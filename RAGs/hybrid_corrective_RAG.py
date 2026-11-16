import os
import re
import json
import logging
import chromadb
import numpy as np

try:
    from logger_config import setup_logging
    from RAGs.chat_history import persist_chat_history, load_file_history
    from RAGs.extract_and_chunk_pdf import initialize_api, decompose_multi_questions, SYS_PROMPT
    
    # 1. Imports from Hybrid RAG 
    from RAGs.hybrid_RAG import (
        refresh_global_bm25,
        GLOBAL_BM25_CACHE,
        bm25_tokenizer,
        global_hybrid_col
    )
    
    # 2. Imports from Legacy CRAG (Reusing Grading Logic)
    from RAG.ask_crag import (
        Log, grade, citation_density, 
        MIN_SIMILARITY, MAX_EXTRA_QUERIES, EXTRA_PER_QUERY, 
        MAX_GRADED, MAX_CONTEXT, REFERENCE_THRESHOLD
    )
except ModuleNotFoundError:
    from logger_config import setup_logging
    from chat_history import persist_chat_history, load_file_history
    from extract_and_chunk_pdf import initialize_api, decompose_multi_questions, SYS_PROMPT
    from hybrid_RAG import refresh_global_bm25, GLOBAL_BM25_CACHE, bm25_tokenizer, global_hybrid_col
    from RAG.ask_crag import Log, grade, citation_density, MIN_SIMILARITY, MAX_EXTRA_QUERIES, EXTRA_PER_QUERY, MAX_GRADED, MAX_CONTEXT, REFERENCE_THRESHOLD

# Initialize advanced logging utilizing the centralized MyApp namespace
setup_logging()
logger = logging.getLogger(f"RAGProj.{__name__}")

# Setup OpenRouter API Client and Environment Variables
client_ai = initialize_api()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")
GLOBAL_SESSION_ID = "global_chat_session"

# ==========================================
# New ChromaDB-Specific Retrieval Logic
# ==========================================

def search_hybrid_rrf(qvec, query_text, k, via_label):
    """Performs Dense + Sparse Hybrid Search via Reciprocal Rank Fusion (RRF) on ChromaDB."""
    logger.debug(f"[{via_label}] Running Hybrid RRF Search for: '{query_text}' (top {k})")
    
    # Refresh the in-memory BM25 cache to sync with the current DB state
    refresh_global_bm25()
    bm25 = GLOBAL_BM25_CACHE["bm25"]
    chunks_cache = GLOBAL_BM25_CACHE["chunks"]
    metas_cache = GLOBAL_BM25_CACHE["metadatas"]
    
    if not bm25 or not chunks_cache:
        logger.warning("BM25 cache is empty. Cannot perform hybrid search.")
        return []

    # Step A: Dense Semantic Search (Vector Embedding Search)
    dense_chunks, dense_ids = [], []
    try:
        dense_results = global_hybrid_col.query(query_embeddings=[qvec], n_results=k, include=["documents", "metadatas", "ids"])
        if dense_results.get("documents") and dense_results["documents"][0]:
            dense_chunks = dense_results["documents"][0]
            dense_ids = dense_results["ids"][0]
            logger.debug(f"[{via_label}] Semantic Dense search found {len(dense_chunks)} chunks.")
    except Exception as e:
        logger.error(f"Dense search failed: {e}", exc_info=True)

    # Step B: Sparse BM25 Keyword Search
    tokenized_q = bm25_tokenizer(query_text)
    sparse_chunks = bm25.get_top_n(tokenized_q, chunks_cache, n=k)
    logger.debug(f"[{via_label}] Sparse BM25 Keyword search found {len(sparse_chunks)} chunks.")

    # Step C: Reciprocal Rank Fusion (RRF) to combine Dense and Sparse results
    rrf_k = 60
    consolidated_scores = {}
    
    # Process Dense Search ranks
    for rank, (doc, cid) in enumerate(zip(dense_chunks, dense_ids)):
        consolidated_scores[doc] = consolidated_scores.get(doc, {"score": 0.0, "id": cid})
        consolidated_scores[doc]["score"] += 1.0 / (rrf_k + rank + 1)
        
    # Process Sparse Search ranks (resolving DB IDs for BM25 results)
    chunk_to_id = {c: cid for c, cid in zip(global_hybrid_col.get()["documents"], global_hybrid_col.get()["ids"])}
    for rank, doc in enumerate(sparse_chunks):
        cid = chunk_to_id.get(doc, f"unknown_{rank}")
        if doc not in consolidated_scores: consolidated_scores[doc] = {"score": 0.0, "id": cid}
        consolidated_scores[doc]["score"] += 1.0 / (rrf_k + rank + 1)

    # Sort combined results based on the highest RRF scores
    ranked_docs = sorted(consolidated_scores.items(), key=lambda item: item[1]["score"], reverse=True)[:k]
    logger.debug(f"[{via_label}] RRF Fusion resolved top {len(ranked_docs)} combined chunks.")
    
    cands = []
    chunk_to_meta = {c: m for c, m in zip(chunks_cache, metas_cache)}
    
    # Format and calculate true cosine similarity for the final candidates
    for doc, info in ranked_docs:
        cid = info["id"]
        meta = chunk_to_meta.get(doc, {})
        try:
            # Fetch raw embedding from DB to compute exact cosine distance against user query
            c_emb = global_hybrid_col.get(ids=[cid], include=["embeddings"])["embeddings"][0]
            sim = float(np.dot(qvec, np.array(c_emb)))
        except: 
            sim = MIN_SIMILARITY # Fallback minimum similarity
             
        cands.append({
            "id": cid, "text": doc, "path": meta.get("source", "Unknown"),
            "loc": f"Page {meta.get('page', '?')}", "sim": sim, "via": via_label
        })
    return cands

def corrective_retrieve_hybrid(question, log):
    """The CRAG loop adapted for ChromaDB Hybrid search pipeline."""
    logger.info(f"--- Starting CRAG Evaluation for Sub-Question: '{question}' ---")
    
    # Generate vector embedding for the user's question
    qvec = client_ai.embeddings.create(model=EMBEDDING_MODEL, input=[question]).data[0].embedding
    log.qvec = qvec

    # 1. Initial Retrieve (Hybrid RRF Search)
    cands = [c for c in search_hybrid_rrf(qvec, question, k=15, via_label="question") if c["sim"] >= MIN_SIMILARITY]
    log.append(f"Retrieved {len(cands)} candidates (similarity >= {MIN_SIMILARITY})")
    
    if not cands: return [], "NOT FOUND", ""

    # 2. Grade retrieved candidates using LLM evaluator
    logger.info("Invoking LLM Grader for Round 1...")
    g = grade(question, cands) 
    relevant = [cands[i - 1] for i in g["relevant"]]
    log.append(f"Round 1: {len(relevant)} of {len(cands)} relevant. Sufficient: {g['sufficient']}")

    # If Round 1 is sufficient, terminate retrieval
    if g["sufficient"] and relevant: return relevant, "CORRECT", g["missing"]

    # 3. Corrective Retrieval: Execute search queries suggested by the Grader
    queries = g["queries"] or [question]
    log.append("New search queries: " + "; ".join(f'"{q}"' for q in queries))

    seen = {c["id"] for c in cands}
    new_cands = []
    
    if queries:
        logger.info(f"Generating embeddings for {len(queries)} corrective queries...")
        qvecs_resp = client_ai.embeddings.create(model=EMBEDDING_MODEL, input=queries)
        
        # Search DB using each newly generated corrective query vector
        for qi, (q_text, q_data) in enumerate(zip(queries, qvecs_resp.data), 1):
            for c in search_hybrid_rrf(q_data.embedding, q_text, k=EXTRA_PER_QUERY, via_label=f"new query {qi}"):
                # Discard already seen chunks and irrelevant bibliography sections
                if c["id"] not in seen and citation_density(c["text"]) < REFERENCE_THRESHOLD:
                    seen.add(c["id"])
                    try:
                        # Re-calculate similarity against the ORIGINAL question vector for fair evaluation
                        c["sim"] = float(np.dot(qvec, global_hybrid_col.get(ids=[c["id"]], include=["embeddings"])["embeddings"][0]))
                    except: pass
                    
                    if c["sim"] >= MIN_SIMILARITY: new_cands.append(c)

    if not new_cands: return relevant, ("PARTIAL" if relevant else "NOT FOUND"), g["missing"]

    # 4. Final Evaluation: Grade the combined pool of original and corrective candidates
    pool = (relevant + new_cands)[:MAX_GRADED]
    logger.info("Invoking LLM Grader for Round 2 (Corrective Search Results)...")
    g2 = grade(question, pool)
    
    relevant2 = [pool[i - 1] for i in g2["relevant"]]
    log.append(f"Round 2: {len(relevant2)} of {len(pool)} relevant.")
    
    return relevant2, ("CORRECTED" if g2["sufficient"] else "PARTIAL"), g2["missing"]

def run_hybrid_corrective_rag(user_query: str, history_limit: int = 5):
    """Main entry point for Global Hybrid CRAG processing."""
    logger.info("=== Starting Global Multi-Query Fused Hybrid CRAG ===")
    logger.info(f"Original User Query: '{user_query}'")
    
    if global_hybrid_col.count() == 0:
        logger.error("Hybrid Collection is empty.")
        return "The database is empty. Please upload and index some PDFs first.", []

    # Decompose the complex query into individual logical sub-questions
    sub_queries = decompose_multi_questions(user_query)
    logger.info(f"Query decomposed into {len(sub_queries)} sub-questions.")

    all_verified_chunks = {}
    all_missing_parts = []

    # Process each sub-question through the CRAG pipeline independently
    for idx, sub_q in enumerate(sub_queries, start=1):
        logger.info(f">>> Processing Sub-Query {idx}/{len(sub_queries)}: '{sub_q}'")
        log = Log()
        chunks, status, missing = corrective_retrieve_hybrid(sub_q, log)

        # Output detailed evaluation log for the current sub-query
        for line in log: 
            logger.info(f"[CRAG Log SQ{idx}] {line}")
            
        if missing:
            all_missing_parts.append(missing)

        # Merge validated chunks into the global collection dict to avoid duplicates
        for c in chunks:
            if c["id"] not in all_verified_chunks:
                all_verified_chunks[c["id"]] = c

    # Halt execution if no relevant context was found across all sub-queries
    if not all_verified_chunks:
        logger.warning("No relevant chunks passed the CRAG filter across all sub-queries.")
        return "The documents do not cover this question. No relevant passages were found.", []

    # Rank the merged validated chunks by similarity score and truncate to MAX_CONTEXT
    final_chunks = sorted(all_verified_chunks.values(), key=lambda c: -c["sim"])[:MAX_CONTEXT]
    logger.info(f"Compiled {len(final_chunks)} unique highly relevant chunks across all sub-queries.")

    # Format the extracted contexts for the LLM prompt insertion
    verified_contexts = [f"[Source: {c['path']} | {c['loc']} | sim: {c['sim']:.3f} | {c['via']}]:\n{c['text']}" for c in final_chunks]
    full_context_text = "\n\n---\n\n".join(verified_contexts)
    logger.info(f"Final Compiled Context for LLM: \n{full_context_text}")

    # Retrieve memory window (chat history limit)
    recent_history = load_file_history(GLOBAL_SESSION_ID)[-history_limit:]
    logger.info(f"Loaded {len(recent_history)} turns of chat history for context.")
    for h in recent_history:
        query_text = str(h.get('query', ''))
        answer_text = str(h.get('answer', ''))[:50]
        logger.debug(f"History - User: {query_text} | Assistant: {answer_text}...")
    
    # Append potential missing data notes to system prompt (CRAG awareness)
    combined_missing = "; ".join(set(all_missing_parts))
    sys_inst = SYS_PROMPT
    if combined_missing:
        sys_inst += f"\n\nNOTE: Likely missing aspects: {combined_missing}. Answer what the sources support and say clearly which parts the documents do not cover."

    # Assemble conversation context including history
    messages = [{"role": "system", "content": sys_inst}]
    for item in recent_history:
        messages.extend([{"role": "user", "content": item["query"]}, {"role": "assistant", "content": item["answer"]}])
    
    # Append the formulated context string with the current user query
    messages.append({"role": "user", "content": f"[Context]:\n{full_context_text}\n\n[Question]:\n{user_query}"})

    # Invoke Generation Model
    logger.info(f"Sending prompt to generation LLM '{CHAT_MODEL}' - for user query={user_query}.")
    try:
        reply = client_ai.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2).choices[0].message.content
        logger.info(f"LLM response generated successfully. Model_reply={reply}.")
    except Exception as e:
        logger.critical(f"Generation failed: {e}", exc_info=True)
        raise e

    # Save to history storage and return to user
    persist_chat_history(GLOBAL_SESSION_ID, user_query, reply, full_context_text)
    return reply, verified_contexts