import os
import logging
import numpy as np

try:
    from logger_config import setup_logging
    from RAGs.chat_history import persist_chat_history, load_file_history
    from RAGs.extract_and_chunk_pdf import initialize_api, decompose_multi_questions, SYS_PROMPT
    
    # 1. Imports from Parent-Child RAG 
    from RAGs.parent_child_RAG import global_child_col, global_parent_col
    
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
    from parent_child_RAG import global_child_col, global_parent_col
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

def search_parents_via_children(qvec, k, via_label):
    """Searches children in ChromaDB, maps to Parents, and calculates Cosine Similarity."""
    logger.debug(f"[{via_label}] Executing Child Search in DB to locate Parent Blocks (top {k})")
    
    # Step 1: Query child chunk collection using dense semantic embedding
    results = global_child_col.query(query_embeddings=[qvec], n_results=k, include=["metadatas"])
    cands = []
    seen_parents = set()
    
    if results.get("metadatas") and results["metadatas"][0]:
        logger.debug(f"[{via_label}] Located {len(results['metadatas'][0])} child hits. Resolving unique parents...")
        
        # Step 2: Iterate child chunks, map to Parent block, prevent duplication
        for meta in results["metadatas"][0]:
            pid = meta.get("parent_id")
            
            if pid and pid not in seen_parents:
                seen_parents.add(pid)
                
                # Fetch Parent context block using the resolved parent ID
                p_data = global_parent_col.get(ids=[pid], include=["documents", "metadatas", "embeddings"])
                
                if p_data["documents"]:
                    p_text, p_meta, p_emb = p_data["documents"][0], p_data["metadatas"][0], p_data["embeddings"][0]
                    
                    # Calculate Cosine distance manually against the raw parent vector
                    sim = float(np.dot(qvec, np.array(p_emb))) 
                    
                    cands.append({
                        "id": pid, "text": p_text, "path": p_meta.get("source", "Unknown"),
                        "loc": "Parent Block", "sim": sim, "via": via_label
                    })
                    
    logger.debug(f"[{via_label}] Extracted {len(cands)} unique parent candidate blocks.")
    return sorted(cands, key=lambda x: -x["sim"])

def corrective_retrieve_parent_child(question, log):
    """The CRAG loop adapted for ChromaDB Hierarchical search pipeline."""
    logger.info(f"--- Starting CRAG Evaluation for Sub-Question: '{question}' ---")
    
    # Generate vector embedding for the user's question
    qvec = client_ai.embeddings.create(model=EMBEDDING_MODEL, input=[question]).data[0].embedding
    log.qvec = qvec

    # 1. Initial Retrieve (Parent-Child Strategy)
    cands = [c for c in search_parents_via_children(qvec, k=10, via_label="question") if c["sim"] >= MIN_SIMILARITY]
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
        for qi, q_data in enumerate(qvecs_resp.data, 1):
            for c in search_parents_via_children(q_data.embedding, k=EXTRA_PER_QUERY, via_label=f"new query {qi}"):
                
                # Discard already seen Parent blocks and irrelevant bibliography sections
                if c["id"] not in seen and citation_density(c["text"]) < REFERENCE_THRESHOLD:
                    seen.add(c["id"])
                    
                    # Re-calculate similarity against the ORIGINAL question vector for fair evaluation
                    c["sim"] = float(np.dot(qvec, global_parent_col.get(ids=[c["id"]], include=["embeddings"])["embeddings"][0]))
                    
                    if c["sim"] >= MIN_SIMILARITY: new_cands.append(c)

    if not new_cands: return relevant, ("PARTIAL" if relevant else "NOT FOUND"), g["missing"]

    # 4. Final Evaluation: Grade the combined pool of original and corrective candidates
    pool = (relevant + new_cands)[:MAX_GRADED]
    logger.info("Invoking LLM Grader for Round 2 (Corrective Search Results)...")
    g2 = grade(question, pool)
    
    relevant2 = [pool[i - 1] for i in g2["relevant"]]
    log.append(f"Round 2: {len(relevant2)} of {len(pool)} relevant.")
    
    return relevant2, ("CORRECTED" if g2["sufficient"] else "PARTIAL"), g2["missing"]

def run_parent_child_corrective_rag(user_query: str, history_limit: int = 5):
    """Main entry point for Global Parent-Child CRAG processing."""
    logger.info("=== Starting Global Multi-Query Fused Parent-Child CRAG ===")
    logger.info(f"Original User Query: '{user_query}'")

    if global_child_col.count() == 0:
        logger.error("Hierarchical DB is empty.")
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
        chunks, status, missing = corrective_retrieve_parent_child(sub_q, log)

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
        logger.warning("No relevant parent blocks passed the CRAG filter across all sub-queries.")
        return "The documents do not cover this question. No relevant passages were found.", []

    # Rank the merged validated Parent blocks by similarity score and truncate to MAX_CONTEXT
    final_chunks = sorted(all_verified_chunks.values(), key=lambda c: -c["sim"])[:MAX_CONTEXT]
    logger.info(f"Compiled {len(final_chunks)} unique highly relevant parent blocks across all sub-queries.")

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
        logger.critical(f"LLM generation failed: {e}", exc_info=True)
        raise e

    # Save to history storage and return to user
    persist_chat_history(GLOBAL_SESSION_ID, user_query, reply, full_context_text)
    return reply, verified_contexts