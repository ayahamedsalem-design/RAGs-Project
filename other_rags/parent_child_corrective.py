"""2-Round Fused Parent-Child Corrective RAG (CRAG) pipeline over hierarchical ChromaDB collections."""

import os
import logging

from hybrid_rag.logging_setup import setup_logging
from hybrid_rag.chat_history import (
    persist_chat_history,
    load_file_history,
    build_history_messages,
)
from hybrid_rag.pdf_processing import initialize_api, decompose_multi_questions, SYS_PROMPT
from hybrid_rag.hybrid_corrective import (
    Log,
    grade,
    citation_density,
    _cosine_similarity,
    MIN_SIMILARITY,
    EXTRA_PER_QUERY,
    MAX_GRADED,
    MAX_CONTEXT,
    REFERENCE_THRESHOLD,
    NOT_COVERED,
)
from other_rags.parent_child import (
    global_child_col,
    global_parent_col,
    get_child_where_filter,
)

# ==========================================
# 1. Setup & Configuration
# ==========================================

setup_logging()
logger = logging.getLogger("Global_Parent_Child_Corrective_RAG")

logger.info("Initializing OpenRouter API Client for Global Parent-Child Corrective RAG...")
client_ai = initialize_api()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")
GLOBAL_SESSION_ID = "global_chat_session"


# ==========================================
# 2. Hierarchical Retrieval (With Page Citations & True Cosine Similarity)
# ==========================================

def search_parents_via_children(
    qvec: list[float],
    k: int,
    via_label: str,
    selected_files: list = None,
) -> list[dict]:
    """Searches child chunks in ChromaDB, batch-fetches unique parent blocks, and computes cosine similarity."""
    where_filter, matching_count = get_child_where_filter(selected_files)
    if matching_count == 0:
        return []

    effective_k = min(k, matching_count)
    results = global_child_col.query(
        query_embeddings=[qvec],
        n_results=effective_k,
        include=["metadatas"],
        where=where_filter,
    )

    seen_parents = set()
    parent_ids_ordered = []
    if results.get("metadatas") and results["metadatas"][0]:
        for meta in results["metadatas"][0]:
            pid = meta.get("parent_id")
            if pid and pid not in seen_parents:
                seen_parents.add(pid)
                parent_ids_ordered.append(pid)

    if not parent_ids_ordered:
        return []

    p_data = global_parent_col.get(
        ids=parent_ids_ordered,
        include=["documents", "metadatas", "embeddings"],
    )
    doc_map = dict(zip(p_data.get("ids", []), p_data.get("documents", [])))
    meta_map = dict(zip(p_data.get("ids", []), p_data.get("metadatas", [])))
    emb_map = dict(zip(p_data.get("ids", []), p_data.get("embeddings", [])))

    cands = []
    for pid in parent_ids_ordered:
        if pid not in doc_map:
            continue
        p_meta = meta_map.get(pid) or {}
        p_emb = emb_map.get(pid)
        sim = _cosine_similarity(qvec, p_emb) if p_emb is not None else MIN_SIMILARITY
        page_label = p_meta.get("page")
        loc = f"Page {page_label}" if page_label else "Parent Block"

        cands.append({
            "id": pid,
            "text": doc_map[pid],
            "path": p_meta.get("source", "Unknown"),
            "loc": loc,
            "sim": sim,
            "via": via_label,
        })

    return sorted(cands, key=lambda x: -x["sim"])


# ==========================================
# 3. 2-Round Corrective Retrieval Loop
# ==========================================

def corrective_retrieve_parent_child(question: str, log: Log, selected_files: list = None):
    """2-Round CRAG loop adapted for ChromaDB Hierarchical Parent-Child search."""
    logger.info(f"--- Starting CRAG Evaluation for Sub-Question: '{question}' ---")

    qvec = client_ai.embeddings.create(model=EMBEDDING_MODEL, input=[question]).data[0].embedding
    log.qvec = qvec

    # 1. Initial Retrieve (Parent-Child)
    cands = [
        c
        for c in search_parents_via_children(
            qvec, k=10, via_label="question", selected_files=selected_files
        )
        if c["sim"] >= MIN_SIMILARITY
    ]
    log.append(f"Retrieved {len(cands)} candidates (similarity >= {MIN_SIMILARITY})")
    if not cands:
        return [], "NOT FOUND", ""

    # 2. Round 1 Batch Grading
    logger.info("Invoking LLM Grader for Round 1...")
    g = grade(question, cands)
    relevant = [cands[i - 1] for i in g["relevant"]]
    log.append(f"Round 1: {len(relevant)} of {len(cands)} relevant. Sufficient: {g['sufficient']}")
    if g["sufficient"] and relevant:
        return relevant, "CORRECT", g["missing"]

    # 3. Round 2 Corrective Retrieval with grader-suggested queries
    queries = g["queries"] or [question]
    log.append("New search queries: " + "; ".join(f'"{q}"' for q in queries))

    seen = {c["id"] for c in cands}
    new_cands = []

    if queries:
        logger.info(f"Generating embeddings for {len(queries)} corrective queries...")
        qvecs_resp = client_ai.embeddings.create(model=EMBEDDING_MODEL, input=queries)

        for qi, q_data in enumerate(qvecs_resp.data, 1):
            for c in search_parents_via_children(
                q_data.embedding,
                k=EXTRA_PER_QUERY,
                via_label=f"new query {qi}",
                selected_files=selected_files,
            ):
                if c["id"] not in seen and citation_density(c["text"]) < REFERENCE_THRESHOLD:
                    seen.add(c["id"])
                    p_emb_resp = global_parent_col.get(ids=[c["id"]], include=["embeddings"])
                    embs = p_emb_resp.get("embeddings")
                    if embs is not None and len(embs) > 0:
                        c["sim"] = _cosine_similarity(qvec, embs[0])
                    if c["sim"] >= MIN_SIMILARITY:
                        new_cands.append(c)

    if not new_cands:
        return relevant, ("PARTIAL" if relevant else "NOT FOUND"), g["missing"]

    # 4. Round 2 Final Grading
    pool = (relevant + new_cands)[:MAX_GRADED]
    logger.info("Invoking LLM Grader for Round 2 (Corrective Search Results)...")
    g2 = grade(question, pool)
    relevant2 = [pool[i - 1] for i in g2["relevant"]]
    log.append(f"Round 2: {len(relevant2)} of {len(pool)} relevant.")
    return relevant2, ("CORRECTED" if g2["sufficient"] else "PARTIAL"), g2["missing"]


# ==========================================
# 4. Main Entry Point
# ==========================================

def run_parent_child_corrective_rag(
    user_query: str,
    history_limit: int = 6,
    selected_files: list = None,
    session_id: str = GLOBAL_SESSION_ID,
):
    """
    Searches across selected (or ALL) indexed documents using 2-Round Fused Parent-Child CRAG
    (Child Search -> Parent Blocks with Page Citations -> Batch Grading -> Round-2 Query Rewriting).
    """
    logger.info("=== Starting Global Multi-Query Fused Parent-Child CRAG ===")
    logger.info(f"Session: '{session_id}' | Original User Query: '{user_query}'")

    if global_child_col.count() == 0 or global_parent_col.count() == 0:
        logger.error("Hierarchical DB is empty.")
        return "The database is empty. Please upload and index some PDFs first.", []

    _, matching_count = get_child_where_filter(selected_files)
    if matching_count == 0:
        logger.warning("No hierarchical chunks matched the selected file filter.")
        return "No indexed chunks were found for the selected file(s) in the Parent-Child DB.", []

    sub_queries = decompose_multi_questions(user_query)
    logger.info(f"Query decomposed into {len(sub_queries)} sub-questions.")

    all_verified_chunks = {}
    all_missing_parts = []

    for idx, sub_q in enumerate(sub_queries, start=1):
        logger.info(f">>> Processing Sub-Query {idx}/{len(sub_queries)}: '{sub_q}'")
        log = Log()
        chunks, status, missing = corrective_retrieve_parent_child(
            sub_q, log, selected_files=selected_files
        )
        logger.info(
            f"[Parent-Child CRAG SQ{idx}] Completed with status={status} ({len(chunks)} relevant block(s))."
        )
        if missing:
            all_missing_parts.append(missing)
        for c in chunks:
            if c["id"] not in all_verified_chunks:
                all_verified_chunks[c["id"]] = c

    if not all_verified_chunks:
        logger.warning("No relevant parent blocks passed the CRAG filter across all sub-queries.")
        persist_chat_history(session_id, user_query, NOT_COVERED, "")
        return NOT_COVERED, []

    final_chunks = sorted(all_verified_chunks.values(), key=lambda c: -c["sim"])[:MAX_CONTEXT]
    logger.info(f"Compiled {len(final_chunks)} unique highly relevant parent blocks across all sub-queries.")

    verified_contexts = [
        f"[Source: {c['path']} | {c['loc']} | sim: {c['sim']:.3f} | {c['via']}]:\n{c['text']}"
        for c in final_chunks
    ]
    full_context_text = "\n\n---\n\n".join(verified_contexts)

    # Load rolling earlier-turn summary (if >6 turns) + latest verbatim turns for the active session
    history_msgs = build_history_messages(session_id, limit=history_limit)

    combined_missing = "; ".join(set(all_missing_parts))
    sys_inst = SYS_PROMPT
    if combined_missing:
        sys_inst += (
            f"\n\nNOTE: Likely missing aspects: {combined_missing}. "
            "Answer what the sources support and say clearly which parts the documents do not cover."
        )

    messages = [{"role": "system", "content": sys_inst}] + history_msgs
    messages.append({
        "role": "user",
        "content": f"[Context from Global Database]:\n{full_context_text}\n\n[User Question(s)]:\n{user_query}",
    })

    logger.info(f"Sending prompt to generation LLM '{CHAT_MODEL}'...")
    try:
        reply = client_ai.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
        ).choices[0].message.content
        logger.info("LLM response generated successfully.")
    except Exception as e:
        logger.critical(f"LLM generation failed: {e}", exc_info=True)
        raise e

    persist_chat_history(session_id, user_query, reply, full_context_text)
    return reply, verified_contexts


if __name__ == "__main__":
    sample_query = (
        "1. What are the general steps in cultured meat production?\n"
        "2. Why is Mycoplasma contamination difficult to detect visually?"
    )
    answer, docs = run_parent_child_corrective_rag(sample_query)
    print("\n" + "=" * 60)
    print("FINAL MODEL ANSWER:")
    print("=" * 60)
    print(answer)
