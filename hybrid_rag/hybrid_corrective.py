import os
import re
import json
import logging
import numpy as np

try:
    from hybrid_rag.logging_setup import setup_logging
    from hybrid_rag.chat_history import persist_chat_history, load_file_history
    from hybrid_rag.file_processing import (
        initialize_api,
        decompose_multi_questions,
        format_location,
        SYS_PROMPT,
    )
    from hybrid_rag.hybrid import (
        get_hybrid_search_state,
        bm25_tokenizer,
        global_hybrid_col,
    )
except ModuleNotFoundError:
    from logging_setup import setup_logging
    from chat_history import persist_chat_history, load_file_history
    from file_processing import (
        initialize_api,
        decompose_multi_questions,
        format_location,
        SYS_PROMPT,
    )
    from hybrid import (
        get_hybrid_search_state,
        bm25_tokenizer,
        global_hybrid_col,
    )

# ==========================================
# 1. Setup & Configuration
# ==========================================

setup_logging()
logger = logging.getLogger("Global_Hybrid_Corrective_RAG")

logger.info("Initializing OpenRouter API Client for Global Hybrid Corrective RAG...")
client_ai = initialize_api()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")
GLOBAL_SESSION_ID = "global_chat_session"

# CRAG thresholds & loop limits
INITIAL_CANDIDATES = 15
MAX_CONTEXT = 6
MIN_SIMILARITY = 0.05
MAX_EXTRA_QUERIES = 4
EXTRA_PER_QUERY = 4
MAX_GRADED = 16
GRADER_TEMPERATURE = 0.0
REFERENCE_THRESHOLD = 12

NOT_COVERED = "The documents do not cover this question. No relevant passages were found."

GRADER_PROMPT = """You check search results for a document question-answering system.
You receive a QUESTION and numbered CANDIDATE excerpts from documents.

1. For each candidate, decide whether it states facts that answer the question, fully or
   partly. Reference lists, bibliography entries, citation titles and passages that only
   mention the same topic without the answer are NOT relevant.
2. Split the question into its separate parts (1 to 4 short parts). A question asking
   about two things, such as "how large are X and Y", has two parts. For each part,
   say whether the relevant candidates actually state the answer to it.
3. Write up to 3 short search queries that could find the unanswered parts in the
   documents, using the documents' likely wording.
4. Say whether the question asks about more than one document (for example
   "each document", "which documents", or a comparison between documents).

Reply with JSON only, no other text, in exactly this form:
{"relevant": [candidate numbers], "parts": [{"part": "...", "answered": true}], "queries": [], "multi_document": false}"""


# ==========================================
# 2. CRAG Utilities (Cosine, Citation Filter, Batch Grader)
# ==========================================

class Log(list):
    """Tracks corrective retrieval steps and logs them as they happen."""
    qvec = None

    def append(self, line):
        logger.info(line, stacklevel=2)
        super().append(line)


def _cosine_similarity(vec_a, vec_b) -> float:
    """Computes normalized cosine similarity safely even if vectors are not unit-length."""
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def citation_density(text: str) -> float:
    """Citation markers per 100 words; reference lists score far higher than normal prose."""
    words = max(len(text.split()), 1)
    years = len(re.findall(r"\b(?:19|20)\d{2}[a-z]?\b", text))
    links = len(re.findall(r"doi|https?://|Available (?:online|at)|Accessed", text, re.I))
    pages = len(re.findall(r"\b\d+\s*\(\d+\)|\bpp?\.\s*\d+|\d+\s*[-\u2013]\s*\d+\.", text))
    return (years + 2 * links + pages) / words * 100


def _read_json(reply: str):
    m = re.search(r"\{.*\}", reply or "", re.DOTALL)
    if not m:
        logger.warning("No JSON found in grader reply: %r", (reply or "")[:300])
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        logger.warning("Grader JSON could not be parsed (%s): %r", e, m.group(0)[:300])
        return None


def _parse_parts(r: dict, have_relevant: bool) -> dict:
    parts = [p for p in r.get("parts", []) if isinstance(p, dict) and str(p.get("part", "")).strip()]
    unanswered = [str(p["part"]).strip() for p in parts if not p.get("answered")]
    queries = []
    for q in unanswered + [q for q in r.get("queries", []) if isinstance(q, str)]:
        q = q.strip()
        if q and q.lower() not in (x.lower() for x in queries):
            queries.append(q)
    return {
        "sufficient": have_relevant and bool(parts) and not unanswered,
        "missing": "; ".join(unanswered),
        "queries": queries[:MAX_EXTRA_QUERIES],
        "multi_document": bool(r.get("multi_document", False)),
    }


def _parse_grade(reply: str, n: int):
    g = _read_json(reply)
    if g is None:
        return None
    relevant = []
    for x in g.get("relevant", []):
        try:
            x = int(x)
        except (TypeError, ValueError):
            continue
        if 1 <= x <= n and x not in relevant:
            relevant.append(x)
    return {"relevant": relevant, **_parse_parts(g, bool(relevant))}


def _is_reference_chunk(cand: dict) -> bool:
    """Only apply bibliography/reference-list filtering to unstructured prose chunks,
    never to structured tabular/JSON chunks (which naturally contain dates/numbers)."""
    if cand.get("data_kind") == "structured":
        return False
    return citation_density(cand["text"]) >= REFERENCE_THRESHOLD


def grade(question: str, cands: list[dict]) -> dict:
    """
    Batch LLM Grader: filters reference lists deterministically first (for unstructured chunks),
    then evaluates all remaining candidates in a single LLM call and proposes corrective queries
    for any unanswered question parts.
    """
    refs = [i for i, c in enumerate(cands, 1) if _is_reference_chunk(c)]
    others = [(i, c) for i, c in enumerate(cands, 1) if i not in refs]

    if not others:
        return {
            "relevant": [],
            "sufficient": False,
            "missing": "",
            "queries": [],
            "multi_document": False,
            "refs": refs,
        }

    listing = "\n\n".join(
        f"[Candidate {k}] (file: {c['path']}, {c['loc']})\n{c['text']}"
        for k, (_, c) in enumerate(others, 1)
    )
    messages = [
        {"role": "system", "content": GRADER_PROMPT},
        {"role": "user", "content": f"QUESTION: {question}\n\nCANDIDATES:\n\n{listing}"},
    ]

    try:
        response = client_ai.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=GRADER_TEMPERATURE,
            extra_body={"reasoning": {"enabled": False}},
        )
        reply = response.choices[0].message.content
    except Exception as e:
        logger.warning(f"Batch grading call failed ({e}); retrying without extra_body...")
        response = client_ai.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=GRADER_TEMPERATURE,
        )
        reply = response.choices[0].message.content

    sub = _parse_grade(reply, len(others))
    if sub is None:
        logger.warning("Batch grader reply unreadable; falling back to top 4 non-reference candidates.")
        return {
            "relevant": [others[k - 1][0] for k in range(1, min(4, len(others)) + 1)],
            "sufficient": True,
            "missing": "",
            "queries": [],
            "multi_document": False,
            "refs": refs,
            "failed": True,
        }

    return {
        **sub,
        "relevant": [others[k - 1][0] for k in sub["relevant"]],
        "refs": refs,
    }


# ==========================================
# 3. Hybrid RRF Retrieval (Fixed ChromaDB 1.x + Cached IDs/Embeddings)
# ==========================================

def search_hybrid_rrf(
    qvec: list[float],
    query_text: str,
    k: int,
    via_label: str,
    selected_files: list = None,
) -> list[dict]:
    """Performs Dense + Sparse Hybrid Search via Reciprocal Rank Fusion (RRF) on ChromaDB."""
    logger.debug(f"[{via_label}] Running Hybrid RRF Search for: '{query_text}' (top {k})")

    bm25, chunks_cache, _, doc_to_id, doc_to_meta, id_to_emb, where_filter = get_hybrid_search_state(
        selected_files
    )

    if not bm25 or not chunks_cache:
        logger.warning("BM25 cache is empty. Cannot perform hybrid search.")
        return []

    effective_k = min(k, len(chunks_cache))

    # Step A: Dense Semantic Search (ChromaDB 1.x always returns ids; do NOT pass "ids" in include)
    dense_chunks, dense_ids = [], []
    try:
        dense_results = global_hybrid_col.query(
            query_embeddings=[qvec],
            n_results=effective_k,
            include=["documents", "metadatas"],
            where=where_filter,
        )
        if dense_results.get("documents") and dense_results["documents"][0]:
            dense_chunks = dense_results["documents"][0]
            dense_ids = dense_results["ids"][0]
            logger.debug(f"[{via_label}] Semantic Dense search found {len(dense_chunks)} chunks.")
    except Exception as e:
        logger.error(f"Dense search failed: {e}", exc_info=True)

    # Step B: Sparse BM25 Keyword Search (uses Arabic-normalized tokenizer from hybrid.py)
    tokenized_q = bm25_tokenizer(query_text)
    sparse_chunks = bm25.get_top_n(tokenized_q, chunks_cache, n=effective_k)
    logger.debug(f"[{via_label}] Sparse BM25 Keyword search found {len(sparse_chunks)} chunks.")

    # Step C: Reciprocal Rank Fusion (RRF)
    rrf_k = 60
    consolidated_scores = {}

    for rank, (doc, cid) in enumerate(zip(dense_chunks, dense_ids)):
        entry = consolidated_scores.setdefault(doc, {"score": 0.0, "id": cid})
        entry["score"] += 1.0 / (rrf_k + rank + 1)

    for rank, doc in enumerate(sparse_chunks):
        cid = doc_to_id.get(doc, f"unknown_{rank}")
        entry = consolidated_scores.setdefault(doc, {"score": 0.0, "id": cid})
        entry["score"] += 1.0 / (rrf_k + rank + 1)

    ranked_docs = sorted(
        consolidated_scores.items(), key=lambda item: item[1]["score"], reverse=True
    )[:effective_k]
    logger.debug(f"[{via_label}] RRF Fusion resolved top {len(ranked_docs)} combined chunks.")

    # Compute exact cosine similarity from cached embeddings (zero per-chunk DB round-trips)
    cands = []
    for doc, info in ranked_docs:
        cid = info["id"]
        meta = doc_to_meta.get(doc, {}) or {}
        c_emb = id_to_emb.get(cid)
        sim = _cosine_similarity(qvec, c_emb) if c_emb is not None else MIN_SIMILARITY

        cands.append({
            "id": cid,
            "text": doc,
            "path": meta.get("source", "Unknown"),
            "loc": format_location(meta),
            "data_kind": meta.get("data_kind", "unstructured"),
            "sim": sim,
            "via": via_label,
        })
    return cands


# ==========================================
# 4. 2-Round Corrective Retrieval Loop
# ==========================================

def corrective_retrieve_hybrid(question: str, log: Log, selected_files: list = None):
    """2-Round CRAG loop adapted for ChromaDB Hybrid RRF retrieval."""
    logger.info(f"--- Starting CRAG Evaluation for Sub-Question: '{question}' ---")

    qvec = client_ai.embeddings.create(model=EMBEDDING_MODEL, input=[question]).data[0].embedding
    log.qvec = qvec

    # 1. Initial Retrieve (Hybrid RRF Search)
    cands = [
        c
        for c in search_hybrid_rrf(
            qvec, question, k=INITIAL_CANDIDATES, via_label="question", selected_files=selected_files
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

    # 3. Round 2 Corrective Retrieval using grader-suggested queries
    queries = g["queries"] or [question]
    log.append("New search queries: " + "; ".join(f'"{q}"' for q in queries))

    _, _, _, _, _, id_to_emb, _ = get_hybrid_search_state(selected_files)
    seen = {c["id"] for c in cands}
    new_cands = []

    if queries:
        logger.info(f"Generating embeddings for {len(queries)} corrective queries...")
        qvecs_resp = client_ai.embeddings.create(model=EMBEDDING_MODEL, input=queries)

        for qi, (q_text, q_data) in enumerate(zip(queries, qvecs_resp.data), 1):
            for c in search_hybrid_rrf(
                q_data.embedding,
                q_text,
                k=EXTRA_PER_QUERY,
                via_label=f"new query {qi}",
                selected_files=selected_files,
            ):
                if c["id"] not in seen and not _is_reference_chunk(c):
                    seen.add(c["id"])
                    c_emb = id_to_emb.get(c["id"])
                    if c_emb is not None:
                        c["sim"] = _cosine_similarity(qvec, c_emb)
                    if c["sim"] >= MIN_SIMILARITY:
                        new_cands.append(c)

    if not new_cands:
        return relevant, ("PARTIAL" if relevant else "NOT FOUND"), g["missing"]

    # 4. Round 2 Final Grading on combined pool
    pool = (relevant + new_cands)[:MAX_GRADED]
    logger.info("Invoking LLM Grader for Round 2 (Corrective Search Results)...")
    g2 = grade(question, pool)
    relevant2 = [pool[i - 1] for i in g2["relevant"]]
    log.append(f"Round 2: {len(relevant2)} of {len(pool)} relevant.")

    return relevant2, ("CORRECTED" if g2["sufficient"] else "PARTIAL"), g2["missing"]


# ==========================================
# 5. Main Entry Point
# ==========================================

def run_hybrid_corrective_rag(
    user_query: str,
    history_limit: int = 5,
    selected_files: list = None,
):
    """
    Searches across selected (or ALL) indexed documents using 2-Round Fused Hybrid CRAG
    (Dense + BM25 via RRF -> Batch Grading -> Corrective Query Rewriting -> Cited Answer).
    """
    logger.info("=== Starting Global Multi-Query Fused Hybrid CRAG ===")
    logger.info(f"Original User Query: '{user_query}'")

    if global_hybrid_col.count() == 0:
        logger.error("Hybrid Collection is empty.")
        return "The database is empty. Please upload and index some PDFs first.", []

    sub_queries = decompose_multi_questions(user_query)
    logger.info(f"Query decomposed into {len(sub_queries)} sub-questions.")

    all_verified_chunks = {}
    all_missing_parts = []

    for idx, sub_q in enumerate(sub_queries, start=1):
        logger.info(f">>> Processing Sub-Query {idx}/{len(sub_queries)}: '{sub_q}'")
        log = Log()
        chunks, status, missing = corrective_retrieve_hybrid(
            sub_q, log, selected_files=selected_files
        )

        for line in log:
            logger.info(f"[CRAG Log SQ{idx}] {line}")

        if missing:
            all_missing_parts.append(missing)

        for c in chunks:
            if c["id"] not in all_verified_chunks:
                all_verified_chunks[c["id"]] = c

    if not all_verified_chunks:
        logger.warning("No relevant chunks passed the CRAG filter across all sub-queries.")
        persist_chat_history(GLOBAL_SESSION_ID, user_query, NOT_COVERED, "")
        return NOT_COVERED, []

    final_chunks = sorted(all_verified_chunks.values(), key=lambda c: -c["sim"])[:MAX_CONTEXT]
    logger.info(f"Compiled {len(final_chunks)} unique highly relevant chunks across all sub-queries.")

    verified_contexts = [
        f"[Source: {c['path']} | {c['loc']} | sim: {c['sim']:.3f} | {c['via']}]:\n{c['text']}"
        for c in final_chunks
    ]
    full_context_text = "\n\n---\n\n".join(verified_contexts)

    global_history = load_file_history(GLOBAL_SESSION_ID)
    recent_history = (
        global_history[-history_limit:] if len(global_history) > history_limit else global_history
    )
    logger.info(
        f"Using {len(recent_history)} recent turns from history (Total saved: {len(global_history)})."
    )

    combined_missing = "; ".join(set(all_missing_parts))
    sys_inst = SYS_PROMPT
    if combined_missing:
        sys_inst += (
            f"\n\nNOTE: Likely missing aspects: {combined_missing}. "
            "Answer what the sources support and say clearly which parts the documents do not cover."
        )

    messages = [{"role": "system", "content": sys_inst}]
    for item in recent_history:
        messages.extend([
            {"role": "user", "content": item["query"]},
            {"role": "assistant", "content": item["answer"]},
        ])

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
        logger.critical(f"Generation failed: {e}", exc_info=True)
        raise e

    persist_chat_history(GLOBAL_SESSION_ID, user_query, reply, full_context_text)
    return reply, verified_contexts


if __name__ == "__main__":
    sample_query = (
        "1. What are the general steps in cultured meat production?\n"
        "2. Why is Mycoplasma contamination difficult to detect visually?"
    )
    answer, docs = run_hybrid_corrective_rag(sample_query)
    print("\n" + "=" * 60)
    print("FINAL MODEL ANSWER:")
    print("=" * 60)
    print(answer)
