"""
Retrieval-only evaluation harness for all 7 RAG pipelines across the 80-question dataset
from docs/ (4 PDFs × 20 Q&As).

Evaluates the retrieved chunks BEFORE the final Answer LLM (never calling the Answer LLM).
For the 3 Corrective pipelines (crag_hybrid, crag_parent_child, crag), captures BOTH:
  - pre_chunks:  raw candidate chunks before the CRAG grader
  - post_chunks: filtered/corrected chunks after the CRAG grader (just before Answer LLM)

Usage:
    python -m evaluation.retrieval_eval index
    python -m evaluation.retrieval_eval run [--only hybrid crag_hybrid ...] [--questions 1-80] [--per-file]
    python -m evaluation.retrieval_eval report
    python -m evaluation.retrieval_eval all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from hybrid_rag import cost
from hybrid_rag.logging_setup import setup_logging
from hybrid_rag.pdf_processing import decompose_multi_questions
from evaluation.retrieval_dataset import (
    DOCS_DIR,
    EVAL_DOCS,
    RESULTS_DIR,
    SOURCE_TO_DOC,
    build_dataset,
    load_dataset,
    locate_chunk,
    score_chunk,
    score_chunk_set_coverage,
)

logger = logging.getLogger("rag.retrieval_eval")
cost.COST_LOG = RESULTS_DIR / "cost_retrieval_eval.csv"

EMBED_CACHE_FILE = RESULTS_DIR / "query_embeddings_cache.json"

ALL_PIPELINES = [
    ("hybrid", "Hybrid RAG", False),
    ("crag_hybrid", "Corrective Hybrid RAG", True),
    ("basic", "Basic RAG", False),
    ("parent_child", "Parent-Child (Hierarchical) RAG", False),
    ("crag_parent_child", "Corrective Parent-Child RAG", True),
    ("naive", "Ext Naive RAG", False),
    ("crag", "Ext Corrective RAG", True),
]
PIPELINE_META = {k: (label, is_crag) for k, label, is_crag in ALL_PIPELINES}


class DailyQuotaExhausted(RuntimeError):
    """Raised when OpenRouter reports the free-tier daily request quota is exhausted."""


def result_file(system: str, per_file: bool = False) -> Path:
    suffix = "_perfile" if per_file else ""
    return RESULTS_DIR / f"retrieval_{system}{suffix}.json"


def _save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _is_daily_limit_error(err: Exception) -> bool:
    msg = str(err).lower()
    return ("429" in msg and ("per-day" in msg or "free-models-per-day" in msg)) or "daily limit" in msg


def _is_transient_error(err: Exception) -> bool:
    msg = str(err).lower()
    return any(tok in msg for tok in (
        "429", "rate", "502", "503", "500", "504", "timeout", "timed out",
        "connection", "temporarily", "inference failed", "overloaded",
    ))


def call_with_backoff(fn, *args, max_retries: int = 7, base_delay: float = 3.0, **kwargs):
    """Calls an API function with exponential backoff on rate limits / transient errors."""
    delay = base_delay
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if _is_daily_limit_error(e):
                raise DailyQuotaExhausted(str(e)) from e
            if _is_transient_error(e) and attempt < max_retries:
                logger.warning("Transient error (attempt %d/%d): %s — sleeping %.1fs",
                               attempt, max_retries, str(e)[:200], delay)
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            raise


# ---------- Cosine Similarity & Query Embedding Cache ----------
def _cosine(a, b) -> float | None:
    if a is None or b is None:
        return None
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return None
    return round(float(np.dot(va, vb) / (na * nb)), 4)


class QueryEmbeddingCache:
    """Caches identical query embeddings across pipelines for the active EMBEDDING_MODEL."""

    def __init__(self, path: Path = EMBED_CACHE_FILE):
        self.path = path
        self.model = os.getenv("EMBEDDING_MODEL", "")
        raw = _load_json(path, {})
        self.data: dict[str, dict[str, list[float]]] = raw if isinstance(raw, dict) else {}
        self.data.setdefault(self.model, {})

    def get_many(self, texts: list[str], client_ai) -> list[list[float]]:
        bucket = self.data[self.model]
        missing = [t for t in dict.fromkeys(texts) if t not in bucket]
        if missing:
            # Batch embed missing query texts (up to 16 per request to stay within Nemotron limits)
            for i in range(0, len(missing), 16):
                batch = missing[i : i + 16]
                resp = call_with_backoff(
                    client_ai.embeddings.create,
                    model=self.model,
                    input=batch,
                )
                for text, item in zip(batch, resp.data):
                    bucket[text] = list(item.embedding)
                time.sleep(0.5)
            _save_json(self.path, self.data)
        return [bucket[t] for t in texts]

    def get_one(self, text: str, client_ai) -> list[float]:
        return self.get_many([text], client_ai)[0]


# ---------- Resumable Indexing Across All 3 Store Groups ----------
def _embed_texts_resumable(client_ai, model: str, texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """Embeds a list of texts in small batches, auto-splitting if a batch is rejected."""
    if not texts:
        return []
    out: list[list[float]] = []
    i = 0
    while i < len(texts):
        batch = texts[i : i + batch_size]
        try:
            resp = call_with_backoff(client_ai.embeddings.create, model=model, input=batch)
            out.extend([item.embedding for item in resp.data])
            i += len(batch)
            time.sleep(0.6)
        except DailyQuotaExhausted:
            raise
        except Exception as e:
            if len(batch) > 1:
                half = max(1, len(batch) // 2)
                logger.warning("Batch of %d failed (%s); splitting into %d + %d",
                               len(batch), str(e)[:120], half, len(batch) - half)
                out.extend(_embed_texts_resumable(client_ai, model, batch[:half], batch_size=half))
                out.extend(_embed_texts_resumable(client_ai, model, batch[half:], batch_size=half))
                i += len(batch)
            else:
                raise
    return out


def _index_hybrid_resumable(doc_key: str, pdf_path: Path):
    from hybrid_rag.hybrid import (
        EMBEDDING_MODEL,
        client_ai,
        global_hybrid_col,
        refresh_global_bm25,
    )
    from hybrid_rag.pdf_processing import extract_and_chunk_standard

    file_name = pdf_path.name
    existing = global_hybrid_col.get(where={"source": file_name}, include=[])
    chunks, metadatas = extract_and_chunk_standard(str(pdf_path), chunk_size=1000, chunk_overlap=200)
    if existing and len(existing.get("ids") or []) >= len(chunks):
        print(f"  [hybrid] {file_name}: already indexed ({len(existing['ids'])} chunks)")
        return

    # If partially indexed with deterministic eval IDs, resume; if random UUIDs exist, skip if >0
    existing_ids = set(existing.get("ids") or [])
    if existing_ids and not any(cid.startswith(f"hyb_{doc_key}_") for cid in existing_ids):
        print(f"  [hybrid] {file_name}: already indexed ({len(existing_ids)} chunks)")
        return

    ids = [f"hyb_{doc_key}_{i}" for i in range(len(chunks))]
    todo = [i for i, cid in enumerate(ids) if cid not in existing_ids]
    print(f"  [hybrid] {file_name}: embedding {len(todo)}/{len(chunks)} chunks...")
    batch_size = 8
    for start in range(0, len(todo), batch_size):
        idxs = todo[start : start + batch_size]
        vecs = _embed_texts_resumable(client_ai, EMBEDDING_MODEL, [chunks[j] for j in idxs], batch_size=batch_size)
        global_hybrid_col.upsert(
            ids=[ids[j] for j in idxs],
            documents=[chunks[j] for j in idxs],
            embeddings=vecs,
            metadatas=[metadatas[j] for j in idxs],
        )
        print(f"    [hybrid] {file_name}: {min(start + batch_size, len(todo))}/{len(todo)} done", flush=True)
    refresh_global_bm25()


def _index_parent_child_resumable(doc_key: str, pdf_path: Path):
    from other_rags.hierarchical_chunking import extract_and_chunk_hierarchical
    from other_rags.parent_child import (
        EMBEDDING_MODEL,
        client_ai,
        global_child_col,
        global_parent_col,
    )

    file_name = pdf_path.name
    # Deterministic chunking (exact same 300/70/15 parameters as other_rags/parent_child.py)
    p_texts, _, c_texts, c_metas_orig, _ = extract_and_chunk_hierarchical(
        str(pdf_path), parent_chunk_words=300, child_chunk_words=70, child_overlap=15
    )

    # Re-assign deterministic parent & child IDs so partial runs resume cleanly
    parent_ids = [f"pc_p_{doc_key}_{i}" for i in range(len(p_texts))]
    # Map each child's original parent_id to its deterministic parent_id
    orig_pid_order: list[str] = []
    orig_to_det: dict[str, str] = {}
    for m in c_metas_orig:
        opid = m["parent_id"]
        if opid not in orig_to_det:
            idx = len(orig_pid_order)
            orig_pid_order.append(opid)
            orig_to_det[opid] = parent_ids[idx]

    child_ids = [f"pc_c_{doc_key}_{i}" for i in range(len(c_texts))]
    child_metas = [{"parent_id": orig_to_det[m["parent_id"]], "source": file_name} for m in c_metas_orig]

    existing_children = set((global_child_col.get(where={"source": file_name}, include=[]) or {}).get("ids") or [])
    existing_parents = set((global_parent_col.get(where={"source": file_name}, include=[]) or {}).get("ids") or [])

    if len(existing_children) >= len(child_ids) and len(existing_parents) >= len(parent_ids):
        print(f"  [parent_child] {file_name}: already indexed ({len(existing_parents)} parents, {len(existing_children)} children)")
        return

    todo_c = [i for i, cid in enumerate(child_ids) if cid not in existing_children]
    if todo_c:
        print(f"  [parent_child] {file_name}: embedding {len(todo_c)}/{len(child_ids)} children...")
        batch_size = 16
        for start in range(0, len(todo_c), batch_size):
            idxs = todo_c[start : start + batch_size]
            vecs = _embed_texts_resumable(client_ai, EMBEDDING_MODEL, [c_texts[j] for j in idxs], batch_size=batch_size)
            global_child_col.upsert(
                ids=[child_ids[j] for j in idxs],
                documents=[c_texts[j] for j in idxs],
                embeddings=vecs,
                metadatas=[child_metas[j] for j in idxs],
            )
            if (start // batch_size) % 10 == 0 or start + batch_size >= len(todo_c):
                print(f"    [parent_child] {file_name}: children {min(start + batch_size, len(todo_c))}/{len(todo_c)} done", flush=True)

    # Populate parent collection using the mean of each parent's real child embeddings
    # (global_parent_col is only ever fetched by ID, never vector-searched; using mean child vectors
    # stores valid 2048-dim embeddings with 0 wasted API calls!)
    todo_p = [i for i, pid in enumerate(parent_ids) if pid not in existing_parents]
    if todo_p:
        print(f"  [parent_child] {file_name}: storing {len(todo_p)}/{len(parent_ids)} parents...")
        child_recs = global_child_col.get(where={"source": file_name}, include=["embeddings", "metadatas"])
        by_parent: dict[str, list[np.ndarray]] = {}
        for emb, meta in zip(child_recs["embeddings"], child_recs["metadatas"]):
            by_parent.setdefault(meta["parent_id"], []).append(np.asarray(emb, dtype=np.float32))

        dim = len(child_recs["embeddings"][0])
        zero_vec = np.zeros(dim, dtype=np.float32)
        p_vecs = []
        for idx in todo_p:
            pid = parent_ids[idx]
            c_embs = by_parent.get(pid)
            vec = np.mean(c_embs, axis=0) if c_embs else zero_vec
            p_vecs.append(vec.tolist())

        # Upsert parents in batches of 200
        for start in range(0, len(todo_p), 200):
            slice_idxs = todo_p[start : start + 200]
            global_parent_col.upsert(
                ids=[parent_ids[j] for j in slice_idxs],
                documents=[p_texts[j] for j in slice_idxs],
                embeddings=p_vecs[start : start + 200],
                metadatas=[{"source": file_name} for _ in slice_idxs],
            )
        print(f"    [parent_child] {file_name}: stored all {len(parent_ids)} parents.")


def _index_ext_resumable(doc_key: str, pdf_path: Path, reuse_hybrid_for_large: bool = True):
    from ext_rag.ingest import file_hash, index_ext_file
    from ext_rag.rag_common import DOC_PREFIX, EMBED_MODEL, check_dims, open_store, stored_files
    from hybrid_rag.hybrid import EMBEDDING_MODEL, global_hybrid_col

    file_name = pdf_path.name
    col = open_store()
    stored = stored_files(col)
    digest = file_hash(pdf_path)

    if digest in stored and stored[digest]["complete"]:
        print(f"  [ext_rag] {file_name}: already indexed ({stored[digest]['stored']} chunks)")
        return
    if any(f["path"] == file_name and f["complete"] for f in stored.values()):
        print(f"  [ext_rag] {file_name}: already indexed by filename")
        return

    # For huge books (like 758-page Bishop) when EMBED_MODEL matches global_hybrid_collection
    # and DOC_PREFIX == "", mirror the exact 1000-char/200-overlap chunks + L2-normalized vectors
    # already embedded in global_hybrid_collection to avoid burning ~650 redundant free-tier calls.
    hybrid_recs = global_hybrid_col.get(
        where={"source": file_name},
        include=["documents", "embeddings", "metadatas"],
    )
    n_hybrid = len(hybrid_recs.get("ids") or [])
    if (
        reuse_hybrid_for_large
        and n_hybrid > 500
        and EMBED_MODEL == EMBEDDING_MODEL
        and DOC_PREFIX == ""
    ):
        print(f"  [ext_rag] {file_name}: importing {n_hybrid} L2-normalized embeddings from global_hybrid_collection (0 API calls)...")
        added = datetime.now().isoformat(timespec="seconds")
        docs = hybrid_recs["documents"]
        embs = hybrid_recs["embeddings"]
        metas = hybrid_recs["metadatas"]

        # Sort by page for deterministic chunk ordering
        order = sorted(range(n_hybrid), key=lambda k: (metas[k].get("page", 0), k))
        check_dims(col, len(embs[0]), record=True)
        for start in range(0, n_hybrid, 250):
            batch_idx = order[start : start + 250]
            norm_vecs = []
            for j in batch_idx:
                v = np.asarray(embs[j], dtype=np.float32)
                norm = float(np.linalg.norm(v)) or 1.0
                norm_vecs.append((v / norm).tolist())
            col.upsert(
                ids=[f"{digest}:{start + offset}" for offset in range(len(batch_idx))],
                embeddings=norm_vecs,
                documents=[docs[j] for j in batch_idx],
                metadatas=[{
                    "path": file_name,
                    "location": f"page {metas[j].get('page', '?')}",
                    "hash": digest,
                    "file_type": "pdf",
                    "added": added,
                    "chunk": start + offset,
                    "chunks": n_hybrid,
                } for offset, j in enumerate(batch_idx)],
            )
        print(f"    [ext_rag] {file_name}: imported {n_hybrid} chunks into ext_chunks_openrouter.")
        return

    print(f"  [ext_rag] {file_name}: running native ext_rag.ingest.index_ext_file...")
    index_ext_file(str(pdf_path))
    print(f"    [ext_rag] {file_name}: indexed via ext_rag.")


def run_index(fresh_ext_bishop: bool = False):
    print("=== Indexing all 4 PDFs from docs/ across all 3 ChromaDB store groups ===")
    for doc_key, info in EVAL_DOCS.items():
        pdf_path = DOCS_DIR / info["pdf"]
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing PDF: {pdf_path}")
        print(f"\n--- {info['label']} ({info['pdf']}) ---")
        with cost.track("retrieval_eval_index", info["pdf"]):
            _index_hybrid_resumable(doc_key, pdf_path)
            _index_ext_resumable(doc_key, pdf_path, reuse_hybrid_for_large=not fresh_ext_bishop)
            _index_parent_child_resumable(doc_key, pdf_path)

    print("\nIndexing complete across all 3 ChromaDB store groups!")


# ---------- Exact Retrieval Runners (Pre-Answer-LLM, Pre/Post Grading) ----------
def _enrich_chunks(q: dict, raw_chunks: list[dict]) -> tuple[list[dict], dict]:
    """Locates true PDF pages and computes deterministic relevance scores for each chunk."""
    enriched = []
    for rank, c in enumerate(raw_chunks, start=1):
        loc = locate_chunk(c.get("text", ""), hint_source=c.get("source"), hint_page=c.get("page"))
        scores = score_chunk(q, c.get("text", ""), loc["doc"], loc["pages"])
        enriched.append({
            "rank": rank,
            "source": c.get("source"),
            "meta_page": c.get("page"),
            "doc": loc["doc"],
            "pages": loc["pages"],
            "located_by": loc["located_by"],
            "sim": c.get("sim"),
            "rrf_score": c.get("rrf_score"),
            "kept": c.get("kept"),
            "ref_filtered": c.get("ref_filtered", False),
            "scores": scores,
            "text": c.get("text", ""),
        })
    cov = score_chunk_set_coverage(q, enriched)
    return enriched, cov


def _grade_binary_with_retry(client_ai, chat_model: str, query: str, doc_content: str) -> bool:
    """Exact same binary grader prompt as corrective_hybrid.py & corrective_parent_child.py, with backoff."""
    eval_prompt = f"""You are an objective document evaluator.
Assess whether the provided context snippet contains information relevant to answering the question.

Question:
{query}

Context Snippet:
{doc_content}

Respond strictly with 'yes' if relevant, or 'no' if irrelevant. Do not provide explanations."""

    def _call():
        resp = client_ai.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.0,
            max_tokens=5,
            extra_body={"reasoning": {"enabled": False}},
        )
        return (resp.choices[0].message.content or "").strip().lower()

    try:
        decision = call_with_backoff(_call)
        return "yes" in decision
    except DailyQuotaExhausted:
        raise
    except Exception as e:
        logger.warning("Grader failed after retries (%s); accepting chunk as fallback.", e)
        return True


def _retrieve_hybrid(q: dict, qcache: QueryEmbeddingCache, selected_files: list | None) -> dict:
    from hybrid_rag.hybrid import (
        bm25_tokenizer,
        client_ai,
        get_bm25_subset,
        global_hybrid_col,
    )

    user_query = q["question"]
    bm25, chunks, metadatas, where_filter = get_bm25_subset(selected_files)
    sub_queries = decompose_multi_questions(user_query)
    qvecs = qcache.get_many(sub_queries, client_ai)

    consolidated_scores: dict[str, float] = {}
    doc_sim: dict[str, float] = {}
    rrf_k = 60
    t_chunks_limit = min(10, len(chunks))

    for sub_q, query_vec in zip(sub_queries, qvecs):
        dense_chunks = []
        dense_results = global_hybrid_col.query(
            query_embeddings=[query_vec],
            n_results=t_chunks_limit,
            include=["documents", "embeddings"],
            where=where_filter,
        )
        if dense_results.get("documents") and dense_results["documents"]:
            dense_chunks = dense_results["documents"][0]
            dense_embs = dense_results["embeddings"][0]
            for d, emb in zip(dense_chunks, dense_embs):
                sim = _cosine(query_vec, emb)
                if sim is not None and (d not in doc_sim or sim > doc_sim[d]):
                    doc_sim[d] = sim

        sparse_chunks = bm25.get_top_n(bm25_tokenizer(sub_q), chunks, n=t_chunks_limit)
        for rank, doc in enumerate(dense_chunks):
            consolidated_scores[doc] = consolidated_scores.get(doc, 0.0) + (1.0 / (rrf_k + rank + 1))
        for rank, doc in enumerate(sparse_chunks):
            consolidated_scores[doc] = consolidated_scores.get(doc, 0.0) + (1.0 / (rrf_k + rank + 1))

    ranked_docs = sorted(consolidated_scores.items(), key=lambda item: item[1], reverse=True)[:6]
    chunk_to_meta = {c: m for c, m in zip(chunks, metadatas)}

    raw_chunks = []
    for doc, rrf in ranked_docs:
        meta = chunk_to_meta.get(doc, {}) or {}
        raw_chunks.append({
            "source": meta.get("source"),
            "page": meta.get("page"),
            "sim": doc_sim.get(doc),
            "rrf_score": round(rrf, 5),
            "text": doc,
        })
    return {
        "pre_chunks": raw_chunks,
        "post_chunks": raw_chunks,
        "expected_embed_requests": len(sub_queries),
        "grader_requests": 0,
    }


def _retrieve_crag_hybrid(q: dict, qcache: QueryEmbeddingCache, selected_files: list | None) -> dict:
    from hybrid_rag.corrective_hybrid import CHAT_MODEL, client_ai as crag_client
    from hybrid_rag.hybrid import (
        bm25_tokenizer,
        client_ai,
        get_bm25_subset,
        global_hybrid_col,
    )

    user_query = q["question"]
    bm25, chunks, metadatas, where_filter = get_bm25_subset(selected_files)
    sub_queries = decompose_multi_questions(user_query)
    qvecs = qcache.get_many(sub_queries, client_ai)

    chunk_to_queries: dict[str, list[str]] = {}
    ordered_chunks: list[str] = []
    doc_sim: dict[str, float] = {}
    doc_rrf: dict[str, float] = {}
    rrf_k = 60
    top_k_per_query = 4
    t_chunks_limit = min(10, len(chunks))

    for sub_q, query_vec in zip(sub_queries, qvecs):
        dense_chunks = []
        dense_results = global_hybrid_col.query(
            query_embeddings=[query_vec],
            n_results=t_chunks_limit,
            include=["documents", "embeddings"],
            where=where_filter,
        )
        if dense_results.get("documents") and dense_results["documents"]:
            dense_chunks = dense_results["documents"][0]
            dense_embs = dense_results["embeddings"][0]
            for d, emb in zip(dense_chunks, dense_embs):
                sim = _cosine(query_vec, emb)
                if sim is not None and (d not in doc_sim or sim > doc_sim[d]):
                    doc_sim[d] = sim

        sparse_chunks = bm25.get_top_n(bm25_tokenizer(sub_q), chunks, n=t_chunks_limit)
        branch_scores: dict[str, float] = {}
        for rank, doc in enumerate(dense_chunks):
            branch_scores[doc] = branch_scores.get(doc, 0.0) + (1.0 / (rrf_k + rank + 1))
        for rank, doc in enumerate(sparse_chunks):
            branch_scores[doc] = branch_scores.get(doc, 0.0) + (1.0 / (rrf_k + rank + 1))

        ranked_branch = sorted(branch_scores.items(), key=lambda item: item[1], reverse=True)[:top_k_per_query]
        for doc, score_val in ranked_branch:
            doc_rrf[doc] = max(doc_rrf.get(doc, 0.0), score_val)
            if doc not in chunk_to_queries:
                chunk_to_queries[doc] = []
                ordered_chunks.append(doc)
            chunk_to_queries[doc].append(sub_q)

    chunk_to_meta = {c: m for c, m in zip(chunks, metadatas)}
    pre_chunks = []
    post_chunks = []
    grader_calls = 0

    for doc in ordered_chunks:
        meta = chunk_to_meta.get(doc, {}) or {}
        is_rel = False
        for sub_q in chunk_to_queries[doc]:
            grader_calls += 1
            if _grade_binary_with_retry(crag_client, CHAT_MODEL, sub_q, doc):
                is_rel = True
                break
            time.sleep(0.3)

        item = {
            "source": meta.get("source"),
            "page": meta.get("page"),
            "sim": doc_sim.get(doc),
            "rrf_score": round(doc_rrf.get(doc, 0.0), 5),
            "kept": is_rel,
            "text": doc,
        }
        pre_chunks.append(item)
        if is_rel:
            post_chunks.append(dict(item))

    return {
        "pre_chunks": pre_chunks,
        "post_chunks": post_chunks,
        "expected_embed_requests": len(sub_queries),
        "grader_requests": grader_calls,
    }


def _retrieve_basic(q: dict, qcache: QueryEmbeddingCache, selected_files: list | None) -> dict:
    from hybrid_rag.hybrid import client_ai, get_bm25_subset, global_hybrid_col

    user_query = q["question"]
    _, chunks, _, where_filter = get_bm25_subset(selected_files)
    sub_queries = decompose_multi_questions(user_query)
    qvecs = qcache.get_many(sub_queries, client_ai)

    n_results = min(3, len(chunks))
    unique_docs: dict[str, dict] = {}
    for sub_q, query_vec in zip(sub_queries, qvecs):
        results = global_hybrid_col.query(
            query_embeddings=[query_vec],
            n_results=n_results,
            include=["documents", "metadatas", "embeddings"],
            where=where_filter,
        )
        if results.get("documents") and results["documents"]:
            for doc, meta, emb in zip(results["documents"][0], results["metadatas"][0], results["embeddings"][0]):
                sim = _cosine(query_vec, emb)
                if doc not in unique_docs:
                    unique_docs[doc] = {
                        "source": meta.get("source") if isinstance(meta, dict) else None,
                        "page": meta.get("page") if isinstance(meta, dict) else None,
                        "sim": sim,
                        "text": doc,
                    }

    raw_chunks = list(unique_docs.values())
    return {
        "pre_chunks": raw_chunks,
        "post_chunks": raw_chunks,
        "expected_embed_requests": len(sub_queries),
        "grader_requests": 0,
    }


def _retrieve_parent_child(q: dict, qcache: QueryEmbeddingCache, selected_files: list | None) -> dict:
    from other_rags.parent_child import (
        client_ai,
        get_child_where_filter,
        global_child_col,
        global_parent_col,
    )

    user_query = q["question"]
    where_filter, matching_count = get_child_where_filter(selected_files)
    child_n_results = min(10, matching_count)
    sub_queries = decompose_multi_questions(user_query)
    qvecs = qcache.get_many(sub_queries, client_ai)

    matched_parent_ids: list[str] = []
    parent_sim: dict[str, float] = {}
    for sub_q, query_vec in zip(sub_queries, qvecs):
        res = global_child_col.query(
            query_embeddings=[query_vec],
            n_results=child_n_results,
            include=["metadatas", "embeddings"],
            where=where_filter,
        )
        if res.get("metadatas") and res["metadatas"]:
            for meta, emb in zip(res["metadatas"][0], res["embeddings"][0]):
                pid = meta.get("parent_id")
                sim = _cosine(query_vec, emb)
                if pid:
                    if pid not in matched_parent_ids:
                        matched_parent_ids.append(pid)
                    if sim is not None and (pid not in parent_sim or sim > parent_sim[pid]):
                        parent_sim[pid] = sim

    top_pids = matched_parent_ids[:6]
    if not top_pids:
        return {"pre_chunks": [], "post_chunks": [], "expected_embed_requests": len(sub_queries), "grader_requests": 0}

    parent_recs = global_parent_col.get(ids=top_pids, include=["documents", "metadatas"])
    doc_map = dict(zip(parent_recs["ids"], parent_recs["documents"]))
    meta_map = dict(zip(parent_recs["ids"], parent_recs["metadatas"]))

    raw_chunks = []
    for pid in top_pids:
        if pid in doc_map:
            meta = meta_map.get(pid, {}) or {}
            raw_chunks.append({
                "source": meta.get("source"),
                "page": None,  # Parent-Child does not store page metadata; locate_chunk resolves true PDF pages
                "sim": parent_sim.get(pid),
                "text": doc_map[pid],
            })
    return {
        "pre_chunks": raw_chunks,
        "post_chunks": raw_chunks,
        "expected_embed_requests": len(sub_queries),
        "grader_requests": 0,
    }


def _retrieve_crag_parent_child(q: dict, qcache: QueryEmbeddingCache, selected_files: list | None) -> dict:
    from other_rags.corrective_parent_child import CHAT_MODEL, client_ai as crag_pc_client
    from other_rags.parent_child import (
        client_ai,
        get_child_where_filter,
        global_child_col,
        global_parent_col,
    )

    user_query = q["question"]
    where_filter, matching_count = get_child_where_filter(selected_files)
    child_n_results = min(10, matching_count)
    sub_queries = decompose_multi_questions(user_query)
    qvecs = qcache.get_many(sub_queries, client_ai)

    parent_to_queries: dict[str, list[str]] = {}
    ordered_parent_ids: list[str] = []
    parent_sim: dict[str, float] = {}
    max_parents_per_query = 4

    for sub_q, query_vec in zip(sub_queries, qvecs):
        res = global_child_col.query(
            query_embeddings=[query_vec],
            n_results=child_n_results,
            include=["metadatas", "embeddings"],
            where=where_filter,
        )
        sub_parents: list[str] = []
        if res.get("metadatas") and res["metadatas"]:
            for meta, emb in zip(res["metadatas"][0], res["embeddings"][0]):
                pid = meta.get("parent_id")
                sim = _cosine(query_vec, emb)
                if pid:
                    if sim is not None and (pid not in parent_sim or sim > parent_sim[pid]):
                        parent_sim[pid] = sim
                    if pid not in sub_parents:
                        sub_parents.append(pid)
                    if len(sub_parents) >= max_parents_per_query:
                        break
        for pid in sub_parents:
            if pid not in parent_to_queries:
                parent_to_queries[pid] = []
                ordered_parent_ids.append(pid)
            parent_to_queries[pid].append(sub_q)

    pre_chunks = []
    post_chunks = []
    grader_calls = 0
    if ordered_parent_ids:
        parent_recs = global_parent_col.get(ids=ordered_parent_ids, include=["documents", "metadatas"])
        doc_map = dict(zip(parent_recs["ids"], parent_recs["documents"]))
        meta_map = dict(zip(parent_recs["ids"], parent_recs["metadatas"]))

        for pid in ordered_parent_ids:
            if pid not in doc_map:
                continue
            text = doc_map[pid]
            meta = meta_map.get(pid, {}) or {}
            is_rel = False
            for sub_q in parent_to_queries[pid]:
                grader_calls += 1
                if _grade_binary_with_retry(crag_pc_client, CHAT_MODEL, sub_q, text):
                    is_rel = True
                    break
                time.sleep(0.3)

            item = {
                "source": meta.get("source"),
                "page": None,
                "sim": parent_sim.get(pid),
                "kept": is_rel,
                "text": text,
            }
            pre_chunks.append(item)
            if is_rel:
                post_chunks.append(dict(item))

    return {
        "pre_chunks": pre_chunks,
        "post_chunks": post_chunks,
        "expected_embed_requests": 1,  # corrective_parent_child batches sub_queries in 1 call
        "grader_requests": grader_calls,
    }


def _ext_where(col, selected_files: list | None):
    from ext_rag.rag_common import stored_files

    all_paths = {f["path"] for f in stored_files(col).values() if f.get("path")}
    if not selected_files or all_paths.issubset(set(selected_files)):
        return None, len(all_paths)
    sel = [p for p in selected_files if p in all_paths]
    return {"path": {"$in": sel}}, len(sel)


def _parse_ext_page(loc: str | None) -> int | None:
    if not loc:
        return None
    import re
    m = re.search(r"(\d+)", str(loc))
    return int(m.group(1)) if m else None


def _retrieve_ext_naive(q: dict, qcache: QueryEmbeddingCache, selected_files: list | None) -> dict:
    from ext_rag.rag_common import TOP_K, open_store, search
    from hybrid_rag.hybrid import client_ai

    col = open_store(check=False)
    where, _ = _ext_where(col, selected_files)
    raw_vec = np.asarray(qcache.get_one(q["question"], client_ai), dtype=np.float32)
    qvec = raw_vec / (float(np.linalg.norm(raw_vec)) or 1.0)
    hits = search(col, qvec, TOP_K, where=where)

    raw_chunks = [
        {
            "source": h["path"],
            "page": _parse_ext_page(h.get("loc")),
            "sim": round(float(h["sim"]), 4),
            "text": h["text"],
        }
        for h in hits
    ]
    return {
        "pre_chunks": raw_chunks,
        "post_chunks": raw_chunks,
        "expected_embed_requests": 1,
        "grader_requests": 0,
    }


def _retrieve_ext_crag(q: dict, qcache: QueryEmbeddingCache, selected_files: list | None) -> dict:
    import ext_rag.ask_crag as ask_crag
    from ext_rag.rag_common import (
        CANDIDATES,
        MAX_CONTEXT,
        MIN_SIMILARITY,
        REFERENCE_THRESHOLD,
        open_store,
    )
    from hybrid_rag.hybrid import client_ai

    col = open_store(check=False)
    where, active_files = _ext_where(col, selected_files)

    # Inject cached query embedding for the initial query so we don't duplicate it,
    # while letting corrective_retrieve run its real grader & any Round-2 extra queries!
    orig_embed = ask_crag.embed_texts

    def _cached_embed(texts, is_query=False, skip_rejected=False, step=None):
        if is_query and len(texts) == 1 and texts[0] == q["question"]:
            v = np.asarray(qcache.get_one(texts[0], client_ai), dtype=np.float32)
            v = v / (float(np.linalg.norm(v)) or 1.0)
            return np.vstack([v])
        return call_with_backoff(orig_embed, texts, is_query=is_query, skip_rejected=skip_rejected, step=step)

    log = ask_crag.Log()
    ask_crag.embed_texts = _cached_embed
    try:
        post_raw, status, missing = call_with_backoff(
            ask_crag.corrective_retrieve,
            col,
            q["question"],
            log,
            where=where,
            active_file_count=active_files,
            max_retries=4,
            base_delay=4.0,
        )
    finally:
        ask_crag.embed_texts = orig_embed

    post_capped = (post_raw or [])[:MAX_CONTEXT]
    post_ids = {c["id"] for c in post_capped}

    # Reconstruct exact initial pre-grading candidates locally (0 API calls) using log.qvec
    pre_cands = [
        c for c in ask_crag.knn(col, log.qvec, CANDIDATES, where=where)
        if c["sim"] >= MIN_SIMILARITY
    ]
    refs_removed = 0
    pre_chunks = []
    for c in pre_cands:
        is_ref = ask_crag.citation_density(c["text"]) >= REFERENCE_THRESHOLD
        if is_ref:
            refs_removed += 1
        pre_chunks.append({
            "source": c["path"],
            "page": _parse_ext_page(c.get("loc")),
            "sim": round(float(c["sim"]), 4),
            "kept": c["id"] in post_ids,
            "ref_filtered": is_ref,
            "text": c["text"],
        })

    post_chunks = [
        {
            "source": c["path"],
            "page": _parse_ext_page(c.get("loc")),
            "sim": round(float(c["sim"]), 4),
            "kept": True,
            "ref_filtered": False,
            "text": c["text"],
        }
        for c in post_capped
    ]

    round2 = any("Round 2:" in line for line in log)
    return {
        "pre_chunks": pre_chunks,
        "post_chunks": post_chunks,
        "status": status,
        "missing": missing,
        "round2_triggered": round2,
        "refs_removed": refs_removed,
        "expected_embed_requests": 1,
        "grader_requests": 2 if round2 else 1,
    }


RUNNERS = {
    "hybrid": _retrieve_hybrid,
    "crag_hybrid": _retrieve_crag_hybrid,
    "basic": _retrieve_basic,
    "parent_child": _retrieve_parent_child,
    "crag_parent_child": _retrieve_crag_parent_child,
    "naive": _retrieve_ext_naive,
    "crag": _retrieve_ext_crag,
}


def _parse_question_ids(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def run_eval(only: list[str] | None = None, question_spec: str | None = None, per_file: bool = False, force: bool = False):
    dataset = load_dataset()
    q_filter = _parse_question_ids(question_spec)
    qcache = QueryEmbeddingCache()

    # Pre-warm query embeddings once (just 5 tiny batch calls for all 80 questions!)
    from hybrid_rag.hybrid import client_ai
    all_sub_queries: list[str] = []
    for q in dataset:
        if q_filter and q["id"] not in q_filter:
            continue
        all_sub_queries.append(q["question"])
        all_sub_queries.extend(decompose_multi_questions(q["question"]))
    print(f"Pre-warming query embedding cache for {len(set(all_sub_queries))} unique query strings...")
    qcache.get_many(all_sub_queries, client_ai)

    for system, label, is_crag in ALL_PIPELINES:
        if only and system not in only:
            continue

        out_path = result_file(system, per_file=per_file)
        saved_list = [] if force else (_load_json(out_path, []) or [])
        results_by_id = {r["id"]: r for r in saved_list if not r.get("error")}

        print(f"\n=== {label} ({system}) | already done: {len(results_by_id)}/{len(dataset)} ===")
        runner = RUNNERS[system]

        for q in dataset:
            qid = q["id"]
            if q_filter and qid not in q_filter:
                continue
            if qid in results_by_id:
                continue

            selected_files = [q["source_file"]] if per_file else None
            started = time.perf_counter()
            try:
                with cost.track(f"ret_{system}", q["question"]) as qc:
                    res = runner(q, qcache, selected_files)
                elapsed = round(time.perf_counter() - started, 3)

                pre_enriched, pre_cov = _enrich_chunks(q, res["pre_chunks"])
                if is_crag:
                    post_enriched, post_cov = _enrich_chunks(q, res["post_chunks"])
                else:
                    post_enriched, post_cov = pre_enriched, pre_cov

                record = {
                    "id": qid,
                    "doc": q["doc"],
                    "category": q["category"],
                    "question": q["question"],
                    "pdf_pages": q["pdf_pages"],
                    "is_crag": is_crag,
                    "per_file": per_file,
                    "seconds": elapsed,
                    "error": None,
                    "status": res.get("status"),
                    "round2_triggered": res.get("round2_triggered", False),
                    "refs_removed": res.get("refs_removed", 0),
                    "pre_chunks": pre_enriched,
                    "pre_coverage": pre_cov,
                    "post_chunks": post_enriched,
                    "post_coverage": post_cov,
                    "expected_embed_requests": res.get("expected_embed_requests", 1),
                    "grader_requests": res.get("grader_requests", 0),
                    "cost": {
                        "usd": qc.total_usd,
                        "measured_requests": qc.n_calls,
                        "total_estimated_requests": res.get("expected_embed_requests", 1) + (
                            qc.n_calls if is_crag else 0
                        ),
                        "prompt_tokens": qc.prompt_tokens,
                        "completion_tokens": qc.completion_tokens,
                        "reasoning_tokens": qc.reasoning_tokens,
                    },
                }
                results_by_id[qid] = record
                _save_json(out_path, [results_by_id[i] for i in sorted(results_by_id)])

                pre_hit = any(c["scores"]["fact_match"] for c in pre_enriched)
                post_hit = any(c["scores"]["fact_match"] for c in post_enriched)
                st_str = f" [{res['status']}]" if res.get("status") else ""
                print(
                    f"  Q{qid:02d} ({q['doc']:7s}) {elapsed:5.1f}s | "
                    f"pre={len(pre_enriched)} (hit={'Y' if pre_hit else 'N'}) -> "
                    f"post={len(post_enriched)} (hit={'Y' if post_hit else 'N'}){st_str}",
                    flush=True,
                )
            except DailyQuotaExhausted as e:
                print(f"\n[QUOTA PAUSE] OpenRouter daily free limit reached at {system} Q{qid}: {e}")
                print(f"Progress saved ({len(results_by_id)} questions in {out_path}). Re-run to resume!")
                return False
            except Exception as e:
                logger.exception("%s Q%d failed", system, qid)
                print(f"  Q{qid:02d} ERROR: {e}", flush=True)
                results_by_id[qid] = {
                    "id": qid,
                    "doc": q["doc"],
                    "category": q["category"],
                    "question": q["question"],
                    "error": f"{type(e).__name__}: {e}",
                    "seconds": round(time.perf_counter() - started, 3),
                }
                _save_json(out_path, [results_by_id[i] for i in sorted(results_by_id)])
    return True


def main():
    setup_logging()
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("rag.retrieval_eval").setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Retrieval-only evaluation across all 7 RAG pipelines")
    parser.add_argument("action", choices=["dataset", "index", "run", "report", "all"])
    parser.add_argument("--only", nargs="*", choices=[k for k, _, _ in ALL_PIPELINES])
    parser.add_argument("--questions", help="Question IDs or ranges, e.g. 1-20,45")
    parser.add_argument("--per-file", action="store_true", help="Filter search to each question's PDF only")
    parser.add_argument("--force", action="store_true", help="Ignore checkpoints and re-run selected questions")
    parser.add_argument("--fresh-ext-bishop", action="store_true", help="Re-embed Bishop from scratch in ext_rag")
    parser.add_argument("--grader-model", help="Optional override for CHAT_MODEL during CRAG grading")
    args = parser.parse_args()

    if args.grader_model:
        os.environ["CHAT_MODEL"] = args.grader_model
        import hybrid_rag.corrective_hybrid as _ch
        import other_rags.corrective_parent_child as _cpc
        import ext_rag.rag_common as _erc
        _ch.CHAT_MODEL = args.grader_model
        _cpc.CHAT_MODEL = args.grader_model
        _erc.LLM_MODEL = args.grader_model
        print(f"Using grader model override: {args.grader_model}")

    if args.action in ("dataset", "all"):
        build_dataset(save_json=True)
    if args.action in ("index", "all"):
        run_index(fresh_ext_bishop=args.fresh_ext_bishop)
    if args.action in ("run", "all"):
        run_eval(only=args.only, question_spec=args.questions, per_file=args.per_file, force=args.force)
    if args.action in ("report", "all"):
        from evaluation.retrieval_report import generate_reports
        generate_reports(per_file=args.per_file)


if __name__ == "__main__":
    main()
