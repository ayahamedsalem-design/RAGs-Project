"""Shared LangChain building blocks for ``hybrid_lc.py`` and ``hybrid_corrective_lc.py``.

Mirrors ``hybrid_rag/hybrid.py`` and ``hybrid_rag/hybrid_corrective.py`` step for step on the
same ChromaDB collection, OpenRouter models, cost hook, Arabic-aware BM25 tokenizer, RRF fusion
(k = 60), and session-scoped chat history so both implementations remain drop-in comparable.
"""

import logging
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import httpx
from dotenv import load_dotenv
from pydantic import ConfigDict
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from hybrid_rag.chat_history import load_file_history
from hybrid_rag.file_processing import (
    SYS_PROMPT,
    decompose_multi_questions,
    format_location,
    record_cost,
)
from hybrid_rag.hybrid import (
    GLOBAL_BM25_CACHE,
    GLOBAL_SESSION_ID,
    bm25_tokenizer,
    chroma_client,
    global_hybrid_col,
    refresh_global_bm25,
)
from hybrid_rag.logging_setup import setup_logging
from hybrid_rag.tracing import traceable

try:
    from flashrank import Ranker, RerankRequest
except ImportError:
    Ranker = None
    RerankRequest = None

setup_logging()
load_dotenv()
logger = logging.getLogger("rag.lc")

OPENROUTER_URL = "https://openrouter.ai/api/v1"
COLLECTION = "global_hybrid_collection"
RRF_K = 60                 # Reciprocal Rank Fusion constant (same as hybrid.py)
PER_BRANCH = 10            # default chunks taken from each search (vector, BM25) per sub-question


# ---------- Models (OpenRouter, with cost tracking) ----------
def _http_client():
    """The HTTP client the models use: the same cost hook as initialize_api(), so every
    LangChain request is recorded in hybrid_rag/cost.py like the original pipelines' ones."""
    return httpx.Client(
        timeout=httpx.Timeout(45.0, connect=15.0),
        event_hooks={"response": [record_cost]},
    )


@lru_cache(maxsize=None)
def get_llm(temperature=0.2, max_tokens=None, disable_reasoning=False):
    """CHAT_MODEL through OpenRouter. Answer steps use temperature 0.2; the CRAG batch
    grader uses temperature 0 with reasoning disabled for fast, deterministic JSON."""
    kwargs = {
        "model": os.getenv("CHAT_MODEL"),
        "base_url": OPENROUTER_URL,
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": 45,
        "max_retries": 1,
        "http_client": _http_client(),
    }
    if disable_reasoning:
        kwargs["extra_body"] = {"reasoning": {"enabled": False}}
    return ChatOpenAI(**kwargs)


@lru_cache(maxsize=None)
def get_embeddings():
    """EMBEDDING_MODEL through OpenRouter. check_embedding_ctx_length=False sends the text
    itself (not OpenAI token ids), as the original pipelines do."""
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        base_url=OPENROUTER_URL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        check_embedding_ctx_length=False,
        timeout=30,
        max_retries=1,
        http_client=_http_client(),
    )


@lru_cache(maxsize=None)
def get_vector_store():
    """The existing collection as a LangChain vector store. It uses hybrid.py's Chroma client,
    so there is only one client for the folder (two with different settings would clash)."""
    return Chroma(
        client=chroma_client,
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
    )


# ---------- BM25 keyword search ----------
class BM25ChunkRetriever(BaseRetriever):
    """BM25 keyword search as a LangChain retriever: rank_bm25 with hybrid.py's Arabic-aware
    tokenizer, so it ranks chunks identically to hybrid.py."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    docs: list
    bm25: BM25Okapi
    k: int = PER_BRANCH

    @classmethod
    def from_documents(cls, docs, k=PER_BRANCH):
        """Builds a BM25Okapi index over ``docs`` using the Arabic-aware tokenizer."""
        return cls(docs=docs, bm25=BM25Okapi([bm25_tokenizer(d.page_content) for d in docs]), k=k)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        """Returns the top-k matching Documents for ``query`` via BM25 keyword scoring."""
        return self.bm25.get_top_n(bm25_tokenizer(query), self.docs, n=self.k)


MAX_BM25_CACHE = 16  # Bound cached file-subset BM25 retrievers to prevent unbounded RAM growth
_CORPUS = {"count": -1, "docs": [], "id_to_idx": {}}
_BM25_CACHE: OrderedDict = OrderedDict()


def all_chunks():
    """Every chunk in the collection as a LangChain Document (with id + source/page metadata),
    sharing hybrid.py's in-memory cache so IDs and embeddings are loaded only once."""
    refresh_global_bm25()
    count = GLOBAL_BM25_CACHE["doc_count"]
    if count != _CORPUS["count"]:
        ids = GLOBAL_BM25_CACHE["ids"]
        texts = GLOBAL_BM25_CACHE["chunks"]
        metas = GLOBAL_BM25_CACHE["metadatas"]
        docs = [
            Document(id=cid, page_content=t, metadata={**(m or {}), "chunk_id": cid})
            for cid, t, m in zip(ids, texts, metas)
        ]
        _CORPUS["docs"] = docs
        _CORPUS["id_to_idx"] = {cid: idx for idx, cid in enumerate(ids)}
        _CORPUS["count"] = count
        _BM25_CACHE.clear()
        logger.debug(f"[LC] Refreshed LangChain Document corpus cache ({count} chunk(s)).")
    return _CORPUS["docs"]


def get_cached_id_maps():
    """Returns (doc_to_id, id_to_emb) from hybrid.py's shared in-memory cache."""
    refresh_global_bm25()
    return GLOBAL_BM25_CACHE["doc_to_id"], GLOBAL_BM25_CACHE["id_to_emb"]


def scope(selected_files=None):
    """(chunks, Chroma filter) for the files selected in the app; all files when none (or all)
    are selected. Same rule as get_bm25_subset() in hybrid.py."""
    docs = all_chunks()
    if selected_files:
        selected = set(selected_files)
        sources = {d.metadata.get("source") for d in docs if d.metadata.get("source")}
        if not sources <= selected:
            return (
                [d for d in docs if d.metadata.get("source") in selected],
                {"source": {"$in": sorted(selected)}},
            )
    return docs, None


def bm25_retriever(docs, where, k):
    """Returns an LRU-cached BM25ChunkRetriever scoped to ``(corpus_count, where_filter, k)``."""
    # Cache key includes corpus count so adding/deleting a PDF automatically invalidates stale indexes
    key = (_CORPUS["count"], str(where), k)
    if key in _BM25_CACHE:
        # Mark entry as most-recently-used for LRU eviction ordering
        _BM25_CACHE.move_to_end(key)
        return _BM25_CACHE[key]
    logger.debug(f"[LC] Building cached BM25ChunkRetriever for {len(docs)} chunk(s) (k={k}).")
    retriever = BM25ChunkRetriever.from_documents(docs, k=k)
    _BM25_CACHE[key] = retriever
    # Evict the least-recently-used BM25 index once cache size exceeds MAX_BM25_CACHE (16)
    if len(_BM25_CACHE) > MAX_BM25_CACHE:
        _BM25_CACHE.popitem(last=False)
    return retriever


def _pages_adjacent(meta_a: dict, meta_b: dict) -> bool:
    """Returns True if two chunks from the same document are on the same or adjacent page."""
    pa, pb = meta_a.get("page"), meta_b.get("page")
    if isinstance(pa, int) and isinstance(pb, int):
        return abs(pa - pb) <= 1
    # Non-integer locations (e.g., Slides/Sections/Sheets) are allowed through
    return True


def expand_with_neighbors(
    docs: list[Document],
    max_total: int = 6,
    max_neighbors: int = 2,
    is_noise_fn=None,
) -> list[Document]:
    """Expands a small verified chunk set with immediate adjacent (+1 / -1) chunks from the
    exact same source file so lists, multi-sentence facts, or numbers split across chunk
    boundaries are recovered without extra API calls."""
    # Skip expansion if no verified chunks survived or we are already at the MAX_CONTEXT budget
    if not docs or len(docs) >= max_total or max_neighbors <= 0:
        return docs

    corpus = all_chunks()
    id_to_idx = _CORPUS["id_to_idx"]
    existing_ids = {d.metadata.get("chunk_id") or d.id for d in docs}
    expanded = list(docs)
    added = 0

    # Prioritise +1 continuation chunks first (lists/explanations usually continue forward);
    # also allow -1 preceding context when only 1 chunk survived grading.
    offsets = (1, -1) if len(docs) == 1 else (1,)
    # Only expand around the top 2 highest-ranked verified chunks to keep context tight
    for base_doc in docs[:2]:
        if len(expanded) >= max_total or added >= max_neighbors:
            break
        base_id = base_doc.metadata.get("chunk_id") or base_doc.id
        base_idx = id_to_idx.get(base_id)
        if base_idx is None:
            continue
        base_src = base_doc.metadata.get("source")
        base_sim = base_doc.metadata.get("sim")

        for delta in offsets:
            if len(expanded) >= max_total or added >= max_neighbors:
                break
            nb_idx = base_idx + delta
            if not (0 <= nb_idx < len(corpus)):
                continue
            nb = corpus[nb_idx]
            nb_id = nb.metadata.get("chunk_id") or nb.id
            # Guardrails: no duplicates, must come from exact same PDF, same/adjacent page, and non-noise
            if nb_id in existing_ids:
                continue
            if nb.metadata.get("source") != base_src:
                continue
            if not _pages_adjacent(base_doc.metadata, nb.metadata):
                continue
            if is_noise_fn is not None and is_noise_fn(nb):
                continue

            existing_ids.add(nb_id)
            nb_meta = {**nb.metadata, "via": f"neighbor ({delta:+d})"}
            # Assign a slightly lower similarity than the anchor chunk so verified chunks remain ranked first
            if base_sim is not None:
                nb_meta["sim"] = max(0.0, float(base_sim) - 0.001)
            expanded.append(Document(id=nb_id, page_content=nb.page_content, metadata=nb_meta))
            added += 1

    if added > 0:
        logger.info(f"[LC] Neighbor expansion added {added} adjacent chunk(s) (total={len(expanded)}).")
    return expanded


# ---------- Local ONNX Multilingual Reranker (FlashRank) ----------
DEFAULT_RERANK_MODEL = "ms-marco-MultiBERT-L-12"
# Serialize ONNX inference across worker threads so multi-threaded CPU ops don't thrash
_RERANK_LOCK = threading.Lock()


def is_rerank_enabled() -> bool:
    """Returns True when local FlashRank ONNX reranking is enabled and installed."""
    if Ranker is None:
        return False
    flag = (os.getenv("RERANK_ENABLED") or "true").strip().lower()
    return flag not in ("0", "false", "no", "off")


@lru_cache(maxsize=1)
def get_reranker():
    """Loads the FlashRank Multilingual ONNX cross-encoder once (no PyTorch/GPU required)."""
    if not is_rerank_enabled():
        return None
    model_name = (os.getenv("RERANK_MODEL") or DEFAULT_RERANK_MODEL).strip()
    # Store the downloaded ~98 MB ONNX model under project-local models/flashrank_cache/
    cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "models",
        "flashrank_cache",
    )
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"[LC] Loading FlashRank ONNX Reranker '{model_name}' (max_length=256)...")
    # max_length=256 keeps CPU ONNX latency fast while covering query + leading chunk evidence
    return Ranker(model_name=model_name, cache_dir=cache_dir, max_length=256)


@traceable(name="flashrank_rerank", run_type="retriever")
def rerank_documents(query: str, docs: list[Document], top_n: int) -> list[Document]:
    """Reranks candidate Documents with FlashRank Multilingual ONNX, using saturation-aware
    rank fusion with the incoming RRF order so strong Qwen3/BM25 Arabic hits are never
    demoted when MultiBERT scores saturate."""
    if not docs:
        return []
    if len(docs) <= 1 or not is_rerank_enabled():
        return docs[:top_n]
    try:
        ranker = get_reranker()
        if ranker is None:
            return docs[:top_n]
        # Trim passages to 600 chars to keep CPU tokenization & ONNX inference bounded
        passages = [{"id": idx, "text": d.page_content[:600]} for idx, d in enumerate(docs)]
        # Acquire _RERANK_LOCK so parallel queries don't contend on ONNX CPU thread pools
        with _RERANK_LOCK:
            ranked = ranker.rerank(RerankRequest(query=query, passages=passages))
        scores = [float(item.get("score", 0.0)) for item in ranked]
        # Score spread reveals whether MultiBERT separated candidates or saturated near 0.999
        spread = (max(scores) - min(scores)) if scores else 0.0

        # Build FlashRank rank map (1-based) and score map
        fr_rank = {int(item["id"]): r for r, item in enumerate(ranked, start=1)}
        fr_score = {int(item["id"]): float(item.get("score", 0.0)) for item in ranked}

        # When MultiBERT scores are saturated (spread < 0.05, common on classical Arabic OCR),
        # keep heavy weight on Qwen3+BM25 RRF order; when discriminative, weight FlashRank higher.
        fr_weight = 0.25 if spread < 0.05 else 1.5
        rrf_weight = 1.0

        # Blend incoming RRF rank and FlashRank cross-encoder rank via weighted Reciprocal Rank Fusion
        blended = []
        for orig_rank, d in enumerate(docs, start=1):
            idx = orig_rank - 1
            comb = (
                rrf_weight / (RRF_K + orig_rank)
                + fr_weight / (RRF_K + fr_rank.get(idx, len(docs)))
            )
            blended.append((comb, idx, d))

        blended.sort(key=lambda t: t[0], reverse=True)
        # Preserve original metadata and attach `rerank_score` for downstream debugging/tracing
        reranked_docs = [
            Document(
                id=d.id,
                page_content=d.page_content,
                metadata={**d.metadata, "rerank_score": round(fr_score.get(idx, 0.0), 4)},
            )
            for _, idx, d in blended[:top_n]
        ]
        logger.debug(
            f"[LC] FlashRank reranked {len(docs)} -> top {len(reranked_docs)} "
            f"(spread={spread:.4f}, fr_weight={fr_weight})."
        )
        return reranked_docs
    except Exception as exc:
        # Fail-open: if ONNX reranking raises any runtime error, fall back cleanly to RRF ordering
        logger.warning(f"[LC] FlashRank reranking failed ({exc}); keeping RRF ordering.")
        return docs[:top_n]


# ---------- Hybrid retrieval ----------
def reciprocal_rank_fusion(ranked_lists, k=RRF_K):
    """Merge ranked lists of Documents: each chunk scores 1 / (k + rank) in every list it appears
    in (rank 1 = best), summed. Chunks are identified by their text, as in hybrid.py."""
    scores, first_seen = {}, {}
    for docs in ranked_lists:
        for rank, d in enumerate(docs, start=1):
            scores[d.page_content] = scores.get(d.page_content, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(d.page_content, d)
    ranked = sorted(scores, key=scores.get, reverse=True)       # stable: ties keep first order
    return [first_seen[text] for text in ranked]


def hybrid_search(question, top_n, selected_files=None):
    """Top top_n chunks for one (sub-)question or a whole question: for each sub-question,
    vector search + BM25 (PER_BRANCH each, parallelized when >1 sub-question), then RRF,
    followed by local FlashRank Multilingual ONNX reranking when enabled."""
    docs, where = scope(selected_files)
    if not docs:
        return []
    n = min(PER_BRANCH, len(docs))
    bm25 = bm25_retriever(docs, where, n)
    store = get_vector_store()
    sub_queries = decompose_multi_questions(question)

    def _search_one(sub_q: str) -> list[list[Document]]:
        try:
            dense = store.similarity_search(sub_q, k=n, filter=where)
        except Exception as e:                       # like hybrid.py: keep the BM25 results
            logger.error(f"Dense search failed for sub-query '{sub_q}': {e}", exc_info=True)
            dense = []
        return [dense, bm25.invoke(sub_q)]

    # Single sub-query runs inline to avoid thread-pool overhead; multi-part queries run in parallel
    ranked_lists = []
    if len(sub_queries) == 1:
        ranked_lists.extend(_search_one(sub_queries[0]))
    else:
        with ThreadPoolExecutor(max_workers=min(4, len(sub_queries))) as pool:
            for pair in pool.map(_search_one, sub_queries):
                ranked_lists.extend(pair)

    # Pull a wider candidate pool (at least 12) for cross-encoder reranking when FlashRank is active
    pool_limit = max(top_n * 2, 12) if is_rerank_enabled() else top_n
    fused = reciprocal_rank_fusion(ranked_lists)[:pool_limit]
    final_docs = rerank_documents(question, fused, top_n=top_n)
    logger.info(f"[LC] Hybrid RRF + Rerank selected top {len(final_docs)} unique chunk(s).")
    return final_docs


def format_chunk(doc):
    """A chunk with its source/location label (Page, Slide, Sheet/Rows, Table, Section)
    plus sim/via when populated by Hybrid Corrective."""
    src = doc.metadata.get("source", "Unknown_File")
    loc = format_location(doc.metadata)
    sim = doc.metadata.get("sim")
    via = doc.metadata.get("via")
    if sim is not None and via:
        return f"[Source: {src} | {loc} | sim: {sim:.3f} | {via}]:\n{doc.page_content}"
    return f"[Source: {src} | {loc}]:\n{doc.page_content}"


def format_context(docs):
    """(full context text for the prompt, list of labelled chunks for the app)."""
    labelled = [format_chunk(d) for d in docs]
    return "\n\n---\n\n".join(labelled), labelled


# ---------- Prompt and chat history ----------
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYS_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "[Context from Global Database]:\n{context}\n\n[User Question(s)]:\n{question}\n"),
])


def history_messages(limit=5, session_id=GLOBAL_SESSION_ID):
    """Loads the last ``limit`` turns for ``session_id`` as LangChain HumanMessage/AIMessage pairs."""
    all_turns = load_file_history(session_id)
    turns = all_turns[-limit:] if limit else []
    logger.info(
        f"[LC] Using {len(turns)} recent turn(s) from session '{session_id}' "
        f"(Total saved: {len(all_turns)})."
    )
    messages = []
    for t in turns:
        messages += [HumanMessage(content=t["query"]), AIMessage(content=t["answer"])]
    return messages
