"""Shared LangChain building blocks for ``hybrid_lc.py`` and ``hybrid_corrective_lc.py``.

Mirrors ``hybrid_rag/hybrid.py`` and ``hybrid_rag/hybrid_corrective.py`` step for step on the
same ChromaDB collection, OpenRouter models, cost hook, Arabic-aware BM25 tokenizer, RRF fusion
(k = 60), and session-scoped chat history so both implementations remain drop-in comparable.
"""

import logging
import os
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

from hybrid_rag.chat_history import (
    DEFAULT_HISTORY_WINDOW,
    load_file_history,
    load_session_summary,
)
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
    return httpx.Client(timeout=30, event_hooks={"response": [record_cost]})


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
        "timeout": 30,
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


# Bound cached file-subset BM25 retrievers to prevent unbounded RAM growth
MAX_BM25_CACHE = 16
# In-memory cache of LangChain Documents (rebuilt only when ChromaDB's chunk count changes)
_CORPUS = {"count": -1, "docs": []}
# LRU cache of BM25ChunkRetriever instances keyed by (corpus_count, file_filter_str, k)
_BM25_CACHE: OrderedDict = OrderedDict()


def all_chunks():
    """Every chunk in the collection as a LangChain Document (with id + source/page metadata),
    sharing hybrid.py's in-memory cache so IDs and embeddings are loaded only once."""
    refresh_global_bm25()
    count = GLOBAL_BM25_CACHE["doc_count"]
    # Rebuild the Document list and invalidate BM25 retrievers only when documents are added/deleted
    if count != _CORPUS["count"]:
        ids = GLOBAL_BM25_CACHE["ids"]
        texts = GLOBAL_BM25_CACHE["chunks"]
        metas = GLOBAL_BM25_CACHE["metadatas"]
        # Store chunk_id inside metadata too so downstream CRAG steps can look up cached embeddings
        _CORPUS["docs"] = [
            Document(id=cid, page_content=t, metadata={**(m or {}), "chunk_id": cid})
            for cid, t, m in zip(ids, texts, metas)
        ]
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
        # Only filter when the user selected a strict subset of the indexed corpus
        if not sources <= selected:
            return (
                [d for d in docs if d.metadata.get("source") in selected],
                {"source": {"$in": sorted(selected)}},
            )
    return docs, None


def bm25_retriever(docs, where, k):
    """Returns an LRU-cached BM25ChunkRetriever scoped to ``(corpus_count, where_filter, k)``."""
    key = (_CORPUS["count"], str(where), k)
    # Build once per unique (corpus version, file scope, k) and evict oldest beyond MAX_BM25_CACHE
    if key in _BM25_CACHE:
        _BM25_CACHE.move_to_end(key)
        return _BM25_CACHE[key]
    logger.debug(f"[LC] Building cached BM25ChunkRetriever for {len(docs)} chunk(s) (k={k}).")
    retriever = BM25ChunkRetriever.from_documents(docs, k=k)
    _BM25_CACHE[key] = retriever
    if len(_BM25_CACHE) > MAX_BM25_CACHE:
        _BM25_CACHE.popitem(last=False)
    return retriever


# ---------- Hybrid retrieval ----------
def reciprocal_rank_fusion(ranked_lists, k=RRF_K):
    """Merge ranked lists of Documents: each chunk scores 1 / (k + rank) in every list it appears
    in (rank 1 = best), summed. Chunks are identified by their text, as in hybrid.py."""
    scores, first_seen = {}, {}
    for docs in ranked_lists:
        for rank, d in enumerate(docs, start=1):
            # Accumulate reciprocal rank contribution across dense + sparse lists & sub-queries
            scores[d.page_content] = scores.get(d.page_content, 0.0) + 1.0 / (k + rank)
            # Keep the first Document object seen for metadata preservation
            first_seen.setdefault(d.page_content, d)
    ranked = sorted(scores, key=scores.get, reverse=True)       # stable: ties keep first order
    return [first_seen[text] for text in ranked]


def hybrid_search(question, top_n, selected_files=None):
    """Top top_n chunks for one (sub-)question or a whole question: for each sub-question,
    vector search + BM25 (PER_BRANCH each, parallelized when >1 sub-question), then RRF."""
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

    # Split compound questions and retrieve each sub-question in parallel when >1 sub-question
    ranked_lists = []
    if len(sub_queries) == 1:
        ranked_lists.extend(_search_one(sub_queries[0]))
    else:
        with ThreadPoolExecutor(max_workers=min(4, len(sub_queries))) as pool:
            for pair in pool.map(_search_one, sub_queries):
                ranked_lists.extend(pair)

    # Fuse all dense & sparse rankings into the final deduplicated top_n Documents
    fused = reciprocal_rank_fusion(ranked_lists)[:top_n]
    logger.info(f"[LC] Hybrid RRF selected top {len(fused)} unique chunk(s) across sub-queries.")
    return fused


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


def history_messages(limit=DEFAULT_HISTORY_WINDOW, session_id=GLOBAL_SESSION_ID):
    """Loads the rolling earlier-turn summary (if any) plus the last ``limit`` verbatim turns
    for ``session_id`` as LangChain messages."""
    # Stateless callers pass limit=0 to avoid injecting chat history
    if not limit:
        return []

    messages = []
    # Prepend the rolling summary of older compacted turns as a SystemMessage so the LLM
    # never loses early conversation context even after 20+ questions.
    earlier_summary, summarized_count = load_session_summary(session_id)
    if earlier_summary:
        messages.append(
            SystemMessage(
                content=(
                    f"[Summary of {summarized_count} earlier conversation turn(s) in this chat]:\n"
                    f"{earlier_summary}"
                )
            )
        )

    # Append the latest `limit` full Q&A turns verbatim (default 6) for exact follow-up handling
    all_turns = load_file_history(session_id)
    turns = all_turns[-limit:]
    logger.info(
        f"[LC] Using {len(turns)} recent turn(s) + "
        f"{summarized_count} summarized earlier turn(s) from session '{session_id}'."
    )
    for t in turns:
        messages += [HumanMessage(content=t["query"]), AIMessage(content=t["answer"])]
    return messages
