# Shared LangChain building blocks for the LangChain versions of the two pipelines.
#
# They mirror hybrid_rag/hybrid.py step for step, on the same data, so the results can be
# compared with the originals:
#   - same ChromaDB collection and vectors (global_hybrid_collection; nothing is re-indexed)
#   - same models (CHAT_MODEL and EMBEDDING_MODEL from .env), through OpenRouter
#   - same cost tracking: the HTTP client carries the hook from pdf_processing.py
#   - same retrieval: for each sub-question, the top 10 chunks by vector search and the top 10
#     by BM25 keyword search, fused with Reciprocal Rank Fusion (k = 60), best top_k kept
#   - same prompt: SYS_PROMPT + the last turns of the chat history + context and question
#
# What LangChain provides here: the chat model and embeddings (ChatOpenAI, OpenAIEmbeddings),
# the vector store (Chroma), the retriever interface (BaseRetriever), documents with metadata
# (Document) and the prompt template (ChatPromptTemplate).

import os
import logging
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
from hybrid_rag.hybrid import GLOBAL_SESSION_ID, bm25_tokenizer, chroma_client, global_hybrid_col
from hybrid_rag.pdf_processing import SYS_PROMPT, _record_cost, decompose_multi_questions

load_dotenv()
logger = logging.getLogger("rag.lc")

OPENROUTER_URL = "https://openrouter.ai/api/v1"
COLLECTION = "global_hybrid_collection"
RRF_K = 60                 # Reciprocal Rank Fusion constant (same as hybrid.py)
PER_BRANCH = 10            # chunks taken from each search (vector, BM25) per sub-question


# ---------- Models (OpenRouter, with cost tracking) ----------
def _http_client():
    """The HTTP client the models use: the same cost hook as initialize_api(), so every
    LangChain request is recorded in hybrid_rag/cost.py like the original pipelines' ones."""
    return httpx.Client(event_hooks={"response": [_record_cost]})


@lru_cache(maxsize=None)
def get_llm(temperature=0.2, max_tokens=None):
    """CHAT_MODEL through OpenRouter. hybrid.py answers at temperature 0.2; the corrective
    grader uses temperature 0 and at most 5 tokens."""
    return ChatOpenAI(model=os.getenv("CHAT_MODEL"), base_url=OPENROUTER_URL,
                      api_key=os.getenv("OPENROUTER_API_KEY"), temperature=temperature,
                      max_tokens=max_tokens, timeout=30, http_client=_http_client())


@lru_cache(maxsize=None)
def get_embeddings():
    """EMBEDDING_MODEL through OpenRouter. check_embedding_ctx_length=False sends the text
    itself (not OpenAI token ids), as the original pipelines do."""
    return OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"), base_url=OPENROUTER_URL,
                            api_key=os.getenv("OPENROUTER_API_KEY"),
                            check_embedding_ctx_length=False, timeout=30,
                            http_client=_http_client())


@lru_cache(maxsize=None)
def get_vector_store():
    """The existing collection as a LangChain vector store. It uses hybrid.py's Chroma client,
    so there is only one client for the folder (two with different settings would clash)."""
    return Chroma(client=chroma_client, collection_name=COLLECTION,
                  embedding_function=get_embeddings())


# ---------- BM25 keyword search ----------
class BM25ChunkRetriever(BaseRetriever):
    """BM25 keyword search as a LangChain retriever: rank_bm25 with hybrid.py's tokenizer, so
    it ranks chunks exactly like the original. (Written here rather than taken from
    langchain-community, which is being discontinued.)"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    docs: list
    bm25: BM25Okapi
    k: int = PER_BRANCH

    @classmethod
    def from_documents(cls, docs, k=PER_BRANCH):
        return cls(docs=docs, bm25=BM25Okapi([bm25_tokenizer(d.page_content) for d in docs]), k=k)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        return self.bm25.get_top_n(bm25_tokenizer(query), self.docs, n=self.k)


_CORPUS = {"count": -1, "docs": []}
_BM25_CACHE = {}


def all_chunks():
    """Every chunk in the collection as a Document (text + source/page), re-read when the
    collection changes (a file indexed or removed)."""
    count = global_hybrid_col.count()
    if count != _CORPUS["count"]:
        got = global_hybrid_col.get(include=["documents", "metadatas"])
        _CORPUS["docs"] = [Document(page_content=t, metadata=m or {})
                           for t, m in zip(got["documents"], got["metadatas"])]
        _CORPUS["count"] = count
        _BM25_CACHE.clear()
    return _CORPUS["docs"]


def scope(selected_files=None):
    """(chunks, Chroma filter) for the files selected in the app; all files when none (or all)
    are selected. Same rule as get_bm25_subset() in hybrid.py."""
    docs = all_chunks()
    if selected_files:
        selected = set(selected_files)
        sources = {d.metadata.get("source") for d in docs if d.metadata.get("source")}
        if not sources <= selected:
            return ([d for d in docs if d.metadata.get("source") in selected],
                    {"source": {"$in": sorted(selected)}})
    return docs, None


def bm25_retriever(docs, where, k):
    key = (_CORPUS["count"], str(where), k)
    if key not in _BM25_CACHE:
        _BM25_CACHE[key] = BM25ChunkRetriever.from_documents(docs, k=k)
    return _BM25_CACHE[key]


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
    vector search + BM25 (PER_BRANCH each), then Reciprocal Rank Fusion over all the lists."""
    docs, where = scope(selected_files)
    if not docs:
        return []
    n = min(PER_BRANCH, len(docs))
    bm25 = bm25_retriever(docs, where, n)
    store = get_vector_store()
    ranked_lists = []
    for sub_q in decompose_multi_questions(question):
        try:
            dense = store.similarity_search(sub_q, k=n, filter=where)
        except Exception as e:                       # like hybrid.py: keep the BM25 results
            logger.error(f"Dense search failed for sub-query '{sub_q}': {e}", exc_info=True)
            dense = []
        ranked_lists += [dense, bm25.invoke(sub_q)]
    return reciprocal_rank_fusion(ranked_lists)[:top_n]


def format_chunk(doc):
    """A chunk with the label the original pipelines give it."""
    return (f"[Source: {doc.metadata.get('source', 'Unknown_File')} | "
            f"Page: {doc.metadata.get('page', '?')}]:\n{doc.page_content}")


def format_context(docs):
    """(full context text for the prompt, list of labelled chunks for the app)."""
    labelled = [format_chunk(d) for d in docs]
    return "\n\n---\n\n".join(labelled), labelled


# ---------- Prompt and chat history ----------
# The same messages hybrid.py sends: system prompt, earlier turns, then context + question.
# SYS_PROMPT is passed as a message (not a template string), so braces in it can't break it.
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYS_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "[Context from Global Database]:\n{context}\n\n[User Question(s)]:\n{question}\n"),
])


def history_messages(limit=5):
    """The last `limit` turns of the shared chat history as LangChain messages."""
    turns = load_file_history(GLOBAL_SESSION_ID)[-limit:] if limit else []
    messages = []
    for t in turns:
        messages += [HumanMessage(content=t["query"]), AIMessage(content=t["answer"])]
    return messages
