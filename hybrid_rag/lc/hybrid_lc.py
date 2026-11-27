"""Hybrid RAG built with LangChain LCEL (LangChain counterpart of ``hybrid_rag/hybrid.py``).

Mirrors the original pipeline step for step so both versions remain drop-in comparable:
1. Hybrid retrieval: Chroma dense vector search + Arabic-aware BM25 per sub-question fused via RRF.
2. Prompt construction: ``SYS_PROMPT`` + recent session turns + formatted context + user query.
3. Generation: configured ``CHAT_MODEL`` (temperature 0.2) via OpenRouter with cost tracking.
4. Persistence: saves the turn to ``chat_history.json`` under the active ``session_id`` so
   multi-chat sessions and history summarization work identically across both implementations.
"""

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from hybrid_rag.chat_history import persist_chat_history
from hybrid_rag.hybrid import GLOBAL_SESSION_ID, global_hybrid_col
from hybrid_rag.lc.components import (
    ANSWER_PROMPT,
    format_context,
    get_llm,
    history_messages,
    hybrid_search,
)
from hybrid_rag.logging_setup import setup_logging
from hybrid_rag.tracing import build_trace_config

setup_logging()
logger = logging.getLogger("rag.lc.hybrid")


def build_chain():
    """Builds the Hybrid RAG LCEL chain.

    Input keys: ``question``, ``top_k``, ``history_limit``, ``selected_files``, ``session_id``.
    Output adds: ``docs``, ``context``, ``history``, and ``answer``.
    """
    return (
        # 1. Retrieve top fused chunks via dense vector search + BM25 + RRF
        RunnablePassthrough.assign(
            docs=lambda x: hybrid_search(x["question"], x["top_k"], x["selected_files"])
        )
        # 2. Format retrieved context and load rolling chat history for the active session
        | RunnablePassthrough.assign(
            context=lambda x: format_context(x["docs"])[0],
            history=lambda x: history_messages(
                x["history_limit"], x.get("session_id", GLOBAL_SESSION_ID)
            ),
        )
        # 3. Render prompt -> invoke ChatOpenAI -> extract plain text answer
        | RunnablePassthrough.assign(answer=ANSWER_PROMPT | get_llm() | StrOutputParser())
    )


_CHAIN = None


def run_hybrid_lc(
    user_query: str,
    top_k: int = 6,
    history_limit: int = 5,
    selected_files: list = None,
    session_id: str = GLOBAL_SESSION_ID,
):
    """Answers one question with Hybrid RAG (LangChain LCEL).

    Returns ``(answer, labelled_contexts)`` — drop-in compatible with
    ``run_hybrid_chat_session()`` in ``hybrid_rag/hybrid.py``.
    """
    global _CHAIN
    logger.info("=== Starting Hybrid RAG (LangChain) ===")
    logger.info(f"Session: '{session_id}' | User Query: '{user_query}'")

    if global_hybrid_col.count() == 0:
        logger.error("Global Hybrid DB is empty. Please upload and index documents first.")
        return "The database is empty. Please upload and index some PDFs first.", []

    if _CHAIN is None:
        _CHAIN = build_chain()

    try:
        result = _CHAIN.invoke(
            {
                "question": user_query,
                "top_k": top_k,
                "history_limit": history_limit,
                "selected_files": selected_files,
                "session_id": session_id,
            },
            # Attach LangSmith run name, searchable tags, and session/file metadata (no-op if tracing off)
            config=build_trace_config(
                run_name="HybridLC",
                tags=["hybrid_lc", "lcel", "bilingual"],
                metadata={
                    "pipeline": "hybrid_lc",
                    "session_id": session_id,
                    "top_k": top_k,
                    "selected_files": selected_files or ["ALL"],
                },
            ),
        )
    except Exception as exc:
        logger.critical(f"LangChain Hybrid RAG execution failed: {exc}", exc_info=True)
        raise

    full_context_text, formatted_contexts = format_context(result["docs"])
    logger.info(
        f"Hybrid RAG (LangChain) completed: {len(formatted_contexts)} chunk(s) retrieved, "
        "answer generated successfully."
    )

    # 4. Persist the turn under session_id so multi-chat follow-ups & summarization stay in sync
    persist_chat_history(session_id, user_query, result["answer"], full_context_text)
    return result["answer"], formatted_contexts
