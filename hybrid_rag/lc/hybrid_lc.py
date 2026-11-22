# Hybrid RAG built with LangChain: the LangChain version of hybrid_rag/hybrid.py.
#
# It does the same steps, on the same data, with the same settings, so the two can be compared:
#   1. hybrid retrieval: vector search + BM25 per sub-question, Reciprocal Rank Fusion, top 6
#   2. prompt: system prompt + last 5 turns of the chat history + context + question
#   3. answer: CHAT_MODEL at temperature 0.2
#   4. the turn is saved to the shared chat history (chat_history.json)
#
# The steps are joined into one LangChain chain (LCEL): each step adds its result to a dict
# that flows to the next one, so the whole pipeline is a single object that can be invoked,
# streamed or traced. Inputs and outputs are the same as run_hybrid_chat_session(), so the app
# can call either.

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from hybrid_rag.chat_history import persist_chat_history
from hybrid_rag.hybrid import GLOBAL_SESSION_ID, global_hybrid_col
from hybrid_rag.lc.components import (ANSWER_PROMPT, format_context, get_llm, history_messages,
                                      hybrid_search)

logger = logging.getLogger("rag.lc.hybrid")


def build_chain():
    """The Hybrid RAG chain. Input: {"question", "top_k", "history_limit", "selected_files"}.
    Output: the same dict plus "docs" (retrieved chunks), "context", "history" and "answer"."""
    return (
        # 1. retrieve the best chunks for the question (hybrid search, see components.py)
        RunnablePassthrough.assign(
            docs=lambda x: hybrid_search(x["question"], x["top_k"], x["selected_files"]))
        # 2. turn them into the prompt's context, and load the earlier turns
        | RunnablePassthrough.assign(
            context=lambda x: format_context(x["docs"])[0],
            history=lambda x: history_messages(x["history_limit"]))
        # 3. prompt -> chat model -> plain text answer
        | RunnablePassthrough.assign(answer=ANSWER_PROMPT | get_llm() | StrOutputParser())
    )


_CHAIN = None


def run_hybrid_lc(user_query: str, top_k: int = 6, history_limit: int = 5, selected_files: list = None):
    """Answer one question with Hybrid RAG (LangChain). Returns (answer, labelled chunks), like
    run_hybrid_chat_session() in hybrid.py."""
    global _CHAIN
    logger.info("=== Starting Hybrid RAG (LangChain) ===")
    logger.info(f"User Query: '{user_query}'")

    if global_hybrid_col.count() == 0:
        logger.error("Global Hybrid DB is empty. Please upload and index documents first.")
        return "The database is empty. Please upload and index some PDFs first.", []

    if _CHAIN is None:
        _CHAIN = build_chain()
    result = _CHAIN.invoke({"question": user_query, "top_k": top_k,
                            "history_limit": history_limit, "selected_files": selected_files})

    full_context_text, formatted_contexts = format_context(result["docs"])
    logger.info(f"Retrieved {len(formatted_contexts)} chunks; answer generated.")

    # 4. save the turn, like hybrid.py, so follow-up questions work in both versions
    persist_chat_history(GLOBAL_SESSION_ID, user_query, result["answer"], full_context_text)
    return result["answer"], formatted_contexts
