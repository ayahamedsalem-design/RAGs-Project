# Corrective Hybrid RAG built with LangChain: the LangChain version of
# hybrid_rag/corrective_hybrid.py.
#
# Same steps, data, prompts and settings as the original, so the two can be compared:
#   1. retrieval: for each sub-question, vector search + BM25 (top 10 each), fused with
#      Reciprocal Rank Fusion, best 4 kept per sub-question (merged, duplicates removed);
#      every chunk remembers which sub-questions found it
#   2. grading: an LLM grader answers "yes"/"no" per chunk (same prompt, temperature 0, at most
#      5 tokens); a chunk is kept if it is relevant to at least one of its sub-questions
#      (checked one sub-question at a time, stopping at the first "yes"); if a grading call
#      fails, the chunk is kept
#   3. answer: system prompt + last 5 turns + the kept chunks + question, temperature 0.2;
#      if no chunk is kept, the context says so
#   4. the turn is saved to the shared chat history
#
# One difference, on purpose: the chunks are graded in parallel (LangChain's .batch()), not one
# after another. The requests and results are the same; only the waiting time changes.

import logging

from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from hybrid_rag.chat_history import persist_chat_history
from hybrid_rag.hybrid import GLOBAL_SESSION_ID, global_hybrid_col
from hybrid_rag.lc.components import (PER_BRANCH, bm25_retriever, format_context, get_llm,
                                      get_vector_store, history_messages, reciprocal_rank_fusion,
                                      scope)
from hybrid_rag.pdf_processing import SYS_PROMPT, decompose_multi_questions

logger = logging.getLogger("rag.lc.corrective_hybrid")

MAX_PARALLEL_GRADES = 8          # grading requests sent at the same time
NO_CONTEXT = "The context does not contain any relevant information for the question."

# The grader prompt of corrective_hybrid.py, word for word
GRADE_PROMPT = ChatPromptTemplate.from_messages([("human", """You are an objective document evaluator.
Assess whether the provided context snippet contains information relevant to answering the question.

Question:
{query}

Context Snippet:
{doc_content}

Respond strictly with 'yes' if relevant, or 'no' if irrelevant. Do not provide explanations.""")])

# The answer prompt of corrective_hybrid.py (as in hybrid.py, but without the final new line)
CORRECTIVE_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYS_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "[Context from Global Database]:\n{context}\n\n[User Question(s)]:\n{question}"),
])


# ---------- 1. Retrieval: top chunks per sub-question ----------
def candidates(question, top_k_per_query=4, selected_files=None):
    """[(chunk Document, [sub-questions that found it]), ...] in the order they were found."""
    docs, where = scope(selected_files)
    if not docs:
        return []
    n = min(PER_BRANCH, len(docs))
    bm25 = bm25_retriever(docs, where, n)
    store = get_vector_store()
    found, order = {}, []
    for sub_q in decompose_multi_questions(question):
        try:
            dense = store.similarity_search(sub_q, k=n, filter=where)
        except Exception as e:                       # like the original: keep the BM25 results
            logger.error(f"Dense search failed for sub-query '{sub_q}': {e}", exc_info=True)
            dense = []
        for doc in reciprocal_rank_fusion([dense, bm25.invoke(sub_q)])[:top_k_per_query]:
            if doc.page_content not in found:
                found[doc.page_content] = (doc, [])
                order.append(doc.page_content)
            found[doc.page_content][1].append(sub_q)
    logger.info(f"Retrieved {len(order)} unique candidate chunks across all sub-queries.")
    return [found[text] for text in order]


# ---------- 2. Grading ----------
_GRADER = None


def grader():
    """GRADE_PROMPT -> CHAT_MODEL (temperature 0, max 5 tokens) -> True if the reply says yes."""
    global _GRADER
    if _GRADER is None:
        _GRADER = (GRADE_PROMPT | get_llm(temperature=0.0, max_tokens=5) | StrOutputParser()
                   | RunnableLambda(read_grade))
    return _GRADER


def read_grade(reply):
    """True for a "yes" reply. An empty reply counts as a failed grading (an error), exactly as in
    corrective_hybrid.py: DeepSeek sometimes spends all 5 allowed tokens on hidden reasoning and
    returns no text; the original then fails on None.strip() and keeps the chunk, so this does too."""
    if not reply.strip():
        raise ValueError("empty grader reply (the 5-token limit was used up by reasoning)")
    return "yes" in reply.strip().lower()


def is_relevant(candidate):
    """Keep a chunk if it is relevant to at least one of the sub-questions that found it,
    asking one sub-question at a time and stopping at the first yes (as the original does)."""
    doc, sub_questions = candidate
    for q in sub_questions:
        try:
            if grader().invoke({"query": q, "doc_content": doc.page_content}):
                return True
        except Exception as e:
            logger.warning(f"Grading failed ({e}), accepting document as fallback.")
            return True
    return False


def grade_all(cands):
    """The kept chunks, in their original order. The chunks are graded in parallel."""
    if not cands:
        return []
    keep = RunnableLambda(is_relevant).batch(cands, config={"max_concurrency": MAX_PARALLEL_GRADES})
    kept = [doc for (doc, _), ok in zip(cands, keep) if ok]
    logger.info(f"CRAG Verification: {len(kept)}/{len(cands)} chunks passed grading.")
    return kept


# ---------- The chain ----------
def build_chain():
    """Input: {"question", "top_k_per_query", "history_limit", "selected_files"}. Output: the
    same dict plus "candidates", "docs" (kept chunks), "context", "history" and "answer"."""
    return (
        # 1. retrieve candidate chunks per sub-question
        RunnablePassthrough.assign(
            candidates=lambda x: candidates(x["question"], x["top_k_per_query"], x["selected_files"]))
        # 2. grade them (in parallel) and keep the relevant ones
        | RunnablePassthrough.assign(docs=lambda x: grade_all(x["candidates"]))
        # 3. context from the kept chunks (or the "nothing relevant" note) + earlier turns
        | RunnablePassthrough.assign(
            context=lambda x: format_context(x["docs"])[0] if x["docs"] else NO_CONTEXT,
            history=lambda x: history_messages(x["history_limit"]))
        # 4. prompt -> chat model -> plain text answer
        | RunnablePassthrough.assign(answer=CORRECTIVE_PROMPT | get_llm() | StrOutputParser())
    )


_CHAIN = None


def run_corrective_hybrid_lc(user_query: str, top_k_per_query: int = 4, history_limit: int = 5,
                             selected_files: list = None):
    """Answer one question with Corrective Hybrid RAG (LangChain). Returns (answer, kept labelled
    chunks), like run_corrective_hybrid_rag() in corrective_hybrid.py."""
    global _CHAIN
    logger.info("=== Starting Corrective Hybrid RAG (LangChain) ===")
    logger.info(f"User Query: '{user_query}'")

    if global_hybrid_col.count() == 0:
        logger.error("Global Hybrid DB is empty. Please upload and index documents first.")
        return "The database is empty. Please upload and index some PDFs first.", []

    if _CHAIN is None:
        _CHAIN = build_chain()
    result = _CHAIN.invoke({"question": user_query, "top_k_per_query": top_k_per_query,
                            "history_limit": history_limit, "selected_files": selected_files})

    _, verified_contexts = format_context(result["docs"])
    # save the turn, like the original, so follow-up questions work in both versions
    persist_chat_history(GLOBAL_SESSION_ID, user_query, result["answer"], result["context"])
    return result["answer"], verified_contexts
