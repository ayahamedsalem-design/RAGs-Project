"""2-Round Fused Hybrid Corrective RAG (CRAG) built with LangChain LCEL.

LangChain counterpart of ``hybrid_rag/hybrid_corrective.py`` sharing the same ChromaDB
collection, OpenRouter models, cost tracking, and session-scoped chat history:
1. Retrieval (per sub-question, parallelized via ``.batch()``): dense vector search + BM25 fused
   via RRF and filtered by cosine similarity.
2. Round 1 Batch Grading: deterministic bibliography pre-filter + single LCEL JSON grading call.
3. Round 2 Corrective Retrieval: runs only when Round 1 is insufficient, using grader-suggested
   queries and re-grading the merged pool.
4. Answer Generation: LCEL ``RunnableBranch`` short-circuits with ``NOT_COVERED`` (0 LLM cost)
   if no verified chunks survive; otherwise generates a cited answer with session history.
5. Persistence: saves the turn to ``chat_history.json`` under the active ``session_id``.
"""

import logging

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough

from hybrid_rag.chat_history import persist_chat_history
from hybrid_rag.file_processing import SYS_PROMPT, decompose_multi_questions, format_location
from hybrid_rag.hybrid import GLOBAL_SESSION_ID, global_hybrid_col
from hybrid_rag.hybrid_corrective import (
    EXTRA_PER_QUERY,
    GRADER_PROMPT,
    GRADER_TEMPERATURE,
    INITIAL_CANDIDATES,
    MAX_CONTEXT,
    MAX_GRADED,
    MIN_SIMILARITY,
    NOT_COVERED,
    REFERENCE_THRESHOLD,
    Log,
    _cosine_similarity,
    _parse_grade,
    citation_density,
)
from hybrid_rag.lc.components import (
    bm25_retriever,
    expand_with_neighbors,
    format_context,
    get_cached_id_maps,
    get_embeddings,
    get_llm,
    get_vector_store,
    history_messages,
    is_rerank_enabled,
    reciprocal_rank_fusion,
    rerank_documents,
    scope,
)
from hybrid_rag.logging_setup import setup_logging
from hybrid_rag.tracing import build_trace_config, traceable

setup_logging()
logger = logging.getLogger("rag.lc.hybrid_corrective")

MAX_PARALLEL_SUBQUERIES = 4

BATCH_GRADER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=GRADER_PROMPT),
    ("human", "QUESTION: {question}\n\nCANDIDATES:\n\n{listing}"),
])


# ---------- 1. Hybrid RRF Retrieval (LangChain Chroma + BM25ChunkRetriever) ----------
def search_hybrid_rrf_lc(
    qvec: list[float],
    query_text: str,
    k: int,
    via_label: str,
    selected_files: list = None,
) -> list[Document]:
    """Dense (Chroma.similarity_search_by_vector) + Sparse (BM25ChunkRetriever) fused via RRF,
    returning LangChain Documents enriched with chunk_id, cosine sim, and via metadata."""
    docs, where = scope(selected_files)
    if not docs:
        return []

    effective_k = min(k, len(docs))
    # Apply FlashRank reranking on Round-1 retrieval (skip redundant ONNX pass on small Round-2 fetches)
    use_rr = is_rerank_enabled() and via_label == "question"
    pool_k = min(max(effective_k, 15), len(docs)) if use_rr else effective_k
    bm25 = bm25_retriever(docs, where, pool_k)
    store = get_vector_store()
    doc_to_id, id_to_emb = get_cached_id_maps()

    try:
        dense = store.similarity_search_by_vector(qvec, k=pool_k, filter=where)
    except Exception as e:
        logger.error(f"[{via_label}] Dense search failed for '{query_text}': {e}", exc_info=True)
        dense = []

    sparse = bm25.invoke(query_text)
    fused = reciprocal_rank_fusion([dense, sparse])[:pool_k]
    # Rerank Round-1 candidates with saturation-aware ONNX FlashRank before passing to the batch grader
    if use_rr:
        fused = rerank_documents(query_text, fused, top_n=effective_k)

    enriched = []
    for idx, d in enumerate(fused):
        cid = d.metadata.get("chunk_id") or d.id or doc_to_id.get(d.page_content, f"unknown_{idx}")
        c_emb = id_to_emb.get(cid)
        sim = _cosine_similarity(qvec, c_emb) if c_emb is not None else MIN_SIMILARITY
        enriched.append(
            Document(
                id=cid,
                page_content=d.page_content,
                metadata={
                    **d.metadata,
                    "chunk_id": cid,
                    "sim": sim,
                    "via": via_label,
                },
            )
        )
    return enriched


# ---------- 2. LCEL Batch Grader Chain ----------
_GRADER_CHAIN = None


def grader_chain():
    """LCEL chain: BATCH_GRADER_PROMPT | ChatOpenAI(temp=0, reasoning off) | StrOutputParser."""
    global _GRADER_CHAIN
    if _GRADER_CHAIN is None:
        _GRADER_CHAIN = (
            BATCH_GRADER_PROMPT
            | get_llm(temperature=GRADER_TEMPERATURE, disable_reasoning=True)
            | StrOutputParser()
        )
    return _GRADER_CHAIN


def _is_reference_doc(doc: Document) -> bool:
    """Only apply bibliography/reference-list filtering to unstructured prose chunks,
    never to structured tabular/JSON chunks."""
    if doc.metadata.get("data_kind") == "structured":
        return False
    return citation_density(doc.page_content) >= REFERENCE_THRESHOLD


@traceable(name="grade_documents_lc", run_type="chain")
def grade_documents_lc(question: str, cands: list[Document]) -> dict:
    """Filters reference lists deterministically (unstructured only), then grades remaining Documents in one LCEL call."""
    refs = [i for i, d in enumerate(cands, 1) if _is_reference_doc(d)]
    others = [(i, d) for i, d in enumerate(cands, 1) if i not in refs]

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
        f"[Candidate {k}] (file: {d.metadata.get('source', 'Unknown')}, {format_location(d.metadata)})\n"
        f"{d.page_content}"
        for k, (_, d) in enumerate(others, 1)
    )

    try:
        reply = grader_chain().invoke({"question": question, "listing": listing})
    except Exception as e:
        logger.warning(f"LCEL batch grading call failed ({e}); retrying without disable_reasoning...")
        fallback_chain = BATCH_GRADER_PROMPT | get_llm(temperature=GRADER_TEMPERATURE) | StrOutputParser()
        reply = fallback_chain.invoke({"question": question, "listing": listing})

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


# ---------- 3. 2-Round Corrective Retrieval per Sub-Question ----------
@traceable(name="corrective_retrieve_sub_query", run_type="retriever")
def corrective_retrieve_sub_query(payload: dict) -> dict:
    """Runs the 2-round CRAG loop for a single sub-question and returns verified Documents + missing note."""
    sub_q = payload["sub_query"]
    selected_files = payload["selected_files"]
    log = Log()

    embedder = get_embeddings()
    qvec = embedder.embed_query(sub_q)
    log.qvec = qvec

    # Round 1: Initial wide Hybrid RRF retrieval (k=15)
    cands = [
        d
        for d in search_hybrid_rrf_lc(
            qvec, sub_q, k=INITIAL_CANDIDATES, via_label="question", selected_files=selected_files
        )
        if d.metadata["sim"] >= MIN_SIMILARITY
    ]
    log.append(f"[LC CRAG] Retrieved {len(cands)} candidates (sim >= {MIN_SIMILARITY}) for '{sub_q}'")
    if not cands:
        return {"docs": [], "status": "NOT FOUND", "missing": ""}

    # Round 1: Batch grade
    g = grade_documents_lc(sub_q, cands)
    relevant = [cands[i - 1] for i in g["relevant"]]
    log.append(f"[LC CRAG] Round 1: {len(relevant)}/{len(cands)} relevant. Sufficient: {g['sufficient']}")
    if g["sufficient"] and relevant:
        return {"docs": relevant, "status": "CORRECT", "missing": g["missing"]}

    # Round 2: Corrective retrieval with grader-suggested queries
    queries = g["queries"] or [sub_q]
    log.append("[LC CRAG] New search queries: " + "; ".join(f'"{q}"' for q in queries))

    _, id_to_emb = get_cached_id_maps()
    seen = {d.metadata["chunk_id"] for d in cands}
    new_cands = []

    if queries:
        qvecs = embedder.embed_documents(queries)
        for qi, (q_text, q_emb) in enumerate(zip(queries, qvecs), 1):
            for d in search_hybrid_rrf_lc(
                q_emb,
                q_text,
                k=EXTRA_PER_QUERY,
                via_label=f"new query {qi}",
                selected_files=selected_files,
            ):
                cid = d.metadata["chunk_id"]
                if cid not in seen and not _is_reference_doc(d):
                    seen.add(cid)
                    c_emb = id_to_emb.get(cid)
                    if c_emb is not None:
                        d.metadata["sim"] = _cosine_similarity(qvec, c_emb)
                    if d.metadata["sim"] >= MIN_SIMILARITY:
                        new_cands.append(d)

    if not new_cands:
        return {
            "docs": relevant,
            "status": "PARTIAL" if relevant else "NOT FOUND",
            "missing": g["missing"],
        }

    # Round 2: Final grading on combined pool
    pool = (relevant + new_cands)[:MAX_GRADED]
    g2 = grade_documents_lc(sub_q, pool)
    relevant2 = [pool[i - 1] for i in g2["relevant"]]
    log.append(f"[LC CRAG] Round 2: {len(relevant2)}/{len(pool)} relevant.")
    return {
        "docs": relevant2,
        "status": "CORRECTED" if g2["sufficient"] else "PARTIAL",
        "missing": g2["missing"],
    }


def retrieve_and_merge_lc(state: dict) -> dict:
    """Decomposes the question, runs 2-round CRAG across sub-questions in parallel (.batch()),
    merges unique verified Documents, sorts by cosine sim, and truncates to MAX_CONTEXT."""
    sub_queries = decompose_multi_questions(state["question"])
    logger.info(f"[LC CRAG] Decomposed query into {len(sub_queries)} sub-question(s).")
    payloads = [{"sub_query": sq, "selected_files": state["selected_files"]} for sq in sub_queries]

    if len(payloads) == 1:
        sub_results = [corrective_retrieve_sub_query(payloads[0])]
    else:
        sub_results = RunnableLambda(corrective_retrieve_sub_query).batch(
            payloads, config={"max_concurrency": MAX_PARALLEL_SUBQUERIES}
        )

    merged_by_id = {}
    missing_parts = []
    for idx, res in enumerate(sub_results, 1):
        logger.info(
            f"[LC CRAG SQ{idx}] Completed with status={res['status']} ({len(res['docs'])} relevant chunk(s))."
        )
        if res["missing"]:
            missing_parts.append(res["missing"])
        for d in res["docs"]:
            cid = d.metadata["chunk_id"]
            if cid not in merged_by_id or d.metadata["sim"] > merged_by_id[cid].metadata["sim"]:
                merged_by_id[cid] = d

    final_docs = sorted(merged_by_id.values(), key=lambda d: -d.metadata["sim"])[:MAX_CONTEXT]
    # Expand small verified chunk sets with immediate adjacent chunks (+1/-1) from the same PDF
    final_docs = expand_with_neighbors(
        final_docs, max_total=MAX_CONTEXT, max_neighbors=2, is_noise_fn=_is_reference_doc
    )
    combined_missing = "; ".join(dict.fromkeys(missing_parts))
    logger.info(f"[LC CRAG] Compiled {len(final_docs)} unique verified chunk(s) for generation.")
    return {"docs": final_docs, "missing": combined_missing}


# ---------- 4. Answer Generation & Full LCEL Chain ----------
def _build_answer_messages(state: dict) -> list:
    """Builds [SystemMessage(+missing note), *history, HumanMessage(context+question)]."""
    sys_inst = SYS_PROMPT
    if state.get("missing"):
        sys_inst += (
            f"\n\nNOTE: Likely missing aspects: {state['missing']}. "
            "Answer what the sources support and say clearly which parts the documents do not cover."
        )
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=sys_inst),
        MessagesPlaceholder("history"),
        ("human", "[Context from Global Database]:\n{context}\n\n[User Question(s)]:\n{question}"),
    ])
    return prompt.format_messages(
        history=state["history"],
        context=state["context"],
        question=state["question"],
    )


def build_chain():
    """
    Full LCEL chain for 2-Round Fused Hybrid Corrective RAG.
    Input:  {"question", "history_limit", "selected_files", "session_id"}
    Output: same dict plus "docs", "missing", "context", "history", and "answer".
    """
    answer_branch = RunnableBranch(
        # Short-circuit with 0 Answer LLM cost when no verified chunks survive grading
        (lambda x: not x["docs"], RunnableLambda(lambda _: NOT_COVERED)),
        RunnableLambda(_build_answer_messages) | get_llm(temperature=0.2) | StrOutputParser(),
    )

    return (
        # 1. 2-round hybrid retrieval + batch grading + corrective query rewriting
        RunnablePassthrough.assign(_crag=RunnableLambda(retrieve_and_merge_lc))
        | RunnablePassthrough.assign(
            docs=lambda x: x["_crag"]["docs"],
            missing=lambda x: x["_crag"]["missing"],
        )
        # 2. Format verified context + load rolling session chat history
        | RunnablePassthrough.assign(
            context=lambda x: format_context(x["docs"])[0] if x["docs"] else "",
            history=lambda x: history_messages(
                x["history_limit"], x.get("session_id", GLOBAL_SESSION_ID)
            ),
        )
        # 3. Short-circuit if empty, else invoke ChatOpenAI via LCEL
        | RunnablePassthrough.assign(answer=answer_branch)
    )


_CHAIN = None


def run_hybrid_corrective_lc(
    user_query: str,
    history_limit: int = 6,
    selected_files: list = None,
    session_id: str = GLOBAL_SESSION_ID,
):
    """
    Answer one question with 2-Round Fused Hybrid Corrective RAG (LangChain LCEL).
    Returns (answer, verified_labelled_contexts) — drop-in compatible with
    run_hybrid_corrective_rag() in hybrid_rag/hybrid_corrective.py.
    """
    global _CHAIN
    logger.info("=== Starting Hybrid Corrective RAG (LangChain LCEL) ===")
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
                "history_limit": history_limit,
                "selected_files": selected_files,
                "session_id": session_id,
            },
            config=build_trace_config(
                run_name="HybridCorrectiveLC",
                tags=["hybrid_corrective_lc", "crag", "lcel", "bilingual"],
                metadata={
                    "pipeline": "hybrid_corrective_lc",
                    "session_id": session_id,
                    "selected_files": selected_files or ["ALL"],
                },
            ),
        )
    except Exception as exc:
        logger.critical(f"LangChain Hybrid CRAG execution failed: {exc}", exc_info=True)
        raise

    full_context_text, verified_contexts = format_context(result["docs"])
    if not verified_contexts:
        logger.warning("[LC CRAG] No verified chunks survived; returned NOT_COVERED.")
    else:
        logger.info(
            f"Hybrid Corrective RAG (LangChain) completed with {len(verified_contexts)} verified chunk(s)."
        )

    # Persist under session_id so multi-chat sessions and history summarization stay in sync
    persist_chat_history(session_id, user_query, result["answer"], full_context_text)
    return result["answer"], verified_contexts


if __name__ == "__main__":
    sample_query = (
        "1. What are the general steps in cultured meat production?\n"
        "2. Why is Mycoplasma contamination difficult to detect visually?"
    )
    answer, docs = run_hybrid_corrective_lc(sample_query)
    print("\n" + "=" * 60)
    print("FINAL MODEL ANSWER (LangChain LCEL):")
    print("=" * 60)
    print(answer)
