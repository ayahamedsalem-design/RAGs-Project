"""Self-Corrective 2-Round Hybrid CRAG implemented as an explicit LangGraph ``StateGraph``.

Graph topology:
    START -> decompose -> retrieve_and_grade
        ├─ (0 verified docs) -> short_circuit (0 Answer LLM cost) -> END
        └─ (>=1 verified docs) -> generate -> verify_answer
                ├─ (grounded & cited, or max retries reached) -> END
                └─ (hallucination / missing citation detected) -> generate (self-correct once)

Reuses the shared ChromaDB client, Arabic-aware BM25 cache, RRF fusion, batch grader,
cost-tracking httpx hook, session-scoped chat history, and LangSmith tracing.
"""

import logging
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

from hybrid_rag.chat_history import persist_chat_history
from hybrid_rag.file_processing import SYS_PROMPT, decompose_multi_questions
from hybrid_rag.hybrid import GLOBAL_SESSION_ID, global_hybrid_col
from hybrid_rag.hybrid_corrective import (
    EXTRA_PER_QUERY,
    GRADER_TEMPERATURE,
    INITIAL_CANDIDATES,
    MAX_CONTEXT,
    MAX_GRADED,
    MIN_SIMILARITY,
    NOT_COVERED,
    _cosine_similarity,
    _read_json,
)
from hybrid_rag.lc.components import (
    expand_with_neighbors,
    format_context,
    get_cached_id_maps,
    get_embeddings,
    get_llm,
    history_messages,
)
from hybrid_rag.lc.hybrid_corrective_lc import (
    MAX_PARALLEL_SUBQUERIES,
    _is_reference_doc,
    grade_documents_lc,
    search_hybrid_rrf_lc,
)
from hybrid_rag.logging_setup import setup_logging
from hybrid_rag.tracing import build_trace_config, traceable

setup_logging()
logger = logging.getLogger("rag.lg.crag")

# Maximum number of self-correction regeneration loops allowed if the Verifier flags a draft
MAX_VERIFY_RETRIES = 1

# System prompt for Node 4 (verify_answer): audits grounding + citations in one fast JSON call
VERIFIER_PROMPT = """You are a strict RAG answer auditor.
Given a QUESTION, the retrieved CONTEXT excerpts, and a DRAFT ANSWER:
1. Check whether every factual claim in the draft answer is grounded in the CONTEXT (or explicitly states what the documents do not cover).
2. Check whether the draft answer includes source/location citations (e.g. file name and page/slide/sheet/section) when stating facts from the CONTEXT. Note: if the draft answer states that the information is not found in the database, citations are not required (set "has_citations": true).

Reply with JSON only, no other text:
{"grounded": true, "has_citations": true, "issue": ""}
If either check fails, set the corresponding boolean to false and write a brief, actionable one-sentence description in "issue"."""

# Prompt template combining the verifier rubric with the question, retrieved context, and draft
VERIFIER_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessage(content=VERIFIER_PROMPT),
    (
        "human",
        "QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nDRAFT ANSWER:\n{answer}",
    ),
])


# ==========================================
# 1. Explicit Typed Graph State
# ==========================================

class CRAGState(TypedDict):
    """State passed and updated across all LangGraph nodes in the CRAG workflow."""
    # --- Inputs supplied by the caller ---
    question: str                  # Original user prompt
    selected_files: list[str] | None  # Sidebar file filter (None = search all indexed files)
    session_id: str                # Active chat session ID for history lookup
    history_limit: int             # Number of prior chat turns to include (0 for stateless calls)
    # --- Intermediate retrieval & grading outputs ---
    sub_queries: list[str]         # Decomposed self-contained sub-questions (from `decompose`)
    docs: list[Document]           # Final verified + neighbor-expanded chunks (capped at MAX_CONTEXT)
    missing: str                   # Combined summary of aspects not covered by the corpus
    statuses: list[str]            # Per-sub-query status: CORRECT / CORRECTED / PARTIAL / NOT FOUND
    round2_triggered: bool         # True if any sub-query needed Round-2 query rewriting
    context: str                   # Formatted "[Source: ... | Page ...]" string passed to LLM
    history: list                  # Loaded LangChain HumanMessage/AIMessage pairs
    # --- Generation & verification outputs ---
    answer: str                    # Latest draft or final answer text
    verified: bool                 # True once `verify_answer` passes (or on short-circuit)
    verification_note: str         # Actionable critique injected into `generate` on retry
    retries: int                   # Number of verifier-triggered regeneration retries used so far


# ==========================================
# 2. Sub-Query 2-Round CRAG Helper
# ==========================================

@traceable(name="crag_graph_sub_query", run_type="retriever")
def _retrieve_sub_query(payload: dict) -> dict:
    """Runs Round-1 Hybrid RRF + Batch Grading + conditional Round-2 rewrite for one sub-query."""
    sub_q = payload["sub_query"]
    selected_files = payload["selected_files"]

    # Embed the sub-query once; this vector is reused for Round-1 dense search and Round-2 cosine re-scoring
    embedder = get_embeddings()
    qvec = embedder.embed_query(sub_q)

    # Round 1: wide Hybrid RRF + FlashRank retrieval (k=15) filtered by cosine similarity floor
    cands = [
        d
        for d in search_hybrid_rrf_lc(
            qvec, sub_q, k=INITIAL_CANDIDATES, via_label="question", selected_files=selected_files
        )
        if d.metadata["sim"] >= MIN_SIMILARITY
    ]
    if not cands:
        return {"docs": [], "status": "NOT FOUND", "missing": "", "round2": False}

    # Round 1: deterministic reference-list filter + single-call batch LLM relevance grader
    g = grade_documents_lc(sub_q, cands)
    relevant = [cands[i - 1] for i in g["relevant"]]
    # Fast exit: if Round-1 evidence already covers the sub-question, skip Round 2 completely
    if g["sufficient"] and relevant:
        return {"docs": relevant, "status": "CORRECT", "missing": g["missing"], "round2": False}

    # Round 2: corrective retrieval using grader-suggested rewritten queries (up to 3)
    queries = g["queries"] or [sub_q]
    _, id_to_emb = get_cached_id_maps()
    seen = {d.metadata["chunk_id"] for d in cands}
    new_cands = []

    if queries:
        # Batch-embed all rewritten queries in a single API call to minimize latency
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
                # Skip duplicates already seen in Round 1 and skip bibliography/reference noise
                if cid not in seen and not _is_reference_doc(d):
                    seen.add(cid)
                    # Re-score candidate similarity against the ORIGINAL sub-query vector for fair ranking
                    c_emb = id_to_emb.get(cid)
                    if c_emb is not None:
                        d.metadata["sim"] = _cosine_similarity(qvec, c_emb)
                    if d.metadata["sim"] >= MIN_SIMILARITY:
                        new_cands.append(d)

    # If Round 2 surfaced no new viable candidates, return whatever Round-1 relevant docs we had
    if not new_cands:
        return {
            "docs": relevant,
            "status": "PARTIAL" if relevant else "NOT FOUND",
            "missing": g["missing"],
            "round2": True,
        }

    # Round 2: final batch grading over the combined Round-1 relevant + Round-2 new pool (capped at MAX_GRADED)
    pool = (relevant + new_cands)[:MAX_GRADED]
    g2 = grade_documents_lc(sub_q, pool)
    relevant2 = [pool[i - 1] for i in g2["relevant"]]
    return {
        "docs": relevant2,
        "status": "CORRECTED" if g2["sufficient"] else "PARTIAL",
        "missing": g2["missing"],
        "round2": True,
    }


# ==========================================
# 3. LangGraph Nodes & Conditional Routers
# ==========================================

def decompose_node(state: CRAGState) -> dict:
    """Node 1: splits compound user questions into focused sub-questions."""
    sub_queries = decompose_multi_questions(state["question"])
    logger.info(f"[LangGraph CRAG] Decomposed query into {len(sub_queries)} sub-question(s).")
    return {"sub_queries": sub_queries}


def retrieve_and_grade_node(state: CRAGState) -> dict:
    """Node 2: executes 2-round CRAG retrieval across sub-questions (parallelized via .batch())."""
    payloads = [
        {"sub_query": sq, "selected_files": state["selected_files"]}
        for sq in state["sub_queries"]
    ]

    # Avoid thread-pool overhead for single-part questions; parallelize multi-part queries via .batch()
    if len(payloads) == 1:
        sub_results = [_retrieve_sub_query(payloads[0])]
    else:
        sub_results = RunnableLambda(_retrieve_sub_query).batch(
            payloads, config={"max_concurrency": MAX_PARALLEL_SUBQUERIES}
        )

    # Deduplicate verified chunks across all sub-queries, keeping each chunk's highest similarity score
    merged_by_id: dict[str, Document] = {}
    missing_parts: list[str] = []
    statuses: list[str] = []
    round2_any = False

    for idx, res in enumerate(sub_results, 1):
        statuses.append(res["status"])
        round2_any = round2_any or bool(res["round2"])
        logger.info(
            f"[LangGraph CRAG SQ{idx}] status={res['status']} "
            f"(relevant={len(res['docs'])}, round2={res['round2']})"
        )
        if res["missing"]:
            missing_parts.append(res["missing"])
        for d in res["docs"]:
            cid = d.metadata["chunk_id"]
            if cid not in merged_by_id or d.metadata["sim"] > merged_by_id[cid].metadata["sim"]:
                merged_by_id[cid] = d

    # Keep the top MAX_CONTEXT verified chunks ordered by descending cosine similarity
    final_docs = sorted(merged_by_id.values(), key=lambda d: -d.metadata["sim"])[:MAX_CONTEXT]
    # Expand small verified chunk sets with immediate adjacent chunks (+1/-1) from the same PDF
    final_docs = expand_with_neighbors(
        final_docs, max_total=MAX_CONTEXT, max_neighbors=2, is_noise_fn=_is_reference_doc
    )
    # Deduplicate any overlapping missing-aspect notes while preserving order
    combined_missing = "; ".join(dict.fromkeys(missing_parts))
    context_text = format_context(final_docs)[0] if final_docs else ""
    hist = history_messages(state["history_limit"], state["session_id"])

    return {
        "docs": final_docs,
        "missing": combined_missing,
        "statuses": statuses,
        "round2_triggered": round2_any,
        "context": context_text,
        "history": hist,
    }


def _route_after_retrieval(state: CRAGState) -> str:
    """Conditional edge: short-circuit at 0 LLM cost if no verified chunks survived."""
    return "short_circuit" if not state["docs"] else "generate"


def short_circuit_node(state: CRAGState) -> dict:
    """Node 3a: immediate grounded refusal with zero Answer LLM cost."""
    logger.warning("[LangGraph CRAG] No verified chunks survived; short-circuiting with NOT_COVERED.")
    # Mark verified=True because NOT_COVERED is an intentional grounded refusal (skips Verifier too)
    return {
        "answer": NOT_COVERED,
        "verified": True,
        "verification_note": "short_circuit_no_relevant_docs",
    }


def generate_node(state: CRAGState) -> dict:
    """Node 3b: generates (or self-corrects) a cited answer from verified context & session history."""
    sys_inst = SYS_PROMPT
    # If grading flagged uncovered sub-parts, instruct the LLM to disclose them transparently
    if state.get("missing"):
        sys_inst += (
            f"\n\nNOTE: Likely missing aspects: {state['missing']}. "
            "Answer what the sources support and say clearly which parts the documents do not cover."
        )
    # On self-correction retries (retries > 0), inject the Verifier's specific critique into the prompt
    if state.get("verification_note") and state.get("retries", 0) > 0:
        sys_inst += (
            f"\n\nQUALITY CORRECTION REQUIRED: A prior draft failed grounding/citation verification "
            f"with the following issue: {state['verification_note']}. "
            "Fix this strictly using ONLY the provided [Context from Global Database] and cite "
            "the exact Source File and Location for every factual claim."
        )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=sys_inst),
        MessagesPlaceholder("history"),
        ("human", "[Context from Global Database]:\n{context}\n\n[User Question(s)]:\n{question}"),
    ])
    chain = prompt | get_llm(temperature=0.2) | StrOutputParser()
    answer = chain.invoke({
        "history": state["history"],
        "context": state["context"],
        "question": state["question"],
    })
    return {"answer": answer}


def verify_answer_node(state: CRAGState) -> dict:
    """Node 4: lightweight Hallucination & Citation Grader auditing the draft answer."""
    # Use temperature=0 with reasoning disabled for fast, deterministic JSON verification
    verifier = (
        VERIFIER_TEMPLATE
        | get_llm(temperature=GRADER_TEMPERATURE, disable_reasoning=True)
        | StrOutputParser()
    )
    try:
        reply = verifier.invoke({
            "question": state["question"],
            "context": state["context"],
            "answer": state["answer"],
        })
    except Exception as exc:
        # Fail-open on transient API errors so a verifier outage never blocks delivering the draft
        logger.warning(f"[LangGraph CRAG] Verifier call failed ({exc}); accepting draft answer.")
        return {"verified": True, "verification_note": ""}

    parsed = _read_json(reply)
    if not isinstance(parsed, dict):
        # Accept draft if the verifier returned unparseable JSON
        return {"verified": True, "verification_note": ""}

    grounded = bool(parsed.get("grounded", True))
    has_citations = bool(parsed.get("has_citations", True))
    issue = str(parsed.get("issue") or "").strip()
    is_ok = grounded and has_citations

    if not is_ok:
        # Increment retry counter and store the critique so `generate_node` can fix it on the next pass
        logger.warning(
            f"[LangGraph CRAG] Verifier flagged answer (grounded={grounded}, "
            f"has_citations={has_citations}): {issue}"
        )
        return {
            "verified": False,
            "verification_note": issue or "Ensure every claim is strictly grounded with source & page citations.",
            "retries": state.get("retries", 0) + 1,
        }

    logger.info("[LangGraph CRAG] Verifier passed: answer is grounded and properly cited.")
    return {"verified": True, "verification_note": ""}


def _route_after_verify(state: CRAGState) -> str:
    """Conditional edge: finish if verified or retry budget exhausted, else regenerate once."""
    if state.get("verified", True) or state.get("retries", 0) > MAX_VERIFY_RETRIES:
        return END
    return "generate"


# ==========================================
# 4. Graph Compilation & Public Runners
# ==========================================

# Module-level compiled graph singleton (built lazily on first query)
_CRAG_GRAPH = None


def build_crag_graph():
    """Compiles the Self-Corrective CRAG LangGraph StateGraph."""
    builder = StateGraph(CRAGState)

    # Register all 5 graph nodes
    builder.add_node("decompose", decompose_node)
    builder.add_node("retrieve_and_grade", retrieve_and_grade_node)
    builder.add_node("short_circuit", short_circuit_node)
    builder.add_node("generate", generate_node)
    builder.add_node("verify_answer", verify_answer_node)

    # Linear entry: START -> decompose -> retrieve_and_grade
    builder.add_edge(START, "decompose")
    builder.add_edge("decompose", "retrieve_and_grade")
    # Branch 1: zero verified chunks -> short_circuit (0 LLM cost); otherwise -> generate
    builder.add_conditional_edges(
        "retrieve_and_grade",
        _route_after_retrieval,
        {"short_circuit": "short_circuit", "generate": "generate"},
    )
    builder.add_edge("short_circuit", END)
    # Every generated draft goes through the Hallucination & Citation Verifier
    builder.add_edge("generate", "verify_answer")
    # Branch 2 (cyclic edge): loop back to `generate` once if flagged, else exit to END
    builder.add_conditional_edges(
        "verify_answer",
        _route_after_verify,
        {"generate": "generate", END: END},
    )
    return builder.compile()


def get_crag_graph():
    """Returns the lazily compiled CRAG StateGraph singleton."""
    global _CRAG_GRAPH
    if _CRAG_GRAPH is None:
        _CRAG_GRAPH = build_crag_graph()
    return _CRAG_GRAPH


def run_crag_graph_stateless(
    user_query: str,
    history_limit: int = 0,
    selected_files: list = None,
    session_id: str = GLOBAL_SESSION_ID,
) -> tuple[str, list[str], dict]:
    """Executes CRAG StateGraph without mutating chat_history.json (used by Agentic RAG tool)."""
    if global_hybrid_col.count() == 0:
        return "The database is empty. Please upload and index some PDFs first.", [], {}

    graph = get_crag_graph()
    initial_state: CRAGState = {
        "question": user_query,
        "selected_files": selected_files,
        "session_id": session_id,
        "history_limit": history_limit,
        "sub_queries": [],
        "docs": [],
        "missing": "",
        "statuses": [],
        "round2_triggered": False,
        "context": "",
        "history": [],
        "answer": "",
        "verified": False,
        "verification_note": "",
        "retries": 0,
    }
    result = graph.invoke(
        initial_state,
        config=build_trace_config(
            run_name="CRAGStateGraph",
            tags=["crag_graph", "langgraph", "self_corrective", "bilingual"],
            metadata={
                "pipeline": "crag_graph",
                "session_id": session_id,
                "selected_files": selected_files or ["ALL"],
            },
        ),
    )
    _, verified_contexts = format_context(result["docs"])
    meta = {
        "statuses": result["statuses"],
        "round2_triggered": result["round2_triggered"],
        "verified": result["verified"],
        "retries": result["retries"],
        "missing": result["missing"],
    }
    return result["answer"], verified_contexts, meta


def run_crag_graph(
    user_query: str,
    history_limit: int = 5,
    selected_files: list = None,
    session_id: str = GLOBAL_SESSION_ID,
):
    """Answers one question with Self-Corrective CRAG StateGraph (LangGraph).

    Returns ``(answer, verified_labelled_contexts)`` — drop-in compatible with ``App.py``.
    """
    logger.info("=== Starting CRAG StateGraph (LangGraph) ===")
    logger.info(f"Session: '{session_id}' | User Query: '{user_query}'")

    if global_hybrid_col.count() == 0:
        logger.error("Global Hybrid DB is empty. Please upload and index documents first.")
        return "The database is empty. Please upload and index some PDFs first.", []

    try:
        answer, verified_contexts, meta = run_crag_graph_stateless(
            user_query=user_query,
            history_limit=history_limit,
            selected_files=selected_files,
            session_id=session_id,
        )
    except Exception as exc:
        logger.critical(f"LangGraph CRAG execution failed: {exc}", exc_info=True)
        raise

    full_context_text = "\n\n---\n\n".join(verified_contexts)
    logger.info(
        f"[LangGraph CRAG] Completed: {len(verified_contexts)} chunk(s), "
        f"round2={meta.get('round2_triggered')}, verified={meta.get('verified')}, "
        f"retries={meta.get('retries')}."
    )
    persist_chat_history(session_id, user_query, answer, full_context_text)
    return answer, verified_contexts
