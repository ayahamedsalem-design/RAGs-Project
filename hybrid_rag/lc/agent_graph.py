"""Tool-Calling Agentic RAG built with LangGraph (``StateGraph`` + ``ToolNode``).

Equips a bilingual research agent with five tools layered directly over the project's
existing retrieval, CRAG StateGraph, structured-data, and history-summarization primitives:
  1. ``list_indexed_documents``   — inspect available files, types, and chunk counts.
  2. ``search_documents_fast``    — single-round Hybrid RRF (Dense + Arabic BM25).
  3. ``search_documents_deep``    — 2-Round Self-Corrective CRAG StateGraph (``crag_graph.py``).
  4. ``search_structured_data``   — targeted retrieval scoped to CSV / Excel / JSON chunks.
  5. ``summarize_conversation_history`` — summarize the active session's chat history on demand.

Graph topology:
    START -> agent
        ├─ (tool_calls & steps < MAX_AGENT_STEPS) -> tools (ToolNode) -> agent
        ├─ (tool_calls & steps >= MAX_AGENT_STEPS) -> finalize (no tools bound) -> END
        └─ (no tool_calls) -> END
"""

import logging
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from hybrid_rag.chat_history import load_file_history, persist_chat_history
from hybrid_rag.hybrid import GLOBAL_BM25_CACHE, GLOBAL_SESSION_ID, global_hybrid_col, refresh_global_bm25
from hybrid_rag.lc.components import format_context, get_llm, history_messages, hybrid_search
from hybrid_rag.lc.crag_graph import run_crag_graph_stateless
from hybrid_rag.logging_setup import setup_logging
from hybrid_rag.summarizer import summarize_cross_file_history
from hybrid_rag.tracing import build_trace_config

setup_logging()
logger = logging.getLogger("rag.lg.agent")

# Hard upper bound on agent reasoning turns before forcing a tool-free final synthesis
MAX_AGENT_STEPS = 4

# System prompt governing autonomous tool selection, bilingual fidelity, and strict grounding
AGENT_SYSTEM_PROMPT = """You are an intelligent bilingual (English & Arabic) Agentic RAG research assistant managing a global knowledge base of PDFs (including scanned Arabic OCR books), Word/PowerPoint files, text/Markdown documents, and structured datasets (CSV, Excel, JSON).

Tool selection policy:
1. NEVER invent document facts from general knowledge — ALWAYS retrieve from the database first.
2. Call `list_indexed_documents` if the user asks what files are available, or when you need exact filenames to target a search.
3. Call `search_documents_fast` for straightforward, single-topic factual lookups.
4. Call `search_documents_deep` (2-Round Self-Corrective CRAG StateGraph) for multi-part, analytical, cross-document, or difficult Arabic OCR questions — or if `search_documents_fast` returns weak/incomplete results.
5. Call `search_structured_data` for questions involving tables, spreadsheets, CSV, Excel, or JSON records.
6. Call `summarize_conversation_history` when the user explicitly asks to summarize the current chat conversation.

Response rules:
- Reply in the same language as the user's question (Arabic or English).
- Cite the exact Source File and Location (Page, Slide, Sheet/Rows, Table, Section) for every factual claim.
- If the retrieved passages do not cover the question (or only cover part of it), state clearly what is supported and what is not found in the database."""


class AgentState(TypedDict):
    """Conversational state tracked across Agent <-> ToolNode iterations."""
    # `add_messages` reducer appends each new AIMessage / ToolMessage instead of overwriting the list
    messages: Annotated[list, add_messages]
    # Monotonic counter of `agent_node` invocations, enforced against MAX_AGENT_STEPS
    steps: int


def _resolve_target_files(
    requested_files: list[str] | None,
    allowed_files: list[str] | None,
) -> list[str] | None:
    """Intersects agent-requested filenames with the user's sidebar selection scope."""
    # Security/scope guardrail: the agent may narrow the user's file selection, never widen it
    if requested_files and allowed_files:
        allowed_set = set(allowed_files)
        scoped = [f for f in requested_files if f in allowed_set]
        # Fall back to the full user-allowed scope if the agent hallucinated an out-of-scope filename
        return scoped or allowed_files
    return requested_files or allowed_files


def _build_tools(
    allowed_files: list[str] | None,
    session_id: str,
    collected_contexts: list[str],
):
    """Creates per-run LangChain tools bound to the active file scope, session, and chunk collector."""

    # Closure helper that deduplicates every chunk returned by any search tool so App.py
    # can display the full multi-hop evidence trail inside its "Retrieved Context Chunks" expander.
    def _record_chunks(labelled: list[str]):
        seen = set(collected_contexts)
        for item in labelled:
            if item not in seen:
                collected_contexts.append(item)
                seen.add(item)

    @tool
    def list_indexed_documents() -> str:
        """List all indexed documents available in the knowledge base with their file type and chunk count."""
        # Ensure the shared in-memory BM25 metadata cache matches the latest ChromaDB state
        refresh_global_bm25()
        counts: dict[str, dict] = {}
        allowed_set = set(allowed_files) if allowed_files else None
        for meta in GLOBAL_BM25_CACHE["metadatas"]:
            if not isinstance(meta, dict):
                continue
            src = meta.get("source")
            if not src:
                continue
            # Respect the user's sidebar file filter so hidden files are never exposed
            if allowed_set is not None and src not in allowed_set:
                continue
            info = counts.setdefault(
                src,
                {
                    "chunks": 0,
                    "file_type": meta.get("file_type", "pdf"),
                    "data_kind": meta.get("data_kind", "unstructured"),
                },
            )
            info["chunks"] += 1

        if not counts:
            return "No indexed documents are available in the current scope."
        lines = [
            f"- {src} (type={info['file_type']}, kind={info['data_kind']}, chunks={info['chunks']})"
            for src, info in sorted(counts.items())
        ]
        return f"Available documents ({len(counts)}):\n" + "\n".join(lines)

    @tool
    def search_documents_fast(query: str, target_files: list[str] | None = None) -> str:
        """Fast single-round Hybrid RRF search (Dense Vector + Arabic-aware BM25, top 6 chunks).
        Use for straightforward factual questions."""
        # Intersect any agent-requested filenames with the sidebar's allowed file scope
        scoped = _resolve_target_files(target_files, allowed_files)
        docs = hybrid_search(query, top_n=6, selected_files=scoped)
        if not docs:
            return "No matching passages found."
        full_text, labelled = format_context(docs)
        _record_chunks(labelled)
        return full_text

    @tool
    def search_documents_deep(query: str, target_files: list[str] | None = None) -> str:
        """Deep 2-Round Self-Corrective CRAG StateGraph search with batch grading, query rewriting,
        and hallucination/citation verification. Use for complex, multi-part, or hard Arabic questions."""
        scoped = _resolve_target_files(target_files, allowed_files)
        # Invoke CRAG StateGraph as a stateless nested sub-graph (history_limit=0, no chat_history write)
        crag_answer, verified_contexts, meta = run_crag_graph_stateless(
            user_query=query,
            history_limit=0,
            selected_files=scoped,
            session_id=session_id,
        )
        _record_chunks(verified_contexts)
        if not verified_contexts:
            return f"[CRAG Result — No Relevant Passages]: {crag_answer}"
        # Return BOTH the verified CRAG synthesis and the underlying cited passages so the agent can verify/combine
        evidence = "\n\n---\n\n".join(verified_contexts)
        return (
            f"[CRAG Verified Synthesis (round2={meta.get('round2_triggered')}, "
            f"verified={meta.get('verified')})]:\n{crag_answer}\n\n"
            f"[Verified Source Passages]:\n{evidence}"
        )

    @tool
    def search_structured_data(query: str, target_files: list[str] | None = None) -> str:
        """Search specifically across structured tabular/JSON documents (CSV, Excel, JSON)."""
        refresh_global_bm25()
        # Collect only sources indexed with data_kind="structured" (spreadsheets, CSV, JSON)
        structured_sources = sorted({
            m.get("source")
            for m in GLOBAL_BM25_CACHE["metadatas"]
            if isinstance(m, dict) and m.get("data_kind") == "structured" and m.get("source")
        })
        if not structured_sources:
            return "No structured files (CSV/Excel/JSON) are currently indexed in the database."
        # Apply user sidebar filter first, then optional agent-requested target_files filter
        if allowed_files:
            allowed_set = set(allowed_files)
            structured_sources = [s for s in structured_sources if s in allowed_set]
        if target_files:
            req_set = set(target_files)
            filtered = [s for s in structured_sources if s in req_set]
            if filtered:
                structured_sources = filtered
        if not structured_sources:
            return "No structured files match the current file selection."

        # Pull a slightly wider top_n=8 window because structured rows are compact
        docs = hybrid_search(query, top_n=8, selected_files=structured_sources)
        if not docs:
            return "No matching rows/records found in structured files."
        full_text, labelled = format_context(docs)
        _record_chunks(labelled)
        return full_text

    @tool
    def summarize_conversation_history() -> str:
        """Summarize the conversation history of the current active chat session."""
        # Read saved turns for the active session and delegate to the shared @traceable summarizer
        turns = load_file_history(session_id)
        if not turns:
            return "There is no prior conversation history in this chat session yet."
        history_text = "\n".join(
            f"Q: {t.get('query')} | A: {t.get('answer')}" for t in turns
        )
        return summarize_cross_file_history(history_text)

    return [
        list_indexed_documents,
        search_documents_fast,
        search_documents_deep,
        search_structured_data,
        summarize_conversation_history,
    ]


def build_agent_graph(tools: list):
    """Compiles a LangGraph tool-calling Agent StateGraph with a hard step-count guardrail."""
    # `llm_with_tools` can emit tool calls; `plain_llm` has NO tools bound for forced final synthesis
    llm_with_tools = get_llm(temperature=0.2).bind_tools(tools)
    plain_llm = get_llm(temperature=0.2)
    # LangGraph's built-in ToolNode executes all tool_calls in the last AIMessage and returns ToolMessages
    tool_node = ToolNode(tools)

    def agent_node(state: AgentState) -> dict:
        # Ask the model to either call one or more tools or produce the final cited answer
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response], "steps": state.get("steps", 0) + 1}

    def finalize_node(state: AgentState) -> dict:
        """Safety synthesis node invoked if MAX_AGENT_STEPS is reached while still requesting tools."""
        # Append a synthetic instruction and invoke `plain_llm` (no tools bound) to guarantee termination
        prompt_msgs = list(state["messages"]) + [
            HumanMessage(
                content=(
                    "You have reached the maximum tool-call budget. Synthesize the best possible "
                    "cited final answer right now using only the tool outputs already gathered."
                )
            )
        ]
        response = plain_llm.invoke(prompt_msgs)
        return {"messages": [response]}

    def _route_agent(state: AgentState) -> str:
        # Inspect the most recent assistant message to decide whether to run tools, finalize, or stop
        last_msg = state["messages"][-1]
        has_tools = isinstance(last_msg, AIMessage) and bool(getattr(last_msg, "tool_calls", None))
        if not has_tools:
            return END
        if state.get("steps", 0) >= MAX_AGENT_STEPS:
            logger.warning(f"[Agentic RAG] Reached MAX_AGENT_STEPS={MAX_AGENT_STEPS}; finalizing.")
            return "finalize"
        return "tools"

    # Assemble the 3-node cyclic agent graph: START -> agent <-> tools, with escape hatch -> finalize -> END
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        _route_agent,
        {"tools": "tools", "finalize": "finalize", END: END},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("finalize", END)
    return builder.compile()


def run_agentic_rag(
    user_query: str,
    history_limit: int = 5,
    selected_files: list = None,
    session_id: str = GLOBAL_SESSION_ID,
):
    """Answers one question with the LangGraph Tool-Calling Agentic RAG.

    Returns ``(answer, collected_labelled_contexts)`` — drop-in compatible with ``App.py``.
    """
    logger.info("=== Starting Agentic RAG (LangGraph) ===")
    logger.info(f"Session: '{session_id}' | User Query: '{user_query}'")

    if global_hybrid_col.count() == 0:
        logger.error("Global Hybrid DB is empty. Please upload and index documents first.")
        return "The database is empty. Please upload and index some PDFs first.", []

    collected_contexts: list[str] = []
    tools = _build_tools(
        allowed_files=selected_files,
        session_id=session_id,
        collected_contexts=collected_contexts,
    )
    graph = build_agent_graph(tools)

    recent_history = history_messages(limit=history_limit, session_id=session_id)
    initial_messages = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        *recent_history,
        HumanMessage(content=user_query),
    ]

    try:
        result = graph.invoke(
            {"messages": initial_messages, "steps": 0},
            config=build_trace_config(
                run_name="AgenticRAG",
                tags=["agentic_rag", "langgraph", "tool_calling", "bilingual"],
                metadata={
                    "pipeline": "agentic_rag",
                    "session_id": session_id,
                    "selected_files": selected_files or ["ALL"],
                },
            ),
        )
    except Exception as exc:
        logger.critical(f"Agentic RAG (LangGraph) execution failed: {exc}", exc_info=True)
        raise

    # Extract the final assistant text message
    final_answer = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            final_answer = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    full_context_text = "\n\n---\n\n".join(collected_contexts)
    logger.info(
        f"[Agentic RAG] Completed in {result.get('steps', 0)} step(s) with "
        f"{len(collected_contexts)} collected context chunk(s)."
    )
    persist_chat_history(session_id, user_query, final_answer, full_context_text)
    return final_answer, collected_contexts
