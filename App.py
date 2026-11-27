"""Main Streamlit web application for the Enterprise Global RAG System.

Provides:
- Multi-format file ingestion and indexing into the shared ChromaDB collection.
- Hybrid RAG and 2-Round Hybrid Corrective RAG (CRAG) query pipelines.
- Multi-session conversation management (New Chat, saved chats switcher, delete/clear).
- Flexible history summarization (current chat, selected chats, or all chats) with cost tracking.
"""

import os
import logging
import streamlit as st
from dotenv import load_dotenv

# The two LangChain LCEL RAG pipelines and shared indexing/history helpers (package hybrid_rag/)
from hybrid_rag.logging_setup import setup_logging
from hybrid_rag.hybrid import (
    initialize_hybrid_index,
    get_indexed_files,
    remove_file_from_index,
)
from hybrid_rag.lc.hybrid_lc import run_hybrid_lc
from hybrid_rag.lc.hybrid_corrective_lc import run_hybrid_corrective_lc
from hybrid_rag.lc.crag_graph import run_crag_graph
from hybrid_rag.lc.agent_graph import run_agentic_rag
from hybrid_rag.file_processing import SUPPORTED_EXTENSIONS
from hybrid_rag.summarizer import summarize_cross_file_history
from hybrid_rag.retry import execute_with_retry
from hybrid_rag.chat_history import (
    load_file_history,
    load_session_summary,
    clear_all_history,
    list_sessions,
    create_session_id,
    load_sessions_history,
    delete_session,
    SUMMARY_ENTRY_TYPE,
)

# Cost tracking (see hybrid_rag/cost.py): tokens and US dollars per question
from hybrid_rag import cost

# Initializes logging (config/logging_config.json)
setup_logging()
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Global Knowledge Base RAG", page_icon="🤖", layout="wide")

# ==========================================
# Cost tracking
# ==========================================
# Every request to OpenRouter reports what it cost (tokens and US dollars). hybrid_rag/cost.py
# adds those up per question and writes one row per question to logs/cost_log.csv, which the
# Usage page reads. cost.track(...) starts a new total before the pipeline runs and saves it
# when it ends; the requests themselves are recorded by the hook in initialize_api()
# (hybrid_rag/file_processing.py).
# Names used in the "system" column of cost_log.csv:
COST_SYSTEM = {
    "Hybrid RAG": "hybrid",
    "Hybrid Corrective RAG": "crag_hybrid",
    "CRAG StateGraph (LangGraph)": "crag_graph",
    "Agentic RAG (LangGraph)": "agentic_rag",
}


def run_tracked(rag_type, rag_func, user_query, selected_files=None, session_id=None):
    """Run one pipeline (with the usual retries) and record what the question cost.
    Requests from a failed attempt that was retried are included: they were paid for too."""
    with cost.track(COST_SYSTEM[rag_type], user_query):
        return execute_with_retry(
            rag_func,
            user_query,
            selected_files=selected_files,
            session_id=session_id or GLOBAL_SESSION_ID,
        )


def show_cost(user_query):
    """One line under the answer: what this question cost (all its model requests together)."""
    qc = cost.last_query()
    if qc is None or not qc.calls or qc.question != user_query:
        return                      # nothing was recorded for this question
    reasoning = f" (+{qc.reasoning_tokens:,} reasoning)" if qc.reasoning_tokens else ""
    st.caption(f"💰 This answer: {qc.n_calls} requests · {qc.prompt_tokens:,} tokens in / "
               f"{qc.completion_tokens:,} out{reasoning} · ${qc.total_usd:.5f}")


st.title("Enterprise Global RAG System ")
st.markdown(
    "Query across **ALL** your indexed documents with Hybrid RAG, Hybrid Corrective RAG, "
    "CRAG StateGraph (LangGraph), or Agentic RAG (LangGraph)."
)
st.divider()

# Default legacy session ID (used when no custom chat session is active)
GLOBAL_SESSION_ID = "global_chat_session"

# Load saved chat sessions and initialize active_session_id once per browser session
saved_sessions = list_sessions()
if "active_session_id" not in st.session_state:
    st.session_state["active_session_id"] = (
        saved_sessions[0]["id"] if saved_sessions else create_session_id()
    )
active_session_id = st.session_state["active_session_id"]

# ==========================================
# 1. Sidebar: Knowledge Base Ingestion & Settings
# ==========================================
with st.sidebar:
    st.header("1. Knowledge Base Files")
    indexed_files = get_indexed_files()

    if indexed_files:
        st.caption("Select files to include in query (none checked = all):")
        selected_files = [
            fname
            for fname in indexed_files
            if st.checkbox(fname, value=True, key=f"query_file_{fname}")
        ]

        files_to_remove = st.multiselect(
            "Select file(s) to remove from DB:",
            options=indexed_files,
        )
        if st.button("Remove Selected File(s)", use_container_width=True, disabled=not files_to_remove):
            for fname in files_to_remove:
                remove_file_from_index(fname)
            logger.info(f"Removed {len(files_to_remove)} file(s) from Global DB: {files_to_remove}")
            st.success(f"Removed {len(files_to_remove)} file(s) from the Global Database!")
            st.rerun()
    else:
        selected_files = []
        st.warning("No files currently embedded in ChromaDB. Upload a file below.")

    st.subheader("Upload New File(s)")
    uploaded_files = st.file_uploader(
        "Upload files (PDF, Word, PowerPoint, Excel, CSV, JSON, TXT/MD, Images):",
        type=list(SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
        key="global_uploader",
    )

    if uploaded_files:
        if st.button("Index File(s) into Global DB", use_container_width=True):
            indexed_count = 0
            for uploaded_file in uploaded_files:
                progress_bar = st.progress(0.0, text=f"Indexing '{uploaded_file.name}'...")
                try:
                    def _update_progress(fraction: float, text: str, _bar=progress_bar):
                        _bar.progress(min(max(float(fraction), 0.0), 1.0), text=text)

                    # Both pipelines search the same collection (global_hybrid_collection).
                    # The embedding cost is saved as an "ingest_global" row in logs/cost_log.csv.
                    with cost.track("ingest_global", uploaded_file.name):
                        initialize_hybrid_index(uploaded_file, progress_callback=_update_progress)

                    logger.info(f"Successfully indexed '{uploaded_file.name}' into the Hybrid DB.")
                    st.success(f"'{uploaded_file.name}' added to the Global Database!")
                    indexed_count += 1
                except Exception as e:
                    logger.error(f"Indexing failed for '{uploaded_file.name}': {e}", exc_info=True)
                    st.error(f"Error during indexing '{uploaded_file.name}': {e}")
                finally:
                    progress_bar.empty()
            if indexed_count > 0:
                st.rerun()

    st.divider()
    st.header("2. Query Architecture")
    rag_type = st.radio(
        "Choose RAG Architecture:",
        (
            "Hybrid RAG",
            "Hybrid Corrective RAG",
            "CRAG StateGraph (LangGraph)",
            "Agentic RAG (LangGraph)",
        ),
        index=0,
    )

    st.divider()
    st.header("3. Conversations")

    # Start a brand-new conversation while keeping all previous chats accessible below
    if st.button("+ New Chat", use_container_width=True, type="primary"):
        new_sid = create_session_id()
        st.session_state["active_session_id"] = new_sid
        st.session_state.pop("history_summary", None)
        logger.info(f"Started new chat session '{new_sid}'.")
        st.rerun()

    saved_ids = [s["id"] for s in saved_sessions]
    title_map = {s["id"]: f"{s['title']} ({s['count']})" for s in saved_sessions}

    # Indicate when the user is currently in a fresh, unsaved conversation
    if active_session_id not in title_map:
        st.caption("💬 Current: New Chat (unsaved)")

    # Clickable list of saved conversations (most recently updated first)
    if saved_sessions:
        st.caption("Saved chats:")
        for sess in saved_sessions:
            sid = sess["id"]
            is_active = sid == active_session_id
            label = f"{'▸ ' if is_active else ''}{sess['title']} ({sess['count']})"
            if st.button(
                label,
                key=f"select_chat_{sid}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                if not is_active:
                    st.session_state["active_session_id"] = sid
                    st.session_state.pop("history_summary", None)
                    logger.info(f"Switched active chat session to '{sid}'.")
                    st.rerun()

    # Delete current chat or wipe all saved conversations
    col_del, col_clear = st.columns(2)
    with col_del:
        if st.button(
            "Delete Chat",
            use_container_width=True,
            disabled=active_session_id not in title_map,
        ):
            delete_session(active_session_id)
            remaining = [s["id"] for s in list_sessions()]
            st.session_state["active_session_id"] = (
                remaining[0] if remaining else create_session_id()
            )
            st.session_state.pop("history_summary", None)
            st.rerun()
    with col_clear:
        if st.button("Clear All", use_container_width=True, disabled=not saved_sessions):
            clear_all_history()
            st.session_state["active_session_id"] = create_session_id()
            st.session_state.pop("history_summary", None)
            st.success("All history cleared!")
            st.rerun()

    st.markdown("---")
    # Flexible summarization: defaults to current chat, supports multi-chat or all chats (empty)
    default_summary_ids = [active_session_id] if active_session_id in saved_ids else []
    chats_to_summarize = st.multiselect(
        "Chats to summarize (empty = all):",
        options=saved_ids,
        default=default_summary_ids,
        format_func=lambda sid: title_map.get(sid, sid),
        key=f"summarize_select_{active_session_id}",
    )

    if st.button("Summarize History", use_container_width=True):
        target_ids = chats_to_summarize or None
        selected_history = load_sessions_history(target_ids)
        if not selected_history:
            st.warning("No history available to summarize.")
        else:
            scope_label = (
                f"{len(chats_to_summarize)} selected chat(s)"
                if chats_to_summarize
                else f"all {len(saved_sessions)} chat(s)"
            )
            logger.info(
                f"Summarizing {len(selected_history)} turn(s) across {scope_label}."
            )
            with st.spinner("Summarizing conversation history..."):
                # Format both compacted earlier-turn summaries and recent verbatim Q&A turns
                # so the manual summary covers 100% of every selected chat session.
                history_lines = []
                for item in selected_history:
                    session_label = title_map.get(item.get("file_name"), item.get("file_name"))
                    if item.get("entry_type") == SUMMARY_ENTRY_TYPE:
                        if item.get("summary"):
                            history_lines.append(
                                f"[{session_label} — Earlier Summary ({item.get('summarized_count', 0)} turns)]: "
                                f"{item.get('summary')}"
                            )
                    else:
                        history_lines.append(
                            f"[{session_label}] Q: {item.get('query')} | A: {item.get('answer')}"
                        )
                history_text = "\n".join(history_lines)
                try:
                    # The summary is a model request too, so it gets its own cost row
                    with cost.track("history_summary", "Summarize History"):
                        st.session_state["history_summary"] = execute_with_retry(
                            summarize_cross_file_history, history_text
                        )
                except Exception as e:
                    logger.error(f"Summary generation failed: {e}", exc_info=True)
                    st.error(f"Failed to generate summary: {e}")


# ==========================================
# 2. Main Area: Chat Interface
# ==========================================
if st.session_state.get("history_summary"):
    with st.expander("📝 History Summary", expanded=True):
        st.markdown(st.session_state["history_summary"])
        show_cost("Summarize History")
        if st.button("Dismiss Summary", key="dismiss_history_summary"):
            st.session_state.pop("history_summary", None)
            st.rerun()

# Load the active session's rolling summary (if >6 turns) plus its latest 6 verbatim Q&A turns
earlier_summary, summarized_count = load_session_summary(active_session_id)
active_history = load_file_history(active_session_id)

# Display older compacted turns inside a single clean collapsible expander above the chat
# so long conversations never clutter the screen while keeping earlier context accessible.
if earlier_summary:
    turn_word = "turn" if summarized_count == 1 else "turns"
    with st.expander(
        f"Earlier conversation summary ({summarized_count} earlier {turn_word} compacted)",
        expanded=False,
    ):
        st.markdown(earlier_summary)

# Render the latest visible Q&A turns (up to 6) for the active conversation
for message in active_history:
    with st.chat_message("user"):
        st.markdown(message["query"])
    with st.chat_message("assistant"):
        st.markdown(message["answer"])
        if "rag_context" in message and message["rag_context"]:
            with st.expander("View Retrieved Context Chunks"):
                st.text(message["rag_context"])

# Handle new user input
user_query = st.chat_input("Ask any question...")

if user_query:
    logger.info(f"Processing query via {rag_type} (session='{active_session_id}')")
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner(f"Running {rag_type}..."):
            try:
                # Route the query to whichever of the 4 architectures is selected in the sidebar
                # (run_tracked wraps each call with execute_with_retry + cost.track)
                if rag_type == "Hybrid RAG":
                    # Architecture 1: Single-round LCEL Hybrid RRF + FlashRank
                    answer, retrieved_contexts = run_tracked(
                        rag_type,
                        run_hybrid_lc,
                        user_query,
                        selected_files=selected_files,
                        session_id=active_session_id,
                    )
                elif rag_type == "Hybrid Corrective RAG":
                    # Architecture 2: 2-Round LCEL CRAG + Neighbor Expansion + RunnableBranch short-circuit
                    answer, retrieved_contexts = run_tracked(
                        rag_type,
                        run_hybrid_corrective_lc,
                        user_query,
                        selected_files=selected_files,
                        session_id=active_session_id,
                    )
                elif rag_type == "CRAG StateGraph (LangGraph)":
                    # Architecture 3: LangGraph cyclic StateGraph adding a Hallucination & Citation Verifier
                    answer, retrieved_contexts = run_tracked(
                        rag_type,
                        run_crag_graph,
                        user_query,
                        selected_files=selected_files,
                        session_id=active_session_id,
                    )
                elif rag_type == "Agentic RAG (LangGraph)":
                    # Architecture 4: LangGraph 5-tool autonomous agent with MAX_AGENT_STEPS=4 cap
                    answer, retrieved_contexts = run_tracked(
                        rag_type,
                        run_agentic_rag,
                        user_query,
                        selected_files=selected_files,
                        session_id=active_session_id,
                    )

                st.subheader("Model Response")
                st.write(answer)
                show_cost(user_query)          # what this question cost (tokens, USD)

                with st.expander("View Retrieved Context Chunks"):
                    if isinstance(retrieved_contexts, list):
                        for idx, chunk in enumerate(retrieved_contexts):
                            st.markdown(f"**Chunk [{idx + 1}]**")
                            if isinstance(chunk, dict):
                                text_content = (
                                    chunk.get("text")
                                    or chunk.get("page_content")
                                    or str(chunk)
                                )
                            else:
                                text_content = str(chunk)
                            st.text(text_content.strip())
                            st.markdown("---")
                    else:
                        st.text(str(retrieved_contexts))

            except Exception as e:
                logger.critical(f"Fatal error during {rag_type} execution: {e}", exc_info=True)
                st.error(f"An error occurred during execution: {e}")