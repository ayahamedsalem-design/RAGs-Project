"""Unified 7-pipeline Streamlit web application for the Enterprise Global RAG System.

Provides:
- Unified PDF ingestion and deletion across all 3 ChromaDB store groups (Standard, Hierarchical, Ext).
- Querying across all 7 RAG architectures (Hybrid, Hybrid Corrective, Basic, Parent-Child,
  Parent-Child Corrective, Ext Naive, and Ext Corrective).
- Multi-session conversation management (New Chat, saved chats switcher, delete/clear).
- Flexible history summarization (current chat, selected chats, or all chats) with cost tracking.
"""

import os
import logging

# Set before any ChromaDB PersistentClient opens ./chroma_db_global so all packages share settings
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import streamlit as st
from dotenv import load_dotenv

# Shared core utilities from hybrid_rag/ (kept unmodified for easy merges from refactor/hybrid-cleanup)
from hybrid_rag.logging_setup import setup_logging
from hybrid_rag.summarizer import summarize_cross_file_history
from hybrid_rag.retry import execute_with_retry
from hybrid_rag.chat_history import (
    load_file_history,
    clear_all_history,
    list_sessions,
    create_session_id,
    load_sessions_history,
    delete_session,
)

# Unified 7-pipeline registry and ChromaDB coordinator (hybrid_rag + other_rags + ext_rag)
from other_rags.registry import (
    COST_SYSTEM,
    PIPELINES,
    get_all_indexed_files,
    index_file_all_stores,
    remove_file_all_stores,
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
# (hybrid_rag/pdf_processing.py).
# Names used in the "system" column of cost_log.csv are defined in other_rags/registry.py (COST_SYSTEM).


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
    "Query across **ALL** your indexed documents with Hybrid, Corrective Hybrid, Basic, "
    "Parent-Child, Corrective Parent-Child, Ext Naive, or Ext Corrective RAG."
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
    indexed_files = get_all_indexed_files()

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
                remove_file_all_stores(fname)
            st.success(f"Removed {len(files_to_remove)} file(s) from the Global Database!")
            st.rerun()
    else:
        selected_files = []
        st.warning("No files currently embedded in ChromaDB. Upload a PDF below.")

    st.subheader("Upload New File(s)")
    uploaded_files = st.file_uploader(
        "Upload PDF(s) to Global DB:",
        type=["pdf"],
        accept_multiple_files=True,
        key="global_uploader",
    )

    if uploaded_files:
        if st.button("Index File(s) into Global DB", use_container_width=True):
            indexed_count = 0
            for uploaded_file in uploaded_files:
                with st.spinner(f"Indexing '{uploaded_file.name}' across Standard, Hierarchical, and Ext DBs..."):
                    try:
                        # Indexes into all 3 ChromaDB store groups (each skips if already present).
                        # Embedding costs are saved as an "ingest_global" row in logs/cost_log.csv.
                        with cost.track("ingest_global", uploaded_file.name):
                            index_file_all_stores(uploaded_file)

                        logger.info(f"Successfully indexed '{uploaded_file.name}' across all Global DBs.")
                        st.success(f"'{uploaded_file.name}' added to the Global Database!")
                        indexed_count += 1
                    except Exception as e:
                        logger.error(f"Indexing failed for '{uploaded_file.name}': {e}", exc_info=True)
                        st.error(f"Error during indexing '{uploaded_file.name}': {e}")
            if indexed_count > 0:
                st.rerun()

    st.divider()
    st.header("2. Query Architecture")
    rag_type = st.radio(
        "Choose RAG Architecture:",
        tuple(PIPELINES.keys()),
        index=0,
    )

    st.divider()
    st.header("3. Conversations")

    # Start a brand-new conversation while keeping all previous chats accessible below
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
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
            logger.info(f"Summarizing {len(selected_history)} turn(s) across {scope_label}.")
            with st.spinner("Summarizing conversation history..."):
                history_text = "\n".join(
                    [
                        f"[{title_map.get(item.get('file_name'), item.get('file_name'))}] "
                        f"Q: {item.get('query')} | A: {item.get('answer')}"
                        for item in selected_history
                    ]
                )
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

active_history = load_file_history(active_session_id)

# Render chat history for the active conversation
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
                # run_tracked = the usual execute_with_retry + cost tracking (see the top)
                answer, retrieved_contexts = run_tracked(
                    rag_type,
                    PIPELINES[rag_type],
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
                                header_parts = [
                                    p for p in (chunk.get("file"), chunk.get("loc")) if p
                                ]
                                if chunk.get("sim"):
                                    header_parts.append(f"sim: {chunk['sim']}")
                                if header_parts:
                                    st.caption(" | ".join(header_parts))
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