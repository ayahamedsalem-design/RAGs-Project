import os
import logging
import streamlit as st
from dotenv import load_dotenv

# The two RAG pipelines and their helpers (package hybrid_rag/)
from hybrid_rag.logging_setup import setup_logging
from hybrid_rag.hybrid import (
    run_hybrid_chat_session,
    initialize_hybrid_index,
    get_indexed_files,
    remove_file_from_index,
)
from hybrid_rag.hybrid_corrective import run_hybrid_corrective_rag
from hybrid_rag.file_processing import SUPPORTED_EXTENSIONS
from hybrid_rag.summarizer import summarize_cross_file_history
from hybrid_rag.retry import execute_with_retry
from hybrid_rag.chat_history import load_file_history, clear_all_history, load_all_history

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
}


def run_tracked(rag_type, rag_func, user_query, selected_files=None):
    """Run one pipeline (with the usual retries) and record what the question cost.
    Requests from a failed attempt that was retried are included: they were paid for too."""
    with cost.track(COST_SYSTEM[rag_type], user_query):
        return execute_with_retry(rag_func, user_query, selected_files=selected_files)


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
    "Query across **ALL** your indexed documents with Hybrid RAG or Hybrid Corrective RAG (2-Round CRAG)."
)
st.divider()

# Unified session ID for global cross-document chat history
GLOBAL_SESSION_ID = "global_chat_session"

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
        ),
        index=0,
    )

    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        clear_all_history()
        logger.info("Global chat history cleared.")
        st.success("All history cleared!")
        st.rerun()

    if st.button("Summarize History", use_container_width=True):
        all_history = load_all_history()
        if not all_history:
            st.warning("No history available to summarize.")
        else:
            with st.spinner("Summarizing conversation history..."):
                history_text = "\n".join(
                    [f"Q: {item.get('query')} | A: {item.get('answer')}" for item in all_history]
                )
                try:
                    # The summary is a model request too, so it gets its own cost row
                    with cost.track("history_summary", "Summarize History"):
                        summary = summarize_cross_file_history(history_text)
                    st.markdown("**📝 Global History Summary**")
                    st.info(summary)
                except Exception as e:
                    logger.error(f"Summary generation failed: {e}", exc_info=True)
                    st.error(f"Failed to generate summary: {e}")


# ==========================================
# 2. Main Area: Chat Interface
# ==========================================
active_history = load_file_history(GLOBAL_SESSION_ID)

# Render chat history
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
    logger.info(f"Processing query via {rag_type}")
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner(f"Running {rag_type}..."):
            try:
                # run_tracked = the usual execute_with_retry + cost tracking (see the top)
                if rag_type == "Hybrid RAG":
                    answer, retrieved_contexts = run_tracked(
                        rag_type, run_hybrid_chat_session, user_query, selected_files=selected_files
                    )
                elif rag_type == "Hybrid Corrective RAG":
                    answer, retrieved_contexts = run_tracked(
                        rag_type, run_hybrid_corrective_rag, user_query, selected_files=selected_files
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