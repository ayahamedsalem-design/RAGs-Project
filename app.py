import os
import logging
import streamlit as st
from dotenv import load_dotenv

# Global RAG imports from RAGs package
from RAGs.logger_config import setup_logging
from RAGs.basic_RAG import run_basic_RAG, index_basic_rag
from RAGs.parent_child_RAG import (
    run_parent_child_chunking_RAG,
    index_parent_child_rag,
)
from RAGs.hybrid_RAG import run_hybrid_chat_session, initialize_hybrid_index
from RAGs.corrective_parent_child_RAG import run_corrective_parent_child_rag
from RAGs.corrective_hybrid_RAG import run_corrective_hybrid_rag
from RAGs.history_summarizer import summarize_cross_file_history
from RAGs.RAG_service import execute_with_retry
from RAGs.chat_history import load_file_history, clear_all_history, load_all_history

# Legacy RAG imports from RAG package
from RAG.app_bridge import run_naive_rag, run_corrective_rag, load_db_history

# Cost tracking for every pipeline (see RAG/cost.py): tokens and US dollars per question
from RAG import cost

# Initializes logging (writes logging_config.json on first run)
setup_logging()
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Global Knowledge Base RAG", page_icon="🤖", layout="wide")

# ==========================================
# Cost tracking
# ==========================================
# Every request to OpenRouter reports what it cost (tokens and US dollars). RAG/cost.py adds
# those up per question and writes one row per question to RAG/cost_log.csv, which the
# evaluation and the Usage page read.
#   - Ext Naive / Ext Corrective RAG track themselves (inside RAG/app_bridge.py).
#   - The RAGs/ pipelines below are tracked here: cost.track(...) starts a new total before the
#     pipeline runs and saves it when it ends. Their requests are recorded by the hook in
#     RAGs/extract_and_chunk_pdf.py (initialize_api).
# Names used in the "system" column of cost_log.csv:
COST_SYSTEM = {
    "Basic RAG": "basic",
    "Parent-Child (Hierarchical) RAG": "parent_child",
    "Hybrid RAG": "hybrid",
    "Corrective Parent-Child RAG": "crag_parent_child",
    "Corrective Hybrid RAG": "crag_hybrid",
}


def run_tracked(rag_type, rag_func, user_query):
    """Run one RAGs/ pipeline (with the usual retries) and record what the question cost.
    Requests from a failed attempt that was retried are included: they were paid for too."""
    with cost.track(COST_SYSTEM[rag_type], user_query):
        return execute_with_retry(rag_func, user_query)


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
    "Query across **ALL** your indexed documents using Global architectures or External RAG pipelines."
)
st.divider()

# Unified session ID for global cross-document chat history
GLOBAL_SESSION_ID = "global_chat_session"

# ==========================================
# 1. Sidebar: Knowledge Base Ingestion & Settings
# ==========================================
with st.sidebar:
    st.header("1. Feed the Knowledge Base")
    st.info("Upload PDF documents here to index them permanently into the Global Database.")

    uploaded_file = st.file_uploader("Upload PDF to Global DB:", type=["pdf"], key="global_uploader")

    if uploaded_file:
        if st.button("Index File into Global DB", use_container_width=True):
            with st.spinner(f"Indexing '{uploaded_file.name}' across Basic, Hierarchical, and Hybrid DBs..."):
                try:
                    # Index into all Global RAG architectures simultaneously.
                    # The embedding cost of all three is saved as one "ingest_global" row in
                    # RAG/cost_log.csv (the Ext pipelines' indexing is "ingest").
                    with cost.track("ingest_global", uploaded_file.name):
                        index_basic_rag(uploaded_file)
                        index_parent_child_rag(uploaded_file)
                        initialize_hybrid_index(uploaded_file)

                    logger.info(f"Successfully indexed '{uploaded_file.name}' into all Global DBs.")
                    st.success(f"'{uploaded_file.name}' added to the Global Database!")
                except Exception as e:
                    logger.error(f"Indexing failed for '{uploaded_file.name}': {e}", exc_info=True)
                    st.error(f"Error during indexing: {e}")

    st.divider()
    st.header("2. Query Architecture")
    rag_type = st.radio(
        "Choose RAG Architecture:",
        (
            "Basic RAG",
            "Parent-Child (Hierarchical) RAG",
            "Hybrid RAG",
            "Corrective Parent-Child RAG",
            "Corrective Hybrid RAG",
            "Ext Naive RAG",
            "Ext Corrective RAG"
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
# Identify if the selected architecture is an external one
is_ext_mode = "Ext" in rag_type

# Load appropriate conversation history based on Global vs Ext mode
if is_ext_mode:
    system_key = "naive" if "Naive" in rag_type else "crag"
    # Assuming GLOBAL_SESSION_ID is used as the session name for external RAGs as well
    active_history = load_db_history(GLOBAL_SESSION_ID, system_name=system_key)
else:
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
                # Route to Global Multi-Document Architectures
                # (run_tracked = the usual execute_with_retry + cost tracking, see the top)
                if rag_type == "Basic RAG":
                    answer, retrieved_contexts = run_tracked(
                        rag_type, run_basic_RAG, user_query
                    )
                elif rag_type == "Parent-Child (Hierarchical) RAG":
                    answer, retrieved_contexts = run_tracked(
                        rag_type, run_parent_child_chunking_RAG, user_query
                    )
                elif rag_type == "Hybrid RAG":
                    answer, retrieved_contexts = run_tracked(
                        rag_type, run_hybrid_chat_session, user_query
                    )
                elif rag_type == "Corrective Parent-Child RAG":
                    answer, retrieved_contexts = run_tracked(
                        rag_type, run_corrective_parent_child_rag, user_query
                    )
                elif rag_type == "Corrective Hybrid RAG":
                    answer, retrieved_contexts = run_tracked(
                        rag_type, run_corrective_hybrid_rag, user_query
                    )

                # Route to External Architectures (Session Name + Query)
                elif rag_type == "Ext Naive RAG":
                    answer, retrieved_contexts = execute_with_retry(
                        run_naive_rag, GLOBAL_SESSION_ID, user_query
                    )
                elif rag_type == "Ext Corrective RAG":
                    result = execute_with_retry(
                        run_corrective_rag, GLOBAL_SESSION_ID, user_query
                    )
                    if len(result) == 3:
                        answer, retrieved_contexts, crag_log = result
                    else:
                        answer, retrieved_contexts = result

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