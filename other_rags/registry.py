"""
Unified registry and ChromaDB coordinator for all 7 RAG pipelines.

Keeps app.py's diff against refactor/hybrid-cleanup minimal so future changes to
hybrid_rag/ and refactor/hybrid-cleanup merge cleanly into feat/all-rags-unified.
"""

import logging
from pathlib import Path

from hybrid_rag.hybrid import (
    run_hybrid_chat_session,
    initialize_hybrid_index,
    get_indexed_files as get_hybrid_indexed_files,
    remove_file_from_index as remove_hybrid_file,
)
from hybrid_rag.corrective_hybrid import run_corrective_hybrid_rag
from other_rags.basic import run_basic_rag
from other_rags.parent_child import (
    run_parent_child_rag,
    index_parent_child_rag,
    get_parent_child_indexed_files,
    remove_parent_child_file,
)
from other_rags.corrective_parent_child import run_corrective_parent_child_rag
from ext_rag.app_bridge import run_ext_naive_rag, run_ext_corrective_rag
from ext_rag.ingest import (
    index_ext_file,
    get_ext_indexed_files,
    remove_ext_file,
)

logger = logging.getLogger(__name__)

# Names used in the "system" column of logs/cost_log.csv
COST_SYSTEM = {
    "Hybrid RAG": "hybrid",
    "Corrective Hybrid RAG": "crag_hybrid",
    "Basic RAG": "basic",
    "Parent-Child (Hierarchical) RAG": "parent_child",
    "Corrective Parent-Child RAG": "crag_parent_child",
    "Ext Naive RAG": "naive",
    "Ext Corrective RAG": "crag",
}

# Ordered mapping of Streamlit radio label -> runner(user_query, selected_files=None)
PIPELINES = {
    "Hybrid RAG": run_hybrid_chat_session,
    "Corrective Hybrid RAG": run_corrective_hybrid_rag,
    "Basic RAG": run_basic_rag,
    "Parent-Child (Hierarchical) RAG": run_parent_child_rag,
    "Corrective Parent-Child RAG": run_corrective_parent_child_rag,
    "Ext Naive RAG": run_ext_naive_rag,
    "Ext Corrective RAG": run_ext_corrective_rag,
}

PIPELINE_BY_KEY = {COST_SYSTEM[label]: fn for label, fn in PIPELINES.items()}


def _reset_stream(pdf_source):
    if hasattr(pdf_source, "seek"):
        try:
            pdf_source.seek(0)
        except Exception:
            pass


def get_all_indexed_files() -> list:
    """Returns a sorted union of unique filenames indexed across all 3 ChromaDB store groups."""
    files = set()
    files.update(get_hybrid_indexed_files())
    files.update(get_parent_child_indexed_files())
    files.update(get_ext_indexed_files())
    return sorted(files)


def index_file_all_stores(pdf_source):
    """
    Indexes a PDF across all 3 ChromaDB store groups:
      1. global_hybrid_collection (serves Basic, Hybrid, and Corrective Hybrid RAG)
      2. global_parent_collection + global_child_collection (serves Parent-Child & Corrective Parent-Child)
      3. ext_chunks_<backend> (serves Ext Naive & Ext Corrective RAG)

    Each indexer skips files already present in its own collection, so re-uploading a PDF
    backfills any missing collection without re-embedding collections that already have it.
    """
    if isinstance(pdf_source, Path):
        pdf_source = str(pdf_source)

    _reset_stream(pdf_source)
    initialize_hybrid_index(pdf_source)

    _reset_stream(pdf_source)
    index_parent_child_rag(pdf_source)

    _reset_stream(pdf_source)
    index_ext_file(pdf_source)


def remove_file_all_stores(file_name: str):
    """Deletes file_name from every ChromaDB collection in sync."""
    logger.info(f"Removing '{file_name}' from all ChromaDB stores...")
    remove_hybrid_file(file_name)
    remove_parent_child_file(file_name)
    remove_ext_file(file_name)
    logger.info(f"Removed '{file_name}' from all ChromaDB stores.")
