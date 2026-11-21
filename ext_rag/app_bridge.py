import logging

from hybrid_rag.logging_setup import setup_logging
import ext_rag.ask as ext_ask
import ext_rag.ask_crag as ext_crag
import ext_rag.memory as memory
from ext_rag.rag_common import (
    RAGError,
    open_store,
    store_exists,
    stored_files,
    TOP_K,
    MAX_CONTEXT,
    chat,
)

setup_logging()
logger = logging.getLogger("Ext_RAG_Bridge")

GLOBAL_SESSION_ID = "global_chat_session"
EMPTY_STORE_MSG = (
    "The Ext knowledge base is empty. Upload and index a PDF in the sidebar "
    "(or run: python -m ext_rag.ingest)."
)


def _open_collection():
    if not store_exists():
        return None
    col = open_store()
    return col if col.count() else None


def _resolve_where_filter(col, selected_files):
    """
    Returns (where_filter, active_file_count).
    If selected_files is empty/None or selects all stored files, where_filter is None.
    """
    all_paths = {f["path"] for f in stored_files(col).values()}
    if not selected_files:
        return None, len(all_paths)
    selected_set = set(selected_files)
    matching = all_paths & selected_set
    if not matching:
        return {"path": {"$in": list(selected_set)}}, 0
    if all_paths.issubset(selected_set):
        return None, len(all_paths)
    return {"path": {"$in": sorted(matching)}}, len(matching)


def _format_context_entries(contexts):
    return [
        f"[Source: {c['file']} | {c['loc']} | sim: {c['sim']}]:\n{c['text']}"
        for c in contexts
    ]


# ----------------- 1. Ext Naive RAG Runner -----------------
def run_ext_naive_rag(user_query: str, selected_files: list = None):
    """
    Matches the standard pipeline contract: (user_query, selected_files=None) -> (answer, contexts).
    Cost tracking is managed by the caller (app.py's run_tracked or evaluation/run_eval.py).
    """
    try:
        col = _open_collection()
    except RAGError as e:
        return str(e), []
    if col is None:
        return EMPTY_STORE_MSG, []

    where_filter, active_files = _resolve_where_filter(col, selected_files)
    if active_files == 0:
        return "No indexed chunks were found for the selected file(s) in the Ext DB.", []

    turns = memory.recent_turns(GLOBAL_SESSION_ID)
    standalone = memory.rewrite_question(user_query, turns)
    results = ext_ask.search(col, standalone, TOP_K, where=where_filter)

    if not results:
        return "No relevant passages were found for the selected file(s).", []

    prompt_msgs = ext_ask.build_prompt(standalone, results)
    messages = memory.add_history_to_prompt(prompt_msgs, turns)
    answer = chat(messages)

    retrieved_contexts = [
        {
            "file": path,
            "loc": loc,
            "sim": f"{sim:.3f}",
            "text": text,
        }
        for sim, path, loc, text in results
    ]

    memory.save_turn(
        GLOBAL_SESSION_ID,
        user_query,
        standalone,
        answer,
        _format_context_entries(retrieved_contexts),
        "naive",
    )
    return answer, retrieved_contexts


# ----------------- 2. Ext Corrective RAG (CRAG) Runner -----------------
def run_ext_corrective_rag_with_log(user_query: str, selected_files: list = None):
    """Same as run_ext_corrective_rag, also returning the CRAG step log (for evaluation)."""
    try:
        col = _open_collection()
    except RAGError as e:
        return str(e), [], ext_crag.Log()
    if col is None:
        return EMPTY_STORE_MSG, [], ext_crag.Log()

    where_filter, active_files = _resolve_where_filter(col, selected_files)
    if active_files == 0:
        return "No indexed chunks were found for the selected file(s) in the Ext DB.", [], ext_crag.Log()

    turns = memory.recent_turns(GLOBAL_SESSION_ID)
    standalone = memory.rewrite_question(user_query, turns)

    log = ext_crag.Log()
    chunks, status, missing = ext_crag.corrective_retrieve(
        col, standalone, log, where=where_filter, active_file_count=active_files
    )

    if not chunks:
        answer = ext_crag.NOT_COVERED
        memory.save_turn(GLOBAL_SESSION_ID, user_query, standalone, answer, [], "crag")
        return answer, [], log

    for c in chunks:
        if c.get("via") != "question":
            c["sim"] = ext_crag.similarity_to(col, log.qvec, c["id"])

    chunks = sorted(chunks, key=lambda c: -c["sim"])[:MAX_CONTEXT]
    results = [(c["sim"], c["path"], c["loc"], c["text"]) for c in chunks]

    messages = ext_crag.build_prompt(standalone, results)
    if status == "PARTIAL":
        gap = f" Likely missing: {missing}." if missing else ""
        messages[1]["content"] += (
            f"\n\nNOTE: the search could not find everything.{gap} "
            "Answer what the sources support and say clearly which parts the documents do not cover."
        )

    memory.add_history_to_prompt(messages, turns)
    answer = chat(messages)

    retrieved_contexts = [
        {
            "file": c["path"],
            "loc": c["loc"],
            "sim": f"{c['sim']:.3f}",
            "via": c.get("via", "question"),
            "text": c["text"],
        }
        for c in chunks
    ]

    memory.save_turn(
        GLOBAL_SESSION_ID,
        user_query,
        standalone,
        answer,
        _format_context_entries(retrieved_contexts),
        "crag",
    )
    return answer, retrieved_contexts, log


def run_ext_corrective_rag(user_query: str, selected_files: list = None):
    """
    Matches the standard pipeline contract: (user_query, selected_files=None) -> (answer, contexts).
    """
    answer, retrieved_contexts, _ = run_ext_corrective_rag_with_log(
        user_query, selected_files=selected_files
    )
    return answer, retrieved_contexts
