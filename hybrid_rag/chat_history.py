"""Persistent JSON-backed chat history store partitioned by session ID.

Each record stores ``file_name`` as the session identifier so legacy single-session
histories (``"global_chat_session"``) remain 100% compatible with multi-chat sessions.
"""

from datetime import datetime
import json
import logging
import os
import uuid

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_PATH = "chat_history.json"


def _read_history_file(filepath: str = DEFAULT_HISTORY_PATH) -> list:
    """Safely loads the raw JSON list from disk, returning an empty list on error."""
    # First-run guard: no history file exists yet
    if not os.path.exists(filepath):
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            # Protect callers against corrupted or non-list JSON structures
            return data if isinstance(data, list) else []
        except json.JSONDecodeError as exc:
            logger.warning(f"Failed to parse chat history file '{filepath}': {exc}")
            return []


def _write_history_file(history_data: list, filepath: str = DEFAULT_HISTORY_PATH) -> None:
    """Writes the chat history list as human-readable UTF-8 JSON (preserving Arabic text)."""
    with open(filepath, "w", encoding="utf-8") as f:
        # ensure_ascii=False keeps Arabic/Unicode readable instead of \uXXXX escapes
        json.dump(history_data, f, indent=4, ensure_ascii=False)


def create_session_id() -> str:
    """Generates a unique, chronologically sortable chat session identifier."""
    # Timestamp prefix makes IDs naturally sortable; 6-hex suffix prevents same-second collisions
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"chat_{stamp}_{uuid.uuid4().hex[:6]}"


def persist_chat_history(
    file_name,
    user_query,
    assistant_reply,
    context,
    max_items=10,
    filepath=DEFAULT_HISTORY_PATH,
):
    """Appends one Q&A turn to ``file_name`` (session ID), capping that session at ``max_items``."""
    # Note: the key is named "file_name" for backward compatibility with older single-session logs
    entry = {
        "file_name": file_name,
        "query": user_query,
        "answer": assistant_reply,
        "rag_context": context,
    }
    history_data = _read_history_file(filepath)

    # Append the new turn and split out other sessions so only the active session is trimmed
    history_data.append(entry)
    other_sessions = [item for item in history_data if item.get("file_name") != file_name]
    current_session = [item for item in history_data if item.get("file_name") == file_name]

    # Cap only the active session to its most recent turns (default: last 10)
    current_session = current_session[-max_items:]

    # Place the active session at the end so list_sessions() surfaces it as most recent
    _write_history_file(other_sessions + current_session, filepath)
    logger.debug(f"Persisted turn to session '{file_name}' (saved turns: {len(current_session)}).")


def load_file_history(file_name, filepath=DEFAULT_HISTORY_PATH):
    """Returns all saved history items belonging to the given session ID."""
    history_data = _read_history_file(filepath)
    return [item for item in history_data if item.get("file_name") == file_name]


def load_all_history(filepath=DEFAULT_HISTORY_PATH):
    """Retrieves the chat history across all saved sessions."""
    return _read_history_file(filepath)


def load_sessions_history(session_ids=None, filepath=DEFAULT_HISTORY_PATH):
    """Returns history entries for the requested session IDs (or all sessions if empty/None)."""
    history_data = _read_history_file(filepath)
    # Empty/None means "summarize across all saved sessions"
    if not session_ids:
        return history_data
    allowed = set(session_ids)
    return [item for item in history_data if item.get("file_name") in allowed]


def list_sessions(filepath=DEFAULT_HISTORY_PATH, max_title_len: int = 42) -> list[dict]:
    """Returns saved sessions ordered from most recently updated to oldest.

    Each item is ``{"id": session_id, "title": preview_title, "count": turn_count}``.
    """
    history_data = _read_history_file(filepath)
    sessions_map: dict[str, dict] = {}
    order: list[str] = []  # Tracks last-seen position of each session ID

    for item in history_data:
        sid = item.get("file_name")
        if not sid:
            continue
        if sid not in sessions_map:
            # Build a compact sidebar preview title from the session's first user question
            raw_query = " ".join(str(item.get("query") or "").split())
            if len(raw_query) > max_title_len:
                title = raw_query[: max_title_len - 1].rstrip() + "…"
            else:
                title = raw_query or sid
            sessions_map[sid] = {"id": sid, "title": title, "count": 0}
        sessions_map[sid]["count"] += 1
        # Move sid to the end of `order` every time a newer turn is seen
        if sid in order:
            order.remove(sid)
        order.append(sid)

    # Reverse so the most recently updated chat appears first in the sidebar
    return [sessions_map[sid] for sid in reversed(order)]


def delete_session(session_id: str, filepath=DEFAULT_HISTORY_PATH) -> int:
    """Deletes all turns belonging to ``session_id`` and returns how many turns were removed."""
    history_data = _read_history_file(filepath)
    remaining = [item for item in history_data if item.get("file_name") != session_id]
    removed = len(history_data) - len(remaining)
    # Only rewrite the file when at least one turn matched the session
    if removed > 0:
        _write_history_file(remaining, filepath)
        logger.info(f"Deleted chat session '{session_id}' ({removed} turn(s) removed).")
    return removed


def clear_all_history(filepath=DEFAULT_HISTORY_PATH):
    """Wipes the entire chat history JSON file across all sessions."""
    _write_history_file([], filepath)
    logger.info(f"Cleared all chat history in '{filepath}'.")
