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
# Maximum number of full recent Q&A turns kept verbatim per session before older turns are
# automatically compacted into the session's rolling summary buffer.
DEFAULT_HISTORY_WINDOW = 6
# Marker value stored in `entry_type` to distinguish a session's rolling summary record from Q&A turns
SUMMARY_ENTRY_TYPE = "summary"


def _is_summary_entry(item: dict) -> bool:
    """Returns True if a JSON history record holds a session's rolling summary rather than a Q&A turn."""
    return isinstance(item, dict) and item.get("entry_type") == SUMMARY_ENTRY_TYPE


def _format_title(raw_text: str, fallback: str, max_title_len: int = 42) -> str:
    """Builds a compact single-line sidebar title from the session's first user question."""
    cleaned = " ".join(str(raw_text or "").split())
    if not cleaned:
        return fallback
    if len(cleaned) > max_title_len:
        return cleaned[: max_title_len - 1].rstrip() + "…"
    return cleaned


def _read_history_file(filepath: str = DEFAULT_HISTORY_PATH) -> list:
    """Safely loads the raw JSON list from disk, returning an empty list on error."""
    if not os.path.exists(filepath):
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError as exc:
            logger.warning(f"Failed to parse chat history file '{filepath}': {exc}")
            return []


def _write_history_file(history_data: list, filepath: str = DEFAULT_HISTORY_PATH) -> None:
    """Atomically writes the chat history list as UTF-8 JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=4, ensure_ascii=False)


def create_session_id() -> str:
    """Generates a unique, chronologically sortable chat session identifier."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"chat_{stamp}_{uuid.uuid4().hex[:6]}"


def load_session_summary(file_name: str, filepath: str = DEFAULT_HISTORY_PATH) -> tuple[str, int]:
    """Returns ``(summary_text, summarized_turn_count)`` for ``file_name`` (session ID).

    Returns ``("", 0)`` if the session has not yet exceeded the active turn window.
    """
    history_data = _read_history_file(filepath)
    for item in history_data:
        if item.get("file_name") == file_name and _is_summary_entry(item):
            return str(item.get("summary") or "").strip(), int(item.get("summarized_count") or 0)
    return "", 0


def build_history_messages(
    session_id: str,
    limit: int = DEFAULT_HISTORY_WINDOW,
    filepath: str = DEFAULT_HISTORY_PATH,
) -> list[dict]:
    """Builds OpenAI-format history messages (optional rolling summary system note + latest
    ``limit`` user/assistant pairs) shared across all native RAG pipelines."""
    if not limit:
        return []

    messages: list[dict] = []
    # Prepend compacted summary of earlier turns so the LLM never loses early conversation context
    earlier_summary, summarized_count = load_session_summary(session_id, filepath=filepath)
    if earlier_summary:
        messages.append({
            "role": "system",
            "content": (
                f"[Summary of {summarized_count} earlier conversation turn(s) in this chat]:\n"
                f"{earlier_summary}"
            ),
        })

    # Append the latest `limit` full Q&A turns verbatim (default 6)
    turns = load_file_history(session_id, filepath=filepath)[-limit:]
    logger.info(
        f"Using {len(turns)} recent turn(s) + {summarized_count} summarized earlier turn(s) "
        f"from session '{session_id}'."
    )
    for item in turns:
        messages.append({"role": "user", "content": item["query"]})
        messages.append({"role": "assistant", "content": item["answer"]})
    return messages


def persist_chat_history(
    file_name,
    user_query,
    assistant_reply,
    context,
    max_items=DEFAULT_HISTORY_WINDOW,
    filepath=DEFAULT_HISTORY_PATH,
    summarizer_fn=None,
):
    """Appends one Q&A turn to ``file_name`` (session ID), keeping the latest ``max_items``
    full turns and automatically compacting older overflow turns into a rolling summary via
    ``summarize_cross_file_history``."""
    entry = {
        "file_name": file_name,
        "query": user_query,
        "answer": assistant_reply,
        "rag_context": context,
    }
    history_data = _read_history_file(filepath)

    # Separate other sessions, any existing rolling summary for this session, and its Q&A turns
    other_sessions = [item for item in history_data if item.get("file_name") != file_name]
    session_items = [item for item in history_data if item.get("file_name") == file_name]
    existing_summary_rec = next((item for item in session_items if _is_summary_entry(item)), None)
    current_turns = [item for item in session_items if not _is_summary_entry(item)]

    # Append the newly completed Q&A turn
    current_turns.append(entry)

    updated_summary_rec = existing_summary_rec
    # When turns exceed the active window (default 6), summarize the oldest overflow turn(s)
    # and merge them into the session's rolling summary instead of silently discarding them.
    if max_items and len(current_turns) > max_items:
        overflow_turns = current_turns[:-max_items]
        kept_turns = current_turns[-max_items:]

        prev_summary = str((existing_summary_rec or {}).get("summary") or "").strip()
        prev_count = int((existing_summary_rec or {}).get("summarized_count") or 0)
        # Preserve the very first user question as the session's permanent sidebar title
        session_title = (existing_summary_rec or {}).get("title") or _format_title(
            overflow_turns[0].get("query", ""), file_name
        )

        # Format overflow turns concisely for the shared summarizer
        overflow_text = "\n\n".join(
            f"User: {t.get('query', '')}\nAssistant: {t.get('answer', '')}"
            for t in overflow_turns
        )

        # Reuse hybrid_rag.summarizer.summarize_cross_file_history lazily to avoid circular imports
        if summarizer_fn is None:
            from hybrid_rag.summarizer import summarize_cross_file_history
            summarizer_fn = summarize_cross_file_history

        try:
            merged_summary = (
                summarizer_fn(overflow_text, existing_summary=prev_summary, rolling=True) or ""
            ).strip()
        except Exception as exc:
            # Fail-safe: never drop the user's new answer if the rolling summary API call fails
            logger.warning(
                f"Rolling summary compaction failed for session '{file_name}' ({exc}); "
                "retaining previous summary text."
            )
            merged_summary = prev_summary

        # Always keep the summary record once overflow occurs so title & summarized count stay accurate
        updated_summary_rec = {
            "file_name": file_name,
            "entry_type": SUMMARY_ENTRY_TYPE,
            "title": session_title,
            "summary": merged_summary,
            "summarized_count": prev_count + len(overflow_turns),
        }
        current_turns = kept_turns
        logger.info(
            f"Compacted {len(overflow_turns)} older turn(s) into rolling summary for session "
            f"'{file_name}' (total summarized={updated_summary_rec['summarized_count']}, "
            f"visible window={len(current_turns)})."
        )

    # Place the active session (summary record first, then latest turns) at the end so
    # list_sessions() surfaces it as the most recently updated conversation.
    active_block = ([updated_summary_rec] if updated_summary_rec else []) + current_turns
    _write_history_file(other_sessions + active_block, filepath)
    logger.debug(f"Persisted turn to session '{file_name}' (visible turns: {len(current_turns)}).")


def load_file_history(file_name, filepath=DEFAULT_HISTORY_PATH):
    """Returns the visible Q&A turns belonging to the given session ID (excludes summary records)."""
    history_data = _read_history_file(filepath)
    return [
        item
        for item in history_data
        if item.get("file_name") == file_name and not _is_summary_entry(item)
    ]


def load_all_history(filepath=DEFAULT_HISTORY_PATH):
    """Retrieves all visible Q&A turns across all saved sessions (excludes summary records)."""
    return [item for item in _read_history_file(filepath) if not _is_summary_entry(item)]


def load_sessions_history(session_ids=None, filepath=DEFAULT_HISTORY_PATH):
    """Returns history entries (including any rolling summaries) for the requested session IDs
    (or all sessions if ``session_ids`` is empty/None)."""
    history_data = _read_history_file(filepath)
    if not session_ids:
        return history_data
    allowed = set(session_ids)
    return [item for item in history_data if item.get("file_name") in allowed]


def list_sessions(filepath=DEFAULT_HISTORY_PATH, max_title_len: int = 42) -> list[dict]:
    """Returns saved sessions ordered from most recently updated to oldest.

    Each item is ``{"id": session_id, "title": preview_title, "count": total_turn_count}``,
    where ``count`` includes both visible recent turns and earlier compacted turns.
    """
    history_data = _read_history_file(filepath)
    sessions_map: dict[str, dict] = {}
    order: list[str] = []

    for item in history_data:
        sid = item.get("file_name")
        if not sid:
            continue
        is_sum = _is_summary_entry(item)
        if sid not in sessions_map:
            # Prefer the preserved first-turn title on the summary record if compaction happened
            initial_title = (
                item.get("title")
                if is_sum and item.get("title")
                else _format_title(item.get("query", ""), sid, max_title_len=max_title_len)
            )
            sessions_map[sid] = {"id": sid, "title": initial_title, "count": 0}
        elif is_sum and item.get("title"):
            sessions_map[sid]["title"] = item["title"]

        # Count summarized earlier turns as well as visible recent turns for accurate badge totals
        sessions_map[sid]["count"] += int(item.get("summarized_count") or 0) if is_sum else 1
        # Track most recent activity position
        if sid in order:
            order.remove(sid)
        order.append(sid)

    return [sessions_map[sid] for sid in reversed(order)]


def delete_session(session_id: str, filepath=DEFAULT_HISTORY_PATH) -> int:
    """Deletes all turns belonging to ``session_id`` and returns how many turns were removed."""
    history_data = _read_history_file(filepath)
    remaining = [item for item in history_data if item.get("file_name") != session_id]
    removed = len(history_data) - len(remaining)
    if removed > 0:
        _write_history_file(remaining, filepath)
        logger.info(f"Deleted chat session '{session_id}' ({removed} turn(s) removed).")
    return removed


def clear_all_history(filepath=DEFAULT_HISTORY_PATH):
    """Wipes the entire chat history JSON file across all sessions."""
    _write_history_file([], filepath)
    logger.info(f"Cleared all chat history in '{filepath}'.")
