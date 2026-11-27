"""Conversation history summarization helper powered by the configured OpenRouter chat model."""

import logging
import os

from hybrid_rag.file_processing import initialize_api
from hybrid_rag.tracing import traceable

logger = logging.getLogger(__name__)

# Shared OpenRouter client (with cost tracking hook) and chat model from .env
client_ai = initialize_api()
CHAT_MODEL = os.getenv("CHAT_MODEL")


@traceable(name="summarize_cross_file_history", run_type="chain")
def summarize_cross_file_history(
    history_text: str,
    existing_summary: str = "",
    rolling: bool = False,
) -> str:
    """Summarizes one or more chat sessions using the configured ``CHAT_MODEL``.

    When ``rolling=True`` (or ``existing_summary`` is provided), merges the prior running
    summary with newly overflowed turns into a compact, bilingual memory summary so long
    conversations stay bounded while preserving earlier facts and entity references.
    """
    is_rolling = rolling or bool(existing_summary and existing_summary.strip())

    if is_rolling:
        # Rolling-buffer compaction mode: merge any prior summary with the turn(s) that just
        # moved outside the active 6-turn window, keeping facts, entities, and citations concise.
        prior_block = (
            f"[Previous Conversation Summary]:\n{existing_summary.strip()}\n\n"
            if existing_summary and existing_summary.strip()
            else ""
        )
        prompt = (
            "You are maintaining a concise running memory summary of an ongoing bilingual "
            "(Arabic & English) RAG conversation. Merge the previous summary (if any) with the "
            "newly archived Q&A turn(s) below into a single compact summary (bullet points, "
            "matching the conversation's language). Preserve key entities, facts, numbers, "
            "cited files/pages, and unanswered topics so follow-up questions remain accurate.\n\n"
            f"{prior_block}"
            f"[Newly Archived Turn(s)]:\n{history_text}\n"
        )
    else:
        # Standard on-demand summarization mode (used by sidebar "Summarize History" and Agentic tool)
        prompt = f"""You are a helpful assistant. Summarize the following chat history covering multiple files. Highlight the main topics discussed, the types of documents analyzed, and the key insights provided:

    [Cross-File History]:
    {history_text}
    """

    messages_to_send = [
        {"role": "system", "content": "You are a helpful AI summarization assistant."},
        {"role": "user", "content": prompt},
    ]

    mode_label = "rolling memory" if is_rolling else "cross-file"
    logger.info(
        f"Generating {mode_label} summary via '{CHAT_MODEL}' ({len(history_text)} input chars)..."
    )
    try:
        # Reuse the shared OpenRouter client_ai and configured CHAT_MODEL (automatically cost-tracked)
        create_kwargs = {
            "model": CHAT_MODEL,
            "messages": messages_to_send,
            "temperature": 0.2 if is_rolling else 0.3,
        }
        # Disable extended reasoning during automatic rolling compaction to keep latency & cost minimal
        if is_rolling:
            create_kwargs["extra_body"] = {"reasoning": {"enabled": False}}

        completion = client_ai.chat.completions.create(**create_kwargs)
        summary = completion.choices[0].message.content
        logger.info(f"History summary generated successfully ({len(summary or '')} chars).")
        return summary
    except Exception as exc:
        logger.error(f"History summarization failed: {exc}", exc_info=True)
        raise
