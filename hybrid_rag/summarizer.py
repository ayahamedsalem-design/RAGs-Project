"""Conversation history summarization helper powered by the configured OpenRouter chat model."""

import logging
import os

from hybrid_rag.file_processing import initialize_api

logger = logging.getLogger(__name__)

# Shared OpenRouter client (with cost tracking hook) and chat model from .env
client_ai = initialize_api()
CHAT_MODEL = os.getenv("CHAT_MODEL")


def summarize_cross_file_history(history_text: str) -> str:
    """Summarizes one or more chat sessions using the configured ``CHAT_MODEL``."""
    # Wrap the raw conversation transcript in a structured summarization prompt
    prompt = f"""You are a helpful assistant. Summarize the following chat history covering multiple files. Highlight the main topics discussed, the types of documents analyzed, and the key insights provided:

    [Cross-File History]:
    {history_text}
    """

    # Build OpenAI-compatible chat message payload (system role + user transcript)
    messages_to_send = [
        {"role": "system", "content": "You are a helpful AI summarization assistant."},
        {"role": "user", "content": prompt},
    ]

    logger.info(
        f"Generating history summary via '{CHAT_MODEL}' ({len(history_text)} input chars)..."
    )
    try:
        # Reuse the shared OpenRouter client_ai so this call is also captured by cost.py
        completion = client_ai.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages_to_send,
            temperature=0.3,  # Slightly higher than 0.2 for natural prose while staying factual
        )
        summary = completion.choices[0].message.content
        logger.info(f"History summary generated successfully ({len(summary or '')} chars).")
        return summary
    except Exception as exc:
        logger.error(f"History summarization failed: {exc}", exc_info=True)
        raise
