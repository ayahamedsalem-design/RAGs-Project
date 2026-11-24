"""Retry helper with exponential backoff for transient LLM/API errors."""

import logging
import time

logger = logging.getLogger(__name__)

# Substrings in exception messages that indicate a retryable, transient API error
TRANSIENT_ERROR_MARKERS = (
    "500",
    "502",
    "503",
    "504",
    "429",
    "timeout",
    "timed out",
    "RateLimit",
    "inference failed",
    "Connection error",
)


def execute_with_retry(rag_func, *args, max_retries: int = 3, delay: float = 2, **kwargs):
    """Calls ``rag_func(*args, **kwargs)``, retrying on transient errors with exponential backoff."""
    current_delay = delay
    for attempt in range(1, max_retries + 1):
        try:
            # Execute the target RAG callable with all forwarded arguments
            return rag_func(*args, **kwargs)
        except Exception as exc:
            err_msg = str(exc)
            # Only retry errors that look temporary (HTTP 5xx, 429 rate limits, timeouts, network drops)
            is_transient = any(marker in err_msg for marker in TRANSIENT_ERROR_MARKERS)
            if is_transient and attempt < max_retries:
                logger.warning(
                    f"Transient error on attempt {attempt}/{max_retries} "
                    f"(retrying in {current_delay}s): {err_msg}"
                )
                time.sleep(current_delay)
                # Exponential backoff (e.g. 2s -> 4s -> 8s) to avoid hammering the API
                current_delay *= 2
                continue
            # Log clearly when retries are exhausted; non-transient errors (e.g. 400/401) fail immediately
            if is_transient:
                logger.error(
                    f"All {max_retries} retry attempts exhausted for {rag_func.__name__}: {err_msg}"
                )
            raise
