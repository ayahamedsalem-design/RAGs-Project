"""Optional LangSmith tracing utilities shared across LCEL, LangGraph, and OpenAI-SDK calls.

Automatically activates whenever ``LANGSMITH_TRACING=true`` and ``LANGSMITH_API_KEY`` are set
in ``.env``, and degrades gracefully to transparent no-ops when disabled or uninstalled.
"""

import os
from dotenv import load_dotenv

# Ensure LangSmith environment variables from .env are loaded before LangChain initializes
load_dotenv()

# Optional import: if the `langsmith` package is not installed, every helper below
# degrades cleanly into a zero-overhead no-op so the RAG pipelines still run normally.
try:
    from langsmith import traceable as _ls_traceable
    from langsmith.wrappers import wrap_openai as _ls_wrap_openai
except ImportError:
    _ls_traceable = None
    _ls_wrap_openai = None


def is_tracing_enabled() -> bool:
    """Returns True when LangSmith tracing is enabled and an API key is configured."""
    # Support both modern (LANGSMITH_*) and legacy (LANGCHAIN_*) env var names
    flag = (
        os.getenv("LANGSMITH_TRACING")
        or os.getenv("LANGCHAIN_TRACING_V2")
        or ""
    ).strip().lower()
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    # Both an explicit opt-in flag AND a non-empty API key must be present
    return flag in ("1", "true", "yes", "on") and bool(api_key)


def traceable(*dargs, **dkwargs):
    """Re-exports ``langsmith.traceable`` with a safe no-op fallback if langsmith is absent."""
    # When langsmith is installed, delegate directly to its official @traceable decorator
    if _ls_traceable is not None:
        return _ls_traceable(*dargs, **dkwargs)

    # Fallback mode: handle bare `@traceable` usage without parentheses
    if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
        return dargs[0]

    # Fallback mode: handle parameterized `@traceable(name=..., run_type=...)` usage
    def _decorator(fn):
        return fn

    return _decorator


def wrap_openai_client(client):
    """Wraps an OpenAI SDK client so direct OpenRouter calls appear inside LangSmith traces."""
    # Only wrap when tracing is actively enabled so disabled runs pay zero wrapper overhead
    if _ls_wrap_openai is not None and is_tracing_enabled():
        return _ls_wrap_openai(client)
    return client


def build_trace_config(
    run_name: str,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> dict:
    """Builds a standard LangChain/LangGraph RunnableConfig dict for clean LangSmith filtering."""
    # Passed as `config=...` to `.invoke()` on LCEL chains and LangGraph graphs so runs
    # show up in LangSmith with friendly names, searchable tags, and session/file metadata.
    return {
        "run_name": run_name,
        "tags": list(tags or []),
        "metadata": dict(metadata or {}),
    }
