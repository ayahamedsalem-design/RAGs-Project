"""Logs viewer page for inspecting application and error logs inside Streamlit.

Reads rotating log files written by ``hybrid_rag/logging_setup.py`` (``logs/app.log``
and ``logs/errors.log``) and provides level filtering, keyword search, tailing, and download.
"""

import re
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"

# Matches standard log entries like: "2026-11-24 01:05:12 [ERROR] Global_Hybrid_RAG (...): ..."
LOG_ENTRY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\[(DEBUG|INFO|WARNING|ERROR|CRITICAL)\]")

st.set_page_config(page_title="System Logs", page_icon="📜", layout="wide")
st.title("System Logs")
st.caption("Inspect and filter application logs (`logs/app.log`) and error logs (`logs/errors.log`).")


def parse_log_records(raw_text: str) -> list[dict]:
    """Groups multiline stack traces with their parent log entry and extracts log level."""
    records: list[dict] = []
    for line in raw_text.splitlines():
        match = LOG_ENTRY_RE.match(line)
        if match:
            records.append({"level": match.group(1), "text": line})
        elif records:
            # Continuation line (e.g., Python traceback) attaches to the preceding entry
            records[-1]["text"] += "\n" + line
        else:
            records.append({"level": "INFO", "text": line})
    return records


# ---------- Controls ----------
col_file, col_level, col_limit, col_refresh = st.columns([2, 2, 1.5, 1])

with col_file:
    selected_file_name = st.selectbox(
        "Log file:",
        options=["app.log", "errors.log"],
        index=0,
    )

with col_level:
    selected_levels = st.multiselect(
        "Filter by level:",
        options=["INFO", "WARNING", "ERROR", "CRITICAL", "DEBUG"],
        default=["INFO", "WARNING", "ERROR", "CRITICAL"],
    )

with col_limit:
    max_entries = st.selectbox(
        "Show latest:",
        options=[50, 150, 500, 1500, "All"],
        index=1,
    )

with col_refresh:
    st.write("")  # Vertical spacing to align with selectboxes
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

search_query = st.text_input("Search logs (keyword, filename, model, or error message):", "")

log_path = LOG_DIR / selected_file_name
if not log_path.exists():
    st.info(f"Log file `{log_path.relative_to(PROJECT_ROOT).as_posix()}` does not exist yet.")
    st.stop()

raw_content = log_path.read_text(encoding="utf-8", errors="replace")
all_records = parse_log_records(raw_content)

# Summary metrics across the full selected log file
error_count = sum(1 for r in all_records if r["level"] in ("ERROR", "CRITICAL"))
warning_count = sum(1 for r in all_records if r["level"] == "WARNING")
info_count = sum(1 for r in all_records if r["level"] == "INFO")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Entries", f"{len(all_records):,}")
m2.metric("Errors / Critical", f"{error_count:,}")
m3.metric("Warnings", f"{warning_count:,}")
m4.metric("Info", f"{info_count:,}")

# Apply level and keyword filters
filtered = all_records
if selected_levels:
    allowed_levels = set(selected_levels)
    filtered = [r for r in filtered if r["level"] in allowed_levels]

if search_query.strip():
    needle = search_query.strip().lower()
    filtered = [r for r in filtered if needle in r["text"].lower()]

# Keep the most recent N entries (newest first toggle)
newest_first = st.checkbox("Show newest entries first", value=True)
if isinstance(max_entries, int):
    filtered = filtered[-max_entries:]
if newest_first:
    filtered = list(reversed(filtered))

st.caption(f"Showing **{len(filtered):,}** matching log entries from `logs/{selected_file_name}`.")

if filtered:
    rendered_text = "\n".join(r["text"] for r in filtered)
    st.code(rendered_text, language="log")
    st.download_button(
        label=f"⬇️ Download {selected_file_name}",
        data=raw_content,
        file_name=selected_file_name,
        mime="text/plain",
    )
else:
    st.warning("No log entries match the current filters.")
