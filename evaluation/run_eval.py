# Ask the 20 evaluation questions to all seven RAG pipelines and save every answer.
#
#   python -m evaluation.run_eval index                 index the two PDFs (documents/) into all stores
#   python -m evaluation.run_eval run                   ask all questions to all seven pipelines
#   python -m evaluation.run_eval run --only hybrid crag --questions 1-5
#
# Each question is asked on its own, with no earlier conversation: chat_history.json (shared by
# all 7 pipelines) is emptied before each question and restored at the end. What each question
# cost is tracked as in the app, but written to evaluation/results/cost_eval.csv so the
# evaluation stays separate from normal use.
#
# Saved per pipeline in evaluation/results/answers_<name>.json: the answer, the chunks given to
# the LLM (with the PDF pages they come from), the corrective status (Ext CRAG), cost and time.

import os
import time
import argparse
import logging

# Set before any ChromaDB PersistentClient opens ./chroma_db_global
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# Same models as the app (the pipelines read these from .env and have no defaults)
os.environ.setdefault("CHAT_MODEL", "deepseek/deepseek-v4-flash")
os.environ.setdefault("EMBEDDING_MODEL", "openai/text-embedding-3-small")

from hybrid_rag import cost
from evaluation.common import (
    ROOT,
    RESULTS,
    PAPERS_DIR,
    DOCS,
    PIPELINES,
    answer_key,
    answers_file,
    load,
    save,
)
from evaluation.locate import locate

logger = logging.getLogger("rag.eval")
cost.COST_LOG = RESULTS / "cost_eval.csv"          # keep evaluation costs out of the app's log
HISTORY = ROOT / "chat_history.json"               # unified chat history shared by all 7 pipelines


# ---------- Indexing ----------
def index():
    """Put both PDFs in all 3 ChromaDB store groups (already indexed files are skipped)."""
    from other_rags.registry import index_file_all_stores

    for name in DOCS.values():
        with cost.track("ingest_global", name) as qc:
            index_file_all_stores(str(PAPERS_DIR / name))
        print(f"{name}: indexed across all stores (cost ${qc.total_usd:.6f})")


# ---------- Asking ----------
def ext_status(contexts, log):
    """Ext CRAG's corrective status, derived from its step log."""
    if not contexts:
        return "NOT FOUND"
    round2 = [l for l in log if l.startswith("Round 2:")]
    if round2:
        return "CORRECTED" if "sufficient: yes" in round2[-1] else "PARTIAL"
    if any(l.startswith("Corrective search found") for l in log):
        return "PARTIAL"
    return "CORRECT"


def normalize_chunks(ctx):
    chunks = []
    for c in ctx:
        if isinstance(c, dict):
            sim = c.get("sim")
            chunks.append(
                {
                    "text": c.get("text") or str(c),
                    "sim": float(sim) if sim is not None else None,
                    "via": c.get("via"),
                }
            )
        else:
            chunks.append({"text": c if isinstance(c, str) else str(c), "sim": None})
    return chunks


def ask(system, question):
    """Same call as the app (execute_with_retry + cost tracking), with an empty chat history."""
    from hybrid_rag.retry import execute_with_retry
    from hybrid_rag.chat_history import clear_all_history
    from other_rags.registry import PIPELINE_BY_KEY

    clear_all_history(filepath=str(HISTORY))
    status, log_lines = None, []

    with cost.track(system, question):
        if system == "crag":
            from ext_rag.app_bridge import run_ext_corrective_rag_with_log

            answer, ctx, log = execute_with_retry(run_ext_corrective_rag_with_log, question)
            log_lines = list(log)
            status = ext_status(ctx, log_lines)
        else:
            answer, ctx = execute_with_retry(PIPELINE_BY_KEY[system], question)

    return answer, normalize_chunks(ctx), status, log_lines


def run(only, question_ids):
    key = answer_key()
    backup = HISTORY.read_bytes() if HISTORY.exists() else None
    try:
        for system, label, _ in PIPELINES:
            if only and system not in only:
                continue
            results = {r["id"]: r for r in load(answers_file(system), [])}
            print(f"\n=== {label} ===")
            for q in key["questions"]:
                if question_ids and q["id"] not in question_ids:
                    continue
                started = time.perf_counter()
                try:
                    answer, chunks, status, log_lines = ask(system, q["question"])
                    error = None
                except Exception as e:                   # keep going; the report shows it
                    logger.exception("%s Q%d failed", system, q["id"])
                    answer, chunks, status, log_lines, error = "", [], None, [], f"{type(e).__name__}: {e}"
                seconds = time.perf_counter() - started
                qc = cost.last_query()
                for c in chunks:
                    c.update(locate(c["text"], PAPERS_DIR))
                record = {
                    "id": q["id"],
                    "question": q["question"],
                    "answer": answer,
                    "chunks": chunks,
                    "error": error,
                    "seconds": round(seconds, 2),
                    "cost": None if qc is None or qc.question != q["question"] else {
                        "usd": qc.total_usd,
                        "requests": qc.n_calls,
                        "prompt_tokens": qc.prompt_tokens,
                        "completion_tokens": qc.completion_tokens,
                        "reasoning_tokens": qc.reasoning_tokens,
                        "steps": qc.by_step(),
                    },
                }
                if status is not None:
                    record["status"] = status
                if log_lines:
                    record["log"] = log_lines
                results[q["id"]] = record
                save(answers_file(system), [results[i] for i in sorted(results)])
                usd = results[q["id"]]["cost"]["usd"] if results[q["id"]]["cost"] else 0
                status_tag = f"  [{status}]" if status else ""
                print(
                    f"  Q{q['id']:>2}  {seconds:5.1f}s  ${usd:.6f}  {len(chunks)} chunks"
                    f"{status_tag}{'  ERROR ' + error if error else ''}"
                )
    finally:
        if backup is None:
            HISTORY.unlink(missing_ok=True)
        else:
            HISTORY.write_bytes(backup)


def parse_ids(text):
    ids = set()
    for part in (text or "").split(","):
        if "-" in part:
            a, b = part.split("-")
            ids |= set(range(int(a), int(b) + 1))
        elif part.strip():
            ids.add(int(part))
    return ids


def main():
    from hybrid_rag.logging_setup import setup_logging

    setup_logging()
    logging.getLogger().setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Run the evaluation questions across all 7 pipelines")
    parser.add_argument("step", choices=["index", "run"])
    parser.add_argument(
        "--only",
        nargs="*",
        help="pipelines: hybrid crag_hybrid basic parent_child crag_parent_child naive crag",
    )
    parser.add_argument("--questions", help="e.g. 1-5,17")
    args = parser.parse_args()
    if args.step == "index":
        index()
    else:
        run(set(args.only or []), parse_ids(args.questions))


if __name__ == "__main__":
    main()
