# Ask the 20 evaluation questions to both RAG pipelines and save every answer.
#
#   python -m evaluation.run_eval index                 index the two PDFs (documents/)
#   python -m evaluation.run_eval run                   ask all questions to both pipelines
#   python -m evaluation.run_eval run --only hybrid --questions 1-5
#
# Each question is asked on its own, with no earlier conversation: chat_history.json is emptied
# before each question and restored at the end. Nothing in the pipelines is changed. What each
# question cost is tracked as in the app, but written to evaluation/results/cost_eval.csv so the
# evaluation stays separate from normal use.
#
# Saved per pipeline in evaluation/results/answers_<name>.json: the answer, the chunks given to
# the LLM (with the PDF pages they come from), the cost and the time.

import os, time, argparse, logging

# Same models as the app (the pipelines read these from .env and have no defaults)
os.environ.setdefault("CHAT_MODEL", "deepseek/deepseek-v4-flash")
os.environ.setdefault("EMBEDDING_MODEL", "openai/text-embedding-3-small")

from hybrid_rag import cost
from evaluation.common import (ROOT, RESULTS, PAPERS_DIR, DOCS, PIPELINES, answer_key,
                               answers_file, load, save)
from evaluation.locate import locate

logger = logging.getLogger("rag.eval")
cost.COST_LOG = RESULTS / "cost_eval.csv"          # keep evaluation costs out of the app's log
HISTORY = ROOT / "chat_history.json"               # the pipelines' chat history (project folder)


# ---------- Indexing ----------
def index():
    """Put both PDFs in the shared hybrid collection (already indexed files are skipped)."""
    from hybrid_rag.hybrid import initialize_hybrid_index
    for name in DOCS.values():
        with cost.track("ingest_global", name) as qc:
            initialize_hybrid_index(str(PAPERS_DIR / name))
        print(f"{name}: indexed (cost ${qc.total_usd:.6f})")


# ---------- Asking ----------
def pipeline_function(system):
    if system == "hybrid":
        from hybrid_rag.hybrid import run_hybrid_chat_session as f
    else:
        from hybrid_rag.corrective_hybrid import run_corrective_hybrid_rag as f
    return f


def ask(system, question):
    """Same call as the app (execute_with_retry + cost tracking), with an empty chat history."""
    from hybrid_rag.retry import execute_with_retry
    from hybrid_rag.chat_history import clear_all_history
    clear_all_history(filepath=str(HISTORY))
    with cost.track(system, question):
        answer, ctx = execute_with_retry(pipeline_function(system), question)
    return answer, [{"text": c if isinstance(c, str) else str(c), "sim": None} for c in ctx]


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
                    answer, chunks = ask(system, q["question"])
                    error = None
                except Exception as e:                   # keep going; the report shows it
                    logger.exception("%s Q%d failed", system, q["id"])
                    answer, chunks, error = "", [], f"{type(e).__name__}: {e}"
                seconds = time.perf_counter() - started
                qc = cost.last_query()
                for c in chunks:
                    c.update(locate(c["text"], PAPERS_DIR))
                results[q["id"]] = {
                    "id": q["id"], "question": q["question"], "answer": answer, "chunks": chunks,
                    "error": error, "seconds": round(seconds, 2),
                    "cost": None if qc is None or qc.question != q["question"] else {
                        "usd": qc.total_usd, "requests": qc.n_calls,
                        "prompt_tokens": qc.prompt_tokens, "completion_tokens": qc.completion_tokens,
                        "reasoning_tokens": qc.reasoning_tokens, "steps": qc.by_step()},
                }
                save(answers_file(system), [results[i] for i in sorted(results)])
                usd = results[q["id"]]["cost"]["usd"] if results[q["id"]]["cost"] else 0
                print(f"  Q{q['id']:>2}  {seconds:5.1f}s  ${usd:.6f}  {len(chunks)} chunks"
                      f"{'  ERROR ' + error if error else ''}")
    finally:
        # put the chat history back the way it was
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
    logging.getLogger().setLevel(logging.WARNING)          # keep the terminal readable
    parser = argparse.ArgumentParser(description="Run the evaluation questions")
    parser.add_argument("step", choices=["index", "run"])
    parser.add_argument("--only", nargs="*", help="pipelines: hybrid crag_hybrid")
    parser.add_argument("--questions", help="e.g. 1-5,17")
    args = parser.parse_args()
    if args.step == "index":
        index()
    else:
        run(set(args.only or []), parse_ids(args.questions))


if __name__ == "__main__":
    main()
