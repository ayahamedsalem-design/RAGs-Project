# Ask the 20 evaluation questions to all seven RAG pipelines and save every answer.
#
#   python -m RAG.eval.run_eval index                 index the two PDFs into every pipeline's store
#   python -m RAG.eval.run_eval run                   ask all questions to all pipelines
#   python -m RAG.eval.run_eval run --only crag basic --questions 1-5
#
# Each question is asked on its own, with no earlier conversation:
#   - Ext pipelines: a fresh chat session per question (deleted afterwards)
#   - RAGs/ pipelines: chat_history.json is emptied before each question and restored at the end
# Nothing in the pipelines is changed. What each question cost is tracked as in the app, but
# written to RAG/eval/results/cost_eval.csv so the evaluation stays separate from normal use.
#
# Saved per pipeline in RAG/eval/results/answers_<name>.json: the answer, the chunks given to
# the LLM (with the PDF pages they come from), the corrective status (Ext CRAG), the cost and time.

import os, sys, json, time, shutil, argparse, logging

# Every Chroma client in this process must see the same settings, or Chroma refuses the second
# one ("An instance of Chroma already exists ... with different settings"). RAG/rag_common.py
# turns telemetry off when it is imported, which can happen after a RAGs/ module has already
# opened the folder, so set it here, before anything opens Chroma.
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Same models for every pipeline (the RAGs/ code reads these and has no defaults)
os.environ.setdefault("CHAT_MODEL", "deepseek/deepseek-v4-flash")
os.environ.setdefault("EMBEDDING_MODEL", "openai/text-embedding-3-small")

from RAG import cost
from RAG.eval.common import (RESULTS, PAPERS_DIR, DOCS, PIPELINES, answer_key, answers_file,
                             load, save)
from RAG.eval.locate import locate

logger = logging.getLogger("rag.eval")
cost.COST_LOG = RESULTS / "cost_eval.csv"          # keep evaluation costs out of the app's log
ROOT = RESULTS.parent.parent.parent                # repo root (Aya's chat_history.json lives here)
HISTORY = ROOT / "chat_history.json"


# ---------- Indexing ----------
def index():
    """Put both PDFs in every store: RAG/documents + ingest (Ext), and Aya's three indexers."""
    from RAG.rag_common import DOCS_DIR
    for name in DOCS.values():
        target = DOCS_DIR / name
        if not target.exists():
            shutil.copy2(PAPERS_DIR / name, target)
            print(f"Copied {name} to {DOCS_DIR}")
    print("\n=== Ext pipelines (RAG/ingest) ===")
    from RAG import ingest
    sys.argv = ["ingest"]
    ingest.run()

    print("\n=== RAGs/ pipelines (Basic, Parent-Child, Hybrid) ===")
    from RAGs.basic_RAG import index_basic_rag
    from RAGs.parent_child_RAG import index_parent_child_rag
    from RAGs.hybrid_RAG import initialize_hybrid_index
    for name in DOCS.values():
        with cost.track("ingest_global", name) as qc:
            path = str(PAPERS_DIR / name)
            index_basic_rag(path)
            index_parent_child_rag(path)
            initialize_hybrid_index(path)
        print(f"{name}: indexed (cost ${qc.total_usd:.6f})")


# ---------- Asking ----------
def ext_status(contexts, log):
    """Ext CRAG's corrective status, read from its step log (app_bridge doesn't return it)."""
    if not contexts:
        return "NOT FOUND"
    round2 = [l for l in log if l.startswith("Round 2:")]
    if round2:
        return "CORRECTED" if "sufficient: yes" in round2[-1] else "PARTIAL"
    if any(l.startswith("Corrective search found") for l in log):
        return "PARTIAL"            # a second search ran but found nothing new
    return "CORRECT"


def ask_ext(system, question, qid):
    from RAG.app_bridge import run_naive_rag, run_corrective_rag
    session = f"eval-{system}-q{qid}"
    if system == "naive":
        answer, ctx = run_naive_rag(session, question)
        log, status = [], None
    else:
        answer, ctx, log = run_corrective_rag(session, question)
        status = ext_status(ctx, list(log))
    delete_session(session)
    chunks = [{"text": c["text"], "sim": float(c["sim"]), "via": c.get("via")} for c in ctx]
    return answer, chunks, status, list(log)


def delete_session(name):
    import RAG.memory as memory
    mem = memory.open_memory()
    sid = memory.get_session(mem, name, create=False)
    if sid:
        mem.execute("DELETE FROM turns WHERE session_id = ?", (sid,))
        mem.execute("DELETE FROM sessions WHERE id = ?", (sid,))
        mem.commit()


class DeepSeekInsteadOfOllama:
    """Basic RAG answers with ollama.chat(model="llama3.2:1b") (hard-coded). For the
    "basic_deepseek" run only, this stands in for the ollama module inside RAGs/basic_RAG.py
    and sends the same messages to CHAT_MODEL on OpenRouter, through Aya's own client (so the
    cost hook records it). Her file is not changed; the app still uses llama3.2:1b."""

    def chat(self, model, messages, options=None):
        import RAGs.basic_RAG as basic
        options = options or {}
        r = basic.client_ai.chat.completions.create(
            model=os.environ["CHAT_MODEL"], messages=messages,
            temperature=options.get("temperature", 0.0), top_p=options.get("top_p", 1.0))
        return {"message": {"content": r.choices[0].message.content}}


def aya_function(system):
    if system == "basic":
        from RAGs.basic_RAG import run_basic_RAG as f
    elif system == "basic_deepseek":
        import RAGs.basic_RAG as basic
        basic.ollama = DeepSeekInsteadOfOllama()       # this process only
        f = basic.run_basic_RAG
    elif system == "parent_child":
        from RAGs.parent_child_RAG import run_parent_child_chunking_RAG as f
    elif system == "hybrid":
        from RAGs.hybrid_RAG import run_hybrid_chat_session as f
    elif system == "crag_parent_child":
        from RAGs.corrective_parent_child_RAG import run_corrective_parent_child_rag as f
    else:
        from RAGs.corrective_hybrid_RAG import run_corrective_hybrid_rag as f
    return f


def ask_aya(system, question):
    """Same call as the app (execute_with_retry + cost tracking), with an empty chat history."""
    from RAGs.RAG_service import execute_with_retry
    from RAGs.chat_history import clear_all_history
    clear_all_history(filepath=str(HISTORY))
    with cost.track(system, question):
        answer, ctx = execute_with_retry(aya_function(system), question)
    chunks = [{"text": c if isinstance(c, str) else str(c), "sim": None, "via": None} for c in ctx]
    return answer, chunks, None, []


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
                    if system in ("naive", "crag"):
                        answer, chunks, status, log = ask_ext(system, q["question"], q["id"])
                    else:
                        answer, chunks, status, log = ask_aya(system, q["question"])
                    error = None
                except Exception as e:                   # keep going; the report shows it
                    logger.exception("%s Q%d failed", system, q["id"])
                    answer, chunks, status, log, error = "", [], None, [], f"{type(e).__name__}: {e}"
                seconds = time.perf_counter() - started
                qc = cost.last_query()
                for c in chunks:
                    c.update(locate(c["text"], PAPERS_DIR))
                results[q["id"]] = {
                    "id": q["id"], "question": q["question"], "answer": answer, "chunks": chunks,
                    "status": status, "log": log, "error": error, "seconds": round(seconds, 2),
                    "cost": None if qc is None or qc.question != q["question"] else {
                        "usd": qc.total_usd, "requests": qc.n_calls,
                        "prompt_tokens": qc.prompt_tokens, "completion_tokens": qc.completion_tokens,
                        "reasoning_tokens": qc.reasoning_tokens, "steps": qc.by_step()},
                }
                save(answers_file(system), [results[i] for i in sorted(results)])
                usd = results[q["id"]]["cost"]["usd"] if results[q["id"]]["cost"] else 0
                print(f"  Q{q['id']:>2}  {seconds:5.1f}s  ${usd:.6f}  {len(chunks)} chunks"
                      f"{'  ' + status if status else ''}{'  ERROR ' + error if error else ''}")
    finally:
        # put Aya's chat history back the way it was
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
    from RAGs.logger_config import setup_logging
    setup_logging()
    logging.getLogger().setLevel(logging.WARNING)          # keep the terminal readable
    parser = argparse.ArgumentParser(description="Run the evaluation questions")
    parser.add_argument("step", choices=["index", "run"])
    parser.add_argument("--only", nargs="*", help="pipelines, e.g. naive crag basic")
    parser.add_argument("--questions", help="e.g. 1-5,17")
    args = parser.parse_args()
    if args.step == "index":
        index()
    else:
        run(set(args.only or []), parse_ids(args.questions))


if __name__ == "__main__":
    main()
