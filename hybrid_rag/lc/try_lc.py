# Try the LangChain pipelines (and the originals) from the terminal.
#
#   python -m hybrid_rag.lc.try_lc "question"                         # Hybrid RAG (LangChain)
#   python -m hybrid_rag.lc.try_lc "question" -p crag_lc              # Hybrid Corrective RAG (LangChain)
#   python -m hybrid_rag.lc.try_lc "question" -p all                  # all four pipelines, side by side
#   python -m hybrid_rag.lc.try_lc "question" --retrieval-only        # LangChain retrieval == original?
#   python -m hybrid_rag.lc.try_lc "question" --files "a.pdf" "b.pdf" # search only these files
#   python -m hybrid_rag.lc.try_lc "question" --keep-history          # keep the turn (for follow-ups)
#
# Each run prints the answer, the chunks used (source and location) and what it cost, and the
# cost is saved to logs/cost_log.csv under the pipeline's name (hybrid_lc, crag_hybrid_lc, ...).
# Questions go to their own chat session ("terminal_test" unless --session is given), so the
# app's chats never feed into them. chat_history.json is put back as it was after the run,
# unless --keep-history is given; with -p all, every pipeline starts from the same history so
# the comparison is fair.

import argparse
import sys
from pathlib import Path

from hybrid_rag import cost
from hybrid_rag.hybrid import retrieve_hybrid_context, run_hybrid_chat_session
from hybrid_rag.hybrid_corrective import run_hybrid_corrective_rag
from hybrid_rag.lc.components import format_context, hybrid_search
from hybrid_rag.lc.hybrid_corrective_lc import run_hybrid_corrective_lc
from hybrid_rag.lc.hybrid_lc import run_hybrid_lc

HISTORY = Path("chat_history.json")      # the path chat_history.py uses (relative to the folder you run from)

# -p name: (function, name in cost_log.csv, title)
PIPELINES = {
    "hybrid_lc": (run_hybrid_lc, "hybrid_lc", "Hybrid RAG (LangChain)"),
    "crag_lc": (run_hybrid_corrective_lc, "crag_hybrid_lc", "Hybrid Corrective RAG (LangChain)"),
    "hybrid": (run_hybrid_chat_session, "hybrid", "Hybrid RAG (original)"),
    "crag": (run_hybrid_corrective_rag, "crag_hybrid", "Hybrid Corrective RAG (original)"),
}


def label(chunk):
    """'[Source: file | location ...]' from a labelled chunk."""
    return chunk.split("]:")[0] + "]"


def run(key, question, files, session):
    fn, system, title = PIPELINES[key]
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    with cost.track(system, question) as qc:
        answer, chunks = fn(question, selected_files=files, session_id=session)
    print(f"\n--- ANSWER ---\n{answer}\n\n--- CHUNKS ({len(chunks)}) ---")
    for c in chunks:
        print("  " + label(c))
    print(f"\n--- COST ---\n{qc.summary()}")


def compare_retrieval(question, files, top_k=6):
    original = retrieve_hybrid_context(question, top_k=top_k, selected_files=files)[1]
    lc = format_context(hybrid_search(question, top_k, files))[1]
    print(f"\n--- ORIGINAL ({len(original)}) ---")
    for c in original:
        print("  " + label(c))
    print(f"\n--- LANGCHAIN ({len(lc)}) ---")
    for c in lc:
        print("  " + label(c))
    print(f"\nSame chunks in the same order: {original == lc}")
    return original == lc


def main():
    ap = argparse.ArgumentParser(description="Try the RAG pipelines from the terminal.")
    ap.add_argument("question")
    ap.add_argument("-p", "--pipeline", default="hybrid_lc", choices=[*PIPELINES, "all"])
    ap.add_argument("--files", nargs="+", help="only search these files (names as in the app)")
    ap.add_argument("--session", default="terminal_test",
                    help="chat session the turns belong to (default: terminal_test)")
    ap.add_argument("--keep-history", action="store_true",
                    help="keep the turn(s) in chat_history.json (needed for follow-up questions)")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="no answer: check the LangChain retrieval gives the original's chunks")
    args = ap.parse_args()

    if args.retrieval_only:
        sys.exit(0 if compare_retrieval(args.question, args.files) else 1)

    saved = HISTORY.read_bytes() if HISTORY.exists() else None

    def restore():
        if saved is None:
            HISTORY.unlink(missing_ok=True)
        else:
            HISTORY.write_bytes(saved)

    keys = list(PIPELINES) if args.pipeline == "all" else [args.pipeline]
    try:
        for i, key in enumerate(keys):
            if i:
                restore()                # every pipeline starts from the same history
            run(key, args.question, args.files, args.session)
    finally:
        if not args.keep_history:
            restore()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")     # answers can contain non-ASCII characters
    main()
