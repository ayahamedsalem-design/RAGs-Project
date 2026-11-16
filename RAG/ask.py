# Ask a question about all your documents: search the vector store for the most
# relevant chunks (from any file), then have the LLM answer using only those.
#
# Usage:
#   python -m RAG.ask "your question"
#   python -m RAG.ask "your question" --show-prompt   also print what is sent to the LLM
#   python -m RAG.ask "your question" --search        only show the matching chunks (no LLM)
#   python -m RAG.ask --list                          list the files in the store
#
# Chat memory (see memory.py):
#   python -m RAG.ask --chat                          interactive chat with follow-up questions
#   python -m RAG.ask --chat --session thesis         continue (or start) the conversation "thesis"
#   python -m RAG.ask "follow-up" --session thesis    one question inside a saved conversation
#   python -m RAG.ask --sessions                      list saved conversations
#   python -m RAG.ask --show thesis                   print a saved conversation
#   python -m RAG.ask --forget thesis                 delete a saved conversation
# A question without --chat or --session is answered on its own and not saved,
# so test runs stay repeatable.

import sys, logging, argparse

from RAG.rag_common import RAGError, TOP_K, embed_texts, chat, open_store, store_exists, stored_files
import RAG.rag_common as rag_common
from RAGs.logger_config import setup_logging
import RAG.memory as memory

logger = logging.getLogger("rag.ask")

SYSTEM_PROMPT = """You answer questions using ONLY the numbered sources provided.
The sources are excerpts from different documents, which may cover different topics.
Rules:
- Use only the sources that are relevant to the question; ignore the others.
- Cite every fact with its source number, like [Source 2]. Use only these labels,
  never author names or references that appear inside the sources.
- If the sources do not contain the answer, say clearly that the documents do not cover it.
  Do not answer from your own knowledge.
- Be clear and concise."""


def search(col, question, k):
    q = embed_texts([question], is_query=True)[0]
    return [(c["sim"], c["path"], c["loc"], c["text"]) for c in rag_common.search(col, q, k)]


def build_prompt(question, results):
    sources = "\n\n".join(
        f"[Source {i}] (file: {path}, {loc})\n{text}"
        for i, (_, path, loc, text) in enumerate(results, 1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"SOURCES:\n\n{sources}\n\nQUESTION: {question}"},
    ]


def list_files(col):
    files = sorted(stored_files(col).values(), key=lambda f: f["path"])
    if not files:
        print("The store is empty. Add files to RAG/documents and run: python -m RAG.ingest")
        return
    for f in files:
        unfinished = "" if f["complete"] else f"   NOT FINISHED: {f['stored']} saved, run ingest again"
        if f["skipped"]:
            unfinished += f"   ({f['skipped']} skipped: refused by the embedding model)"
        print(f"{f['path']:50} {f['file_type']:5} {f['chunks']:4} chunks   added {f['added']}{unfinished}")
    print(f"\n{len(files)} files, {sum(f['chunks'] for f in files)} chunks")


def answer(col, question, turns, args):
    """Answer one question (a follow-up if turns holds earlier ones), print it, and
    return (answer, sources, standalone question) for the chat memory."""
    logger.info("Question: %s", question)
    standalone = memory.rewrite_question(question, turns)
    if standalone != question:
        print(f"(searching for: {standalone})\n")
    results = search(col, standalone, args.k)
    for i, (sim, path, loc, _) in enumerate(results, 1):
        logger.info("Source %d: %s | %s | similarity %.3f", i, path, loc, sim)

    messages = memory.add_history_to_prompt(build_prompt(standalone, results), turns)
    logger.debug("Prompt:\n%s", "\n".join(f"--- {m['role']} ---\n{m['content']}" for m in messages))
    if args.show_prompt:
        print("=== Prompt sent to the LLM ===\n")
        for m in messages:
            print(f"--- {m['role']} ---\n{m['content']}\n")

    reply = chat(messages)
    logger.info("Answer: %s", reply)
    print("=== Answer ===\n")
    print(reply)

    print("\n=== Sources used ===")
    for i, (sim, path, loc, _) in enumerate(results, 1):
        print(f"[Source {i}] {path} | {loc} | similarity {sim:.3f}")
    return reply, [f"{path} | {loc}" for _, path, loc, _ in results], standalone


def run():
    parser = argparse.ArgumentParser(description="Ask questions about your documents")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--k", type=int, default=TOP_K, help="how many chunks to use")
    parser.add_argument("--search", action="store_true", help="only show matching chunks")
    parser.add_argument("--show-prompt", action="store_true", help="print the prompt sent to the LLM")
    parser.add_argument("--list", action="store_true", help="list files in the store")
    memory.add_memory_args(parser)
    args = parser.parse_args()

    if memory.handle_memory_commands(args):
        return
    if not store_exists():
        rag_common.fail("The vector store doesn't exist yet. First run: python -m RAG.ingest")
    col = open_store()

    if args.list:
        list_files(col)
        return
    if not args.question and not args.chat:
        sys.exit('Usage: python -m RAG.ask "your question"   (or --chat)')
    if col.count() == 0:
        rag_common.fail("The store is empty. Add files to RAG/documents and run: python -m RAG.ingest")

    if args.chat:
        memory.run_chat(lambda q, turns: answer(col, q, turns, args), "naive",
                        args.session or "default")
        return

    question = " ".join(args.question)
    if args.search:
        for i, (sim, path, loc, text) in enumerate(search(col, question, args.k), 1):
            print(f"[{i}] {path} | {loc} | similarity {sim:.3f}")
            print(f"    {text[:300]}...\n")
        return

    if not args.session:
        answer(col, question, [], args)     # no memory: answered on its own, not saved
        return
    mem = memory.open_memory()
    sid = memory.get_session(mem, args.session)
    reply, sources, standalone = answer(col, question, memory.recent_turns(mem, sid), args)
    memory.save_turn(mem, sid, question, standalone, reply, sources, "naive")


def main():
    setup_logging()
    try:
        run()
    except RAGError as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()