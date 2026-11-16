# Corrective RAG (CRAG): ask a question about your documents, but check the
# search results before answering, and fix the search when they are not good enough.
#
#   1. Retrieve:  get the CANDIDATES closest chunks (more than the 4 used by ask.py)
#   2. Grade:     an LLM "grader" marks each chunk relevant or not, says whether
#                 the relevant ones answer EVERY part of the question, and
#                 suggests new search queries for anything missing
#   3. Correct:   if something is missing, search again with those queries
#                 (and once per document for questions about several documents),
#                 then grade the new chunks
#   4. Answer:    the LLM answers from the relevant chunks only. If no chunk is
#                 relevant, the system says the documents don't cover it,
#                 without asking the LLM to answer at all.
#
# Usage:
#   python -m RAG.ask_crag "your question"
#   python -m RAG.ask_crag "your question" --trace        show every grading decision
#   python -m RAG.ask_crag "your question" --quiet        hide the corrective steps
#   python -m RAG.ask_crag "your question" --show-prompt  print the prompt sent to the LLM
#   python -m RAG.ask_crag --chat                         interactive chat with follow-up questions
#   (--session, --sessions, --show and --forget work as in ask.py; see memory.py)
#
# ask.py (the original, naive RAG) answers the same way as before, so both can be compared
# on the same questions. Follow-ups are rewritten into a stand-alone question first,
# and that question is what gets retrieved and graded.

import sys, re, json, logging, argparse

import numpy as np

from RAG.rag_common import (RAGError, CANDIDATES, MAX_CONTEXT, MIN_SIMILARITY, MAX_EXTRA_QUERIES,
                        EXTRA_PER_QUERY, PER_FILE, MAX_GRADED, GRADER_TEMPERATURE, GRADING,
                        REFERENCE_THRESHOLD, embed_texts, chat, open_store, store_exists,
                        stored_files, get_embedding, search, fail)
from RAG.ask import build_prompt
from RAGs.logger_config import setup_logging
import RAG.memory as memory
from RAG import cost

logger = logging.getLogger("rag.crag")

NOT_COVERED = "The documents do not cover this question. No relevant passages were found."

GRADER_PROMPT = """You check search results for a document question-answering system.
You receive a QUESTION and numbered CANDIDATE excerpts from documents.

1. For each candidate, decide whether it states facts that answer the question, fully or
   partly. Reference lists, bibliography entries, citation titles and passages that only
   mention the same topic without the answer are NOT relevant.
2. Split the question into its separate parts (1 to 4 short parts). A question asking
   about two things, such as "how large are X and Y", has two parts. For each part,
   say whether the relevant candidates actually state the answer to it.
3. Write up to 3 short search queries that could find the unanswered parts in the
   documents, using the documents' likely wording.
4. Say whether the question asks about more than one document (for example
   "each document", "which documents", or a comparison between documents).

Reply with JSON only, no other text, in exactly this form:
{"relevant": [candidate numbers], "parts": [{"part": "...", "answered": true}], "queries": [], "multi_document": false}"""


# The exact shape the grader must reply in (enforced by Ollama for local models)
GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {"type": "array", "items": {"type": "integer"}},
        "parts": {"type": "array", "items": {
            "type": "object",
            "properties": {"part": {"type": "string"}, "answered": {"type": "boolean"}},
            "required": ["part", "answered"]}},
        "queries": {"type": "array", "items": {"type": "string"}},
        "multi_document": {"type": "boolean"},
    },
    "required": ["relevant", "parts", "queries", "multi_document"],
}


# Per-chunk grading (small local models judge one passage at a time far more reliably)
RELEVANCE_PROMPT = """You decide whether ONE passage from a document helps answer a question.
Answer true only if the passage itself states facts that answer the question or part of it.
Answer false if the passage only mentions the same topic without giving the answer, or if it
is a reference list, bibliography or list of citations.
Reply with JSON only: {"relevant": true} or {"relevant": false}"""

RELEVANCE_SCHEMA = {"type": "object", "properties": {"relevant": {"type": "boolean"}},
                    "required": ["relevant"]}

SUFFICIENCY_PROMPT = """You check whether the passages found so far answer a question completely.
1. Split the question into its separate parts (1 to 4 short parts). A question asking
   about two things, such as "how large are X and Y", has two parts.
2. For each part, say whether the passages actually state the answer to it.
3. Write up to 3 short search queries that could find the unanswered parts in the
   documents, using the documents' likely wording.
4. Say whether the question asks about more than one document (for example
   "each document", "which documents", or a comparison between documents).
Reply with JSON only, in exactly this form:
{"parts": [{"part": "...", "answered": true}], "queries": [], "multi_document": false}"""

SUFFICIENCY_SCHEMA = {
    "type": "object",
    "properties": {
        "parts": {"type": "array", "items": {
            "type": "object",
            "properties": {"part": {"type": "string"}, "answered": {"type": "boolean"}},
            "required": ["part", "answered"]}},
        "queries": {"type": "array", "items": {"type": "string"}},
        "multi_document": {"type": "boolean"},
    },
    "required": ["parts", "queries", "multi_document"],
}


def citation_density(text):
    """Citation markers per 100 words. Reference lists score far higher than normal text
    (measured on the test PDFs: reference chunks 13.7 or more, normal text 7.7 or less)."""
    words = max(len(text.split()), 1)
    years = len(re.findall(r"\b(?:19|20)\d{2}[a-z]?\b", text))
    links = len(re.findall(r"doi|https?://|Available (?:online|at)|Accessed", text, re.I))
    pages = len(re.findall(r"\b\d+\s*\(\d+\)|\bpp?\.\s*\d+|\d+\s*[-\u2013]\s*\d+\.", text))
    return (years + 2 * links + pages) / words * 100


def read_json(reply):
    m = re.search(r"\{.*\}", reply or "", re.DOTALL)
    if not m:
        logger.warning("No JSON in the grader's reply: %r", (reply or "")[:300])
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        logger.warning("Grader's JSON could not be read (%s): %r", e, m.group(0)[:300])
        return None


# ---------- Retrieval ----------
def knn(col, qvec, k):
    """Nearest chunks to one query vector, with their ids (so duplicates can be removed)."""
    return [{**c, "via": "question"} for c in search(col, qvec, k)]


def similarity_to(col, qvec, chunk_id):
    """Similarity between the original question and a chunk found by another query."""
    return float(np.dot(qvec, get_embedding(col, chunk_id)))


def knn_per_file(col, qvec, n, skip=lambda c: False):
    """The n closest usable chunks from EACH document, so no document is crowded out.
    Chunks for which skip(chunk) is true (already seen, reference lists) don't count."""
    best = {}
    for c in knn(col, qvec, min(col.count(), 500)):
        group = best.setdefault(c["path"], [])
        if len(group) < n and not skip(c):
            group.append(c)
    logger.debug("Closest %d new chunks per document: %s", n,
                 ", ".join(f"{path}: {len(g)}" for path, g in best.items()))
    return [c for group in best.values() for c in group]


class Log(list):
    """The list of steps shown to the user (also carries the question vector).
    Every step is also written to the log file, including the ones only shown with --trace."""
    qvec = None

    def append(self, line):
        logger.info(line, stacklevel=2)     # name the function that added the step
        super().append(line)


# ---------- Grading ----------
def parse_parts(r, have_relevant):
    """Turn the grader's list of question parts into: complete?, what's missing, new queries.
    Unanswered parts are searched for in their own words first, then the grader's queries."""
    parts = [p for p in r.get("parts", []) if isinstance(p, dict) and str(p.get("part", "")).strip()]
    unanswered = [str(p["part"]).strip() for p in parts if not p.get("answered")]
    queries = []
    for q in unanswered + [q for q in r.get("queries", []) if isinstance(q, str)]:
        q = q.strip()
        if q and q.lower() not in (x.lower() for x in queries):
            queries.append(q)
    return {"sufficient": have_relevant and bool(parts) and not unanswered,
            "missing": "; ".join(unanswered),
            "queries": queries[:MAX_EXTRA_QUERIES],
            "multi_document": bool(r.get("multi_document", False))}


def parse_grade(reply, n):
    """Read the batch grader's JSON; return None if it can't be understood."""
    g = read_json(reply)
    if g is None:
        return None
    relevant = []
    for x in g.get("relevant", []):
        try:
            x = int(x)
        except (TypeError, ValueError):
            continue
        if 1 <= x <= n and x not in relevant:
            relevant.append(x)
    return {"relevant": relevant, **parse_parts(g, bool(relevant))}


def grade(question, cands, round=1):
    """Decide which candidates are relevant and what is still missing.
    Reference lists are removed first without asking the LLM.
    round (1 or 2) only names the grading requests in the cost record."""
    refs = [i for i, c in enumerate(cands, 1) if citation_density(c["text"]) >= REFERENCE_THRESHOLD]
    others = [(i, c) for i, c in enumerate(cands, 1) if i not in refs]
    logger.debug("Grading %d chunks (%s mode); %d reference-list chunks removed first",
                 len(others), GRADING, len(refs))
    if not others:
        g = grade_sufficiency(question, [], round) or {"sufficient": False, "missing": "", "queries": [],
                                                "multi_document": False}
        return {**g, "relevant": [], "refs": refs}
    if GRADING == "per_chunk":
        g = grade_per_chunk(question, others, round)
    else:
        sub = grade_batch(question, [c for _, c in others], round)
        g = {**sub, "relevant": [others[k - 1][0] for k in sub["relevant"]]}
    g["refs"] = refs
    return g


def step_name(step, round):
    return step if round == 1 else f"{step}_round2"     # grade, grade_round2, sufficiency, ...


def grade_per_chunk(question, numbered, round=1):
    """One short yes/no call per chunk, then one call to check what is missing."""
    relevant, unreadable = [], 0
    for i, c in numbered:
        messages = [{"role": "system", "content": RELEVANCE_PROMPT},
                    {"role": "user", "content": f"QUESTION: {question}\n\n"
                                                f"PASSAGE (file: {c['path']}, {c['loc']}):\n{c['text']}"}]
        r = read_json(chat(messages, temperature=GRADER_TEMPERATURE, json_schema=RELEVANCE_SCHEMA,
                           step=step_name("grade", round)))
        if r is None:
            unreadable += 1
            relevant.append(i)          # can't tell: keep it, the answer step can still ignore it
            logger.warning("Chunk %d (%s | %s): grader reply unreadable, kept", i, c["path"], c["loc"])
        elif bool(r.get("relevant")):
            relevant.append(i)
        logger.debug("Chunk %d of %d (%s | %s): %s", i, len(numbered), c["path"], c["loc"],
                     "relevant" if i in relevant else "not relevant")
    kept = [c for i, c in numbered if i in relevant]
    g = grade_sufficiency(question, kept, round)
    if g is None:
        logger.warning("Completeness check unreadable; treating the kept chunks as enough")
        g = {"sufficient": bool(kept), "missing": "", "queries": [], "multi_document": False}
    return {**g, "relevant": relevant, "unreadable": unreadable}


def grade_sufficiency(question, kept, round=1):
    if kept:
        listing = "\n\n".join(f"[Passage {k}] (file: {c['path']}, {c['loc']})\n{c['text']}"
                               for k, c in enumerate(kept, 1))
    else:
        listing = "(none of the search results were relevant)"
    messages = [{"role": "system", "content": SUFFICIENCY_PROMPT},
                {"role": "user", "content": f"QUESTION: {question}\n\nPASSAGES:\n\n{listing}"}]
    r = read_json(chat(messages, temperature=GRADER_TEMPERATURE, json_schema=SUFFICIENCY_SCHEMA,
                       step=step_name("sufficiency", round)))
    if r is None:
        return None
    g = parse_parts(r, bool(kept))
    logger.debug("Completeness check on %d passages: sufficient=%s, missing=%r, multi_document=%s",
                 len(kept), g["sufficient"], g["missing"], g["multi_document"])
    return g


def grade_batch(question, cands, round=1):
    """All candidates in one grading call (used for online models)."""
    listing = "\n\n".join(f"[Candidate {i}] (file: {c['path']}, {c['loc']})\n{c['text']}"
                          for i, c in enumerate(cands, 1))
    messages = [{"role": "system", "content": GRADER_PROMPT},
                {"role": "user", "content": f"QUESTION: {question}\n\nCANDIDATES:\n\n{listing}"}]
    g = parse_grade(chat(messages, temperature=GRADER_TEMPERATURE, json_schema=GRADE_SCHEMA,
                         step=step_name("grade", round)), len(cands))
    if g is None:
        # Grader reply unreadable: fall back to plain RAG behaviour (top 4 by similarity)
        logger.warning("Batch grader reply unreadable; falling back to the top 4 chunks")
        return {"relevant": list(range(1, min(4, len(cands)) + 1)), "sufficient": True,
                "missing": "", "queries": [], "multi_document": False, "failed": True}
    return g


def log_decisions(log, cands, g):
    if g.get("refs"):
        log.append(f"Removed {len(g['refs'])} reference-list chunks before grading")
    if g.get("unreadable"):
        log.append(f"{g['unreadable']} grader replies could not be read (those chunks were kept)")
    for i, c in enumerate(cands, 1):
        verdict = ("drop: reference list" if i in g.get("refs", []) else
                   "KEEP" if i in g["relevant"] else "drop")
        log.append(f"   [{verdict}] {c['path']} | {c['loc']} | similarity {c['sim']:.3f}")


# ---------- The corrective pipeline ----------
def corrective_retrieve(col, question, log):
    n_files = len(stored_files(col))
    qvec = embed_texts([question], is_query=True)[0]
    log.qvec = qvec

    # 1. Retrieve wider than plain RAG, and drop chunks that are clearly unrelated
    cands = [c for c in knn(col, qvec, CANDIDATES) if c["sim"] >= MIN_SIMILARITY]
    log.append(f"Retrieved {len(cands)} candidates (similarity >= {MIN_SIMILARITY})")
    if not cands:
        return [], "NOT FOUND", ""

    # 2. Grade them
    g = grade(question, cands)
    if g.get("failed"):
        log.append("Grader reply could not be read; using the top 4 chunks (plain RAG)")
    relevant = [cands[i - 1] for i in g["relevant"]]
    log.append(f"Round 1: {len(relevant)} of {len(cands)} candidates relevant, "
               f"sufficient: {'yes' if g['sufficient'] else 'no'}")
    log_decisions(log, cands, g)

    # A question about several documents is only answered once every document was checked
    files_covered = {c["path"] for c in relevant}
    needs_all_docs = g["multi_document"] and len(files_covered) < n_files
    if needs_all_docs:
        log.append(f"Question covers several documents, but only {len(files_covered)} of "
                   f"{n_files} were found: searching each document")

    if g["sufficient"] and relevant and not needs_all_docs:
        return relevant, "CORRECT", g["missing"]

    # 3. Corrective retrieval: new queries from the grader, plus one search per document
    queries = g["queries"] or [question]
    if g["missing"]:
        log.append(f"Missing: {g['missing']}")
    log.append("New search queries: " + "; ".join(f'"{q}"' for q in queries))

    seen = {c["id"] for c in cands}
    new = []
    qvecs = embed_texts(queries, is_query=True, step="embed_extra")
    for qi, v in enumerate(qvecs, 1):
        for c in knn(col, v, EXTRA_PER_QUERY):
            if (c["id"] not in seen and c["sim"] >= MIN_SIMILARITY
                    and citation_density(c["text"]) < REFERENCE_THRESHOLD):
                c["via"] = f"new query {qi}"
                seen.add(c["id"]); new.append(c)
    if g["multi_document"]:
        # the closest chunks from each document, for the question and for every new query
        usable_seen = lambda c: c["id"] in seen or citation_density(c["text"]) >= REFERENCE_THRESHOLD
        for v in [qvec, *qvecs]:
            for c in knn_per_file(col, v, PER_FILE, skip=usable_seen):
                if c["id"] not in seen:
                    c["via"] = "per-document search"
                    seen.add(c["id"]); new.append(c)
    log.append(f"Corrective search found {len(new)} new candidates")
    if len(relevant + new) > MAX_GRADED:
        logger.warning("Only the first %d of %d candidates can be graded (MAX_GRADED)",
                       MAX_GRADED, len(relevant + new))
    if not new:
        return relevant, ("PARTIAL" if relevant else "NOT FOUND"), g["missing"]

    # Grade the kept chunks and the new ones together
    pool = (relevant + new)[:MAX_GRADED]
    g2 = grade(question, pool, round=2)
    if g2.get("failed"):
        log.append("Grader reply could not be read in round 2; using the top 4 chunks")
    relevant2 = [pool[i - 1] for i in g2["relevant"]]
    log.append(f"Round 2: {len(relevant2)} of {len(pool)} candidates relevant, "
               f"sufficient: {'yes' if g2['sufficient'] else 'no'}")
    log_decisions(log, pool, g2)

    if not relevant2:
        return [], "NOT FOUND", g2["missing"]
    return relevant2, ("CORRECTED" if g2["sufficient"] else "PARTIAL"), g2["missing"]


def answer(col, question, turns, args):
    """Answer one question (a follow-up if turns holds earlier ones), print it and what it
    cost, and return (answer, sources, standalone question, cost in USD) for the chat memory."""
    with cost.track("crag", question) as qc:
        reply, sources, standalone = _answer(col, question, turns, args)
    print("\n=== Cost ===\n" + qc.summary())
    return reply, sources, standalone, qc.total_usd


def _answer(col, question, turns, args):
    logger.info("Question: %s", question)
    standalone = memory.rewrite_question(question, turns)
    if standalone != question:
        print(f"(searching for: {standalone})\n")

    log = Log()
    chunks, status, missing = corrective_retrieve(col, standalone, log)
    logger.info("Corrective step: %s", status)

    print(f"=== Corrective step: {status} ===")
    if not args.quiet:
        for line in log:
            if args.trace or not line.startswith("   ["):
                print(line)

    if not chunks:
        print("\n=== Answer ===\n")
        print(NOT_COVERED)
        return NOT_COVERED, [], standalone

    # 4. Answer from the relevant chunks only, most similar to the question first.
    # Chunks found by a new query get their similarity to the ORIGINAL question,
    # so the numbers stay comparable with ask.py.
    for c in chunks:
        if c["via"] != "question":
            c["sim"] = similarity_to(col, log.qvec, c["id"])
    chunks = sorted(chunks, key=lambda c: -c["sim"])[:MAX_CONTEXT]
    for i, c in enumerate(chunks, 1):
        logger.info("Source %d: %s | %s | similarity %.3f | via %s",
                    i, c["path"], c["loc"], c["sim"], c["via"])
    results = [(c["sim"], c["path"], c["loc"], c["text"]) for c in chunks]
    messages = build_prompt(standalone, results)
    if status == "PARTIAL":
        gap = f" Likely missing: {missing}." if missing else ""
        messages[1]["content"] += ("\n\nNOTE: the search could not find everything." + gap +
                                   " Answer what the sources support and say clearly "
                                   "which parts the documents do not cover.")
    memory.add_history_to_prompt(messages, turns)
    logger.debug("Prompt:\n%s", "\n".join(f"--- {m['role']} ---\n{m['content']}" for m in messages))
    if args.show_prompt:
        print("\n=== Prompt sent to the LLM ===\n")
        for m in messages:
            print(f"--- {m['role']} ---\n{m['content']}\n")

    reply = chat(messages)
    logger.info("Answer: %s", reply)
    print("\n=== Answer ===\n")
    print(reply)

    print("\n=== Sources used ===")
    for i, c in enumerate(chunks, 1):
        found = "" if c["via"] == "question" else f" | found by {c['via']}"
        print(f"[Source {i}] {c['path']} | {c['loc']} | similarity {c['sim']:.3f}{found}")
    return reply, [f"{c['path']} | {c['loc']}" for c in chunks], standalone


def run():
    parser = argparse.ArgumentParser(description="Ask questions with Corrective RAG")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--trace", action="store_true", help="show every grading decision")
    parser.add_argument("--quiet", action="store_true", help="don't print the corrective steps")
    parser.add_argument("--show-prompt", action="store_true", help="print the prompt sent to the LLM")
    memory.add_memory_args(parser)
    args = parser.parse_args()

    if memory.handle_memory_commands(args):
        return
    if not store_exists():
        fail("The vector store doesn't exist yet. First run: python -m RAG.ingest")
    if not args.question and not args.chat:
        sys.exit('Usage: python -m RAG.ask_crag "your question"   (or --chat)')
    col = open_store()

    if args.chat:
        memory.run_chat(lambda q, turns: answer(col, q, turns, args), "crag",
                        args.session or "default")
        return

    question = " ".join(args.question)
    if not args.session:
        answer(col, question, [], args)     # no memory: answered on its own, not saved
        return
    mem = memory.open_memory()
    sid = memory.get_session(mem, args.session)
    reply, sources, standalone, cost_usd = answer(col, question, memory.recent_turns(mem, sid), args)
    memory.save_turn(mem, sid, question, standalone, reply, sources, "crag", cost_usd)


def main():
    setup_logging()
    try:
        run()
    except RAGError as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()