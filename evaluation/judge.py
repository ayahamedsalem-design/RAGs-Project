# Score every saved answer against the answer key with an LLM judge.
#
#   python -m evaluation.judge                  judge all pipelines that have answers
#   python -m evaluation.judge --only hybrid    one pipeline (re-judges it)
#
# The judge sees: the question, the expected answer split into parts (or "unanswerable"),
# the chunks the pipeline gave its LLM (numbered, with the PDF pages they come from), and the
# pipeline's answer. It returns JSON: a verdict per expected part, a 0-2 correctness score,
# faithfulness (is every claim supported by the chunks?), citations, and refusals.
# Retrieval (hit rate, coverage, rank) is not judged here: it is computed from the chunk pages
# in report.py. The verdicts are saved in results/judged_<name>.json, checked by hand
# afterwards (corrections go in results/overrides.json), then turned into the report.

import os, json, argparse, logging
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("CHAT_MODEL", "deepseek/deepseek-v4-flash")
JUDGE_MODEL = os.environ["CHAT_MODEL"]             # same model as the pipelines, with reasoning on

from hybrid_rag import cost
from hybrid_rag.pdf_processing import initialize_api   # OpenRouter client with the cost hook
from evaluation.common import RESULTS, PIPELINES, answer_key, answers_file, judged_file, load, save

logger = logging.getLogger("rag.eval.judge")
cost.COST_LOG = RESULTS / "cost_eval.csv"

JUDGE_PROMPT = """You grade the answer of a document question-answering system against an answer key.
You get the QUESTION, the EXPECTED ANSWER split into numbered parts (or a note that the documents
cannot answer it), the CHUNKS the system gave its language model (numbered in the order given,
with the file and PDF pages each comes from), and the system's ANSWER.

Page numbers: chunk pages are PDF pages. The battery paper (HummesETAL...) also prints journal page
numbers, which are the PDF page + 93 (journal page 97 = PDF page 4); the cultured meat advice prints
"Pagina X van 33", which equals PDF page X. A citation using either numbering is correct when it
points to the chunk's page. A citation without a page number is fine if the file is right.

Judge strictly but fairly. Wording may differ; meaning and numbers must match. Details marked
"optional detail" in the expected answer are not required for a part to be correct.

Return ONLY a JSON object with these keys:
- "parts": for each expected part, {"part": <number>, "verdict": "correct" | "partial" | "missing" | "wrong"}.
  Empty list if the question is unanswerable.
- "correct": 2 if every part is correct, 1 if at least one part is correct or partially correct but
  not all, 0 if none is. Use null if the question is unanswerable.
- "faithful": true if every factual claim in the answer is supported by the CHUNKS (not by outside
  knowledge); false otherwise. A statement that the documents don't cover something counts as
  supported when the chunks indeed don't contain it.
- "unsupported": short quote of the first unsupported claim, or "".
- "citations": "ok" if the answer cites its sources and every citation points to a chunk that
  contains the fact (chunk numbers like [Source 2] count in the order given; file/page citations
  must match the chunk's file and pages); "wrong" if any citation points to the wrong chunk, file
  or page; "missing" if facts are stated without any citation; "n/a" if the answer states no facts
  (e.g. a refusal). Only factual claims need citations: sentences saying the documents do not
  contain something need none.
- "refused": true if the answer says the documents do not contain the answer (fully or for the
  whole question) instead of answering it; false otherwise.
- "invented": for unanswerable questions only: true if the answer gives a specific answer that the
  documents do not support (e.g. a price or a country); false otherwise; null for answerable ones.
- "false_refusal": for answerable questions: "full" if the answer says the information is not in
  the documents for the whole question, "partial" if it says so for one part that the answer key
  does contain, "none" otherwise; null for unanswerable questions.
- "note": at most 20 words explaining the main issue, or "" if none."""


def format_chunks(chunks):
    if not chunks:
        return "(no chunks were given to the language model)"
    out = []
    for i, c in enumerate(chunks, 1):
        where = f"{c['doc']} PDF, page(s) {', '.join(map(str, c['pages']))}" if c["doc"] else "location not found"
        out.append(f"[Chunk {i}] ({where})\n{c['text'][:3000]}")
    return "\n\n".join(out)


def expected(q, docs):
    if not q["parts"]:
        return f"UNANSWERABLE from these documents. {q.get('unanswerable_note', '')}"
    return "\n".join(f"Part {i}: {p['answer']} ({docs[p['doc']]}, page(s) {', '.join(map(str, p['pages']))})"
                     for i, p in enumerate(q["parts"], 1))


_client = None


def client():
    global _client
    if _client is None:
        _client = initialize_api()
    return _client


def judge_one(system, q, r, docs):
    user = (f"QUESTION: {q['question']}\n\nEXPECTED ANSWER:\n{expected(q, docs)}\n\n"
            f"CHUNKS:\n{format_chunks(r['chunks'])}\n\nANSWER:\n{r['answer'] or '(no answer: the pipeline failed)'}")
    with cost.track("judge", f"{system} Q{q['id']}"):
        r = client().chat.completions.create(
            model=JUDGE_MODEL, temperature=0, extra_body={"reasoning": {"enabled": True}},
            messages=[{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}])
    reply = r.choices[0].message.content
    verdict = parse(reply)
    if verdict is None:
        logger.warning("%s Q%d: judge reply unreadable: %r", system, q["id"], reply[:300])
        verdict = {"error": "unreadable judge reply", "raw": reply[:2000]}
    verdict["id"] = q["id"]
    return verdict


def parse(reply):
    import re
    m = re.search(r"\{.*\}", reply or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main():
    from hybrid_rag.logging_setup import setup_logging
    setup_logging()
    logging.getLogger().setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Judge the evaluation answers")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    key = answer_key()
    questions = {q["id"]: q for q in key["questions"]}
    for system, label, _ in PIPELINES:
        if args.only and system not in args.only:
            continue
        answers = load(answers_file(system))
        if not answers:
            continue
        with ThreadPoolExecutor(max_workers=8) as pool:
            verdicts = list(pool.map(lambda r: judge_one(system, questions[r["id"]], r, key["documents"]),
                                     answers))
        save(judged_file(system), verdicts)
        scored = [v for v in verdicts if "error" not in v]
        print(f"{label}: {len(scored)}/{len(verdicts)} judged; points "
              f"{sum(v.get('correct') or 0 for v in scored)}")


if __name__ == "__main__":
    main()
