# Turn the saved answers and judgements into the evaluation report (one Word document,
# a full set of tables for each pipeline, in the layout of the earlier
# Corrective RAG evaluation).
#
#   python -m evaluation.report [--template PATH] [--out PATH]
#
# Retrieval metrics come from the PDF pages of the chunks each pipeline gave its LLM
# (results/answers_*.json, located with locate.py). Answer metrics come from the LLM judge
# (results/judged_*.json) with my corrections from results/overrides.json applied on top.

import json, argparse
from collections import defaultdict
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Cm, RGBColor

from evaluation.locate import words, strip_header
from evaluation.common import (RESULTS, PIPELINES, LABEL, DOCS, answer_key, answers_file,
                             judged_file, load)

OVERRIDES = RESULTS / "overrides.json"
TYPE_ORDER = ["Single fact", "Definition", "Numbers", "Paraphrase", "Acronyms", "Table",
              "Multi-part", "Cross-document", "Unanswerable"]
TYPE_GROUPS = [("Single fact / definition", {"Single fact", "Definition"}),
               ("Numbers", {"Numbers"}), ("Paraphrased wording", {"Paraphrase"}),
               ("Acronyms / tables", {"Acronyms", "Table"}), ("Multi-part", {"Multi-part"}),
               ("Cross-document", {"Cross-document"}), ("Unanswerable", {"Unanswerable"})]

# What each pipeline does, for its setup table (all read from the code)
SETUP = {
    "hybrid": [
        ("Retrieval", "Per sub-question: vector search + BM25 keyword search, merged with Reciprocal Rank Fusion; top 6 chunks"),
        ("Chunks given to the LLM", "6 chunks labelled [Source: file | Page: n]"),
        ("Answer LLM", "CHAT_MODEL (provider default reasoning)")],
    "crag_hybrid": [
        ("Retrieval", "Per sub-question: hybrid search (vector + BM25, Reciprocal Rank Fusion), top 4 chunks"),
        ("Grading", "One LLM call per chunk: relevant yes/no"),
        ("Chunks given to the LLM", "The chunks that pass grading, labelled [Source: file | Page: n]"),
        ("Answer LLM", "CHAT_MODEL (provider default reasoning)")],
}


# ---------- Scores ----------
def apply_overrides(system, judged):
    over = (load(OVERRIDES, {}) or {}).get(system, {})
    for j in judged:
        o = over.get(str(j["id"]))
        if o:
            j.update({k: v for k, v in o.items() if k != "reason"})
            j["override"] = o.get("reason", "corrected by hand")
    return judged


def has_evidence(chunk_text, evidence):
    """True if the chunk contains the evidence phrase ("a | b" = either, "a && b" = both),
    compared word by word, ignoring punctuation and case."""
    text = " " + " ".join(words(strip_header(chunk_text))) + " "
    return all(any(" " + " ".join(words(alt)) + " " in text for alt in group.split("|"))
               for group in evidence.split("&&"))


def retrieval(q, chunks):
    """Rank of the first chunk that contains part of the answer, and whether every part was
    given to the LLM. A chunk counts only if it contains the part's evidence phrase: being on
    the right page is not enough (a neighbouring chunk of the same page may miss the answer)."""
    def matches(c, part):
        return has_evidence(c["text"], part["evidence"])
    rank = next((i for i, c in enumerate(chunks, 1) if any(matches(c, p) for p in q["parts"])), None)
    covered = [any(matches(c, p) for c in chunks) for p in q["parts"]]
    return rank, all(covered) and bool(covered), covered


def score(system, key):
    answers = {r["id"]: r for r in load(answers_file(system), [])}
    judged = {j["id"]: j for j in apply_overrides(system, load(judged_file(system), []))}
    rows = []
    for q in key["questions"]:
        a, j = answers.get(q["id"]), judged.get(q["id"], {})
        if a is None:
            continue
        answerable = bool(q["parts"])
        rank, full, covered = retrieval(q, a["chunks"]) if answerable else (None, None, [])
        sims = [c["sim"] for c in a["chunks"] if c.get("sim") is not None]
        # The judge sometimes leaves these empty when an answerable question was refused:
        # no score = 0 points, and a refusal of an answerable question is a full false refusal.
        correct = (j.get("correct") or 0) if answerable else None
        # False refusals, the same rule for every pipeline (the judge's own labels were not
        # consistent): "full" = the answer refuses an answerable question and scores 0;
        # "partial" = it says part of the answer is not in the documents; "none" otherwise.
        false_refusal = None
        if answerable:
            said_missing = j.get("refused") or j.get("false_refusal") in ("full", "partial")
            false_refusal = ("full" if j.get("refused") and correct == 0 else
                             "partial" if said_missing else "none")
            false_refusal = j.get("false_refusal_override", false_refusal)
        rows.append({
            "q": q, "a": a, "j": j, "answerable": answerable, "rank": rank, "full": full,
            "covered": covered, "correct": correct,
            "faithful": bool(j.get("faithful")), "citations": j.get("citations"),
            "refused": bool(j.get("refused")), "invented": bool(j.get("invented")),
            "false_refusal": false_refusal or "none",
            "top_sim": max(sims) if sims else None,
            "usd": (a["cost"] or {}).get("usd", 0.0), "requests": (a["cost"] or {}).get("requests", 0),
            "reasoning": (a["cost"] or {}).get("reasoning_tokens", 0), "seconds": a["seconds"],
        })
    return rows


def metrics(rows):
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    n, m, N = len(ans), len(una), len(rows)
    hits = [r for r in ans if r["rank"]]
    full = [r for r in ans if r["full"]]
    rr = sum(1 / r["rank"] for r in hits)
    points = sum(r["correct"] or 0 for r in ans)
    faithful = [r for r in rows if r["faithful"]]
    cit_ok = [r for r in rows if r["citations"] in ("ok", "n/a")]
    refused_ok = [r for r in una if r["refused"] and not r["invented"]]
    fr_full = [r for r in ans if r["false_refusal"] == "full"]
    fr_part = [r for r in ans if r["false_refusal"] == "partial"]
    return {
        "n": n, "m": m, "N": N, "hits": hits, "full": full, "rr": rr, "points": points,
        "faithful": faithful, "cit_ok": cit_ok, "refused_ok": refused_ok,
        "fr_full": fr_full, "fr_part": fr_part,
        "hit_rate": len(hits) / n, "coverage": len(full) / n, "mrr": rr / n,
        "accuracy": points / (2 * n), "faithfulness": len(faithful) / N,
        "citation": len(cit_ok) / N, "refusal": len(refused_ok) / m if m else 0,
        "requests": sum(r["requests"] for r in rows) / N, "usd": sum(r["usd"] for r in rows) / N,
        "usd_total": sum(r["usd"] for r in rows),
        # median, not mean: a few requests stalled for minutes (see "slowest")
        "seconds": sorted(r["seconds"] for r in rows)[N // 2] if N % 2 else
                   sum(sorted(r["seconds"] for r in rows)[N // 2 - 1:N // 2 + 1]) / 2,
        "slowest": max(rows, key=lambda r: r["seconds"]),
        "reasoning": sum(r["reasoning"] for r in rows),
    }


def qlist(rs):
    return ", ".join(f"Q{r['q']['id']}" for r in rs) or "none"


def pct(x):
    return f"{100 * x:.0f}%" if abs(100 * x - round(100 * x)) < 0.05 else f"{100 * x:.1f}%"


# ---------- Word helpers ----------
def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    for k, v in (("w:val", "clear"), ("w:color", "auto"), ("w:fill", fill)):
        shd.set(qn(k), v)
    tcPr.append(shd)


def borders(t, color="A6A6A6"):
    """Thin borders on every cell (the template has no table style to borrow)."""
    tblPr = t._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{side}")
        for k, v in (("w:val", "single"), ("w:sz", "4"), ("w:space", "0"), ("w:color", color)):
            e.set(qn(k), v)
        b.append(e)
    tblPr.append(b)


def table(doc, headers, rows, widths, size=9):
    t = doc.add_table(rows=1, cols=len(headers))
    borders(t)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold, r.font.size = True, Pt(size)
        shade(c, "D9E2F3")
    trPr = t.rows[0]._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader")
    th.set(qn("w:val"), "true")
    trPr.append(th)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run("" if v is None else str(v))
            run.font.size = Pt(size)
    sec = doc.sections[-1]
    usable = (sec.page_width - sec.left_margin - sec.right_margin) / 360000      # cm
    scale = usable / sum(widths)
    for row in t.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = Cm(w * scale)
    doc.add_paragraph()
    return t


def caption(doc, text):
    p = doc.add_paragraph(text)
    p.runs[0].italic = True
    p.paragraph_format.keep_with_next = True


def heading(doc, text, level):
    """The template's own heading style (its style names don't match python-docx's lookup)."""
    style = next((st for st in doc.styles if st.name in (f"Heading {level}", f"heading {level}")), None)
    return doc.add_paragraph(text, style=style)


def clear_body(doc):
    body = doc.element.body
    for el in list(body):
        if el.tag != qn("w:sectPr"):
            body.remove(el)


# ---------- Report ----------
def interpretation(name, m, rows):
    miss_hit = [r for r in rows if r["answerable"] and not r["rank"]]
    miss_full = [r for r in rows if r["answerable"] and not r["full"]]
    lost = [r for r in rows if r["answerable"] and (r["correct"] or 0) < 2]
    unfaithful = [r for r in rows if not r["faithful"]]
    bad_cit = [r for r in rows if r["citations"] not in ("ok", "n/a")]
    not_refused = [r for r in rows if not r["answerable"] and not (r["refused"] and not r["invented"])]
    return {
        "hit": "Every answerable question got at least one relevant chunk" if not miss_hit
               else f"No relevant chunk for {qlist(miss_hit)}",
        "coverage": "All parts of every answer were retrieved" if not miss_full
                    else f"Some part missing for {qlist(miss_full)}",
        "mrr": "The relevant chunk is usually ranked first" if m["mrr"] >= 0.8
               else "Relevant chunks often appear below other chunks" if m["hits"] else "No relevant chunks ranked",
        "accuracy": "All answerable questions fully correct" if not lost else f"Points lost on {qlist(lost)}",
        "faithful": "No invented information" if not unfaithful else f"Unsupported claims in {qlist(unfaithful)}",
        "citation": "Every citation pointed to the right chunk" if not bad_cit
                    else f"Missing or wrong citations in {qlist(bad_cit)}",
        "refusal": "All unanswerable questions refused, including the traps" if not not_refused
                   else f"Answered instead of refusing: {qlist(not_refused)}",
        "false_refusal": (f"Full: {qlist(m['fr_full'])}; partial: {qlist(m['fr_part'])}"
                          if m["fr_full"] or m["fr_part"] else "No available answer reported as missing"),
    }


def model_section(doc, number, system, rows, m, key, all_metrics):
    label = LABEL[system]
    heading(doc, f"{number}. {label}", 1)
    n, N, M = m["n"], m["N"], m["m"]
    t = 1
    caption(doc, f"Table {number}.{t}: Evaluation setup")
    table(doc, ["Item", "Value"], SETUP[system] + [
        ("Test questions", f"{N}: {n} answerable and {M} unanswerable"),
        ("Grading of answers", "LLM judge against the answer key, every verdict checked by hand")],
        [4.2, 12.4])
    t += 1
    caption(doc, f"Table {number}.{t}: Metrics and calculations")
    rr_detail = defaultdict(int)
    for r in m["hits"]:
        rr_detail[r["rank"]] += 1
    rr_text = ", ".join(f"rank {k} for {v}" for k, v in sorted(rr_detail.items())) or "no hits"
    pts = defaultdict(int)
    for r in rows:
        if r["answerable"]:
            pts[r["correct"] or 0] += 1
    table(doc, ["Metric", "What it measures", "Formula", "Calculation in this evaluation"], [
        ("Hit rate", "At least one chunk containing part of the answer was given to the LLM", "Hits ÷ answerable questions",
         f"{len(m['hits'])} ÷ {n} = {m['hit_rate']:.2f} ({pct(m['hit_rate'])})"),
        ("Full-answer coverage", "Every part of the answer was given to the LLM", "Fully covered ÷ answerable questions",
         f"{len(m['full'])} ÷ {n} = {m['coverage']:.2f} ({pct(m['coverage'])})"),
        ("MRR", "How high the first relevant chunk was ranked", "Average of 1 ÷ rank (not found = 0)",
         f"{rr_text}: total {m['rr']:.2f} ÷ {n} = {m['mrr']:.2f}"),
        ("Answer accuracy", "2 = fully correct, 1 = partly correct, 0 = wrong", f"Points ÷ ({n} × 2)",
         f"{pts[2]} × 2 + {pts[1]} × 1 = {m['points']} ÷ {2 * n} = {m['accuracy']:.2f} ({pct(m['accuracy'])})"),
        ("Faithfulness", "Every claim is supported by the chunks given", "Faithful answers ÷ all answers",
         f"{len(m['faithful'])} ÷ {N} = {m['faithfulness']:.2f} ({pct(m['faithfulness'])})"),
        ("Citation accuracy", "Citations point to the chunk that contains the fact", "Correct ÷ all answers",
         f"{len(m['cit_ok'])} ÷ {N} = {m['citation']:.2f} ({pct(m['citation'])})"),
        ("Refusal accuracy", "Says \"not covered\" for questions the documents can't answer",
         "Correct refusals ÷ unanswerable questions",
         f"{len(m['refused_ok'])} ÷ {M} = {m['refusal']:.2f} ({pct(m['refusal'])})"),
        ("False refusal rate", "An available answer reported as missing", "False refusals ÷ answerable questions",
         f"Full: {len(m['fr_full'])} ÷ {n} = {pct(len(m['fr_full']) / n)}. Partial: {len(m['fr_part'])} ÷ {n} = {pct(len(m['fr_part']) / n)}"),
        ("API requests per question", "Model requests to OpenRouter (measured)", "Requests ÷ questions",
         f"{round(m['requests'] * N)} ÷ {N} = {m['requests']:.1f}"),
        ("Cost per question", "US dollars charged by OpenRouter (measured)", "Total cost ÷ questions",
         f"${m['usd_total']:.5f} ÷ {N} = ${m['usd']:.6f}"),
    ], [3.0, 4.6, 3.6, 5.4], size=8.5)
    t += 1
    it = interpretation(system, m, rows)
    caption(doc, f"Table {number}.{t}: Results")
    table(doc, ["Metric", "Result", "Interpretation"], [
        ("Hit rate", f"{pct(m['hit_rate'])} ({len(m['hits'])}/{n})", it["hit"]),
        ("Full-answer coverage", f"{pct(m['coverage'])} ({len(m['full'])}/{n})", it["coverage"]),
        ("MRR", f"{m['mrr']:.2f}", it["mrr"]),
        ("Answer accuracy", f"{pct(m['accuracy'])} ({m['points']}/{2 * n})", it["accuracy"]),
        ("Faithfulness", f"{pct(m['faithfulness'])} ({len(m['faithful'])}/{N})", it["faithful"]),
        ("Citation accuracy", f"{pct(m['citation'])} ({len(m['cit_ok'])}/{N})", it["citation"]),
        ("Refusal accuracy", f"{pct(m['refusal'])} ({len(m['refused_ok'])}/{M})", it["refusal"]),
        ("False refusal rate", f"{pct(len(m['fr_full']) / n)} full, {pct(len(m['fr_part']) / n)} partial", it["false_refusal"]),
        ("Requests / question", f"{m['requests']:.1f}", "Measured from OpenRouter's usage data"),
        ("Cost / question", f"${m['usd']:.6f}", f"Total ${m['usd_total']:.5f}; median {m['seconds']:.1f} s per question (slowest: Q{m['slowest']['q']['id']}, {m['slowest']['seconds']:.0f} s)"
         + (f"; {m['reasoning']:,} reasoning tokens in total" if m["reasoning"] else "")),
    ], [3.4, 3.4, 9.8])
    t += 1
    caption(doc, f"Table {number}.{t}: Results by question type")
    trows = []
    for name, types in TYPE_GROUPS:
        rs = [r for r in rows if r["q"]["type"] in types]
        if not rs:
            continue
        ids = ", ".join(f"Q{r['q']['id']}" for r in rs)
        if name == "Unanswerable":
            ok = sum(1 for r in rs if r["refused"] and not r["invented"])
            trows.append((name, ids, "—", f"{ok}/{len(rs)} refused correctly ({pct(ok / len(rs))})"))
        else:
            cov = sum(1 for r in rs if r["full"])
            p = sum(r["correct"] or 0 for r in rs)
            trows.append((name, ids, f"{cov}/{len(rs)}", f"{p}/{2 * len(rs)} ({pct(p / (2 * len(rs)))})"))
    table(doc, ["Question type", "Questions", "Full-answer coverage", "Answer accuracy"], trows, [4, 4.6, 3.6, 4.4])

    if any(r["top_sim"] is not None for r in rows):
        t += 1
        caption(doc, f"Table {number}.{t}: Similarity of the best chunk given to the LLM")
        groups = [("Fully correct", [r for r in rows if r["answerable"] and r["correct"] == 2]),
                  ("Partly correct", [r for r in rows if r["answerable"] and r["correct"] == 1]),
                  ("Wrong", [r for r in rows if r["answerable"] and (r["correct"] or 0) == 0]),
                  ("Unanswerable", [r for r in rows if not r["answerable"]])]
        grows = []
        for name, rs in groups:
            sims = [r["top_sim"] for r in rs if r["top_sim"] is not None]
            grows.append((name, f"{len(rs)}" + (f" ({qlist(rs)})" if 0 < len(rs) <= 4 else ""),
                          f"{sum(sims) / len(sims):.3f}" if sims else "—",
                          f"{len(sims)} answers with chunks" if sims else "No chunks given to the LLM"))
        table(doc, ["Group", "Questions", "Average top-1 similarity", "Note"], grows, [3.2, 5.2, 3.6, 4.6])

    t += 1
    caption(doc, f"Table {number}.{t}: Per-question scores")
    headers = ["#", "Question (short)", "Type"] + \
              ["Rank", "Full answer", "Correct (0–2)", "Faithful", "Citations", "Refused", "Cost $", "Notes"]
    prow = []
    for r in rows:
        q, j = r["q"], r["j"]
        note = j.get("override") or j.get("note") or ""
        if r["a"].get("error"):
            note = f"Pipeline error: {r['a']['error'][:80]}"
        prow.append([q["id"], q["short"], q["type"]] + [
            r["rank"] or ("—" if not r["answerable"] else "none"),
            ("Yes" if r["full"] else "No") if r["answerable"] else "—",
            r["correct"] if r["answerable"] else "—",
            "Yes" if r["faithful"] else "No",
            {"ok": "Yes", "n/a": "—", "wrong": "Wrong", "missing": "Missing"}.get(r["citations"], "?"),
            "Yes" if r["refused"] else "—",
            f"{r['usd']:.5f}", note])
    widths = [0.7, 3.0, 1.9] + [1.0, 1.2, 1.3, 1.3, 1.5, 1.3, 1.5, 4.0]
    table(doc, headers, prow, widths, size=7.5)


def findings(doc, M, R, systems):
    """Short written conclusions under the comparison table; every number comes from the results."""
    if not {"hybrid", "crag_hybrid"} <= set(systems):
        return
    h, c = M["hybrid"], M["crag_hybrid"]
    caption(doc, "Key findings")
    pts = {r["q"]["id"]: sum(R[s][i]["correct"] or 0 for s in systems)
           for i, r in enumerate(R["hybrid"]) if r["answerable"]}
    hardest = sorted(pts, key=pts.get)[:3]
    lines = [
        ("Hybrid RAG: ",
         f"{pct(h['accuracy'])} answer accuracy, {pct(h['coverage'])} full-answer coverage, "
         f"{pct(h['faithfulness'])} faithfulness and {pct(h['citation'])} citation accuracy, at "
         f"${h['usd']:.6f} per question. Combining keyword search (BM25) with vector search finds exact terms "
         "such as table values that vector search alone tends to miss."),
        ("Corrective Hybrid RAG: ",
         f"{pct(c['accuracy'])} answer accuracy and {pct(c['coverage'])} full-answer coverage. Its per-chunk "
         f"grader removed some relevant chunks (coverage {pct(h['coverage'])} without grading), and it makes "
         f"{c['requests']:.0f} requests per question instead of {h['requests']:.0f}."),
        ("Unanswerable questions: ",
         f"refused correctly in {pct(h['refusal'])} (Hybrid) and {pct(c['refusal'])} (Corrective Hybrid) of cases."),
        ("Hardest questions: ",
         ", ".join(f"Q{q} ({pts[q]}/{2 * len(systems)} points)" for q in hardest) + "."),
        ("Reasoning tokens: ",
         f"neither pipeline switches DeepSeek's reasoning off; they spent {h['reasoning']:,} (Hybrid) and "
         f"{c['reasoning']:,} (Corrective Hybrid) reasoning tokens over the 20 questions."),
    ]
    for lead, text in lines:
        p = doc.add_paragraph(style=next((st for st in doc.styles if st.name in ("List Bullet", "list bullet")), None))
        r = p.add_run(lead)
        r.bold = True
        p.add_run(text)


def build(template, out):
    key = answer_key()
    doc = Document(template)
    clear_body(doc)
    systems = [s for s, _, _ in PIPELINES if load(judged_file(s))]
    all_rows = {s: score(s, key) for s in systems}
    all_m = {s: metrics(all_rows[s]) for s in systems}

    doc.add_paragraph("RAG Evaluation: Hybrid and Corrective Hybrid RAG", style="Title")
    doc.add_paragraph(
        "Evaluation of the project's two RAG pipelines on the same 20 test questions, two PDF documents and "
        "answer key used for the earlier plain and Corrective RAG evaluations, with the same metrics so the "
        "results can be compared directly.")

    heading(doc, "1. Common setup", 1)
    caption(doc, "Table 1: Evaluation setup shared by all pipelines")
    table(doc, ["Item", "Value"], [
        ("Documents", "Hummes et al. (2023), battery geometries for EVs: 21 pages, 91 chunks. "
                      "NVWA/BuRO (2025), cultured meat risk advice: 33 pages, 201 chunks."),
        ("Chunking", "1,000 characters with 200 overlap (LangChain RecursiveCharacterTextSplitter), page by page"),
        ("Vector store", "ChromaDB (chroma_db_global), collection global_hybrid_collection, shared by both pipelines"),
        ("Embedding model", "openai/text-embedding-3-small (1,536 dimensions), via OpenRouter"),
        ("Chat model", "deepseek/deepseek-v4-flash via OpenRouter (CHAT_MODEL)"),
        ("Test questions", "20: 16 answerable (9 battery, 5 cultured meat, 2 cross-document) and 4 unanswerable, "
                           "two of them traps with closely related content"),
        ("Run", "Each question asked once, on its own, with an empty chat history; nothing in the pipelines changed"),
        ("Grading", "Retrieval: a chunk given to the LLM counts as relevant when its text contains the evidence phrase "
                    "of that part of the answer (taken from the PDF); being on the right page is not enough. Answers: an LLM judge (deepseek-v4-flash with reasoning) against the answer "
                    "key; every verdict was then checked by hand and corrected where needed (marked in the notes)."),
        ("Cost", "Measured: the US dollars OpenRouter reported for each request (hybrid_rag/cost.py)"),
    ], [3.6, 13])

    heading(doc, "2. Comparison of all pipelines", 1)
    caption(doc, "Table 2: Results of every pipeline on the same 20 questions")
    order = ["hit_rate", "coverage", "mrr", "accuracy", "faithfulness", "citation", "refusal"]
    names = {"hit_rate": "Hit rate", "coverage": "Full-answer coverage", "mrr": "MRR",
             "accuracy": "Answer accuracy", "faithfulness": "Faithfulness",
             "citation": "Citation accuracy", "refusal": "Refusal accuracy"}
    rows = []
    for k in order:
        rows.append([names[k]] + [f"{all_m[s][k]:.2f}" if k == "mrr" else pct(all_m[s][k]) for s in systems])
    rows.append(["False refusals (full / partial)"] + [f"{len(all_m[s]['fr_full'])} / {len(all_m[s]['fr_part'])}" for s in systems])
    rows.append(["Requests / question"] + [f"{all_m[s]['requests']:.1f}" for s in systems])
    rows.append(["Cost / question ($)"] + [f"{all_m[s]['usd']:.6f}" for s in systems])
    rows.append(["Median time / question (s)"] + [f"{all_m[s]['seconds']:.1f}" for s in systems])
    short = {"hybrid": "Hybrid RAG", "crag_hybrid": "Corrective Hybrid RAG"}
    table(doc, ["Metric"] + [short[s] for s in systems], rows, [3.4] + [13.2 / len(systems)] * len(systems), size=8)

    findings(doc, all_m, all_rows, systems)

    heading(doc, "3. Metrics", 1)
    doc.add_paragraph(
        "The metrics and formulas are the same as in the earlier plain RAG and Corrective RAG evaluations. "
        "Retrieval metrics are measured on the chunks given to the answer LLM (after grading, for the corrective "
        "pipelines). A chunk counts as relevant when its text contains the evidence phrase the answer key gives for "
        "that part of the answer (e.g. \"battery cost 5 1\" and \"power density 1 5\" for the Table 4 weights): being "
        "on the right page is not enough, since a neighbouring chunk of the same page can miss the answer. Rank is the "
        "position of that chunk among the chunks given to the LLM. Each pipeline's section shows the calculation with "
        "its own numbers.")

    for i, s in enumerate(systems, 4):
        model_section(doc, i, s, all_rows[s], all_m[s], key, all_m)

    heading(doc, f"{4 + len(systems)}. Limitations", 1)
    doc.add_paragraph(
        "20 questions is a small test set, and each question was asked once. Retrieval is judged by evidence phrases: "
        "a chunk that states the answer in different words would not count. The answer judge is an LLM; every "
        "verdict was checked by hand, but borderline partial-credit calls remain a judgement. Neither pipeline "
        "switches reasoning off, so DeepSeek may spend reasoning tokens (counted in their cost).")

    heading(doc, "Appendix: Answer key", 1)
    akrows = []
    for q in key["questions"]:
        exp = " | ".join(f"{p['answer']} [{DOCS[p['doc']].split('_')[0].split(' ')[0]} p. {', '.join(map(str, p['pages']))}]"
                         for p in q["parts"]) or f"Unanswerable. {q.get('unanswerable_note', '')}"
        akrows.append((q["id"], q["question"], q["type"], exp))
    table(doc, ["#", "Question", "Type", "Expected answer [document, pages]"], akrows, [0.7, 5.2, 2.0, 8.7], size=8)
    doc.save(out)
    return all_m


def main():
    parser = argparse.ArgumentParser(description="Build the evaluation report")
    parser.add_argument("--template", default=str(Path.home() / "Downloads" / "Corrective_RAG_Evaluation_Tables.docx"))
    parser.add_argument("--out", default=str(Path.home() / "Downloads" / "RAG_Evaluation_All_Pipelines.docx"))
    args = parser.parse_args()
    m = build(args.template, args.out)
    for s, v in m.items():
        print(f"{LABEL[s]:32} hit {pct(v['hit_rate']):>5}  cov {pct(v['coverage']):>5}  mrr {v['mrr']:.2f}  "
              f"acc {pct(v['accuracy']):>5}  faith {pct(v['faithfulness']):>5}  cit {pct(v['citation']):>5}  "
              f"ref {pct(v['refusal']):>5}  ${v['usd']:.6f}/q  {v['seconds']:.1f}s")
    print("saved", args.out)


if __name__ == "__main__":
    main()
