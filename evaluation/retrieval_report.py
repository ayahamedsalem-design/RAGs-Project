"""
Generates the Retrieval Evaluation Report in three formats:
  1. Markdown: evaluation/results/Retrieval_Evaluation_Report.md
  2. Word:     evaluation/results/Retrieval_Evaluation_Report.docx (styled after docs/Corrective_RAG_Evaluation_Tables.docx)
  3. JSON:     evaluation/results/retrieval_summary.json

Combines standard IR retrieval metrics (Document Accuracy, Page Hit@K, Page MRR,
Answer-Fact Hit@K, Full-Answer Coverage, Precision@K, MRR, nDCG@K, and CRAG Pre- vs Post-Grading
Noise Removed / False Drops) with the table structure and analytical layout of the existing
Corrective_RAG_Evaluation_Tables.docx report.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

from evaluation.retrieval_dataset import (
    DOCS_DIR,
    EVAL_DOCS,
    RESULTS_DIR,
    load_dataset,
    score_chunk,
    score_chunk_set_coverage,
)
from evaluation.retrieval_eval import ALL_PIPELINES, PIPELINE_META, result_file

TEMPLATE_DOCX = DOCS_DIR / "Corrective_RAG_Evaluation_Tables.docx"

SHORT_LABEL = {
    "hybrid": "Hybrid",
    "crag_hybrid": "CRAG Hybrid",
    "basic": "Basic",
    "parent_child": "Parent-Child",
    "crag_parent_child": "CRAG Parent-Child",
    "naive": "Ext Naive",
    "crag": "Ext CRAG",
}

PIPELINE_SETUP = {
    "hybrid": [
        ("Vector store", "ChromaDB (chroma_db_global / global_hybrid_collection)"),
        ("Chunking", "1,000 characters, 200 overlap (RecursiveCharacterTextSplitter)"),
        ("Pre-LLM Retrieval", "Per sub-question: Dense vector (top 10) + BM25 keyword (top 10), fused via RRF (k=60) -> top 6 chunks"),
        ("Grading / Filter", "None (all 6 chunks go directly to the Answer LLM)"),
    ],
    "crag_hybrid": [
        ("Vector store", "ChromaDB (chroma_db_global / global_hybrid_collection)"),
        ("Chunking", "1,000 characters, 200 overlap (RecursiveCharacterTextSplitter)"),
        ("Pre-Grading Retrieval", "Per sub-question: Dense (top 10) + BM25 (top 10) via RRF -> top 4 candidate chunks/sub-query"),
        ("CRAG Grader", "1 binary LLM call (yes/no, temperature=0) per candidate chunk before Answer LLM"),
    ],
    "basic": [
        ("Vector store", "ChromaDB (chroma_db_global / global_hybrid_collection)"),
        ("Chunking", "1,000 characters, 200 overlap (RecursiveCharacterTextSplitter)"),
        ("Pre-LLM Retrieval", "Pure dense vector search: top 3 chunks per sub-question (deduplicated)"),
        ("Grading / Filter", "None"),
    ],
    "parent_child": [
        ("Vector store", "ChromaDB (global_child_collection + global_parent_collection)"),
        ("Chunking", "300-word parents / 70-word children with 15-word overlap (no page metadata stored)"),
        ("Pre-LLM Retrieval", "Top 10 child chunks per sub-question -> up to 6 unique parent blocks"),
        ("Grading / Filter", "None"),
    ],
    "crag_parent_child": [
        ("Vector store", "ChromaDB (global_child_collection + global_parent_collection)"),
        ("Chunking", "300-word parents / 70-word children with 15-word overlap"),
        ("Pre-Grading Retrieval", "Top 10 child chunks per sub-question -> up to 4 unique parent blocks/sub-query"),
        ("CRAG Grader", "1 binary LLM call (yes/no, temperature=0) per parent candidate before Answer LLM"),
    ],
    "naive": [
        ("Vector store", "ChromaDB (chroma_db_global / ext_chunks_openrouter)"),
        ("Chunking", "1,000 characters, 200 overlap (L2-normalized embeddings)"),
        ("Pre-LLM Retrieval", "Cosine similarity top 4 chunks"),
        ("Grading / Filter", "None"),
    ],
    "crag": [
        ("Vector store", "ChromaDB (chroma_db_global / ext_chunks_openrouter)"),
        ("Chunking", "1,000 characters, 200 overlap (L2-normalized embeddings)"),
        ("Pre-Grading Retrieval", "Top 10 cosine candidates (sim >= 0.05) + reference-list filter (>= 12 citations/100w removed)"),
        ("CRAG Grader & Correction", "Batch LLM grader + sufficiency check; Round-2 query rewriting & per-file search if incomplete; capped at 6 chunks"),
    ],
}

CATEGORY_ORDER = [
    "Single fact",
    "Numbers / Hyperparameters",
    "Definition / Concept",
    "Mechanism / Why",
    "Multi-part / Comparison",
]


def pct(x: float) -> str:
    return f"{100 * x:.0f}%" if abs(100 * x - round(100 * x)) < 0.05 else f"{100 * x:.1f}%"


def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _ndcg_binary(rels: list[int]) -> float:
    if not rels or not any(rels):
        return 0.0
    dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rels, start=1) if r)
    n_rel = sum(1 for r in rels if r)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _eval_chunk_list(chunks: list[dict], cov: dict) -> dict:
    k = len(chunks)
    doc_flags = [1 if c["scores"]["doc_match"] else 0 for c in chunks]
    page_flags = [1 if c["scores"]["page_match"] else 0 for c in chunks]
    fact_flags = [1 if c["scores"]["fact_match"] else 0 for c in chunks]

    doc_hit = any(doc_flags)
    doc_prec = (sum(doc_flags) / k) if k else 0.0
    page_hit = any(page_flags)
    page_rank = next((i for i, f in enumerate(page_flags, start=1) if f), None)
    fact_hit = any(fact_flags)
    fact_rank = next((i for i, f in enumerate(fact_flags, start=1) if f), None)
    prec_at_k = (sum(fact_flags) / k) if k else 0.0
    ndcg = _ndcg_binary(fact_flags)
    sims = [c["sim"] for c in chunks if c.get("sim") is not None]

    return {
        "k": k,
        "doc_hit": doc_hit,
        "doc_precision": doc_prec,
        "page_hit": page_hit,
        "page_rank": page_rank,
        "page_rr": (1.0 / page_rank) if page_rank else 0.0,
        "fact_hit": fact_hit,
        "fact_rank": fact_rank,
        "fact_rr": (1.0 / fact_rank) if fact_rank else 0.0,
        "full_coverage": bool(cov.get("full_coverage")),
        "term_recall": float(cov.get("combined_term_recall", 0.0)),
        "precision": prec_at_k,
        "ndcg": ndcg,
        "top_sim": max(sims) if sims else None,
        "rel_count": sum(fact_flags),
        "noise_count": k - sum(fact_flags),
    }


def compute_pipeline_metrics(system: str, dataset: list[dict], records: list[dict]) -> dict:
    label, is_crag = PIPELINE_META[system]
    q_by_id = {q["id"]: q for q in dataset}
    valid = [r for r in records if not r.get("error") and r["id"] in q_by_id]
    errors = [r for r in records if r.get("error")]

    rows = []
    for r in valid:
        q = q_by_id[r["id"]]
        pre_chunks = r.get("pre_chunks", [])
        post_chunks = r.get("post_chunks", []) if is_crag else pre_chunks

        for c in pre_chunks:
            c["scores"] = score_chunk(q, c.get("text", ""), c.get("doc"), c.get("pages", []))
        pre_cov = score_chunk_set_coverage(q, pre_chunks)
        if is_crag:
            for c in post_chunks:
                c["scores"] = score_chunk(q, c.get("text", ""), c.get("doc"), c.get("pages", []))
            post_cov = score_chunk_set_coverage(q, post_chunks)
        else:
            post_cov = pre_cov

        pre_m = _eval_chunk_list(pre_chunks, pre_cov)
        post_m = _eval_chunk_list(post_chunks, post_cov) if is_crag else pre_m

        # Grader impact on pre_chunks (for CRAG pipelines)
        pre_rel_total = sum(1 for c in pre_chunks if c["scores"]["fact_match"])
        pre_rel_dropped = sum(1 for c in pre_chunks if c["scores"]["fact_match"] and not c.get("kept"))
        pre_noise_total = sum(1 for c in pre_chunks if not c["scores"]["fact_match"])
        pre_noise_dropped = sum(1 for c in pre_chunks if not c["scores"]["fact_match"] and not c.get("kept"))

        cost_info = r.get("cost") or {}
        rows.append({
            "q": q,
            "r": r,
            "pre": pre_m,
            "post": post_m,
            "pre_rel_total": pre_rel_total,
            "pre_rel_dropped": pre_rel_dropped,
            "pre_noise_total": pre_noise_total,
            "pre_noise_dropped": pre_noise_dropped,
            "lost_by_grader": is_crag and pre_m["fact_hit"] and not post_m["fact_hit"],
            "gained_by_correction": is_crag and (not pre_m["fact_hit"]) and post_m["fact_hit"],
            "seconds": float(r.get("seconds", 0.0)),
            "usd": float(cost_info.get("usd", 0.0)),
            "requests": int(cost_info.get("total_estimated_requests", 1)),
            "grader_requests": int(r.get("grader_requests", 0)),
            "status": r.get("status"),
            "refs_removed": int(r.get("refs_removed", 0)),
            "round2_triggered": bool(r.get("round2_triggered", False)),
        })

    def _agg(stage: str) -> dict:
        n = len(rows)
        if n == 0:
            return {}
        doc_hits = [x for x in rows if x[stage]["doc_hit"]]
        page_hits = [x for x in rows if x[stage]["page_hit"]]
        fact_hits = [x for x in rows if x[stage]["fact_hit"]]
        full_covs = [x for x in rows if x[stage]["full_coverage"]]
        return {
            "n": n,
            "avg_k": sum(x[stage]["k"] for x in rows) / n,
            "doc_hit_count": len(doc_hits),
            "doc_hit_rate": len(doc_hits) / n,
            "doc_precision": sum(x[stage]["doc_precision"] for x in rows) / n,
            "page_hit_count": len(page_hits),
            "page_hit_rate": len(page_hits) / n,
            "page_rr_sum": sum(x[stage]["page_rr"] for x in rows),
            "page_mrr": sum(x[stage]["page_rr"] for x in rows) / n,
            "fact_hit_count": len(fact_hits),
            "fact_hit_rate": len(fact_hits) / n,
            "fact_rr_sum": sum(x[stage]["fact_rr"] for x in rows),
            "mrr": sum(x[stage]["fact_rr"] for x in rows) / n,
            "full_cov_count": len(full_covs),
            "full_coverage": len(full_covs) / n,
            "avg_term_recall": sum(x[stage]["term_recall"] for x in rows) / n,
            "precision": sum(x[stage]["precision"] for x in rows) / n,
            "ndcg": sum(x[stage]["ndcg"] for x in rows) / n,
        }

    pre_agg = _agg("pre")
    post_agg = _agg("post")

    total_pre_noise = sum(x["pre_noise_total"] for x in rows)
    total_pre_noise_dropped = sum(x["pre_noise_dropped"] for x in rows)
    total_pre_rel = sum(x["pre_rel_total"] for x in rows)
    total_pre_rel_dropped = sum(x["pre_rel_dropped"] for x in rows)

    grader_stats = None
    if is_crag and rows:
        grader_stats = {
            "total_pre_chunks": sum(x["pre"]["k"] for x in rows),
            "total_post_chunks": sum(x["post"]["k"] for x in rows),
            "total_pre_noise": total_pre_noise,
            "total_pre_noise_dropped": total_pre_noise_dropped,
            "noise_removed_rate": (total_pre_noise_dropped / total_pre_noise) if total_pre_noise else 0.0,
            "total_pre_rel": total_pre_rel,
            "total_pre_rel_dropped": total_pre_rel_dropped,
            "false_drop_rate": (total_pre_rel_dropped / total_pre_rel) if total_pre_rel else 0.0,
            "lost_qids": [x["q"]["id"] for x in rows if x["lost_by_grader"]],
            "gained_qids": [x["q"]["id"] for x in rows if x["gained_by_correction"]],
            "status_counts": {
                st: [x["q"]["id"] for x in rows if x["status"] == st]
                for st in ("CORRECT", "CORRECTED", "PARTIAL", "NOT FOUND")
            } if system == "crag" else None,
            "total_refs_removed": sum(x["refs_removed"] for x in rows),
            "round2_count": sum(1 for x in rows if x["round2_triggered"]),
        }

    n = len(rows) or 1
    return {
        "system": system,
        "label": label,
        "is_crag": is_crag,
        "n_valid": len(rows),
        "n_errors": len(errors),
        "rows": rows,
        "pre": pre_agg,
        "post": post_agg,
        "grader": grader_stats,
        "median_seconds": _median([x["seconds"] for x in rows]),
        "slowest": max(rows, key=lambda x: x["seconds"]) if rows else None,
        "avg_requests": sum(x["requests"] for x in rows) / n,
        "total_requests": sum(x["requests"] for x in rows),
        "usd_total": sum(x["usd"] for x in rows),
        "usd_per_q": sum(x["usd"] for x in rows) / n,
    }


def _store_chunk_counts() -> dict[str, dict[str, int]]:
    """Inspects current chunk counts per PDF in each of the 3 ChromaDB store groups."""
    counts = {"hybrid": {}, "parent": {}, "child": {}, "ext": {}}
    try:
        from hybrid_rag.hybrid import global_hybrid_col
        for info in EVAL_DOCS.values():
            fn = info["pdf"]
            counts["hybrid"][fn] = len((global_hybrid_col.get(where={"source": fn}, include=[]) or {}).get("ids") or [])
    except Exception:
        pass
    try:
        from other_rags.parent_child import global_child_col, global_parent_col
        for info in EVAL_DOCS.values():
            fn = info["pdf"]
            counts["parent"][fn] = len((global_parent_col.get(where={"source": fn}, include=[]) or {}).get("ids") or [])
            counts["child"][fn] = len((global_child_col.get(where={"source": fn}, include=[]) or {}).get("ids") or [])
    except Exception:
        pass
    try:
        from ext_rag.rag_common import open_store, store_exists
        if store_exists():
            col = open_store(check=False)
            for info in EVAL_DOCS.values():
                fn = info["pdf"]
                counts["ext"][fn] = len((col.get(where={"path": fn}, include=[]) or {}).get("ids") or [])
    except Exception:
        pass
    return counts


def _qlist(qids: list[int], limit: int = 12) -> str:
    if not qids:
        return "none"
    if len(qids) <= limit:
        return ", ".join(f"Q{i}" for i in qids)
    return ", ".join(f"Q{i}" for i in qids[:limit]) + f" (+{len(qids) - limit} more)"


def _rr_breakdown(rows: list[dict], stage: str, key: str = "fact_rank") -> str:
    counts = defaultdict(int)
    for x in rows:
        rk = x[stage][key]
        if rk:
            counts[rk] += 1
    return ", ".join(f"rank {k} for {v}" for k, v in sorted(counts.items())) or "no hits"


def _interpret_pipeline(pm: dict) -> dict[str, str]:
    st = "post" if pm["is_crag"] else "pre"
    m = pm[st]
    rows = pm["rows"]
    miss_doc = [x["q"]["id"] for x in rows if not x[st]["doc_hit"]]
    miss_page = [x["q"]["id"] for x in rows if not x[st]["page_hit"]]
    miss_fact = [x["q"]["id"] for x in rows if not x[st]["fact_hit"]]
    miss_cov = [x["q"]["id"] for x in rows if not x[st]["full_coverage"]]

    return {
        "doc_hit": "Every question retrieved from the right PDF" if not miss_doc else f"Wrong PDF only for {_qlist(miss_doc)}",
        "doc_prec": "Minimal cross-document bleed" if m["doc_precision"] >= 0.85 else "Noticeable chunks pulled from other PDFs",
        "page_hit": "Every question hit the exact ground-truth PDF page" if not miss_page else f"Missed exact page on {len(miss_page)} Qs ({_qlist(miss_page, 8)})",
        "fact_hit": "Every question retrieved at least one chunk with the answer evidence" if not miss_fact else f"No answer-evidence chunk on {len(miss_fact)} Qs ({_qlist(miss_fact, 8)})",
        "coverage": "Full answer facts retrieved for every question" if not miss_cov else f"Incomplete facts on {len(miss_cov)} Qs ({_qlist(miss_cov, 8)})",
        "precision": "High signal-to-noise ratio" if m["precision"] >= 0.5 else "Moderate noise among top-K chunks" if m["precision"] >= 0.25 else "High noise: most returned chunks lack the answer",
        "mrr": "First relevant chunk is usually rank 1" if m["mrr"] >= 0.75 else "Relevant chunks often appear below rank 1",
        "ndcg": "Strong top-heavy ranking of relevant chunks" if m["ndcg"] >= 0.75 else "Relevant chunks spread down the list",
    }


def _build_key_findings(metrics_by_sys: dict[str, dict]) -> list[tuple[str, str]]:
    systems = list(metrics_by_sys.keys())
    if not systems:
        return []

    def _primary(s):
        pm = metrics_by_sys[s]
        return pm["post"] if pm["is_crag"] else pm["pre"]

    best_hit = max(systems, key=lambda s: (_primary(s)["fact_hit_rate"], _primary(s)["mrr"]))
    best_prec = max(systems, key=lambda s: (_primary(s)["precision"], _primary(s)["fact_hit_rate"]))
    best_mrr = max(systems, key=lambda s: (_primary(s)["mrr"], _primary(s)["ndcg"]))

    findings = [
        (
            "Best Retrieval Hit Rate (Pre-Answer-LLM): ",
            f"{metrics_by_sys[best_hit]['label']} led with {pct(_primary(best_hit)['fact_hit_rate'])} Answer-Fact Hit@K "
            f"({pct(_primary(best_hit)['page_hit_rate'])} Page Hit@K, {pct(_primary(best_hit)['full_coverage'])} full coverage, "
            f"MRR {_primary(best_hit)['mrr']:.2f}).",
        ),
        (
            "Best Precision / Lowest Noise: ",
            f"{metrics_by_sys[best_prec]['label']} achieved {pct(_primary(best_prec)['precision'])} Precision@K "
            f"(avg {_primary(best_prec)['avg_k']:.1f} chunks handed to the Answer LLM).",
        ),
        (
            "Best Ranking Quality (MRR / nDCG): ",
            f"{metrics_by_sys[best_mrr]['label']} scored MRR {_primary(best_mrr)['mrr']:.2f} and "
            f"nDCG@K {_primary(best_mrr)['ndcg']:.2f}.",
        ),
    ]

    if "hybrid" in metrics_by_sys and "basic" in metrics_by_sys:
        h, b = metrics_by_sys["hybrid"]["pre"], metrics_by_sys["basic"]["pre"]
        findings.append((
            "Hybrid (Dense + BM25) vs Basic (Dense-Only) on the same store: ",
            f"Hybrid achieved {pct(h['fact_hit_rate'])} Fact Hit@K (MRR {h['mrr']:.2f}, Page Hit {pct(h['page_hit_rate'])}) "
            f"vs Basic's {pct(b['fact_hit_rate'])} Fact Hit@K (MRR {b['mrr']:.2f}, Page Hit {pct(b['page_hit_rate'])}).",
        ))

    # CRAG Grader impact summary
    for s in ("crag_hybrid", "crag_parent_child", "crag"):
        if s in metrics_by_sys and metrics_by_sys[s]["grader"]:
            pm = metrics_by_sys[s]
            g = pm["grader"]
            findings.append((
                f"{pm['label']} Grader Impact (Pre -> Post): ",
                f"Cut avg chunks from {pm['pre']['avg_k']:.1f} -> {pm['post']['avg_k']:.1f}, "
                f"removed {pct(g['noise_removed_rate'])} of irrelevant chunks, moved Precision@K from "
                f"{pct(pm['pre']['precision'])} -> {pct(pm['post']['precision'])}, with a "
                f"{pct(g['false_drop_rate'])} relevant-chunk false drop rate "
                f"(questions lost: {len(g['lost_qids'])}"
                + (f"; gained via Round-2 correction: {len(g['gained_qids'])}" if s == "crag" else "")
                + ").",
            ))

    # Hardest document across all pipelines
    doc_avg_hit = {}
    for dkey, info in EVAL_DOCS.items():
        rates = []
        for s in systems:
            pm = metrics_by_sys[s]
            st = "post" if pm["is_crag"] else "pre"
            drows = [x for x in pm["rows"] if x["q"]["doc"] == dkey]
            if drows:
                rates.append(sum(1 for x in drows if x[st]["fact_hit"]) / len(drows))
        if rates:
            doc_avg_hit[info["label"]] = sum(rates) / len(rates)
    if doc_avg_hit:
        hardest_doc = min(doc_avg_hit, key=doc_avg_hit.get)
        easiest_doc = max(doc_avg_hit, key=doc_avg_hit.get)
        findings.append((
            "Document Difficulty (across all pipelines): ",
            f"Hardest document was {hardest_doc} (avg {pct(doc_avg_hit[hardest_doc])} Fact Hit@K); "
            f"easiest was {easiest_doc} (avg {pct(doc_avg_hit[easiest_doc])} Fact Hit@K).",
        ))

    return findings


# ---------- Markdown Report Generator ----------
def _md_table(headers: list[str], rows: list[list]) -> str:
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        cells = [("" if v is None else str(v)).replace("\n", " ").replace("|", "\\|") for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown_report(
    out_path: Path,
    dataset: list[dict],
    metrics_by_sys: dict[str, dict],
    store_counts: dict,
    per_file: bool,
):
    systems = list(metrics_by_sys.keys())
    embed_model = os.getenv("EMBEDDING_MODEL", "unknown")
    chat_model = os.getenv("CHAT_MODEL", "unknown")
    scope_label = "Per-file filtered search (`selected_files=[doc]`)" if per_file else "Global search across all 4 PDFs simultaneously"

    md: list[str] = []
    md.append("# Retrieval Evaluation Report: All 7 RAG Pipelines (Pre-Answer-LLM)")
    md.append("")
    md.append(
        f"Generated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}` | "
        f"Test set: **{len(dataset)} questions** (4 PDFs × 20 Q&As from `docs/`) | "
        f"Scope: **{scope_label}**"
    )
    md.append("")
    md.append(
        "> **Focus:** Pure retrieval quality of the chunks retrieved by each RAG architecture **before the final Answer LLM**. "
        "For the 3 Corrective pipelines (`Corrective Hybrid`, `Corrective Parent-Child`, `Ext Corrective`), "
        "both **Pre-Grading (raw retrieval)** and **Post-Grading (filtered chunks handed to the Answer LLM)** are measured side by side."
    )
    md.append("")

    # Section 1: Common Setup
    md.append("## 1. Common Evaluation Setup")
    doc_summary = "; ".join(
        f"{info['label']} (`{info['pdf']}`: "
        f"hybrid={store_counts['hybrid'].get(info['pdf'], 0)}, "
        f"ext={store_counts['ext'].get(info['pdf'], 0)}, "
        f"parents={store_counts['parent'].get(info['pdf'], 0)}, "
        f"children={store_counts['child'].get(info['pdf'], 0)})"
        for info in EVAL_DOCS.values()
    )
    md.append(_md_table(
        ["Item", "Value"],
        [
            ["Documents (4 PDFs)", doc_summary],
            ["Total Chunks per Store",
             f"Standard (`global_hybrid_collection`): {sum(store_counts['hybrid'].values()):,} | "
             f"Ext (`ext_chunks_openrouter`): {sum(store_counts['ext'].values()):,} | "
             f"Hierarchical (`parent` / `child`): {sum(store_counts['parent'].values()):,} / {sum(store_counts['child'].values()):,}"],
            ["Embedding Model", f"`{embed_model}` via OpenRouter"],
            ["Grader LLM (CRAG only)", f"`{chat_model}` via OpenRouter (Answer LLM NOT called)"],
            ["Search Scope", scope_label],
            ["Test Questions", f"{len(dataset)} answerable questions (20 Battery, 20 Bishop PRML, 20 Cultured Meat, 20 AlexNet)"],
            ["Relevance Grading",
             "Deterministic dual check: (1) Exact PDF & Page match via 8-word text anchor locator (works for Parent-Child too); "
             "(2) Answer-Fact match verifying distinctive answer terms & numbers inside the chunk text."],
        ],
    ))
    md.append("")

    # Section 2: Master Comparison Table
    md.append("## 2. Comparison of All 7 Pipelines (Pre-Answer-LLM)")
    md.append("")
    md.append("### Table 2.1: Final Chunks Handed to Answer LLM (Post-Grading for CRAG, Raw Retrieval for Non-CRAG)")
    comp_headers = ["Metric"] + [SHORT_LABEL[s] for s in systems]

    def _p(s):
        return metrics_by_sys[s]["post"] if metrics_by_sys[s]["is_crag"] else metrics_by_sys[s]["pre"]

    comp_rows = [
        ["Questions Evaluated"] + [f"{metrics_by_sys[s]['n_valid']}/{len(dataset)}" for s in systems],
        ["Avg Chunks Returned (K)"] + [f"{_p(s)['avg_k']:.1f}" for s in systems],
        ["Document Accuracy (Source Hit@K)"] + [pct(_p(s)["doc_hit_rate"]) for s in systems],
        ["Document Precision (Right-PDF %)"] + [pct(_p(s)["doc_precision"]) for s in systems],
        ["Page Hit Rate@K"] + [pct(_p(s)["page_hit_rate"]) for s in systems],
        ["Page MRR"] + [f"{_p(s)['page_mrr']:.2f}" for s in systems],
        ["Answer-Fact Hit Rate@K"] + [pct(_p(s)["fact_hit_rate"]) for s in systems],
        ["Full-Answer Coverage (Recall)"] + [pct(_p(s)["full_coverage"]) for s in systems],
        ["Avg Answer-Term Recall"] + [pct(_p(s)["avg_term_recall"]) for s in systems],
        ["Precision@K (Signal / Total K)"] + [pct(_p(s)["precision"]) for s in systems],
        ["MRR (Answer-Fact)"] + [f"{_p(s)['mrr']:.2f}" for s in systems],
        ["nDCG@K (Answer-Fact)"] + [f"{_p(s)['ndcg']:.2f}" for s in systems],
        ["Est. API Requests / Question"] + [f"{metrics_by_sys[s]['avg_requests']:.1f}" for s in systems],
        ["Measured Cost / Question ($)"] + [f"${metrics_by_sys[s]['usd_per_q']:.6f}" for s in systems],
        ["Median Retrieval Time (s)"] + [f"{metrics_by_sys[s]['median_seconds']:.2f}s" for s in systems],
    ]
    md.append(_md_table(comp_headers, comp_rows))
    md.append("")

    # Table 2.2: Pre-Grading vs Post-Grading comparison for CRAG pipelines
    crag_systems = [s for s in systems if metrics_by_sys[s]["is_crag"] and metrics_by_sys[s]["grader"]]
    if crag_systems:
        md.append("### Table 2.2: CRAG Grader Impact — Raw Retrieval (Pre-Grading) vs Filtered (Post-Grading)")
        crag_rows = []
        for s in crag_systems:
            pm = metrics_by_sys[s]
            pre, post, g = pm["pre"], pm["post"], pm["grader"]
            crag_rows.append([
                pm["label"],
                f"{pre['avg_k']:.1f} -> {post['avg_k']:.1f}",
                f"{pct(pre['page_hit_rate'])} -> {pct(post['page_hit_rate'])}",
                f"{pct(pre['fact_hit_rate'])} -> {pct(post['fact_hit_rate'])}",
                f"{pct(pre['precision'])} -> {pct(post['precision'])}",
                f"{pre['mrr']:.2f} -> {post['mrr']:.2f}",
                pct(g["noise_removed_rate"]),
                pct(g["false_drop_rate"]),
                f"-{len(g['lost_qids'])}" + (f" / +{len(g['gained_qids'])}" if s == "crag" else ""),
            ])
        md.append(_md_table(
            ["Pipeline", "Avg K (Pre->Post)", "Page Hit", "Fact Hit", "Precision@K", "MRR", "Noise Removed", "False Drops", "Qs Lost / Gained"],
            crag_rows,
        ))
        md.append("")

    # Key Findings
    md.append("### Key Findings")
    for lead, text in _build_key_findings(metrics_by_sys):
        md.append(f"- **{lead.strip()}** {text}")
    md.append("")

    # Section 3: Metrics Definitions
    md.append("## 3. Metrics & Formulas")
    md.append(_md_table(
        ["Metric", "What it measures", "Formula"],
        [
            ["Document Accuracy", "At least one retrieved chunk comes from the target PDF", "Doc hits ÷ N"],
            ["Document Precision", "Share of retrieved chunks belonging to the target PDF (1 - cross-doc bleed)", "Mean(right-PDF chunks ÷ K)"],
            ["Page Hit Rate@K", "At least one retrieved chunk spans the exact ground-truth PDF page", "Page hits ÷ N"],
            ["Page MRR", "Ranking position of the first chunk from the exact ground-truth page", "Mean(1 ÷ page_rank), miss = 0"],
            ["Answer-Fact Hit Rate@K", "At least one retrieved chunk contains verified ground-truth answer evidence", "Fact hits ÷ N"],
            ["Full-Answer Coverage", "Combined target-PDF chunks cover >=65% of verified answer terms & >=80% critical numbers", "Fully covered ÷ N"],
            ["Precision@K", "Fraction of retrieved chunks that actually contain relevant answer evidence", "Mean(relevant chunks ÷ K)"],
            ["MRR (Answer-Fact)", "How high the first relevant answer-evidence chunk is ranked", "Mean(1 ÷ fact_rank), miss = 0"],
            ["nDCG@K", "Ranking quality across all relevant chunks in top-K (higher = relevant chunks at top)", "Mean(DCG@K ÷ IDCG@K)"],
            ["Noise Removed % (CRAG)", "Irrelevant pre-grading candidates correctly dropped by the CRAG grader", "Dropped noise ÷ total pre noise"],
            ["False Drop Rate % (CRAG)", "Relevant pre-grading candidates mistakenly dropped by the CRAG grader", "Dropped relevant ÷ total pre relevant"],
        ],
    ))
    md.append("")

    # Sections 4+: Per-Pipeline Detailed Breakdown
    for sec_idx, s in enumerate(systems, start=4):
        pm = metrics_by_sys[s]
        st = "post" if pm["is_crag"] else "pre"
        m = pm[st]
        rows = pm["rows"]
        n = m["n"]
        it = _interpret_pipeline(pm)

        md.append(f"## {sec_idx}. {pm['label']}")
        md.append("")
        md.append(f"### Table {sec_idx}.1: Setup")
        md.append(_md_table(["Item", "Value"], PIPELINE_SETUP[s] + [["Questions Evaluated", f"{n} ({pm['n_errors']} errors)"]]))
        md.append("")

        md.append(f"### Table {sec_idx}.2: Metrics, Calculations & Interpretation")
        rr_txt = _rr_breakdown(rows, st, "fact_rank")
        calc_rows = [
            ["Document Accuracy", f"{m['doc_hit_count']} ÷ {n} = {pct(m['doc_hit_rate'])}", it["doc_hit"]],
            ["Document Precision", f"Mean right-PDF share = {pct(m['doc_precision'])}", it["doc_prec"]],
            ["Page Hit Rate@K", f"{m['page_hit_count']} ÷ {n} = {pct(m['page_hit_rate'])}", it["page_hit"]],
            ["Page MRR", f"{m['page_rr_sum']:.2f} ÷ {n} = {m['page_mrr']:.2f}", f"Ranks: {_rr_breakdown(rows, st, 'page_rank')}"],
            ["Answer-Fact Hit Rate@K", f"{m['fact_hit_count']} ÷ {n} = {pct(m['fact_hit_rate'])}", it["fact_hit"]],
            ["Full-Answer Coverage", f"{m['full_cov_count']} ÷ {n} = {pct(m['full_coverage'])}", it["coverage"]],
            ["Avg Answer-Term Recall", f"Mean combined term recall = {pct(m['avg_term_recall'])}", "Fraction of verified answer terms present in context"],
            ["Precision@K", f"Mean relevant/K (avg K={m['avg_k']:.1f}) = {pct(m['precision'])}", it["precision"]],
            ["MRR (Answer-Fact)", f"{m['fact_rr_sum']:.2f} ÷ {n} = {m['mrr']:.2f} ({rr_txt})", it["mrr"]],
            ["nDCG@K", f"Mean nDCG@K = {m['ndcg']:.2f}", it["ndcg"]],
            ["Requests / Cost / Latency",
             f"{pm['avg_requests']:.1f} req/q · ${pm['usd_per_q']:.6f}/q (total ${pm['usd_total']:.5f}) · median {pm['median_seconds']:.2f}s",
             f"Slowest: Q{pm['slowest']['q']['id']} ({pm['slowest']['seconds']:.1f}s)" if pm["slowest"] else ""],
        ]
        md.append(_md_table(["Metric", "Calculation & Result", "Interpretation"], calc_rows))
        md.append("")

        # Per-Document Breakdown
        md.append(f"### Table {sec_idx}.3: Results by Document (20 questions each)")
        drows_tbl = []
        for dkey, info in EVAL_DOCS.items():
            sub = [x for x in rows if x["q"]["doc"] == dkey]
            if not sub:
                continue
            dn = len(sub)
            drows_tbl.append([
                info["label"],
                dn,
                pct(sum(1 for x in sub if x[st]["doc_hit"]) / dn),
                pct(sum(1 for x in sub if x[st]["page_hit"]) / dn),
                pct(sum(1 for x in sub if x[st]["fact_hit"]) / dn),
                pct(sum(1 for x in sub if x[st]["full_coverage"]) / dn),
                pct(sum(x[st]["precision"] for x in sub) / dn),
                f"{sum(x[st]['fact_rr'] for x in sub) / dn:.2f}",
                f"{sum(x[st]['ndcg'] for x in sub) / dn:.2f}",
            ])
        md.append(_md_table(
            ["Document", "Qs", "Doc Hit", "Page Hit", "Fact Hit", "Coverage", "Precision@K", "MRR", "nDCG@K"],
            drows_tbl,
        ))
        md.append("")

        # Per-Category Breakdown
        md.append(f"### Table {sec_idx}.4: Results by Question Category")
        crows_tbl = []
        for cat in CATEGORY_ORDER:
            sub = [x for x in rows if x["q"]["category"] == cat]
            if not sub:
                continue
            cn = len(sub)
            crows_tbl.append([
                cat,
                cn,
                pct(sum(1 for x in sub if x[st]["page_hit"]) / cn),
                pct(sum(1 for x in sub if x[st]["fact_hit"]) / cn),
                pct(sum(1 for x in sub if x[st]["full_coverage"]) / cn),
                pct(sum(x[st]["precision"] for x in sub) / cn),
                f"{sum(x[st]['fact_rr'] for x in sub) / cn:.2f}",
            ])
        md.append(_md_table(["Category", "Qs", "Page Hit", "Fact Hit", "Coverage", "Precision@K", "MRR"], crows_tbl))
        md.append("")

        t_num = 5
        # Ext CRAG Corrective Status Table (mirrors Table 5 of Corrective_RAG_Evaluation_Tables.docx)
        if s == "crag" and pm["grader"] and pm["grader"]["status_counts"]:
            md.append(f"### Table {sec_idx}.{t_num}: Corrective Status Breakdown (Ext CRAG)")
            t_num += 1
            meanings = {
                "CORRECT": "Round-1 retrieval after reference filter & grading was sufficient",
                "CORRECTED": "Round-2 query rewriting / per-document search succeeded",
                "PARTIAL": "Some relevant chunks kept, but grader marked parts still missing",
                "NOT FOUND": "No chunks survived grading",
            }
            st_rows = [
                [st_name, len(qids), pct(len(qids) / n), _qlist(qids, 10), meanings[st_name]]
                for st_name, qids in pm["grader"]["status_counts"].items()
            ]
            md.append(_md_table(["Status", "Count", "Share", "Questions", "Meaning"], st_rows))
            md.append(f"*Reference-list chunks filtered before grading: {pm['grader']['total_refs_removed']} | Round-2 triggered on {pm['grader']['round2_count']}/{n} questions.*")
            md.append("")

        # Similarity Score Analysis (mirrors Table 6 of Corrective_RAG_Evaluation_Tables.docx)
        hit_sims = [x[st]["top_sim"] for x in rows if x[st]["fact_hit"] and x[st]["top_sim"] is not None]
        miss_sims = [x[st]["top_sim"] for x in rows if not x[st]["fact_hit"] and x[st]["top_sim"] is not None]
        if hit_sims or miss_sims:
            md.append(f"### Table {sec_idx}.{t_num}: Top-1 Cosine Similarity Analysis")
            md.append(_md_table(
                ["Group", "Questions", "Avg Top-1 Cosine Similarity"],
                [
                    ["Answer-Fact Hit", len(hit_sims), f"{sum(hit_sims)/len(hit_sims):.3f}" if hit_sims else "—"],
                    ["Answer-Fact Miss", len(miss_sims), f"{sum(miss_sims)/len(miss_sims):.3f}" if miss_sims else "—"],
                ],
            ))
            md.append("")

    # Limitations
    md.append(f"## {4 + len(systems)}. Limitations & Important Code Findings")
    md.append(
        "- **Production Bug Found in `corrective_hybrid.py` & `corrective_parent_child.py` Grader**: "
        "Both files call `chat.completions.create(..., max_tokens=5)` without disabling reasoning. "
        "With `CHAT_MODEL=nvidia/nemotron-3.5-super-vl:free` (which enables reasoning by default), all 5 output tokens "
        "are consumed as reasoning tokens and `message.content` returns `None`, triggering `except Exception: return True` "
        "(accepting 100% of chunks with zero filtering). In this retrieval evaluation, `extra_body={\"reasoning\": {\"enabled\": False}}` "
        "was enabled on the grader call (matching `ext_rag/rag_common.py`) so real grading behavior could be measured.\n"
        "- **Deterministic Relevance Matching**: Answer-Fact Hit@K and Full-Answer Coverage match verified ground-truth "
        "answer terms/numbers and exact PDF pages (via 8-word text anchors). A chunk conveying an answer using completely "
        "different synonyms on a non-target page may be under-counted, whereas an on-page chunk mentioning key words without "
        "stating the full relation may be generously counted.\n"
        "- **No Answer LLM Calls**: By design, this evaluation stops strictly before the final Answer LLM so retrieval "
        "quality is isolated from LLM synthesis and hallucination.\n"
        "- **Query Embedding Cache**: Identical query strings reuse cached vectors from the exact same `EMBEDDING_MODEL` "
        "across pipelines so free-tier quotas are preserved; retrieval outputs are mathematically identical."
    )
    md.append("")

    # Appendix: Per-Question Matrix across all 80 questions
    md.append("## Appendix: Per-Question Retrieval Matrix (All 80 Questions)")
    app_headers = ["#", "Doc", "PDF p.", "Category", "Question (short)"] + [SHORT_LABEL[s] for s in systems]
    app_rows = []
    row_lookup = {s: {x["q"]["id"]: x for x in metrics_by_sys[s]["rows"]} for s in systems}
    for q in dataset:
        qid = q["id"]
        line = [qid, q["doc"], ",".join(map(str, q["pdf_pages"])), q["category"], q["short"]]
        for s in systems:
            x = row_lookup[s].get(qid)
            if not x:
                line.append("ERR")
                continue
            if metrics_by_sys[s]["is_crag"]:
                pr = x["pre"]["fact_rank"] or "-"
                po = x["post"]["fact_rank"] or "-"
                cell = f"pre:{pr} / post:{po}"
                if x["lost_by_grader"]:
                    cell += " (LOST)"
                elif x["gained_by_correction"]:
                    cell += " (+R2)"
                line.append(cell)
            else:
                rk = x["pre"]["fact_rank"]
                pg = "P" if x["pre"]["page_hit"] else "-"
                line.append(f"#{rk} ({pg})" if rk else f"MISS ({pg})")
        app_rows.append(line)
    md.append(_md_table(app_headers, app_rows))
    md.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")


# ---------- Word (.docx) Report Generator (styled after Corrective_RAG_Evaluation_Tables.docx) ----------
def _shade(cell, fill: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    for k, v in (("w:val", "clear"), ("w:color", "auto"), ("w:fill", fill)):
        shd.set(qn(k), v)
    tcPr.append(shd)


def _borders(t, color: str = "A6A6A6"):
    tblPr = t._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{side}")
        for k, v in (("w:val", "single"), ("w:sz", "4"), ("w:space", "0"), ("w:color", color)):
            e.set(qn(k), v)
        b.append(e)
    tblPr.append(b)


def _docx_table(doc, headers: list[str], rows: list[list], widths: list[float], size: float = 8.5):
    t = doc.add_table(rows=1, cols=len(headers))
    _borders(t)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(str(h))
        r.bold, r.font.size = True, Pt(size)
        _shade(c, "D9E2F3")
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
    usable = (sec.page_width - sec.left_margin - sec.right_margin) / 360000  # cm
    scale = usable / sum(widths)
    for row in t.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = Cm(w * scale)
    doc.add_paragraph()
    return t


def _docx_heading(doc, text: str, level: int):
    style = next((st for st in doc.styles if st.name in (f"Heading {level}", f"heading {level}")), None)
    return doc.add_paragraph(text, style=style)


def _docx_caption(doc, text: str):
    p = doc.add_paragraph(text)
    p.runs[0].italic = True
    p.paragraph_format.keep_with_next = True


def _clear_body(doc):
    body = doc.element.body
    for el in list(body):
        if el.tag != qn("w:sectPr"):
            body.remove(el)


def write_docx_report(
    out_path: Path,
    dataset: list[dict],
    metrics_by_sys: dict[str, dict],
    store_counts: dict,
    per_file: bool,
):
    systems = list(metrics_by_sys.keys())
    doc = Document(str(TEMPLATE_DOCX)) if TEMPLATE_DOCX.exists() else Document()
    _clear_body(doc)

    embed_model = os.getenv("EMBEDDING_MODEL", "unknown")
    chat_model = os.getenv("CHAT_MODEL", "unknown")
    scope_label = "Per-file filtered search" if per_file else "Global search across all 4 PDFs simultaneously"

    doc.add_paragraph("RAG Retrieval Evaluation: All Seven Pipelines", style="Title")
    doc.add_paragraph(
        f"Pre-Answer-LLM retrieval evaluation of all seven RAG pipelines across {len(dataset)} test questions "
        f"from four PDF documents (Battery Technology, Bishop PRML, Cultured Meat, and AlexNet). "
        f"For the three Corrective pipelines, both Pre-Grading (raw candidates) and Post-Grading "
        f"(filtered chunks handed to the Answer LLM) are reported."
    )

    # 1. Common Setup
    _docx_heading(doc, "1. Common setup", 1)
    _docx_caption(doc, "Table 1: Evaluation setup shared across all pipelines")
    _docx_table(doc, ["Item", "Value"], [
        ["Documents (4 PDFs)", "; ".join(f"{info['label']} ({info['pdf']})" for info in EVAL_DOCS.values())],
        ["Chunks per Store",
         f"Standard (global_hybrid_collection): {sum(store_counts['hybrid'].values()):,} | "
         f"Ext (ext_chunks_openrouter): {sum(store_counts['ext'].values()):,} | "
         f"Hierarchical (parents / children): {sum(store_counts['parent'].values()):,} / {sum(store_counts['child'].values()):,}"],
        ["Embedding Model", f"{embed_model} via OpenRouter"],
        ["Grader LLM (CRAG only)", f"{chat_model} via OpenRouter (final Answer LLM NOT called)"],
        ["Search Scope", scope_label],
        ["Test Questions", f"{len(dataset)}: 20 Battery, 20 Bishop PRML, 20 Cultured Meat, 20 AlexNet"],
        ["Relevance Criteria",
         "Dual deterministic check: (1) Exact PDF & Page match via 8-word anchor locator; "
         "(2) Answer-Fact match verifying key answer terms & numbers inside chunk text."],
    ], [3.8, 12.8], size=8.5)

    # 2. Comparison Table
    _docx_heading(doc, "2. Comparison of all pipelines", 1)
    _docx_caption(doc, "Table 2.1: Retrieval metrics on final chunks handed to the Answer LLM (80 questions)")

    def _p(s):
        return metrics_by_sys[s]["post"] if metrics_by_sys[s]["is_crag"] else metrics_by_sys[s]["pre"]

    comp_rows = [
        ["Avg Chunks (K)"] + [f"{_p(s)['avg_k']:.1f}" for s in systems],
        ["Document Accuracy"] + [pct(_p(s)["doc_hit_rate"]) for s in systems],
        ["Document Precision"] + [pct(_p(s)["doc_precision"]) for s in systems],
        ["Page Hit Rate@K"] + [pct(_p(s)["page_hit_rate"]) for s in systems],
        ["Page MRR"] + [f"{_p(s)['page_mrr']:.2f}" for s in systems],
        ["Answer-Fact Hit@K"] + [pct(_p(s)["fact_hit_rate"]) for s in systems],
        ["Full-Answer Coverage"] + [pct(_p(s)["full_coverage"]) for s in systems],
        ["Precision@K"] + [pct(_p(s)["precision"]) for s in systems],
        ["MRR (Answer-Fact)"] + [f"{_p(s)['mrr']:.2f}" for s in systems],
        ["nDCG@K"] + [f"{_p(s)['ndcg']:.2f}" for s in systems],
        ["Est. Requests / q"] + [f"{metrics_by_sys[s]['avg_requests']:.1f}" for s in systems],
        ["Cost / q ($)"] + [f"${metrics_by_sys[s]['usd_per_q']:.6f}" for s in systems],
        ["Median Time (s)"] + [f"{metrics_by_sys[s]['median_seconds']:.2f}s" for s in systems],
    ]
    _docx_table(
        doc,
        ["Metric"] + [SHORT_LABEL[s] for s in systems],
        comp_rows,
        [3.6] + [13.0 / max(1, len(systems))] * len(systems),
        size=7.5,
    )

    crag_systems = [s for s in systems if metrics_by_sys[s]["is_crag"] and metrics_by_sys[s]["grader"]]
    if crag_systems:
        _docx_caption(doc, "Table 2.2: CRAG Grader Impact (Pre-Grading raw retrieval -> Post-Grading filtered)")
        c_rows = []
        for s in crag_systems:
            pm = metrics_by_sys[s]
            pre, post, g = pm["pre"], pm["post"], pm["grader"]
            c_rows.append([
                pm["label"],
                f"{pre['avg_k']:.1f} -> {post['avg_k']:.1f}",
                f"{pct(pre['fact_hit_rate'])} -> {pct(post['fact_hit_rate'])}",
                f"{pct(pre['precision'])} -> {pct(post['precision'])}",
                f"{pre['mrr']:.2f} -> {post['mrr']:.2f}",
                pct(g["noise_removed_rate"]),
                pct(g["false_drop_rate"]),
                f"-{len(g['lost_qids'])}" + (f" / +{len(g['gained_qids'])}" if s == "crag" else ""),
            ])
        _docx_table(
            doc,
            ["Pipeline", "Avg K", "Fact Hit", "Precision@K", "MRR", "Noise Rem.", "False Drops", "Lost/Gained"],
            c_rows,
            [3.2, 1.8, 2.2, 2.2, 1.8, 1.8, 1.8, 1.8],
            size=7.5,
        )

    _docx_caption(doc, "Key findings")
    bullet_style = next((st for st in doc.styles if st.name in ("List Bullet", "list bullet")), None)
    for lead, text in _build_key_findings(metrics_by_sys):
        p = doc.add_paragraph(style=bullet_style)
        r = p.add_run(lead)
        r.bold = True
        p.add_run(text)

    # 3. Metrics & Formulas
    _docx_heading(doc, "3. Evaluation metrics and formulas", 1)
    _docx_caption(doc, "Table 3: Retrieval metrics definitions and formulas")
    _docx_table(doc, ["Metric", "What it measures", "Formula"], [
        ["Document Accuracy", "At least one chunk comes from the target PDF", "Doc hits ÷ N"],
        ["Document Precision", "Fraction of retrieved chunks from the target PDF", "Mean(right-PDF chunks ÷ K)"],
        ["Page Hit Rate@K", "At least one chunk spans the exact ground-truth PDF page", "Page hits ÷ N"],
        ["Page MRR", "Rank of the first chunk from the ground-truth page", "Mean(1 ÷ page_rank)"],
        ["Answer-Fact Hit@K", "At least one chunk contains verified ground-truth answer evidence", "Fact hits ÷ N"],
        ["Full-Answer Coverage", "Combined right-PDF chunks cover >=65% answer terms & >=80% critical numbers", "Fully covered ÷ N"],
        ["Precision@K", "Fraction of returned chunks that contain relevant answer evidence", "Mean(relevant ÷ K)"],
        ["MRR (Answer-Fact)", "Rank of the first relevant answer-evidence chunk", "Mean(1 ÷ fact_rank)"],
        ["nDCG@K", "Position-weighted ranking quality across top-K", "Mean(DCG@K ÷ IDCG@K)"],
        ["Noise Removed %", "Irrelevant pre-grading candidates removed by CRAG grader", "Dropped noise ÷ pre noise"],
        ["False Drop Rate %", "Relevant pre-grading candidates mistakenly dropped by CRAG grader", "Dropped rel ÷ pre rel"],
    ], [3.5, 8.2, 4.9], size=8.0)

    # 4+. Per-Pipeline Sections
    for sec_idx, s in enumerate(systems, start=4):
        pm = metrics_by_sys[s]
        st = "post" if pm["is_crag"] else "pre"
        m = pm[st]
        rows = pm["rows"]
        n = m["n"]
        it = _interpret_pipeline(pm)

        _docx_heading(doc, f"{sec_idx}. {pm['label']}", 1)
        _docx_caption(doc, f"Table {sec_idx}.1: Evaluation setup")
        _docx_table(doc, ["Item", "Value"], PIPELINE_SETUP[s] + [["Evaluated", f"{n} questions ({pm['n_errors']} errors)"]], [4.2, 12.4], size=8.5)

        _docx_caption(doc, f"Table {sec_idx}.2: Results, calculations and interpretation")
        _docx_table(doc, ["Metric", "Calculation & Result", "Interpretation"], [
            ["Document Accuracy", f"{m['doc_hit_count']} ÷ {n} = {pct(m['doc_hit_rate'])}", it["doc_hit"]],
            ["Document Precision", f"{pct(m['doc_precision'])}", it["doc_prec"]],
            ["Page Hit Rate@K", f"{m['page_hit_count']} ÷ {n} = {pct(m['page_hit_rate'])}", it["page_hit"]],
            ["Page MRR", f"{m['page_rr_sum']:.2f} ÷ {n} = {m['page_mrr']:.2f}", _rr_breakdown(rows, st, "page_rank")],
            ["Answer-Fact Hit@K", f"{m['fact_hit_count']} ÷ {n} = {pct(m['fact_hit_rate'])}", it["fact_hit"]],
            ["Full-Answer Coverage", f"{m['full_cov_count']} ÷ {n} = {pct(m['full_coverage'])}", it["coverage"]],
            ["Precision@K", f"{pct(m['precision'])} (avg K={m['avg_k']:.1f})", it["precision"]],
            ["MRR (Answer-Fact)", f"{m['fact_rr_sum']:.2f} ÷ {n} = {m['mrr']:.2f}", it["mrr"]],
            ["nDCG@K", f"{m['ndcg']:.2f}", it["ndcg"]],
            ["Requests & Cost", f"{pm['avg_requests']:.1f} req/q · ${pm['usd_per_q']:.6f}/q", f"Median {pm['median_seconds']:.2f}s/q"],
        ], [3.4, 5.4, 7.8], size=8.0)

        _docx_caption(doc, f"Table {sec_idx}.3: Results by document")
        drows = []
        for dkey, info in EVAL_DOCS.items():
            sub = [x for x in rows if x["q"]["doc"] == dkey]
            if not sub:
                continue
            dn = len(sub)
            drows.append([
                info["label"], dn,
                pct(sum(1 for x in sub if x[st]["doc_hit"]) / dn),
                pct(sum(1 for x in sub if x[st]["page_hit"]) / dn),
                pct(sum(1 for x in sub if x[st]["fact_hit"]) / dn),
                pct(sum(1 for x in sub if x[st]["full_coverage"]) / dn),
                pct(sum(x[st]["precision"] for x in sub) / dn),
                f"{sum(x[st]['fact_rr'] for x in sub) / dn:.2f}",
                f"{sum(x[st]['ndcg'] for x in sub) / dn:.2f}",
            ])
        _docx_table(doc, ["Document", "Qs", "Doc", "Page", "Fact", "Cov.", "Prec@K", "MRR", "nDCG"],
                    drows, [3.6, 1.0, 1.6, 1.6, 1.6, 1.6, 1.8, 1.8, 1.8], size=8.0)

        _docx_caption(doc, f"Table {sec_idx}.4: Results by question category")
        crows = []
        for cat in CATEGORY_ORDER:
            sub = [x for x in rows if x["q"]["category"] == cat]
            if not sub:
                continue
            cn = len(sub)
            crows.append([
                cat, cn,
                pct(sum(1 for x in sub if x[st]["page_hit"]) / cn),
                pct(sum(1 for x in sub if x[st]["fact_hit"]) / cn),
                pct(sum(1 for x in sub if x[st]["full_coverage"]) / cn),
                pct(sum(x[st]["precision"] for x in sub) / cn),
                f"{sum(x[st]['fact_rr'] for x in sub) / cn:.2f}",
            ])
        _docx_table(doc, ["Category", "Qs", "Page Hit", "Fact Hit", "Coverage", "Precision@K", "MRR"],
                    crows, [4.2, 1.0, 2.2, 2.2, 2.2, 2.4, 2.2], size=8.0)

        t_num = 5
        if s == "crag" and pm["grader"] and pm["grader"]["status_counts"]:
            _docx_caption(doc, f"Table {sec_idx}.{t_num}: Corrective status distribution (Ext CRAG)")
            t_num += 1
            st_rows = [
                [st_name, len(qids), pct(len(qids) / n), _qlist(qids, 8)]
                for st_name, qids in pm["grader"]["status_counts"].items()
            ]
            _docx_table(doc, ["Status", "Count", "Share", "Questions"], st_rows, [2.8, 1.6, 2.0, 10.2], size=8.0)

        hit_sims = [x[st]["top_sim"] for x in rows if x[st]["fact_hit"] and x[st]["top_sim"] is not None]
        miss_sims = [x[st]["top_sim"] for x in rows if not x[st]["fact_hit"] and x[st]["top_sim"] is not None]
        if hit_sims or miss_sims:
            _docx_caption(doc, f"Table {sec_idx}.{t_num}: Similarity of the best retrieved chunk")
            _docx_table(doc, ["Group", "Questions", "Average top-1 cosine similarity"], [
                ["Answer-Fact Hit", len(hit_sims), f"{sum(hit_sims)/len(hit_sims):.3f}" if hit_sims else "—"],
                ["Answer-Fact Miss", len(miss_sims), f"{sum(miss_sims)/len(miss_sims):.3f}" if miss_sims else "—"],
            ], [4.5, 3.0, 9.1], size=8.0)

    # Appendix: Per-Question Matrix
    _docx_heading(doc, "Appendix: Per-question retrieval matrix (80 questions)", 1)
    _docx_caption(doc, "Table A.1: First Answer-Fact rank per question across all pipelines")
    row_lookup = {s: {x["q"]["id"]: x for x in metrics_by_sys[s]["rows"]} for s in systems}
    app_rows = []
    for q in dataset:
        qid = q["id"]
        line = [qid, q["doc"], ",".join(map(str, q["pdf_pages"])), q["short"]]
        for s in systems:
            x = row_lookup[s].get(qid)
            if not x:
                line.append("ERR")
                continue
            if metrics_by_sys[s]["is_crag"]:
                pr = x["pre"]["fact_rank"] or "-"
                po = x["post"]["fact_rank"] or "-"
                tag = "!" if x["lost_by_grader"] else ("+" if x["gained_by_correction"] else "")
                line.append(f"{pr}->{po}{tag}")
            else:
                rk = x["pre"]["fact_rank"]
                line.append(f"#{rk}" if rk else "MISS")
        app_rows.append(line)
    _docx_table(
        doc,
        ["#", "Doc", "PDF p.", "Question"] + [SHORT_LABEL[s] for s in systems],
        app_rows,
        [0.7, 1.3, 1.1, 4.4] + [9.1 / max(1, len(systems))] * len(systems),
        size=6.5,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))


def generate_reports(per_file: bool = False):
    dataset = load_dataset()
    metrics_by_sys: dict[str, dict] = {}
    for system, _, _ in ALL_PIPELINES:
        p = result_file(system, per_file=per_file)
        if not p.exists():
            continue
        recs = json.loads(p.read_text(encoding="utf-8"))
        if recs:
            metrics_by_sys[system] = compute_pipeline_metrics(system, dataset, recs)

    if not metrics_by_sys:
        print("No retrieval evaluation result files found yet. Run `python -m evaluation.retrieval_eval run` first.")
        return

    store_counts = _store_chunk_counts()
    suffix = "_PerFile" if per_file else ""
    md_out = RESULTS_DIR / f"Retrieval_Evaluation_Report{suffix}.md"
    docx_out = RESULTS_DIR / f"Retrieval_Evaluation_Report{suffix}.docx"
    summary_out = RESULTS_DIR / f"retrieval_summary{suffix.lower()}.json"

    write_markdown_report(md_out, dataset, metrics_by_sys, store_counts, per_file=per_file)
    write_docx_report(docx_out, dataset, metrics_by_sys, store_counts, per_file=per_file)

    # Save compact machine-readable JSON summary (without bulky per-row chunk texts)
    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "per_file": per_file,
        "total_questions": len(dataset),
        "store_counts": store_counts,
        "pipelines": {
            s: {
                "label": pm["label"],
                "is_crag": pm["is_crag"],
                "n_valid": pm["n_valid"],
                "n_errors": pm["n_errors"],
                "pre": pm["pre"],
                "post": pm["post"],
                "grader": pm["grader"],
                "avg_requests": pm["avg_requests"],
                "usd_per_q": pm["usd_per_q"],
                "usd_total": pm["usd_total"],
                "median_seconds": pm["median_seconds"],
            }
            for s, pm in metrics_by_sys.items()
        },
    }
    summary_out.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Retrieval Evaluation Summary ===")
    for s, pm in metrics_by_sys.items():
        st = "post" if pm["is_crag"] else "pre"
        m = pm[st]
        extra = ""
        if pm["is_crag"] and pm["grader"]:
            extra = f" | pre->post fact: {pct(pm['pre']['fact_hit_rate'])}->{pct(m['fact_hit_rate'])}, noise_rem={pct(pm['grader']['noise_removed_rate'])}, false_drop={pct(pm['grader']['false_drop_rate'])}"
        print(
            f"{pm['label']:32} n={m['n']:2d} | doc={pct(m['doc_hit_rate']):>5} "
            f"page={pct(m['page_hit_rate']):>5} fact={pct(m['fact_hit_rate']):>5} "
            f"cov={pct(m['full_coverage']):>5} prec={pct(m['precision']):>5} "
            f"mrr={m['mrr']:.2f} ndcg={m['ndcg']:.2f}{extra}"
        )
    print(f"\nSaved Markdown: {md_out}")
    print(f"Saved Word:     {docx_out}")
    print(f"Saved Summary:  {summary_out}")


if __name__ == "__main__":
    generate_reports()
