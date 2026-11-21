"""
Unified 80-question retrieval evaluation dataset built from the four Q&A files in docs/.

Each question is parsed from docs/*Q & A.txt, mapped to its exact 1-based PDF page(s)
and validated against the actual PDF text using a generalized 4-PDF word-anchor locator
(same mechanism as evaluation/locate.py, extended with ligature & hyphen normalization).
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
RESULTS_DIR = ROOT / "evaluation" / "results"
DATASET_JSON = RESULTS_DIR / "retrieval_dataset_80.json"

EVAL_DOCS = {
    "battery": {
        "label": "Battery Technology",
        "pdf": "HummesETAL_Lajer_2023-2.pdf",
        "qa_file": "Battery Technology Q & A.txt",
        "page_offset": -93,  # Journal page 94 -> PDF page 1
    },
    "bishop": {
        "label": "Bishop PRML",
        "pdf": "Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf",
        "qa_file": "Bishop Q & A.txt",
        "page_offset": 20,   # Book page 1 -> PDF page 21
    },
    "meat": {
        "label": "Cultured Meat",
        "pdf": "Advice For Risk Associated with Cultured Meat.pdf",
        "qa_file": "Cultured Meat Q & A.txt",
        "page_offset": 0,    # Exact PDF pages
    },
    "alexnet": {
        "label": "Neural Network (AlexNet)",
        "pdf": "imagenet-classification-with-deep-convolutional-neural-networks-Paper.pdf",
        "qa_file": "Nueral Network Q & A.txt",
        "page_offset": 0,    # Exact PDF pages
    },
}

SOURCE_TO_DOC = {info["pdf"]: key for key, info in EVAL_DOCS.items()}

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for", "of", "by",
    "with", "from", "as", "into", "about", "that", "this", "these", "those", "it", "its",
    "they", "them", "their", "we", "our", "you", "your", "he", "she", "his", "her", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "can", "not", "no", "nor",
    "so", "than", "then", "too", "very", "just", "only", "also", "such", "more", "most",
    "less", "least", "much", "many", "some", "any", "all", "each", "every", "both", "few",
    "other", "another", "same", "which", "who", "whom", "whose", "what", "when", "where",
    "why", "how", "while", "whereas", "during", "between", "through", "over", "under",
    "above", "below", "up", "down", "out", "off", "after", "before", "because", "since",
    "until", "although", "though", "whether", "either", "neither", "there", "here", "one",
    "two", "three", "four", "five", "used", "using", "use", "uses", "based", "given",
    "called", "known", "made", "make", "making", "per", "via", "etc", "eg", "ie",
    "approximately", "around", "about", "usually", "typically", "generally", "mainly",
    "primarily", "specifically", "particularly", "strictly", "directly", "exact", "exactly",
    "different", "new", "first", "second", "third", "final", "initial", "main", "major",
    "primary", "specific", "certain", "various", "several", "number", "value", "values",
    "set", "data", "model", "models", "method", "approach", "problem", "process", "system",
    "study", "paper", "network", "cells", "cell", "battery", "batteries", "meat", "cultured",
}

ANCHOR = 8


def normalize_text(text: str) -> str:
    """Normalizes ligatures, unicode math/punctuation, LaTeX commands, and line-break hyphens."""
    if not text:
        return ""
    # Common PDF ligatures
    for src, dst in (
        ("\ufb00", "ff"), ("\ufb01", "fi"), ("\ufb02", "fl"),
        ("\ufb03", "ffi"), ("\ufb04", "ffl"),
        ("\u2013", "-"), ("\u2014", "-"), ("\u2212", "-"),
        ("\u00a0", " "), ("\u00b5", "u"), ("\u03bc", "u"),
    ):
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKC", text)
    # Remove LaTeX commands while keeping inner alphanumeric content
    text = re.sub(r"\\(mathcal|mathbf|mathrm|text|operatorname)\{([^}]*)\}", r" \2 ", text)
    text = re.sub(r"\\(times|propto|vert|sigma|mu|alpha|beta|tanh|cdot|approx|le|ge)\b", " ", text)
    text = text.replace("$", " ")
    # Join hyphenated PDF line wraps (e.g. "over-\nfitting" -> "overfitting" or "over fitting")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return text


def norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z]+|[0-9]+(?:\.[0-9]+)?", normalize_text(text).lower())


def strip_chunk_header(text: str) -> str:
    """Remove '[Source: file | Page: n]:' or '[Source: file]:' prefixes added by pipelines."""
    return re.sub(r"^\s*\[[^\]]*\]\s*:\s*", "", text or "", count=1)


# ---------- 4-PDF Word-Anchor Locator ----------
class EvalDocument:
    def __init__(self, doc_key: str, pdf_path: Path):
        self.key = doc_key
        self.path = pdf_path
        self.page_of: list[int] = []
        self.page_texts: dict[int, str] = {}
        self.page_word_sets: dict[int, set[str]] = {}
        tokens: list[str] = []

        reader = PdfReader(str(pdf_path))
        self.num_pages = len(reader.pages)
        for page_no, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            w = norm_words(raw)
            self.page_texts[page_no] = " ".join(w)
            self.page_word_sets[page_no] = set(w)
            tokens.extend(w)
            self.page_of.extend([page_no] * len(w))

        self.n = len(tokens)
        self.runs: dict[tuple[str, ...], int | None] = {}
        for i in range(max(0, self.n - ANCHOR + 1)):
            run = tuple(tokens[i : i + ANCHOR])
            self.runs[run] = None if run in self.runs else i

    def find(self, chunk_words: list[str]) -> tuple[int, int] | None:
        if len(chunk_words) < ANCHOR:
            return None
        for offset in range(0, len(chunk_words) - ANCHOR + 1):
            start = self.runs.get(tuple(chunk_words[offset : offset + ANCHOR]))
            if start is not None:
                first = max(start - offset, 0)
                return first, min(first + len(chunk_words) - 1, self.n - 1)
        return None


@lru_cache(maxsize=1)
def load_eval_documents() -> dict[str, EvalDocument]:
    return {
        key: EvalDocument(key, DOCS_DIR / info["pdf"])
        for key, info in EVAL_DOCS.items()
    }


def locate_chunk(text: str, hint_source: str | None = None, hint_page: int | None = None) -> dict:
    """
    Locates the true PDF document key and 1-based PDF page list for a chunk.
    Uses unique 8-word anchor matching first (works even for Parent-Child chunks with no page metadata),
    and falls back to chunk metadata source/page if an anchor wasn't unique.
    """
    docs = load_eval_documents()
    w = norm_words(strip_chunk_header(text))
    hint_doc = SOURCE_TO_DOC.get(Path(hint_source).name) if hint_source else None

    search_order = ([hint_doc] if hint_doc in docs else []) + [k for k in docs if k != hint_doc]
    if len(w) >= ANCHOR:
        for key in search_order:
            span = docs[key].find(w)
            if span:
                first, last = span
                pages = sorted(set(docs[key].page_of[first : last + 1]))
                return {"doc": key, "pages": pages, "located_by": "anchor"}

    if hint_doc and hint_page:
        return {"doc": hint_doc, "pages": [int(hint_page)], "located_by": "metadata"}
    return {"doc": hint_doc, "pages": [], "located_by": "none"}


# ---------- Parsers for the 4 Q&A txt files ----------
def _parse_page_numbers(raw_page: str) -> list[int]:
    """Extracts page numbers and expands ranges like '6-7' or '5–6'."""
    cleaned = re.sub(r"\([^)]*van\s+\d+[^)]*\)", "", raw_page, flags=re.I)
    cleaned = cleaned.replace("\u2013", "-").replace("\u2014", "-")
    pages: set[int] = set()
    for a, b in re.findall(r"(\d+)\s*-\s*(\d+)", cleaned):
        lo, hi = int(a), int(b)
        if lo <= hi <= lo + 10:
            pages.update(range(lo, hi + 1))
    cleaned_no_ranges = re.sub(r"\d+\s*-\s*\d+", " ", cleaned)
    for num in re.findall(r"\d+", cleaned_no_ranges):
        pages.add(int(num))
    return sorted(pages)


def _classify_question(question: str, answer: str) -> str:
    q_low = question.lower()
    a_low = answer.lower()
    if any(k in q_low for k in ("how much", "how many", "how long", "percentage", "error rate", "hyperparameter", "values", "kernel size", "batch size", "cost", "weight reduction")):
        return "Numbers / Hyperparameters"
    if any(k in q_low for k in ("difference between", "differ from", "compared to", "versus", "vs", " and ", "both ", "two ", "three ", "four ")):
        if "why" not in q_low[:10]:
            return "Multi-part / Comparison"
    if any(k in q_low for k in ("why ", "how does", "how is", "how were", "how do", "effect did", "what risk", "what primary risk", "what chemical hazards")):
        return "Mechanism / Why"
    if any(k in q_low for k in ("define", "what does the term", "what is the phenomenon", "what is the field", "what is a ", "what are microcarriers", "what is the \"", "what is s-fold", "what is the credit")):
        return "Definition / Concept"
    if re.search(r"\d+(\.\d+)?%", a_low):
        return "Numbers / Hyperparameters"
    return "Single fact"


def _make_short_title(question: str) -> str:
    words_list = question.strip().rstrip("?").split()
    return " ".join(words_list[:7]) + ("..." if len(words_list) > 7 else "")


def _parse_battery_qa(text: str) -> list[dict]:
    items = []
    pattern = re.compile(
        r"Question:\s*(.*?)\s*\nAnswer:\s*(.*?)\s*\nPage:\s*([^\n]+)",
        re.DOTALL,
    )
    for idx, m in enumerate(pattern.finditer(text), start=1):
        items.append({
            "doc_q_num": idx,
            "question": " ".join(m.group(1).split()),
            "answer": " ".join(m.group(2).split()),
            "raw_page": m.group(3).strip(),
        })
    return items


def _parse_bishop_qa(text: str) -> list[dict]:
    items = []
    pattern = re.compile(
        r"(\d+)\.\s*\*\*Question:\*\*\s*(.*?)\s*\*\*Answer:\*\*\s*(.*?)\s*\*\*Page:\*\*\s*([^\n]+)",
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        items.append({
            "doc_q_num": int(m.group(1)),
            "question": " ".join(m.group(2).split()),
            "answer": " ".join(m.group(3).split()),
            "raw_page": m.group(4).strip().rstrip("."),
        })
    return items


def _parse_numbered_qa(text: str) -> list[dict]:
    """Parses Cultured Meat Q & A.txt and Nueral Network Q & A.txt."""
    items = []
    pattern = re.compile(
        r"(?:^|\n)\s*(\d+)\.\s*(.*?)\s*Answer:\s*(.*?)\s*Page Number:\s*([^\n]+)",
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        items.append({
            "doc_q_num": int(m.group(1)),
            "question": " ".join(m.group(2).split()),
            "answer": " ".join(m.group(3).split()),
            "raw_page": m.group(4).strip(),
        })
    return items


def _extract_verified_fact_terms(
    answer: str,
    question: str,
    doc: EvalDocument,
    pdf_pages: list[int],
) -> tuple[list[str], list[str], list[str]]:
    """
    Extracts ground-truth answer terms & numeric tokens verified to actually exist in the
    target PDF page(s) (or adjacent page +/-1 for boundary spans).

    Returns:
      - verified_terms: distinctive answer content words verified on the ground-truth PDF page(s)
      - critical_numbers: key numeric tokens from the answer verified on the ground-truth page(s)
      - verified_bigrams: consecutive 2-word answer phrases verified on the ground-truth page(s)
    """
    # Combine text of target PDF pages (plus immediate neighbors only if needed)
    exact_page_text = " ".join(doc.page_texts.get(p, "") for p in pdf_pages)
    exact_words = set(exact_page_text.split())

    neighbor_pages = sorted({
        p2 for p in pdf_pages for p2 in (p - 1, p, p + 1) if 1 <= p2 <= doc.num_pages
    })
    near_page_text = " ".join(doc.page_texts.get(p, "") for p in neighbor_pages)
    near_words = set(near_page_text.split())

    q_words = set(norm_words(question))
    a_words = norm_words(answer)

    # Prefer terms that appear in the answer, aren't stopwords, and exist in the actual PDF page
    seen: set[str] = set()
    verified_terms: list[str] = []
    critical_numbers: list[str] = []

    for w in a_words:
        if w in seen:
            continue
        is_num = bool(re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", w))
        if is_num:
            if w in near_words and (len(w) >= 2 or w not in {"1", "2", "3", "4", "5", "0"} or w in exact_words):
                critical_numbers.append(w)
                seen.add(w)
                verified_terms.append(w)
            continue
        if len(w) <= 2 or w in STOPWORDS:
            continue
        if w in exact_words or w in near_words:
            seen.add(w)
            verified_terms.append(w)

    # Also build distinctive answer bigrams (pairs of words appearing contiguously in the PDF)
    verified_bigrams: list[str] = []
    seen_bi: set[str] = set()
    padded_near = f" {near_page_text} "
    for w1, w2 in zip(a_words, a_words[1:]):
        if w1 in STOPWORDS and w2 in STOPWORDS:
            continue
        if w1 in q_words and w2 in q_words:
            continue
        bi = f"{w1} {w2}"
        if bi not in seen_bi and f" {bi} " in padded_near:
            seen_bi.add(bi)
            verified_bigrams.append(bi)

    return verified_terms, critical_numbers, verified_bigrams


def build_dataset(save_json: bool = True) -> list[dict]:
    docs = load_eval_documents()
    parsers = {
        "battery": _parse_battery_qa,
        "bishop": _parse_bishop_qa,
        "meat": _parse_numbered_qa,
        "alexnet": _parse_numbered_qa,
    }

    dataset: list[dict] = []
    qid = 1
    for doc_key, info in EVAL_DOCS.items():
        qa_path = DOCS_DIR / info["qa_file"]
        raw = qa_path.read_text(encoding="utf-8")
        parsed = parsers[doc_key](raw)
        if len(parsed) != 20:
            raise ValueError(f"Expected 20 Q&As in {qa_path.name}, found {len(parsed)}")

        doc = docs[doc_key]
        offset = info["page_offset"]

        for item in parsed:
            doc_pages = _parse_page_numbers(item["raw_page"])
            pdf_pages = [
                p + offset for p in doc_pages if 1 <= (p + offset) <= doc.num_pages
            ]
            verified_terms, critical_numbers, verified_bigrams = _extract_verified_fact_terms(
                item["answer"], item["question"], doc, pdf_pages
            )
            dataset.append({
                "id": qid,
                "doc": doc_key,
                "doc_label": info["label"],
                "source_file": info["pdf"],
                "doc_q_num": item["doc_q_num"],
                "short": _make_short_title(item["question"]),
                "category": _classify_question(item["question"], item["answer"]),
                "question": item["question"],
                "answer": item["answer"],
                "raw_page": item["raw_page"],
                "doc_pages": doc_pages,
                "pdf_pages": pdf_pages,
                "verified_terms": verified_terms,
                "critical_numbers": critical_numbers,
                "verified_bigrams": verified_bigrams,
            })
            qid += 1

    if save_json:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        DATASET_JSON.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    return dataset


def load_dataset() -> list[dict]:
    if DATASET_JSON.exists():
        return json.loads(DATASET_JSON.read_text(encoding="utf-8"))
    return build_dataset(save_json=True)


# ---------- Deterministic Chunk Relevance Scoring ----------
def score_chunk(q: dict, chunk_text: str, located_doc: str | None, located_pages: list[int]) -> dict:
    """
    Evaluates a single retrieved chunk against ground truth q:
      - doc_match: True iff chunk comes from the question's target PDF
      - page_match: True iff doc_match and chunk spans at least one ground-truth PDF page
      - term_recall: fraction of verified answer terms found in this chunk
      - bigram_hits: count of verified distinctive answer bigrams found in this chunk
      - number_recall: fraction of verified critical answer numbers found in this chunk
      - fact_match: True iff the chunk genuinely contains the ground-truth answer evidence
                    (not merely sitting on the right page with unrelated text)
    """
    doc_match = located_doc == q["doc"]
    target_pages = set(q["pdf_pages"])
    near_pages = {p2 for p in target_pages for p2 in (p - 1, p, p + 1)}
    loc_set = set(located_pages or [])

    page_match = doc_match and bool(loc_set & target_pages)
    near_page_match = doc_match and bool(loc_set & near_pages)

    w_list = norm_words(strip_chunk_header(chunk_text))
    w_set = set(w_list)
    padded_text = " " + " ".join(w_list) + " "

    v_terms = q["verified_terms"]
    c_nums = q["critical_numbers"]
    v_bigrams = q["verified_bigrams"]

    matched_terms = [t for t in v_terms if t in w_set]
    term_recall = (len(matched_terms) / len(v_terms)) if v_terms else 0.0

    matched_nums = [n for n in c_nums if n in w_set]
    num_recall = (len(matched_nums) / len(c_nums)) if c_nums else 1.0

    matched_bigrams = [b for b in v_bigrams if f" {b} " in padded_text]
    bigram_recall = (len(matched_bigrams) / len(v_bigrams)) if v_bigrams else 0.0

    # Relevance decision: must come from the target document
    fact_match = False
    if doc_match:
        if page_match or near_page_match:
            # On (or straddling) the ground-truth page: require real answer evidence in the chunk
            if not v_terms:
                fact_match = True
            elif len(v_terms) <= 2:
                fact_match = term_recall >= 0.5 and (not c_nums or num_recall >= 0.5)
            else:
                fact_match = (
                    (term_recall >= 0.40 and (not c_nums or num_recall >= 0.50))
                    or (len(matched_bigrams) >= 2 and (not c_nums or num_recall >= 0.50))
                    or (len(matched_bigrams) >= 3 and term_recall >= 0.35)
                    or (bool(c_nums) and num_recall == 1.0 and term_recall >= 0.25)
                )
        else:
            # Another page of the same PDF (e.g. abstract/conclusion repeating the exact answer):
            # require stronger evidence so index/bibliography pages don't count as false positives
            if len(v_terms) <= 3:
                fact_match = term_recall == 1.0 and (not c_nums or num_recall == 1.0)
            else:
                fact_match = (
                    (term_recall >= 0.75 and (not c_nums or num_recall == 1.0))
                    or (bigram_recall >= 0.50 and len(matched_bigrams) >= 2 and (not c_nums or num_recall == 1.0))
                )

    return {
        "doc_match": doc_match,
        "page_match": page_match,
        "fact_match": fact_match,
        "term_recall": round(term_recall, 4),
        "num_recall": round(num_recall, 4),
        "bigram_hits": len(matched_bigrams),
        "matched_terms": matched_terms,
    }


def score_chunk_set_coverage(q: dict, chunks_with_loc: list[dict]) -> dict:
    """
    Evaluates full-answer coverage across the combined set of retrieved chunks from the right PDF.
    A question has full coverage when the retrieved chunks collectively cover the key answer facts
    (>= 70% of verified answer terms AND all critical numbers, or >= 1 chunk with term_recall >= 0.65
    and full critical numbers).
    """
    v_terms = q["verified_terms"]
    c_nums = q["critical_numbers"]

    covered_terms: set[str] = set()
    covered_nums: set[str] = set()
    for c in chunks_with_loc:
        if c.get("doc") != q["doc"]:
            continue
        w_set = set(norm_words(strip_chunk_header(c.get("text", ""))))
        covered_terms.update(t for t in v_terms if t in w_set)
        covered_nums.update(n for n in c_nums if n in w_set)

    combined_term_recall = (len(covered_terms) / len(v_terms)) if v_terms else 1.0
    combined_num_recall = (len(covered_nums) / len(c_nums)) if c_nums else 1.0

    # Full-answer coverage requires at least one fact_match chunk plus strong collective fact recall
    any_fact_hit = any(c.get("scores", {}).get("fact_match") for c in chunks_with_loc)
    full_coverage = bool(
        any_fact_hit
        and combined_term_recall >= 0.65
        and (not c_nums or combined_num_recall >= 0.80)
    )
    return {
        "combined_term_recall": round(combined_term_recall, 4),
        "combined_num_recall": round(combined_num_recall, 4),
        "full_coverage": full_coverage,
    }


if __name__ == "__main__":
    ds = build_dataset(save_json=True)
    docs = load_eval_documents()
    print(f"Built dataset with {len(ds)} questions -> {DATASET_JSON}")

    # Sanity-validate all 80 questions against their ground-truth PDF pages
    gt_ok = 0
    weak = []
    for q in ds:
        doc = docs[q["doc"]]
        gt_text = " ".join(doc.page_texts.get(p, "") for p in q["pdf_pages"])
        res = score_chunk(q, gt_text, q["doc"], q["pdf_pages"])
        if res["fact_match"]:
            gt_ok += 1
        else:
            weak.append((q["id"], q["doc"], q["pdf_pages"], q["verified_terms"], res))

    # Also test negative specificity: check every Q against a random wrong page of the same PDF
    fp = 0
    for q in ds:
        doc = docs[q["doc"]]
        wrong_page = next((p for p in range(doc.num_pages, 0, -1) if p not in q["pdf_pages"] and (p - 1) not in q["pdf_pages"] and (p + 1) not in q["pdf_pages"]), None)
        if wrong_page:
            res = score_chunk(q, doc.page_texts[wrong_page], q["doc"], [wrong_page])
            if res["fact_match"]:
                fp += 1

    print(f"Ground-truth page self-check: {gt_ok}/{len(ds)} passed | Far-page false positives: {fp}/{len(ds)}")
    for w in weak:
        print("  WEAK GT:", w)
