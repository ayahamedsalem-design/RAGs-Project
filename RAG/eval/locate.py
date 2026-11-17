# Find which PDF pages a retrieved chunk comes from, by looking its text up in the document.
#
# The pipelines store page numbers differently (my chunks: "page 12"; Aya's Hybrid chunks:
# "Page: 12"; her Parent-Child chunks: no page at all), so for a fair retrieval score every
# chunk is located the same way: by its own words. The PDF is read page by page into one long
# list of words, each word labelled with its page. A chunk is found by searching a run of 8 of
# its words; its pages are the pages its words span.

import re
from functools import lru_cache
from pathlib import Path

from pypdf import PdfReader

ANCHOR = 8                      # words searched at once


def words(text):
    return re.findall(r"\w+", text.lower())


def strip_header(text):
    """Remove the labels the pipelines put in front of a chunk ("[Source: x | Page: 3]:", ":")."""
    return re.sub(r"^\s*(\[[^\]]*\])?\s*:\s*", "", text, count=1)


class Document:
    def __init__(self, name, path):
        self.name = name
        self.page_of = []                      # page number of every word
        tokens = []
        for page_no, page in enumerate(PdfReader(path).pages, 1):
            w = words(page.extract_text() or "")
            tokens += w
            self.page_of += [page_no] * len(w)
        self.n = len(tokens)
        self.text = " ".join(tokens)           # the whole document, normalised (for evidence checks)
        # Every run of ANCHOR words -> where it starts, or None if it occurs more than once.
        # Only unique runs are used: the running headers and footers repeat on every page
        # ("Hummes et al. | Latin American Journal ...", "Pagina 12 van 33", the download
        # notice), so matching those would give the wrong page.
        self.runs = {}
        for i in range(self.n - ANCHOR + 1):
            run = tuple(tokens[i:i + ANCHOR])
            self.runs[run] = None if run in self.runs else i

    def find(self, chunk_words):
        """(first word, last word) of the chunk in this document, or None."""
        for offset in range(0, max(1, len(chunk_words) - ANCHOR + 1)):
            start = self.runs.get(tuple(chunk_words[offset:offset + ANCHOR]))
            if start is not None:
                first = max(start - offset, 0)
                return first, min(first + len(chunk_words) - 1, self.n - 1)
        return None


@lru_cache(maxsize=None)
def documents(docs_dir):
    """{short name: Document} for the evaluation PDFs."""
    from RAG.eval.common import DOCS
    return {short: Document(short, Path(docs_dir) / name) for short, name in DOCS.items()}


def locate(text, docs_dir):
    """{"doc": short name, "pages": [..]} for a chunk, or {"doc": None, "pages": []}."""
    w = words(strip_header(text))
    if len(w) < 3:
        return {"doc": None, "pages": []}
    for short, doc in documents(str(docs_dir)).items():
        span = doc.find(w)
        if span:
            first, last = span
            return {"doc": short, "pages": sorted(set(doc.page_of[first:last + 1]))}
    return {"doc": None, "pages": []}
