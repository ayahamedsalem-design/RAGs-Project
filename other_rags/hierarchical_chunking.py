import os
import re
import uuid
import logging
from pathlib import Path
from pypdf import PdfReader

from hybrid_rag.logging_setup import setup_logging
from hybrid_rag.pdf_processing import (
    get_valid_pdf_path,
    resolve_file_name,
    is_ocr_enabled,
    ocr_pdf_pages,
    OCR_MIN_PAGE_CHARS,
)

setup_logging()
logger = logging.getLogger(__name__)


def _format_page_range(pages: list[int]) -> str:
    lo, hi = min(pages), max(pages)
    return str(lo) if lo == hi else f"{lo}-{hi}"


def extract_and_chunk_hierarchical_with_pages(
    pdf_path,
    parent_chunk_words: int = 300,
    child_chunk_words: int = 70,
    child_overlap: int = 15,
    use_ocr: bool | None = None,
    ocr_lang: str | None = None,
    ocr_dpi: int | None = None,
):
    """
    Hierarchical chunking splitting text into large parent blocks and smaller overlapping
    child chunks while preserving exact PDF page numbers ('page': '5' or '5-6') on every
    parent and child, with automatic OCR fallback for scanned pages.
    """
    if isinstance(pdf_path, Path):
        pdf_path = str(pdf_path)

    is_existing_path = isinstance(pdf_path, str) or (
        hasattr(pdf_path, "name")
        and isinstance(pdf_path.name, str)
        and os.path.exists(pdf_path.name)
    )
    valid_pdf_path = get_valid_pdf_path(pdf_path)
    file_name = resolve_file_name(pdf_path)

    try:
        reader = PdfReader(valid_pdf_path)
        total_pages = len(reader.pages)
        page_texts: dict[int, str] = {}
        pages_needing_ocr: list[int] = []

        for page_idx, page in enumerate(reader.pages):
            extracted = (page.extract_text() or "").strip()
            page_texts[page_idx] = extracted
            if len(re.sub(r"\s+", "", extracted)) < OCR_MIN_PAGE_CHARS:
                pages_needing_ocr.append(page_idx)

        if pages_needing_ocr and is_ocr_enabled(use_ocr):
            resolved_lang = ocr_lang or os.getenv("OCR_LANG", "ara+eng")
            resolved_dpi = int(ocr_dpi or os.getenv("OCR_DPI", "250"))
            resolved_workers = int(os.getenv("OCR_WORKERS", str(min(4, os.cpu_count() or 2))))
            has_text_pages = len(pages_needing_ocr) < total_pages
            try:
                ocr_texts = ocr_pdf_pages(
                    valid_pdf_path,
                    pages_needing_ocr,
                    ocr_lang=resolved_lang,
                    dpi=resolved_dpi,
                    max_workers=resolved_workers,
                )
                for p_idx, ocr_text in ocr_texts.items():
                    if len(ocr_text.strip()) > len(page_texts[p_idx]):
                        page_texts[p_idx] = ocr_text.strip()
            except RuntimeError as exc:
                if has_text_pages:
                    logger.warning(
                        f"[OCR] Skipping OCR on {len(pages_needing_ocr)} page(s) in '{file_name}': {exc}"
                    )
                else:
                    raise

        word_stream: list[tuple[str, int]] = []
        for page_idx in range(total_pages):
            text = page_texts.get(page_idx, "")
            if not text:
                continue
            for w in text.split():
                word_stream.append((w, page_idx + 1))

        parent_ids, parent_texts, parent_metadatas = [], [], []
        child_ids, child_chunks, child_metadatas = [], [], []

        step = max(1, child_chunk_words - child_overlap)
        for p_idx in range(0, len(word_stream), parent_chunk_words):
            parent_slice = word_stream[p_idx : p_idx + parent_chunk_words]
            if not parent_slice:
                continue

            parent_id = f"parent_{uuid.uuid4().hex[:8]}"
            parent_words = [w for w, _ in parent_slice]
            parent_pages = [pg for _, pg in parent_slice]
            parent_text = " ".join(parent_words)

            parent_ids.append(parent_id)
            parent_texts.append(parent_text)
            parent_metadatas.append({
                "source": file_name,
                "page": _format_page_range(parent_pages),
                "start_page": min(parent_pages),
                "end_page": max(parent_pages),
            })

            for c_idx in range(0, len(parent_slice), step):
                child_slice = parent_slice[c_idx : c_idx + child_chunk_words]
                child_text = " ".join(w for w, _ in child_slice)
                if len(child_text.strip()) > 20:
                    child_pages = [pg for _, pg in child_slice]
                    child_ids.append(f"child_{uuid.uuid4().hex[:8]}")
                    child_chunks.append(child_text)
                    child_metadatas.append({
                        "parent_id": parent_id,
                        "source": file_name,
                        "page": _format_page_range(child_pages),
                    })

        logger.info(
            f"[Hierarchical Chunking] Created {len(parent_texts)} parents and "
            f"{len(child_chunks)} children (with page metadata) for '{file_name}'."
        )
        return parent_texts, parent_ids, parent_metadatas, child_chunks, child_metadatas, child_ids
    finally:
        if not is_existing_path and valid_pdf_path and os.path.exists(valid_pdf_path):
            try:
                os.remove(valid_pdf_path)
            except OSError:
                pass


def extract_and_chunk_hierarchical(
    pdf_path,
    parent_chunk_words: int = 300,
    child_chunk_words: int = 70,
    child_overlap: int = 15,
):
    """Backward-compatible wrapper returning (parent_texts, parent_ids, child_chunks, child_metadatas, child_ids)."""
    parent_texts, parent_ids, _, child_chunks, child_metadatas, child_ids = (
        extract_and_chunk_hierarchical_with_pages(
            pdf_path,
            parent_chunk_words=parent_chunk_words,
            child_chunk_words=child_chunk_words,
            child_overlap=child_overlap,
        )
    )
    return parent_texts, parent_ids, child_chunks, child_metadatas, child_ids
