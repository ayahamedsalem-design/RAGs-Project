import uuid
import logging
from pathlib import Path
from pypdf import PdfReader

from hybrid_rag.logging_setup import setup_logging
from hybrid_rag.pdf_processing import get_valid_pdf_path, resolve_file_name

setup_logging()
logger = logging.getLogger(__name__)


def get_full_text(pdf_path) -> str:
    """Extracts and concatenates all text across pages of a PDF."""
    if isinstance(pdf_path, Path):
        pdf_path = str(pdf_path)
    valid_pdf_path = get_valid_pdf_path(pdf_path)
    reader = PdfReader(valid_pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_and_chunk_hierarchical(
    pdf_path,
    parent_chunk_words: int = 300,
    child_chunk_words: int = 70,
    child_overlap: int = 15,
):
    """
    Hierarchical chunking splitting text into large parent blocks and smaller overlapping
    child chunks. Used by Parent-Child RAG and Corrective Parent-Child RAG.
    """
    file_name = resolve_file_name(pdf_path)
    full_text = get_full_text(pdf_path)
    words = full_text.split()

    parent_ids = []
    parent_texts = []
    child_ids = []
    child_chunks = []
    child_metadatas = []

    step = max(1, child_chunk_words - child_overlap)
    for p_idx in range(0, len(words), parent_chunk_words):
        parent_id = f"parent_{uuid.uuid4().hex[:8]}"
        parent_text = " ".join(words[p_idx : p_idx + parent_chunk_words])
        parent_texts.append(parent_text)
        parent_ids.append(parent_id)

        parent_words = parent_text.split()
        for c_idx in range(0, len(parent_words), step):
            child_text = " ".join(parent_words[c_idx : c_idx + child_chunk_words])
            if len(child_text.strip()) > 20:
                child_chunks.append(child_text)
                child_ids.append(f"child_{uuid.uuid4().hex[:8]}")
                child_metadatas.append({"parent_id": parent_id, "source": file_name})

    logger.info(
        f"[Hierarchical Chunking] Created {len(parent_texts)} parents and "
        f"{len(child_chunks)} children for '{file_name}'."
    )
    return parent_texts, parent_ids, child_chunks, child_metadatas, child_ids
