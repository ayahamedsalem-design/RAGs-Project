"""Unified OpenRouter API setup, query decomposition, OCR, and multi-format file chunking.

Supports both unstructured documents (PDF with OCR fallback, DOCX, PPTX, TXT/MD, Images)
and structured data files (CSV, Excel, JSON), producing consistent text chunks and location metadata.
"""

import os
import re
import csv
import json
import shutil
import logging
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None

try:
    import fitz  # PyMuPDF (optional renderer)
except ImportError:
    fitz = None

try:
    from pdf2image import convert_from_path  # Poppler renderer (lightweight default in Docker)
except ImportError:
    convert_from_path = None

try:
    from hybrid_rag.logging_setup import setup_logging
except ModuleNotFoundError:
    from logging_setup import setup_logging

# Cost tracking (hybrid_rag/cost.py). If this file is run on its own from inside hybrid_rag/,
# the package can't be imported: cost tracking is then simply off and nothing else changes.
try:
    import httpx
    from hybrid_rag import cost
except ImportError:
    cost = None

# ==========================================
# 1. Setup, Prompts & Supported Formats
# ==========================================

setup_logging()
logger = logging.getLogger(__name__)

UNSTRUCTURED_EXTENSIONS = ("pdf", "docx", "pptx", "txt", "md", "png", "jpg", "jpeg", "tiff", "bmp", "webp")
STRUCTURED_EXTENSIONS = ("csv", "xlsx", "xls", "json")
SUPPORTED_EXTENSIONS = UNSTRUCTURED_EXTENSIONS + STRUCTURED_EXTENSIONS

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}

SYS_PROMPT = """You are an expert analytical research assistant managing a vast knowledge base of structured and unstructured files (PDFs, Word documents, PowerPoint presentations, spreadsheets, CSV/JSON datasets, and text files).
Answer all aspects of the user's question(s) thoroughly and accurately based on BOTH the provided context below AND the conversation history.
- Use the conversation history to understand follow-up questions, pronouns, or previously discussed topics.
- If the question requires new facts, rely on the [Context from Global Database].
- If multiple questions are asked, address each question in a separate, clearly structured section.
- **IMPORTANT**: Cite the Source File and Location (Page, Slide, Sheet/Rows, Table, Records, or Section) mentioned in the context headers when stating facts.
- If an answer or detail cannot be found in the context or the conversation history, explicitly state: 'The answer does not exist in the available database'."""


def format_location(meta: dict) -> str:
    """Formats a chunk's location cleanly for any file type (backward-compatible with legacy 'page' metadata)."""
    if not isinstance(meta, dict):
        return "Unknown"
    loc = meta.get("loc")
    if loc:
        return str(loc)
    page = meta.get("page")
    if page is not None and str(page).strip() != "":
        page_str = str(page).strip()
        if any(page_str.lower().startswith(p) for p in ("page", "slide", "sheet", "row", "table", "section", "record", "key", "image")):
            return page_str
        return f"Page {page_str}"
    return "Unknown"


# ==========================================
# 2. OpenRouter Client & Cost Hook
# ==========================================

def initialize_api():
    """Creates an OpenAI-compatible OpenRouter client wired with the automatic cost-tracking hook."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")

    # If cost tracking is unavailable (standalone script execution), return a plain OpenAI client
    if cost is None:
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=30)
    # Attach record_cost to httpx response events so every chat/embedding request is logged automatically
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=30,
        http_client=httpx.Client(event_hooks={"response": [record_cost]}),
    )


def record_cost(response):
    """Called by httpx for every reply from OpenRouter (chat and embeddings)."""
    try:
        # Ignore failed HTTP responses since OpenRouter does not bill non-200 errors
        if response.status_code != 200:
            return
        # Ensure the response body is buffered before parsing JSON usage fields
        response.read()
        data = response.json()
        step = "embedding" if "/embeddings" in str(response.url) else "chat"
        cost.record(step, data.get("model"), data.get("usage"), response.elapsed.total_seconds())
    except Exception as e:
        # Never break a RAG answer just because cost parsing failed on an unexpected payload
        logger.debug(f"Cost tracking skipped one reply: {e}")


# Backward-compatible alias for older modules/branches
_record_cost = record_cost


# ==========================================
# 3. File Path & Query Decomposition Helpers
# ==========================================

def resolve_file_name(file_source) -> str:
    """Safely extracts the base filename from either a file path string, Path, or UploadedFile object."""
    if hasattr(file_source, "name") and isinstance(file_source.name, str):
        return os.path.basename(file_source.name)
    return os.path.basename(str(file_source))


def get_valid_file_path(file_source) -> str:
    """Ensures a valid readable local file path while preserving the original file extension."""
    # Case 1: already a local filesystem path
    if isinstance(file_source, (str, Path)):
        return str(file_source)

    # Case 2: file-like object whose `.name` already points to an existing file on disk
    if hasattr(file_source, "name") and isinstance(file_source.name, str) and os.path.exists(file_source.name):
        return file_source.name

    # Case 3: in-memory upload (e.g., Streamlit UploadedFile) -> spill to a temp file keeping its extension
    file_name = resolve_file_name(file_source)
    suffix = Path(file_name).suffix.lower() or ".pdf"

    if hasattr(file_source, "getbuffer"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_source.getbuffer())
            return tmp.name

    if hasattr(file_source, "read"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_source.read())
            return tmp.name

    raise ValueError("File type not supported or invalid")


# Backward-compatible alias
get_valid_pdf_path = get_valid_file_path


def decompose_multi_questions(user_query: str) -> list:
    """Decomposes compound queries using an LLM to guarantee high accuracy."""
    logger.info(f"Using LLM to decompose query: '{user_query}'")

    client_ai = initialize_api()
    chat_model = os.getenv("CHAT_MODEL")

    extraction_prompt = f"""You are a strict text extraction assistant. Your ONLY task is to identify and extract all distinct questions from the provided user input.

Rules:
1. DO NOT answer the questions.
2. DO NOT generate, hallucinate, or add any new questions.
3. Extract only the questions implied or stated in the user input.
4. ONLY split into multiple questions when they ask about genuinely separate topics or numbered items. If two clauses ask about the SAME entity using pronouns (e.g. "Who wrote X and what is its relation to Y?" or "من مؤلف X وما علاقته بـ Y؟"), keep them together as ONE single question — OR if you split them, explicitly replace every pronoun with the full entity name so EVERY sub-question is 100% self-contained.
5. If there is no question in the input, return an empty list.

Output your response STRICTLY in valid JSON format using the following structure:
{{"questions": ["question 1", "question 2"]}}

User Input:
{user_query}"""

    try:
        # Disable chain-of-thought reasoning for fast, low-cost deterministic JSON decomposition
        try:
            response = client_ai.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                extra_body={"reasoning": {"enabled": False}},
            )
        except Exception:
            # Fallback for providers/models that reject the OpenRouter `reasoning` parameter
            response = client_ai.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

        result_text = response.choices[0].message.content
        parsed_json = json.loads(result_text)
        extracted_questions = parsed_json.get("questions", [])

        if not extracted_questions:
            return [user_query.strip()]

        logger.info(f"LLM extracted {len(extracted_questions)} sub-questions: {extracted_questions}")
        return extracted_questions
    except Exception as e:
        logger.error(f"LLM Decomposition failed ({e}), falling back to original query.")
        return [user_query.strip()]


# ==========================================
# 4. OCR Support (for Scanned PDFs & Standalone Images)
# ==========================================

OCR_MIN_PAGE_CHARS = 30


def find_tesseract_cmd() -> str | None:
    """Locate the Tesseract binary via env var, PATH, or common Windows paths."""
    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd and os.path.isfile(env_cmd):
        return env_cmd

    found = shutil.which("tesseract")
    if found:
        return found

    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def is_ocr_enabled(use_ocr: bool | None) -> bool:
    """Returns whether OCR fallback is enabled (explicit flag overrides ``OCR_ENABLED`` env var)."""
    if use_ocr is not None:
        return bool(use_ocr)
    return os.getenv("OCR_ENABLED", "true").strip().lower() not in ("0", "false", "no", "off")


def _resolve_ocr_language(ocr_lang: str) -> str:
    """Validates the Tesseract binary and resolves requested language packs against installed ones."""
    tesseract_cmd = find_tesseract_cmd()
    if not tesseract_cmd:
        raise RuntimeError(
            "Tesseract OCR binary was not found. Run the app via Docker (./rebuild.ps1) "
            "or install Tesseract OCR with Arabic & English language packs "
            "(tesseract-ocr-ara, tesseract-ocr-eng) and set TESSERACT_CMD if needed."
        )
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    effective_lang = ocr_lang
    try:
        installed = set(pytesseract.get_languages(config=""))
        requested = [lang.strip() for lang in ocr_lang.split("+") if lang.strip()]
        missing = [lang for lang in requested if lang not in installed]
        available = [lang for lang in requested if lang in installed]
        if missing:
            logger.warning(
                f"[OCR] Missing Tesseract language pack(s): {missing}. Installed: {sorted(installed)}"
            )
        if available:
            effective_lang = "+".join(available)
        else:
            fallback = [lang for lang in sorted(installed) if lang != "osd"]
            if not fallback:
                raise RuntimeError("No Tesseract language packs installed.")
            effective_lang = fallback[0]
            logger.warning(f"[OCR] Falling back to installed language: '{effective_lang}'")
    except RuntimeError:
        raise
    except Exception as exc:
        logger.debug(f"[OCR] Could not query installed Tesseract languages: {exc}")
    return effective_lang


def _group_contiguous_pages(page_indices: list[int], max_chunk: int) -> list[list[int]]:
    """Groups sorted 0-based page indices into contiguous runs of at most max_chunk pages."""
    if not page_indices:
        return []
    groups: list[list[int]] = []
    current: list[int] = [page_indices[0]]
    for idx in page_indices[1:]:
        if idx == current[-1] + 1 and len(current) < max_chunk:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    groups.append(current)
    return groups


def ocr_pdf_pages(
    pdf_path: str,
    page_indices: list[int],
    ocr_lang: str,
    dpi: int,
    max_workers: int,
    progress_callback=None,
) -> dict[int, str]:
    """
    Runs Tesseract OCR on specific 0-based page indices of a PDF using either PyMuPDF
    or pdf2image/poppler for rendering (both support JBIG2, CCITT, JPEG2000, etc.)
    and parallel Tesseract workers.
    """
    if pytesseract is None or Image is None:
        raise RuntimeError(
            "OCR dependencies are missing. Install 'pytesseract' and 'pillow' "
            "(pip install pytesseract pillow)."
        )
    if fitz is None and convert_from_path is None:
        raise RuntimeError(
            "No PDF renderer installed for OCR. Install 'pdf2image' (with poppler-utils) "
            "or 'pymupdf' (pip install pdf2image  OR  pip install pymupdf)."
        )

    effective_lang = _resolve_ocr_language(ocr_lang)

    def _run_tesseract(img) -> str:
        try:
            return pytesseract.image_to_string(img, lang=effective_lang, config="--psm 3")
        finally:
            img.close()

    total_ocr = len(page_indices)
    results: dict[int, str] = {}
    batch_size = max(4, max_workers * 4)
    done = 0
    renderer_name = "pymupdf" if fitz is not None else "poppler (pdf2image)"

    logger.info(
        f"[OCR] Running OCR on {total_ocr} page(s) via {renderer_name} "
        f"(lang='{effective_lang}', dpi={dpi}, workers={max_workers})..."
    )

    if fitz is not None:
        doc = fitz.open(pdf_path)
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for start in range(0, total_ocr, batch_size):
                    batch_indices = page_indices[start : start + batch_size]
                    rendered_batch = []

                    for p_idx in batch_indices:
                        pix = doc[p_idx].get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
                        img = Image.frombytes("L", [pix.width, pix.height], pix.samples)
                        rendered_batch.append((p_idx, img))

                    futures = [
                        (p_idx, executor.submit(_run_tesseract, img))
                        for p_idx, img in rendered_batch
                    ]

                    for p_idx, future in futures:
                        try:
                            results[p_idx] = future.result() or ""
                        except Exception as exc:
                            logger.warning(f"[OCR] Failed on page {p_idx + 1}: {exc}")
                            results[p_idx] = ""
                        done += 1
                        if progress_callback:
                            progress_callback(done, total_ocr, f"OCR page {done}/{total_ocr}...")

                    if done % 25 == 0 or done == total_ocr:
                        logger.info(f"[OCR] Progress: {done}/{total_ocr} page(s) processed.")
        finally:
            doc.close()
    else:
        groups = _group_contiguous_pages(sorted(page_indices), max_chunk=batch_size)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for group in groups:
                first_page = group[0] + 1
                last_page = group[-1] + 1
                images = convert_from_path(
                    pdf_path,
                    dpi=dpi,
                    grayscale=True,
                    first_page=first_page,
                    last_page=last_page,
                    thread_count=max_workers,
                )
                futures = [
                    (p_idx, executor.submit(_run_tesseract, img))
                    for p_idx, img in zip(group, images)
                ]
                for p_idx, future in futures:
                    try:
                        results[p_idx] = future.result() or ""
                    except Exception as exc:
                        logger.warning(f"[OCR] Failed on page {p_idx + 1}: {exc}")
                        results[p_idx] = ""
                    done += 1
                    if progress_callback:
                        progress_callback(done, total_ocr, f"OCR page {done}/{total_ocr}...")

                if done % 25 == 0 or done == total_ocr:
                    logger.info(f"[OCR] Progress: {done}/{total_ocr} page(s) processed.")

    return results


# ==========================================
# 5. Structured Data Loaders (Smart Row-Batching & JSON Record-Batching)
# ==========================================

def _read_text_with_encoding(file_path: str) -> str:
    """Reads text/CSV/JSON files safely with Arabic & international encoding fallback."""
    for enc in ("utf-8-sig", "utf-8", "cp1256", "cp1252", "latin-1"):
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _detect_csv_delimiter(sample_text: str) -> str:
    """Sniffs the column delimiter from the first 4 KB of a CSV file, defaulting to comma."""
    try:
        dialect = csv.Sniffer().sniff(sample_text[:4096], delimiters=",;\t|")
        return dialect.delimiter
    except Exception:
        return ","


def _format_cell_value(val) -> str:
    """Normalizes a tabular cell value to clean single-line text, treating NaN/None as empty."""
    if val is None:
        return ""
    # Handle pandas NaN / NaT without importing pandas at top-level
    try:
        if val != val:  # NaN check
            return ""
    except Exception:
        pass
    s = str(val).strip()
    if s.lower() in ("nan", "nat", "none"):
        return ""
    return re.sub(r"\s+", " ", s)


def _chunk_dataframe_smart(
    df,
    file_name: str,
    file_type: str,
    sheet_name: str | None = None,
    section_prefix: str = "Sheet",
    chunk_size: int = 1000,
    max_rows_per_chunk: int = 25,
) -> tuple[list[str], list[dict]]:
    """
    Smart Row-Batching for CSV and Excel DataFrames:
      - Drops completely empty rows/columns and preserves clean column headers.
      - Repeats a structured header ([Table | Sheet | Columns]) at the top of EVERY chunk
        so no row ever loses its column context.
      - Greedily packs consecutive rows into balanced chunks (~chunk_size chars, capped at
        max_rows_per_chunk) without splitting a row across chunks unless a single row itself
        exceeds chunk_size.
    """
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return [], []

    columns = [
        re.sub(r"\s+", " ", str(c)).strip() or f"Column_{i + 1}"
        for i, c in enumerate(df.columns)
    ]
    df.columns = columns

    header_parts = [f"Table: {file_name}"]
    section_tag = f"{section_prefix} '{sheet_name}'" if (sheet_name and section_prefix) else (sheet_name or "")
    if section_tag:
        header_parts.append(section_tag)
    header_parts.append(f"Columns: {', '.join(columns)}")
    header_line = "[" + " | ".join(header_parts) + "]"

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(200, chunk_size - len(header_line) - 10),
        chunk_overlap=100,
        separators=[" | ", "\n", " ", ""],
    )

    chunks: list[str] = []
    metadatas: list[dict] = []

    batch_lines: list[str] = []
    batch_start_row: int | None = None
    batch_end_row: int | None = None
    batch_len = len(header_line)

    def _flush_batch():
        nonlocal batch_lines, batch_start_row, batch_end_row, batch_len
        if not batch_lines or batch_start_row is None:
            return
        row_range = (
            f"Row {batch_start_row}"
            if batch_start_row == batch_end_row
            else f"Rows {batch_start_row}-{batch_end_row}"
        )
        loc = f"{section_tag} ({row_range})" if section_tag else row_range
        chunk_text = header_line + "\n" + "\n".join(batch_lines)
        chunks.append(chunk_text)
        metadatas.append({
            "source": file_name,
            "file_type": file_type,
            "data_kind": "structured",
            "loc": loc,
            "page": loc,
        })
        batch_lines = []
        batch_start_row = None
        batch_end_row = None
        batch_len = len(header_line)

    for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
        pairs = []
        for col in columns:
            val_str = _format_cell_value(row[col])
            if val_str:
                pairs.append(f"{col}: {val_str}")
        if not pairs:
            continue

        row_line = f"Row {row_idx}: " + " | ".join(pairs)

        # Handle an unusually long single row (e.g., long text/FAQ cell) gracefully
        if len(header_line) + 1 + len(row_line) > chunk_size and len(row_line) > chunk_size * 0.85:
            _flush_batch()
            sub_parts = text_splitter.split_text(row_line)
            for part_idx, part in enumerate(sub_parts, start=1):
                suffix = f" (Part {part_idx}/{len(sub_parts)})" if len(sub_parts) > 1 else ""
                loc = (
                    f"{section_tag} (Row {row_idx}{suffix})"
                    if section_tag
                    else f"Row {row_idx}{suffix}"
                )
                chunks.append(header_line + "\n" + part.strip())
                metadatas.append({
                    "source": file_name,
                    "file_type": file_type,
                    "data_kind": "structured",
                    "loc": loc,
                    "page": loc,
                })
            continue

        # Flush current batch if adding this row would overflow chunk_size or row cap
        if batch_lines and (
            batch_len + 1 + len(row_line) > chunk_size
            or len(batch_lines) >= max_rows_per_chunk
        ):
            _flush_batch()

        if batch_start_row is None:
            batch_start_row = row_idx
        batch_end_row = row_idx
        batch_lines.append(row_line)
        batch_len += 1 + len(row_line)

    _flush_batch()
    return chunks, metadatas


def _extract_csv(file_path: str, file_name: str, chunk_size: int) -> tuple[list[str], list[dict]]:
    """Loads a CSV file with auto-detected encoding/delimiter and chunks rows with header context."""
    import pandas as pd

    raw_text = _read_text_with_encoding(file_path)
    delimiter = _detect_csv_delimiter(raw_text)
    from io import StringIO

    df = pd.read_csv(StringIO(raw_text), sep=delimiter, dtype=str, keep_default_na=False)
    return _chunk_dataframe_smart(df, file_name=file_name, file_type="csv", chunk_size=chunk_size)


def _extract_excel(file_path: str, file_name: str, chunk_size: int) -> tuple[list[str], list[dict]]:
    """Extracts and smart-chunks every worksheet in an Excel workbook (.xlsx/.xls)."""
    import pandas as pd

    try:
        sheets = pd.read_excel(file_path, sheet_name=None, dtype=str)
    except ImportError as exc:
        raise RuntimeError(
            "Excel support requires 'openpyxl'. Install it via: uv pip install openpyxl"
        ) from exc

    all_chunks: list[str] = []
    all_metas: list[dict] = []
    ext = Path(file_name).suffix.lstrip(".").lower() or "xlsx"

    for sheet_name, df in sheets.items():
        c, m = _chunk_dataframe_smart(
            df,
            file_name=file_name,
            file_type=ext,
            sheet_name=str(sheet_name),
            chunk_size=chunk_size,
        )
        all_chunks.extend(c)
        all_metas.extend(m)

    return all_chunks, all_metas


def _flatten_record(obj, prefix: str = "") -> list[str]:
    """Flattens a JSON dict/value into compact 'key: value' pairs for clean retrieval."""
    pairs = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_k = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                pairs.extend(_flatten_record(v, full_k))
            elif isinstance(v, list):
                if v and all(not isinstance(x, (dict, list)) for x in v):
                    pairs.append(f"{full_k}: {', '.join(str(x) for x in v)}")
                else:
                    pairs.append(f"{full_k}: {json.dumps(v, ensure_ascii=False)}")
            elif v is not None and str(v).strip() != "":
                pairs.append(f"{full_k}: {re.sub(r'\\s+', ' ', str(v)).strip()}")
    else:
        val = re.sub(r"\s+", " ", str(obj)).strip()
        if val:
            pairs.append(f"{prefix}: {val}" if prefix else val)
    return pairs


def _extract_json(file_path: str, file_name: str, chunk_size: int, chunk_overlap: int) -> tuple[list[str], list[dict]]:
    """Flattens JSON arrays/objects into searchable key-value records and chunks them cleanly."""
    raw = _read_text_with_encoding(file_path)
    data = json.loads(raw)

    chunks: list[str] = []
    metadatas: list[dict] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n", " | ", " ", ""]
    )

    if isinstance(data, list):
        header_line = f"[JSON Dataset: {file_name} | Total Records: {len(data)}]"
        batch_lines: list[str] = []
        start_idx: int | None = None
        end_idx: int | None = None
        batch_len = len(header_line)

        def _flush():
            nonlocal batch_lines, start_idx, end_idx, batch_len
            if not batch_lines or start_idx is None:
                return
            loc = f"Record {start_idx}" if start_idx == end_idx else f"Records {start_idx}-{end_idx}"
            chunks.append(header_line + "\n" + "\n".join(batch_lines))
            metadatas.append({
                "source": file_name,
                "file_type": "json",
                "data_kind": "structured",
                "loc": loc,
                "page": loc,
            })
            batch_lines, start_idx, end_idx, batch_len = [], None, None, len(header_line)

        for idx, item in enumerate(data, start=1):
            rec_text = f"Record {idx}: " + " | ".join(_flatten_record(item))
            if len(header_line) + 1 + len(rec_text) > chunk_size and len(rec_text) > chunk_size * 0.85:
                _flush()
                for sub_i, part in enumerate(splitter.split_text(rec_text), start=1):
                    loc = f"Record {idx} (Part {sub_i})"
                    chunks.append(header_line + "\n" + part.strip())
                    metadatas.append({
                        "source": file_name,
                        "file_type": "json",
                        "data_kind": "structured",
                        "loc": loc,
                        "page": loc,
                    })
                continue

            if batch_lines and (batch_len + 1 + len(rec_text) > chunk_size or len(batch_lines) >= 20):
                _flush()

            if start_idx is None:
                start_idx = idx
            end_idx = idx
            batch_lines.append(rec_text)
            batch_len += 1 + len(rec_text)
        _flush()
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
                # Nested list of records (common API/export JSON shape)
                import pandas as pd
                df = pd.json_normalize(value)
                c, m = _chunk_dataframe_smart(
                    df,
                    file_name=file_name,
                    file_type="json",
                    sheet_name=str(key),
                    section_prefix="Key",
                    chunk_size=chunk_size,
                )
                chunks.extend(c)
                metadatas.extend(m)
            else:
                text = f"Key '{key}': " + (" | ".join(_flatten_record(value)) if isinstance(value, dict) else json.dumps(value, ensure_ascii=False))
                for part in splitter.split_text(text):
                    cleaned = re.sub(r"\s+", " ", part).strip()
                    if len(cleaned) > 15:
                        loc = f"Key '{key}'"
                        chunks.append(cleaned)
                        metadatas.append({
                            "source": file_name,
                            "file_type": "json",
                            "data_kind": "structured",
                            "loc": loc,
                            "page": loc,
                        })
    return chunks, metadatas


# ==========================================
# 6. Unstructured Data Loaders (PDF, DOCX, PPTX, TXT/MD, Images)
# ==========================================

def _extract_pdf(
    valid_path: str,
    file_name: str,
    chunk_size: int,
    chunk_overlap: int,
    use_ocr: bool | None,
    ocr_lang: str | None,
    ocr_dpi: int | None,
    progress_callback=None,
) -> tuple[list[str], list[dict]]:
    """Extracts text per PDF page, runs OCR fallback on scanned pages, and splits into page-tagged chunks."""
    # Threshold below which a PDF page is considered scanned/image-only and routed to OCR
    reader = PdfReader(valid_path)
    total_pages = len(reader.pages)

    page_texts: dict[int, str] = {}
    pages_needing_ocr: list[int] = []

    # First pass: fast native text extraction via pypdf; flag near-empty pages for OCR
    for page_idx, page in enumerate(reader.pages):
        extracted = (page.extract_text() or "").strip()
        page_texts[page_idx] = extracted
        if len(re.sub(r"\s+", "", extracted)) < OCR_MIN_PAGE_CHARS:
            pages_needing_ocr.append(page_idx)

    # Second pass (conditional): run parallel Tesseract OCR only on flagged scanned pages
    if pages_needing_ocr and is_ocr_enabled(use_ocr):
        resolved_lang = ocr_lang or os.getenv("OCR_LANG", "ara+eng")
        resolved_dpi = int(ocr_dpi or os.getenv("OCR_DPI", "250"))
        resolved_workers = int(os.getenv("OCR_WORKERS", str(min(4, os.cpu_count() or 2))))
        has_text_pages = len(pages_needing_ocr) < total_pages

        logger.info(
            f"[OCR] Detected {len(pages_needing_ocr)}/{total_pages} scanned/empty page(s) in '{file_name}'."
        )
        try:
            ocr_texts = ocr_pdf_pages(
                valid_path,
                pages_needing_ocr,
                ocr_lang=resolved_lang,
                dpi=resolved_dpi,
                max_workers=resolved_workers,
                progress_callback=progress_callback,
            )
            # Only overwrite native pypdf text when OCR actually recovered more characters
            for p_idx, ocr_text in ocr_texts.items():
                if len(ocr_text.strip()) > len(page_texts[p_idx]):
                    page_texts[p_idx] = ocr_text.strip()
        except RuntimeError as exc:
            # If the PDF is mixed (some native pages), warn and keep native pages; if 100% scanned, fail loudly
            if has_text_pages:
                logger.warning(f"[OCR] Skipping OCR on {len(pages_needing_ocr)} page(s) in '{file_name}': {exc}")
            else:
                logger.error(f"[OCR] Cannot process scanned PDF '{file_name}': {exc}")
                raise

    # Third pass: split each page's text into overlapping chunks tagged with 1-based Page metadata
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )
    chunks, metadatas = [], []

    for page_idx in range(total_pages):
        page_text = page_texts.get(page_idx, "")
        if not page_text:
            continue
        for chunk in splitter.split_text(page_text):
            cleaned = re.sub(r"\s+", " ", chunk).strip()
            # Drop tiny header/footer fragments (< 31 chars) that add retrieval noise
            if len(cleaned) > 30:
                chunks.append(cleaned)
                metadatas.append({
                    "page": page_idx + 1,
                    "loc": f"Page {page_idx + 1}",
                    "source": file_name,
                    "file_type": "pdf",
                    "data_kind": "unstructured",
                })
    return chunks, metadatas


def _table_rows_to_chunks(
    rows: list[list[str]],
    file_name: str,
    file_type: str,
    table_label: str,
    chunk_size: int,
) -> tuple[list[str], list[dict]]:
    """Converts a Word/PowerPoint table into smart row-batched chunks keeping its header row."""
    cleaned_rows = []
    for row in rows:
        vals = [re.sub(r"\s+", " ", str(c or "")).strip() for c in row]
        if any(vals):
            cleaned_rows.append(vals)
    if not cleaned_rows:
        return [], []

    import pandas as pd

    if len(cleaned_rows) >= 2:
        headers = [h or f"Col_{i + 1}" for i, h in enumerate(cleaned_rows[0])]
        body = [
            r + [""] * (len(headers) - len(r)) if len(r) < len(headers) else r[: len(headers)]
            for r in cleaned_rows[1:]
        ]
        df = pd.DataFrame(body, columns=headers)
        return _chunk_dataframe_smart(
            df,
            file_name=file_name,
            file_type=file_type,
            sheet_name=table_label,
            section_prefix="",
            chunk_size=chunk_size,
        )

    # Single-row table fallback
    line = f"[{table_label}]: " + " | ".join(v for v in cleaned_rows[0] if v)
    return [line], [{
        "source": file_name,
        "file_type": file_type,
        "data_kind": "structured",
        "loc": table_label,
        "page": table_label,
    }]


def _extract_docx(file_path: str, file_name: str, chunk_size: int, chunk_overlap: int) -> tuple[list[str], list[dict]]:
    """Extracts Word document paragraphs (grouped by headings) and embedded tables in reading order."""
    try:
        from docx import Document as DocxDocument
        from docx.document import Document as _DocxDoc
        from docx.oxml.table import CT_Tbl
        from docx.oxml.text.paragraph import CT_P
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as exc:
        raise RuntimeError(
            "Word (.docx) support requires 'python-docx'. Install it via: uv pip install python-docx"
        ) from exc

    doc = DocxDocument(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )

    chunks: list[str] = []
    metadatas: list[dict] = []
    para_buffer: list[str] = []
    section_title = "Section 1"
    section_num = 1
    table_num = 0

    def _flush_paragraphs():
        nonlocal para_buffer
        if not para_buffer:
            return
        text = "\n\n".join(para_buffer)
        for part in splitter.split_text(text):
            cleaned = re.sub(r"\s+", " ", part).strip()
            if len(cleaned) > 30:
                chunks.append(cleaned)
                metadatas.append({
                    "source": file_name,
                    "file_type": "docx",
                    "data_kind": "unstructured",
                    "loc": section_title,
                    "page": section_title,
                })
        para_buffer = []

    # Walk raw XML body children so paragraphs and tables stay in exact document reading order
    body = doc.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            p = Paragraph(child, doc)
            text = p.text.strip()
            if not text:
                continue
            # Flush accumulated text on every Word Heading so each chunk keeps its real section title
            style_name = (p.style.name or "").lower() if p.style else ""
            if style_name.startswith("heading"):
                _flush_paragraphs()
                section_num += 1
                section_title = f"Section: {text[:60]}"
            para_buffer.append(text)
        elif isinstance(child, CT_Tbl):
            # Flush prose before the table, then chunk the table with smart row-batching
            _flush_paragraphs()
            table_num += 1
            tbl = Table(child, doc)
            raw_rows = [[cell.text for cell in row.cells] for row in tbl.rows]
            t_chunks, t_metas = _table_rows_to_chunks(
                raw_rows, file_name, "docx", f"Table {table_num}", chunk_size
            )
            chunks.extend(t_chunks)
            metadatas.extend(t_metas)

    _flush_paragraphs()
    return chunks, metadatas


def _extract_pptx(file_path: str, file_name: str, chunk_size: int, chunk_overlap: int) -> tuple[list[str], list[dict]]:
    """Extracts PowerPoint slide text, tables, and speaker notes with per-slide location metadata."""
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise RuntimeError(
            "PowerPoint (.pptx) support requires 'python-pptx'. Install it via: uv pip install python-pptx"
        ) from exc

    prs = Presentation(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )
    chunks: list[str] = []
    metadatas: list[dict] = []

    for slide_idx, slide in enumerate(prs.slides, start=1):
        slide_texts: list[str] = []
        slide_table_idx = 0

        for shape in slide.shapes:
            if shape.has_text_frame:
                txt = "\n".join(p.text.strip() for p in shape.text_frame.paragraphs if p.text.strip())
                if txt:
                    slide_texts.append(txt)
            if shape.has_table:
                slide_table_idx += 1
                tbl = shape.table
                raw_rows = [[cell.text for cell in row.cells] for row in tbl.rows]
                t_chunks, t_metas = _table_rows_to_chunks(
                    raw_rows,
                    file_name,
                    "pptx",
                    f"Slide {slide_idx} Table {slide_table_idx}",
                    chunk_size,
                )
                chunks.extend(t_chunks)
                metadatas.extend(t_metas)

        # Include presenter notes if present
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                slide_texts.append(f"[Speaker Notes]: {notes}")

        combined = "\n".join(slide_texts).strip()
        if not combined:
            continue

        loc = f"Slide {slide_idx}"
        for part in splitter.split_text(combined):
            cleaned = re.sub(r"\s+", " ", part).strip()
            if len(cleaned) > 20:
                chunks.append(cleaned)
                metadatas.append({
                    "source": file_name,
                    "file_type": "pptx",
                    "data_kind": "unstructured",
                    "loc": loc,
                    "page": slide_idx,
                })
    return chunks, metadatas


def _extract_text_or_markdown(
    file_path: str, file_name: str, file_type: str, chunk_size: int, chunk_overlap: int
) -> tuple[list[str], list[dict]]:
    """Extracts plain text or Markdown (splitting by headings to keep section titles in metadata)."""
    raw = _read_text_with_encoding(file_path).strip()
    if not raw:
        return [], []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )
    chunks: list[str] = []
    metadatas: list[dict] = []

    if file_type == "md":
        # Split by markdown headings to preserve section titles in metadata
        sections = re.split(r"(?m)^(#{1,6}\s+.+)$", raw)
        current_heading = "Section 1"
        sec_idx = 1
        buffer = []

        def _emit(title: str, body_text: str):
            for part in splitter.split_text(body_text):
                cleaned = re.sub(r"\s+", " ", part).strip()
                if len(cleaned) > 20:
                    chunks.append(cleaned)
                    metadatas.append({
                        "source": file_name,
                        "file_type": "md",
                        "data_kind": "unstructured",
                        "loc": title,
                        "page": title,
                    })

        for piece in sections:
            piece_s = piece.strip()
            if not piece_s:
                continue
            if re.match(r"^#{1,6}\s+", piece_s):
                if buffer:
                    _emit(current_heading, "\n\n".join(buffer))
                    buffer = []
                sec_idx += 1
                heading_clean = re.sub(r"^#+\s+", "", piece_s).strip()[:60]
                current_heading = f"Section: {heading_clean}"
                buffer.append(piece_s)
            else:
                buffer.append(piece_s)
        if buffer:
            _emit(current_heading, "\n\n".join(buffer))
    else:
        for idx, part in enumerate(splitter.split_text(raw), start=1):
            cleaned = re.sub(r"\s+", " ", part).strip()
            if len(cleaned) > 20:
                loc = f"Section {idx}"
                chunks.append(cleaned)
                metadatas.append({
                    "source": file_name,
                    "file_type": file_type,
                    "data_kind": "unstructured",
                    "loc": loc,
                    "page": loc,
                })
    return chunks, metadatas


def _extract_image_ocr(
    file_path: str, file_name: str, file_type: str, chunk_size: int, chunk_overlap: int, ocr_lang: str | None
) -> tuple[list[str], list[dict]]:
    """Runs Tesseract OCR on an image file and splits the extracted text into chunks."""
    if pytesseract is None or Image is None:
        raise RuntimeError(
            "Image OCR requires 'pytesseract' and 'pillow' (pip install pytesseract pillow)."
        )
    resolved_lang = _resolve_ocr_language(ocr_lang or os.getenv("OCR_LANG", "ara+eng"))
    with Image.open(file_path) as img:
        text = pytesseract.image_to_string(img.convert("L"), lang=resolved_lang, config="--psm 3").strip()
    if not text:
        return [], []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )
    chunks, metadatas = [], []
    for part in splitter.split_text(text):
        cleaned = re.sub(r"\s+", " ", part).strip()
        if len(cleaned) > 15:
            chunks.append(cleaned)
            metadatas.append({
                "source": file_name,
                "file_type": file_type,
                "data_kind": "unstructured",
                "loc": "Image (OCR)",
                "page": "Image (OCR)",
            })
    return chunks, metadatas


# ==========================================
# 7. Unified File Extraction & Chunking Entry Point
# ==========================================

def extract_and_chunk_file(
    file_source,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    use_ocr: bool | None = None,
    ocr_lang: str | None = None,
    ocr_dpi: int | None = None,
    progress_callback=None,
) -> tuple[list[str], list[dict]]:
    """
    Unified multi-format file loader & chunker for BOTH structured and unstructured files:
      - Unstructured: .pdf (with automatic OCR), .docx (paragraphs + tables),
                      .pptx (slides + tables + notes), .txt, .md, and images (.png/.jpg/...)
      - Structured:   .csv, .xlsx/.xls (multi-sheet Smart Row-Batching), .json (Smart Record-Batching)

    Returns (chunks, metadatas) where each metadata dict carries:
      {"source", "file_type", "data_kind", "loc", "page"}.
    """
    # Track whether file_source was already on disk so finally: only deletes temp files we created
    is_existing_path = isinstance(file_source, (str, Path)) or (
        hasattr(file_source, "name")
        and isinstance(file_source.name, str)
        and os.path.exists(file_source.name)
    )
    valid_path = get_valid_file_path(file_source)
    file_name = resolve_file_name(file_source)
    ext = Path(file_name).suffix.lower() or Path(valid_path).suffix.lower()

    try:
        # Dispatch to the format-specific extractor based on normalized file extension
        if ext == ".pdf":
            chunks, metadatas = _extract_pdf(
                valid_path, file_name, chunk_size, chunk_overlap, use_ocr, ocr_lang, ocr_dpi, progress_callback
            )
        elif ext == ".docx":
            chunks, metadatas = _extract_docx(valid_path, file_name, chunk_size, chunk_overlap)
        elif ext == ".pptx":
            chunks, metadatas = _extract_pptx(valid_path, file_name, chunk_size, chunk_overlap)
        elif ext == ".csv":
            chunks, metadatas = _extract_csv(valid_path, file_name, chunk_size)
        elif ext in (".xlsx", ".xls"):
            chunks, metadatas = _extract_excel(valid_path, file_name, chunk_size)
        elif ext == ".json":
            chunks, metadatas = _extract_json(valid_path, file_name, chunk_size, chunk_overlap)
        elif ext in (".txt", ".md"):
            chunks, metadatas = _extract_text_or_markdown(
                valid_path, file_name, ext.lstrip("."), chunk_size, chunk_overlap
            )
        elif ext in IMAGE_EXTENSIONS:
            chunks, metadatas = _extract_image_ocr(
                valid_path, file_name, ext.lstrip("."), chunk_size, chunk_overlap, ocr_lang
            )
        else:
            raise ValueError(
                f"Unsupported file type '{ext}' for '{file_name}'. "
                f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        logger.info(f"[File Processing] Created {len(chunks)} chunks for '{file_name}' (type={ext}).")
        return chunks, metadatas
    finally:
        # Clean up temporary files spilled from Streamlit UploadedFile buffers
        if not is_existing_path and valid_path and os.path.exists(valid_path):
            try:
                os.remove(valid_path)
            except OSError:
                pass


# Backward-compatible alias so any existing callers keep working seamlessly
extract_and_chunk_standard = extract_and_chunk_file
