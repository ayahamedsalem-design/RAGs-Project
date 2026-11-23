import os
import re
import json
import shutil
import logging
import tempfile
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
# 1. Utility Functions (Setup & File Handling)
# ==========================================

setup_logging()

logger = logging.getLogger(__name__)

# prompts.py

SYS_PROMPT = """You are an expert analytical research assistant managing a vast knowledge base.
Answer all aspects of the user's question(s) thoroughly and accurately based on BOTH the provided context below AND the conversation history.
- Use the conversation history to understand follow-up questions, pronouns, or previously discussed topics.
- If the question requires new facts, rely on the [Context from Global Database].
- If multiple questions are asked, address each question in a separate, clearly structured section.
- **IMPORTANT**: Cite the Source File and Page number mentioned in the context headers when stating facts.
- If an answer or detail cannot be found in the context or the conversation history, explicitly state: 'The answer does not exist in the available database'."""

# Initialization and Environment Setup
def initialize_api():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if cost is None:
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=30)
    # COST TRACKING: both pipelines (and the summarizer) get their client from this function,
    # so this is the only place that needs it. The http_client below is the same HTTP client the OpenAI
    # library would use anyway, plus one "event hook": after each reply arrives from OpenRouter,
    # record_cost() reads the token counts and the price from it. The requests, prompts and
    # answers are not changed in any way.
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=30,
                  http_client=httpx.Client(event_hooks={"response": [record_cost]}))


def record_cost(response):
    """Called by httpx for every reply from OpenRouter (chat and embeddings).

    OpenRouter adds a "usage" block to each reply: tokens in / out / reasoning / cached, and
    "cost" = the US dollars it actually charged for that request. This hands that block to
    hybrid_rag/cost.py, which adds it to the question being tracked (app.py starts the tracking
    for each question with cost.track(...)) and, at the end, writes one row to logs/cost_log.csv.

    It only reads the reply. If anything goes wrong here, it is ignored, so cost tracking can
    never break or change an answer."""
    try:
        if response.status_code != 200:
            return                                  # failed attempts are not billed
        response.read()                             # load the body (the OpenAI library reuses it)
        data = response.json()
        # Requests are labelled by type only: "embedding" (search vectors) or "chat" (answers,
        # grading, summaries).
        step = "embedding" if "/embeddings" in str(response.url) else "chat"
        cost.record(step, data.get("model"), data.get("usage"),
                    response.elapsed.total_seconds())
    except Exception as e:
        logger.debug(f"Cost tracking skipped one reply: {e}")


def resolve_file_name(pdf_source) -> str:
    """Safely extracts the base filename from either a file path string or an UploadedFile object."""
    if hasattr(pdf_source, "name") and isinstance(pdf_source.name, str):
        return os.path.basename(pdf_source.name)
    return os.path.basename(str(pdf_source))


def get_valid_pdf_path(pdf_source):
    """Ensures a valid readable local file path for the PDF."""
    if isinstance(pdf_source, str):
        return pdf_source
     
     # If a Gradio File object   
    if hasattr(pdf_source, "name") and isinstance(pdf_source.name, str) and os.path.exists(pdf_source.name):
        return pdf_source.name
        
     # If the Streamlit UploadedFile object
    if hasattr(pdf_source, "getbuffer"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_source.getbuffer())
            return tmp.name
     # Attempting to read bytes if read() exists       
    if hasattr(pdf_source, "read"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_source.read())
            return tmp.name
    raise ValueError("File type not supported or invalid")


def decompose_multi_questions(user_query: str) -> list:
    """Decomposes compound queries using an LLM to guarantee high accuracy."""

    logger.info(f"Using LLM to decompose query: '{user_query}'")

    client_ai = initialize_api()

    CHAT_MODEL = os.getenv("CHAT_MODEL")
    
    extraction_prompt = f"""You are a strict text extraction assistant. Your ONLY task is to identify and extract all distinct questions from the provided user input.

Rules:
1. DO NOT answer the questions.
2. DO NOT generate, hallucinate, or add any new questions.
3. Extract only the questions exactly as they are implied or stated in the user input.
4. If the user input contains multiple questions merged together, separate them logically.
5. If there is no question in the input, return an empty list.

Output your response STRICTLY in valid JSON format using the following structure:
{{"questions": ["question 1", "question 2"]}}

User Input:
{user_query}"""

    try:
        response = client_ai.chat.completions.create(
            model=CHAT_MODEL, 
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.0, 
            response_format={"type": "json_object"} 
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
# 2. OCR Support (for Scanned / Image-Only PDFs)
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
    if use_ocr is not None:
        return bool(use_ocr)
    return os.getenv("OCR_ENABLED", "true").strip().lower() not in ("0", "false", "no", "off")


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

    tesseract_cmd = find_tesseract_cmd()
    if not tesseract_cmd:
        raise RuntimeError(
            "Tesseract OCR binary was not found. Run the app via Docker (./rebuild.ps1) "
            "or install Tesseract OCR with Arabic & English language packs "
            "(tesseract-ocr-ara, tesseract-ocr-eng) and set TESSERACT_CMD if needed."
        )
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    # Validate requested OCR languages against installed tessdata
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

                    # Render on the main thread (PyMuPDF documents are not thread-safe)
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
# 3. Chunking: standard recursive chunks (used by Hybrid and Corrective Hybrid)
# ==========================================

def extract_and_chunk_standard(
    pdf_path,
    chunk_size=1000,
    chunk_overlap=200,
    use_ocr: bool | None = None,
    ocr_lang: str | None = None,
    ocr_dpi: int | None = None,
    progress_callback=None,
):
    """
    Standard recursive chunking with page number and source filename metadata.
    Automatically falls back to Tesseract OCR for scanned / image-only pages.
    Used by: Hybrid RAG, Corrective Hybrid RAG
    """
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

        # 1. Extract embedded text page by page and detect scanned/empty pages
        page_texts: dict[int, str] = {}
        pages_needing_ocr: list[int] = []

        for page_idx, page in enumerate(reader.pages):
            extracted = (page.extract_text() or "").strip()
            page_texts[page_idx] = extracted
            if len(re.sub(r"\s+", "", extracted)) < OCR_MIN_PAGE_CHARS:
                pages_needing_ocr.append(page_idx)

        # 2. Run OCR fallback on scanned / image-only pages if enabled
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
                    valid_pdf_path,
                    pages_needing_ocr,
                    ocr_lang=resolved_lang,
                    dpi=resolved_dpi,
                    max_workers=resolved_workers,
                    progress_callback=progress_callback,
                )
                for p_idx, ocr_text in ocr_texts.items():
                    if len(ocr_text.strip()) > len(page_texts[p_idx]):
                        page_texts[p_idx] = ocr_text.strip()
            except RuntimeError as exc:
                if has_text_pages:
                    # Mixed PDF (mostly text + a few blank/figure pages): warn and continue with extracted text
                    logger.warning(f"[OCR] Skipping OCR on {len(pages_needing_ocr)} page(s) in '{file_name}': {exc}")
                else:
                    # Fully scanned PDF with no extractable text: propagate actionable error
                    logger.error(f"[OCR] Cannot process scanned PDF '{file_name}': {exc}")
                    raise

        # 3. Set up the Splitter for smart cutting that preserves paragraphs and sentences.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = []
        metadatas = []

        # 4. Split extracted/OCR text page by page and save page number in metadata
        for page_idx in range(total_pages):
            page_text = page_texts.get(page_idx, "")
            if not page_text:
                continue

            page_chunks = text_splitter.split_text(page_text)
            for chunk in page_chunks:
                cleaned_chunk = re.sub(r"\s+", " ", chunk).strip()
                if len(cleaned_chunk) > 30:
                    chunks.append(cleaned_chunk)
                    metadatas.append({"page": page_idx + 1, "source": file_name})

        logger.info(f"[Standard Chunking] Created successfully {len(chunks)} chunks for {file_name}.")
        return chunks, metadatas
    finally:
        if not is_existing_path and valid_pdf_path and os.path.exists(valid_pdf_path):
            try:
                os.remove(valid_pdf_path)
            except OSError:
                pass
