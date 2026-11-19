import os
import re
import logging
import tempfile
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    # _record_cost() reads the token counts and the price from it. The requests, prompts and
    # answers are not changed in any way.
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=30,
                  http_client=httpx.Client(event_hooks={"response": [_record_cost]}))


def _record_cost(response):
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
    """Decomposes compound queries into individual sub-questions."""
    parts = re.split(r"[\n?؟;]+", user_query)
    sub_queries = [p.strip() for p in parts if len(p.strip()) > 5]
    if not sub_queries:
        sub_queries = [user_query.strip()]
    logger.info(f"Query decomposed into {len(sub_queries)} sub-questions: {sub_queries}")
    return sub_queries

# ==========================================
# 2. Chunking: standard recursive chunks (used by Hybrid and Corrective Hybrid)
# ==========================================

def extract_and_chunk_standard(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    Standard recursive chunking with page number and source filename metadata.
    Used by: Hybrid RAG, Corrective Hybrid RAG
    """
    valid_pdf_path = get_valid_pdf_path(pdf_path)
    file_name = resolve_file_name(pdf_path)
    reader = PdfReader(valid_pdf_path)

    # 1. Set up the Splitter for smart cutting that preserves paragraphs and sentences.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []
    metadatas = []

    # 2. Extract the text page by page and save the page number in the metadata
    for page_idx, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if not page_text:
            continue

        page_chunks = text_splitter.split_text(page_text)
        for chunk in page_chunks:
            # Clean up extra spaces
            cleaned_chunk = re.sub(r"\s+", " ", chunk).strip()
            if len(cleaned_chunk) > 30:
                chunks.append(cleaned_chunk)
                metadatas.append({"page": page_idx, "source": file_name})

    logger.info(f"[Standard Chunking] Created successfully {len(chunks)} chunks for {file_name}.")
    return chunks, metadatas
