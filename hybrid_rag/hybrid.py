"""Global Hybrid RAG pipeline combining dense vector search (ChromaDB) and sparse BM25 via RRF.

Handles multi-format document indexing with batch OpenRouter embeddings, Arabic-aware BM25
tokenization, Reciprocal Rank Fusion (RRF) retrieval, and session-scoped chat memory.
"""

import os
import re
import time
import uuid
import logging
import chromadb
from rank_bm25 import BM25Okapi

try:    
    from hybrid_rag.logging_setup import setup_logging
    from hybrid_rag.chat_history import persist_chat_history, load_file_history
    from hybrid_rag.file_processing import (    
        initialize_api,
        resolve_file_name, 
        decompose_multi_questions, 
        extract_and_chunk_file,
        format_location,
        SYS_PROMPT
    )
except ModuleNotFoundError:
    from logging_setup import setup_logging
    from chat_history import persist_chat_history, load_file_history
    from file_processing import (    
        initialize_api,
        resolve_file_name, 
        decompose_multi_questions,
        extract_and_chunk_file,
        format_location,
        SYS_PROMPT
    )

# ==========================================
# 1. Setup & Configuration
# ==========================================

setup_logging()
logger = logging.getLogger("Global_Hybrid_RAG")

logger.info("Initializing Global Hybrid RAG System...")
client_ai = initialize_api()

# Unified persistent ChromaDB collection for Global Hybrid RAG
chroma_client = chromadb.PersistentClient(path="./chroma_db_global")
global_hybrid_col = chroma_client.get_or_create_collection(name="global_hybrid_collection")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")

logger.info(f"Embedding Model: {EMBEDDING_MODEL}")
logger.info(f"Chat Model: {CHAT_MODEL}")

# Global runtime cache for the consolidated BM25 index + chunk IDs & embeddings across all documents
GLOBAL_BM25_CACHE = {
    "bm25": None,
    "ids": [],
    "chunks": [],
    "metadatas": [],
    "doc_to_id": {},
    "doc_to_meta": {},
    "id_to_emb": {},
    "doc_count": 0,
}

GLOBAL_SESSION_ID = "global_chat_session"


# Unicode range covering Arabic harakat (diacritics: Fatha..Sukun, dagger Alef) and Tatweel (kashida)
ARABIC_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u0640]")


def normalize_arabic(text: str) -> str:
    """Strips Arabic diacritics/tatweel and unifies common letter variants for BM25 matching."""
    # Remove vowel marks and elongation characters so e.g. "العِلْم" matches "العلم"
    text = ARABIC_DIACRITICS_RE.sub("", text)
    # Normalize all Alef forms (أ إ آ ٱ) to bare Alef (ا)
    text = re.sub(r"[أإآٱ]", "ا", text)
    # Normalize Alef Maqsura (ى) to Ya (ي) to bridge OCR/spelling variations
    text = text.replace("ى", "ي")
    return text


def bm25_tokenizer(text: str) -> list:
    """Cleans, normalizes Arabic, and tokenizes text into lowercase words for BM25."""
    normalized = normalize_arabic(text.lower())
    # Extract alphanumeric word tokens across both Latin and Arabic Unicode scripts
    return re.findall(r"\b\w+\b", normalized)


def refresh_global_bm25():
    """
    Ensures the in-memory BM25 index (plus cached IDs & embeddings) reflects all
    chunks currently stored in the global ChromaDB collection.
    """
    total_db_docs = global_hybrid_col.count()
    # Reset all cached structures when the collection is empty
    if total_db_docs == 0:
        GLOBAL_BM25_CACHE["bm25"] = None
        GLOBAL_BM25_CACHE["ids"] = []
        GLOBAL_BM25_CACHE["chunks"] = []
        GLOBAL_BM25_CACHE["metadatas"] = []
        GLOBAL_BM25_CACHE["doc_to_id"] = {}
        GLOBAL_BM25_CACHE["doc_to_meta"] = {}
        GLOBAL_BM25_CACHE["id_to_emb"] = {}
        GLOBAL_BM25_CACHE["doc_count"] = 0
        return

    # Rebuild BM25 only if cache is empty or new documents were added/removed in ChromaDB
    if GLOBAL_BM25_CACHE["bm25"] is None or GLOBAL_BM25_CACHE["doc_count"] != total_db_docs:
        logger.info(f"Building consolidated BM25 index for {total_db_docs} chunks from Global DB...")
        # Single bulk fetch of texts, metadata, and vectors so downstream lookups need zero extra DB calls
        db_records = global_hybrid_col.get(include=["documents", "metadatas", "embeddings"])
        all_ids = db_records.get("ids", [])
        all_chunks = db_records.get("documents", [])
        all_metadatas = db_records.get("metadatas", [])
        all_embeddings = db_records.get("embeddings")
        if all_embeddings is None:
            all_embeddings = []

        # Build BM25Okapi inverted index and fast O(1) lookup dictionaries shared with CRAG & LCEL
        tokenized_corpus = [bm25_tokenizer(chunk) for chunk in all_chunks]
        GLOBAL_BM25_CACHE["bm25"] = BM25Okapi(tokenized_corpus)
        GLOBAL_BM25_CACHE["ids"] = all_ids
        GLOBAL_BM25_CACHE["chunks"] = all_chunks
        GLOBAL_BM25_CACHE["metadatas"] = all_metadatas
        GLOBAL_BM25_CACHE["doc_to_id"] = {doc: cid for doc, cid in zip(all_chunks, all_ids)}
        GLOBAL_BM25_CACHE["doc_to_meta"] = {doc: meta for doc, meta in zip(all_chunks, all_metadatas)}
        GLOBAL_BM25_CACHE["id_to_emb"] = {cid: emb for cid, emb in zip(all_ids, all_embeddings)}
        GLOBAL_BM25_CACHE["doc_count"] = total_db_docs
        logger.info("Consolidated BM25 index built successfully.")


# ==========================================
# 2. Indexing (Feed Global DB Once per File)
# ==========================================

def initialize_hybrid_index(pdf_source, progress_callback=None):
    """
    Indexes a PDF document into the global hybrid collection and refreshes the BM25 index.
    Skips indexing if the file already exists in the database.
    """
    file_name = resolve_file_name(pdf_source)
    logger.info(f"--- Checking Global Hybrid DB for '{file_name}' ---")

    # Check if file is already indexed in the Global DB
    existing_docs = global_hybrid_col.get(where={"source": file_name}, limit=1)
    if existing_docs and existing_docs.get("ids") and len(existing_docs["ids"]) > 0:
        logger.info(f"Document '{file_name}' is already indexed in Global Hybrid DB. Skipping.")
        refresh_global_bm25()
        if progress_callback:
            progress_callback(1.0, f"'{file_name}' is already indexed.")
        return

    # Extract and split text (with automatic OCR fallback for scanned pages)
    logger.info(f"Starting Extraction & Standard Chunking for '{file_name}'...")
    if progress_callback:
        progress_callback(0.02, f"Extracting text from '{file_name}'...")

    def _ocr_progress(done: int, total: int, message: str):
        if progress_callback and total > 0:
            progress_callback(0.05 + 0.50 * (done / total), f"{file_name}: {message}")

    chunks, metadatas = extract_and_chunk_file(
        pdf_source,
        chunk_size=1000,
        chunk_overlap=200,
        progress_callback=_ocr_progress,
    )

    if not chunks:
        logger.warning(f"No valid chunks extracted from '{file_name}'.")
        raise ValueError(
            f"No extractable text was found in '{file_name}' (0 chunks created). "
            "The file may be empty, corrupted, or scanned without OCR enabled."
        )

    # Generate dense embeddings in batches with retry logic
    embeddings_list = []
    batch_size = 15
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    logger.info(f"Generating embeddings via OpenRouter in {total_batches} batches...")

    for idx, i in enumerate(range(0, len(chunks), batch_size), start=1):
        batch = chunks[i : i + batch_size]
        success = False
        last_error = None

        if progress_callback:
            progress_callback(
                0.55 + 0.40 * ((idx - 1) / total_batches),
                f"{file_name}: Embedding batch {idx}/{total_batches} ({len(chunks)} chunks)...",
            )

        # Map OCR page progress into the [5% .. 55%] band of the Streamlit progress bar
        for attempt in range(3):
            try:
                response = client_ai.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                )
                embeddings_list.extend([item.embedding for item in response.data])
                success = True
                time.sleep(3)  # Gentle pacing to respect OpenRouter rate limits
                break
            except Exception as e:
                last_error = e
                status_code = getattr(e, "status_code", None)
                # Fail fast on permanent client errors (e.g. 400/401/404/410) except 429 rate-limit
                if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                    logger.error(f"Batch {idx} failed with non-retryable HTTP {status_code}: {e}")
                    break
                logger.warning(f"Batch {idx} attempt {attempt + 1} failed: {e}. Retrying in 2s...")
                time.sleep(2)

        if not success:
            logger.error(f"Failed to generate embeddings for batch {idx}. Halting indexing: {last_error}")
            raise RuntimeError(
                f"Embeddings failed on batch {idx}/{total_batches}: {last_error}"
            ) from last_error

        time.sleep(1)

    # Save to Global Hybrid Collection with unique UUIDs (batched for large books)
    if progress_callback:
        progress_callback(0.96, f"{file_name}: Saving {len(chunks)} chunks to ChromaDB...")
    chunk_ids = [f"hybrid_{uuid.uuid4().hex[:8]}" for _ in range(len(chunks))]
    add_batch_size = 1000
    for i in range(0, len(chunks), add_batch_size):
        global_hybrid_col.add(
            documents=chunks[i : i + add_batch_size],
            embeddings=embeddings_list[i : i + add_batch_size],
            metadatas=metadatas[i : i + add_batch_size],
            ids=chunk_ids[i : i + add_batch_size],
        )
    logger.info(f"Successfully added {len(chunks)} chunks from '{file_name}' to Global Hybrid DB.")

    # Refresh global BM25 cache to include the newly added chunks
    refresh_global_bm25()
    if progress_callback:
        progress_callback(1.0, f"Indexed {len(chunks)} chunks from '{file_name}'.")


def get_indexed_files() -> list:
    """Returns a sorted list of unique source filenames currently embedded in ChromaDB."""
    refresh_global_bm25()
    sources = {
        m.get("source")
        for m in GLOBAL_BM25_CACHE["metadatas"]
        if isinstance(m, dict) and m.get("source")
    }
    return sorted(sources)


def remove_file_from_index(file_name: str):
    """Removes all chunks belonging to file_name from ChromaDB and refreshes the BM25 cache."""
    logger.info(f"Removing '{file_name}' from Global Hybrid DB...")
    global_hybrid_col.delete(where={"source": file_name})
    GLOBAL_BM25_CACHE["bm25"] = None
    refresh_global_bm25()
    logger.info(f"Successfully removed '{file_name}' from Global Hybrid DB.")


def get_bm25_subset(selected_files: list = None):
    """
    Returns (bm25, chunks, metadatas, where_filter) scoped to selected_files.
    Falls back to all indexed files when selected_files is empty/None or selects all files.
    """
    refresh_global_bm25()
    bm25 = GLOBAL_BM25_CACHE["bm25"]
    chunks = GLOBAL_BM25_CACHE["chunks"]
    metadatas = GLOBAL_BM25_CACHE["metadatas"]
    where_filter = None

    if selected_files:
        selected_set = set(selected_files)
        all_sources = {
            m.get("source") for m in metadatas if isinstance(m, dict) and m.get("source")
        }
        # Only build a scoped BM25 index + Chroma `where` filter when a strict subset is selected
        if not all_sources.issubset(selected_set):
            where_filter = {"source": {"$in": list(selected_set)}}
            filtered_pairs = [
                (c, m) for c, m in zip(chunks, metadatas)
                if isinstance(m, dict) and m.get("source") in selected_set
            ]
            chunks = [c for c, _ in filtered_pairs]
            metadatas = [m for _, m in filtered_pairs]
            bm25 = BM25Okapi([bm25_tokenizer(c) for c in chunks]) if chunks else None

    return bm25, chunks, metadatas, where_filter


def get_hybrid_search_state(selected_files: list = None):
    """
    Returns (bm25, chunks, metadatas, doc_to_id, doc_to_meta, id_to_emb, where_filter)
    scoped to selected_files, reusing the in-memory ChromaDB cache without extra DB scans.
    """
    bm25, chunks, metadatas, where_filter = get_bm25_subset(selected_files)
    return (
        bm25,
        chunks,
        metadatas,
        GLOBAL_BM25_CACHE["doc_to_id"],
        GLOBAL_BM25_CACHE["doc_to_meta"],
        GLOBAL_BM25_CACHE["id_to_emb"],
        where_filter,
    )


# ==========================================
# 3. Querying (Global Multi-Question Hybrid Search)
# ==========================================

def retrieve_hybrid_context(user_query: str, top_k: int = 6, selected_files: list = None):
    """
    Performs Hybrid Retrieval (Dense Vector + Sparse BM25 via RRF) across selected (or ALL)
    indexed documents for each sub-question in the user query.
    """
    bm25, chunks, metadatas, where_filter = get_bm25_subset(selected_files)

    if not chunks or bm25 is None:
        return "", []

    sub_queries = decompose_multi_questions(user_query)
    consolidated_scores = {}
    rrf_k = 60
    t_chunks_limit = min(10, len(chunks))

    for idx, sub_q in enumerate(sub_queries, start=1):
        logger.debug(f"Running Hybrid Retrieval for sub-query #{idx}: '{sub_q}'")

        # Step A: Dense Semantic Search
        dense_chunks = []
        try:
            query_vec = client_ai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[sub_q],
            ).data[0].embedding

            dense_results = global_hybrid_col.query(
                query_embeddings=[query_vec],
                n_results=t_chunks_limit,
                include=["documents"],
                where=where_filter,
            )
            if dense_results.get("documents") and dense_results["documents"]:
                dense_chunks = dense_results["documents"][0]
        except Exception as e:
            logger.error(f"Dense search failed for sub-query '{sub_q}': {e}", exc_info=True)

        # Step B: Sparse BM25 Keyword Search
        tokenized_q = bm25_tokenizer(sub_q)
        sparse_chunks = bm25.get_top_n(tokenized_q, chunks, n=t_chunks_limit)

        # Step C: Reciprocal Rank Fusion (RRF) for this branch
        for rank, doc in enumerate(dense_chunks):
            consolidated_scores[doc] = consolidated_scores.get(doc, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, doc in enumerate(sparse_chunks):
            consolidated_scores[doc] = consolidated_scores.get(doc, 0.0) + (1.0 / (rrf_k + rank + 1))

    # Rank all retrieved documents across all sub-queries by combined RRF score
    ranked_docs = sorted(consolidated_scores.items(), key=lambda item: item[1], reverse=True)
    top_chunks = [doc for doc, _ in ranked_docs[:top_k]]
    logger.info(f"RRF selected top {len(top_chunks)} unique context chunks across all sub-queries.")

    # Map chunks to their source filename and page number
    chunk_to_meta = {c: m for c, m in zip(chunks, metadatas)}
    formatted_contexts = []

    for doc in top_chunks:
        meta_info = chunk_to_meta.get(doc, {})
        source_file = meta_info.get("source", "Unknown_File") if isinstance(meta_info, dict) else "Unknown_File"
        loc_label = format_location(meta_info)
        formatted_contexts.append(f"[Source: {source_file} | {loc_label}]:\n{doc}")

    full_context_text = "\n\n---\n\n".join(formatted_contexts)
    return full_context_text, formatted_contexts


def run_hybrid_chat_session(
    user_query: str,
    top_k: int = 6,
    history_limit: int = 5,
    selected_files: list = None,
    session_id: str = GLOBAL_SESSION_ID,
):
    """
    Searches across selected (or ALL) indexed documents using Hybrid RAG (Dense + BM25),
    supports multi-questions, and maintains a rolling window of chat history per session.
    """
    logger.info("=== Starting Global Multi-Query Hybrid RAG ===")
    logger.info(f"Session: '{session_id}' | User Query: '{user_query}'")

    if global_hybrid_col.count() == 0:
        logger.error("Global Hybrid DB is empty. Please upload and index documents first.")
        return "The database is empty. Please upload and index some PDFs first.", []

    # 1. Retrieve consolidated hybrid context
    full_context_text, formatted_contexts = retrieve_hybrid_context(
        user_query, top_k=top_k, selected_files=selected_files
    )

    # 2. Load rolling window history for the active chat session
    logger.info(f"Loading conversation history for session '{session_id}'...")
    global_history = load_file_history(session_id)
    recent_history = (
        global_history[-history_limit:] if len(global_history) > history_limit else global_history
    )
    logger.info(f"Using {len(recent_history)} recent turns from history (Total saved: {len(global_history)}).")

    # 3. Build prompt and invoke LLM
    current_turn_prompt = f"""[Context from Global Database]:
{full_context_text}

[User Question(s)]:
{user_query}
"""

    messages_to_send = [{"role": "system", "content": SYS_PROMPT}]
    for item in recent_history:
        messages_to_send.append({"role": "user", "content": item["query"]})
        messages_to_send.append({"role": "assistant", "content": item["answer"]})

    messages_to_send.append({"role": "user", "content": current_turn_prompt})

    logger.info(f"Invoking LLM '{CHAT_MODEL}'...")
    try:
        completion = client_ai.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages_to_send,
            temperature=0.2,
        )
        assistant_reply = completion.choices[0].message.content
        logger.info("LLM global response generated successfully.")
    except Exception as e:
        logger.critical(f"LLM generation failed: {e}", exc_info=True)
        raise e

    # 4. Persist chat history under the active session ID
    persist_chat_history(session_id, user_query, assistant_reply, full_context_text)
    return assistant_reply, formatted_contexts


if __name__ == "__main__":
    # 1. Index a PDF once into Global Hybrid DB:
    # sample_pdf = r"D:\GitProjects\Machine-Learning\Docs\Advice For Risk Associated with Cultured Meat.pdf"
    # initialize_hybrid_index(sample_pdf)

    # 2. Query across all indexed PDFs:
    sample_query = """1. What are the general steps in cultured meat production?
2. Why is Mycoplasma contamination difficult to detect visually?"""

    answer, contexts = run_hybrid_chat_session(sample_query)
    print("\n--- Final Answer ---\n", answer)