import math
import os
import time
import logging
import numpy as np

from hybrid_rag.logging_setup import setup_logging
from hybrid_rag.chat_history import persist_chat_history, load_file_history
from hybrid_rag.pdf_processing import (
    initialize_api,
    resolve_file_name,
    decompose_multi_questions,
    SYS_PROMPT,
)
from hybrid_rag.hybrid import chroma_client
from other_rags.hierarchical_chunking import extract_and_chunk_hierarchical_with_pages

# ==========================================
# 1. Setup & Configuration
# ==========================================

setup_logging()
logger = logging.getLogger("Global_Parent_Child_RAG")

logger.info("Initializing OpenRouter API Client for Parent-Child RAG...")
client_ai = initialize_api()

# Reuse hybrid_rag's shared ChromaDB PersistentClient to guarantee identical client settings
global_child_col = chroma_client.get_or_create_collection(name="global_child_collection")
global_parent_col = chroma_client.get_or_create_collection(name="global_parent_collection")

GLOBAL_SESSION_ID = "global_chat_session"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")


def _embed_in_batches(texts: list, label: str, batch_size: int = 32) -> list:
    """Embeds texts via OpenRouter in batches with retry logic."""
    embeddings_list = []
    total_batches = math.ceil(len(texts) / batch_size)
    logger.info(f"Generating embeddings for {len(texts)} {label} in {total_batches} batches...")

    for idx, i in enumerate(range(0, len(texts), batch_size), start=1):
        batch = texts[i : i + batch_size]
        success = False

        for attempt in range(3):
            try:
                response = client_ai.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                )
                for item in response.data:
                    arr = np.asarray(item.embedding, dtype=np.float32)
                    norm = float(np.linalg.norm(arr))
                    embeddings_list.append((arr / norm).tolist() if norm > 0.0 else arr.tolist())
                success = True
                time.sleep(1)
                break
            except Exception as e:
                logger.warning(f"{label} batch {idx} attempt {attempt + 1} failed: {e}. Retrying in 2s...")
                time.sleep(2)

        if not success:
            logger.error(f"Failed to generate embeddings for {label} batch {idx}. Halting indexing.")
            raise RuntimeError(f"Embeddings service failed to respond for {label}.")

    return embeddings_list


# ==========================================
# 2. Indexing & File Management
# ==========================================

def index_parent_child_rag(pdf_source):
    """
    Indexes a PDF document into the global parent and child collections (300/70 words).
    Embeds both parents and children via EMBEDDING_MODEL to avoid local ONNX downloads.
    Skips indexing if the file already exists in the database.
    """
    file_name = resolve_file_name(pdf_source)
    logger.info(f"--- Checking Global Hierarchical DB for '{file_name}' ---")

    existing_docs = global_child_col.get(where={"source": file_name}, limit=1)
    if existing_docs and existing_docs.get("ids") and len(existing_docs["ids"]) > 0:
        logger.info(f"Document '{file_name}' is already indexed in Global Hierarchical DB. Skipping.")
        return

    logger.info(f"Starting Hierarchical Extraction & Chunking for '{file_name}'...")
    parent_texts, parent_ids, parent_metadatas, child_chunks, child_metadatas, child_ids = (
        extract_and_chunk_hierarchical_with_pages(
            pdf_source,
            parent_chunk_words=300,
            child_chunk_words=70,
            child_overlap=15,
        )
    )

    if not child_chunks or not parent_texts:
        logger.warning(f"No valid chunks extracted from '{file_name}'.")
        raise ValueError(
            f"No extractable text was found in '{file_name}' (0 parent/child chunks created)."
        )

    parent_embeddings = _embed_in_batches(parent_texts, label="parent chunks")
    logger.info(f"Saving {len(parent_texts)} parent blocks (with page metadata) to Global Parent DB...")
    for i in range(0, len(parent_texts), 1000):
        global_parent_col.add(
            documents=parent_texts[i : i + 1000],
            embeddings=parent_embeddings[i : i + 1000],
            metadatas=parent_metadatas[i : i + 1000],
            ids=parent_ids[i : i + 1000],
        )

    child_embeddings = _embed_in_batches(child_chunks, label="child chunks")
    for i in range(0, len(child_chunks), 1000):
        global_child_col.add(
            documents=child_chunks[i : i + 1000],
            embeddings=child_embeddings[i : i + 1000],
            metadatas=child_metadatas[i : i + 1000],
            ids=child_ids[i : i + 1000],
        )
    logger.info(f"Successfully indexed '{file_name}' into Global Parent-Child DB.")


def get_parent_child_indexed_files() -> list:
    """Returns a sorted list of unique source filenames currently in the Hierarchical DB."""
    if global_child_col.count() == 0:
        return []
    metas = global_child_col.get(include=["metadatas"]).get("metadatas") or []
    sources = {m.get("source") for m in metas if isinstance(m, dict) and m.get("source")}
    return sorted(sources)


def remove_parent_child_file(file_name: str):
    """Removes all parent and child chunks belonging to file_name from ChromaDB."""
    logger.info(f"Removing '{file_name}' from Global Hierarchical DB...")
    if global_child_col.count() > 0:
        global_child_col.delete(where={"source": file_name})
    if global_parent_col.count() > 0:
        global_parent_col.delete(where={"source": file_name})
    logger.info(f"Successfully removed '{file_name}' from Global Hierarchical DB.")


def get_child_where_filter(selected_files: list = None):
    """
    Returns (where_filter, matching_child_count) scoped to selected_files.
    Falls back to all indexed files when selected_files is empty/None or covers all files.
    """
    total = global_child_col.count()
    if total == 0:
        return None, 0

    if not selected_files:
        return None, total

    selected_set = set(selected_files)
    all_sources = set(get_parent_child_indexed_files())
    if all_sources and all_sources.issubset(selected_set):
        return None, total

    where_filter = {"source": {"$in": list(selected_set)}}
    matching_ids = global_child_col.get(where=where_filter, include=[]).get("ids") or []
    return where_filter, len(matching_ids)


# ==========================================
# 3. Querying (Global Multi-Question Hierarchical Search)
# ==========================================

def run_parent_child_rag(
    user_query: str,
    max_parents: int = 6,
    history_limit: int = 5,
    selected_files: list = None,
):
    """
    Searches across selected (or ALL) indexed documents using child embeddings, retrieves
    unique parent contexts, supports multi-questions, and maintains chat history.
    """
    logger.info("=== Starting Global Multi-Query Parent-Child RAG ===")
    logger.info(f"User Query: '{user_query}'")

    if global_child_col.count() == 0 or global_parent_col.count() == 0:
        logger.error("Global Hierarchical DB is empty. Please upload and index documents first.")
        return "The database is empty. Please upload and index some PDFs first.", []

    where_filter, matching_count = get_child_where_filter(selected_files)
    if matching_count == 0:
        logger.warning("No hierarchical chunks matched the selected file filter.")
        return "No indexed chunks were found for the selected file(s) in the Parent-Child DB.", []

    child_n_results = min(10, matching_count)

    # 1. Decompose query and retrieve matching parent IDs across all sub-queries
    sub_queries = decompose_multi_questions(user_query)
    matched_parent_ids = []

    for idx, sub_q in enumerate(sub_queries, start=1):
        logger.debug(f"Searching children for sub-query #{idx}: '{sub_q}'")
        try:
            query_resp = client_ai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[sub_q],
            )
            query_vec = query_resp.data[0].embedding

            search_results = global_child_col.query(
                query_embeddings=[query_vec],
                n_results=child_n_results,
                include=["metadatas"],
                where=where_filter,
            )

            if search_results.get("metadatas") and search_results["metadatas"]:
                for meta in search_results["metadatas"][0]:
                    pid = meta.get("parent_id")
                    if pid and pid not in matched_parent_ids:
                        matched_parent_ids.append(pid)
        except Exception as e:
            logger.error(f"Child search failed for sub-query '{sub_q}': {e}", exc_info=True)

    # 2. Fetch unique parent documents preserving retrieval order
    top_unique_parents = matched_parent_ids[:max_parents]
    logger.info(f"Selected {len(top_unique_parents)} unique parent IDs across all sub-queries.")

    if not top_unique_parents:
        return "No relevant context found in the indexed documents.", []

    parent_records = global_parent_col.get(
        ids=top_unique_parents,
        include=["documents", "metadatas"],
    )
    doc_map = dict(zip(parent_records.get("ids", []), parent_records.get("documents", [])))
    meta_map = dict(zip(parent_records.get("ids", []), parent_records.get("metadatas", [])))

    formatted_contexts = []
    for pid in top_unique_parents:
        if pid in doc_map:
            meta = meta_map.get(pid, {})
            source_file = meta.get("source", "Unknown_File") if isinstance(meta, dict) else "Unknown_File"
            page_label = meta.get("page") if isinstance(meta, dict) else None
            header = f"[Source: {source_file} | Page: {page_label}]" if page_label else f"[Source: {source_file}]"
            formatted_contexts.append(f"{header}:\n{doc_map[pid]}")

    full_context_text = "\n\n---\n\n".join(formatted_contexts)

    # 3. Load rolling window history
    logger.info("Loading global conversation history...")
    global_history = load_file_history(GLOBAL_SESSION_ID)
    recent_history = (
        global_history[-history_limit:] if len(global_history) > history_limit else global_history
    )
    logger.info(
        f"Using {len(recent_history)} recent turns from history (Total saved: {len(global_history)})."
    )

    # 4. Build prompt and invoke LLM
    prompt = f"""[Context from Global Database]:
{full_context_text}

[User Question(s)]:
{user_query}
"""

    messages_to_send = [{"role": "system", "content": SYS_PROMPT}]
    for item in recent_history:
        messages_to_send.append({"role": "user", "content": item["query"]})
        messages_to_send.append({"role": "assistant", "content": item["answer"]})

    messages_to_send.append({"role": "user", "content": prompt})

    logger.info(f"Sending prompt to generation model '{CHAT_MODEL}'...")
    try:
        completion = client_ai.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages_to_send,
            temperature=0.2,
        )
        assistant_reply = completion.choices[0].message.content
        logger.info("LLM response generated successfully.")
    except Exception as e:
        logger.critical(f"LLM generation failed: {e}", exc_info=True)
        raise e

    # 5. Persist chat history
    persist_chat_history(GLOBAL_SESSION_ID, user_query, assistant_reply, full_context_text)
    return assistant_reply, formatted_contexts


if __name__ == "__main__":
    sample_query = (
        "1. What are the general steps in cultured meat production?\n"
        "2. Why is Mycoplasma contamination difficult to detect visually?"
    )
    answer, contexts = run_parent_child_rag(sample_query)
    print("\n--- Final Answer ---\n", answer)
