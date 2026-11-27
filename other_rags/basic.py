"""Global Basic RAG pipeline using pure dense vector search over ``global_hybrid_collection``."""

import os
import logging

from hybrid_rag.logging_setup import setup_logging
from hybrid_rag.chat_history import (
    persist_chat_history,
    load_file_history,
    build_history_messages,
)
from hybrid_rag.pdf_processing import (
    initialize_api,
    decompose_multi_questions,
    SYS_PROMPT,
)
from hybrid_rag.hybrid import global_hybrid_col, get_bm25_subset

# ==========================================
# 1. Setup & Configuration
# ==========================================

setup_logging()
logger = logging.getLogger("Global_Basic_RAG")

logger.info("Initializing OpenRouter API Client for Basic RAG...")
client_ai = initialize_api()

GLOBAL_SESSION_ID = "global_chat_session"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")


# ==========================================
# 2. Querying (Pure Dense Vector Search on Shared Standard Collection)
# ==========================================

def run_basic_rag(
    user_query: str,
    top_k_per_query: int = 3,
    history_limit: int = 6,
    selected_files: list = None,
    session_id: str = GLOBAL_SESSION_ID,
):
    """
    Searches across selected (or ALL) indexed documents using pure dense vector search
    over the shared global_hybrid_collection (1000-char / 200-overlap standard chunks),
    supports multi-questions, and maintains a rolling window of chat history per session.
    """
    logger.info("=== Starting Global Multi-Query Basic RAG ===")
    logger.info(f"Session: '{session_id}' | User Query: '{user_query}'")

    if global_hybrid_col.count() == 0:
        logger.error("Global standard DB is empty. Please upload and index documents first.")
        return "The database is empty. Please upload and index some PDFs first.", []

    # Reuse hybrid's subset resolution so selected_files filtering behaves identically
    _, chunks, _, where_filter = get_bm25_subset(selected_files)
    if not chunks:
        logger.warning("No chunks matched the selected file filter for Basic RAG.")
        return "No indexed chunks were found for the selected file(s).", []

    n_results = min(top_k_per_query, len(chunks))

    # 1. Decompose query and retrieve unique chunks across all sub-queries
    sub_queries = decompose_multi_questions(user_query)
    unique_chunks_dict = {}

    for idx, sub_q in enumerate(sub_queries, start=1):
        logger.debug(f"Retrieving chunks for sub-query #{idx}: '{sub_q}'")
        try:
            query_res = client_ai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[sub_q],
            )
            query_vector = query_res.data[0].embedding

            results = global_hybrid_col.query(
                query_embeddings=[query_vector],
                n_results=n_results,
                include=["documents", "metadatas"],
                where=where_filter,
            )

            if results.get("documents") and results["documents"]:
                docs = results["documents"][0]
                metas = results["metadatas"][0]

                for doc, meta in zip(docs, metas):
                    source_file = meta.get("source", "Unknown_File") if isinstance(meta, dict) else "Unknown_File"
                    page_num = meta.get("page", "?") if isinstance(meta, dict) else "?"
                    formatted_chunk = f"[Source: {source_file} | Page: {page_num}]:\n{doc}"
                    unique_chunks_dict[doc] = formatted_chunk

        except Exception as e:
            logger.error(f"Search failed for sub-query '{sub_q}': {e}", exc_info=True)

    # 2. Consolidate context
    formatted_contexts = list(unique_chunks_dict.values())
    if not formatted_contexts:
        full_context_text = "The context does not contain any relevant information for the question."
    else:
        full_context_text = "\n\n---\n\n".join(formatted_contexts)
    logger.info(f"Retrieved {len(formatted_contexts)} unique chunks across all sub-queries.")

    # 3. Load rolling earlier-turn summary (if >6 turns) + latest verbatim turns for the active session
    logger.info(f"Loading conversation history for session '{session_id}'...")
    history_msgs = build_history_messages(session_id, limit=history_limit)

    # 4. Build prompt and invoke OpenRouter CHAT_MODEL (tracked automatically via initialize_api)
    prompt = f"""[Context from Global Database]:
{full_context_text}

[User Question(s)]:
{user_query}
"""

    messages_to_send = [{"role": "system", "content": SYS_PROMPT}] + history_msgs
    messages_to_send.append({"role": "user", "content": prompt})

    logger.info(f"Sending prompt to generation model '{CHAT_MODEL}'...")
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

    # 5. Persist chat history under the active session ID
    persist_chat_history(session_id, user_query, assistant_reply, full_context_text)
    return assistant_reply, formatted_contexts


if __name__ == "__main__":
    sample_query = (
        "1. What are the general steps in cultured meat production?\n"
        "2. Why is Mycoplasma contamination difficult to detect visually?"
    )
    answer, contexts = run_basic_rag(sample_query)
    print("\n--- Final Answer ---\n", answer)
