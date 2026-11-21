# Enterprise Global RAG System (All 7 Pipelines)

Ask questions about PDF documents across **seven** Retrieval-Augmented Generation (RAG) pipelines in a single Streamlit app, with unified ChromaDB file management (multi-upload, list, filter, remove), unified chat history (`chat_history.json`), per-answer cost tracking, and a 7-pipeline evaluation suite.

## Supported Pipelines

| Pipeline | Package | Vector Store Collection | Retrieval & Strategy |
| --- | --- | --- | --- |
| **Hybrid RAG** | `hybrid_rag/hybrid.py` | `global_hybrid_collection` | Dense vector + BM25 fused with Reciprocal Rank Fusion (top 6) |
| **Corrective Hybrid RAG** | `hybrid_rag/corrective_hybrid.py` | `global_hybrid_collection` | Hybrid RRF + 1-round per-chunk yes/no LLM grader |
| **Basic RAG** | `other_rags/basic.py` | `global_hybrid_collection` (shared) | Pure dense vector search over shared 1000/200 chunks (OpenRouter `CHAT_MODEL`) |
| **Parent-Child (Hierarchical) RAG** | `other_rags/parent_child.py` | `global_parent_collection` + `global_child_collection` | Searches 70-word child chunks, returns 300-word parent blocks |
| **Corrective Parent-Child RAG** | `other_rags/corrective_parent_child.py` | `global_parent_collection` + `global_child_collection` | Hierarchical retrieval + 1-round per-chunk yes/no LLM grader |
| **Ext Naive RAG** | `ext_rag/ask.py` | `ext_chunks_openrouter` | Cosine similarity search (top 4) + LLM follow-up question rewriting |
| **Ext Corrective RAG** | `ext_rag/ask_crag.py` | `ext_chunks_openrouter` | Cosine search + reference filter + batch LLM grader + Round 2 corrective queries |

## Project Structure

```text
app.py                          Streamlit chat: upload/filter/remove PDFs, pick any of the 7 pipelines
pages/
  Usage.py                      Cost dashboard across all 7 pipelines (spend, requests, tokens, avg cost)
hybrid_rag/                     Kept 100% unmodified from refactor/hybrid-cleanup for clean git merges
  hybrid.py                     Hybrid RAG indexing & answering (global_hybrid_collection)
  corrective_hybrid.py          Corrective Hybrid RAG
  pdf_processing.py             PDF reading, standard chunking, question splitting, OpenRouter client, prompt
  chat_history.py               Unified conversation history (chat_history.json)
  summarizer.py                 "Summarize History" in the sidebar
  retry.py                      Retries for transient API errors
  cost.py                       Tokens & US dollars per question (logs/cost_log.csv)
  logging_setup.py              Logging (config/logging_config.json)
other_rags/                     Additive Global pipelines + unified coordinator
  registry.py                   Unified multi-store index/list/remove & 7-pipeline dispatch table
  basic.py                      Basic RAG (shares global_hybrid_collection, OpenRouter CHAT_MODEL)
  parent_child.py               Parent-Child RAG (embeds parents & children via EMBEDDING_MODEL)
  corrective_parent_child.py    Corrective Parent-Child RAG (1-round per-chunk yes/no grader)
  hierarchical_chunking.py      Hierarchical word chunker (300-word parents / 70-word children)
ext_rag/                        ChromaDB-backed Ext Naive & Ext Corrective pipelines
  rag_common.py                 Shared ChromaDB client, OpenRouter/Ollama helpers, hybrid_rag.cost hook
  ingest.py                     CLI multi-format ingest + Streamlit UI index/list/remove helpers
  memory.py                     Follow-up question rewriting backed by unified chat_history.json
  ask.py                        Ext Naive RAG (supports selected_files filter & CLI)
  ask_crag.py                   Ext Corrective RAG (supports selected_files filter & CLI)
  app_bridge.py                 Streamlit/evaluation bridge matching (user_query, selected_files=None)
  show_db.py                    CLI inspector for all collections in chroma_db_global/
evaluation/                     Evaluation of all 7 pipelines on 20 test questions
  answer_key.json               Questions, expected answers, pages and evidence phrases
  common.py                     7-pipeline registry for evaluation scripts
  run_eval.py                   Index documents/ across all stores and run all 7 pipelines
  judge.py                      LLM judge scoring answers against answer_key.json
  report.py                     Word report with comparison tables across all 7 pipelines
  locate.py                     Maps retrieved chunks back to PDF pages
documents/                      The two test PDFs used by evaluation/
config/
  logging_config.json
chroma_db_global/               Local vector store (not in git)
logs/                           app.log, errors.log, cost_log.csv (not in git)
```

## Setup

Requirements: Python 3.13 and [uv](https://docs.astral.sh/uv/).

```powershell
uv venv
uv pip install -r requirements.txt
```

Create a `.env` file in the project folder:

```env
OPENROUTER_API_KEY=<your OpenRouter key>
CHAT_MODEL=deepseek/deepseek-v4-flash
EMBEDDING_MODEL=openai/text-embedding-3-small
```

All three are required: the Global pipelines read `CHAT_MODEL` and `EMBEDDING_MODEL` from `.env` with no defaults.

## Running the App

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501.

1. **Index documents**: upload one or more PDFs in the sidebar and click *Index File(s) into Global DB*. Files are indexed across all 3 ChromaDB store groups (`global_hybrid_collection`, `global_parent/child_collection`, and `ext_chunks_openrouter`). Already-indexed collections skip existing files automatically.
2. **Filter or remove files**: check/uncheck indexed files in the sidebar to scope queries (`selected_files`) across all 7 pipelines, or remove selected files from every collection in sync.
3. **Ask**: choose any of the 7 architectures and ask a question. All 7 pipelines share `chat_history.json` (*Clear Chat History* and *Summarize History* cover all turns) and log per-answer cost to `logs/cost_log.csv`.
4. **Usage**: open the *Usage* page in the sidebar to compare spend, requests, tokens, and average cost per question across all 7 pipelines.

With Docker Compose: `docker compose up --build`.

With `docker build` + `docker run` (single-command rebuild):

```powershell
.\rebuild.ps1
```

This builds `rags_img`, replaces `global_rag_app` only if the build succeeds, mounts `.env` read-only, and mounts `chroma_db_global/`, `logs/`, and `chat_history.json`.

## Evaluation

Evaluate all 7 pipelines on the 20 test questions about the two PDFs in `documents/`:

```powershell
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python.exe -m evaluation.run_eval index
.venv\Scripts\python.exe -m evaluation.run_eval run
.venv\Scripts\python.exe -m evaluation.run_eval run --only hybrid basic naive crag --questions 1-5
.venv\Scripts\python.exe -m evaluation.judge
.venv\Scripts\python.exe -m evaluation.report
```

## Keeping in Sync with `refactor/hybrid-cleanup`

This branch (`feat/all-rags-unified`) is branched directly from `refactor/hybrid-cleanup` and keeps **`hybrid_rag/` completely unmodified**. When teammates push updates to `refactor/hybrid-cleanup`, pull them in cleanly with:

```powershell
git fetch origin
git merge origin/refactor/hybrid-cleanup
```

Changes inside `hybrid_rag/` merge without conflicts; only thin integration lines in `app.py`, `pages/Usage.py`, or `evaluation/` may occasionally need a quick conflict review.

## Technology

Python 3.13 · Streamlit · ChromaDB · NumPy · rank-bm25 · LangChain text splitters · OpenRouter (OpenAI SDK) · DeepSeek V4 Flash · OpenAI text-embedding-3-small
