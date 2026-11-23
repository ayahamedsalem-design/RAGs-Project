# Hybrid RAG

Ask questions about PDF documents with two Retrieval-Augmented Generation (RAG) pipelines, in a Streamlit app that also shows what every answer cost.

- **Hybrid RAG**: semantic (vector) search and keyword (BM25) search, merged with Reciprocal Rank Fusion, so neither exact terms nor meaning are missed.
- **Hybrid Corrective RAG**: 2-round fused CRAG — hybrid search (top 15 candidates), reference-list filtering, batch LLM grading with part-sufficiency check, automatic Round-2 corrective query rewriting when parts are missing, and refusal short-circuit when no relevant chunks exist.

Both pipelines split multi-part questions into sub-questions, keep a rolling window of the conversation (last 5 turns), and cite the source file and page of every fact.

## Project structure

```
app.py                     Streamlit chat: pick a pipeline, upload PDFs, ask questions
pages/
  Usage.py                 cost dashboard (spend, requests, tokens, cost per question)
hybrid_rag/                the two pipelines and everything they share
  hybrid.py                Hybrid RAG: indexing (initialize_hybrid_index) and answering
  hybrid_corrective.py     Hybrid Corrective RAG: 2-round fused CRAG with batch grading & query rewriting
  pdf_processing.py        PDF reading, chunking, question splitting, OpenRouter client, prompt
  chat_history.py          conversation history (chat_history.json)
  summarizer.py            "Summarize History" in the sidebar
  retry.py                 retries for temporary API errors
  cost.py                  tokens and US dollars per question (logs/cost_log.csv)
  logging_setup.py         logging (config/logging_config.json)
evaluation/                evaluation of both pipelines on 20 test questions
  answer_key.json          questions, expected answers, pages and evidence phrases
  run_eval.py              index the PDFs and ask every question
  judge.py                 LLM judge: scores each answer against the answer key
  report.py                Word report with all metrics and tables
  locate.py                finds the PDF pages a retrieved chunk comes from
  results/                 saved answers, judge verdicts, hand corrections (overrides.json)
documents/                 the two PDFs the questions are about
config/
  logging_config.json
chroma_db_global/          vector store (built locally, not in git)
logs/                      app.log, errors.log, cost_log.csv (not in git)
```

## Setup

Requirements: Python 3.13 and [uv](https://docs.astral.sh/uv/).

```powershell
uv venv
uv pip install -r requirements.txt
```

Create a `.env` file in the project folder:

```
OPENROUTER_API_KEY=<your OpenRouter key>
CHAT_MODEL=deepseek/deepseek-v4-flash
EMBEDDING_MODEL=openai/text-embedding-3-small
```

All three are required: the pipelines read the model names from `.env` and have no defaults.

## Running the app

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501.

1. **Index documents**: upload a PDF in the sidebar and click *Index File into Global DB*. Both pipelines search the same collection, so a file is indexed once. Files already indexed are skipped.
2. **Ask**: choose *Hybrid RAG* or *Corrective Hybrid RAG* and type a question. Under the answer you see the retrieved chunks and what the answer cost (requests, tokens, US dollars).
3. **Usage**: the *Usage* page in the sidebar shows spend, requests and tokens over time, cost per question per pipeline, and OpenRouter's own figures for the API key.

With Docker Compose: `docker compose up --build`, then open the same address. The vector store (`chroma_db_global/`) is kept between restarts.

With `docker build` + `docker run` (single-command rebuild):

```powershell
.\rebuild.ps1
```

This builds `rags_img`, replaces the `global_rag_app` container only if the build succeeds, mounts `.env` read-only, and mounts `chroma_db_global/`, `logs/`, and `chat_history.json` so your vector store, cost logs, and chat history are preserved.

## Cost tracking

Every reply from OpenRouter includes the tokens used and the US dollars charged. The client created by `initialize_api()` (`hybrid_rag/pdf_processing.py`) hands that to `hybrid_rag/cost.py`, which adds up all requests of one question and writes one row to `logs/cost_log.csv`. If a reply has no cost, it is calculated from OpenRouter's price list (cached in `logs/model_prices.json`).

## Evaluation

Both pipelines are evaluated on 20 questions about the two PDFs in `documents/` (16 answerable, 4 unanswerable), with the metrics of the earlier evaluations: hit rate, full-answer coverage, MRR, answer accuracy, faithfulness, citation accuracy, refusal accuracy, false refusals, requests and cost per question.

```powershell
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python.exe -m evaluation.run_eval index    # index the two PDFs
.venv\Scripts\python.exe -m evaluation.run_eval run      # ask all 20 questions to both pipelines
.venv\Scripts\python.exe -m evaluation.judge             # LLM judge scores every answer
.venv\Scripts\python.exe -m evaluation.report            # Word report in your Downloads folder
```

- Each question is asked on its own with an empty chat history (the history is restored afterwards).
- A retrieved chunk counts as relevant only if it contains the answer's evidence phrase from the PDF (`answer_key.json`).
- Judge verdicts are checked by hand; corrections go in `evaluation/results/overrides.json` with a reason.
- The evaluation's costs go to `evaluation/results/cost_eval.csv`, not the app's log.

## Technology

Python 3.13 · Streamlit · ChromaDB · rank-bm25 · LangChain text splitters · OpenRouter (OpenAI SDK) · DeepSeek V4 Flash · OpenAI text-embedding-3-small
