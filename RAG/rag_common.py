# Shared settings and helpers for the multi-file RAG pipeline.
# Used by ingest.py (add documents), ask.py (plain RAG) and ask_crag.py (Corrective RAG).
#
# Settings read from .env (repo root) or the environment:
#   OPENROUTER_API_KEY   required for the openrouter backend
#   CHAT_MODEL           OpenRouter chat model     (default deepseek/deepseek-v4-flash)
#   EMBEDDING_MODEL      OpenRouter embedding model (default openai/text-embedding-3-small)
#                        CHAT_MODEL and EMBEDDING_MODEL are the names the RAGs/ pipelines read
#                        too, so every pipeline in the evaluation runs on the same models.
#                        A different EMBEDDING_MODEL needs:  python -m RAG.ingest --rebuild
#   RAG_BACKEND          "openrouter" (default) or "ollama" (local models, see below)
#   RAG_EMBED_DIMS       optional: keep only the first N numbers of each vector (needs --rebuild).
#                        Unset = the model's full vector length.
#   RAG_REASONING        "on" lets the chat model think before answering (thinking tokens are
#                        billed as output). Default off: chat requests send reasoning disabled.

import os, re, sys, time, logging
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

from RAG import cost

load_dotenv()
# Chroma's telemetry is turned off with an env var, not with Settings(anonymized_telemetry=False):
# the RAGs/ modules open the same folder with default settings in the same process, and Chroma
# refuses a second client for one folder when the settings differ.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logger = logging.getLogger("rag.common")

# ---------- Settings ----------
# Where the models run:
#   "openrouter" = online models through OpenRouter (default; needs OPENROUTER_API_KEY)
#   "ollama"     = on this computer (offline, free, slower)
# To use the local models for one terminal:  $env:RAG_BACKEND = "ollama"
class RAGError(RuntimeError):
    """A problem the user can fix (missing key, empty store, ...). The CLIs print it and exit;
    the Streamlit app shows it instead of dying."""


BACKEND = os.environ.get("RAG_BACKEND", "openrouter").lower()

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / "documents"        # put all your files here (any topic, any mix)
STORE_DIR = HERE.parent / "chroma_db_global"   # shared ChromaDB folder (the RAGs/ pipelines use it too)
COLLECTION = f"ext_chunks_{BACKEND}"           # "ext_" keeps it apart from the RAGs/ collections

if BACKEND == "ollama":
    EMBED_MODEL = "nomic-embed-text"     # install once with:  ollama pull nomic-embed-text
    LLM_MODEL = "gemma3:4b"
    QUERY_PREFIX, DOC_PREFIX = "search_query: ", "search_document: "   # nomic-embed-text expects these
elif BACKEND == "openrouter":
    # "or": an empty CHAT_MODEL= line in .env also falls back to the default
    LLM_MODEL = os.environ.get("CHAT_MODEL") or "deepseek/deepseek-v4-flash"
    EMBED_MODEL = os.environ.get("EMBEDDING_MODEL") or "openai/text-embedding-3-small"
    QUERY_PREFIX, DOC_PREFIX = "", ""
else:
    raise RAGError(f'Unknown RAG_BACKEND "{BACKEND}". Use "ollama" or "openrouter".')

# The vector length comes from the model itself (it is recorded in the collection on the first
# ingest and checked after that). RAG_EMBED_DIMS only cuts the vectors shorter when set.
DIMS = int(os.environ["RAG_EMBED_DIMS"]) if os.environ.get("RAG_EMBED_DIMS") else None

# Thinking tokens are billed as output, and grading/answering from given sources doesn't need them
REASONING = os.environ.get("RAG_REASONING", "off").strip().lower() in ("1", "on", "true", "yes")

OLLAMA_NUM_CTX = 16384   # how much text the local LLM can read at once (grading needs ~13,000 tokens)
# Chunking is the same as the RAGs/ pipelines (Basic and Hybrid RAG), so every RAG in the
# project searches identically sized chunks: 1000 characters, 200 overlap (LangChain's
# RecursiveCharacterTextSplitter, see RAG/ingest.py).
CHUNK_CHARS = 1000
OVERLAP_CHARS = 200
# Embedding requests: at most BATCH texts and EMBED_MAX_TOKENS estimated tokens each.
# NVIDIA's free Nemotron embedder on OpenRouter rejects big requests (HTTP 502 "service
# inference failed"; measured 2026-11-07): one chunk of 1,900 tokens works, two chunks of
# ~900 tokens don't, many tiny texts do. For Nemotron only, (number of texts) x (longest text)
# must stay within 1,000 estimated tokens. Other models take big requests.
NEMOTRON_EMBED = "nemotron" in EMBED_MODEL.lower()
BATCH = 16 if NEMOTRON_EMBED else 64     # also: chunks saved at a time by ingest.py
EMBED_MAX_TOKENS = 1000 if NEMOTRON_EMBED else 100_000
TOP_K = 4               # chunks given to the LLM
TEMPERATURE = 0.2

# Corrective RAG settings (used by ask_crag.py only; ask.py ignores them)
CANDIDATES = 10          # chunks retrieved and checked by the grader
MAX_CONTEXT = 6          # most chunks given to the LLM after grading
MIN_SIMILARITY = 0.05    # chunks below this are dropped before grading (the grader does the real filtering)
MAX_EXTRA_QUERIES = 4    # new search queries per question (unanswered parts + grader's suggestions)
EXTRA_PER_QUERY = 4      # chunks retrieved for each new query
PER_FILE = 2             # chunks per document, per query, for questions about several documents
MAX_GRADED = 16          # most chunks shown to the grader at once
GRADER_TEMPERATURE = 0.0 # grading should be consistent, not creative
# How the grader works: "per_chunk" asks a short yes/no question for each chunk (best for small
# local models), "batch" grades all chunks in one call (fewer requests for online models)
GRADING = "per_chunk" if BACKEND == "ollama" else "batch"
REFERENCE_THRESHOLD = 12 # chunks this dense with citations (years, DOIs, URLs) are reference lists

OPENROUTER_URL = "https://openrouter.ai/api/v1"
REQUEST_DEADLINE = 180   # seconds before an OpenRouter request counts as stuck and is retried


def fail(message):
    """Stop with a message for the user: log it, then raise RAGError (never sys.exit, which
    would kill a Streamlit run)."""
    logger.error(message, stacklevel=2)     # stacklevel=2: name the function that called fail()
    raise RAGError(message)


class ContentRejected(RAGError):
    """OpenRouter's content safety screen refused the input. Sending the same text again
    gets the same answer, so it is never retried. (Seen on the free embedding model with
    reference-list chunks that start with a bare DOI link.)"""


# ---------- OpenRouter ----------
def _headers():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        fail("OPENROUTER_API_KEY is not set. Add it to the .env file in the project folder.")
    return {"Authorization": f"Bearer {key}"}


def _post_with_deadline(url, payload):
    """requests' timeout only fires when the connection goes silent, but OpenRouter keeps a
    slow request alive by sending "still processing" bytes, so a stuck request could hang for
    half an hour. Read the reply in pieces and give up after REQUEST_DEADLINE seconds."""
    started = time.perf_counter()
    with requests.post(url, headers=_headers(), json=payload, timeout=(15, 60), stream=True) as r:
        body = b""
        for piece in r.iter_content(8192):
            body += piece
            if time.perf_counter() - started > REQUEST_DEADLINE:
                raise requests.Timeout(f"no complete reply after {REQUEST_DEADLINE}s")
        r._content = body          # so r.text and r.json() work as usual
        return r


def post(endpoint, payload, retries=6, can_split=False, step=None):
    """POST to OpenRouter, retrying when the free models are busy.
    can_split=True: on a server error, return None at once (no retries), so the caller
    can send the texts in smaller requests instead.
    Every successful reply is recorded in cost.py under the given step (what it cost)."""
    payload = {**payload, "usage": {"include": True}}   # ask for the cost in every reply
    for attempt in range(retries):
        try:
            started = time.perf_counter()
            logger.debug("POST %s/%s (model %s)", OPENROUTER_URL, endpoint, payload.get("model"))
            r = _post_with_deadline(f"{OPENROUTER_URL}/{endpoint}", payload)
            logger.debug("HTTP %s after %.1fs", r.status_code, time.perf_counter() - started)
            if r.status_code == 200:
                data = r.json()
                if "error" not in data:
                    cost.record(step, payload.get("model"), data.get("usage"),
                                time.perf_counter() - started)
                    return data
                logger.warning("OpenRouter %s API error: %s", endpoint, str(data["error"])[:500])
            elif r.status_code == 422 and "content safety" in r.text:
                raise ContentRejected(f"OpenRouter's content safety screen refused the {endpoint} input")
            elif r.status_code in (400, 401, 402, 403, 404):
                # These won't fix themselves by waiting (bad key, bad model name, ...)
                fail(f"HTTP {r.status_code}: {r.text[:300]}")
            elif r.status_code == 429 and "per-day" in r.text:
                # The daily free quota is used up; retrying within seconds can't help
                fail("OpenRouter's daily limit for free models is used up "
                     "(50 requests per day, or 1,000 after buying 10 credits).\n"
                     "It resets at midnight UTC (03:00 Cairo time). Everything embedded so far "
                     "is saved: run the same command again then, and ingest continues where it stopped.")
            else:
                logger.warning("OpenRouter %s HTTP %s: %s", endpoint, r.status_code, r.text[:500])
                if can_split and r.status_code >= 500:
                    return None
        except requests.RequestException as e:
            logger.warning("OpenRouter %s request error: %s", endpoint, e)
        wait = 2 ** attempt
        logger.info("Retrying in %ds (attempt %d of %d)", wait, attempt + 1, retries)
        time.sleep(wait)
    fail("Request failed after several retries. Try again in a few minutes.")


# ---------- Ollama (local) ----------
def ollama_call(fn, step=None, **kwargs):
    """Call the local Ollama server with clear messages for the usual problems.
    The call is recorded in cost.py as free ("local"), with its token counts."""
    import ollama
    try:
        started = time.perf_counter()
        r = getattr(ollama, fn)(**kwargs)
        cost.record(step, kwargs.get("model"),
                    {"prompt_tokens": r.get("prompt_eval_count") or 0,
                     "completion_tokens": r.get("eval_count") or 0},
                    time.perf_counter() - started, local=True)
        return r
    except ollama.ResponseError as e:
        if "not found" in str(e).lower():
            fail(f"The model {kwargs.get('model')} is not installed. Run:  ollama pull {kwargs.get('model')}")
        fail(f"Ollama error: {e}")
    except (ConnectionError, OSError) as e:
        fail(f"Can't reach Ollama ({e}). Open the Ollama app (llama icon near the clock) and try again.")


# ---------- Embeddings and chat (same functions for both backends) ----------
def embed_texts(texts, is_query=False, skip_rejected=False, step=None):
    """Embed a list of texts -> float32 array (len(texts), DIMS), normalized.
    Questions use is_query=True (some embedding models expect a different prefix).
    skip_rejected=True (ingest): texts the content safety screen refuses get None instead of
    a vector, and a list is returned. Otherwise a refused text stops with ContentRejected.
    step names the request in the cost record (default: embed_query / embed_chunks)."""
    prefix = QUERY_PREFIX if is_query else DOC_PREFIX
    step = step or ("embed_query" if is_query else "embed_chunks")
    out, started = [], time.perf_counter()
    batches = embed_batches([prefix + t for t in texts])
    for batch in batches:
        for v in _embed_batch(batch, skip_rejected, step):
            if v is None:
                out.append(None)
                continue
            v = np.array(v, dtype=np.float32)
            if DIMS:
                v = v[:DIMS]
            out.append(v / (np.linalg.norm(v) or 1.0))
    logger.debug("embed %d %s with %s in %d requests, %.2fs", len(texts),
                 "queries" if is_query else "chunks", EMBED_MODEL, len(batches),
                 time.perf_counter() - started)
    return out if skip_rejected else np.vstack(out)


def estimate_tokens(text):
    """A quick upper bound on a text's token count, without calling the model. Measured on the
    test PDFs: real counts are 75-95% of it (OpenRouter's and nomic-embed-text's tokenizers)."""
    return len(re.findall(r"[A-Za-z]+", text)) * 1.3 + len(re.findall(r"[^A-Za-z\s]", text))


def embed_batches(texts):
    """Group texts into requests of at most BATCH texts and EMBED_MAX_TOKENS estimated tokens:
    the sum of the texts, or for Nemotron (number of texts) x (longest text). A single text
    too long for that still gets a request of its own."""
    batches, current, longest, total = [], [], 0, 0
    for t in texts:
        tokens = estimate_tokens(t)
        if NEMOTRON_EMBED:
            size = (len(current) + 1) * max(longest, tokens)
        else:
            size = total + tokens
        if current and (len(current) == BATCH or size > EMBED_MAX_TOKENS):
            batches.append(current)
            current, longest, total = [], 0, 0
        current.append(t)
        longest = max(longest, tokens)
        total += tokens
    if current:
        batches.append(current)
    return batches


def _embed_batch(batch, skip_rejected=False, step=None):
    """Embed one request's texts. If OpenRouter rejects a request of several texts as too
    big after all, send each half on its own instead of retrying the same request.
    The same halving finds a text the content safety screen refuses; with skip_rejected
    it gets None, so the other texts of the request are still embedded."""
    if BACKEND == "ollama":
        return ollama_call("embed", step=step, model=EMBED_MODEL, input=batch)["embeddings"]
    try:
        data = post("embeddings", {"model": EMBED_MODEL, "input": batch},
                    can_split=len(batch) > 1, step=step)
    except ContentRejected:
        if len(batch) > 1:
            half = len(batch) // 2
            logger.info("Content safety screen refused a request of %d texts; "
                        "sending %d and %d separately to find the refused one",
                        len(batch), half, len(batch) - half)
            return (_embed_batch(batch[:half], skip_rejected, step) +
                    _embed_batch(batch[half:], skip_rejected, step))
        if skip_rejected:
            logger.warning("Content safety screen refused this text, skipped: %r", batch[0][:200])
            return [None]
        fail("The embedding model's content safety screen refused this text. Try rephrasing it.")
    if data is None:
        half = len(batch) // 2
        logger.warning("OpenRouter rejected %d texts in one request; sending %d and %d separately",
                       len(batch), half, len(batch) - half)
        return (_embed_batch(batch[:half], skip_rejected, step) +
                _embed_batch(batch[half:], skip_rejected, step))
    data = sorted(data["data"], key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in data]


def chat(messages, temperature=None, json_schema=None, step="answer"):
    """Send messages to the LLM and return its reply. json_schema forces a JSON reply
    of that shape (local models only; online models are asked for JSON in the prompt).
    step names the request in the cost record (rewrite, grade, answer, ...)."""
    temp = TEMPERATURE if temperature is None else temperature
    started = time.perf_counter()
    if BACKEND == "ollama":
        r = ollama_call("chat", step=step, model=LLM_MODEL, messages=messages, format=json_schema,
                        options={"temperature": temp, "num_ctx": OLLAMA_NUM_CTX})
        reply = r["message"]["content"]
        # Ollama silently drops the start of a prompt that doesn't fit, system prompt included
        tokens = r.get("prompt_eval_count") or 0
        if tokens >= OLLAMA_NUM_CTX:
            logger.warning("Prompt of %d tokens filled num_ctx=%d and was probably cut short; "
                           "raise OLLAMA_NUM_CTX", tokens, OLLAMA_NUM_CTX)
        logger.debug("prompt tokens evaluated: %d of num_ctx %d", tokens, OLLAMA_NUM_CTX)
    else:
        data = post("chat/completions", {"model": LLM_MODEL, "messages": messages,
                                         "temperature": temp,
                                         "reasoning": {"enabled": REASONING}}, step=step)
        reply = data["choices"][0]["message"]["content"]
    logger.debug("chat %s temperature=%s json=%s, %d prompt chars, %.2fs -> %r",
                 LLM_MODEL, temp, json_schema is not None,
                 sum(len(m["content"]) for m in messages), time.perf_counter() - started,
                 reply[:300])
    return reply


# ---------- Vector store (ChromaDB) ----------
# One collection per backend, in the same folder, so switching backends doesn't need a rebuild.
# Each chunk is stored with its embedding, its text and where it came from:
#   id        "<file hash>:<chunk number>", so the same file always gets the same ids
#   metadata  path, location, hash, file_type, added, chunk, chunks (= chunks in that file)
def _client():
    import chromadb
    # Default settings on purpose: see the telemetry note at the top of this file.
    return chromadb.PersistentClient(path=str(STORE_DIR))


def store_exists():
    return STORE_DIR.exists() and COLLECTION in [c.name for c in _client().list_collections()]


def delete_store():
    if store_exists():
        _client().delete_collection(COLLECTION)
        logger.warning("Deleted collection %s", COLLECTION)


# Settings that decide what the stored chunks and vectors look like
# (the vector length is recorded separately, by check_dims(), once the first vectors exist)
STORE_SETTINGS = {"embed_model": EMBED_MODEL,
                  "chunk_chars": CHUNK_CHARS, "overlap_chars": OVERLAP_CHARS}


def open_store(check=True):
    # embedding_function=None: we always pass our own vectors, Chroma never embeds anything.
    # cosine space: distance = 1 - cosine similarity
    col = _client().get_or_create_collection(
        COLLECTION, embedding_function=None, configuration={"hnsw": {"space": "cosine"}},
        metadata=STORE_SETTINGS)
    # The stored vectors must use the same model and size as new questions, and the same
    # chunking (otherwise unchanged files would keep their old chunks, as files are embedded once).
    stored = col.metadata or {}
    changed = [f"{k}: {stored.get(k)} -> {v}" for k, v in STORE_SETTINGS.items() if stored.get(k) != v]
    if changed and check:
        fail(f"The {COLLECTION} store was built with different settings ({'; '.join(changed)}).\n"
             "Run:  python -m RAG.ingest --rebuild")
    logger.debug("Opened %s in %s (%d chunks)", COLLECTION, STORE_DIR, col.count())
    return col


def check_dims(col, n, record=False):
    """The collection's vectors all have the same length, recorded as "dims" in its metadata.
    record=True (ingest): the first vectors set it. A different length later means the model
    or RAG_EMBED_DIMS changed since the store was built."""
    meta = dict(col.metadata or {})
    stored = meta.get("dims")
    if stored is None:
        if record:
            meta["dims"] = n
            col.modify(metadata=meta)
            logger.info("%s holds vectors of %d numbers (%s)", COLLECTION, n, EMBED_MODEL)
        return
    if stored != n:
        fail(f"The {COLLECTION} store holds vectors of {stored} numbers, but {EMBED_MODEL} "
             f"now gives {n} (embedding model or RAG_EMBED_DIMS changed).\n"
             "Run:  python -m RAG.ingest --rebuild")


def search(col, qvec, k):
    """The k chunks closest to a query vector, most similar first, as dicts:
    {"id", "sim", "path", "loc", "text"}."""
    k = min(k, col.count())
    if k == 0:
        return []
    check_dims(col, len(qvec))
    r = col.query(query_embeddings=[qvec], n_results=k,
                  include=["distances", "documents", "metadatas"])
    hits = [{"id": cid, "sim": 1 - d, "path": m["path"], "loc": m["location"], "text": t}
            for cid, d, m, t in zip(r["ids"][0], r["distances"][0],
                                    r["metadatas"][0], r["documents"][0])]
    logger.debug("search k=%d -> %s", k, ", ".join(f"{h['id'][:8]}..{h['id'][-4:]} "
                 f"{h['path']} {h['loc']} {h['sim']:.3f}" for h in hits))
    return hits


def get_embedding(col, chunk_id):
    return np.asarray(col.get(ids=[chunk_id], include=["embeddings"])["embeddings"][0],
                      dtype=np.float32)


def stored_files(col):
    """Every file in the store, by hash:
    {hash: {path, file_type, added, chunks, skipped, stored, complete}}.
    A file is complete when all its chunks were saved (an import cut short leaves fewer),
    apart from the ones the embedding model refused ("skipped", set once the import ends)."""
    files = {}
    for m in col.get(include=["metadatas"])["metadatas"]:
        f = files.setdefault(m["hash"], {"path": m["path"], "file_type": m["file_type"],
                                         "added": m["added"], "chunks": m["chunks"],
                                         "skipped": m.get("skipped", 0), "stored": 0})
        f["stored"] += 1
    for f in files.values():
        f["complete"] = f["stored"] + f["skipped"] >= f["chunks"]
    logger.debug("Store holds %d files (%d incomplete)", len(files),
                 sum(not f["complete"] for f in files.values()))
    return files
