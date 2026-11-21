# Shared settings and helpers for the Ext multi-file RAG pipelines.
# Used by ingest.py (add documents), ask.py (plain RAG) and ask_crag.py (Corrective RAG).
#
# Settings read from .env (repo root) or the environment:
#   OPENROUTER_API_KEY   required for the openrouter backend
#   CHAT_MODEL           OpenRouter chat model      (default deepseek/deepseek-v4-flash)
#   EMBEDDING_MODEL      OpenRouter embedding model (default openai/text-embedding-3-small)
#                        A different EMBEDDING_MODEL needs:  python -m ext_rag.ingest --rebuild
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

from hybrid_rag import cost

load_dotenv()
# Chroma's telemetry is turned off with an env var, not with Settings(anonymized_telemetry=False):
# all packages open the same folder with default settings in the same process, and Chroma
# refuses a second client for one folder when the settings differ.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logger = logging.getLogger("rag.common")


class RAGError(RuntimeError):
    """A problem the user can fix (missing key, empty store, ...). The CLIs print it and exit;
    the Streamlit app shows it instead of dying."""


BACKEND = os.environ.get("RAG_BACKEND", "openrouter").lower()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DOCS_DIR = ROOT / "documents"                  # shared test PDFs folder
STORE_DIR = ROOT / "chroma_db_global"          # shared ChromaDB folder
COLLECTION = f"ext_chunks_{BACKEND}"           # keeps Ext chunks apart from global_* collections

if BACKEND == "ollama":
    EMBED_MODEL = "nomic-embed-text"           # install once with:  ollama pull nomic-embed-text
    LLM_MODEL = "gemma3:4b"
    QUERY_PREFIX, DOC_PREFIX = "search_query: ", "search_document: "
elif BACKEND == "openrouter":
    LLM_MODEL = os.environ.get("CHAT_MODEL") or "deepseek/deepseek-v4-flash"
    EMBED_MODEL = os.environ.get("EMBEDDING_MODEL") or "openai/text-embedding-3-small"
    QUERY_PREFIX, DOC_PREFIX = "", ""
else:
    raise RAGError(f'Unknown RAG_BACKEND "{BACKEND}". Use "ollama" or "openrouter".')

DIMS = int(os.environ["RAG_EMBED_DIMS"]) if os.environ.get("RAG_EMBED_DIMS") else None
REASONING = os.environ.get("RAG_REASONING", "off").strip().lower() in ("1", "on", "true", "yes")

OLLAMA_NUM_CTX = 16384
CHUNK_CHARS = 1000
OVERLAP_CHARS = 200
NEMOTRON_EMBED = "nemotron" in EMBED_MODEL.lower()
BATCH = 16 if NEMOTRON_EMBED else 64
EMBED_MAX_TOKENS = 1000 if NEMOTRON_EMBED else 100_000
TOP_K = 4
TEMPERATURE = 0.2

# Corrective RAG settings (used by ask_crag.py only; ask.py ignores them)
CANDIDATES = 10
MAX_CONTEXT = 6
MIN_SIMILARITY = 0.05
MAX_EXTRA_QUERIES = 4
EXTRA_PER_QUERY = 4
PER_FILE = 2
MAX_GRADED = 16
GRADER_TEMPERATURE = 0.0
GRADING = "per_chunk" if BACKEND == "ollama" else "batch"
REFERENCE_THRESHOLD = 12

OPENROUTER_URL = "https://openrouter.ai/api/v1"
REQUEST_DEADLINE = 180


def fail(message):
    """Stop with a message for the user: log it, then raise RAGError (never sys.exit, which
    would kill a Streamlit run)."""
    logger.error(message, stacklevel=2)
    raise RAGError(message)


class ContentRejected(RAGError):
    """OpenRouter's content safety screen refused the input."""


# ---------- OpenRouter ----------
def _headers():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        fail("OPENROUTER_API_KEY is not set. Add it to the .env file in the project folder.")
    return {"Authorization": f"Bearer {key}"}


def _post_with_deadline(url, payload):
    started = time.perf_counter()
    with requests.post(url, headers=_headers(), json=payload, timeout=(15, 60), stream=True) as r:
        body = b""
        for piece in r.iter_content(8192):
            body += piece
            if time.perf_counter() - started > REQUEST_DEADLINE:
                raise requests.Timeout(f"no complete reply after {REQUEST_DEADLINE}s")
        r._content = body
        return r


def post(endpoint, payload, retries=6, can_split=False, step=None):
    """POST to OpenRouter, retrying transient errors and recording usage in hybrid_rag.cost."""
    payload = {**payload, "usage": {"include": True}}
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
                fail(f"HTTP {r.status_code}: {r.text[:300]}")
            elif r.status_code == 429 and "per-day" in r.text:
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


# ---------- Ollama (optional local backend) ----------
def ollama_call(fn, step=None, **kwargs):
    try:
        import ollama
    except ImportError:
        fail("The 'ollama' package is not installed. Use RAG_BACKEND=openrouter or install ollama.")
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
        fail(f"Can't reach Ollama ({e}). Open the Ollama app and try again.")


# ---------- Embeddings and chat ----------
def embed_texts(texts, is_query=False, skip_rejected=False, step=None):
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
    return len(re.findall(r"[A-Za-z]+", text)) * 1.3 + len(re.findall(r"[^A-Za-z\s]", text))


def embed_batches(texts):
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
    temp = TEMPERATURE if temperature is None else temperature
    started = time.perf_counter()
    if BACKEND == "ollama":
        r = ollama_call("chat", step=step, model=LLM_MODEL, messages=messages, format=json_schema,
                        options={"temperature": temp, "num_ctx": OLLAMA_NUM_CTX})
        reply = r["message"]["content"]
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
def _client():
    # Reuse hybrid_rag's PersistentClient if it is already open on the same directory,
    # avoiding any duplicate-client settings conflict in the same process.
    hybrid_mod = sys.modules.get("hybrid_rag.hybrid")
    if hybrid_mod is not None and hasattr(hybrid_mod, "chroma_client"):
        try:
            if Path("./chroma_db_global").resolve() == STORE_DIR.resolve():
                return hybrid_mod.chroma_client
        except OSError:
            pass
    import chromadb
    return chromadb.PersistentClient(path=str(STORE_DIR))


def store_exists():
    return STORE_DIR.exists() and COLLECTION in [c.name for c in _client().list_collections()]


def delete_store():
    if store_exists():
        _client().delete_collection(COLLECTION)
        logger.warning("Deleted collection %s", COLLECTION)


STORE_SETTINGS = {"embed_model": EMBED_MODEL,
                  "chunk_chars": CHUNK_CHARS, "overlap_chars": OVERLAP_CHARS}


def open_store(check=True):
    col = _client().get_or_create_collection(
        COLLECTION, embedding_function=None, configuration={"hnsw": {"space": "cosine"}},
        metadata=STORE_SETTINGS)
    stored = col.metadata or {}
    changed = [f"{k}: {stored.get(k)} -> {v}" for k, v in STORE_SETTINGS.items() if stored.get(k) != v]
    if changed and check:
        fail(f"The {COLLECTION} store was built with different settings ({'; '.join(changed)}).\n"
             "Run:  python -m ext_rag.ingest --rebuild")
    logger.debug("Opened %s in %s (%d chunks)", COLLECTION, STORE_DIR, col.count())
    return col


def check_dims(col, n, record=False):
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
             "Run:  python -m ext_rag.ingest --rebuild")


def search(col, qvec, k, where=None):
    """The k chunks closest to a query vector, most similar first, as dicts:
    {"id", "sim", "path", "loc", "text"}. Supports optional Chroma `where` metadata filtering."""
    if where is not None:
        available = len(col.get(where=where, include=[]).get("ids") or [])
    else:
        available = col.count()
    k = min(k, available)
    if k == 0:
        return []
    check_dims(col, len(qvec))
    query_kwargs = {
        "query_embeddings": [qvec],
        "n_results": k,
        "include": ["distances", "documents", "metadatas"],
    }
    if where is not None:
        query_kwargs["where"] = where
    r = col.query(**query_kwargs)
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
    {hash: {path, file_type, added, chunks, skipped, stored, complete}}."""
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
