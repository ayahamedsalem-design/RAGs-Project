"""Token and USD cost tracker for OpenRouter chat and embedding requests.

Every request made through ``initialize_api()`` is intercepted by an httpx response hook
and aggregated per question/document via a ``ContextVar``, then persisted to ``logs/cost_log.csv``.
"""

# What each question (and each ingested document) cost: tokens and US dollars, per step.
#
# Every request to OpenRouter goes through the client made by initialize_api()
# (hybrid_rag/pdf_processing.py), whose hook calls record() here for each reply. So cost is
# recorded in one place instead of in every pipeline.
#
#   with cost.track("hybrid", question) as qc:    # or start_query() ... finish_query()
#       ...model calls...                          # each one is recorded as a Call
#   print(qc.summary())                            # the "=== Cost ===" block
#
# Where the numbers come from:
#   "reported"    the cost OpenRouter actually charged (usage.cost in its reply). Preferred.
#   "calculated"  the reply had no cost: tokens x the model's prices from OpenRouter's model
#                 list (cached in logs/model_prices.json, refreshed after 24 hours)
#   "local"       Ollama: $0, but the tokens are still counted
#
# Saved to logs/cost_log.csv (one row per question or document; the Usage page reads it).
# The tracking lives in a ContextVar, so parallel Streamlit sessions never mix their costs.

import csv, json, time, logging, contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import requests

logger = logging.getLogger("rag.cost")

ROOT = Path(__file__).resolve().parent.parent          # the project folder
COST_LOG = ROOT / "logs" / "cost_log.csv"
PRICES_FILE = ROOT / "logs" / "model_prices.json"
PRICES_MAX_AGE = 24 * 3600
PRICE_LISTS = ["https://openrouter.ai/api/v1/models",              # chat models
               "https://openrouter.ai/api/v1/embeddings/models"]   # embedding models
CSV_FIELDS = ["time", "system", "backend", "chat_model", "embedding_model", "question", "n_calls",
              "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_usd", "steps"]


@dataclass
class Call:
    """One model request."""
    step: str               # rewrite | embed_query | grade | sufficiency | embed_extra |
                            # grade_round2 | sufficiency_round2 | answer | embed_chunks
    model: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cached_tokens: int
    cost_usd: float
    source: str             # reported | calculated | local
    seconds: float


@dataclass
class QueryCost:
    """Everything one question (or one ingested document) cost."""
    system: str             # naive | crag | ingest
    question: str
    calls: list = field(default_factory=list)

    @property
    def total_usd(self):
        return sum(c.cost_usd for c in self.calls)

    @property
    def prompt_tokens(self):
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def completion_tokens(self):
        return sum(c.completion_tokens for c in self.calls)

    @property
    def reasoning_tokens(self):
        return sum(c.reasoning_tokens for c in self.calls)

    @property
    def cached_tokens(self):
        return sum(c.cached_tokens for c in self.calls)

    @property
    def n_calls(self):
        return len(self.calls)

    @property
    def seconds(self):
        return sum(c.seconds for c in self.calls)

    def by_step(self):
        """{step: {calls, prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens,
        usd, source}} in the order the steps first happened."""
        steps = {}
        for c in self.calls:
            s = steps.setdefault(c.step, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                          "reasoning_tokens": 0, "cached_tokens": 0, "usd": 0.0,
                                          "sources": set()})
            s["calls"] += 1
            s["prompt_tokens"] += c.prompt_tokens
            s["completion_tokens"] += c.completion_tokens
            s["reasoning_tokens"] += c.reasoning_tokens
            s["cached_tokens"] += c.cached_tokens
            s["usd"] += c.cost_usd
            s["sources"].add(c.source)
        for s in steps.values():
            s["source"] = "/".join(sorted(s.pop("sources")))
        return steps

    def summary(self):
        """A short block for the terminal and the log."""
        lines = [f"Total: ${self.total_usd:.6f}  |  {self.n_calls} calls  |  tokens in "
                 f"{self.prompt_tokens:,} / out {self.completion_tokens:,} / reasoning "
                 f"{self.reasoning_tokens:,}  |  {self.seconds:.1f}s in model calls",
                 f"{'STEP':20} {'CALLS':>5} {'IN':>8} {'OUT':>7} {'REASON':>7} {'USD':>11}  SOURCE"]
        for step, s in self.by_step().items():
            lines.append(f"{step:20} {s['calls']:5} {s['prompt_tokens']:8,} {s['completion_tokens']:7,} "
                         f"{s['reasoning_tokens']:7,} {s['usd']:11.6f}  {s['source']}")
        return "\n".join(lines)


_current = contextvars.ContextVar("rag_cost_current", default=None)
_last = contextvars.ContextVar("rag_cost_last", default=None)


def start_query(system, question):
    """Begin tracking the cost of one question (or document)."""
    qc = QueryCost(system, question)
    _current.set(qc)
    return qc


def finish_query(save=True):
    """Stop tracking; save the row to cost_log.csv (unless save=False) and return it."""
    qc = _current.get()
    _current.set(None)
    if qc is None:
        return None
    _last.set(qc)
    if not qc.calls:
        return qc                 # nothing was asked of a model (e.g. the store is empty)
    if save:
        _save(qc)
    logger.info("Cost of this %s %s: $%.6f, %d calls, %d tokens in / %d out / %d reasoning",
                qc.system, "document" if qc.system == "ingest" else "question", qc.total_usd,
                qc.n_calls, qc.prompt_tokens, qc.completion_tokens, qc.reasoning_tokens)
    return qc


@contextmanager
def track(system, question):
    """start_query() ... finish_query(). If the work fails half-way, the money already
    spent is logged as a warning but not saved as a row (it isn't a finished question)."""
    qc = start_query(system, question)
    try:
        yield qc
    except BaseException:
        finish_query(save=False)
        if qc.calls:
            logger.warning("Stopped after %d calls that cost $%.6f (not saved to %s)",
                           qc.n_calls, qc.total_usd, COST_LOG.name)
        raise
    finish_query()


def last_query():
    """The QueryCost of the last question finished in this session (for the UI)."""
    return _last.get()


def record(step, model, usage, seconds, local=False):
    """Add one request to the question being tracked. usage is OpenRouter's usage object
    (or, for Ollama, {"prompt_tokens", "completion_tokens"})."""
    usage = usage or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    reasoning = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    if local:
        usd, source = 0.0, "local"
    elif usage.get("cost") is not None:
        usd, source = float(usage["cost"]), "reported"
    else:
        usd, source = calculate_cost(model, prompt, completion, cached), "calculated"
    call = Call(step or "other", model, prompt, completion, reasoning, cached, usd, source,
                round(seconds, 3))
    logger.debug("cost %s %s: %d in / %d out / %d reasoning / %d cached tokens, $%.6f (%s)",
                 call.step, model, prompt, completion, reasoning, cached, usd, source)
    qc = _current.get()
    if qc is None:
        logger.debug("No question is being tracked; this call is not counted")
    else:
        qc.calls.append(call)
    return call


# ---------- Fallback: calculate the cost from OpenRouter's price list ----------
def calculate_cost(model, prompt_tokens, completion_tokens, cached_tokens=0):
    """Tokens x prices (USD per token). Cached prompt tokens are billed at the cache-read price."""
    p = model_prices().get(model)
    if p is None:
        logger.warning("No price found for %s; counting its cost as $0", model)
        return 0.0
    cache_read = p["input_cache_read"] if p["input_cache_read"] is not None else p["prompt"]
    return (prompt_tokens * p["prompt"] + completion_tokens * p["completion"]
            - cached_tokens * (p["prompt"] - cache_read))


def model_prices():
    """{model id: {prompt, completion, input_cache_read}} in USD per token, from the cache
    file if it is less than a day old, otherwise from OpenRouter."""
    try:
        cached = json.loads(PRICES_FILE.read_text(encoding="utf-8"))
        if time.time() - cached["fetched"] < PRICES_MAX_AGE:
            return cached["prices"]
    except (OSError, ValueError, KeyError):
        cached = None
    prices = {}
    try:
        for url in PRICE_LISTS:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            for m in r.json()["data"]:
                pr = m.get("pricing") or {}
                num = lambda k: float(pr[k]) if pr.get(k) not in (None, "") else None
                prices[m["id"]] = {"prompt": num("prompt") or 0.0,
                                   "completion": num("completion") or 0.0,
                                   "input_cache_read": num("input_cache_read")}
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.warning("Could not load OpenRouter's price list (%s)", e)
        return cached["prices"] if cached else {}
    PRICES_FILE.parent.mkdir(parents=True, exist_ok=True)
    PRICES_FILE.write_text(json.dumps({"fetched": time.time(), "prices": prices}), encoding="utf-8")
    logger.debug("Loaded prices of %d models from OpenRouter", len(prices))
    return prices


# ---------- Saving ----------
def _save(qc):
    """Appends one completed QueryCost summary row to ``logs/cost_log.csv``."""
    import os
    steps = {k: {**v, "usd": round(v["usd"], 8)} for k, v in qc.by_step().items()}
    row = {"time": datetime.now().isoformat(timespec="seconds"), "system": qc.system,
           "backend": "openrouter", "chat_model": os.getenv("CHAT_MODEL"),
           "embedding_model": os.getenv("EMBEDDING_MODEL"),
           "question": qc.question, "n_calls": qc.n_calls, "prompt_tokens": qc.prompt_tokens,
           "completion_tokens": qc.completion_tokens, "reasoning_tokens": qc.reasoning_tokens,
           "total_usd": f"{qc.total_usd:.8f}", "steps": json.dumps(steps)}
    COST_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not COST_LOG.exists()
    try:
        with open(COST_LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
    except OSError as e:          # e.g. the file is open in Excel: never lose the answer over it
        logger.warning("Could not write %s (%s); cost was $%.6f", COST_LOG.name, e, qc.total_usd)
