# Shared settings for the evaluation of all seven RAG pipelines (see run_eval.py).

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"                         # answers, judgements, cost log of the runs
ANSWER_KEY = HERE / "answer_key.json"
PAPERS_DIR = Path.home() / "Desktop" / "papers"    # the two test PDFs

DOCS = {"battery": "HummesETAL_Lajer_2023-2.pdf",
        "meat": "Advice For Risk Associated with Cultured Meat.pdf"}

# (name in the results and cost log, name in the app, who built it)
PIPELINES = [
    ("naive", "Ext Naive RAG", "RAG/"),
    ("crag", "Ext Corrective RAG", "RAG/"),
    ("basic", "Basic RAG (llama3.2:1b, as in the app)", "RAGs/"),
    ("basic_deepseek", "Basic RAG (DeepSeek)", "RAGs/"),
    ("parent_child", "Parent-Child (Hierarchical) RAG", "RAGs/"),
    ("hybrid", "Hybrid RAG", "RAGs/"),
    ("crag_parent_child", "Corrective Parent-Child RAG", "RAGs/"),
    ("crag_hybrid", "Corrective Hybrid RAG", "RAGs/"),
]
LABEL = {name: label for name, label, _ in PIPELINES}


def answer_key():
    return json.loads(ANSWER_KEY.read_text(encoding="utf-8"))


def answers_file(system):
    return RESULTS / f"answers_{system}.json"


def judged_file(system):
    return RESULTS / f"judged_{system}.json"


def load(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
