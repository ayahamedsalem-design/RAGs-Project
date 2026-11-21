# Shared settings for the evaluation of all seven RAG pipelines (see run_eval.py).

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                 # the project folder
RESULTS = HERE / "results"                         # answers, judgements, cost log of the runs
ANSWER_KEY = HERE / "answer_key.json"
PAPERS_DIR = ROOT / "documents"                    # the two test PDFs

DOCS = {"battery": "HummesETAL_Lajer_2023-2.pdf",
        "meat": "Advice For Risk Associated with Cultured Meat.pdf"}

# (name in the results and cost log, name in the app, module)
PIPELINES = [
    ("hybrid", "Hybrid RAG", "hybrid_rag/hybrid.py"),
    ("crag_hybrid", "Corrective Hybrid RAG", "hybrid_rag/corrective_hybrid.py"),
    ("basic", "Basic RAG", "other_rags/basic.py"),
    ("parent_child", "Parent-Child (Hierarchical) RAG", "other_rags/parent_child.py"),
    ("crag_parent_child", "Corrective Parent-Child RAG", "other_rags/corrective_parent_child.py"),
    ("naive", "Ext Naive RAG", "ext_rag/ask.py"),
    ("crag", "Ext Corrective RAG", "ext_rag/ask_crag.py"),
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
