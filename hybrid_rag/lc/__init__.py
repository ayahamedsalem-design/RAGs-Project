"""LangChain LCEL implementations of the Hybrid RAG and 2-Round Hybrid CRAG pipelines.

Contains:
- ``components.py``: Shared OpenRouter models (with cost hook), ChromaDB vector store,
  Arabic-aware BM25 retriever, RRF fusion, and session-scoped chat history loader.
- ``hybrid_lc.py``: LangChain LCEL equivalent of ``hybrid_rag/hybrid.py``.
- ``hybrid_corrective_lc.py``: LangChain LCEL equivalent of ``hybrid_rag/hybrid_corrective.py``.

Named ``lc`` instead of ``langchain`` so it never shadows the installed ``langchain`` package.
"""
