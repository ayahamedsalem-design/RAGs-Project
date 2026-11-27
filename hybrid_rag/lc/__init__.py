"""LangChain LCEL and LangGraph implementations of the RAG pipelines.

Contains:
- ``components.py``: Shared OpenRouter models (with cost hook), ChromaDB vector store,
  Arabic-aware BM25 retriever, RRF fusion, and session-scoped chat history loader.
- ``hybrid_lc.py``: LangChain LCEL equivalent of ``hybrid_rag/hybrid.py``.
- ``hybrid_corrective_lc.py``: LangChain LCEL equivalent of ``hybrid_rag/hybrid_corrective.py``.
- ``crag_graph.py``: Self-Corrective 2-Round CRAG LangGraph ``StateGraph`` with answer verification.
- ``agent_graph.py``: Tool-Calling Agentic RAG LangGraph orchestrating fast/deep CRAG, structured search, and history summarization.

Named ``lc`` instead of ``langchain`` so it never shadows the installed ``langchain`` package.
"""
