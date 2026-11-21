# Retrieval Evaluation Report: All 7 RAG Pipelines (Pre-Answer-LLM)

Generated: `2026-11-22 00:21:25` | Test set: **80 questions** (4 PDFs × 20 Q&As from `docs/`) | Scope: **Global search across all 4 PDFs simultaneously**

> **Focus:** Pure retrieval quality of the chunks retrieved by each RAG architecture **before the final Answer LLM**. For the 3 Corrective pipelines (`Corrective Hybrid`, `Corrective Parent-Child`, `Ext Corrective`), both **Pre-Grading (raw retrieval)** and **Post-Grading (filtered chunks handed to the Answer LLM)** are measured side by side.

## 1. Common Evaluation Setup
| Item | Value |
| --- | --- |
| Documents (4 PDFs) | Battery Technology (`HummesETAL_Lajer_2023-2.pdf`: hybrid=91, ext=91, parents=33, children=196); Bishop PRML (`Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf`: hybrid=2315, ext=2315, parents=951, children=5705); Cultured Meat (`Advice For Risk Associated with Cultured Meat.pdf`: hybrid=201, ext=201, parents=72, children=431); Neural Network (AlexNet) (`imagenet-classification-with-deep-convolutional-neural-networks-Paper.pdf`: hybrid=47, ext=47, parents=19, children=113) |
| Total Chunks per Store | Standard (`global_hybrid_collection`): 2,654 \| Ext (`ext_chunks_openrouter`): 2,654 \| Hierarchical (`parent` / `child`): 1,075 / 6,445 |
| Embedding Model | `nvidia/nemotron-3-embed-1b:free` via OpenRouter |
| Grader LLM (CRAG only) | `nvidia/nemotron-3.5-super-vl:free` via OpenRouter (Answer LLM NOT called) |
| Search Scope | Global search across all 4 PDFs simultaneously |
| Test Questions | 80 answerable questions (20 Battery, 20 Bishop PRML, 20 Cultured Meat, 20 AlexNet) |
| Relevance Grading | Deterministic dual check: (1) Exact PDF & Page match via 8-word text anchor locator (works for Parent-Child too); (2) Answer-Fact match verifying distinctive answer terms & numbers inside the chunk text. |

## 2. Comparison of All 7 Pipelines (Pre-Answer-LLM)

### Table 2.1: Final Chunks Handed to Answer LLM (Post-Grading for CRAG, Raw Retrieval for Non-CRAG)
| Metric | Hybrid | CRAG Hybrid | Basic | Parent-Child | CRAG Parent-Child | Ext Naive | Ext CRAG |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Questions Evaluated | 80/80 | 80/80 | 80/80 | 80/80 | 80/80 | 80/80 | 80/80 |
| Avg Chunks Returned (K) | 6.0 | 2.6 | 3.0 | 5.8 | 2.6 | 4.0 | 2.5 |
| Document Accuracy (Source Hit@K) | 100% | 98.8% | 100% | 100% | 100% | 100% | 100% |
| Document Precision (Right-PDF %) | 93.1% | 98.3% | 97.1% | 91% | 98.8% | 95.6% | 100% |
| Page Hit Rate@K | 100% | 92.5% | 96.2% | 98.8% | 96.2% | 97.5% | 97.5% |
| Page MRR | 0.81 | 0.82 | 0.85 | 0.89 | 0.90 | 0.86 | 0.91 |
| Answer-Fact Hit Rate@K | 98.8% | 90% | 92.5% | 98.8% | 96.2% | 95% | 98.8% |
| Full-Answer Coverage (Recall) | 91.2% | 78.8% | 83.8% | 97.5% | 92.5% | 86.2% | 91.2% |
| Avg Answer-Term Recall | 92.8% | 84.6% | 88% | 96.8% | 92.6% | 90.7% | 92.7% |
| Precision@K (Signal / Total K) | 33.5% | 61.3% | 50.4% | 34.8% | 63.2% | 42.2% | 83.4% |
| MRR (Answer-Fact) | 0.83 | 0.83 | 0.81 | 0.89 | 0.90 | 0.82 | 0.93 |
| nDCG@K (Answer-Fact) | 0.84 | 0.84 | 0.84 | 0.89 | 0.91 | 0.85 | 0.94 |
| Est. API Requests / Question | 1.0 | 5.0 | 1.0 | 1.0 | 5.0 | 1.0 | 2.0 |
| Measured Cost / Question ($) | $0.000000 | $0.000000 | $0.000000 | $0.000000 | $0.000000 | $0.000000 | $0.000000 |
| Median Retrieval Time (s) | 0.02s | 5.88s | 0.01s | 0.01s | 4.31s | 0.20s | 1.51s |

### Table 2.2: CRAG Grader Impact — Raw Retrieval (Pre-Grading) vs Filtered (Post-Grading)
| Pipeline | Avg K (Pre->Post) | Page Hit | Fact Hit | Precision@K | MRR | Noise Removed | False Drops | Qs Lost / Gained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Corrective Hybrid RAG | 4.0 -> 2.6 | 93.8% -> 92.5% | 93.8% -> 90% | 41.9% -> 61.3% | 0.82 -> 0.83 | 52.2% | 10.4% | -3 |
| Corrective Parent-Child RAG | 4.0 -> 2.6 | 98.8% -> 96.2% | 97.5% -> 96.2% | 42.5% -> 63.2% | 0.89 -> 0.90 | 52.2% | 13.2% | -1 |
| Ext Corrective RAG | 10.0 -> 2.5 | 100% -> 97.5% | 98.8% -> 98.8% | 25.2% -> 83.4% | 0.83 -> 0.93 | 92% | 27.2% | -0 / +0 |

### Key Findings
- **Best Retrieval Hit Rate (Pre-Answer-LLM):** Ext Corrective RAG led with 98.8% Answer-Fact Hit@K (97.5% Page Hit@K, 91.2% full coverage, MRR 0.93).
- **Best Precision / Lowest Noise:** Ext Corrective RAG achieved 83.4% Precision@K (avg 2.5 chunks handed to the Answer LLM).
- **Best Ranking Quality (MRR / nDCG):** Ext Corrective RAG scored MRR 0.93 and nDCG@K 0.94.
- **Hybrid (Dense + BM25) vs Basic (Dense-Only) on the same store:** Hybrid achieved 98.8% Fact Hit@K (MRR 0.83, Page Hit 100%) vs Basic's 92.5% Fact Hit@K (MRR 0.81, Page Hit 96.2%).
- **Corrective Hybrid RAG Grader Impact (Pre -> Post):** Cut avg chunks from 4.0 -> 2.6, removed 52.2% of irrelevant chunks, moved Precision@K from 41.9% -> 61.3%, with a 10.4% relevant-chunk false drop rate (questions lost: 3).
- **Corrective Parent-Child RAG Grader Impact (Pre -> Post):** Cut avg chunks from 4.0 -> 2.6, removed 52.2% of irrelevant chunks, moved Precision@K from 42.5% -> 63.2%, with a 13.2% relevant-chunk false drop rate (questions lost: 1).
- **Ext Corrective RAG Grader Impact (Pre -> Post):** Cut avg chunks from 10.0 -> 2.5, removed 92% of irrelevant chunks, moved Precision@K from 25.2% -> 83.4%, with a 27.2% relevant-chunk false drop rate (questions lost: 0; gained via Round-2 correction: 0).
- **Document Difficulty (across all pipelines):** Hardest document was Battery Technology (avg 90.7% Fact Hit@K); easiest was Neural Network (AlexNet) (avg 97.9% Fact Hit@K).

## 3. Metrics & Formulas
| Metric | What it measures | Formula |
| --- | --- | --- |
| Document Accuracy | At least one retrieved chunk comes from the target PDF | Doc hits ÷ N |
| Document Precision | Share of retrieved chunks belonging to the target PDF (1 - cross-doc bleed) | Mean(right-PDF chunks ÷ K) |
| Page Hit Rate@K | At least one retrieved chunk spans the exact ground-truth PDF page | Page hits ÷ N |
| Page MRR | Ranking position of the first chunk from the exact ground-truth page | Mean(1 ÷ page_rank), miss = 0 |
| Answer-Fact Hit Rate@K | At least one retrieved chunk contains verified ground-truth answer evidence | Fact hits ÷ N |
| Full-Answer Coverage | Combined target-PDF chunks cover >=65% of verified answer terms & >=80% critical numbers | Fully covered ÷ N |
| Precision@K | Fraction of retrieved chunks that actually contain relevant answer evidence | Mean(relevant chunks ÷ K) |
| MRR (Answer-Fact) | How high the first relevant answer-evidence chunk is ranked | Mean(1 ÷ fact_rank), miss = 0 |
| nDCG@K | Ranking quality across all relevant chunks in top-K (higher = relevant chunks at top) | Mean(DCG@K ÷ IDCG@K) |
| Noise Removed % (CRAG) | Irrelevant pre-grading candidates correctly dropped by the CRAG grader | Dropped noise ÷ total pre noise |
| False Drop Rate % (CRAG) | Relevant pre-grading candidates mistakenly dropped by the CRAG grader | Dropped relevant ÷ total pre relevant |

## 4. Hybrid RAG

### Table 4.1: Setup
| Item | Value |
| --- | --- |
| Vector store | ChromaDB (chroma_db_global / global_hybrid_collection) |
| Chunking | 1,000 characters, 200 overlap (RecursiveCharacterTextSplitter) |
| Pre-LLM Retrieval | Per sub-question: Dense vector (top 10) + BM25 keyword (top 10), fused via RRF (k=60) -> top 6 chunks |
| Grading / Filter | None (all 6 chunks go directly to the Answer LLM) |
| Questions Evaluated | 80 (0 errors) |

### Table 4.2: Metrics, Calculations & Interpretation
| Metric | Calculation & Result | Interpretation |
| --- | --- | --- |
| Document Accuracy | 80 ÷ 80 = 100% | Every question retrieved from the right PDF |
| Document Precision | Mean right-PDF share = 93.1% | Minimal cross-document bleed |
| Page Hit Rate@K | 80 ÷ 80 = 100% | Every question hit the exact ground-truth PDF page |
| Page MRR | 65.18 ÷ 80 = 0.81 | Ranks: rank 1 for 57, rank 2 for 9, rank 3 for 6, rank 4 for 3, rank 5 for 3, rank 6 for 2 |
| Answer-Fact Hit Rate@K | 79 ÷ 80 = 98.8% | No answer-evidence chunk on 1 Qs (Q7) |
| Full-Answer Coverage | 73 ÷ 80 = 91.2% | Incomplete facts on 7 Qs (Q7, Q29, Q38, Q64, Q73, Q74, Q78) |
| Avg Answer-Term Recall | Mean combined term recall = 92.8% | Fraction of verified answer terms present in context |
| Precision@K | Mean relevant/K (avg K=6.0) = 33.5% | Moderate noise among top-K chunks |
| MRR (Answer-Fact) | 66.10 ÷ 80 = 0.83 (rank 1 for 59, rank 2 for 8, rank 3 for 4, rank 4 for 4, rank 5 for 3, rank 6 for 1) | First relevant chunk is usually rank 1 |
| nDCG@K | Mean nDCG@K = 0.84 | Strong top-heavy ranking of relevant chunks |
| Requests / Cost / Latency | 1.0 req/q · $0.000000/q (total $0.00000) · median 0.02s | Slowest: Q1 (0.5s) |

### Table 4.3: Results by Document (20 questions each)
| Document | Qs | Doc Hit | Page Hit | Fact Hit | Coverage | Precision@K | MRR | nDCG@K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Battery Technology | 20 | 100% | 100% | 95% | 95% | 30.8% | 0.78 | 0.76 |
| Bishop PRML | 20 | 100% | 100% | 100% | 90% | 35% | 0.79 | 0.82 |
| Cultured Meat | 20 | 100% | 100% | 100% | 100% | 29.2% | 0.87 | 0.89 |
| Neural Network (AlexNet) | 20 | 100% | 100% | 100% | 80% | 39.2% | 0.87 | 0.90 |

### Table 4.4: Results by Question Category
| Category | Qs | Page Hit | Fact Hit | Coverage | Precision@K | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| Single fact | 19 | 100% | 100% | 89.5% | 35.1% | 0.93 |
| Numbers / Hyperparameters | 11 | 100% | 100% | 100% | 36.4% | 0.73 |
| Definition / Concept | 9 | 100% | 100% | 88.9% | 29.6% | 0.77 |
| Mechanism / Why | 17 | 100% | 94.1% | 82.4% | 29.4% | 0.77 |
| Multi-part / Comparison | 24 | 100% | 100% | 95.8% | 35.4% | 0.85 |

### Table 4.5: Top-1 Cosine Similarity Analysis
| Group | Questions | Avg Top-1 Cosine Similarity |
| --- | --- | --- |
| Answer-Fact Hit | 79 | 0.544 |
| Answer-Fact Miss | 1 | 0.495 |

## 5. Corrective Hybrid RAG

### Table 5.1: Setup
| Item | Value |
| --- | --- |
| Vector store | ChromaDB (chroma_db_global / global_hybrid_collection) |
| Chunking | 1,000 characters, 200 overlap (RecursiveCharacterTextSplitter) |
| Pre-Grading Retrieval | Per sub-question: Dense (top 10) + BM25 (top 10) via RRF -> top 4 candidate chunks/sub-query |
| CRAG Grader | 1 binary LLM call (yes/no, temperature=0) per candidate chunk before Answer LLM |
| Questions Evaluated | 80 (0 errors) |

### Table 5.2: Metrics, Calculations & Interpretation
| Metric | Calculation & Result | Interpretation |
| --- | --- | --- |
| Document Accuracy | 79 ÷ 80 = 98.8% | Wrong PDF only for Q51 |
| Document Precision | Mean right-PDF share = 98.3% | Minimal cross-document bleed |
| Page Hit Rate@K | 74 ÷ 80 = 92.5% | Missed exact page on 6 Qs (Q1, Q2, Q19, Q26, Q39, Q51) |
| Page MRR | 65.42 ÷ 80 = 0.82 | Ranks: rank 1 for 59, rank 2 for 9, rank 3 for 5, rank 4 for 1 |
| Answer-Fact Hit Rate@K | 72 ÷ 80 = 90% | No answer-evidence chunk on 8 Qs (Q2, Q7, Q19, Q26, Q29, Q39, Q51, Q52) |
| Full-Answer Coverage | 63 ÷ 80 = 78.8% | Incomplete facts on 17 Qs (Q1, Q2, Q7, Q8, Q15, Q19, Q26, Q29 (+9 more)) |
| Avg Answer-Term Recall | Mean combined term recall = 84.6% | Fraction of verified answer terms present in context |
| Precision@K | Mean relevant/K (avg K=2.6) = 61.3% | High signal-to-noise ratio |
| MRR (Answer-Fact) | 66.75 ÷ 80 = 0.83 (rank 1 for 63, rank 2 for 5, rank 3 for 3, rank 4 for 1) | First relevant chunk is usually rank 1 |
| nDCG@K | Mean nDCG@K = 0.84 | Strong top-heavy ranking of relevant chunks |
| Requests / Cost / Latency | 5.0 req/q · $0.000000/q (total $0.00000) · median 5.88s | Slowest: Q28 (20.7s) |

### Table 5.3: Results by Document (20 questions each)
| Document | Qs | Doc Hit | Page Hit | Fact Hit | Coverage | Precision@K | MRR | nDCG@K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Battery Technology | 20 | 100% | 85% | 85% | 70% | 52.9% | 0.78 | 0.79 |
| Bishop PRML | 20 | 100% | 90% | 85% | 80% | 50.8% | 0.76 | 0.77 |
| Cultured Meat | 20 | 95% | 95% | 90% | 85% | 59.6% | 0.88 | 0.87 |
| Neural Network (AlexNet) | 20 | 100% | 100% | 100% | 80% | 81.7% | 0.93 | 0.94 |

### Table 5.4: Results by Question Category
| Category | Qs | Page Hit | Fact Hit | Coverage | Precision@K | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| Single fact | 19 | 89.5% | 94.7% | 73.7% | 61.8% | 0.91 |
| Numbers / Hyperparameters | 11 | 100% | 100% | 100% | 78% | 0.84 |
| Definition / Concept | 9 | 88.9% | 77.8% | 77.8% | 48.1% | 0.72 |
| Mechanism / Why | 17 | 100% | 88.2% | 70.6% | 57.4% | 0.81 |
| Multi-part / Comparison | 24 | 87.5% | 87.5% | 79.2% | 60.8% | 0.83 |

### Table 5.5: Top-1 Cosine Similarity Analysis
| Group | Questions | Avg Top-1 Cosine Similarity |
| --- | --- | --- |
| Answer-Fact Hit | 72 | 0.545 |
| Answer-Fact Miss | 7 | 0.500 |

## 6. Basic RAG

### Table 6.1: Setup
| Item | Value |
| --- | --- |
| Vector store | ChromaDB (chroma_db_global / global_hybrid_collection) |
| Chunking | 1,000 characters, 200 overlap (RecursiveCharacterTextSplitter) |
| Pre-LLM Retrieval | Pure dense vector search: top 3 chunks per sub-question (deduplicated) |
| Grading / Filter | None |
| Questions Evaluated | 80 (0 errors) |

### Table 6.2: Metrics, Calculations & Interpretation
| Metric | Calculation & Result | Interpretation |
| --- | --- | --- |
| Document Accuracy | 80 ÷ 80 = 100% | Every question retrieved from the right PDF |
| Document Precision | Mean right-PDF share = 97.1% | Minimal cross-document bleed |
| Page Hit Rate@K | 77 ÷ 80 = 96.2% | Missed exact page on 3 Qs (Q13, Q16, Q62) |
| Page MRR | 67.67 ÷ 80 = 0.85 | Ranks: rank 1 for 61, rank 2 for 8, rank 3 for 8 |
| Answer-Fact Hit Rate@K | 74 ÷ 80 = 92.5% | No answer-evidence chunk on 6 Qs (Q2, Q7, Q13, Q22, Q52, Q62) |
| Full-Answer Coverage | 67 ÷ 80 = 83.8% | Incomplete facts on 13 Qs (Q1, Q2, Q7, Q8, Q13, Q15, Q22, Q52 (+5 more)) |
| Avg Answer-Term Recall | Mean combined term recall = 88% | Fraction of verified answer terms present in context |
| Precision@K | Mean relevant/K (avg K=3.0) = 50.4% | High signal-to-noise ratio |
| MRR (Answer-Fact) | 65.00 ÷ 80 = 0.81 (rank 1 for 58, rank 2 for 10, rank 3 for 6) | First relevant chunk is usually rank 1 |
| nDCG@K | Mean nDCG@K = 0.84 | Strong top-heavy ranking of relevant chunks |
| Requests / Cost / Latency | 1.0 req/q · $0.000000/q (total $0.00000) · median 0.01s | Slowest: Q31 (0.0s) |

### Table 6.3: Results by Document (20 questions each)
| Document | Qs | Doc Hit | Page Hit | Fact Hit | Coverage | Precision@K | MRR | nDCG@K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Battery Technology | 20 | 100% | 90% | 85% | 70% | 38.3% | 0.72 | 0.75 |
| Bishop PRML | 20 | 100% | 100% | 95% | 95% | 58.3% | 0.77 | 0.81 |
| Cultured Meat | 20 | 100% | 100% | 95% | 90% | 46.7% | 0.88 | 0.89 |
| Neural Network (AlexNet) | 20 | 100% | 95% | 95% | 80% | 58.3% | 0.89 | 0.89 |

### Table 6.4: Results by Question Category
| Category | Qs | Page Hit | Fact Hit | Coverage | Precision@K | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| Single fact | 19 | 94.7% | 100% | 89.5% | 52.6% | 0.89 |
| Numbers / Hyperparameters | 11 | 81.8% | 81.8% | 81.8% | 45.5% | 0.76 |
| Definition / Concept | 9 | 100% | 88.9% | 88.9% | 48.1% | 0.69 |
| Mechanism / Why | 17 | 100% | 88.2% | 76.5% | 47.1% | 0.78 |
| Multi-part / Comparison | 24 | 100% | 95.8% | 83.3% | 54.2% | 0.85 |

### Table 6.5: Top-1 Cosine Similarity Analysis
| Group | Questions | Avg Top-1 Cosine Similarity |
| --- | --- | --- |
| Answer-Fact Hit | 74 | 0.552 |
| Answer-Fact Miss | 6 | 0.467 |

## 7. Parent-Child (Hierarchical) RAG

### Table 7.1: Setup
| Item | Value |
| --- | --- |
| Vector store | ChromaDB (global_child_collection + global_parent_collection) |
| Chunking | 300-word parents / 70-word children with 15-word overlap (no page metadata stored) |
| Pre-LLM Retrieval | Top 10 child chunks per sub-question -> up to 6 unique parent blocks |
| Grading / Filter | None |
| Questions Evaluated | 80 (0 errors) |

### Table 7.2: Metrics, Calculations & Interpretation
| Metric | Calculation & Result | Interpretation |
| --- | --- | --- |
| Document Accuracy | 80 ÷ 80 = 100% | Every question retrieved from the right PDF |
| Document Precision | Mean right-PDF share = 91% | Minimal cross-document bleed |
| Page Hit Rate@K | 79 ÷ 80 = 98.8% | Missed exact page on 1 Qs (Q62) |
| Page MRR | 71.25 ÷ 80 = 0.89 | Ranks: rank 1 for 66, rank 2 for 7, rank 3 for 3, rank 4 for 3 |
| Answer-Fact Hit Rate@K | 79 ÷ 80 = 98.8% | No answer-evidence chunk on 1 Qs (Q62) |
| Full-Answer Coverage | 78 ÷ 80 = 97.5% | Incomplete facts on 2 Qs (Q62, Q74) |
| Avg Answer-Term Recall | Mean combined term recall = 96.8% | Fraction of verified answer terms present in context |
| Precision@K | Mean relevant/K (avg K=5.8) = 34.8% | Moderate noise among top-K chunks |
| MRR (Answer-Fact) | 71.17 ÷ 80 = 0.89 (rank 1 for 66, rank 2 for 7, rank 3 for 3, rank 4 for 2, rank 6 for 1) | First relevant chunk is usually rank 1 |
| nDCG@K | Mean nDCG@K = 0.89 | Strong top-heavy ranking of relevant chunks |
| Requests / Cost / Latency | 1.0 req/q · $0.000000/q (total $0.00000) · median 0.01s | Slowest: Q1 (0.5s) |

### Table 7.3: Results by Document (20 questions each)
| Document | Qs | Doc Hit | Page Hit | Fact Hit | Coverage | Precision@K | MRR | nDCG@K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Battery Technology | 20 | 100% | 100% | 100% | 100% | 37% | 0.87 | 0.88 |
| Bishop PRML | 20 | 100% | 100% | 100% | 100% | 34.2% | 0.85 | 0.85 |
| Cultured Meat | 20 | 100% | 100% | 100% | 100% | 31.8% | 0.92 | 0.91 |
| Neural Network (AlexNet) | 20 | 100% | 95% | 95% | 90% | 36.2% | 0.93 | 0.91 |

### Table 7.4: Results by Question Category
| Category | Qs | Page Hit | Fact Hit | Coverage | Precision@K | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| Single fact | 19 | 100% | 100% | 100% | 46.1% | 0.94 |
| Numbers / Hyperparameters | 11 | 90.9% | 90.9% | 90.9% | 33% | 0.80 |
| Definition / Concept | 9 | 100% | 100% | 100% | 26.9% | 0.85 |
| Mechanism / Why | 17 | 100% | 100% | 94.1% | 30.4% | 0.89 |
| Multi-part / Comparison | 24 | 100% | 100% | 100% | 32.8% | 0.91 |

### Table 7.5: Top-1 Cosine Similarity Analysis
| Group | Questions | Avg Top-1 Cosine Similarity |
| --- | --- | --- |
| Answer-Fact Hit | 79 | 0.586 |
| Answer-Fact Miss | 1 | 0.457 |

## 8. Corrective Parent-Child RAG

### Table 8.1: Setup
| Item | Value |
| --- | --- |
| Vector store | ChromaDB (global_child_collection + global_parent_collection) |
| Chunking | 300-word parents / 70-word children with 15-word overlap |
| Pre-Grading Retrieval | Top 10 child chunks per sub-question -> up to 4 unique parent blocks/sub-query |
| CRAG Grader | 1 binary LLM call (yes/no, temperature=0) per parent candidate before Answer LLM |
| Questions Evaluated | 80 (0 errors) |

### Table 8.2: Metrics, Calculations & Interpretation
| Metric | Calculation & Result | Interpretation |
| --- | --- | --- |
| Document Accuracy | 80 ÷ 80 = 100% | Every question retrieved from the right PDF |
| Document Precision | Mean right-PDF share = 98.8% | Minimal cross-document bleed |
| Page Hit Rate@K | 77 ÷ 80 = 96.2% | Missed exact page on 3 Qs (Q2, Q7, Q62) |
| Page MRR | 71.83 ÷ 80 = 0.90 | Ranks: rank 1 for 68, rank 2 for 5, rank 3 for 4 |
| Answer-Fact Hit Rate@K | 77 ÷ 80 = 96.2% | No answer-evidence chunk on 3 Qs (Q2, Q7, Q62) |
| Full-Answer Coverage | 74 ÷ 80 = 92.5% | Incomplete facts on 6 Qs (Q1, Q2, Q7, Q57, Q62, Q74) |
| Avg Answer-Term Recall | Mean combined term recall = 92.6% | Fraction of verified answer terms present in context |
| Precision@K | Mean relevant/K (avg K=2.6) = 63.2% | High signal-to-noise ratio |
| MRR (Answer-Fact) | 71.83 ÷ 80 = 0.90 (rank 1 for 68, rank 2 for 5, rank 3 for 4) | First relevant chunk is usually rank 1 |
| nDCG@K | Mean nDCG@K = 0.91 | Strong top-heavy ranking of relevant chunks |
| Requests / Cost / Latency | 5.0 req/q · $0.000000/q (total $0.00000) · median 4.31s | Slowest: Q41 (18.8s) |

### Table 8.3: Results by Document (20 questions each)
| Document | Qs | Doc Hit | Page Hit | Fact Hit | Coverage | Precision@K | MRR | nDCG@K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Battery Technology | 20 | 100% | 90% | 90% | 85% | 58.3% | 0.84 | 0.86 |
| Bishop PRML | 20 | 100% | 100% | 100% | 100% | 50.4% | 0.85 | 0.88 |
| Cultured Meat | 20 | 100% | 100% | 100% | 95% | 62.5% | 0.95 | 0.95 |
| Neural Network (AlexNet) | 20 | 100% | 95% | 95% | 90% | 81.7% | 0.95 | 0.95 |

### Table 8.4: Results by Question Category
| Category | Qs | Page Hit | Fact Hit | Coverage | Precision@K | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| Single fact | 19 | 100% | 100% | 94.7% | 74.6% | 1.00 |
| Numbers / Hyperparameters | 11 | 90.9% | 90.9% | 90.9% | 75.8% | 0.85 |
| Definition / Concept | 9 | 100% | 100% | 100% | 48.1% | 0.85 |
| Mechanism / Why | 17 | 94.1% | 94.1% | 88.2% | 57.8% | 0.88 |
| Multi-part / Comparison | 24 | 95.8% | 95.8% | 91.7% | 58% | 0.87 |

### Table 8.5: Top-1 Cosine Similarity Analysis
| Group | Questions | Avg Top-1 Cosine Similarity |
| --- | --- | --- |
| Answer-Fact Hit | 77 | 0.585 |
| Answer-Fact Miss | 3 | 0.464 |

## 9. Ext Naive RAG

### Table 9.1: Setup
| Item | Value |
| --- | --- |
| Vector store | ChromaDB (chroma_db_global / ext_chunks_openrouter) |
| Chunking | 1,000 characters, 200 overlap (L2-normalized embeddings) |
| Pre-LLM Retrieval | Cosine similarity top 4 chunks |
| Grading / Filter | None |
| Questions Evaluated | 80 (0 errors) |

### Table 9.2: Metrics, Calculations & Interpretation
| Metric | Calculation & Result | Interpretation |
| --- | --- | --- |
| Document Accuracy | 80 ÷ 80 = 100% | Every question retrieved from the right PDF |
| Document Precision | Mean right-PDF share = 95.6% | Minimal cross-document bleed |
| Page Hit Rate@K | 78 ÷ 80 = 97.5% | Missed exact page on 2 Qs (Q13, Q16) |
| Page MRR | 68.50 ÷ 80 = 0.86 | Ranks: rank 1 for 62, rank 2 for 8, rank 3 for 6, rank 4 for 2 |
| Answer-Fact Hit Rate@K | 76 ÷ 80 = 95% | No answer-evidence chunk on 4 Qs (Q2, Q7, Q13, Q52) |
| Full-Answer Coverage | 69 ÷ 80 = 86.2% | Incomplete facts on 11 Qs (Q1, Q2, Q7, Q8, Q13, Q15, Q52, Q54 (+3 more)) |
| Avg Answer-Term Recall | Mean combined term recall = 90.7% | Fraction of verified answer terms present in context |
| Precision@K | Mean relevant/K (avg K=4.0) = 42.2% | Moderate noise among top-K chunks |
| MRR (Answer-Fact) | 65.83 ÷ 80 = 0.82 (rank 1 for 58, rank 2 for 12, rank 3 for 4, rank 4 for 2) | First relevant chunk is usually rank 1 |
| nDCG@K | Mean nDCG@K = 0.85 | Strong top-heavy ranking of relevant chunks |
| Requests / Cost / Latency | 1.0 req/q · $0.000000/q (total $0.00000) · median 0.20s | Slowest: Q1 (0.3s) |

### Table 9.3: Results by Document (20 questions each)
| Document | Qs | Doc Hit | Page Hit | Fact Hit | Coverage | Precision@K | MRR | nDCG@K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Battery Technology | 20 | 100% | 90% | 85% | 70% | 31.2% | 0.71 | 0.74 |
| Bishop PRML | 20 | 100% | 100% | 100% | 100% | 48.8% | 0.80 | 0.84 |
| Cultured Meat | 20 | 100% | 100% | 95% | 90% | 37.5% | 0.88 | 0.89 |
| Neural Network (AlexNet) | 20 | 100% | 100% | 100% | 85% | 51.2% | 0.90 | 0.91 |

### Table 9.4: Results by Question Category
| Category | Qs | Page Hit | Fact Hit | Coverage | Precision@K | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| Single fact | 19 | 94.7% | 100% | 89.5% | 44.7% | 0.88 |
| Numbers / Hyperparameters | 11 | 90.9% | 90.9% | 90.9% | 43.2% | 0.78 |
| Definition / Concept | 9 | 100% | 100% | 100% | 41.7% | 0.76 |
| Mechanism / Why | 17 | 100% | 88.2% | 76.5% | 38.2% | 0.78 |
| Multi-part / Comparison | 24 | 100% | 95.8% | 83.3% | 42.7% | 0.85 |

### Table 9.5: Top-1 Cosine Similarity Analysis
| Group | Questions | Avg Top-1 Cosine Similarity |
| --- | --- | --- |
| Answer-Fact Hit | 76 | 0.547 |
| Answer-Fact Miss | 4 | 0.514 |

## 10. Ext Corrective RAG

### Table 10.1: Setup
| Item | Value |
| --- | --- |
| Vector store | ChromaDB (chroma_db_global / ext_chunks_openrouter) |
| Chunking | 1,000 characters, 200 overlap (L2-normalized embeddings) |
| Pre-Grading Retrieval | Top 10 cosine candidates (sim >= 0.05) + reference-list filter (>= 12 citations/100w removed) |
| CRAG Grader & Correction | Batch LLM grader + sufficiency check; Round-2 query rewriting & per-file search if incomplete; capped at 6 chunks |
| Questions Evaluated | 80 (0 errors) |

### Table 10.2: Metrics, Calculations & Interpretation
| Metric | Calculation & Result | Interpretation |
| --- | --- | --- |
| Document Accuracy | 80 ÷ 80 = 100% | Every question retrieved from the right PDF |
| Document Precision | Mean right-PDF share = 100% | Minimal cross-document bleed |
| Page Hit Rate@K | 78 ÷ 80 = 97.5% | Missed exact page on 2 Qs (Q7, Q55) |
| Page MRR | 72.42 ÷ 80 = 0.91 | Ranks: rank 1 for 68, rank 2 for 7, rank 3 for 2, rank 4 for 1 |
| Answer-Fact Hit Rate@K | 79 ÷ 80 = 98.8% | No answer-evidence chunk on 1 Qs (Q7) |
| Full-Answer Coverage | 73 ÷ 80 = 91.2% | Incomplete facts on 7 Qs (Q7, Q15, Q52, Q60, Q73, Q74, Q78) |
| Avg Answer-Term Recall | Mean combined term recall = 92.7% | Fraction of verified answer terms present in context |
| Precision@K | Mean relevant/K (avg K=2.5) = 83.4% | High signal-to-noise ratio |
| MRR (Answer-Fact) | 74.25 ÷ 80 = 0.93 (rank 1 for 71, rank 2 for 5, rank 3 for 1, rank 4 for 1, rank 6 for 1) | First relevant chunk is usually rank 1 |
| nDCG@K | Mean nDCG@K = 0.94 | Strong top-heavy ranking of relevant chunks |
| Requests / Cost / Latency | 2.0 req/q · $0.000000/q (total $0.00000) · median 1.51s | Slowest: Q8 (3.7s) |

### Table 10.3: Results by Document (20 questions each)
| Document | Qs | Doc Hit | Page Hit | Fact Hit | Coverage | Precision@K | MRR | nDCG@K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Battery Technology | 20 | 100% | 95% | 95% | 90% | 80.7% | 0.86 | 0.89 |
| Bishop PRML | 20 | 100% | 100% | 100% | 100% | 78.8% | 0.92 | 0.94 |
| Cultured Meat | 20 | 100% | 95% | 100% | 90% | 88.3% | 0.96 | 0.96 |
| Neural Network (AlexNet) | 20 | 100% | 100% | 100% | 85% | 85.9% | 0.97 | 0.98 |

### Table 10.4: Results by Question Category
| Category | Qs | Page Hit | Fact Hit | Coverage | Precision@K | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| Single fact | 19 | 94.7% | 100% | 94.7% | 83.4% | 0.97 |
| Numbers / Hyperparameters | 11 | 100% | 100% | 100% | 91.5% | 0.93 |
| Definition / Concept | 9 | 100% | 100% | 100% | 74.1% | 0.89 |
| Mechanism / Why | 17 | 94.1% | 94.1% | 76.5% | 67.6% | 0.85 |
| Multi-part / Comparison | 24 | 100% | 100% | 91.7% | 94.4% | 0.96 |

### Table 10.5: Corrective Status Breakdown (Ext CRAG)
| Status | Count | Share | Questions | Meaning |
| --- | --- | --- | --- | --- |
| CORRECT | 79 | 98.8% | Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q9, Q10, Q11 (+69 more) | Round-1 retrieval after reference filter & grading was sufficient |
| CORRECTED | 1 | 1.2% | Q8 | Round-2 query rewriting / per-document search succeeded |
| PARTIAL | 0 | 0% | none | Some relevant chunks kept, but grader marked parts still missing |
| NOT FOUND | 0 | 0% | none | No chunks survived grading |
*Reference-list chunks filtered before grading: 25 | Round-2 triggered on 1/80 questions.*

### Table 10.6: Top-1 Cosine Similarity Analysis
| Group | Questions | Avg Top-1 Cosine Similarity |
| --- | --- | --- |
| Answer-Fact Hit | 79 | 0.538 |
| Answer-Fact Miss | 1 | 0.491 |

## 11. Limitations & Important Code Findings
- **Production Bug Found in `corrective_hybrid.py` & `corrective_parent_child.py` Grader**: Both files call `chat.completions.create(..., max_tokens=5)` without disabling reasoning. With `CHAT_MODEL=nvidia/nemotron-3.5-super-vl:free` (which enables reasoning by default), all 5 output tokens are consumed as reasoning tokens and `message.content` returns `None`, triggering `except Exception: return True` (accepting 100% of chunks with zero filtering). In this retrieval evaluation, `extra_body={"reasoning": {"enabled": False}}` was enabled on the grader call (matching `ext_rag/rag_common.py`) so real grading behavior could be measured.
- **Deterministic Relevance Matching**: Answer-Fact Hit@K and Full-Answer Coverage match verified ground-truth answer terms/numbers and exact PDF pages (via 8-word text anchors). A chunk conveying an answer using completely different synonyms on a non-target page may be under-counted, whereas an on-page chunk mentioning key words without stating the full relation may be generously counted.
- **No Answer LLM Calls**: By design, this evaluation stops strictly before the final Answer LLM so retrieval quality is isolated from LLM synthesis and hallucination.
- **Query Embedding Cache**: Identical query strings reuse cached vectors from the exact same `EMBEDDING_MODEL` across pipelines so free-tier quotas are preserved; retrieval outputs are mathematically identical.

## Appendix: Per-Question Retrieval Matrix (All 80 Questions)
| # | Doc | PDF p. | Category | Question (short) | Hybrid | CRAG Hybrid | Basic | Parent-Child | CRAG Parent-Child | Ext Naive | Ext CRAG |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | battery | 1 | Single fact | What is the primary objective of this... | #1 (P) | pre:1 / post:1 | #3 (P) | #1 (P) | pre:1 / post:1 | #4 (P) | pre:4 / post:2 |
| 2 | battery | 2 | Multi-part / Comparison | Which commercial vehicle was chosen as the... | #4 (P) | pre:4 / post:- (LOST) | MISS (P) | #1 (P) | pre:1 / post:- (LOST) | MISS (P) | pre:7 / post:1 |
| 3 | battery | 1 | Single fact | Which battery geometry demonstrated the best overall... | #1 (P) | pre:1 / post:1 | #2 (P) | #2 (P) | pre:2 / post:1 | #2 (P) | pre:2 / post:1 |
| 4 | battery | 3 | Mechanism / Why | Why do cylindrical cells typically have better... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 5 | battery | 3 | Single fact | What specific safety features can be integrated... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 6 | battery | 3 | Definition / Concept | What is a significant mechanical disadvantage of... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 7 | battery | 3 | Mechanism / Why | Why are prismatic cells still considered to... | MISS (P) | pre:- / post:- | MISS (P) | #6 (P) | pre:- / post:- | MISS (P) | pre:- / post:- |
| 8 | battery | 3 | Multi-part / Comparison | How does the casing of a pouch... | #1 (P) | pre:1 / post:1 | #2 (P) | #1 (P) | pre:1 / post:1 | #2 (P) | pre:2 / post:2 |
| 9 | battery | 4 | Multi-part / Comparison | What is a major thermal and safety... | #3 (P) | pre:3 / post:3 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 10 | battery | 4 | Multi-part / Comparison | What two functions are combined in a... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 11 | battery | 4 | Multi-part / Comparison | What are the three carbon fiber reinforced... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 12 | battery | 5 | Mechanism / Why | How is the assembly of the blade... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 13 | battery | 5,7 | Numbers / Hyperparameters | What is the estimated manufacturing cost per... | #4 (P) | pre:4 / post:4 | MISS (-) | #4 (P) | pre:4 / post:3 | MISS (-) | pre:5 / post:4 |
| 14 | battery | 10 | Single fact | According to the study's decision criteria, which... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 15 | battery | 12 | Mechanism / Why | Why did the structural battery score low... | #2 (P) | pre:2 / post:1 | #1 (P) | #2 (P) | pre:2 / post:2 | #1 (P) | pre:1 / post:1 |
| 16 | battery | 7 | Single fact | Which pouch battery model achieved the highest... | #1 (P) | pre:1 / post:1 | #2 (-) | #1 (P) | pre:1 / post:1 | #2 (-) | pre:2 / post:1 |
| 17 | battery | 14 | Numbers / Hyperparameters | In the capacity and weight analysis, how... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 18 | battery | 14 | Single fact | Which current-generation battery occupies the most volume... | #1 (P) | pre:1 / post:1 | #2 (P) | #1 (P) | pre:1 / post:1 | #2 (P) | pre:2 / post:1 |
| 19 | battery | 17 | Single fact | According to the range estimates, which future... | #4 (P) | pre:4 / post:- (LOST) | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 20 | battery | 17 | Multi-part / Comparison | While the blade battery ranked highest overall,... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 21 | bishop | 21 | Definition / Concept | What is the field of pattern recognition... | #1 (P) | pre:1 / post:1 | #2 (P) | #3 (P) | pre:3 / post:3 | #2 (P) | pre:2 / post:1 |
| 22 | bishop | 22 | Definition / Concept | In a supervised machine learning approach, what... | #2 (P) | pre:2 / post:2 | MISS (P) | #1 (P) | pre:1 / post:1 | #2 (P) | pre:2 / post:1 |
| 23 | bishop | 22 | Definition / Concept | What does the term generalization refer to... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 24 | bishop | 22 | Single fact | What is another term used for the... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 25 | bishop | 23 | Multi-part / Comparison | What is the primary difference between supervised... | #1 (P) | pre:1 / post:1 | #1 (P) | #2 (P) | pre:2 / post:2 | #1 (P) | pre:1 / post:1 |
| 26 | bishop | 23 | Definition / Concept | What defines a classification problem | #6 (P) | pre:- / post:- | #3 (P) | #1 (P) | pre:1 / post:1 | #2 (P) | pre:2 / post:2 |
| 27 | bishop | 23 | Definition / Concept | What is the credit assignment problem in... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 28 | bishop | 27 | Single fact | What does the root-mean-square (RMS) error allow... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 29 | bishop | 26,27 | Definition / Concept | What is the phenomenon of over-fitting in... | #4 (P) | pre:4 / post:- (LOST) | #3 (P) | #3 (P) | pre:3 / post:3 | #3 (P) | pre:3 / post:2 |
| 30 | bishop | 30 | Mechanism / Why | How does regularization help control the over-fitting... | #3 (P) | pre:3 / post:3 | #3 (P) | #1 (P) | pre:1 / post:1 | #3 (P) | pre:3 / post:3 |
| 31 | bishop | 33,34 | Multi-part / Comparison | What are the two fundamental rules of... | #1 (P) | pre:1 / post:1 | #2 (P) | #1 (P) | pre:1 / post:1 | #2 (P) | pre:2 / post:1 |
| 32 | bishop | 42 | Multi-part / Comparison | How is Bayes' theorem stated in words... | #1 (P) | pre:1 / post:1 | #1 (P) | #2 (P) | pre:2 / post:2 | #1 (P) | pre:1 / post:1 |
| 33 | bishop | 43 | Mechanism / Why | From a frequentist viewpoint, how is the... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 34 | bishop | 44 | Multi-part / Comparison | What are the two distinct parameters that... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 35 | bishop | 49 | Single fact | When assuming a Gaussian noise distribution on... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 36 | bishop | 52,53 | Definition / Concept | What is S-fold cross-validation used for | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 37 | bishop | 55,56 | Definition / Concept | What is the "curse of dimensionality" | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 38 | bishop | 61 | Single fact | In decision theory, what is the specific... | #3 (P) | pre:3 / post:3 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 39 | bishop | 63 | Multi-part / Comparison | What is the structural difference between generative... | #5 (P) | pre:- / post:- | #3 (P) | #4 (P) | pre:4 / post:3 | #3 (P) | pre:3 / post:1 |
| 40 | bishop | 70 | Single fact | According to Shannon's noiseless coding theorem, what... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 41 | meat | 5,8 | Single fact | Under which European Union regulation is cultured... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 42 | meat | 1 | Multi-part / Comparison | What four steps of risk assessment according... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 43 | meat | 8,22 | Multi-part / Comparison | Which country was the first to formally... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 44 | meat | 2,10 | Multi-part / Comparison | What are the three general steps in... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 45 | meat | 4,21 | Single fact | According to the Dutch Code of Practice... | #1 (P) | pre:1 / post:1 | #1 (P) | #3 (P) | pre:3 / post:1 | #1 (P) | pre:1 / post:1 |
| 46 | meat | 9 | Single fact | What professional disciplines must comprise the expert... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 47 | meat | 9 | Multi-part / Comparison | What are the procedural participant limits and... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 48 | meat | 12 | Mechanism / Why | Why could the common enteric bacteria (e.g.,... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 49 | meat | 16 | Mechanism / Why | Why is visual detection of Mycoplasma contamination... | #2 (P) | pre:2 / post:2 | #2 (P) | #1 (P) | pre:1 / post:1 | #2 (P) | pre:2 / post:1 |
| 50 | meat | 17 | Mechanism / Why | Why is cultured meat generally paler in... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 51 | meat | 11 | Multi-part / Comparison | What are microcarriers, and what role do... | #5 (P) | pre:- / post:- | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 52 | meat | 3,13,14 | Mechanism / Why | What primary risk is associated with the... | #5 (P) | pre:- / post:- | MISS (P) | #2 (P) | pre:2 / post:2 | MISS (P) | pre:6 / post:6 |
| 53 | meat | 18 | Mechanism / Why | Why are residues of bovine growth factors... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 54 | meat | 3,19 | Multi-part / Comparison | What chemical hazards can arise from the... | #1 (P) | pre:1 / post:1 | #1 (P) | #2 (P) | pre:2 / post:2 | #1 (P) | pre:1 / post:1 |
| 55 | meat | 15 | Single fact | Can consuming cells that have undergone uncontrolled... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 56 | meat | 20 | Single fact | What are examples of chemical process contaminants... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 57 | meat | 23 | Multi-part / Comparison | How is regulatory oversight of cell-cultured livestock,... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 58 | meat | 22 | Mechanism / Why | How does the Singapore Food Agency (SFA)... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 59 | meat | 3,4,20,21 | Mechanism / Why | Why was BuRO unable to conduct a... | #2 (P) | pre:2 / post:1 | #2 (P) | #1 (P) | pre:1 / post:1 | #2 (P) | pre:2 / post:1 |
| 60 | meat | 6 | Multi-part / Comparison | What main advice did BuRO give to... | #1 (P) | pre:1 / post:1 | #2 (P) | #1 (P) | pre:1 / post:1 | #2 (P) | pre:2 / post:1 |
| 61 | alexnet | 1 | Numbers / Hyperparameters | What error rates did the network achieve... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 62 | alexnet | 1 | Numbers / Hyperparameters | How many parameters and neurons does the... | #2 (P) | pre:2 / post:2 | MISS (-) | MISS (-) | pre:- / post:- | #4 (P) | pre:4 / post:1 |
| 63 | alexnet | 1 | Numbers / Hyperparameters | What was the winning top-5 test error... | #2 (P) | pre:2 / post:2 | #3 (P) | #1 (P) | pre:1 / post:1 | #3 (P) | pre:3 / post:1 |
| 64 | alexnet | 2 | Mechanism / Why | How were the variable-resolution images in the... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 65 | alexnet | 2,7 | Numbers / Hyperparameters | What hardware was used to train the... | #3 (P) | pre:3 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 66 | alexnet | 3 | Mechanism / Why | Why did the authors choose Rectified Linear... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 67 | alexnet | 3 | Multi-part / Comparison | How do the two GPUs communicate across... | #2 (P) | pre:2 / post:2 | #2 (P) | #1 (P) | pre:1 / post:1 | #2 (P) | pre:2 / post:2 |
| 68 | alexnet | 4 | Numbers / Hyperparameters | What are the specific hyperparameter values chosen... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 69 | alexnet | 4 | Numbers / Hyperparameters | How is overlapping pooling defined, and what... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 70 | alexnet | 4 | Numbers / Hyperparameters | What are the kernel size, number of... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 71 | alexnet | 4 | Single fact | Which convolutional layers are followed by max-pooling... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 72 | alexnet | 4,5 | Numbers / Hyperparameters | How many neurons are there in each... | #2 (P) | pre:2 / post:1 | #1 (P) | #2 (P) | pre:2 / post:1 | #1 (P) | pre:1 / post:1 |
| 73 | alexnet | 5 | Multi-part / Comparison | What is the first form of data... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 74 | alexnet | 5,6 | Mechanism / Why | How does the second form of data... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 75 | alexnet | 6 | Multi-part / Comparison | In which layers is dropout applied, and... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 76 | alexnet | 6 | Numbers / Hyperparameters | What values were used for batch size,... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 77 | alexnet | 6 | Multi-part / Comparison | How were the weights and neuron biases... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 78 | alexnet | 6,7 | Single fact | What was the learning rate schedule used... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 79 | alexnet | 7 | Multi-part / Comparison | What specialization was observed between the kernels... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
| 80 | alexnet | 2,8 | Mechanism / Why | What effect did removing a single convolutional... | #1 (P) | pre:1 / post:1 | #1 (P) | #1 (P) | pre:1 / post:1 | #1 (P) | pre:1 / post:1 |
