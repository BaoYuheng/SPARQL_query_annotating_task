# SPARQL_query_annotating_task

This repository contains the code implementation for the ESWC 2026 paper **"SQA: SPARQL Query Annotating with Question-Answer Pairs"**.

> **Note**: Due to the complexity and long development timeline of this project, the code is still under active reorganization. If you encounter issues, please open a GitHub issue.

## Table of Contents

- [Overview](#overview)
- [Project Organization](#project-organization)
- [Environment Setup](#environment-setup)
- [How to Run](#how-to-run)
- [Code Architecture](#code-architecture)
- [Technical Features & Correction](#technical-features--correction)
- [Known Issues](#known-issues)
- [Citation](#citation)

---

## Overview

### Task Definition

**SPARQL Query Annotating (SQA)**: Given a natural language question $q$, its answer $a$, and a knowledge base $\mathcal{K}$, automatically construct a SPARQL query $s$ such that executing $s$ over $\mathcal{K}$ yields results that reflect $a$.

Based on this task, this repository further implements evaluation metrics (TSS, TED), proposes the QuAD method, and includes all experiments from the paper.

### Contributions

| # | Contribution | Description |
|---|-------------|-------------|
| 1 | **SQA Task** | A new task: automatically constructing SPARQL queries from question-answer pairs |
| 2 | **TSS Metric** | Test Suite Score — measures semantic equivalence via execution consistency |
| 3 | **ReQUMA Dataset** | 200 real-world questions for evaluating SQA's utility in manual annotation |
| 4 | **QuAD Method** | Question-Answer Driven approach using answers to guide search space pruning |

---

## Project Organization

```
SPARQL_query_annotating_task/
│
├── data/                              # Data directory (released as assets)
│   ├── FB_ontology/                   # Freebase KG ontology data
│   │   ├── fb_roles                   # Freebase relation list
│   │   ├── fb_types                   # Freebase type information
│   │   ├── reverse_properties         # Reverse relation dictionary
│   │   ├── full_reverse_properties.json
│   │   ├── domain_dict / domain_info  # Relation domain information
│   │   └── README.md
│   │
│   ├── WD_ontology/                   # Wikidata KG ontology data
│   │   ├── all_wikidata_property.csv  # All Wikidata properties
│   │   ├── wikidata_classes_all.json  # Wikidata classes
│   │   ├── wikidata_numeral_property_dict.json  # Numerical property dictionary
│   │   └── wikidata_property_id2label_dict.json # Property ID→label mapping
│   │
│   ├── original_dataset/              # Original KBQA datasets (golden SPARQL)
│   │   ├── ComplexWebQuestions_test.json
│   │   ├── grailqa_v1.0_dev.json
│   │   ├── webqsp_test.json
│   │   └── requama_test.json          # ReQUMA dataset
│   │
│   ├── input/                         # Preprocessed input data
│   │   ├── ReQUMA/dataset.json        # ReQUMA dataset (with question decomposition)
│   │   ├── cwq/                       # CWQ: entity linking results, test suites, etc.
│   │   ├── GrailQA_v1.0/              # GrailQA: entity linking results, test suites, etc.
│   │   └── WebQSP/                    # WebQSP: entity linking results, test suites, etc.
│   │
│   ├── output/                        # Experiment outputs
│   │   ├── quad_search_results/       # QuAD search & rerank outputs
│   │   │   ├── cwq/                   # CWQ search results
│   │   │   ├── grailqa/               # GrailQA search results
│   │   │   ├── webqsp/                # WebQSP search results
│   │   │   └── requama/               # ReQUMA search results
│   │   ├── annotation_performance_experiment/  # Exp 5.3: SQA method evaluation
│   │   ├── kbqa_performance_experiment/        # Exp 5.2: KBQA performance evaluation
│   │   └── manual_annotation_experiment/       # Exp 5.1: Manual annotation experiment
│   │
│   └── test/                          # Test data
│
├── src/                               # Source code (modular organization)
│   ├── core/                          # Core: constants, enums, utility functions
│   │   ├── common.py                  # DATASET/Dataset/Method enums, Freebase/Wikidata constants
│   │   └── utils.py                   # JSON I/O, logging, PRF1 computation
│   │
│   ├── sparql/                        # SPARQL: execution, parsing, preprocessing
│   │   ├── executor.py                # SPARQL query executor (ODBC + SPARQLWrapper dual backend)
│   │   ├── sparql_utils.py            # SPARQL parsing, syntax tree editing, TED computation
│   │   ├── sparql_utils_new.py        # Extended: Wikidata inverse relation support
│   │   ├── process.py                 # SPARQL preprocessing (normalization, token replacement)
│   │   └── awudima/                   # Third-party PLY SPARQL grammar parser
│   │
│   ├── logical_form/                  # Logical forms: S-expressions, graphs, transformations
│   │   ├── s_expression_utils.py      # S-expression ↔ SPARQL conversion (basic)
│   │   ├── s_expression_utils_new.py  # Extended: Test Suite Score functionality
│   │   ├── logic_form_util.py         # Lisp/S-expression → SPARQL top-level conversion
│   │   ├── graph_utils.py             # Graph processing & edit distance computation
│   │   ├── simple_graph.py            # Simple graph definitions (Node, Edge, SimpleGraph)
│   │   └── sparql_to_sexp_utils.py    # SPARQL → S-expression reverse parsing
│   │
│   ├── linking/                       # Entity linking & semantic similarity
│   │   ├── entity_linker.py           # Entity linker (FACC1, Freebase Class, SUTime)
│   │   ├── facc1_index.py             # FACC1 entity surface form index (in-memory)
│   │   ├── semantic_sim.py            # PLM embedding-based semantic similarity
│   │   ├── sentence_bert.py           # SentenceBERT similarity predictor
│   │   └── faiss_indexer.py           # FAISS vector index construction
│   │
│   ├── quad/                          # QuAD method core: query search & decomposition
│   │   ├── search.py                  # Beam Search query construction algorithm (core module)
│   │   ├── decomposition.py           # Complex question decomposition into sub-questions
│   │   └── rerank.py                  # GPT/LLM candidate query reranking
│   │
│   ├── llm/                           # LLM invocation
│   │   └── agent.py                   # OpenAI API request wrapper, query intent classification
│   │
│   ├── concurrent/                    # Concurrent execution
│   │   └── executor.py                # General-purpose concurrent execution framework
│   │
│   └── metrics/                       # Evaluation metrics (pure algorithms, no I/O)
│       ├── minimum_TSS.py             # TSS implementation: Method 1 (all-at-once perturbation) etc.
│       └── minimum_TED.py             # Tree edit distance (ZSS algorithm)
│
├── scripts/                           # Experiment & evaluation scripts
│   ├── experiment_runner.py           # Unified experiment entry: search, rerank, TSS, TED, pipeline
│   ├── annotation_performance_evaluation.py  # Exp 5.3: SQA method evaluation
│   ├── manual_annotation_evaluation.py       # Exp 5.1: Manual annotation evaluation
│   ├── kbqa_performance_evaluation.py        # Exp 5.2: KBQA performance evaluation
│   └── recompute_annotation_metrics.py       # Recompute annotation metrics
│
├── tests/                             # Tests
│   └── test_executor_backends.py      # Executor dual-backend tests
│
├── paper.tex                          # ESWC 2026 paper source
├── requirements.txt                   # Python dependencies
├── README.md                          # English README (this file)
├── README_CN.md                       # Chinese README
└── .gitignore
```

---

## Environment Setup

### Conda Environment

We recommend using the `sqa_verify` conda environment:

```bash
conda create -n sqa_verify python=3.12
conda activate sqa_verify
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:

| Package | Purpose |
|---------|---------|
| `torch`, `transformers`, `sentence-transformers` | Semantic similarity (BGE-reranker) |
| `openai` | LLM calls (GPT-3.5-Turbo for decomposition and reranking) |
| `ply` | SPARQL grammar parsing (awudima parser) |
| `networkx` | Graph structure processing |
| `pyodbc` | Freebase/Wikidata Virtuoso ODBC connection (recommended, superior performance for large results) |
| `SPARQLWrapper` | HTTP SPARQL endpoint connection (fallback, no Virtuoso driver needed) |
| `anytree` | Syntax tree structures |
| `zss` | Tree edit distance (Zhang-Shasha) |
| `scikit-learn`, `scipy`, `numpy` | General scientific computing |

> **Note**: NumPy < 2.0 is required (scipy/sklearn compatibility). The `sqa_verify` conda environment is pre-configured.

### External Services

| Service | Description |
|---------|-------------|
| **Freebase Virtuoso** | ODBC or HTTP SPARQL endpoint for CWQ / GrailQA / WebQSP experiments |
| **Wikidata 2023 Endpoint** | HTTP SPARQL endpoint for ReQUMA experiments |
| **OpenAI API** | Used for question decomposition and query reranking in QuAD |

### Entity Linking

- **Freebase**: FACC1 surface form matching
- **Wikidata**: CLOCQ entity linking tool

> **Note**: The entity linking pipeline is lengthy, and question decomposition is simple prompt-based in-context learning. Therefore, this repository does not provide executable pipelines for these steps. Decomposition and linking results are already included in each dataset directory under `data/input/`. The corresponding source code (`src/linking/`, `src/quad/decomposition.py`) is provided for reference only.

---

## How to Run

All experiments are run through the unified `scripts/experiment_runner.py` CLI.

```bash
python scripts/experiment_runner.py <dataset> <step> [options]
```

### Available Steps

| Step | Description |
|------|-------------|
| `search` | Beam Search SPARQL query construction |
| `rerank` | GPT-based candidate query reranking |
| `tss` | Method 1 (Paper): all-at-once KB perturbation, deterministic |
| `tss3` | Method 3 (Precomputed): pre-generated test suite + sampled F1 |
| `pipeline` | End-to-end: search → rerank |

**Options**:

| Option | Description | Default |
|--------|-------------|---------|
| `--size N` | Process N questions | 100 |
| `--model M` | GPT model name | gpt-4o-mini |
| `--suffix S` | Output subdirectory suffix | none |
| `--rerank-file` | Custom rerank input file | auto-derived |
| `--test-suite-file` | Method 3 test suite path | auto-derived/generated |
| `--test-cases-limit` | Method 3 samples per question | 5 |

### Examples

```bash
# Search
python scripts/experiment_runner.py CWQ search --size 100
python scripts/experiment_runner.py ReQUMA search --size 100

# Rerank
python scripts/experiment_runner.py CWQ rerank --model gpt-4o-mini --size 100

# TSS Method 1 (Paper) — best for Freebase
python scripts/experiment_runner.py CWQ tss --size 100
python scripts/experiment_runner.py WebQSP tss --size 100

# TSS Method 3 (Precomputed) — best for Wikidata
# Test suite is auto-generated if absent
python scripts/experiment_runner.py ReQUMA tss3 --size 100

# End-to-end pipeline
python scripts/experiment_runner.py CWQ pipeline --size 100 --model gpt-4o-mini

# Batch run across all datasets
python scripts/experiment_runner.py all search --size 100
```

### TSS Method Selection

| Method | Principle | Best For |
|--------|-----------|----------|
| Method 1 (Paper) | All-at-once KB perturbation, single query, deterministic | Freebase (CWQ/GrailQA/WebQSP) |
| Method 3 (Precomputed) | Pre-generated test suite + sampling + F1 | Wikidata (ReQUMA) |

> In essence, Method 3 is a random-sampling approximation of the theoretical version (Method 1): it uses one query's perturbation space to represent the full perturbation spaces of both queries. Since the returned data volume is small, efficiency is manageable. Although F1 is introduced, it is conceptually consistent with Jaccard similarity and does not affect overall trends.

> Wikidata is much larger in scale; Method 1 takes ~10s per query and places heavy load on the endpoint. We recommend using Method 3 for Wikidata.

### Reproducing Paper Experiments

#### Experiment 5.1: Manual Annotation Evaluation (ReQUMA)

```bash
python scripts/manual_annotation_evaluation.py
```

Input: `data/original_dataset/requama_test.json`. Output: `data/output/manual_annotation_experiment/`.

#### Experiment 5.2: KBQA Performance Evaluation (Freebase)

```bash
python scripts/kbqa_performance_evaluation.py
```

#### Experiment 5.3: SQA Method Performance Evaluation

```bash
# Step 1: QuAD search + rerank
python scripts/experiment_runner.py all pipeline --size 100

# Step 2: Evaluate TSS + TED
python scripts/annotation_performance_evaluation.py
```

---

## Code Architecture

Three-layer architecture:

```
Layer 3 (Directory-level): scripts/annotation_performance_evaluation.py
                            scripts/manual_annotation_evaluation.py
                            └─ Multi-method traversal, table/chart generation, batch processing

Layer 2 (File-level):      scripts/experiment_runner.py
                            └─ Querier construction, data I/O, batch entry points, pipeline orchestration

Layer 1 (Algorithm-level): src/metrics/minimum_TSS.py / minimum_TED.py
                            └─ Pure algorithms, no file I/O, no querier construction
```

**Module dependencies**:

```
scripts/         → core/ + sparql/ + logical_form/ + linking/ + quad/ + metrics/
quad/            → core/ + llm/ + logical_form/ + linking/
metrics/         → core/ + sparql/
linking/         → core/
sparql/          → logical_form/
logical_form/    → core/
```

---

## Technical Features & Correction

### 1. TED Implementation Correction (Important Update)

After publication, we reviewed the TED implementation and fixed several bugs.

**Typical issue** (Freebase): QuAD generates `FILTER NOT EXISTS {?cvt ns:name ?name}` to constrain CVT nodes via the absence of a name. This is semantically harmless — Freebase's property schema guarantees CVT nodes do not have names. However, this **unfairly penalizes QuAD** with inflated TED scores.

**Corrected Table 4 TED results**:

| Methods | CWQ TED | GrailQA TED | WebQSP TED |
|---------|:-------:|:-----------:|:----------:|
| QGG⁺ | 0.431 | -- | 0.191 |
| QGG-A | 0.568 | -- | 0.187 |
| KB-BINDER | -- | 0.407 | 0.466 |
| LSQ-A | 0.732 | 0.438 | 0.422 |
| QueryAgent | 0.644 | 0.275 | 0.252 |
| QuAD | 0.451 | 0.375 | 0.220 |
| QuAD$_{+g}$ | **0.355** | **0.188** | **0.135** |

> ⚠️ The corrected results are more favorable to QuAD. This sensitivity to specific edge cases further illustrates the inherent instability of TED implementations.

---

### 2. TSS Implementation Details & Corrections

#### 2.1 Freebase TSS (Method 1, Theoretical Version)

- **Principle**: Strictly follows the paper's theoretical description — takes **all** possible perturbations from the KG
- **Used in**: Experiment 5.3 (method evaluation on Freebase, Table 4)
- **Key parameter**: `MaxReturnRows`

**Correction notes**:

The original paper's TSS results were computed **without** setting MaxReturnRows. After reconfiguring Virtuoso, MaxReturnRows was set to 50K.

In practice, `MaxReturnRows` is an important parameter:
- **Too small** → low overlap probability for random perturbation sets, high variance
- **Too large** → long execution time, risk of timeouts
- **Recommendation**: maximize it while keeping efficiency manageable

After setting `MaxReturnRows` to **200K** and re-running experiments, the corrected results are:

**TSS Comparison (Old vs New)**:

| Method | CWQ Old | CWQ New | GrailQA Old | GrailQA New | WebQSP Old | WebQSP New |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| QGG⁺ | 0.132 | 0.142 | -- | -- | 0.602 | 0.618 |
| QGG-A | 0.076 | 0.039 | -- | -- | 0.665 | 0.686 |
| KB-BINDER | -- | -- | 0.429 | 0.429 | 0.301 | 0.312 |
| LSQ-A | 0.036 | 0.004 | 0.160 | 0.162 | 0.202 | 0.224 |
| QueryAgent | 0.106 | 0.060 | 0.527 | 0.530 | 0.551 | 0.566 |
| QuAD | 0.236 | 0.270 | 0.548 | 0.550 | 0.554 | 0.630 |
| QuAD$_{+g}$ | 0.368 | 0.423 | 0.710 | 0.715 | 0.669 | 0.747 |

> QuAD's advantage is more pronounced in the corrected results.

**Average TSS computation time (MaxReturnRows=200K, seconds/query)**:

| Method | CWQ | GrailQA | WebQSP |
| :--- | :---: | :---: | :---: |
| QGG⁺ | 8.62 | -- | 16.65 |
| QGG-A | 5.31 | -- | 15.67 |
| KB-BINDER | -- | 5.10 | 14.80 |
| LSQ-A | 0.19 | 1.09 | 3.34 |
| QueryAgent | 6.02 | 2.02 | 15.62 |
| QuAD | 9.32 | 2.30 | 15.80 |
| QuAD$_{+g}$ | 13.80 | 3.98 | 18.56 |

> Performance can be significantly improved through query result caching and other optimizations.

This repository defaults to the **ODBC** backend (configurable in `src/core/common.py`). ODBC significantly outperforms SPARQLWrapper HTTP for large result sets. The measured ODBC TSS Method 1 timings are in the same order of magnitude as the table above:

| Dataset | Setting | Items | Per Item |
|---------|---------|------:|:---------:|
| CWQ | quad golden | 869 | 3.4s |
| CWQ | quad linked | 841 | 2.2s |
| GrailQA | quad golden | 867 | 1.3s |
| GrailQA | quad linked | 692 | 0.7s |
| WebQSP | quad golden | 966 | 8.0s |
| WebQSP | quad linked | 922 | 3.9s |

> If ODBC is unavailable, switch to SPARQLWrapper by setting `service="sparql_wrapper"` in `build_sparql_querier()`.

#### 2.2 Wikidata TSS (Method 3, Practical Version)

For Wikidata-scale KGs, all-at-once perturbation (Method 1) is infeasible. Method 3 uses a precomputed test suite with random sampling:

1. Pre-generate test suite examples on the golden query (one-time, reusable)
2. Apply the test suite to the simulated query and compute F1 scores
3. Average over multiple sampling rounds as the TSS

> This is a random-sampling approximation of the theoretical version. Since query result sizes are small, efficiency is manageable.

**Wikidata Method 1 compatibility**:
- Code supports Wikidata Method 1, but note the performance implications of large-scale data
- Fixed recognition of abbreviated `xsd:` prefix forms (`src/core/common.py`)
- Fixed golden SPARQL missing PREFIX declarations causing syntax tree node mismatches (`src/metrics/minimum_TSS.py`)

---

### 3. Wikidata Data Type Support

This repository has fixed handling of Wikidata typed literals:
- Unified support for full URI forms (`^^<http://www.w3.org/2001/XMLSchema#dateTime>`) and abbreviated forms (`^^xsd:dateTime`)
- Supported types: `decimal`, `integer`, `float`, `date`, `gYear`, `gYearMonth`, `dateTime`
- Fix location: `WikidataConstantForConstruction.get_constant_type()` in `src/core/common.py`

---

### 4. AI Usage Statement

Code reorganization in this repository was assisted by Claude Code (with DeepSeekV4-pro as the LLM backbone).

---

## Known Issues

### Code Structure

| # | Location | Issue |
|---|----------|-------|
| 1 | `src/sparql/sparql_utils_Old.py` | Legacy file with hardcoded absolute paths; retained for reference only |
| 2 | `src/linking/semantic_sim.py` | Embedding file (~1GB) path points to external storage; annotated with comments |

### DATASET Enum Mixing

The legacy `DATASET` enum in `src/core/common.py` conflates two semantics:
- **Dataset** (GRAIL, CWQ, WEBQ, LC2, QALD) — determines KB type and SPARQL dialect
- **Method source** (QGG, QUERYAGENT, BINDER, LSQ, etc.) — determines SPARQL preprocessing mode

The codebase has introduced separate `Dataset` and `Method` enums; some entry-point functions have migrated to the new types. However, low-level modules such as `SyntaxTreeEditor` still use the old `DATASET` enum for direct comparisons.

**Current status: does not affect usability.** All call chains pass correct values; only the type system is not enforced. A full refactoring would require bottom-up changes starting from `SyntaxTreeEditor`.

---

## Citation

```bibtex
@inproceedings{DBLP:conf/esws/BaoZWHXQQ26,
  author       = {Yuheng Bao and
                  Wenhao Zhou and
                  Xuan Wu and
                  Wei Hu and
                  Dingkun Xu and
                  Mingjia Qian and
                  Yuzhong Qu},
  editor       = {Maribel Acosta and
                  Marieke van Erp and
                  Sebastian Rudolph and
                  Olaf Hartig and
                  Blerina Spahiu and
                  Anisa Rula and
                  Daniel Garijo and
                  Francesco Osborne},
  title        = {{SQA:} {SPARQL} Query Annotating with Question-Answer Pairs},
  booktitle    = {The Semantic Web - 23rd European Semantic Web Conference, {ESWC} 2026,
                  Dubrovnik, Croatia, May 10-14, 2026, Proceedings, Part {I}},
  series       = {Lecture Notes in Computer Science},
  pages        = {42--62},
  publisher    = {Springer},
  year         = {2026},
  url          = {https://doi.org/10.1007/978-3-032-25156-5\_3},
  doi          = {10.1007/978-3-032-25156-5\_3},
  timestamp    = {Wed, 03 Jun 2026 10:10:13 +0200},
  biburl       = {https://dblp.org/rec/conf/esws/BaoZWHXQQ26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
