# src 文件结构总述

> 生成日期: 2026-05-20
> 这是对原始扁平化 `src/` 目录进行模块化重组后的代码结构。
> 原始 `src/` 为扁平单层目录，`src/` 按功能拆分为多个子包。

## 目录树

```
src/
├── __init__.py                    # 顶级包入口
│
├── unused/                        # 未使用 / 已废弃的代码
│   └── fuzzing_similarity.py      # SPARQL 实体替换工具（死代码，无引用）
│
├── core/                          # 核心：常量定义 & 工具函数
│   ├── __init__.py                # 导出所有 public API
│   ├── common.py                  # 枚举、常量、知识库类型定义
│   └── utils.py                   # JSON读写、日志、PRF1计算、时间/数值处理
│
├── sparql/                        # SPARQL：执行、解析、预处理
│   ├── __init__.py
│   ├── executor.py                # SPARQL 查询执行器（ODBC/HTTP，FB/Wikidata）
│   ├── sparql_utils.py            # SPARQL 解析 & 语法树编辑（旧版）
│   ├── sparql_utils_new.py        # SPARQL 解析扩展版（支持 Wikidata 逆关系）
│   ├── process.py                 # SPARQL 查询预处理（调用 awudima parser）
│   └── awudima/                   # 第三方 SPARQL 语法解析器
│       ├── __init__.py            # 解析器类定义（SelectQuery, Query, AskQuery, BGP...）
│       ├── parser.py              # PLY 解析器实现（3756行）
│       ├── parser.out             # PLY 生成文件
│       └── parsetab.py            # PLY 解析表（419行）
│
├── logical_form/                  # 逻辑形式：S-expression、图结构、转换
│   ├── __init__.py
│   ├── s_expression_utils.py      # S-expression → SPARQL 转换（基础版，Freebase/Wikidata）
│   ├── s_expression_utils_new.py  # 扩展版：增加 Test Suite Score 评估功能
│   ├── logic_form_util.py         # Lisp/S-expression → SPARQL 顶层转换
│   ├── graph_utils.py             # 图结构处理 & 编辑距离计算（GraphEquivalenceUtil）
│   ├── simple_graph.py            # 简单图定义（Node, Edge, SimpleGraph）
│   └── sparql_to_sexp_utils.py    # SPARQL → S-expression 反向转换（SparqlParserLC2）
│
├── linking/                       # 实体链接 & 语义相似度
│   ├── __init__.py
│   ├── entity_linker.py           # 实体链接器（FACC1, FreebaseClass, SUTime）
│   ├── facc1_index.py             # FACC1 实体表面形式索引（内存版）
│   ├── semantic_sim.py            # 基于 PLM embedding 的语义相似度
│   ├── sentence_bert.py           # SentenceBERT 相似度预测器
│   └── faiss_indexer.py           # FAISS 向量索引构建
│
├── quad/                          # 查询搜索 & 问题分解
│   ├── __init__.py
│   ├── search.py                  # Best-First Search 搜索算法（核心模块，1629行）
│   ├── decomposition.py           # 复杂问题分解
│   └── rerank.py                  # GPT 对候选查询重排序
│
├── llm/                           # LLM 调用
│   ├── __init__.py
│   └── agent.py                   # OpenAI API 请求器、查询意图分类（94行）
│
├── concurrent/                    # 并发执行
│   ├── __init__.py
│   └── executor.py                # 通用并发执行器（多线程/多进程，244行）
│
└── scripts/                       # 实验 & 数据处理脚本
    ├── experiment.py              # 主实验流程（GrailQA, CWQ, WebQSP, Wikidata）
    ├── data_process.py            # 数据预处理 & 特征工程（2994行）
    └── analyze.py                 # (已移至 unused/) 结果分析工具，含 broken imports
```

## 模块职责

### 1. `core/` — 基础设施层
| 文件 | 原始对应 | 行数 | 职责 |
|------|----------|------|------|
| `common.py` | `src/common.py` | 325 | DATASET, KB_TYPE 枚举；Freebase/Wikidata 常量类；操作符映射；WIKIDATA_PREFIX_LIST |
| `utils.py` | `src/utils.py` | 370 | JSON 读写、日志初始化、PRF1 指标计算、KBNumberItem/KBTimeItem 比较、N-gram 生成 |

### 2. `sparql/` — SPARQL 查询层
| 文件 | 原始对应 | 行数 | 职责 |
|------|----------|------|------|
| `executor.py` | `src/sparql_executor.py` | 5323 | SPARQL 查询执行（ODBC Virtuoso + HTTP SPARQLWrapper）；支持 Freebase 和 Wikidata；含并发查询能力 |
| `sparql_utils.py` | `src/sparql_utils.py` | 1339 | SPARQL 解析、语法树构建（SyntaxTreeEditor）；关系字典加载 |
| `sparql_utils_new.py` | `src/sparql_utils_new.py` | 1547 | 扩展版：支持 Wikidata 逆关系；TreeConstructor, TreeEditDistance |
| `process.py` | `src/sparql_process.py` | 1186 | SPARQL 查询预处理：标准化、token 替换、awudima parser 调用 |
| `awudima/` | `src/awudima/` | ~5300 | 第三方 PLY 语法解析器，提供完整 SPARQL AST 定义 |

### 3. `logical_form/` — 逻辑形式层
| 文件 | 原始对应 | 行数 | 职责 |
|------|----------|------|------|
| `s_expression_utils.py` | `src/s_expression_utils.py` | 1483 | S-expression 构建函数（JOIN/AND/ARG/CMP/COUNT/R）；Freebase & Wikidata SPARQL 转换；编辑距离格式转换 |
| `s_expression_utils_new.py` | `src/s_expression_utils_new.py` | 2134 | 扩展版：新增 `sexp_to_sparql_for_test_suite()` 和 `sexp_to_sparql_wikidata_for_test_suite()` |
| `logic_form_util.py` | `src/logic_form_util.py` | 723 | Lisp/S-expression → SPARQL 顶层入口函数 `lisp_to_sparql()` |
| `graph_utils.py` | `src/graph_utils.py` | 1430 | 图结构处理、同构检测、图编辑距离（GraphEquivalenceUtil） |
| `simple_graph.py` | `src/simple_graph.py` | 649 | SimpleGraph/Node/Edge 数据结构定义 |
| `sparql_to_sexp_utils.py` | `src/sparql_to_sexp_utils.py` | 722 | SPARQL 查询反向解析为 S-expression（SparqlParserLC2） |

### 4. `linking/` — 实体链接与语义层
| 文件 | 原始对应 | 行数 | 职责 |
|------|----------|------|------|
| `entity_linker.py` | `src/entity_utils.py` | 293 | 实体链接（FACC1 表面形式匹配）；Freebase Class 链接；SUTime 时间识别 |
| `facc1_index.py` | `src/facc1_surface_index_memory.py` | 334 | FACC1 实体内存索引构建与查询 |
| `semantic_sim.py` | `src/semantic_sim_utils.py` | 135 | 基于 PLM embedding 的关系/问题语义相似度计算（FBEmbeddingBasedSimilarity, PLMSimRanker） |
| `sentence_bert.py` | `src/sentence_bert_utils.py` | 137 | SentenceBERT 模型加载与语义相似度预测 |
| `faiss_indexer.py` | `src/faiss_indexers.py` | 139 | FAISS 向量索引构建（Flat/IVF） |

### 5. `quad/` — 查询搜索与分解
| 文件 | 原始对应 | 行数 | 职责 |
|------|----------|------|------|
| `search.py` | `src/quad/search.py` | 1629 | Best-First Search 查询搜索算法；原子查询生成、组合、剪枝；多跳关系支持 |
| `decomposition.py` | `src/quad/question_decomposition.py` | 167 | 复杂问题分解为子问题 |
| `rerank.py` | `src/quad/gpt_rerank.py` | 167 | 使用 GPT/LLM 对候选查询进行重排序 |

**已知问题**:
- `search.py` 有 hardcoded path: `sys.path.append("/home5/yhbao/SPARQLannotation")`

### 6. `metrics/` — 评估指标
| 文件 | 原始对应 | 行数 | 职责 |
|------|----------|------|------|
| `minimum_TSS.py` | 新建 | 319 | TSS Method 1 (ALL-at-once)，轻量入口 |
| `minimum_tss.py` | `src/tss/metrics_calculation.py` | 1527 | TSS 完整实现：Method 1/2/3 + test suite 生成 + runner |
| `minimum_TED.py` | 新建 | ~100 | 树编辑距离计算入口 |

### 7. `llm/` — LLM 集成
| 文件 | 原始对应 | 行数 | 职责 |
|------|----------|------|------|
| `agent.py` | `src/llm_agent.py` | 94 | OpenAI API 请求封装；QueryIntent 意图分类 |

### 8. `concurrent/` — 并发执行
| 文件 | 原始对应 | 行数 | 职责 |
|------|----------|------|------|
| `executor.py` | `src/concurrent_executor.py` | 244 | 通用并发执行框架；支持多线程/多进程 |

### 9. `scripts/` — 实验脚本
| 文件 | 原始对应 | 行数 | 职责 |
|------|----------|------|------|
| `experiment.py` | `scripts/experiment.py` | 790 | 主实验入口：多数据集搜索、Wikidata 标注、GPT 重排序 |
| `data_process.py` | `scripts/data_process.py` | 2994 | 数据预处理流水线：分词、实体链接、特征提取 |
| `analyze.py` | (已移至 unused/) | 159 | 已废弃：含 broken imports (`from src.utils`)，移至 unused/ |

## 文件命名映射: src/ → src/

| 原始文件 (src/) | 新位置 (src/) | 说明 |
|---|---|---|
| `common.py` | `core/common.py` | 归入 core 模块 |
| `utils.py` | `core/utils.py` | 归入 core 模块 |
| `sparql_executor.py` | `sparql/executor.py` | 归入 sparql 模块 |
| `sparql_utils.py` | `sparql/sparql_utils.py` | 保留原名 |
| `sparql_utils_new.py` | `sparql/sparql_utils_new.py` | 保留原名 |
| `sparql_process.py` | `sparql/process.py` | 重命名 |
| `s_expression_utils.py` | `logical_form/s_expression_utils.py` | 归入 logical_form 模块 |
| `s_expression_utils_new.py` | `logical_form/s_expression_utils_new.py` | 归入 logical_form 模块 |
| `logic_form_util.py` | `logical_form/logic_form_util.py` | 保留原名 |
| `graph_utils.py` | `logical_form/graph_utils.py` | 归入 logical_form 模块 |
| `simple_graph.py` | `logical_form/simple_graph.py` | 归入 logical_form 模块 |
| `sparql_to_sexp_utils.py` | `logical_form/sparql_to_sexp_utils.py` | 归入 logical_form 模块 |
| `entity_utils.py` | `linking/entity_linker.py` | 重命名 |
| `facc1_surface_index_memory.py` | `linking/facc1_index.py` | 重命名 |
| `semantic_sim_utils.py` | `linking/semantic_sim.py` | 重命名 |
| `sentence_bert_utils.py` | `linking/sentence_bert.py` | 重命名 |
| `faiss_indexers.py` | `linking/faiss_indexer.py` | 重命名 |
| `fuzzing_similarity.py` | `linking/fuzzing_similarity.py` | 保留原名 |
| `detection_utils.py` | **未迁移** | `quad/__init__.py` 仍引用 `.detection` |
| `quad/search.py` | `quad/search.py` | 保留原名 |
| `quad/question_decomposition.py` | `quad/decomposition.py` | 重命名 |
| `quad/gpt_rerank.py` | `quad/rerank.py` | 重命名 |
| `tss/metrics_calculation.py` | `metrics/minimum_tss.py` | 重命名并迁移至 metrics/ |
| `llm_agent.py` | `llm/agent.py` | 归入 llm 模块，重命名 |
| `concurrent_executor.py` | `concurrent/executor.py` | 归入 concurrent 模块 |
| `awudima/` | `sparql/awudima/` | 归入 sparql 模块 |

## 已知问题清单

1. **`src/sparql/sparql_utils_Old.py`**: 旧版文件，含有 hardcoded 绝对路径（`/home5/yhbao/data/...`），保留仅供参考
2. **`src/linking/semantic_sim.py`**: embedding 文件路径指向外部存储（`/home5/yhbao/SPARQLannotation/...`），因文件体积 ~1GB 未纳入仓库，已添加双语注释说明

## 统计

| 指标 | 数值 |
|------|------|
| 总文件数 | 40 (.py) + 2 (.out/.tab) |
| Python 总行数 | ~34,262 |
| 子包数 | 8 (core, sparql, logical_form, linking, quad, metrics, llm, concurrent) + unused/ |
| 最大模块 | `sparql/executor.py` (5323行) |
| 最大子包 | `sparql/` (~11,400行, 含 awudima) |

## 依赖关系图（包级别）

```
scripts/
  ├── core/
  ├── logical_form/
  ├── linking/
  ├── quad/
  └── sparql/

quad/
  ├── core/
  ├── llm/
  ├── logical_form/
  └── linking/

linking/
  └── core/

sparql/
  └── logical_form/

logical_form/
  └── core/
```

> **注意**: 以上依赖关系基于 import 语句分析。
