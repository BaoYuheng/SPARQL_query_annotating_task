# SPARQL_query_annotating_task

本仓库是 ESWC 2026 论文 **"SQA: SPARQL Query Annotating with Question-Answer Pairs"** 的代码实现。


由于该项目代码比较复杂而且时间跨度较长，目前仍在进一步整理过程中。如果发现问题，欢迎留言。


## 目录

- [项目概述](#项目概述)
- [项目组织](#项目组织)
- [环境配置](#环境配置)
- [How to Run](#how-to-run)
- [代码架构](#代码架构)
- [Technical Features & Correction](#technical-features--correction)
- [已知问题](#已知问题)
- [引用](#引用)

---

## 项目概述

### 任务定义

**SPARQL Query Annotating (SQA)** 任务：给定一个自然语言问题 $q$、其答案 $a$、以及知识库 $\mathcal{K}$，自动构造一条 SPARQL 查询 $s$，使得 $s$ 在 $\mathcal{K}$ 上执行的结果能够反映 $a$。

基于此任务，本仓库进一步实现了评估指标（TSS、TED）、提出了 QuAD 方法，并完成了论文中的全部实验。

### 论文贡献

| # | 贡献 | 说明 |
|---|------|------|
| 1 | **SQA 任务** | 提出从 question-answer 对自动构造 SPARQL 查询的新任务 |
| 2 | **TSS 指标** | 提出 Test Suite Score，通过执行一致性衡量查询的语义等价性 |
| 3 | **ReQuMA 数据集** | 收集 200 条真实问题，用于评估 SQA 任务对人工标注的辅助效果 |
| 4 | **QuAD 方法** | 提出 Question-Answer Driven 方法，利用答案引导搜索空间剪枝 |

---

## 项目组织

```
SPARQL_query_annotating_task/
│
├── data/                              # 数据目录
│   ├── FB_ontology/                   # Freebase KG 本体数据
│   │   ├── fb_roles                   # Freebase 关系列表
│   │   ├── fb_types                   # Freebase 类型信息
│   │   ├── reverse_properties         # 逆向关系字典
│   │   ├── full_reverse_properties.json
│   │   ├── domain_dict / domain_info  # 关系 domain 信息
│   │   └── README.md
│   │
│   ├── WD_ontology/                   # Wikidata KG 本体数据
│   │   ├── all_wikidata_property.csv  # 全部 Wikidata 属性
│   │   ├── wikidata_classes_all.json  # Wikidata 类别
│   │   ├── wikidata_numeral_property_dict.json  # 数值属性字典
│   │   └── wikidata_property_id2label_dict.json # 属性 ID→标签映射
│   │
│   ├── original_dataset/              # 原始 KBQA 数据集（golden SPARQL）
│   │   ├── ComplexWebQuestions_test.json
│   │   ├── grailqa_v1.0_dev.json
│   │   ├── webqsp_test.json
│   │   └── requama_test.json          # ReQUMA 数据集
│   │
│   ├── input/                         # 预处理后的输入数据
│   │   ├── ReQUMA/dataset.json        # ReQUMA 数据集（含问题分解）
│   │   ├── cwq/                       # CWQ: 实体链接结果、test suite 等
│   │   ├── GrailQA_v1.0/              # GrailQA: 实体链接结果、test suite 等
│   │   └── WebQSP/                    # WebQSP: 实体链接结果、test suite 等
│   │
│   ├── output/                        # 实验输出
│   │   ├── quad_search_results/       # QuAD 搜索 & Rerank 输出
│   │   │   ├── cwq/                   # CWQ 搜索结果
│   │   │   ├── grailqa/               # GrailQA 搜索结果
│   │   │   ├── webqsp/                # WebQSP 搜索结果
│   │   │   └── requama/               # ReQUMA 搜索结果
│   │   ├── annotation_performance_experiment/  # 实验 5.3: SQA 方法性能评估
│   │   ├── kbqa_performance_experiment/        # 实验 5.2: KBQA 性能评估
│   │   └── manual_annotation_experiment/       # 实验 5.1: 人工标注实验
│   │
│   └── test/                          # 测试数据
│
├── src/                               # 源代码（模块化组织）
│   ├── core/                          # 核心：常量、枚举、工具函数
│   │   ├── common.py                  # DATASET/Dataset/Method 枚举，Freebase/Wikidata 常量
│   │   └── utils.py                   # JSON 读写、日志、PRF1 计算
│   │
│   ├── sparql/                        # SPARQL 执行、解析、预处理
│   │   ├── executor.py                # SPARQL 查询执行器（ODBC + SPARQLWrapper 双后端）
│   │   ├── sparql_utils.py            # SPARQL 解析、语法树编辑、TED 计算
│   │   ├── sparql_utils_new.py        # 扩展版：支持 Wikidata 逆关系
│   │   ├── process.py                 # SPARQL 预处理（标准化、token 替换）
│   │   └── awudima/                   # 第三方 PLY SPARQL 语法解析器
│   │
│   ├── logical_form/                  # 逻辑形式：S-expression、图结构、转换
│   │   ├── s_expression_utils.py      # S-expression ↔ SPARQL 转换（基础版）
│   │   ├── s_expression_utils_new.py  # 扩展版：增加 Test Suite Score 功能
│   │   ├── logic_form_util.py         # Lisp/S-expression → SPARQL 顶层转换
│   │   ├── graph_utils.py             # 图结构处理 & 编辑距离计算
│   │   ├── simple_graph.py            # 简单图定义（Node, Edge, SimpleGraph）
│   │   └── sparql_to_sexp_utils.py    # SPARQL → S-expression 反向解析
│   │
│   ├── linking/                       # 实体链接 & 语义相似度
│   │   ├── entity_linker.py           # 实体链接（FACC1, Freebase Class, SUTime）
│   │   ├── facc1_index.py             # FACC1 实体表面形式索引（内存版）
│   │   ├── semantic_sim.py            # 基于 PLM embedding 的语义相似度
│   │   ├── sentence_bert.py           # SentenceBERT 相似度预测器
│   │   └── faiss_indexer.py           # FAISS 向量索引构建
│   │
│   ├── quad/                          # QuAD 方法核心：查询搜索 & 问题分解
│   │   ├── search.py                  # Beam Search 查询搜索算法（核心模块）
│   │   ├── decomposition.py           # 复杂问题分解为子问题
│   │   └── rerank.py                  # GPT/LLM 对候选查询重排序
│   │
│   ├── llm/                           # LLM 调用
│   │   └── agent.py                   # OpenAI API 请求封装、查询意图分类
│   │
│   ├── concurrent/                    # 并发执行
│   │   └── executor.py                # 通用并发执行框架（多线程/多进程）
│   │
│   └── metrics/                       # 评估指标（纯算法，无 I/O）
│       ├── minimum_TSS.py             # TSS 实现：Method 1（全量扰动）等
│       └── minimum_TED.py             # 树编辑距离计算（ZSS 算法）
│
├── scripts/                           # 实验 & 评估脚本
│   ├── experiment_runner.py           # 统一实验入口：搜索、Rerank、TSS、TED、Pipeline
│   ├── annotation_performance_evaluation.py  # 实验 5.3: SQA 方法评估
│   ├── manual_annotation_evaluation.py       # 实验 5.1: 人工标注评估
│   ├── kbqa_performance_evaluation.py        # 实验 5.2: KBQA 性能评估
│   └── recompute_annotation_metrics.py       # 重算标注指标
│
├── tests/                             # 测试
│   └── test_executor_backends.py      # executor 双后端测试
│
├── paper.tex                          # ESWC 2026 论文源文件
├── requirements.txt                   # Python 依赖
├── README.md                          # 英文 README
├── README_CN.md                       # 本文件
└── .gitignore
```

---

## 环境配置

### Conda 环境

推荐使用 conda 环境 `sqa_verify`：

```bash
conda create -n sqa_verify python=3.12
conda activate sqa_verify
```

### 依赖安装

```bash
pip install -r requirements.txt
```

主要依赖：

| 库 | 用途 |
|---|------|
| `torch`, `transformers`, `sentence-transformers` | 语义相似度计算（BGE-reranker） |
| `openai` | LLM 调用（GPT-3.5-Turbo 用于问题分解和重排序） |
| `ply` | SPARQL 语法解析（awudima parser） |
| `networkx` | 图结构处理 |
| `pyodbc` | Freebase/Wikidata Virtuoso ODBC 连接（推荐，大结果集性能显著优于 HTTP） |
| `SPARQLWrapper` | HTTP SPARQL 端点连接（备选后端，无需 Virtuoso driver） |
| `anytree` | 语法树结构 |
| `zss` | 树编辑距离计算（Zhang-Shasha） |
| `scikit-learn`, `scipy`, `numpy` | 通用科学计算 |

> **注意**：NumPy 需 < 2.0（scipy/sklearn 兼容性要求）。conda 环境 `sqa_verify` 已配置好。

### 外部服务

| 服务 | 说明 |
|------|------|
| **Freebase Virtuoso** | ODBC 或 HTTP SPARQL 端点，用于 CWQ / GrailQA / WebQSP 实验 |
| **Wikidata 2023 端点** | HTTP SPARQL 端点，用于 ReQUMA 实验 |
| **OpenAI API** | 用于 QuAD 方法中的问题分解和查询重排序 |

### 实体链接

- **Freebase**：使用 FACC1 表面形式匹配
- **Wikidata**：使用 CLOCQ 实体链接工具

> **注意**：实体链接流水线较长，问题分解为简单的 prompt 上下文学习，因此本仓库不提供对应的运行流水线。分解与链接结果已附着在 `data/input/` 各数据集目录中，对应代码（`src/linking/`、`src/quad/decomposition.py`）仅供参考。

---

## How to Run

所有实验通过 `scripts/experiment_runner.py` CLI 统一入口运行。

```bash
python scripts/experiment_runner.py <dataset> <step> [options]
```

### 可用命令

| Step | 说明 |
|------|------|
| `search` | Beam Search 构造 SPARQL 查询 |
| `rerank` | GPT 对候选查询重排序 |
| `tss` | Method 1 (Paper): 全量 KB 扰动，确定性 |
| `tss3` | Method 3 (Precomputed): 预生成 test suite + 采样 F1 |
| `pipeline` | 一条龙：search → rerank |

**参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--size N` | 处理 N 题 | 100 |
| `--model M` | GPT 模型名 | gpt-4o-mini |
| `--suffix S` | 输出子目录后缀 | 无 |
| `--rerank-file` | 指定 rerank 输入文件 | 自动推导 |
| `--test-suite-file` | Method 3 test suite 路径 | 自动推导/生成 |
| `--test-cases-limit` | Method 3 每题采样数 | 5 |

### 示例

```bash
# 搜索
python scripts/experiment_runner.py CWQ search --size 100
python scripts/experiment_runner.py ReQUMA search --size 100

# Rerank
python scripts/experiment_runner.py CWQ rerank --model gpt-4o-mini --size 100

# TSS Method 1 (Paper) — 适合 Freebase
python scripts/experiment_runner.py CWQ tss --size 100
python scripts/experiment_runner.py WebQSP tss --size 100

# TSS Method 3 (Precomputed) — 适合 Wikidata
# test suite 不存在时自动生成
python scripts/experiment_runner.py ReQUMA tss3 --size 100

# 一条龙 Pipeline
python scripts/experiment_runner.py CWQ pipeline --size 100 --model gpt-4o-mini

# 批量运行所有数据集
python scripts/experiment_runner.py all search --size 100
```

### TSS 方法选择

| 方法 | 原理 | 适用 |
|------|------|------|
| Method 1 (Paper) | 全量 KB 扰动，单次查询，确定性 | Freebase (CWQ/GrailQA/WebQSP) |
| Method 3 (Precomputed) | 预生成 test suite + 采样 + F1 | Wikidata (ReQUMA) |

> 本质上 Method 3 是理论版本（Method 1）的随机采样近似：用一个查询的可能扰动空间代表两个查询的完整扰动空间。由于查询返回的数据量级较小，效率可控。虽然引入 F1 分数，但 F1 与 Jaccard 相似度思想一致，不影响趋势。

> Wikidata 数据量大，Method 1 单题 ~10s 且对端点压力大，建议优先用 Method 3。

### 复现论文实验

#### 实验 5.1：人工标注评估（ReQUMA）

```bash
python scripts/manual_annotation_evaluation.py
```

输入 `data/original_dataset/requama_test.json`，输出到 `data/output/manual_annotation_experiment/`。

#### 实验 5.2：KBQA 性能评估（Freebase）

```bash
python scripts/kbqa_performance_evaluation.py
```

#### 实验 5.3：SQA 方法性能评估

```bash
# Step 2: 评估 TSS + TED
python scripts/annotation_performance_evaluation.py
```

---

## 代码架构

三层架构：

```
Layer 3 (目录级):  scripts/annotation_performance_evaluation.py
                   scripts/manual_annotation_evaluation.py
                   └─ 多方法遍历、表格/图表生成、目录级批量处理

Layer 2 (文件级):  scripts/experiment_runner.py
                   └─ querier 构建、数据 I/O、批量入口、Pipeline 编排

Layer 1 (算法级):  src/metrics/minimum_TSS.py / minimum_TED.py
                   └─ 纯算法，无文件 I/O、无 querier 构造
```

**模块依赖**：

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

### 1. TED 实现的修正（重要更新）

论文发表后，我们检查了 TED 的实现并修复了若干 bug。

**典型问题示例**（Freebase）：QuAD 会生成 `FILTER NOT EXISTS {?cvt ns:name ?name}`，通过名字的不存在性来约束 CVT 节点。理论上这不影响语义——Freebase 的 property schema 已保证 CVT 节点不应有 name。但这会**单向惩罚 QuAD**，导致其 TED 偏大。

**修正后的 Table 4 TED 数据**：

| Methods | CWQ TED | GrailQA TED | WebQSP TED |
|---------|:-------:|:-----------:|:----------:|
| QGG⁺ | 0.431 | -- | 0.191 |
| QGG-A | 0.568 | -- | 0.187 |
| KB-BINDER | -- | 0.407 | 0.466 |
| LSQ-A | 0.732 | 0.438 | 0.422 |
| QueryAgent | 0.644 | 0.275 | 0.252 |
| QuAD | 0.451 | 0.375 | 0.220 |
| QuAD$_{+g}$ | **0.355** | **0.188** | **0.135** |

> ⚠️ 修正后的结果对 QuAD 更为有利。这种对具体特例判断的敏感性进一步说明了 TED 实现的不稳定性。

---

### 2. TSS 实现的细节与修正

#### 2.1 Freebase TSS（Method 1，理论版本）

- **原理**：严格遵循论文理论描述，取 KG 上**全部**的可能扰动
- **用于**：实验 5.3（Freebase 上的方法评估，Table 4）
- **关键参数**：`MaxReturnRows`

**修正说明**：

原始论文数据中 TSS 基于**不设置** MaxReturnRows 的结果计算。但在重新配置 Virtuoso 后，MaxReturnRows 被设置为 50K。

在实际使用场景下，`MaxReturnRows` 是一个重要参数：
- **太小** → 随机扰动集合重合概率低，误差大
- **太大** → 耗时长且易超时
- **建议**：在保证效率可控的情况下尽可能增大

将 `MaxReturnRows` 设置为 **200K** 后重跑实验，得到如下修正结果：

**TSS 对比（Old vs New）**：

| Method | CWQ Old | CWQ New | GrailQA Old | GrailQA New | WebQSP Old | WebQSP New |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| QGG⁺ | 0.132 | 0.142 | -- | -- | 0.602 | 0.618 |
| QGG-A | 0.076 | 0.039 | -- | -- | 0.665 | 0.686 |
| KB-BINDER | -- | -- | 0.429 | 0.429 | 0.301 | 0.312 |
| LSQ-A | 0.036 | 0.004 | 0.160 | 0.162 | 0.202 | 0.224 |
| QueryAgent | 0.106 | 0.060 | 0.527 | 0.530 | 0.551 | 0.566 |
| QuAD | 0.236 | 0.270 | 0.548 | 0.550 | 0.554 | 0.630 |
| QuAD$_{+g}$ | 0.368 | 0.423 | 0.710 | 0.715 | 0.669 | 0.747 |

> 修正结果上 QuAD 的优势更加明显。

**平均 TSS 计算耗时（MaxReturnRows=200K，秒/题）**：

| Method | CWQ | GrailQA | WebQSP |
| :--- | :---: | :---: | :---: |
| QGG⁺ | 8.62 | -- | 16.65 |
| QGG-A | 5.31 | -- | 15.67 |
| KB-BINDER | -- | 5.10 | 14.80 |
| LSQ-A | 0.19 | 1.09 | 3.34 |
| QueryAgent | 6.02 | 2.02 | 15.62 |
| QuAD | 9.32 | 2.30 | 15.80 |
| QuAD$_{+g}$ | 13.80 | 3.98 | 18.56 |

> 通过添加查询结果缓存等技巧，可以大幅加速。

本仓库默认使用 **ODBC** 后端（`src/core/common.py` KB 连接配置区），大结果集场景下 ODBC 显著快于 SPARQLWrapper HTTP。ODBC 后端 TSS Method 1 实测耗时与上表同一量级：

| Dataset | Setting | Items | Per Item |
|---------|---------|------:|:---------:|
| CWQ | quad golden | 869 | 3.4s |
| CWQ | quad linked | 841 | 2.2s |
| GrailQA | quad golden | 867 | 1.3s |
| GrailQA | quad linked | 692 | 0.7s |
| WebQSP | quad golden | 966 | 8.0s |
| WebQSP | quad linked | 922 | 3.9s |

> 如无 ODBC 环境，可切换为 SPARQLWrapper：将 `build_sparql_querier()` 的 `service` 参数改为 `"sparql_wrapper"`。

#### 2.2 Wikidata TSS（Method 3，实用版本）

对于 Wikidata 这种量级的 KG，全量扰动（Method 1）在效率上不可行。Method 3 采用预计算 test suite + 随机采样的策略：

1. 在 golden query 上预生成 test suite examples（一次生成，可复用）
2. 将 test suite 应用于 simulated query，计算 F1 分数
3. 取多次采样的平均值作为 TSS

> 这是理论版本的随机采样近似。由于查询返回数据量较小，效率可控。

---

### 4. AI 使用声明

本项目的代码整理借助了 Claude Code（DeepSeekV4-pro 作为 LLM backbone）完成。

---

## 已知问题

### 代码结构

| # | 位置 | 问题 |
|---|------|------|
| 1 | `src/sparql/sparql_utils_Old.py` | 旧版文件，含 hardcoded 绝对路径，保留仅供参考 |
| 2 | `src/linking/semantic_sim.py` | embedding 文件（~1GB）路径指向外部存储，已添加注释说明 |

### DATASET 枚举混用

`src/core/common.py` 中旧的 `DATASET` 枚举将两种语义混在一起：
- **数据集**（GRAIL, CWQ, WEBQ, LC2, QALD）— 决定 KB 类型和 SPARQL 方言
- **方法来源**（QGG, QUERYAGENT, BINDER, LSQ 等）— 决定 SPARQL 预处理模式

代码已新增 `Dataset` 和 `Method` 两个独立枚举，部分入口函数已迁移至新类型。但 `SyntaxTreeEditor` 等底层模块仍使用旧 `DATASET` 枚举做直接比较。

**当前状态：不影响使用。** 所有调用链传递的值都是正确的，仅类型系统未强制执行。

---

## 引用

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
