#!/usr/bin/env python3
"""
实验运行器 / Experiment Runner — SPARQL Query Annotating 统一 CLI 入口.

用法:
  python scripts/experiment_runner.py <dataset> <step> [options]

数据集 (case-insensitive):  cwq | webqsp | grailqa | ReQUMA | all
步骤:                       search | rerank | tss | tss3 | pipeline

Examples:
  # 搜索 100 条
  python scripts/experiment_runner.py cwq search --size 100

  # GPT-4o-mini rerank
  python scripts/experiment_runner.py cwq rerank --model gpt-4o-mini

  # TSS Method 1 (Paper, 全量扰动)
  python scripts/experiment_runner.py cwq tss --size 100

  # TSS Method 3 (Precomputed test suite, 适合 Wikidata)
  python scripts/experiment_runner.py requama tss3 --size 100

  # 完整流水线: search → rerank
  python scripts/experiment_runner.py cwq pipeline --size 100 --model gpt-4o-mini
"""

import sys, os, time, re, argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.core.common import (Dataset, DATASET,
    ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab,
    ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official,
    ODBC_CONFIG_WIKIDATA_2023, SPARQL_wrapper_path_wikidata_2023,
    WIKIDATA_PREFIX_LIST,
    TSS_TEST_SUITE_GEN_LIMIT, TSS_RANDOM_SEED)
from src.core.utils import load_json, dump_json, setup_custom_logger
from src.sparql.executor import SparqlOdbcQuerierNoSexpr, SparqlOdbcQuerierNoSexprWikidata

# 端点分配 / Endpoint Assignment:
#   CWQ/WebQSP → official (8890): 纯日期格式，与 linking 答案一致
#   GrailQA    → dkilab  (8896):  已有实验基于此，保持兼容
#   ReQUMA     → Wikidata (2023): Wikidata KB


# ═══════════════════════════════════════════════════════════════
# 数据路径 / Data Paths (按需修改)
# ═══════════════════════════════════════════════════════════════

DATA = {
    "cwq": {
        "dataset":           Dataset.CWQ,
        "ds_enum":           DATASET.CWQ,
        "input":             "data/input/cwq/cwq_test_0_1000_linking.json",
        "test_suite":        "data/input/cwq/cwq_test_0_1000_test_suite.json",
        "test_suite_method3": "data/input/cwq/cwq_test_suite_method3.json",  # pre-generated test_suite_examples for Method 3
        "output_dir":        "data/output/quad_search_results/cwq",
        "sw_path":           SPARQL_wrapper_path_official,
        "odbc_config":       ODBC_CONFIG_OFFICIAL,
    },
    "webqsp": {
        "dataset":           Dataset.WEBQ,
        "ds_enum":           DATASET.WEBQ,
        "input":             "data/input/WebQSP/webqsp_test_0_1000_linking.json",
        "test_suite":        "data/input/WebQSP/webqsp_test_0_1000_test_suite.json",
        "test_suite_method3": "data/input/WebQSP/webqsp_test_suite_method3.json",
        "output_dir":        "data/output/quad_search_results/webqsp",
        "sw_path":           SPARQL_wrapper_path_official,
        "odbc_config":       ODBC_CONFIG_OFFICIAL,
    },
    "grailqa": {
        "dataset":           Dataset.GRAIL,
        "ds_enum":           DATASET.GRAIL,
        "input":             "data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_linking.json",
        "test_suite":        "data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_test_suite.json",
        "test_suite_method3": "data/input/GrailQA_v1.0/grailqa_test_suite_method3.json",
        "output_dir":        "data/output/quad_search_results/grailqa",
        "sw_path":           SPARQL_wrapper_path_dkilab,
        "odbc_config":       ODBC_CONFIG_DKILAB,
    },
    "ReQUMA": {
        "dataset":           Dataset.REQUMA,
        "ds_enum":           DATASET.REQUMA,
        "input":             "data/input/ReQUMA/dataset.json",
        "test_suite":        "data/original_dataset/requama_test.json",  # reformatted: golden_sparql → golden_sparql_query
        "test_suite_method3": "data/input/ReQUMA/requama_test_suite_method3.json",
        "output_dir":        "data/output/quad_search_results/requama",
        "sw_path":           SPARQL_wrapper_path_wikidata_2023,
        "odbc_config":       ODBC_CONFIG_WIKIDATA_2023,
    },
}


# ═══════════════════════════════════════════════════════════════
# SPARQL Querier Builders (file-level: construct queriers from config)
# ═══════════════════════════════════════════════════════════════

def build_sparql_querier(dataset: Dataset, logger, timeout: int = 60,
                          service: str = "sparql_wrapper",
                          sw_path: str = None, odbc_config: str = None) -> SparqlOdbcQuerierNoSexpr:
    """Build SPARQL querier for Freebase datasets."""
    if dataset == Dataset.GRAIL:
        return SparqlOdbcQuerierNoSexpr(
            sparql_wrapper_path=sw_path or SPARQL_wrapper_path_dkilab,
            logger=logger, service=service,
            odbc_config=odbc_config or ODBC_CONFIG_DKILAB, timeout=timeout)
    elif dataset in [Dataset.CWQ, Dataset.WEBQ]:
        return SparqlOdbcQuerierNoSexpr(
            sparql_wrapper_path=sw_path or SPARQL_wrapper_path_official,
            logger=logger, service=service,
            odbc_config=odbc_config or ODBC_CONFIG_OFFICIAL, timeout=timeout)
    else:
        raise NotImplementedError(
            f"dataset: {dataset} — use build_sparql_querier_wikidata for Wikidata datasets.")


def build_sparql_querier_wikidata(dataset: DATASET, logger, timeout: int = 300,
                                   service: str = "sparql_wrapper",
                                   sw_path: str = None, odbc_config: str = None) -> SparqlOdbcQuerierNoSexprWikidata:
    """Build SPARQL querier for Wikidata datasets."""
    if sw_path is None:
        sw_path = SPARQL_wrapper_path_wikidata_2023
    if odbc_config is None:
        odbc_config = ODBC_CONFIG_WIKIDATA_2023
    if dataset in [DATASET.LC2, DATASET.QALD, DATASET.SIMULATED_WIKIDATA, DATASET.REQUMA]:
        return SparqlOdbcQuerierNoSexprWikidata(
            sparql_wrapper_path=sw_path, logger=logger, service=service,
            odbc_config=odbc_config, timeout=timeout)
    else:
        raise NotImplementedError(
            f"dataset: {dataset} — Wikidata querier not available. Use build_sparql_querier for Freebase.")


def _get_sparql_prefix(dataset: DATASET) -> str:
    """返回数据集的 SPARQL 前缀."""
    if dataset in [DATASET.CWQ, DATASET.WEBQ, DATASET.GRAIL]:
        return "PREFIX ns: <http://rdf.freebase.com/ns/> "
    else:
        return WIKIDATA_PREFIX_LIST


# ═══════════════════════════════════════════════════════════════
# Test Suite Generation (file-level: I/O + querier + algorithm call)
# ═══════════════════════════════════════════════════════════════

def run_generate_test_suite(golden_linking_file: str, output_file: str,
                             dataset: DATASET, logger_path: str,
                             test_cases_limit: int = TSS_TEST_SUITE_GEN_LIMIT,
                             service: str = "sparql_wrapper",
                             sw_path: str = None, odbc_config: str = None):
    """Generate test_suite_examples for all golden questions and save to file.

    This is the data preparation step for Method 3 (run_tss_precomputed).
    """
    from src.metrics.minimum_TSS import generate_test_suites_batch

    logger = setup_custom_logger(logger_path)
    try:
        querier = build_sparql_querier(dataset, logger, service=service, timeout=300,
                                        sw_path=sw_path, odbc_config=odbc_config)
    except NotImplementedError:
        querier = build_sparql_querier_wikidata(dataset, logger, service=service, timeout=300,
                                                  sw_path=sw_path, odbc_config=odbc_config)

    golden_data = load_json(golden_linking_file)
    if isinstance(golden_data, dict) and 'results' in golden_data:
        src_items = golden_data['results']
    elif isinstance(golden_data, list):
        src_items = golden_data
    else:
        src_items = list(golden_data.values())

    logger.info(f"Total golden items: {len(src_items)}")
    logger.info(f"test_cases_limit: {test_cases_limit}")

    generate_test_suites_batch(src_items, dataset, querier, logger, test_cases_limit)
    dump_json(src_items, output_file)
    logger.info(f"Saved to: {output_file}")


def run_tss_precomputed(method_output_file: str, test_suite_file: str,
                         output_file: str, dataset: DATASET, method: DATASET,
                         logger_path: str, service: str = "sparql_wrapper",
                         timeout: int = 300,
                         sw_path: str = None, odbc_config: str = None):
    """Run Method 3 (Precomputed Test Suite) on a batch.

    Requires test_suite_file with pre-generated test_suite_examples.
    """
    from src.metrics.minimum_TSS import execute_precomputed_test_suite

    logger = setup_custom_logger(logger_path)
    try:
        querier = build_sparql_querier(dataset, logger, service=service, timeout=timeout,
                                        sw_path=sw_path, odbc_config=odbc_config)
    except NotImplementedError:
        querier = build_sparql_querier_wikidata(dataset, logger, service=service, timeout=timeout,
                                                  sw_path=sw_path, odbc_config=odbc_config)

    constructed_data = load_json(method_output_file)
    test_suite_data = load_json(test_suite_file)
    result = execute_precomputed_test_suite(
        constructed_data, test_suite_data, dataset, method, querier, logger)
    dump_json(result, output_file)
    return result


# ═══════════════════════════════════════════════════════════════
# Step 1: Search
# ═══════════════════════════════════════════════════════════════

def _output_dir(cfg, suffix):
    """构建带后缀的输出目录."""
    base = cfg['output_dir']
    return f"{base}/{suffix}" if suffix else base

def run_search(ds_key: str, size: int = 100, use_golden: bool = True, suffix: str = None):
    """运行 LF 搜索 / Run LF search."""
    cfg = DATA[ds_key]
    od = _output_dir(cfg, suffix)
    out_file = f"{od}/{ds_key}_test_{size}_golden.json"
    log_file = f"{od}/../log/{ds_key}_search_{size}.log"

    if os.path.exists(out_file):
        print(f"[SKIP] {out_file} exists ({len(load_json(out_file))} items)")
        return out_file

    from src.quad.search import LFSearchAlgorithm

    data = load_json(cfg["input"])[:size]
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    t0 = time.time()
    search = LFSearchAlgorithm(
        dataset=cfg["dataset"], log_file_dir=log_file,
        use_golden_entity=use_golden, use_neighbor_entity=False)
    search.sparql_executor.direct_manage_2hop = True
    results, _ = search.begin_search(data, return_sexpr=True, only_keep_accurate=False)
    dump_json(results, out_file)

    n_found = sum(1 for r in results if r.get("search_results"))
    print(f"[Search] {ds_key}: {n_found}/{len(results)} found ({time.time()-t0:.0f}s)")
    return out_file


# ═══════════════════════════════════════════════════════════════
# Step 2: GPT Rerank
# ═══════════════════════════════════════════════════════════════

def run_rerank(ds_key: str, search_file: str = None, model: str = "gpt-4o-mini",
               size: int = 100, max_rounds: int = 5, suffix: str = None):
    """GPT-4o 重排序，失败自动重试 / GPT rerank with auto-retry."""
    cfg = DATA[ds_key]
    od = _output_dir(cfg, suffix)

    if search_file is None:
        search_file = f"{od}/{ds_key}_test_{size}_golden.json"
    out_file = f"{od}/{ds_key}_test_{size}_golden_{model.replace('-', '')}_rerank.json"

    from src.llm.agent import OpenAiRequestor, QueryIntent
    from src.quad.rerank import prompt_freebase

    origin = {it['qid']: it for it in load_json(cfg["input"])}

    # Load or init
    if os.path.exists(out_file):
        reranked = load_json(out_file)
        if not isinstance(reranked, dict):
            reranked = {it['qid']: it for it in reranked}
    else:
        raw = load_json(search_file)
        reranked = {it['qid']: it for it in raw} if isinstance(raw, list) else dict(raw)

    for rnd in range(max_rounds):
        needs = {}
        for qid, item in reranked.items():
            sr = item.get("search_results", [])
            if len(sr) <= 1:
                continue
            gpt = item.get("gpt_select_top1")
            if not gpt or gpt == "error":
                needs[qid] = item

        if not needs:
            print(f"[Rerank] {ds_key}: all OK (round {rnd})")
            break

        print(f"[Rerank] round {rnd+1}: {len(needs)} items")

        # Build prompts
        prompts = {}
        for qid, item in needs.items():
            txt = prompt_freebase + "\n#\nQuestions:\n" + item["question"] + "\nSPARQLs:\n"
            id2label = {}
            id2label.update(origin.get(qid, {}).get("linked_item_to_label", {}))
            id2label.update(origin.get(qid, {}).get("golden_item_to_label", {}))
            for idx, r in enumerate(item["search_results"]):
                s = r["sparql"].replace("\n", "")
                for mid, label in id2label.items():
                    s = s.replace("ns:" + mid, label.replace(" ", "_"))
                txt += f"{idx+1}. {s}\n"
            prompts[qid] = txt

        msgs = list(prompts.values())
        msg2qid = {v: k for k, v in prompts.items()}
        asker = OpenAiRequestor(model)
        raw_results = asker.concurrent_query(msgs, min(10, len(msgs)), intent=QueryIntent.QUESTION)

        for msg, output in raw_results:
            qid = msg2qid.get(msg)
            if qid is None:
                continue
            try:
                idx = int(re.findall(r"\d+", output)[0]) - 1
                reranked[qid]["gpt_select_top1"] = reranked[qid]["search_results"][idx]
            except:
                reranked[qid]["gpt_select_top1"] = "error"

        dump_json(reranked, out_file)

    n_need = sum(1 for it in reranked.values() if len(it.get("search_results", [])) > 1)
    n_ok = sum(1 for it in reranked.values()
               if len(it.get("search_results", [])) > 1
               and it.get("gpt_select_top1") and it["gpt_select_top1"] != "error")
    print(f"[Rerank] {ds_key}: {n_ok}/{n_need} OK, {n_need - n_ok} errors")
    return out_file


# ═══════════════════════════════════════════════════════════════
# TSS Method 1 (Paper): all-at-once KB perturbation, deterministic
# ═══════════════════════════════════════════════════════════════

def run_tss_method1(ds_key: str, rerank_file: str = None, model: str = "gpt-4o-mini",
                    size: int = 100, suffix: str = None):
    """Method 1 (Paper) TSS: all-at-once KB perturbation, single query, deterministic.

    Supports Freebase (CWQ/WebQSP/GrailQA) and Wikidata (ReQUMA) datasets.
    """
    import logging
    from src.metrics.minimum_TSS import new_test_suite_similarity

    cfg = DATA[ds_key]
    od = _output_dir(cfg, suffix)

    if rerank_file is None:
        model_short = model.replace("-", "")
        rerank_file = f"{od}/{ds_key}_test_{size}_golden_{model_short}_rerank.json"
    tss_file = f"{od}/{ds_key}_test_{size}_golden_{model.replace('-', '')}_tss_method1.json"
    log_file = f"{od}/../log/{ds_key}_tss1_{size}.log"

    reranked = load_json(rerank_file)
    items = list(reranked.values()) if isinstance(reranked, dict) else reranked
    golden = load_json(cfg["test_suite"])
    qid_to_golden = {str(it["qid"]): it for it in golden}

    logger = logging.getLogger("tss")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.WARNING)

    # Build querier — dispatch by dataset type
    if cfg["ds_enum"] in (DATASET.CWQ, DATASET.WEBQ, DATASET.GRAIL):
        querier = build_sparql_querier(
            cfg["dataset"], logger, service="sparql_wrapper",
            sw_path=cfg["sw_path"], odbc_config=cfg["odbc_config"], timeout=60)
        method = DATASET.SIMULATED_FREEBASE
    else:
        querier = build_sparql_querier_wikidata(
            cfg["ds_enum"], logger, service="sparql_wrapper",
            sw_path=cfg.get("sw_path"), odbc_config=cfg.get("odbc_config"), timeout=60)
        method = DATASET.SIMULATED_WIKIDATA
    querier.timeout = 60

    prefix = _get_sparql_prefix(cfg["ds_enum"])

    gpt_tss, first_tss = [], []
    t0 = time.time()

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            gpt_tss.append(0.0); first_tss.append(0.0); continue
        qid = str(item.get("qid", ""))
        sr = item.get("search_results", [])
        if not sr or qid not in qid_to_golden:
            gpt_tss.append(0.0); first_tss.append(0.0); continue

        golden_item = qid_to_golden[qid]

        # First result TSS
        s0 = sr[0].get("sparql", str(sr[0]))
        try:
            sim0, _ = new_test_suite_similarity(
                golden_item, {"qid": qid, "simulated_sparql_query": prefix + "\n" + s0},
                cfg["ds_enum"], method, querier, logger)
        except:
            sim0 = 0.0
        first_tss.append(sim0)

        # GPT-selected TSS
        gpt = item.get("gpt_select_top1")
        sparql = gpt.get("sparql", s0) if (gpt and gpt != "error") else s0
        try:
            sim, _ = new_test_suite_similarity(
                golden_item, {"qid": qid, "simulated_sparql_query": prefix + "\n" + sparql},
                cfg["ds_enum"], method, querier, logger)
        except:
            sim = 0.0
        gpt_tss.append(sim)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}] first={sum(first_tss)/len(first_tss):.4f}  gpt={sum(gpt_tss)/len(gpt_tss):.4f}  ({time.time()-t0:.0f}s)")

    f_avg = sum(first_tss) / len(first_tss) if first_tss else 0
    g_avg = sum(gpt_tss) / len(gpt_tss) if gpt_tss else 0

    dump_json({"test_suite_s": {"average": g_avg, "count": len(gpt_tss), "first_only_avg": f_avg},
                "results": items}, tss_file)

    print(f"[TSS-M1] {ds_key}: first={f_avg:.4f}  gpt={g_avg:.4f}  gain={g_avg-f_avg:+.4f}  ({time.time()-t0:.0f}s)")
    return tss_file


# ═══════════════════════════════════════════════════════════════
# TSS Method 3 (Precomputed): pre-generated test_suite_examples
# ═══════════════════════════════════════════════════════════════

def run_tss_method3(ds_key: str, rerank_file: str = None,
                    test_suite_file: str = None,
                    model: str = "gpt-4o-mini", size: int = 100,
                    suffix: str = None,
                    test_cases_limit: int = TSS_TEST_SUITE_GEN_LIMIT,
                    random_seed: int = TSS_RANDOM_SEED):
    """Method 3 (Precomputed) TSS: pre-generated test suite evaluation.

    与 Method 1 (all-at-once KB 扰动) 不同，Method 3 使用预生成的
    test_suite_examples {(binding) → answer} 进行评估。
    如果 test_suite_file 不存在，自动从 golden data 生成并缓存到
    {output_dir}/test_suite/ 下。
    """
    import logging

    cfg = DATA[ds_key]
    od = _output_dir(cfg, suffix)

    # ── 路径推导 ──
    if rerank_file is None:
        rerank_file = f"{od}/{ds_key}_test_{size}_golden_{model.replace('-', '')}_rerank.json"

    if test_suite_file is None:
        ts_dir = f"{od}/generated_test_suite"
        test_suite_file = f"{ts_dir}/{ds_key}_test_suite_method3.json"

    tss_file = f"{od}/{ds_key}_test_{size}_golden_{model.replace('-', '')}_tss_method3.json"
    log_file = f"{od}/../log/{ds_key}_tss3_{size}.log"

    # ── 自动生成 test suite（一次性操作）──
    if not os.path.exists(test_suite_file):
        print(f"[TSS-M3] Test suite 不存在: {test_suite_file}")
        print(f"[TSS-M3] ⚠ 开始自动生成（一次性操作，每条 golden 需多次 KB 查询，"
              f"test_cases_limit={test_cases_limit}）...")

        if cfg["test_suite"] is None:
            print(f"[TSS-M3] ❌ {ds_key}: test_suite 路径未配置，无法生成")
            return None

        os.makedirs(os.path.dirname(test_suite_file), exist_ok=True)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        t0 = time.time()
        run_generate_test_suite(
            golden_linking_file=cfg["test_suite"],
            output_file=test_suite_file,
            dataset=cfg["ds_enum"],
            logger_path=log_file,
            test_cases_limit=test_cases_limit,
            service="sparql_wrapper",
            sw_path=cfg.get("sw_path"),
            odbc_config=cfg.get("odbc_config"),
        )
        print(f"[TSS-M3] Test suite 已生成 → {test_suite_file}"
              f"  ({time.time() - t0:.0f}s)")

    # ── 评估 ──
    # 构造查询对应的 KB 类型
    if cfg["ds_enum"] in (DATASET.CWQ, DATASET.WEBQ, DATASET.GRAIL):
        method = DATASET.SIMULATED_FREEBASE
    else:
        method = DATASET.SIMULATED_WIKIDATA  # ReQUMA

    os.makedirs(os.path.dirname(tss_file), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    t0 = time.time()
    run_tss_precomputed(
        method_output_file=rerank_file,
        test_suite_file=test_suite_file,
        output_file=tss_file,
        dataset=cfg["ds_enum"],
        method=method,
        logger_path=log_file,
        service="sparql_wrapper",
        timeout=300,
        sw_path=cfg.get("sw_path"),
        odbc_config=cfg.get("odbc_config"),
    )

    # ── 简要结果 ──
    result = load_json(tss_file)
    summary = result.get("summary", {}).get("test_suite", {})
    avg_f1 = summary.get("average_f1", 0.0)
    n_items = len([it for it in (result.get("results", result.values()) if isinstance(result, dict) else result)
                   if isinstance(it, dict) and it.get("test_suite") is not None])

    print(f"[TSS-M3] {ds_key}: avg_f1={avg_f1:.4f}  n={n_items}"
          f"  ({time.time() - t0:.0f}s)  → {tss_file}")
    return tss_file


# ═══════════════════════════════════════════════════════════════
# TED: Tree Edit Distance (file-level)
# ═══════════════════════════════════════════════════════════════

def compute_ted(golden_sparql: str, predicted_sparql: str,
                golden_dataset, predicted_dataset, logger=None) -> float:
    """Compute Tree Edit Distance between a golden and predicted SPARQL query.

    Thin wrapper around minimum_ted for convenient import from runner.
    """
    from src.metrics.minimum_TED import minimum_ted
    return minimum_ted(golden_sparql, predicted_sparql,
                       golden_dataset, predicted_dataset, logger)


# ═══════════════════════════════════════════════════════════════
# Pipeline: search → rerank
# ═══════════════════════════════════════════════════════════════

def run_pipeline(ds_key: str, size: int = 100, model: str = "gpt-4o-mini", suffix: str = None):
    """完整流水线: search + rerank. TSS 是独立快速检查入口，不包含在 pipeline 中."""
    print(f"\n{'='*60}")
    print(f"  Pipeline: {ds_key}  (size={size}, model={model}, suffix={suffix})")
    print(f"{'='*60}")

    search_file = run_search(ds_key, size, suffix=suffix)
    rerank_file = run_rerank(ds_key, search_file, model, size, suffix=suffix)

    print(f"\n  Done: {rerank_file}")
    return rerank_file


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    ds_names = list(DATA.keys())
    parser = argparse.ArgumentParser(
        description="SPARQL Query Annotating — 实验运行器 / Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/experiment_runner.py cwq search --size 100
  python scripts/experiment_runner.py cwq rerank --model gpt-4o-mini
  python scripts/experiment_runner.py cwq tss --size 100
  python scripts/experiment_runner.py requama tss3 --size 100
  python scripts/experiment_runner.py cwq pipeline --size 100""")
    parser.add_argument("dataset", nargs="?", default=None,
                        help=f"数据集名 (case-insensitive): {', '.join(ds_names)}, all")
    parser.add_argument("step", nargs="?", default=None,
                        choices=["search", "rerank", "tss", "tss3", "pipeline"],
                        help="步骤: search | rerank | tss (Method1) | tss3 (Method3) | pipeline")
    parser.add_argument("--size", type=int, default=100, help="问题数 (default: 100)")
    parser.add_argument("--model", default="gpt-4o-mini", help="GPT 模型 (default: gpt-4o-mini)")
    parser.add_argument("--search-file", help="自定义搜索文件路径")
    parser.add_argument("--rerank-file", help="自定义 rerank 文件路径")
    parser.add_argument("--suffix", default=None, help="输出子目录后缀")
    parser.add_argument("--test-suite-file", default=None, help="Method 3 test suite 文件路径")
    parser.add_argument("--test-cases-limit", type=int, default=TSS_TEST_SUITE_GEN_LIMIT,
                        help=f"Method 3 每题采样数 (default: {TSS_TEST_SUITE_GEN_LIMIT})")
    args = parser.parse_args()

    if args.dataset is None:
        parser.print_help()
        return

    # Case-insensitive dataset lookup
    ds_lower_map = {k.lower(): k for k in ds_names}
    ds_input = args.dataset.lower()
    if ds_input == "all":
        datasets = ds_names
    elif ds_input in ds_lower_map:
        datasets = [ds_lower_map[ds_input]]
    else:
        print(f"Unknown dataset: {args.dataset}. Available: {', '.join(ds_names)}, all")
        return

    for ds in datasets:
        if args.step == "search":
            run_search(ds, size=args.size, suffix=args.suffix)
        elif args.step == "rerank":
            run_rerank(ds, search_file=args.search_file, model=args.model, size=args.size, suffix=args.suffix)
        elif args.step == "pipeline":
            run_pipeline(ds, size=args.size, model=args.model, suffix=args.suffix)
        elif args.step == "tss":
            run_tss_method1(ds, rerank_file=args.rerank_file, model=args.model, size=args.size, suffix=args.suffix)
        elif args.step == "tss3":
            run_tss_method3(ds, rerank_file=args.rerank_file,
                            test_suite_file=args.test_suite_file,
                            model=args.model, size=args.size, suffix=args.suffix,
                            test_cases_limit=args.test_cases_limit)


if __name__ == "__main__":
    main()
