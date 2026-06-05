#!/usr/bin/env python3
"""
Annotation Performance Evaluation — TED & TSS Computation.

读取 method_outputs 中各方法的输出，计算 TED 和 TSS 指标。
/ Reads method outputs, computes TED and TSS metrics.
"""

import sys
import os
import math
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.common import DATASET, to_dataset
from src.core.utils import load_json, dump_json, setup_custom_logger
from scripts.experiment_runner import compute_ted
from src.metrics.minimum_TSS import new_test_suite_similarity
from scripts.experiment_runner import build_sparql_querier, _get_sparql_prefix


# ============================================================
# 数据集推断 / Dataset inference from subdir name
# ============================================================
DATASET_DIR_MAP = {
    'cwq': DATASET.CWQ,
    'grailqa': DATASET.GRAIL,
    'webqsp': DATASET.WEBQ,
}

GOLDEN_FILES = {
    'cwq': 'data/input/cwq/cwq_test_0_1000_linking.json',
    'grailqa': 'data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_linking.json',
    'webqsp': 'data/input/WebQSP/webqsp_test_0_1000_linking.json',
}


# ============================================================
# TSS 辅助 / TSS helpers
# ============================================================

def _build_simulated_sparql(item: dict, prefix: str) -> str:
    """构造 simulated_sparql_query / Build simulated SPARQL query for TSS."""
    if "gpt_select_top1" in item and "sparql" in item.get("gpt_select_top1", {}):
        return prefix + "\n" + item["gpt_select_top1"]["sparql"]
    if item.get("search_results") and "sparql" in item["search_results"][0]:
        return prefix + "\n" + item["search_results"][0]["sparql"]
    if item.get("search_results"):
        return prefix + "\n" + item["search_results"][0]
    return prefix


def _normalize_sparql(s: str) -> str:
    """规范化 SPARQL 字符串用于比较 / Normalize SPARQL for comparison."""
    import re
    s = s.replace("PREFIX ns: <http://rdf.freebase.com/ns/>", "")
    return re.sub(r'\s+', ' ', s).strip()


def _find_gpt_top1_sparql(item: dict) -> str | None:
    """在 search_results 中找到与 gpt_select_top1 匹配的 SPARQL.
    / Find the search_result SPARQL matching gpt_select_top1."""
    if "gpt_select_top1" not in item or "search_results" not in item:
        return None
    gpt_sparql = _normalize_sparql(item["gpt_select_top1"].get("sparql", ""))
    if not gpt_sparql:
        return None
    for sr in item["search_results"]:
        if _normalize_sparql(sr.get("sparql", "")) == gpt_sparql:
            return sr["sparql"]
    # 模糊匹配：去空格后比较 / Fallback: compare without spaces
    gpt_compact = gpt_sparql.replace(" ", "")
    for sr in item["search_results"]:
        if _normalize_sparql(sr.get("sparql", "")).replace(" ", "") == gpt_compact:
            return sr["sparql"]
    return None


def infer_method(filename_stem: str) -> DATASET:
    """从文件名推断方法类型 / Infer method type from filename stem."""
    stem = filename_stem.lower().replace('-', '_')
    if 'qgg_a' in stem:
        return DATASET.QGG
    if 'qgg' in stem:
        return DATASET.QGG
    if 'lsq' in stem:
        return DATASET.LSQ
    if 'queryagent' in stem:
        return DATASET.QUERYAGENT
    if 'kbbinder' in stem or 'binder' in stem:
        return DATASET.BINDER
    if 'quad_golden' in stem:
        return DATASET.SIMULATED_FREEBASE
    if 'quad_link' in stem:
        return DATASET.SIMULATED_FREEBASE
    raise ValueError(f"Cannot infer method from filename: {filename_stem}")


# ============================================================
# 发现输入文件 / Discover input files
# ============================================================
def discover_files(input_dir: str) -> list:
    """扫描 input_dir 下的子目录，返回待处理文件列表.
    / Scan subdirs under input_dir, return list of file configs.
    """
    tasks = []
    for subdir in sorted(os.listdir(input_dir)):
        subdir_path = os.path.join(input_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        if subdir not in DATASET_DIR_MAP:
            print(f"  [WARN] Unknown subdir: {subdir}, skipping")
            continue

        golden_ds = DATASET_DIR_MAP[subdir]
        for fname in sorted(os.listdir(subdir_path)):
            if not fname.endswith('.json'):
                continue
            stem = os.path.splitext(fname)[0]
            try:
                predicted_ds = infer_method(stem)
            except ValueError as e:
                print(f"  [WARN] {subdir}/{fname}: {e}, skipping")
                continue
            tasks.append({
                "path": os.path.join(subdir_path, fname),
                "subdir": subdir,
                "stem": stem,
                "golden_dataset": golden_ds,
                "predicted_dataset": predicted_ds,
            })
    return tasks


# ============================================================
# 单文件 TED 计算 / Single-file TED worker
# ============================================================
def _compute_ted_for_file(cfg: dict) -> dict:
    """对单个方法输出文件计算 TED / Compute TED for a single method output file.

    cfg keys:
        mode: "gpt_top1" — 只计算 gpt 选中的查询的 TED
              "best_of_all" — 计算所有 search_result 的 TED，取最小值
    """
    input_path = cfg["input_path"]
    golden_file = cfg["golden_file"]
    golden_dataset = cfg["golden_dataset"]
    predicted_dataset = cfg["predicted_dataset"]
    output_path = cfg["output_path"]
    label = cfg["label"]
    mode = cfg.get("mode", "gpt_top1")

    # 跳过已完成的结果 / Skip if already computed
    if os.path.exists(output_path):
        existing = load_json(output_path)
        if isinstance(existing, dict) and "tree_edit_distance_summary" in existing:
            summary = existing["tree_edit_distance_summary"]
            print(f"[SKIP] {label} already exists, TED={summary['average']:.4f}")
            return {"label": label, "average": summary["average"],
                    "isomorphism_rate": summary["isomorphism_rate"]}

    log_name = label.replace("/", "_")
    logger = setup_custom_logger(f"data/test/ted_{log_name}.log")

    src_data = load_json(golden_file)
    detection_result = load_json(input_path)

    qid_to_src = {str(example["qid"]): example for example in src_data}
    edit_distance_list = []

    # 确定数据格式 / Determine data format (dict-of-dicts vs results-list)
    if isinstance(detection_result, dict) and "results" in detection_result:
        raw_items = detection_result["results"]
        items = [(item["qid"], item) for item in raw_items if "qid" in item]
    elif isinstance(detection_result, dict):
        items = [(qid, item) for qid, item in detection_result.items()
                 if isinstance(item, dict) and "qid" in item]
    else:
        items = [(item["qid"], item) for item in detection_result if "qid" in item]

    gpt_top1_count = 0  # 实际使用 gpt_top1 的 item 数

    if mode == "gpt_top1":
        print(f"[{label}] Mode: gpt_top1, Processing {len(items)} items")
    else:
        print(f"[{label}] Mode: best_of_all, Processing {len(items)} items")

    for qid, detection_item in items:
        qid = str(qid)
        try:
            lf_key = "search_results" if "search_results" in detection_item else "searched_queries"

            if not detection_item.get(lf_key):
                edit_distance_list.append(1.0)
                detection_item["TED"] = 1.0
                continue

            if qid not in qid_to_src:
                edit_distance_list.append(1.0)
                detection_item["TED"] = 1.0
                continue

            golden_sparql = qid_to_src[qid]["golden_sparql_query"]

            # 确定要计算的 SPARQL 列表 / Determine which SPARQLs to evaluate
            if mode == "gpt_top1":
                gpt_sparql = _find_gpt_top1_sparql(detection_item)
                if gpt_sparql:
                    candidate_sparqls = [gpt_sparql]
                    gpt_top1_count += 1
                else:
                    # gpt_select_top1 不存在或无法匹配 → 退回 best_of_all
                    candidate_sparqls = [r["sparql"] for r in detection_item[lf_key]]
            else:
                candidate_sparqls = [r["sparql"] for r in detection_item[lf_key]]

            ted_list = []
            for predicted_result in candidate_sparqls:
                predicted_sparql = 'PREFIX ns: <http://rdf.freebase.com/ns/> ' + predicted_result
                ted = compute_ted(
                    golden_sparql=golden_sparql,
                    predicted_sparql=predicted_sparql,
                    golden_dataset=golden_dataset,
                    predicted_dataset=predicted_dataset,
                    logger=logger,
                )
                ted_list.append(ted)

            final_ted = min(ted_list)
            detection_item["TED"] = final_ted
            detection_item["TED_list"] = ted_list
            edit_distance_list.append(final_ted)

        except Exception as e:
            logger.error(f"file: {input_path}, qid: {qid}; error: {e}")
            detection_item["TED"] = 1.0
            edit_distance_list.append(1.0)

    if mode == "gpt_top1":
        print(f"[{label}] gpt_top1 matched on {gpt_top1_count}/{len(items)} items")

    n_total = len(src_data)
    average = sum(edit_distance_list) / n_total if n_total else 0.0
    isomorphism_rate = (len([x for x in edit_distance_list if math.isclose(x, 0.0)]) / n_total
                        if n_total else 0.0)
    detection_result["tree_edit_distance_summary"] = {
        "average": average,
        "isomorphism_rate": isomorphism_rate,
    }

    dump_json(detection_result, output_path)
    print(f"[{label}] Done: avg TED={average:.4f}, isomorphism_rate={isomorphism_rate:.4f}")
    return {"label": label, "average": average, "isomorphism_rate": isomorphism_rate}


# ============================================================
# 主流程 / Main entry point
# ============================================================
def run_ted_evaluation(input_dir: str, output_dir: str, num_workers: int = 8,
                       mode: str = "gpt_top1"):
    """读取 method_outputs，计算 TED，输出结果.

    Args:
        mode: "gpt_top1" — 只计算 gpt 选中查询的 TED（默认，仅 QuAD 系生效）
              "best_of_all" — 计算所有 search_result 的 TED，取最小值
    """
    if mode not in ("gpt_top1", "best_of_all"):
        raise ValueError(f"Unknown mode: {mode}")

    os.makedirs(output_dir, exist_ok=True)

    tasks = discover_files(input_dir)
    if not tasks:
        print("No files found to process.")
        return

    # 为每个任务生成完整配置 / Build full config for each task
    configs = []
    for t in tasks:
        golden_file = GOLDEN_FILES.get(t["subdir"])
        if not golden_file or not os.path.exists(golden_file):
            print(f"  [WARN] Golden file not found for {t['subdir']}: {golden_file}")
            continue
        label = f"{t['subdir']}/{t['stem']}"
        output_path = os.path.join(output_dir, f"{t['subdir']}_{t['stem']}_edit_distance.json")
        configs.append({
            "input_path": t["path"],
            "golden_file": golden_file,
            "golden_dataset": t["golden_dataset"],
            "predicted_dataset": t["predicted_dataset"],
            "output_path": output_path,
            "label": label,
            "mode": mode,
        })

    num_workers = min(num_workers, len(configs))
    print(f"Processing {len(configs)} files with {num_workers} workers, mode={mode}...")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print()

    with Pool(processes=num_workers) as pool:
        results = pool.map(_compute_ted_for_file, configs)

    print(f"\n{'='*60}")
    print("  TED Evaluation Summary")
    print(f"{'='*60}")
    for r in sorted(results, key=lambda x: x['label']):
        print(f"  {r['label']:40s}  avg TED = {r['average']:.4f},  iso_rate = {r['isomorphism_rate']:.4f}")


# ============================================================
# TSS: 单文件计算 / Single-file TSS worker
# ============================================================
def _compute_tss_for_file(cfg: dict) -> dict:
    """对单个方法输出文件计算 TSS / Compute TSS for a single method output file."""
    input_path = cfg["input_path"]
    golden_file = cfg["golden_file"]
    golden_dataset = cfg["golden_dataset"]
    predicted_dataset = cfg["predicted_dataset"]
    output_path = cfg["output_path"]
    label = cfg["label"]
    log_dir = cfg.get("log_dir")

    # 跳过已完成的结果 / Skip if already computed
    if os.path.exists(output_path):
        existing = load_json(output_path)
        if isinstance(existing, dict) and "test_suite_s" in existing:
            summary = existing["test_suite_s"]
            print(f"[SKIP] {label} already exists, TSS={summary['average']:.4f}")
            return {"label": label, "average": summary["average"]}

    log_name = label.replace("/", "_")
    log_path = os.path.join(log_dir, f"tss_{log_name}.log") if log_dir else f"data/test/tss_{log_name}.log"
    logger = setup_custom_logger(log_path)

    golden_data = load_json(golden_file)
    constructed_data = load_json(input_path)

    # 标准化 golden_data → dict keyed by qid
    if isinstance(golden_data, list):
        qid_to_golden = {str(item["qid"]): item for item in golden_data}
    elif "results" in golden_data:
        qid_to_golden = {str(item["qid"]): item for item in golden_data["results"]}
    else:
        qid_to_golden = {str(qid): item for qid, item in golden_data.items()
                         if isinstance(item, dict)}

    # 确定迭代来源 / Determine iteration source
    if "results" in constructed_data:
        src_items = constructed_data["results"]
    else:
        src_items = [v for v in constructed_data.values() if isinstance(v, dict)]

    # 字段名归一化：非 Quad 方法用 searched_queries，统一为 search_results
    # Normalize field name: non-Quad methods use searched_queries, unify to search_results
    for item in src_items:
        if isinstance(item, dict) and "searched_queries" in item and "search_results" not in item:
            item["search_results"] = item["searched_queries"]

    sparql_querier = build_sparql_querier(
        to_dataset(golden_dataset), logger, service="sparql_wrapper")
    prefix = _get_sparql_prefix(golden_dataset)

    f1_list = []
    skipped = {"no_qid": 0, "no_golden": 0, "no_results": 0}
    valid_items = []

    for item in src_items:
        if "qid" not in item:
            item["test_suite_similarity"] = 0.0
            f1_list.append(0.0)
            skipped["no_qid"] += 1
            continue
        qid = str(item["qid"])
        if qid not in qid_to_golden:
            item["test_suite_similarity"] = 0.0
            f1_list.append(0.0)
            skipped["no_golden"] += 1
            continue
        if not item.get("search_results"):
            item["test_suite_similarity"] = 0.0
            f1_list.append(0.0)
            skipped["no_results"] += 1
            continue
        valid_items.append(item)

    logger.info(f"Valid (will compute TSS): {len(valid_items)}, Skipped: {skipped}")
    print(f"[{label}] {len(valid_items)} items (skipped: {skipped})", flush=True)

    overall_start = time.time()
    n_total = len(valid_items)
    last_report = overall_start

    for i, item in enumerate(valid_items):
        t0 = time.time()
        qid = str(item["qid"])
        golden_item = qid_to_golden[qid]

        # 构造 simulated_sparql_query，优先使用已有的
        if "simulated_sparql_query" not in item:
            item["simulated_sparql_query"] = _build_simulated_sparql(item, prefix)

        similarity, num_replace = new_test_suite_similarity(
            golden_item, item, golden_dataset, predicted_dataset, sparql_querier, logger)

        elapsed = time.time() - t0
        item["test_suite_similarity"] = similarity
        item["all_replace_num"] = num_replace
        logger.info(f"[{i+1}/{n_total}] qid: {qid}, TSS: {similarity:.4f}, "
                    f"|T|: {num_replace}, time: {elapsed:.1f}s")
        f1_list.append(similarity)

        # 每 30s 输出一次进度 / Print progress every 30s
        now = time.time()
        if now - last_report >= 30:
            pct = (i + 1) / n_total * 100
            avg_t = (now - overall_start) / (i + 1)
            eta = avg_t * (n_total - i - 1)
            print(f"[{label}] {i+1}/{n_total} ({pct:.0f}%) | "
                  f"avg {avg_t:.1f}s/item | ETA {eta/60:.0f}m", flush=True)
            last_report = now

    total_elapsed = time.time() - overall_start
    avg_time = total_elapsed / n_total if n_total else 0
    logger.info(f"Completed {n_total} items in {total_elapsed:.1f}s ({avg_time:.1f}s/item)")

    valid_f1 = [s for s in f1_list if s >= 0]
    constructed_data["test_suite_s"] = {
        "average": sum(valid_f1) / len(valid_f1) if valid_f1 else 0.0,
        "count": len(valid_f1),
    }

    dump_json(constructed_data, output_path)
    avg = constructed_data["test_suite_s"]["average"]
    print(f"[{label}] Done: avg TSS={avg:.4f}, count={len(valid_f1)}")
    return {"label": label, "average": avg}


# ============================================================
# TSS: 主流程 / Main entry point
# ============================================================
def run_tss_evaluation(input_dir: str, output_dir: str):
    """读取 method_outputs，计算 TSS，输出结果（单进程，避免数据库压力）."""
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "log")
    os.makedirs(log_dir, exist_ok=True)

    tasks = discover_files(input_dir)
    if not tasks:
        print("No files found to process.")
        return

    print(f"Processing {len(tasks)} files (single-process)...")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Log:    {log_dir}")
    print()

    overall_start = time.time()
    results = []
    for i, t in enumerate(tasks):
        golden_file = GOLDEN_FILES.get(t["subdir"])
        if not golden_file or not os.path.exists(golden_file):
            print(f"  [WARN] Golden file not found for {t['subdir']}: {golden_file}")
            continue
        label = f"{t['subdir']}/{t['stem']}"
        output_path = os.path.join(output_dir, f"{t['subdir']}_{t['stem']}_tss.json")
        cfg = {
            "input_path": t["path"],
            "golden_file": golden_file,
            "golden_dataset": t["golden_dataset"],
            "predicted_dataset": t["predicted_dataset"],
            "output_path": output_path,
            "label": label,
            "log_dir": log_dir,
        }
        print(f"[{i+1}/{len(tasks)}] {label} ...", flush=True)
        r = _compute_tss_for_file(cfg)
        results.append(r)

    elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"  TSS Evaluation Summary ({elapsed:.0f}s / {elapsed/60:.1f}m)")
    print(f"{'='*60}")
    for r in sorted(results, key=lambda x: x['label']):
        print(f"  {r['label']:40s}  avg TSS = {r['average']:.4f}")


# ============================================================
# TED 表格生成 / TED Markdown table generator
# ============================================================

METHOD_DISPLAY = {
    'qgg': 'QGG⁺',
    'qgg_a': 'QGG-A',
    'qgg-a': 'QGG-A',
    'lsq': 'LSQ-A',
    'queryagent': 'QueryAgent',
    'kbbinder': 'KB-BINDER',
    'quad_golden': 'QuAD₊𝓰',
    'quad_link': 'QuAD',
    'quad_linked': 'QuAD',
}

DATASET_DISPLAY = {
    'cwq': 'CWQ',
    'grailqa': 'GrailQA',
    'webqsp': 'WebQSP',
}

DATASET_ORDER = ['cwq', 'grailqa', 'webqsp']
METHOD_ORDER = ['qgg', 'qgg_a', 'qgg-a', 'kbbinder', 'lsq', 'queryagent',
                'quad_link', 'quad_linked', 'quad_golden']


def _extract_ted_avg(data: dict) -> float | None:
    """从 per-item TED 重新计算平均值 / Recompute average from per-item TED."""
    if "results" in data:
        items = data["results"]
    else:
        items = [v for k, v in data.items() if isinstance(v, dict) and "TED" in v]
    if not items:
        if "tree_edit_distance_summary" in data:
            return data["tree_edit_distance_summary"]["average"]
        return None
    teds = [it.get("TED", 1.0) for it in items]
    return sum(teds) / len(teds)


def generate_ted_table(ted_dir: str) -> str:
    """扫描 TED 结果目录，生成 Markdown 对比表格并保存到目录下.
    / Scan TED results dir, generate Markdown comparison table.

    Returns the table string.
    """
    # 收集数据 / Collect data
    results = {}  # {(dataset, method_display): ted_avg}
    datasets_seen = set()
    methods_seen = set()

    for fname in sorted(os.listdir(ted_dir)):
        if not fname.endswith('_edit_distance.json'):
            continue
        stem = fname.replace('_edit_distance.json', '')
        parts = stem.split('_', 1)
        if len(parts) != 2:
            continue
        ds_key, method_key = parts[0], parts[1]
        if ds_key not in DATASET_DISPLAY:
            continue
        method_display = METHOD_DISPLAY.get(method_key)
        if not method_display:
            continue

        data = load_json(os.path.join(ted_dir, fname))
        ted = _extract_ted_avg(data) if data else None
        if ted is not None:
            results[(ds_key, method_display)] = ted
            datasets_seen.add(ds_key)
            methods_seen.add(method_display)

    datasets = [d for d in DATASET_ORDER if d in datasets_seen]
    methods = [m for m in METHOD_ORDER if METHOD_DISPLAY.get(m) in methods_seen]
    # dedup method displays (qgg_a / qgg-a → same display)
    seen_displays = set()
    methods_unique = []
    for m in methods:
        d = METHOD_DISPLAY[m]
        if d not in seen_displays:
            seen_displays.add(d)
            methods_unique.append(d)

    if not datasets or not methods_unique:
        print("No TED data found.")
        return ""

    # 按列判定 best / second-best (TED 越小越好)
    ranks = {}  # {(ds, method): 'best' | 'second'}
    for ds in datasets:
        vals = [(m, results[(ds, m)]) for m in methods_unique if (ds, m) in results]
        vals.sort(key=lambda x: x[1])
        if vals:
            best_val = vals[0][1]
            for m, v in vals:
                if abs(v - best_val) < 0.0005:
                    ranks[(ds, m)] = "best"
                else:
                    break
            second_entries = [(m, v) for m, v in vals if ranks.get((ds, m)) != "best"]
            if second_entries:
                second_val = second_entries[0][1]
                for m, v in second_entries:
                    if abs(v - second_val) < 0.0005:
                        ranks[(ds, m)] = "second"
                    else:
                        break

    def md_cell(ds, m):
        ted = results.get((ds, m))
        if ted is None:
            return "--"
        s = f"{ted:.3f}"
        rank = ranks.get((ds, m))
        if rank == "best":
            return f"**{s}**"
        elif rank == "second":
            return f"<u>{s}</u>"
        return s

    # 构建 Markdown 表格
    header = "| Methods | " + " | ".join(f"{DATASET_DISPLAY[d]} TED" for d in datasets) + " |"
    sep = "|---------|" + "|".join(":-------:" for _ in datasets) + "|"
    lines = [header, sep]

    for m in methods_unique:
        cells = [md_cell(ds, m) for ds in datasets]
        # 只保留至少有一个非空的方法行
        if all(c == "--" for c in cells):
            continue
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    table = "\n".join(lines) + "\n"

    # 保存
    out_path = os.path.join(ted_dir, "ted_comparison.md")
    with open(out_path, "w") as f:
        f.write(table)
    print(f"TED comparison table saved to {out_path}")
    print()
    print(table)

    return table


# ============================================================
# TSS 表格生成 / TSS Markdown table generator
# ============================================================

def _extract_tss_avg(data: dict) -> float | None:
    """从 TSS 结果中提取平均值 / Extract average TSS from result data."""
    if "test_suite_s" in data:
        val = data["test_suite_s"]
        return val["average"] if isinstance(val, dict) else val
    # fallback: recompute from per-item
    if "results" in data:
        items = data["results"]
    else:
        items = [v for k, v in data.items() if isinstance(v, dict) and "qid" in v]
    vals = [it.get("test_suite_similarity") for it in items]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def generate_tss_table(tss_dir: str) -> str:
    """扫描 TSS 结果目录，生成 Markdown 对比表格并保存到目录下.
    / Scan TSS results dir, generate Markdown comparison table.

    TSS 越大越好 / TSS: higher is better.
    """
    results = {}  # {(dataset, method_display): tss_avg}
    datasets_seen = set()
    methods_seen = set()

    for fname in sorted(os.listdir(tss_dir)):
        if not fname.endswith('_tss.json'):
            continue
        stem = fname.replace('_tss.json', '')
        parts = stem.split('_', 1)
        if len(parts) != 2:
            continue
        ds_key, method_key = parts[0], parts[1]
        if ds_key not in DATASET_DISPLAY:
            continue
        method_display = METHOD_DISPLAY.get(method_key)
        if not method_display:
            continue

        data = load_json(os.path.join(tss_dir, fname))
        tss = _extract_tss_avg(data) if data else None
        if tss is not None:
            results[(ds_key, method_display)] = tss
            datasets_seen.add(ds_key)
            methods_seen.add(method_display)

    datasets = [d for d in DATASET_ORDER if d in datasets_seen]
    methods = [m for m in METHOD_ORDER if METHOD_DISPLAY.get(m) in methods_seen]
    seen_displays = set()
    methods_unique = []
    for m in methods:
        d = METHOD_DISPLAY[m]
        if d not in seen_displays:
            seen_displays.add(d)
            methods_unique.append(d)

    if not datasets or not methods_unique:
        print("No TSS data found.")
        return ""

    # 按列判定 best / second-best (TSS 越大越好)
    ranks = {}
    for ds in datasets:
        vals = [(m, results[(ds, m)]) for m in methods_unique if (ds, m) in results]
        vals.sort(key=lambda x: x[1], reverse=True)  # higher is better
        if vals:
            best_val = vals[0][1]
            for m, v in vals:
                if abs(v - best_val) < 0.0005:
                    ranks[(ds, m)] = "best"
                else:
                    break
            second_entries = [(m, v) for m, v in vals if ranks.get((ds, m)) != "best"]
            if second_entries:
                second_val = second_entries[0][1]
                for m, v in second_entries:
                    if abs(v - second_val) < 0.0005:
                        ranks[(ds, m)] = "second"
                    else:
                        break

    def md_cell(ds, m):
        tss = results.get((ds, m))
        if tss is None:
            return "--"
        s = f"{tss:.3f}"
        rank = ranks.get((ds, m))
        if rank == "best":
            return f"**{s}**"
        elif rank == "second":
            return f"<u>{s}</u>"
        return s

    header = "| Methods | " + " | ".join(f"{DATASET_DISPLAY[d]} TSS" for d in datasets) + " |"
    sep = "|---------|" + "|".join(":-------:" for _ in datasets) + "|"
    lines = [header, sep]

    for m in methods_unique:
        cells = [md_cell(ds, m) for ds in datasets]
        if all(c == "--" for c in cells):
            continue
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    table = "\n".join(lines) + "\n"

    out_path = os.path.join(tss_dir, "tss_comparison.md")
    with open(out_path, "w") as f:
        f.write(table)
    print(f"TSS comparison table saved to {out_path}")
    print()
    print(table)

    return table


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    # Step 1: TED
    run_ted_evaluation(
        input_dir="data/output/annotation_performance_experiment/method_outputs",
        output_dir="data/output/annotation_performance_experiment/annotation_analysis/ted_new",
        num_workers=8,
        mode="gpt_top1",
    )

    # Step 2: TSS
    run_tss_evaluation(
        input_dir="data/output/annotation_performance_experiment/method_outputs",
        output_dir="data/output/annotation_performance_experiment/annotation_analysis/tss_new_odbc",
    )

    # Step 3: 生成 TED 对比表格
    generate_ted_table(
        ted_dir="data/output/annotation_performance_experiment/annotation_analysis/ted_new",
    )

    # Step 4: 生成 TSS 对比表格
    generate_tss_table(
        tss_dir="data/output/annotation_performance_experiment/annotation_analysis/tss_new_odbc",
    )
