"""
Manual annotation experiment: metric recomputation + evaluation + visualization.
One script to rule them all.

Usage:
  # Full pipeline: recompute TSS+TED, then generate tables & figures
  conda run -n KBQA_research python scripts/manual_annotation_evaluation.py --recompute

  # Generate tables/figures only (from existing metrics file)
  conda run -n KBQA_research python scripts/manual_annotation_evaluation.py \\
      --input annotation_result_with_metric_original.json --suffix _original

  # Recompute + generate for reproduce
  conda run -n KBQA_research python scripts/manual_annotation_evaluation.py \\
      --recompute --suffix _reproduce
"""

import json
import math
import os
import sys
import re
import time
import random
import logging
import argparse
from enum import Enum
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── Project paths ──
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

DATA_DIR = os.path.join(_project_root, "data", "output", "manual_annotation_experiment", "annotation_analysis")
REQUMA_PATH = os.path.join(_project_root, "data", "original_dataset", "requama_test.json")

# ── Bypass heavy ML imports for SPARQL modules ──
_dummy_st = type(sys)("dummy")
_dummy_st.__file__ = "dummy"
for _mod in ["sentence_transformers", "sentence_transformers.util", "sentence_transformers.backend",
             "sklearn", "sklearn.metrics", "scipy", "scipy.sparse"]:
    sys.modules[_mod] = type(sys)("dummy")

from src.core.common import (DATASET, WIKIDATA_PREFIX_LIST,
    TSS_TEST_SUITE_GEN_LIMIT, TSS_RANDOM_SEED, DEFAULT_SPARQL_SERVICE)
from src.core.utils import load_json, dump_json, setup_custom_logger, get_PRF1
from src.sparql.sparql_utils import SyntaxTreeEditor
from src.metrics.minimum_TSS import (
    _generate_test_suite_for_golden, _get_sparql_prefix,
    TARGET_VARIABLE_NAME, TEST_SUITE_GEN_LIMIT,
)
from scripts.experiment_runner import compute_ted
from scripts.experiment_runner import build_sparql_querier_wikidata

RANDOM_SEED = 374
LOG_FILE = os.path.join(DATA_DIR, "evaluation.log")
ANNOTATION_ORIGINAL = os.path.join(DATA_DIR, "annotation_result_with_metric_original.json")
ANNOTATION_REPRODUCE = os.path.join(DATA_DIR, "annotation_result_with_metric_reproduce.json")
FP_FN_DATA = os.path.join(DATA_DIR, "annotation_fp_fn_original.json")


# ═══════════════════════════════════════════════════════════════
# Metric Recompution
# ═══════════════════════════════════════════════════════════════

class Metric(Enum):
    TED = 1
    TSS = 2


def recompute_tss(data: list, logger: logging.Logger) -> dict:
    """TSS Method 3: Precomputed Test Suite. Returns {idx: tss_value}."""
    logger.info("=" * 60)
    logger.info("Phase 1: TSS Method 3")
    logger.info("=" * 60)

    sparql_querier = build_sparql_querier_wikidata(
        DATASET.REQUMA, logger, service=DEFAULT_SPARQL_SERVICE, timeout=300)
    dataset = DATASET.REQUMA
    method = DATASET.SIMULATED_WIKIDATA

    golden_to_indices = {}
    skip_indices = set()
    for idx, entry in enumerate(data):
        constructed = entry.get('simulated_LF', {}).get('sparql', '')
        golden = entry.get('golden_sparql_query', '')
        if not constructed or not constructed.strip():
            skip_indices.add(idx)
        elif not golden or not golden.strip():
            skip_indices.add(idx)
        else:
            golden_to_indices.setdefault(golden, []).append(idx)

    logger.info(f"Entries: {len(data)}, Skip: {len(skip_indices)}, "
                f"Unique goldens: {len(golden_to_indices)}")

    # Step 1: test suite per unique golden
    test_suite_cache = {}
    golden_errors = 0
    for i, (gs, indices) in enumerate(golden_to_indices.items()):
        if (i + 1) % 50 == 0:
            logger.info(f"  Test suite: {i+1}/{len(golden_to_indices)}")
        try:
            entities = [m.group(0) for m in re.finditer(r'wd:Q\d+', gs)]
            golden_item = {"qid": f"g{i}", "golden_sparql_query": gs,
                           "golden_grounded_items": [{"mid": e} for e in entities]}
            ts = _generate_test_suite_for_golden(
                golden_item, dataset, sparql_querier, logger,
                test_cases_limit=TEST_SUITE_GEN_LIMIT)
            test_suite_cache[gs] = ts
        except Exception as e:
            logger.error(f"  Test suite error golden {i}: {e}")
            test_suite_cache[gs] = []
            golden_errors += 1

    logger.info(f"Test suites: {len(test_suite_cache)} ({golden_errors} errors)")

    # Step 2: evaluate
    tss_results = {}
    eval_count, eval_errors = 0, 0
    prefix = _get_sparql_prefix(dataset)

    for idx, entry in enumerate(data):
        if idx in skip_indices:
            tss_results[idx] = 0.0
            continue
        golden = entry.get('golden_sparql_query', '')
        constructed = entry.get('simulated_LF', {}).get('sparql', '')
        ts = test_suite_cache.get(golden, [])
        if not ts:
            tss_results[idx] = 0.0
            continue

        full_sparql = (prefix + "\n" + constructed) if not constructed.strip().lower().startswith('prefix') else constructed
        try:
            editor = SyntaxTreeEditor(full_sparql, method, logger)
            editor.preprocess_test_suite(f"?{TARGET_VARIABLE_NAME}")
            processed = editor.sparql_txt
            f1_list = []
            for tc in ts:
                te = SyntaxTreeEditor(processed, method, logger)
                # Apply ALL bindings first (not any() — would short-circuit after first match!)
                overlap = False
                for b, a in tc["binding"].items():
                    if te.update_leaf_value(b, a):
                        overlap = True
                if not overlap:
                    continue
                try:
                    pred = sparql_querier.get_execution_result_one_variable_sparql_wrapper(te.sparql_txt, retry=1)
                except Exception:
                    continue
                _, _, f1 = get_PRF1(pred, tc["answer"])
                f1_list.append(f1)
            tss_results[idx] = sum(f1_list) / len(f1_list) if f1_list else 0.0
            eval_count += 1
        except Exception as e:
            logger.warning(f"  [{idx}] TSS eval error: {e}")
            tss_results[idx] = 0.0
            eval_errors += 1
        if eval_count % 100 == 0:
            logger.info(f"  TSS eval: {eval_count}")

    logger.info(f"TSS done: {eval_count} eval, {eval_errors} errors")
    return tss_results


def recompute_ted(data: list, logger: logging.Logger) -> dict:
    """TED with fixed _break_wdt_wikidata + canonicalization. Returns {idx: ted_value}."""
    logger.info("=" * 60)
    logger.info("Phase 2: TED")
    logger.info("=" * 60)
    ted_results = {}
    errors = 0
    for idx, entry in enumerate(data):
        golden_raw = entry.get('golden_sparql_query', '')
        pred_raw = entry.get('simulated_LF', {}).get('sparql', '')
        if not pred_raw or not pred_raw.strip():
            ted_results[idx] = 1.0
        elif re.match(r'\s*SELECT\s+\*', pred_raw, re.IGNORECASE):
            ted_results[idx] = 1.0
        elif not golden_raw or not golden_raw.strip():
            ted_results[idx] = 1.0
        else:
            try:
                ted = compute_ted(
                    WIKIDATA_PREFIX_LIST + "\n" + golden_raw,
                    WIKIDATA_PREFIX_LIST + "\n" + pred_raw,
                    DATASET.REQUMA, DATASET.SIMULATED_WIKIDATA, logger=None)
                ted_results[idx] = ted
            except Exception as e:
                logger.warning(f"  [{idx}] TED error: {e}")
                ted_results[idx] = 1.0
                errors += 1
        if (idx + 1) % 200 == 0:
            logger.info(f"  TED: {idx+1}/{len(data)}")
    logger.info(f"TED done: {len(ted_results)} results, {errors} errors")
    return ted_results


def recompute_metrics(input_file: str, output_file: str):
    """Recompute TSS (Method 3) and TED, save to output_file."""
    logger = setup_custom_logger(LOG_FILE)
    random.seed(RANDOM_SEED)

    t0 = time.time()
    logger.info(f"Loading: {input_file}")
    data = load_json(input_file)
    logger.info(f"Total entries: {len(data)}")

    # TSS
    tss_results = recompute_tss(data, logger)

    # TED
    ted_results = recompute_ted(data, logger)

    # Merge
    for idx, entry in enumerate(data):
        if idx in tss_results:
            entry['test_suite'] = tss_results[idx]
        if idx in ted_results:
            entry['TED'] = ted_results[idx]

    dump_json(data, output_file)
    elapsed = time.time() - t0

    tss_vals = [e['test_suite'] for e in data if isinstance(e.get('test_suite'), (int, float))]
    ted_vals = [e['TED'] for e in data if isinstance(e.get('TED'), (int, float))]

    logger.info(f"Saved to: {output_file} ({elapsed:.1f}s / {elapsed/60:.1f}min)")
    if tss_vals:
        logger.info(f"TSS: mean={sum(tss_vals)/len(tss_vals):.4f}, "
                    f"==0: {sum(1 for v in tss_vals if v==0)}, ==1: {sum(1 for v in tss_vals if v==1)}")
    if ted_vals:
        logger.info(f"TED: mean={sum(ted_vals)/len(ted_vals):.4f}, "
                    f"==1: {sum(1 for v in ted_vals if v==1)}")

    print(f"\nRecomputed metrics → {output_file}")
    print(f"  TSS: mean={sum(tss_vals)/len(tss_vals):.4f}, ==0: {sum(1 for v in tss_vals if v==0)}, ==1: {sum(1 for v in tss_vals if v==1)}")
    print(f"  TED: mean={sum(ted_vals)/len(ted_vals):.4f}, ==1: {sum(1 for v in ted_vals if v==1)}")


# ═══════════════════════════════════════════════════════════════
# Data Loading & Filtering
# ═══════════════════════════════════════════════════════════════

def load_data_rel(input_path: str) -> list:
    if not os.path.isabs(input_path):
        input_path = os.path.join(DATA_DIR, input_path)
    with open(input_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries from {input_path}")
    return data


def add_golden_sparql(data: list):
    if all('golden_sparql_query' in e for e in data):
        return data
    requma = json.load(open(REQUMA_PATH))
    qid_to_golden = {str(e['qid']): e.get('golden_sparql_query', e.get('golden_sparql', '')) for e in requma}
    added = 0
    for e in data:
        if 'golden_sparql_query' not in e:
            qid = str(e['qid'])
            if qid in qid_to_golden:
                e['golden_sparql_query'] = qid_to_golden[qid]
                added += 1
    print(f"Added golden_sparql_query to {added} entries")
    return data


def filter_by_individual_parse(data: list) -> list:
    valid = []; errors = 0
    for item in data:
        annota = item.get('annota', [])
        if not annota or not isinstance(annota, list) or not annota[0]:
            errors += 1; continue
        try:
            int(str(annota[0]).split('#')[0])
            valid.append(item)
        except (ValueError, IndexError):
            errors += 1
    print(f"Valid by individual parse: {len(valid)}/{len(data)} ({errors} parse errors excluded)")
    return valid


def filter_valid_by_qid(data: list) -> list:
    abandoned_qids = set()
    for item in data:
        time_val = item.get('time', 0)
        annota = item.get('annota', [])
        if time_val == 9999:
            abandoned_qids.add(item['qid'])
        elif annota and isinstance(annota, list) and annota[0]:
            first = str(annota[0]).lower()
            if "abandon" in first or "放弃" in first:
                abandoned_qids.add(item['qid'])
    valid = [item for item in data if item['qid'] not in abandoned_qids]
    print(f"Valid by qid: {len(valid)}/{len(data)} (removed {len(abandoned_qids)} abandoned qids)")
    return valid


# ═══════════════════════════════════════════════════════════════
# Hop Counting
# ═══════════════════════════════════════════════════════════════

def get_annotation_hop(item: dict) -> int:
    annota = item.get('annota', [])
    if annota and isinstance(annota, list) and annota[0]:
        hop = get_number_of_hop(annota[0])
        if hop > 0: return hop
    sim = item.get('simulated_LF', {})
    return get_number_of_hop(sim.get('sparql', ''))


def get_number_of_hop(lf: str) -> int:
    sparql = str(lf)
    return (len(re.findall(r" p:", sparql)) + len(re.findall(r"wdt:", sparql)) +
            len(re.findall(r" ps:", sparql)) + len(re.findall(r" pq:", sparql)) +
            len(re.findall("filter", sparql.lower())))


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def avg(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def calculate_avg_anno_time(data: list) -> float:
    return sum(item['time'] for item in data) / len(data) if data else 0.0


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

METHOD_MAP = {
    'ONLYLINK': 'Baseline', 'QUERYAGENT LINKED': 'QueryAgent',
    'QUERYAGENT GOLDEN': 'QueryAgent_{+g}', 'QUAD LINKED': 'QuAD', 'QUAD GOLDEN': 'QuAD_{+g}',
}
METHOD_ORDER = ['Baseline', 'QueryAgent', 'QueryAgent_{+g}', 'QuAD', 'QuAD_{+g}']
PAPER_VALUES = {
    'Baseline': ('--', '--', '112.1s'), 'QueryAgent': ('0.168', '0.720', '92.1s'),
    'QueryAgent_{+g}': ('0.366', '0.533', '75.8s'), 'QuAD': ('0.336', '0.602', '71.2s'),
    'QuAD_{+g}': ('0.535', '0.451', '52.2s'),
}

TSS_BINS = [('0', lambda v: v == 0), ('(0,0.25]', lambda v: 0 < v <= 0.25),
            ('(0.25,0.75]', lambda v: 0.25 < v <= 0.75), ('(0.75,1]', lambda v: 0.75 < v <= 1)]
TED_BINS = [('1', lambda v: v == 1), ('[0.75,1)', lambda v: 0.75 <= v < 1),
            ('[0.25,0.75)', lambda v: 0.25 <= v < 0.75), ('[0,0.25)', lambda v: 0 <= v < 0.25)]
HOP_LABELS = ['1', '2', '3', '>3']


# ═══════════════════════════════════════════════════════════════
# Table 1: Method Summary
# ═══════════════════════════════════════════════════════════════

def generate_method_summary_table(data_all: list, output_dir: str, suffix: str):
    """Table 1: Average TSS, TED, Annotation Time per method."""
    by_method = defaultdict(list)
    for item in data_all:
        by_method[METHOD_MAP.get(item.get('type', '?'), item.get('type', '?'))].append(item)

    data_parsed = filter_by_individual_parse(data_all)
    by_method_time_all = defaultdict(list)
    for item in data_parsed:
        by_method_time_all[METHOD_MAP.get(item.get('type', '?'), item.get('type', '?'))].append(item)

    data_qid = filter_valid_by_qid(data_all)
    data_859 = [it for it in data_qid if get_annotation_hop(it) > 0]
    by_method_time_859 = defaultdict(list)
    for item in data_859:
        by_method_time_859[METHOD_MAP.get(item.get('type', '?'), item.get('type', '?'))].append(item)

    rows = []
    for mname in METHOD_ORDER:
        items = by_method.get(mname, [])
        items_time_all = by_method_time_all.get(mname, [])
        items_time_859 = by_method_time_859.get(mname, [])

        avg_tss = avg([it.get('test_suite', 0) for it in items])
        avg_ted = avg([it.get('TED', 1) for it in items])
        tss_str, ted_str = ('--', '--') if mname == 'Baseline' else (f'{avg_tss:.3f}', f'{avg_ted:.3f}')

        rows.append({'Method': mname, 'A. TSS': tss_str, 'A. TED': ted_str,
                     'A. ATQ (all)': f'{calculate_avg_anno_time(items_time_all):.1f}s',
                     'A. ATQ (859)': f'{calculate_avg_anno_time(items_time_859):.1f}s',
                     'N (all)': len(items), 'N (time)': len(items_time_all), 'N (859)': len(items_time_859),
                     '_tss': avg_tss, '_ted': avg_ted,
                     '_time_all': calculate_avg_anno_time(items_time_all),
                     '_time_859': calculate_avg_anno_time(items_time_859)})

    # Terminal print
    print(f"\n{'='*100}")
    print(f"Table 1: Method Summary ({suffix})")
    print(f"{'='*100}")
    hdr = f"{'Method':<20} {'TSS':>8} {'TED':>8} {'ATQ(all)':>10} {'ATQ(859)':>10} {'N(all)':>7} {'N(859)':>7}"
    print(hdr); print('-' * 70)
    for r in rows:
        print(f"{r['Method']:<20} {r['A. TSS']:>8} {r['A. TED']:>8} {r['A. ATQ (all)']:>10} "
              f"{r['A. ATQ (859)']:>10} {r['N (all)']:>7} {r['N (859)']:>7}")

    # Paper comparison
    print(f"\nPaper vs Our:")
    print(f"{'Method':<20} {'P.TSS':>8} {'O.TSS':>8} {'P.TED':>8} {'O.TED':>8} {'P.ATQ':>10} {'O.ATQ(all)':>12} {'O.ATQ(859)':>12}")
    print('-' * 86)
    for r in rows:
        m = r['Method']; pv = PAPER_VALUES.get(m, ('?', '?', '?'))
        print(f"{m:<20} {pv[0]:>8} {r['A. TSS']:>8} {pv[1]:>8} {r['A. TED']:>8} "
              f"{pv[2]:>10} {r['A. ATQ (all)']:>12} {r['A. ATQ (859)']:>12}")

    non_bl = [r for r in rows if r['Method'] != 'Baseline']
    if non_bl:
        print(f"\nBest TSS: {max(non_bl, key=lambda r: r['_tss'])['Method']}, "
              f"Best TED: {min(non_bl, key=lambda r: r['_ted'])['Method']}, "
              f"Best Time: {min(non_bl, key=lambda r: r['_time_all'])['Method']}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, f"method_summary{suffix}.json")
    with open(out_json, 'w') as f: json.dump(rows, f, indent=2, ensure_ascii=False)

    # MD output
    best_method = max(non_bl, key=lambda r: r['_tss'])['Method'] if non_bl else ''
    n_all_time = sum(r['N (time)'] for r in rows)
    n_859 = sum(r['N (859)'] for r in rows)
    md = [f"# Table 1: Evaluation Metrics of SQA Task Outputs and Average Annotation Time\n",
          f"TSS and TED are averaged over all 1000 assigned tasks (200 questions × 5 methods).  ",
          f"ATQ (all) is computed on {n_all_time} entries where the annotator's SPARQL could be parsed.  ",
          f"ATQ (859) is an additional column computed on the {n_859} valid annotations ",
          f"(abandoned qids removed, hop>0), matching the subset used for the scatter plot and hop×metric table.\n",
          f"| Method | A. TSS | A. TED | A. ATQ (all) | A. ATQ (859) | N (all) | N (859) |",
          f"|---|---|---|---|---|---|---|"]
    for r in rows:
        b = '**' if r['Method'] == best_method else ''
        md.append(f"| {b}{r['Method']}{b} | {r['A. TSS']} | {r['A. TED']} | "
                  f"{r['A. ATQ (all)']} | {r['A. ATQ (859)']} | {r['N (all)']} | {r['N (859)']} |")
    md.append(f"\n> **Note.** The ATQ (859) column yields slightly lower absolute times since it "
              f"excludes abandoned annotations. The relative ordering and all trends remain "
              f"identical: {best_method} consistently achieves the best performance, "
              f"golden settings outperform linked settings, and QuAD outperforms QueryAgent.")
    out_md = os.path.join(output_dir, f"method_summary{suffix}.md")
    with open(out_md, 'w') as f: f.write('\n'.join(md))
    print(f"Saved: {out_json}, {out_md}")
    return rows


# ═══════════════════════════════════════════════════════════════
# Table 2: Hops × Metric
# ═══════════════════════════════════════════════════════════════

def generate_hop_metric_table(data: list, output_dir: str, suffix: str):
    def bin_by_hop(items):
        bins = {k: [] for k in HOP_LABELS}
        for item in items:
            hop = get_annotation_hop(item)
            if hop == 1: bins['1'].append(item)
            elif hop == 2: bins['2'].append(item)
            elif hop == 3: bins['3'].append(item)
            elif hop > 3: bins['>3'].append(item)
        return bins

    def build_subtable(metric_bins, metric_key):
        rows = []
        for hl in HOP_LABELS:
            rd = {'#Hops': hl if hl != '>3' else '$>$3'}
            for bn, bf in metric_bins:
                bi = [it for it in hop_bins[hl] if bf(it.get(metric_key, 0))]
                rd[bn] = f'{calculate_avg_anno_time(bi):.1f}s' if bi else '--'
            rows.append(rd)
        return rows

    hop_bins = bin_by_hop(data)
    tss_rows = build_subtable(TSS_BINS, 'test_suite')
    ted_rows = build_subtable(TED_BINS, 'TED')

    for label, rows, bins in [('TSS', tss_rows, TSS_BINS), ('TED', ted_rows, TED_BINS)]:
        print(f"\n{'='*80}\nTable 2: Hops × {label} ({suffix})\n{'='*80}")
        hdrs = ['#Hops'] + [b[0] for b in bins]
        print(f"{'#Hops':<8} " + " ".join(f"{h:>12}" for h in hdrs[1:]))
        for r in rows:
            print(f"{r['#Hops']:<8} " + " ".join(f"{r.get(h, '--'):>12}" for h in hdrs[1:]))

    hop_counts = {k: len(v) for k, v in hop_bins.items()}
    print(f"\nHop distribution: {hop_counts}  Total={sum(hop_counts.values())}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"hop_metric_table{suffix}.json")
    with open(out_path, 'w') as f:
        json.dump({'tss': tss_rows, 'ted': ted_rows, 'hop_counts': hop_counts}, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {out_path}")


# ═══════════════════════════════════════════════════════════════
# Figure: Scatter Plot
# ═══════════════════════════════════════════════════════════════

def generate_scatter_plot(data: list, output_dir: str, suffix: str):
    valid = [it for it in data if it.get('test_suite') is not None and it.get('TED') is not None]
    x_tss, y_tss = [it['test_suite'] for it in valid], [it['time'] for it in valid]
    x_ted, y_ted = [it['TED'] for it in valid], [it['time'] for it in valid]
    print(f"Scatter points: {len(valid)}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
    for ax, x, y, c, xl in [(ax1, x_tss, y_tss, 'salmon', 'TSS'), (ax2, x_ted, y_ted, 'deepskyblue', 'TED')]:
        sns.scatterplot(x=x, y=y, ax=ax, color=c, alpha=0.6, s=30)
        ax.set_ylabel('Annotation Time (s)'); ax.set_xlabel(xl); ax.set_xlim(-0.05, 1.05)
    plt.subplots_adjust(hspace=0.25)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"scatter{suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Scatter: {out_path}")


# ═══════════════════════════════════════════════════════════════
# Figure: FP/FN vs Threshold
# ═══════════════════════════════════════════════════════════════

def generate_fp_fn_plot(fp_fn_path: str, output_dir: str, suffix: str):
    data = load_data_rel(fp_fn_path)
    thresholds = [1.0 - i * 0.01 for i in range(20)]

    tss_fn_counts, tss_fp_counts = [], []
    ted_fn_counts, ted_fp_counts = [], []
    f1_fp_counts = []

    for tau in thresholds:
        tss_fn = tss_fp = ted_fn = ted_fp = f1_fp = 0
        for item in data:
            is_pos = item.get('annotated') == 'positive'
            is_neg = item.get('annotated') == 'negative'
            tss_val = item.get('ts', 0)
            ted_val = item.get('TED', 1)
            f1_val = item.get('f1', 0)
            ted_inv = 1.0 - tau
            if is_neg:
                if tss_val >= tau or math.isclose(tss_val, tau): tss_fp += 1
                if math.isclose(f1_val, 1.0): f1_fp += 1
                if ted_val <= ted_inv or math.isclose(ted_val, ted_inv): ted_fp += 1
            elif is_pos:
                if not (tss_val >= tau or math.isclose(tss_val, tau)): tss_fn += 1
                if not (ted_val <= ted_inv or math.isclose(ted_val, ted_inv)): ted_fn += 1
        tss_fn_counts.append(tss_fn); tss_fp_counts.append(tss_fp)
        ted_fn_counts.append(ted_fn); ted_fp_counts.append(ted_fp)
        f1_fp_counts.append(f1_fp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.set_xlim(1.0, 0.81)
    ax1.plot(thresholds, tss_fn_counts, label='TSS', linewidth=3)
    ax1.plot(thresholds, ted_fn_counts, label='TED', linewidth=3)
    ax1.legend(prop={'size': 8}); ax1.grid(); ax1.tick_params(axis='both', labelsize=9)
    ax1.set_xlabel(r'Threshold $\tau$', fontsize=12); ax1.set_ylabel('Number of FNs', fontsize=12)
    ax2.set_xlim(1.0, 0.81)
    ax2.plot(thresholds, tss_fp_counts, label='TSS', linewidth=3)
    ax2.plot(thresholds, ted_fp_counts, label='TED', linewidth=3)
    ax2.plot(thresholds, f1_fp_counts, label='F1', linewidth=3)
    ax2.legend(prop={'size': 8}); ax2.grid(); ax2.tick_params(axis='both', labelsize=9)
    ax2.set_xlabel(r'Threshold $\tau$', fontsize=12); ax2.set_ylabel('Number of FPs', fontsize=12)
    plt.subplots_adjust(wspace=0.3)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"fp_fn{suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"FP/FN plot: {out_path}")


# ═══════════════════════════════════════════════════════════════
# Full Evaluation Pipeline
# ═══════════════════════════════════════════════════════════════

def run_evaluation(input_file: str, suffix: str):
    """Generate all tables and figures from the given metrics file."""
    data = load_data_rel(input_file)
    data = add_golden_sparql(data)

    tables_dir = os.path.join(DATA_DIR, f"tables{suffix}")
    figures_dir = os.path.join(DATA_DIR, f"figures{suffix}")
    print(f"\nOutput: tables → {tables_dir}/\n        figures → {figures_dir}/")

    generate_method_summary_table(data, tables_dir, suffix)
    data_valid = filter_valid_by_qid(data)
    generate_hop_metric_table(data_valid, tables_dir, suffix)
    generate_scatter_plot(data_valid, figures_dir, suffix)
    if os.path.exists(FP_FN_DATA):
        generate_fp_fn_plot(FP_FN_DATA, figures_dir, suffix)
    else:
        print(f"FP/FN data not found: {FP_FN_DATA}")

    print(f"\nDone! → {tables_dir}/  &  {figures_dir}/")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Manual annotation experiment: recompute + evaluate')
    parser.add_argument('--recompute', action='store_true',
                        help='Recompute TSS (Method 3) and TED before evaluation')
    parser.add_argument('--input', type=str, default=None,
                        help='Input metrics file (for --evaluate mode)')
    parser.add_argument('--suffix', type=str, default='_original',
                        help='Output suffix: _original or _reproduce')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Only generate tables/figures (skip recompute)')
    args = parser.parse_args()

    if args.recompute:
        # Step 1: Recompute metrics → _reproduce.json
        recompute_metrics(ANNOTATION_ORIGINAL, ANNOTATION_REPRODUCE)

        # Step 2: Generate tables/figures from reproduced metrics
        run_evaluation(ANNOTATION_REPRODUCE, '_reproduce')

    elif args.evaluate_only:
        input_file = args.input or ANNOTATION_ORIGINAL
        run_evaluation(input_file, args.suffix)

    else:
        # Default: generate from original metrics with given suffix
        input_file = args.input or ANNOTATION_ORIGINAL
        run_evaluation(input_file, args.suffix)


if __name__ == "__main__":
    main()
