"""
Recompute TSS (Method 3) and TED for the manual annotation experiment data.

TSS Method 3: Precomputed test suite — sample perturbation bindings from golden,
reuse across constructed queries for evaluation.
TED: minimum tree edit distance with fixed _break_wdt_wikidata + canonicalization.

Usage:
  conda run -n KBQA_research python scripts/recompute_annotation_metrics.py
"""

import sys, os, json, re, math, time, random, logging

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

# Bypass heavy ML imports
_dummy_st = type(sys)("dummy")
_dummy_st.__file__ = "dummy"
for _mod in ["sentence_transformers", "sentence_transformers.util", "sentence_transformers.backend",
             "sklearn", "sklearn.metrics", "scipy", "scipy.sparse"]:
    sys.modules[_mod] = type(sys)("dummy")

from src.core.common import (
    DATASET, ODBC_CONFIG_WIKIDATA_2023, SPARQL_wrapper_path_wikidata_2023,
    WIKIDATA_PREFIX_LIST, DEFAULT_SPARQL_SERVICE,
)
from src.core.utils import load_json, dump_json, setup_custom_logger, get_PRF1
from src.sparql.executor import SparqlOdbcQuerierNoSexprWikidata
from src.sparql.sparql_utils import SyntaxTreeEditor
from src.metrics.minimum_TSS import (
    _generate_test_suite_for_golden, _get_sparql_prefix,
    TARGET_VARIABLE_NAME, TEST_SUITE_GEN_LIMIT,
)
from src.metrics.minimum_TED import minimum_ted

# ── Config ──
DATA_DIR = os.path.join(_project_root, "data", "output", "manual_annotation_experiment", "annotation_analysis")
INPUT_FILE = os.path.join(DATA_DIR, "annotation_result_with_metric_original.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "annotation_result_with_metric_reproduce.json")
LOG_FILE = os.path.join(DATA_DIR, "recompute_metrics.log")
RANDOM_SEED = 374


def compute_tss(data: list, logger: logging.Logger) -> dict:
    """Compute TSS Method 3 for all entries. Returns {(idx, qid): tss_value}."""
    logger.info("=" * 60)
    logger.info("Phase 1: TSS Method 3 (Precomputed Test Suite)")
    logger.info("=" * 60)

    # Connect to SPARQL endpoint
    sparql_querier = SparqlOdbcQuerierNoSexprWikidata(
        sparql_wrapper_path=SPARQL_wrapper_path_wikidata_2023,
        logger=logger, service=DEFAULT_SPARQL_SERVICE,
        odbc_config=ODBC_CONFIG_WIKIDATA_2023, timeout=300,
    )
    dataset = DATASET.LC2
    method = DATASET.SIMULATED_WIKIDATA

    # Build golden→entries mapping
    golden_to_indices = {}
    skip_indices = set()

    for idx, entry in enumerate(data):
        constructed = entry.get('simulated_LF', {}).get('sparql', '')
        golden = entry.get('golden_sparql_query', '')

        if not constructed or not constructed.strip():
            skip_indices.add(idx)
            continue
        if not golden or not golden.strip():
            skip_indices.add(idx)
            continue

        if golden not in golden_to_indices:
            golden_to_indices[golden] = []
        golden_to_indices[golden].append(idx)

    logger.info(f"Entries: {len(data)}, Skip: {len(skip_indices)}, "
                f"Unique golden SPARQLs: {len(golden_to_indices)}")

    # Step 1: Generate test suites per unique golden
    test_suite_cache = {}
    golden_errors = 0

    for i, (golden_sparql, indices) in enumerate(golden_to_indices.items()):
        if (i + 1) % 50 == 0:
            logger.info(f"  Test suite progress: {i+1}/{len(golden_to_indices)}")

        try:
            golden_item = {
                "qid": f"g{i}",
                "golden_sparql_query": golden_sparql,
                "golden_grounded_items": [
                    {"mid": m.group(0)}
                    for m in re.finditer(r'wd:Q\d+', golden_sparql)
                ],
            }
            ts_examples = _generate_test_suite_for_golden(
                golden_item, dataset, sparql_querier, logger,
                test_cases_limit=TEST_SUITE_GEN_LIMIT,
            )
            test_suite_cache[golden_sparql] = ts_examples
        except Exception as e:
            logger.error(f"  Test suite error for golden {i}: {e}")
            test_suite_cache[golden_sparql] = []
            golden_errors += 1

    logger.info(f"Test suites generated: {len(test_suite_cache)} ({golden_errors} errors)")

    # Step 2: Evaluate each constructed query
    tss_results = {}
    eval_count, eval_errors = 0, 0
    prefix = _get_sparql_prefix(dataset)

    for idx, entry in enumerate(data):
        if idx in skip_indices:
            tss_results[idx] = 0.0
            continue

        golden = entry.get('golden_sparql_query', '')
        constructed = entry.get('simulated_LF', {}).get('sparql', '')
        ts_examples = test_suite_cache.get(golden, [])

        if not ts_examples:
            tss_results[idx] = 0.0
            continue

        # Add prefix if needed
        if not constructed.strip().lower().startswith('prefix'):
            full_sparql = prefix + "\n" + constructed
        else:
            full_sparql = constructed

        try:
            editor = SyntaxTreeEditor(full_sparql, method, logger)
            editor.preprocess_test_suite(f"?{TARGET_VARIABLE_NAME}")
            processed = editor.sparql_txt

            f1_list = []
            for test_case in ts_examples:
                test_editor = SyntaxTreeEditor(processed, method, logger)
                overlap = False
                for before, after in test_case["binding"].items():
                    if test_editor.update_leaf_value(before, after):
                        overlap = True
                if not overlap:
                    continue

                try:
                    predicted = sparql_querier.get_execution_result_one_variable_sparql_wrapper(
                        test_editor.sparql_txt, retry=1)
                except Exception:
                    continue

                gold = test_case["answer"]
                _, _, f1 = get_PRF1(predicted, gold)
                f1_list.append(f1)

            tss_results[idx] = sum(f1_list) / len(f1_list) if f1_list else 0.0
            eval_count += 1

        except Exception as e:
            logger.warning(f"  [{idx}] TSS eval error: {e}")
            tss_results[idx] = 0.0
            eval_errors += 1

        if eval_count % 100 == 0:
            logger.info(f"  TSS eval progress: {eval_count}")

    logger.info(f"TSS done: {eval_count} evaluated, {eval_errors} errors")
    return tss_results


def compute_ted(data: list, logger: logging.Logger) -> dict:
    """Compute TED for all entries. Returns {(idx, qid): ted_value}."""
    logger.info("=" * 60)
    logger.info("Phase 2: TED (Tree Edit Distance)")
    logger.info("=" * 60)

    ted_results = {}
    errors = 0

    for idx, entry in enumerate(data):
        golden_raw = entry.get('golden_sparql_query', '')
        pred_raw = entry.get('simulated_LF', {}).get('sparql', '')

        if not pred_raw or not pred_raw.strip():
            ted_results[idx] = 1.0
            continue
        if re.match(r'\s*SELECT\s+\*', pred_raw, re.IGNORECASE):
            ted_results[idx] = 1.0
            continue
        if not golden_raw or not golden_raw.strip():
            ted_results[idx] = 1.0
            continue

        golden_with_prefix = WIKIDATA_PREFIX_LIST + "\n" + golden_raw
        pred_with_prefix = WIKIDATA_PREFIX_LIST + "\n" + pred_raw

        try:
            ted = minimum_ted(
                golden_with_prefix, pred_with_prefix,
                DATASET.LC2, DATASET.SIMULATED_WIKIDATA,
                logger=None,
            )
            ted_results[idx] = ted
        except Exception as e:
            logger.warning(f"  [{idx}] TED error: {e}")
            ted_results[idx] = 1.0
            errors += 1

        if (idx + 1) % 200 == 0:
            logger.info(f"  TED progress: {idx+1}/{len(data)}")

    logger.info(f"TED done: {len(ted_results)} results, {errors} errors")
    return ted_results


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    logger = setup_custom_logger(LOG_FILE)
    random.seed(RANDOM_SEED)

    t_start = time.time()
    logger.info(f"Loading: {INPUT_FILE}")
    data = load_json(INPUT_FILE)
    logger.info(f"Total entries: {len(data)}")

    # ── Phase 1: TSS Method 3 ──
    try:
        tss_results = compute_tss(data, logger)
        tss_ok = True
    except Exception as e:
        logger.error(f"TSS computation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        tss_results = {}
        tss_ok = False

    # ── Phase 2: TED ──
    try:
        ted_results = compute_ted(data, logger)
        ted_ok = True
    except Exception as e:
        logger.error(f"TED computation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        ted_results = {}
        ted_ok = False

    # ── Merge results ──
    for idx, entry in enumerate(data):
        if tss_ok and idx in tss_results:
            entry['test_suite'] = tss_results[idx]
        if ted_ok and idx in ted_results:
            entry['TED'] = ted_results[idx]

    # ── Save ──
    dump_json(data, OUTPUT_FILE)
    t_end = time.time()

    # ── Summary ──
    tss_vals = [e['test_suite'] for e in data if isinstance(e.get('test_suite'), (int, float))]
    ted_vals = [e['TED'] for e in data if isinstance(e.get('TED'), (int, float))]

    logger.info("=" * 60)
    logger.info(f"Output saved to: {OUTPUT_FILE}")
    logger.info(f"Total time: {t_end - t_start:.1f}s ({(t_end - t_start)/60:.1f} min)")
    if tss_vals:
        logger.info(f"TSS: min={min(tss_vals):.4f}, max={max(tss_vals):.4f}, "
                    f"mean={sum(tss_vals)/len(tss_vals):.4f}, "
                    f"==0: {sum(1 for v in tss_vals if v==0)}, "
                    f"==1: {sum(1 for v in tss_vals if v==1)}")
    if ted_vals:
        logger.info(f"TED: min={min(ted_vals):.4f}, max={max(ted_vals):.4f}, "
                    f"mean={sum(ted_vals)/len(ted_vals):.4f}, "
                    f"==1: {sum(1 for v in ted_vals if v==1)}")
    logger.info("Done!")

    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Log: {LOG_FILE}")


if __name__ == "__main__":
    main()
