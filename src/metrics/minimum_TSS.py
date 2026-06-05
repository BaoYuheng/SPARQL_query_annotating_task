"""
Minimum TSS Computation Module (Pure Algorithm Layer)
=====================================================
Contains only core TSS algorithms — no file I/O, no SPARQL querier construction.

Three TSS Methods
-----------------
  Method 1 (Paper):  new_test_suite_similarity       — ALL-at-once, single KB query, deterministic
  Method 2 (S1+S2): new_quad_execute_test_suite      — Two-stage with reground + answer consistency
  Method 3 (Precom): execute_precomputed_test_suite   — Requires pre-generated test_suite_examples

Test Suite Generation
---------------------
  _generate_test_suite_for_golden — Per-question test suite generation (pure algorithm)
  generate_test_suites_batch      — Batch in-memory wrapper (no file I/O)

  Each test case has a "binding" dict {tree_string → value} serving as a primary
  key, making test cases reusable across different constructed queries.

File-level orchestration (file I/O, querier construction) is in scripts/experiment_runner.py.
Directory-level batch evaluation is in scripts/annotation_performance_evaluation.py
and scripts/manual_annotation_evaluation.py.

Reproducibility
---------------
  Method 1 is fully deterministic (no random ops, no sampling).
  Method 2 uses random.shuffle for S2; seed is fixed at the entry point.
  Method 3 test suite generation uses random.shuffle with fixed seed (TSS_RANDOM_SEED).
"""

import re
import math
import copy
import random
import itertools
from collections import defaultdict
from typing import Union
from tqdm import tqdm

# ── src imports (relative, works when imported from within the package) ──
from ..core.common import (
    DATASET, Dataset,
    FreebaseConstantForConstruction, FREEBASE_CONSTANT_TYPE,
    WikidataConstantForConstruction, WIKIDATA_CONSTANT_TYPE,
    get_syntax_tree_string, get_syntax_tree_value,
    NS_PREFIX, WIKIDATA_PREFIX_LIST,
    to_dataset, to_kb_type, KB_TYPE, FB_DATASETS, WD_DATASETS,
    TSS_QUERY_MAX_ROWS, TSS_TEST_SUITE_LIMIT, TSS_TEST_SUITE_GEN_LIMIT, TSS_RANDOM_SEED,
)
from ..core.utils import get_PRF1
from ..sparql.executor import SparqlOdbcQuerierNoSexpr, SparqlOdbcQuerierNoSexprWikidata
from ..sparql.sparql_utils import SyntaxTreeEditor, TreeConstructor, TreeEditDistance


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

REPLACED_VARIABLE_NAME = "replace"
TARGET_VARIABLE_NAME = "target"
# TSS_QUERY_MAX_ROWS, TSS_TEST_SUITE_LIMIT, TSS_TEST_SUITE_GEN_LIMIT, TSS_RANDOM_SEED
#   imported from ..core.common — see above.
TEST_SUITE_GEN_LIMIT = TSS_TEST_SUITE_GEN_LIMIT  # backward-compatible alias

WIKIDATA_PREFIX_BLOCK = "\n".join([
    "PREFIX p: <http://www.wikidata.org/prop/>",
    "PREFIX pq: <http://www.wikidata.org/prop/qualifier/>",
    "PREFIX ps: <http://www.wikidata.org/prop/statement/>",
    "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>",
    "PREFIX wd: <http://www.wikidata.org/entity/>",
    "PREFIX wdt: <http://www.wikidata.org/prop/direct/>",
    "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>",
])


# ═══════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════

def jaccard_similarity(set1, set2):
    """Jaccard similarity between two sets."""
    if len(set1) == 0 or len(set2) == 0:
        return 0.0
    set1 = set(set1)
    set2 = set(set2)
    return float(len(set1.intersection(set2)) / len(set1.union(set2)))


def _modified_wikidata_sparql(sparql: str) -> str:
    """
    Simplify Wikidata SPARQL: p:Pxx/ps:Pxx → wdt:Pxx when both PIDs match.
    当 p:/ps: 路径两端的 PID 相同时，简化为 wdt: 直接属性。
    """
    pattern = r"p:P\d+/ps:P\d+"
    props = re.findall(pattern, sparql)
    for p in props:
        pids = re.findall(r'\d+', p)
        if pids and pids[0] == pids[1]:
            sparql = sparql.replace(p, "wdt:P" + pids[0])
    return sparql


def _get_sparql_prefix(dataset: DATASET) -> str:
    """Return the appropriate SPARQL prefix block for a dataset."""
    if to_dataset(dataset) in FB_DATASETS:
        return "PREFIX ns: <http://rdf.freebase.com/ns/> "
    elif to_dataset(dataset) in WD_DATASETS or dataset == DATASET.SIMULATED_WIKIDATA:
        return WIKIDATA_PREFIX_LIST
    else:
        raise NotImplementedError(f"dataset: {dataset}")


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Overlap Detection  (used by Method 1)
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_replaceable_items(editor: SyntaxTreeEditor, replaceable_items: list,
                               dataset: DATASET, logger) -> dict:
    """
    Detect which golden replaceable items also appear in the simulated editor's
    syntax tree. Returns a reground map: syntax-tree-string → [candidate values].

    For entities:  map to a variable  (?replaceN)
    For quantities: map to [self]     (original value only in paper version)
    For times:     map to [self]      (no meaningful perturbation)
    """
    ds = to_dataset(dataset)  # normalize DATASET → Dataset
    reground_map = {}
    for idx, item in enumerate(replaceable_items):
        if ds in FB_DATASETS:
            _type = FreebaseConstantForConstruction.get_constant_type(item)
            if _type is FREEBASE_CONSTANT_TYPE.QUANTITY:
                old_value = item.split('^^')[0][1:-1]
                reground_map[get_syntax_tree_string(item, ds)] = [f'"{old_value}"']
            elif _type is FREEBASE_CONSTANT_TYPE.TIME:
                reground_map[get_syntax_tree_string(item, ds)] = [get_syntax_tree_value(item, ds)]
            else:  # entity / class → variable
                tree_str = get_syntax_tree_string(item, ds)
                reground_map[tree_str] = [f"?{REPLACED_VARIABLE_NAME}{idx}"]

        elif ds in WD_DATASETS:
            _type = WikidataConstantForConstruction.get_constant_type(item)
            if _type is WIKIDATA_CONSTANT_TYPE.QUANTITY:
                old_value = item.split('^^')[0][1:-1]
                reground_map[get_syntax_tree_string(item, ds)] = [f'"{old_value}"']
            elif _type is WIKIDATA_CONSTANT_TYPE.TIME:
                reground_map[get_syntax_tree_string(item, ds)] = [get_syntax_tree_value(item, ds)]
            else:
                tree_str = get_syntax_tree_string(item, ds)
                reground_map[tree_str] = [f"?{REPLACED_VARIABLE_NAME}{idx}"]
        else:
            raise NotImplementedError(f"dataset: {dataset}")

    # Filter: keep only items whose syntax tree string exists in the simulated editor
    filtered = {}
    for key in reground_map:
        if editor.is_consist_node(key):
            filtered[key] = reground_map[key]
    return filtered


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: ALL-at-once Perturbation Execution  (used by Method 1)
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_perturbation_and_execute(editor: SyntaxTreeEditor,
                                    intersection_items: set,
                                    reground_map: dict,
                                    sparql_querier: SparqlOdbcQuerierNoSexpr,
                                    dataset: DATASET, logger) -> dict:
    """
    Paper Implementation: replace ALL overlapping entities as projected variables
    simultaneously, then execute a single SPARQL query to get the complete
    perturbation → answer mapping.

    Returns: answer_map = {"replace_val_1,replace_val_2,...": [target_answer_1, ...]}
    """
    for item_key in intersection_items:
        value = reground_map[item_key][0]
        if value.startswith('?'):  # entity → variable
            if not editor.replace_leaf_with_variable(item_key, value):
                raise ValueError(f"key: {item_key} not found in syntax tree")
        else:  # literal → value
            if not editor.update_leaf_value(item_key, value):
                raise ValueError(f"key: {item_key} not found in syntax tree")

    sparql_txt = editor.sparql_txt
    # Add row limit to prevent unbounded ODBC fetches; matches Virtuoso MaxReturnRows.
    # 加行数限制防止 ODBC 无限 fetch；与 Virtuoso HTTP 端点的 MaxReturnRows 对齐。
    if 'LIMIT' not in sparql_txt.upper():
        sparql_txt = sparql_txt.rstrip() + f" LIMIT {TSS_QUERY_MAX_ROWS}"
    logger.info(sparql_txt)
    rows = sparql_querier.execute_query(sparql_txt, retry=1)

    ds = to_dataset(dataset)  # normalize DATASET → Dataset
    answer_map = defaultdict(list)
    for row in rows:
        try:
            replace_map = {}
            for variable in row.keys():
                if variable.startswith(REPLACED_VARIABLE_NAME):
                    var_idx = variable.replace(REPLACED_VARIABLE_NAME, '')
                    if ds in FB_DATASETS:
                        replace_map[var_idx] = get_syntax_tree_value(
                            FreebaseConstantForConstruction(
                                row[variable]['type'], row[variable]['value'],
                                row[variable].get('datatype', None), row[variable].get('xml:lang', None)
                            ).__repr__(), ds)
                    elif ds in WD_DATASETS:
                        replace_map[var_idx] = get_syntax_tree_value(
                            WikidataConstantForConstruction(
                                row[variable]['type'], row[variable]['value'],
                                row[variable].get('datatype', None), row[variable].get('xml:lang', None)
                            ).__repr__(), ds)
                elif variable == TARGET_VARIABLE_NAME:
                    if ds in FB_DATASETS:
                        answer = get_syntax_tree_value(
                            FreebaseConstantForConstruction(
                                row[variable]['type'], row[variable]['value'],
                                row[variable].get('datatype', None), row[variable].get('xml:lang', None)
                            ).__repr__(), ds)
                    elif ds in WD_DATASETS:
                        answer = get_syntax_tree_value(
                            WikidataConstantForConstruction(
                                row[variable]['type'], row[variable]['value'],
                                row[variable].get('datatype', None), row[variable].get('xml:lang', None)
                            ).__repr__(), ds)
            replace_result = [replace_map[key] for key in sorted(replace_map.keys(), key=int)]
            answer_map[','.join(replace_result)].append(answer)
        except Exception:
            continue
    return answer_map


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: KB Regrounding for a Single Item  (used by Method 2)
# ═══════════════════════════════════════════════════════════════════════════════

def _reground_single_item(sparql_txt: str, replaceable_items: list,
                          sparql_querier: SparqlOdbcQuerierNoSexpr,
                          dataset: DATASET, logger,
                          items_are_tree_strings: bool = False) -> dict:
    """
    Execute the abstracted query for each overlapping item to find its actual KB
    reground values. Returns reground_map: item_key → [original, value1, value2, ...]

    Unlike _detect_replaceable_items (which only checks syntax-tree existence),
    this actually queries the KB to discover viable perturbations.

    Parameters
    ----------
    items_are_tree_strings : bool
        If True, items are already syntax tree strings (keys from golden reground_map).
        If False, items are raw mids (from generate_test_suite_data).
    """
    ds = to_dataset(dataset)  # normalize DATASET → Dataset
    reground_map = {}

    for idx, item in enumerate(replaceable_items):
        if items_are_tree_strings:
            # Items already in syntax tree string form — use directly as key.
            # All treated as variable replacements for KB reground discovery.
            tree_str = item
            reground_map[tree_str] = [f"?{REPLACED_VARIABLE_NAME}{idx}"]

        elif ds in FB_DATASETS:
            _type = FreebaseConstantForConstruction.get_constant_type(item)
            if _type is FREEBASE_CONSTANT_TYPE.QUANTITY:
                old_value = item.split('^^')[0][1:-1]
                suffix = item.split('^^')[1]
                if suffix == "<http://www.w3.org/2001/XMLSchema#integer>":
                    reground_map[get_syntax_tree_string(item, ds)] = [
                        f'"{old_value}"', f'"{str(int(old_value) * 2)}"',
                        f'"{str(int(old_value) * 3)}"']
                else:
                    reground_map[get_syntax_tree_string(item, ds)] = [
                        f'"{old_value}"', f'"{str(float(old_value) * 2.0)}"',
                        f'"{str(float(old_value) * 3.0)}"']
            elif _type is FREEBASE_CONSTANT_TYPE.TIME:
                reground_map[get_syntax_tree_string(item, ds)] = [
                    get_syntax_tree_value(item, ds)]
            else:
                tree_str = get_syntax_tree_string(item, ds)
                reground_map[tree_str] = [f"?{REPLACED_VARIABLE_NAME}{idx}"]

        elif ds in WD_DATASETS:
            _type = WikidataConstantForConstruction.get_constant_type(item)
            if _type is WIKIDATA_CONSTANT_TYPE.QUANTITY:
                old_value = item.split('^^')[0][1:-1]
                suffix = item.split('^^')[1]
                if suffix == "<http://www.w3.org/2001/XMLSchema#integer>":
                    reground_map[get_syntax_tree_string(item, ds)] = [
                        f'"{old_value}"', f'"{str(int(old_value) * 2)}"',
                        f'"{str(int(old_value) * 3)}"']
                else:
                    reground_map[get_syntax_tree_string(item, ds)] = [
                        f'"{old_value}"', f'"{str(float(old_value) * 2.0)}"',
                        f'"{str(float(old_value) * 3.0)}"']
            elif _type is WIKIDATA_CONSTANT_TYPE.TIME:
                reground_map[get_syntax_tree_string(item, ds)] = [
                    get_syntax_tree_value(item, ds)]
            else:
                tree_str = get_syntax_tree_string(item, ds)
                reground_map[tree_str] = [f"?{REPLACED_VARIABLE_NAME}{idx}"]
        else:
            raise NotImplementedError(f"dataset: {dataset}")

    # Execute abstracted query for each item to find KB reground values
    for item_key in reground_map:
        editor = SyntaxTreeEditor(sparql_txt, dataset, logger)
        editor.remove_projection_variable(f"?{TARGET_VARIABLE_NAME}")
        value = reground_map[item_key][0]

        if value.startswith('?'):  # entity → variable
            if not editor.replace_leaf_with_variable(item_key, value):
                raise ValueError(f"key: {item_key} not found in syntax tree")
            reground_map[item_key] = [item_key]  # keep original as fallback
            # Execute to discover additional values from KB
            query_txt = editor.sparql_txt
            if 'LIMIT' not in query_txt.upper():
                query_txt = query_txt.rstrip() + f" LIMIT {TSS_QUERY_MAX_ROWS}"
            rows = sparql_querier.execute_query(query_txt, retry=1)
            for row in rows:
                try:
                    if ds in FB_DATASETS:
                        reground_map[item_key].append(get_syntax_tree_value(
                            FreebaseConstantForConstruction(
                                row[value[1:]]['type'], row[value[1:]]['value'],
                                row[value[1:]].get('datatype', None), row[value[1:]].get('xml:lang', None)
                            ).__repr__(), ds))
                    elif ds in WD_DATASETS:
                        reground_map[item_key].append(get_syntax_tree_value(
                            WikidataConstantForConstruction(
                                row[value[1:]]['type'], row[value[1:]]['value'],
                                row[value[1:]].get('datatype', None), row[value[1:]].get('xml:lang', None)
                            ).__repr__(), ds))
                except Exception:
                    pass
        else:
            # For literal types, use the pre-defined values directly
            if not editor.update_leaf_value(item_key, value):
                raise ValueError(f"key: {item_key} not found in syntax tree")

    return reground_map


# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Paper TSS  (ALL-at-once, deterministic, single KB query)
# ═══════════════════════════════════════════════════════════════════════════════

def new_test_suite_similarity(
    sparql_item_golden: dict,           # has: golden_sparql_query, golden_grounded_items
    sparql_item_constructed: dict,       # has: simulated_sparql_query
    dataset: DATASET,
    method: DATASET,                     # dataset type of the constructed query (e.g. SIMULATED_FREEBASE)
    sparql_querier: SparqlOdbcQuerierNoSexpr,
    logger,
) -> tuple:
    """
    Paper Sec 3.2: Test Suite Score (TSS).

    Algorithm:
      1. Identify overlapping entities E_O between golden and constructed query.
      2. Abstract all E_O as projection variables simultaneously.
      3. Execute ONE SPARQL query to get the complete perturbation → answer mapping.
      4. Compute average Jaccard similarity across all perturbations.

    Returns:
      (tss_score, num_perturbations)
      tss_score ∈ [0, 1]; num_perturbations = |T| (test suite size)
    """
    assert str(sparql_item_golden['qid']) == str(sparql_item_constructed['qid']), \
        f"qid mismatch: {sparql_item_golden['qid']} vs {sparql_item_constructed['qid']}"

    logger.info(f"qid: {sparql_item_golden['qid']}")

    original_sparql = sparql_item_golden["golden_sparql_query"]

    # NOTE: _modified_wikidata_sparql disabled — p:P/ps:P → wdt:P
    # simplification causes semantic drift for multi-valued entity properties.
    # if dataset == DATASET.LC2:
    #     original_sparql = _modified_wikidata_sparql(original_sparql)

    # Add PREFIX declarations for Wikidata golden SPARQL so the parser expands
    # wd:/wdt:/p:/ps:/pq: to full IRIs, matching the simulated SPARQL format.
    if to_dataset(dataset) in WD_DATASETS:
        original_sparql = WIKIDATA_PREFIX_LIST + '\n' + original_sparql

    replaceable_items = [item['mid'] for item in sparql_item_golden["golden_grounded_items"]]

    # Dataset-specific fixes
    if dataset is DATASET.GRAIL:
        for idx, item in enumerate(replaceable_items):
            if any(item.endswith(t) for t in ['gYear>', 'gYearMonth>', 'date>']):
                replaceable_items[idx] = (
                    f'"{item.split("^^")[0][1:-1] + "-08:00"}"^^{item.split("^^")[1]}')

    try:
        golden_editor = SyntaxTreeEditor(original_sparql, dataset, logger)
        golden_editor.preprocess_test_suite(f"?{TARGET_VARIABLE_NAME}")

        simulated_sparql = sparql_item_constructed["simulated_sparql_query"]
        # NOTE: _modified_wikidata_sparql disabled
        # if dataset == DATASET.LC2:
        #     simulated_sparql = _modified_wikidata_sparql(simulated_sparql)

        simulated_editor = SyntaxTreeEditor(simulated_sparql, method, logger)
        simulated_editor.preprocess_test_suite(f"?{TARGET_VARIABLE_NAME}")

        logger.info(simulated_editor.sparql_txt)
        logger.info(golden_editor.sparql_txt)

        # Exact match shortcut
        if simulated_editor.sparql_txt == golden_editor.sparql_txt:
            return 1.0, -1

        if len(replaceable_items) == 0:
            return 0.0, -1

        # Step 1: identify overlapping entities
        golden_reground = _detect_replaceable_items(golden_editor, replaceable_items, dataset, logger)
        simulated_reground = _detect_replaceable_items(simulated_editor, replaceable_items, dataset, logger)
        intersection_items = set(golden_reground.keys()) & set(simulated_reground.keys())

        if len(intersection_items) == 0:
            return 0.0, -1

        logger.info(f"intersection_items: {intersection_items}")

        # Step 2 & 3: ALL-at-once perturbation + single query execution
        golden_result = _apply_perturbation_and_execute(
            golden_editor, intersection_items, golden_reground, sparql_querier, dataset, logger)
        logger.info(f"golden_result: {len(golden_result)}")

        simu_result = _apply_perturbation_and_execute(
            simulated_editor, intersection_items, golden_reground, sparql_querier, dataset, logger)
        logger.info(f"simu_result: {len(simu_result)}")

        # Step 4: compute average Jaccard
        all_replace = set(golden_result.keys()) | set(simu_result.keys())
        logger.info(f"all_replace: {len(all_replace)}")
        similarity_list = []
        for replace_key in all_replace:
            golden_answers = golden_result.get(replace_key, [])
            simu_answers = simu_result.get(replace_key, [])
            similarity_list.append(jaccard_similarity(golden_answers, simu_answers))

        num_exact = sum(1 for s in similarity_list if s == 1.0)
        logger.info(f"exact_matches: {num_exact}/{len(similarity_list)}")

        return sum(similarity_list) / len(similarity_list), len(all_replace)

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"qid: {sparql_item_constructed['qid']}; error: {e}")
        return 0.0, -1


# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: S1+S2 TSS  (reground consistency × answer consistency)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_test_suite_data(golden_item: dict, dataset: DATASET,
                             sparql_querier: SparqlOdbcQuerierNoSexpr, logger) -> tuple:
    """
    Pre-generate reground_map and processed golden SPARQL for one question.
    This is the data preparation step required BEFORE running Method 2.

    Returns: (reground_map, processed_golden_sparql)
    """
    original_sparql = golden_item["golden_sparql_query"]
    replaceable_items = [item['mid'] for item in golden_item["golden_grounded_items"]]

    if dataset is DATASET.GRAIL:
        for idx, item in enumerate(replaceable_items):
            if any(item.endswith(t) for t in ['gYear>', 'gYearMonth>', 'date>']):
                replaceable_items[idx] = (
                    f'"{item.split("^^")[0][1:-1] + "-08:00"}"^^{item.split("^^")[1]}')

    editor = SyntaxTreeEditor(original_sparql, dataset, logger)
    editor.preprocess_test_suite(f"?{TARGET_VARIABLE_NAME}")
    sparql_processed = editor.sparql_txt

    reground_map = _reground_single_item(sparql_processed, replaceable_items,
                                         sparql_querier, dataset, logger)
    return reground_map, sparql_processed


def new_quad_execute_test_suite(
    golden_data: dict,                    # golden linking data, keyed by qid, with reground_map and golden_sparql_process
    constructed_data: dict,               # method output, keyed by qid, with search_results
    dataset: DATASET,
    method: DATASET,
    sparql_querier: SparqlOdbcQuerierNoSexpr,
    logger,
    test_suite_limit: int = 200,
    random_seed: int = 374,
) -> dict:
    """
    S1+S2 TSS: Decompose TSS into reground consistency (S1) and answer
    consistency (S2). More expensive but provides finer-grained analysis.

    S1: For each overlapping item, compare reground value sets via Jaccard.
    S2: Cartesian-product combinations of reground values → sample → compare answers.

    Returns the constructed_data dict with 'test_suite_s1', 'test_suite_s2',
    and 'test_suite_similarity' (= S1 * S2) added per item.
    """
    random.seed(random_seed)

    qid_to_golden = {str(item["qid"]): item for item in golden_data.values()}
    f1_list = []

    for qid, detection_item in constructed_data.items():
        if 'qid' not in detection_item:
            continue
        if str(detection_item['qid']) not in qid_to_golden:
            detection_item["test_suite_similarity"] = 0.0
            f1_list.append(0.0)
            continue
        if not detection_item.get('search_results'):
            detection_item["test_suite_similarity"] = 0.0
            f1_list.append(0.0)
            continue

        try:
            src_item = qid_to_golden[str(detection_item['qid'])]
            prefix = _get_sparql_prefix(dataset)

            # Determine which SPARQL to use
            if "gpt_select_top1" in detection_item and "sparql" in detection_item.get('gpt_select_top1', {}):
                simulated_sparql = prefix + "\n" + detection_item['gpt_select_top1']['sparql']
            elif 'sparql' in detection_item["search_results"][0]:
                simulated_sparql = prefix + "\n" + detection_item["search_results"][0]["sparql"]
            else:
                simulated_sparql = prefix + "\n" + detection_item["search_results"][0]

            detection_item['simulated_sparql_query'] = simulated_sparql

            editor = SyntaxTreeEditor(simulated_sparql, method, logger)
            editor.preprocess_test_suite(f"?{TARGET_VARIABLE_NAME}")
            simulated_sparql_processed = editor.sparql_txt

            # Check overlap: which golden items exist in the simulated query?
            overlap_items = []
            for golden_key in src_item.get('reground_map', {}).keys():
                test_editor = SyntaxTreeEditor(simulated_sparql_processed, method, logger)
                if test_editor.update_leaf_value(golden_key, src_item['reground_map'][golden_key][0]):
                    overlap_items.append(golden_key)

            if len(overlap_items) == 0:
                detection_item["test_suite_similarity"] = 0.0
                f1_list.append(0.0)
                continue

            # Reground simulated query against KB
            # overlap_items are syntax tree strings from golden reground_map
            simulated_reground = _reground_single_item(
                simulated_sparql_processed, overlap_items, sparql_querier, dataset, logger,
                items_are_tree_strings=True
            )

            # ── S1: per-item reground consistency ──
            s1_list = []
            overlap_intersection = {}
            for k in simulated_reground:
                simu_set = set(simulated_reground[k])
                golden_set = set(src_item['reground_map'].get(k, []))
                s1_list.append(jaccard_similarity(simu_set, golden_set))
                overlap_intersection[k] = simu_set & golden_set

            detection_item["test_suite_s1"] = {"s1": sum(s1_list) / len(s1_list)}

            # ── S2: Cartesian-product answer consistency ──
            key_list = list(overlap_intersection.keys())
            value_lists = [list(overlap_intersection[k]) for k in key_list]
            combinations = list(itertools.product(*value_lists))
            random.shuffle(combinations)

            golden_sparql_processed = src_item.get("golden_sparql_process",
                                                    src_item.get("golden_sparql_query", ""))

            s2_list = []
            for combo in combinations[:test_suite_limit]:
                simu_editor = SyntaxTreeEditor(simulated_sparql_processed, method, logger)
                golden_editor = SyntaxTreeEditor(golden_sparql_processed, dataset, logger)

                for key, value in zip(key_list, combo):
                    golden_editor.update_leaf_value(key, value)
                    simu_editor.update_leaf_value(key, value)

                simu_answer = sparql_querier.get_execution_result_one_variable_sparql_wrapper(
                    simu_editor.sparql_txt, retry=1)
                golden_answer = sparql_querier.get_execution_result_one_variable_sparql_wrapper(
                    golden_editor.sparql_txt, retry=1)

                if len(golden_answer) == 0 and len(simu_answer) == 0:
                    s2_list.append(1.0)
                else:
                    s2_list.append(jaccard_similarity(simu_answer, golden_answer))

            detection_item["test_suite_s2"] = {"s2": sum(s2_list) / len(s2_list)}
            tss = (sum(s1_list) / len(s1_list)) * (sum(s2_list) / len(s2_list))
            detection_item['test_suite_similarity'] = tss
            f1_list.append(tss)

        except Exception as e:
            logger.error(f"qid: {detection_item.get('qid', '?')}; error: {e}")
            detection_item["test_suite_similarity"] = 0.0
            f1_list.append(0.0)

    constructed_data["test_suite_s"] = {"average": sum(f1_list) / len(f1_list)}
    return constructed_data


# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: Precomputed Test Suite  (requires pre-generated test_suite_examples)
# ═══════════════════════════════════════════════════════════════════════════════

def execute_precomputed_test_suite(
    constructed_data: dict,               # method output, keyed by qid or {'results': [...]}
    test_suite_data: list,                # pre-generated test_suite_examples (list of dicts keyed by qid)
    dataset: DATASET,
    method: DATASET,
    sparql_querier: SparqlOdbcQuerierNoSexpr,
    logger,
) -> dict:
    """
    Evaluate using pre-generated test suite examples.
    Requires test_suite_data where each item has:
      - test_suite_examples: list of {original_sparql, binding, answer}

    For each constructed query:
      1. Check which test cases have overlapping items.
      2. Apply the test case's binding (perturbation) to the constructed query.
      3. Execute and compare answers using P/R/F1.
      4. Average across overlapping test cases.
    """
    # Normalize constructed_data format
    if 'results' in constructed_data:
        items_iter = constructed_data['results']
    elif isinstance(constructed_data, dict):
        items_iter = constructed_data.values()
    else:
        items_iter = constructed_data

    qid_to_test_suite = {}
    for item in test_suite_data:
        qid_to_test_suite[str(item["qid"])] = item

    f1_list = []
    for detection_item in items_iter:
        if 'qid' not in detection_item:
            continue
        qid = str(detection_item['qid'])
        try:
            # Determine the simulated SPARQL
            prefix = _get_sparql_prefix(dataset)
            if "gpt_select_top1" in detection_item and "sparql" in detection_item.get('gpt_select_top1', {}):
                simulated_sparql = prefix + "\n" + detection_item['gpt_select_top1']['sparql']
            elif 'simulated_sparql_query' in detection_item:
                simulated_sparql = detection_item['simulated_sparql_query']
                if not simulated_sparql.strip().lower().startswith('prefix'):
                    simulated_sparql = prefix + "\n" + simulated_sparql
            elif detection_item.get('search_results'):
                if 'sparql' in detection_item['search_results'][0]:
                    simulated_sparql = prefix + "\n" + detection_item["search_results"][0]["sparql"]
                else:
                    simulated_sparql = prefix + "\n" + detection_item["search_results"][0]
            else:
                detection_item["test_suite"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
                f1_list.append(0.0)
                continue

            detection_item['simulated_sparql_query'] = simulated_sparql

            editor = SyntaxTreeEditor(simulated_sparql, method, logger)
            editor.preprocess_test_suite(f"?{TARGET_VARIABLE_NAME}")
            simulated_sparql_processed = editor.sparql_txt

            if qid not in qid_to_test_suite:
                continue

            src_item = qid_to_test_suite[qid]
            test_suite_examples = src_item.get("test_suite_examples", [])

            if len(test_suite_examples) == 0:
                detection_item["test_suite"] = {"precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0}
                f1_list.append(1.0)
                continue

            prec_list, recall_list, f1_scores, acc_list = [], [], [], []
            for test_case in test_suite_examples:
                test_editor = SyntaxTreeEditor(simulated_sparql_processed, dataset, logger)
                overlap = False
                for before, after in test_case["binding"].items():
                    if test_editor.update_leaf_value(before, after):
                        overlap = True
                if not overlap:
                    continue

                predicted = sparql_querier.get_execution_result_one_variable_sparql_wrapper(
                    test_editor.sparql_txt, retry=1)
                gold = test_case["answer"]
                p, r, f1 = get_PRF1(predicted, gold)
                prec_list.append(p)
                recall_list.append(r)
                f1_scores.append(f1)
                acc_list.append(1.0 if math.isclose(f1, 1.0) else 0.0)

            if len(f1_scores) == 0:
                detection_item["test_suite"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
                f1_list.append(0.0)
            else:
                avg_f1 = sum(f1_scores) / len(f1_scores)
                detection_item["test_suite"] = {
                    "precision": sum(prec_list) / len(prec_list),
                    "recall": sum(recall_list) / len(recall_list),
                    "f1": avg_f1,
                    "accuracy": sum(acc_list) / len(acc_list),
                }
                f1_list.append(avg_f1)

        except Exception as e:
            logger.error(f"qid: {qid}; error: {e}")
            detection_item["test_suite"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
            f1_list.append(0.0)

    if 'summary' not in constructed_data:
        constructed_data["summary"] = {}
    constructed_data["summary"]["test_suite"] = {
        "average_f1": sum(f1_list) / len(f1_list),
    }
    return constructed_data


def _generate_test_cases(test_suite_item: dict, dataset: DATASET,
                          sparql_querier, logger,
                          test_cases_limit: int = TSS_TEST_SUITE_GEN_LIMIT) -> list:
    """
    Sample bindings from the all-at-once KB query result, execute each sampled
    binding to obtain the golden answer, and return test cases.

    Only test cases with non-empty answers are kept (paper constraint: "the
    output of the new test case is non-empty").

    Parameters
    ----------
    test_suite_item : dict
        {"original_sparql": str, "bindings": [dict, ...]}
        Each binding is {tree_string: concrete_value, ...} from KB grounding.
    dataset : DATASET
    sparql_querier : SparqlOdbcQuerierNoSexpr | SparqlOdbcQuerierNoSexprWikidata
    logger
    test_cases_limit : int
        Max number of test cases to generate (randomly sampled).

    Returns
    -------
    list of dict
        [{"original_sparql": str, "binding": dict, "answer": list}, ...]
    """
    test_cases = []
    bindings = copy.deepcopy(test_suite_item["bindings"])
    random.shuffle(bindings)
    # Cap the number of bindings to try — for high-cardinality properties
    # (P50, P175, P577, etc.) the abstracted query may return tens of thousands
    # of bindings, most of which produce empty answers.  Without a cap, the
    # per-binding loop can issue thousands of SPARQL queries before finding
    # test_cases_limit non-empty ones.  max_tries limits this to a reasonable
    # upper bound, trading a small risk of fewer test cases for speed.
    # 限制尝试 binding 的最大数量。高基数属性（P50/P175/P577 等）的抽象查询
    # 可能返回数万 binding，大部分产生空答案。不加限制会导致数千次 SPARQL 查询。
    max_tries = max(test_cases_limit * 5, 250)
    for idx, binding in enumerate(bindings):
        if idx >= max_tries or len(test_cases) >= test_cases_limit:
            break
        editor = SyntaxTreeEditor(test_suite_item["original_sparql"], dataset, logger)
        for old_str, new_v in binding.items():
            editor.update_leaf_value(old_str, new_v)
        sparql_txt = editor.sparql_txt
        execution_result = sparql_querier.get_execution_result_one_variable_sparql_wrapper(
            sparql_txt, retry=2)
        if not execution_result:
            # Non-empty constraint (paper: ensure test case is meaningful)
            continue
        test_cases.append({
            "original_sparql": test_suite_item["original_sparql"],
            "binding": binding,
            "answer": list(execution_result),
        })
    return test_cases

def _generate_test_suite_for_golden_per_binding(
    sparql_before_replace: str,
    key_list: list,
    value_list: list,
    ds,
    sparql_querier,
    logger,
    test_cases_limit: int,
    test_cases_collection: list,
):
    """
    Method 3 per-binding 实现：
    抽象查询仅获取实体绑定 (bindings)，移除 ?target 投影。
    对每个采样到的 binding，单独执行具体 SPARQL 查询获取精确答案。
    max_tries 限制最大尝试次数，避免高基数属性的无限循环。

    Method 3 per-binding implementation: removes ?target from the abstracted
    query projection so only entity bindings are returned.  Answers are obtained
    by executing a separate concrete SPARQL query for each sampled binding.
    This avoids LIMIT-truncation of answers.  max_tries caps the number of
    bindings attempted to prevent runaway loops on high-cardinality properties.
    """
    for value_after_replace in itertools.product(*value_list):
        editor2 = SyntaxTreeEditor(sparql_before_replace, ds, logger)
        editor2.remove_projection_variable(f"?{TARGET_VARIABLE_NAME}")

        for key, value in zip(key_list, value_after_replace):
            if value.startswith('?'):
                if not editor2.replace_leaf_with_variable(key, value):
                    raise ValueError(f"key: {key} not found in syntax tree")
            else:
                if not editor2.update_leaf_value(key, value):
                    raise ValueError(f"key: {key} not found in syntax tree")

        sparql_after_replace = editor2.sparql_txt
        # Guard against unbounded all-at-once queries on both Freebase & Wikidata.
        # 对 Freebase 和 Wikidata 统一加行数上限，避免 ODBC 无限 fetch。
        if 'LIMIT' not in sparql_after_replace.upper():
            limit = TSS_QUERY_MAX_ROWS if ds in FB_DATASETS else max(test_cases_limit * 10, 50000)
            sparql_after_replace = sparql_after_replace.rstrip() + f" LIMIT {limit}"
        rows = sparql_querier.execute_query(sparql_after_replace, retry=1)

        # Parse each subgraph row → concrete entity groundings
        test_suite_bindings = []
        for row in rows:
            row_result = {}
            skip_flag = False
            for key, value in zip(key_list, value_after_replace):
                if value.startswith('?'):
                    try:
                        if ds in FB_DATASETS:
                            row_result[key] = get_syntax_tree_value(
                                FreebaseConstantForConstruction(
                                    row[value[1:]]['type'], row[value[1:]]['value'],
                                    row[value[1:]].get('datatype', None),
                                    row[value[1:]].get('xml:lang', None)
                                ).__repr__(), ds)
                        elif ds in WD_DATASETS:
                            row_result[key] = get_syntax_tree_value(
                                WikidataConstantForConstruction(
                                    row[value[1:]]['type'], row[value[1:]]['value'],
                                    row[value[1:]].get('datatype', None),
                                    row[value[1:]].get('xml:lang', None)
                                ).__repr__(), ds)
                    except Exception:
                        skip_flag = True
                else:
                    row_result[key] = value
            if not skip_flag:
                test_suite_bindings.append(row_result)

        if test_suite_bindings:
            test_suite_item = {
                "original_sparql": sparql_before_replace,
                "bindings": test_suite_bindings,
            }
            test_cases = _generate_test_cases(
                test_suite_item, ds, sparql_querier, logger, test_cases_limit)
            test_cases_collection.extend(test_cases)


def _generate_test_suite_for_golden(golden_item: dict, dataset: DATASET,
                                     sparql_querier, logger,
                                     test_cases_limit: int = TSS_TEST_SUITE_GEN_LIMIT) -> list:
    """
    Generate test_suite_examples for a single golden query.

    为单条 golden query 生成 test suite examples（Method 3 的核心）。
    选择一个查询为主查询（golden），将其 entity 抽象化后在 KG 上采样
    扰动集合并记录对应答案。该扰动集合作为"主键"可跨 constructed query 复用。

    Method 3 (Precomputed Test Suite TSS):
      1. 选择主查询 (golden)，提取其中的 grounded entities
      2. 构建 replace_map: entities → [?replaceN], quantities → [old, ×2, ×3]
      3. 对每种替换组合执行 all-at-once KB 查询 → 实体绑定 (bindings)
      4. 对每个采样 binding 执行具体 SPARQL 查询获取精确答案
      5. binding 作为主键 {tree_string → value}，可跨 constructed query 复用
      6. max_tries 限制最大尝试次数，避免高基数属性上的无限循环

    答案通过逐条查询（per-binding）获取，不受 all-at-once LIMIT 截断影响。
    应答格式与 get_execution_result_one_variable_sparql_wrapper 一致
    （WikidataConstantForConstruction.__repr__()），确保评估时 gold/pred 可比。

    Algorithm (paper Figure 2):
      1. Build replace_map: entities → [?replaceN], quantities → [old,×2,×3]
      2. For each combination in itertools.product(*value_list):
         a. Replace all entities as projection variables simultaneously
         b. Execute ONE SPARQL query on the KB → entity bindings
         c. For each binding, execute concrete SPARQL → precise answer
      3. Sample up to test_cases_limit with non-empty answers, capped by max_tries.

    Also includes the original (unperturbed) test case from the dataset as the
    first entry, ensuring the ground-truth is always tested.

    Parameters
    ----------
    golden_item : dict
        Must contain: golden_sparql_query, golden_grounded_items.
        May contain: answer (for the original test case).
    dataset : DATASET
    sparql_querier : SparqlOdbcQuerierNoSexpr | SparqlOdbcQuerierNoSexprWikidata
    logger
    test_cases_limit : int
        Max test cases per perturbation combination (default TSS_TEST_SUITE_GEN_LIMIT=50).

    Returns
    -------
    list of dict
        [{"original_sparql": str, "binding": dict, "answer": list}, ...]
        answer format matches get_execution_result_one_variable_sparql_wrapper output.
    """
    original_sparql = golden_item["golden_sparql_query"]
    replaceable_items = [item['mid'] for item in golden_item["golden_grounded_items"]]

    # ── Normalize dataset (DATASET → Dataset) for internal API compatibility ──
    ds = to_dataset(dataset)

    # Add prefix declarations so that prefixed names (wd:Qxxx) can be resolved
    _prefix = _get_sparql_prefix(dataset)
    if not original_sparql.strip().lower().startswith('prefix'):
        original_sparql = _prefix + "\n" + original_sparql

    # Dataset-specific fixes
    if dataset is DATASET.GRAIL:
        for idx, item in enumerate(replaceable_items):
            if any(item.endswith(t) for t in ['gYear>', 'gYearMonth>', 'date>']):
                replaceable_items[idx] = (
                    f'"{item.split("^^")[0][1:-1] + "-08:00"}"^^{item.split("^^")[1]}')

    # NOTE: _modified_wikidata_sparql simplification (p:P/ps:P → wdt:P)
    # is skipped for Method 3 because it causes semantic differences for
    # multi-valued entity properties. The test suite's original_sparql must
    # match the constructed queries' query form for accurate evaluation.
    # if ds == Dataset.LC2:
    #     original_sparql = _modified_wikidata_sparql(original_sparql)

    editor = SyntaxTreeEditor(original_sparql, ds, logger)
    editor.preprocess_test_suite(f"?{TARGET_VARIABLE_NAME}")
    sparql_before_replace = editor.sparql_txt

    # ── Fix: BIND AS variable not renamed by _rename_target_variable ──
    # _rename_target_variable renames ?x → ?target in SELECT and leaf nodes,
    # but BIND(... AS ?x) variables are not in the leafs list. This causes
    # ?target to be unbound for queries like:
    #   SELECT ?x WHERE { ... BIND(... AS ?x) }
    # We fix it textually: extract old projection var and replace AS bindings.
    _old_proj_match = re.search(
        r'(?i)select\s+(?:distinct\s+)?\?(\w+)',
        original_sparql)
    if _old_proj_match:
        _old_proj_var = _old_proj_match.group(1)
        if _old_proj_var != TARGET_VARIABLE_NAME:
            sparql_before_replace = re.sub(
                r'\bAS\s+\?' + _old_proj_var + r'\b',
                f'AS ?{TARGET_VARIABLE_NAME}',
                sparql_before_replace)

    # ── Always include the original test case (unperturbed, from dataset) ──
    test_cases_collection = []
    if "answer" in golden_item and golden_item["answer"]:
        original_answer = [item['mid'] if isinstance(item, dict) else item
                           for item in golden_item["answer"]]
    else:
        # Fall back to KB execution for the original answer
        original_answer = list(sparql_querier.get_execution_result_one_variable_sparql_wrapper(
            sparql_before_replace, retry=2))
    original_binding = {}
    for item in replaceable_items:
        try:
            tree_str = get_syntax_tree_string(item, ds)
            original_binding[tree_str] = get_syntax_tree_value(item, ds)
        except Exception:
            pass
    test_cases_collection.append({
        "original_sparql": sparql_before_replace,
        "answer": original_answer,
        "binding": original_binding,
    })

    # ── Build replace_map ──
    # entity/class → [?replaceN]; quantity → [old, ×2, ×3]; time → [self]
    replace_map = {}
    for idx, item in enumerate(replaceable_items):
        if ds in FB_DATASETS:
            _type = FreebaseConstantForConstruction.get_constant_type(item)
            if _type is FREEBASE_CONSTANT_TYPE.QUANTITY:
                old_value = item.split('^^')[0][1:-1]
                suffix = item.split('^^')[1]
                if suffix == "<http://www.w3.org/2001/XMLSchema#integer>":
                    replace_map[get_syntax_tree_string(item, ds)] = [
                        f'"{old_value}"',
                        f'"{str(int(old_value) * 2)}"',
                        f'"{str(int(old_value) * 3)}"',
                    ]
                else:
                    replace_map[get_syntax_tree_string(item, ds)] = [
                        f'"{old_value}"',
                        f'"{str(float(old_value) * 2.0)}"',
                        f'"{str(float(old_value) * 3.0)}"',
                    ]
            elif _type is FREEBASE_CONSTANT_TYPE.TIME:
                replace_map[get_syntax_tree_string(item, ds)] = [
                    get_syntax_tree_value(item, ds)]
            else:  # entity / class → variable
                replace_map[get_syntax_tree_string(item, ds)] = [
                    f"?{REPLACED_VARIABLE_NAME}{idx}"]

        elif ds in WD_DATASETS:
            _type = WikidataConstantForConstruction.get_constant_type(item)
            if _type is WIKIDATA_CONSTANT_TYPE.QUANTITY:
                old_value = item.split('^^')[0][1:-1]
                suffix = item.split('^^')[1]
                if suffix == "<http://www.w3.org/2001/XMLSchema#integer>":
                    replace_map[get_syntax_tree_string(item, ds)] = [
                        f'"{old_value}"',
                        f'"{str(int(old_value) * 2)}"',
                        f'"{str(int(old_value) * 3)}"',
                    ]
                else:
                    replace_map[get_syntax_tree_string(item, ds)] = [
                        f'"{old_value}"',
                        f'"{str(float(old_value) * 2.0)}"',
                        f'"{str(float(old_value) * 3.0)}"',
                    ]
            elif _type is WIKIDATA_CONSTANT_TYPE.TIME:
                replace_map[get_syntax_tree_string(item, ds)] = [
                    get_syntax_tree_value(item, ds)]
            else:  # entity / class → variable
                replace_map[get_syntax_tree_string(item, ds)] = [
                    f"?{REPLACED_VARIABLE_NAME}{idx}"]
        else:
            raise NotImplementedError(f"dataset: {dataset}")

    if len(replace_map) == 0:
        return test_cases_collection

    key_list = list(replace_map.keys())
    value_list = [replace_map[key] for key in key_list]

    # ── Per-binding perturbation + KB query ──
    # 抽象查询获取实体绑定 (bindings)，对每个采样 binding 执行具体 SPARQL
    # 查询获取精确答案。不受 LIMIT 截断影响。
    _generate_test_suite_for_golden_per_binding(
        sparql_before_replace, key_list, value_list, ds,
        sparql_querier, logger, test_cases_limit, test_cases_collection)

    return test_cases_collection


# ═══════════════════════════════════════════════════════════════════════════════
# Batch In-Memory Wrapper (no file I/O)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_test_suites_batch(golden_data: list, dataset: DATASET,
                                sparql_querier, logger,
                                test_cases_limit: int = TSS_TEST_SUITE_GEN_LIMIT) -> list:
    """Generate test_suite_examples for all golden items. Pure in-memory.

    Parameters
    ----------
    golden_data : list of dict
        Each item must contain: golden_sparql_query, golden_grounded_items.
    dataset : DATASET
    sparql_querier : SparqlOdbcQuerierNoSexpr or SparqlOdbcQuerierNoSexprWikidata
    logger
    test_cases_limit : int

    Returns
    -------
    list of dict — the golden_data with 'test_suite_examples' added to each item.
    """
    random.seed(TSS_RANDOM_SEED)
    error_count = 0
    for example_item in tqdm(golden_data, desc="Generating test suites", unit="q"):
        try:
            example_item['test_suite_examples'] = _generate_test_suite_for_golden(
                example_item, dataset, sparql_querier, logger, test_cases_limit)
        except Exception as e:
            logger.error(f"qid: {example_item.get('qid', '?')}; error: {e}")
            example_item['test_suite_examples'] = []
            error_count += 1
    logger.info(f"generate_test_suites_batch done. Errors: {error_count}/{len(golden_data)}")
    return golden_data