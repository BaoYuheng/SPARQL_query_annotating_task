#!/usr/bin/env python3
"""
KBQA Performance Evaluation -- Refactored OOP Version.

Provides a unified class hierarchy for computing KBQA metrics across
different datasets and methods.  The primary interface is :meth:`Evaluator.evaluate`:
given gold and prediction file paths, it returns a metrics dictionary.

Architecture
------------
Evaluator (ABC)
├── PanguEvaluator
│   ├── PanguWebQSPEvaluator   (Dataset.WEBQ)
│   └── PanguGrailQAEvaluator  (Dataset.GRAIL)
└── GMTKBQAEvaluator           (Dataset.CWQ)

This code is sorted and refined using Claude Code with DeepSeekV4-pro.
Date:2026-05-30
"""

from __future__ import annotations

import json
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional

import networkx as nx
from tqdm import tqdm

# Ensure the project root is on sys.path AND is the CWD so that
# src imports work (it uses relative paths like data/test/sparql.txt).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

# -- GMT-KBQA dependencies (lazy-imported — src triggers heavy ML deps) --
# load_json / dump_json are imported inside GMTKBQAEvaluator._ensure_utils()
# SparqlOdbcQuerierNoSexpr is imported inside GMTKBQAEvaluator._get_executor()
import logging

# -- Dataset enum (prefer the canonical definition in src.core) ----
try:
    from src.core.common import Dataset
except ImportError:  # pragma: no cover
    from enum import Enum

    class Dataset(Enum):  # type: ignore[no-redef]
        GRAIL = 1
        CWQ = 2
        WEBQ = 3


# ====================================================================
#  Base Evaluator
# ====================================================================

class Evaluator(ABC):
    """Abstract base for all KBQA evaluators.

    Subclasses implement :meth:`evaluate`, which reads *gold_file* and
    *pred_file* (both JSON) and returns a ``dict`` of metrics.
    """

    @abstractmethod
    def evaluate(self, gold_file: str, pred_file: str) -> Dict[str, Any]:
        """Compute evaluation metrics.

        Args:
            gold_file: Path to the gold-standard dataset file (JSON).
            pred_file: Path to the prediction output file (JSON).

        Returns:
            A dictionary of computed metrics.
        """
        ...


class PanguEvaluator(Evaluator):
    """Intermediate base for Pangu-family evaluators (WebQSP & GrailQA).

    Both datasets target **Freebase** and share Pangu's official evaluation
    methodology.

    Original source:
    https://www.microsoft.com/en-us/download/details.aspx?id=52763
    """

    _dataset: Dataset


# ====================================================================
#  Pangu / WebQSP  —  Dataset.WEBQ
# ====================================================================
#
#  Official Pangu Evaluation Script for WebQSP.
#  来源: https://www.microsoft.com/en-us/download/details.aspx?id=52763
#  原始代码为 Python 2，此处做了格式适配。
# ====================================================================


def _FindInList(entry: str, elist: List[str]) -> bool:
    """Return True if *entry* is in *elist*."""
    for item in elist:
        if entry == item:
            return True
    return False


def _CalculatePRF1(goldAnswerList: List[Dict[str, str]],
                   predAnswerList: List[str]) -> List[float]:
    """Compute precision, recall and F1 for a single question.

    Args:
        goldAnswerList: list of ``{"AnswerArgument": ...}`` or
            ``{"answer_argument": ...}`` dicts (key auto-detected).
        predAnswerList: list of predicted answer strings.

    Returns:
        ``[precision, recall, f1]``.
    """
    if len(goldAnswerList) == 0:
        if len(predAnswerList) == 0:
            return [1.0, 1.0, 1.0]
        else:
            return [0.0, 1.0, 0.0]
    elif len(predAnswerList) == 0:
        return [1.0, 0.0, 0.0]
    else:
        # Auto-detect answer key: original Pangu uses "AnswerArgument",
        # normalised format uses "answer_argument"
        key = ("AnswerArgument" if "AnswerArgument" in goldAnswerList[0]
               else "answer_argument")
        glist = [x[key] for x in goldAnswerList]
        plist = predAnswerList

        tp = 1e-40  # numerical trick to avoid division-by-zero
        fp = 0.0
        fn = 0.0

        for gentry in glist:
            if _FindInList(gentry, plist):
                tp += 1
            else:
                fn += 1
        for pentry in plist:
            if not _FindInList(pentry, glist):
                fp += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
        return [precision, recall, f1]


class PanguWebQSPEvaluator(PanguEvaluator):
    """Pangu evaluator for **WebQSP** (Freebase KB).

    Original source:
    https://www.microsoft.com/en-us/download/details.aspx?id=52763
    (Python 2 → Python 3 adaptation)

    Corresponding function in the legacy script:
    ``Pangu_webqsp_main()``
    """

    _dataset = Dataset.WEBQ

    # ------------------------------------------------------------------
    #  Format conversion utility
    # ------------------------------------------------------------------

    @staticmethod
    def convert_prediction_format(folder: str) -> str:
        """Convert line-based prediction file to the format expected by
        :meth:`evaluate`.

        Reads ``predictions/{folder}/predictions_oracle_entity_linking.txt``
        and writes ``predictions/{folder}/predictions_for_evaluation.json``.

        Args:
            folder: sub-directory name under ``predictions/``.

        Returns:
            Path to the generated JSON file.
        """
        # Original source: convert_prediction_format() (WebQSP section)
        old_file = f"predictions/{folder}/predictions_oracle_entity_linking.txt"
        data: List[Dict[str, Any]] = []
        with open(old_file, 'r') as f:
            for line in f:
                line_data = json.loads(line)
                data.append({
                    "QuestionId": line_data["qid"],
                    "Answers": line_data["answer"],
                })
        out_path = f"predictions/{folder}/predictions_for_evaluation.json"
        json.dump(data, open(out_path, 'w'))
        return out_path

    # ------------------------------------------------------------------
    #  Main evaluation
    # ------------------------------------------------------------------

    def evaluate(self, gold_file: str, pred_file: str) -> Dict[str, Any]:
        """Compute WebQSP evaluation metrics.

        Supports two gold formats (auto-detected):

        1. **Original Pangu format**::

               {"Questions": [{"QuestionId": ..., "Parses": [
                   {"Answers": [{"AnswerArgument": ...}, ...],
                    "AnnotatorComment": {"QuestionQuality": ..., "ParseQuality": ...}},
                   ...]}, ...]}

        2. **Normalised list format** (used by this project)::

               [{"qid": ..., "answer": [{"answer_argument": ..., "answer_type": ..., "entity_name": ...}, ...]}, ...]

        Pred format (expected)::

            [{"QuestionId": ..., "Answers": [...]}, ...]

        Returns:
            Dict with keys: ``num_questions``, ``avg_precision``,
            ``avg_recall``, ``avg_f1``, ``f1_of_avg``, ``true_accuracy``.
        """
        # Original source: Pangu_webqsp_main()
        goldData = json.loads(open(gold_file).read())
        predAnswers = json.loads(open(pred_file).read())

        # -- Auto-detect gold format --------------------------------------------------
        if isinstance(goldData, dict) and "Questions" in goldData:
            # Format 1: original Pangu {"Questions": [...]}
            questions = goldData["Questions"]
            _fmt_v1 = True
        elif isinstance(goldData, list):
            # Format 2: normalised list [{"qid": ..., "answer": ...}, ...]
            questions = goldData
            _fmt_v1 = False
        else:
            raise ValueError(f"Unrecognised WebQSP gold format: {type(goldData)}")

        # -- Build gold lookup by qid (list of answer sets, one per parse) ------------
        GoldById: Dict[str, List[List[Dict[str, str]]]] = {}
        for entry in questions:
            if _fmt_v1:
                qid = entry["QuestionId"]
                parses = entry.get("Parses", [])
                # Filter: only keep parses with Good question quality + Complete parse
                GoldById[qid] = [
                    p["Answers"] for p in parses
                    if (p.get("AnnotatorComment", {}).get("QuestionQuality") == "Good" and
                        p.get("AnnotatorComment", {}).get("ParseQuality") == "Complete")
                ]
            else:
                qid = str(entry["qid"])
                # Normalised format: single "parse" equivalent, no quality filter
                GoldById[qid] = [entry["answer"]]

        # -- Build pred lookup --------------------------------------------------------
        PredAnswersById: Dict[str, List[str]] = {}
        for item in predAnswers:
            PredAnswersById[str(item["QuestionId"])] = item["Answers"]

        total = 0.0
        f1sum = 0.0
        recSum = 0.0
        precSum = 0.0
        numCorrect = 0

        for qid, parse_answers_list in GoldById.items():
            if not parse_answers_list:
                continue

            total += 1

            if qid not in PredAnswersById:
                print(f"The problem {qid} is not in the prediction set")
                print("Continue to evaluate the other entries")
                continue

            predAnswers_q = PredAnswersById[qid]

            bestf1 = -9999.0
            bestf1Rec = -9999.0
            bestf1Prec = -9999.0
            for pidxAnswers in parse_answers_list:
                prec, rec, f1 = _CalculatePRF1(pidxAnswers, predAnswers_q)
                if f1 > bestf1:
                    bestf1 = f1
                    bestf1Rec = rec
                    bestf1Prec = prec

            f1sum += bestf1
            recSum += bestf1Rec
            precSum += bestf1Prec
            if bestf1 == 1.0:
                numCorrect += 1

        avg_p = precSum / total
        avg_r = recSum / total
        avg_f1 = f1sum / total
        f1_of_avg = (2 * avg_r * avg_p / (avg_r + avg_p)) if (avg_r + avg_p) > 0 else 0.0

        metrics: Dict[str, Any] = {
            "num_questions": int(total),
            "avg_precision": round(avg_p, 4),
            "avg_recall": round(avg_r, 4),
            "avg_f1": round(avg_f1, 4),
            "f1_of_avg": round(f1_of_avg, 4),
            "true_accuracy": round(numCorrect / total, 4),
        }

        # Also print (matching original behaviour)
        print(f"Number of questions: {int(total)}")
        print(f"Average precision over questions: {avg_p:.4f}")
        print(f"Average recall over questions: {avg_r:.4f}")
        print(f"Average f1 over questions (accuracy): {avg_f1:.4f}")
        print(f"F1 of average recall and average precision: {f1_of_avg:.4f}")
        print(f"True accuracy (ratio of questions answered exactly correctly): {numCorrect / total:.4f}")

        return metrics


# ====================================================================
#  Pangu / GrailQA  —  Dataset.GRAIL
# ====================================================================
#
#  Official Pangu Evaluation Script for GrailQA.
#  来源: https://worksheets.codalab.org/rest/bundles/0x2d13989c17e44690ab62cc4edc0b900d/contents/blob/
# ====================================================================

function_map = {'le': '<=', 'ge': '>=', 'lt': '<', 'gt': '>'}


def process_ontology(fb_roles_file: str, fb_types_file: str,
                     reverse_properties_file: str):
    """Load Freebase ontology files.

    Original source: ``process_ontology()`` in Pangu GrailQA script.
    """
    reverse_properties: Dict[str, str] = {}
    with open(reverse_properties_file, 'r') as f:
        for line in f:
            reverse_properties[line.split('\t')[0]] = \
                line.split('\t')[1].replace('\n', '')

    with open(fb_roles_file, 'r') as f:
        content = f.readlines()

    relation_dr: Dict[str, tuple] = {}
    relations: set = set()
    for line in content:
        fields = line.split()
        relation_dr[fields[1]] = (fields[0], fields[2])
        relations.add(fields[1])

    with open(fb_types_file, 'r') as f:
        content = f.readlines()

    upper_types: defaultdict = defaultdict(lambda: set())
    types: set = set()
    for line in content:
        fields = line.split()
        upper_types[fields[0]].add(fields[2])
        types.add(fields[0])
        types.add(fields[2])

    return reverse_properties, relation_dr, relations, upper_types, types


def lisp_to_nested_expression(lisp_string: str) -> List:
    """Parse a Lisp S-expression string into a nested list.

    E.g. ``"(count (division first))"`` →
    ``['count', ['division', 'first']]``.

    Original source: ``lisp_to_nested_expression()`` in Pangu GrailQA script.
    """
    stack: List = []
    current_expression: List = []
    tokens = lisp_string.split()
    for token in tokens:
        while token[0] == '(':
            nested_expression: List = []
            current_expression.append(nested_expression)
            stack.append(current_expression)
            current_expression = nested_expression
            token = token[1:]
        current_expression.append(token.replace(')', ''))
        while token[-1] == ')':
            current_expression = stack.pop()
            token = token[:-1]
    return current_expression[0]


class SemanticMatcher:
    """Graph-based semantic matcher for GrailQA logical forms.

    Converts logical forms (S-expressions) to NetworkX graphs and checks
    for isomorphism to determine equivalence.

    Original source: ``SemanticMatcher`` class in Pangu GrailQA script.
    Source URL:
    https://worksheets.codalab.org/rest/bundles/0x2d13989c17e44690ab62cc4edc0b900d/contents/blob/
    """

    def __init__(self, reverse_properties: Dict[str, str],
                 relation_dr: Dict[str, tuple],
                 relations: set,
                 upper_types: defaultdict,
                 types: set):
        self.reverse_properties = reverse_properties
        self.relation_dr = relation_dr
        self.relations = relations
        self.upper_types = upper_types
        self.types = types

    def same_logical_form(self, form1: str, form2: str) -> bool:
        if form1.__contains__("@@UNKNOWN@@") or form2.__contains__("@@UNKNOWN@@"):
            return False
        try:
            G1 = self.logical_form_to_graph(lisp_to_nested_expression(form1))
        except Exception:
            return False
        try:
            G2 = self.logical_form_to_graph(lisp_to_nested_expression(form2))
        except Exception:
            return False

        def node_match(n1, n2):
            if n1['id'] == n2['id'] and n1['type'] == n2['type']:
                func1 = n1.pop('function', 'none')
                func2 = n2.pop('function', 'none')
                tc1 = n1.pop('tc', 'none')
                tc2 = n2.pop('tc', 'none')
                if func1 == func2 and tc1 == tc2:
                    return True
                else:
                    return False
            else:
                return False

        def multi_edge_match(e1, e2):
            if len(e1) != len(e2):
                return False
            values1 = []
            values2 = []
            for v in e1.values():
                values1.append(v['relation'])
            for v in e2.values():
                values2.append(v['relation'])
            return sorted(values1) == sorted(values2)

        return nx.is_isomorphic(G1, G2, node_match=node_match,
                                edge_match=multi_edge_match)

    def get_symbol_type(self, symbol: str) -> int:
        if symbol.__contains__('^^'):  # typed literal
            return 2
        elif symbol in self.types:
            return 3
        elif symbol in self.relations:
            return 4
        else:
            return 1

    def logical_form_to_graph(self, expression: List) -> nx.MultiGraph:
        G = self._get_graph(expression)
        G.nodes[len(G.nodes())]['question_node'] = 1
        return G

    def _get_graph(self, expression: List) -> nx.MultiDiGraph:
        if isinstance(expression, str):
            G = nx.MultiDiGraph()
            if self.get_symbol_type(expression) == 1:
                G.add_node(1, id=expression, type='entity')
            elif self.get_symbol_type(expression) == 2:
                G.add_node(1, id=expression, type='literal')
            elif self.get_symbol_type(expression) == 3:
                G.add_node(1, id=expression, type='class')
            elif self.get_symbol_type(expression) == 4:
                domain, rang = self.relation_dr[expression]
                G.add_node(1, id=rang, type='class')
                G.add_node(2, id=domain, type='class')
                G.add_edge(2, 1, relation=expression)
                if expression in self.reverse_properties:
                    G.add_edge(1, 2,
                               relation=self.reverse_properties[expression])
            return G

        if expression[0] == 'R':
            G = self._get_graph(expression[1])
            size = len(G.nodes())
            mapping = {}
            for n in G.nodes():
                mapping[n] = size - n + 1
            G = nx.relabel_nodes(G, mapping)
            return G

        elif expression[0] in ['JOIN', 'le', 'ge', 'lt', 'gt']:
            G1 = self._get_graph(expression=expression[1])
            G2 = self._get_graph(expression=expression[2])
            size = len(G2.nodes())
            qn_id = size
            if G1.nodes[1]['type'] == G2.nodes[qn_id]['type'] == 'class':
                if G2.nodes[qn_id]['id'] in self.upper_types[G1.nodes[1]['id']]:
                    G2.nodes[qn_id]['id'] = G1.nodes[1]['id']
            mapping = {}
            for n in G1.nodes():
                mapping[n] = n + size - 1
            G1 = nx.relabel_nodes(G1, mapping)
            G = nx.compose(G1, G2)
            if expression[0] != 'JOIN':
                G.nodes[1]['function'] = function_map[expression[0]]
            return G

        elif expression[0] == 'AND':
            G1 = self._get_graph(expression[1])
            G2 = self._get_graph(expression[2])
            size1 = len(G1.nodes())
            size2 = len(G2.nodes())
            if G1.nodes[size1]['type'] == G2.nodes[size2]['type'] == 'class':
                G2.nodes[size2]['id'] = G1.nodes[size1]['id']
            mapping = {}
            for n in G1.nodes():
                mapping[n] = n + size2 - 1
            G1 = nx.relabel_nodes(G1, mapping)
            G2 = nx.relabel_nodes(G2, {size2: size1 + size2 - 1})
            G = nx.compose(G1, G2)
            return G

        elif expression[0] == 'COUNT':
            G = self._get_graph(expression[1])
            size = len(G.nodes())
            G.nodes[size]['function'] = 'count'
            return G

        elif expression[0].__contains__('ARG'):
            G1 = self._get_graph(expression[1])
            size1 = len(G1.nodes())
            G2 = self._get_graph(expression[2])
            size2 = len(G2.nodes())
            G2.nodes[1]['id'] = 0
            G2.nodes[1]['type'] = 'literal'
            G2.nodes[1]['function'] = expression[0].lower()
            if G1.nodes[size1]['type'] == G2.nodes[size2]['type'] == 'class':
                G2.nodes[size2]['id'] = G1.nodes[size1]['id']
            mapping = {}
            for n in G1.nodes():
                mapping[n] = n + size2 - 1
            G1 = nx.relabel_nodes(G1, mapping)
            G2 = nx.relabel_nodes(G2, {size2: size1 + size2 - 1})
            G = nx.compose(G1, G2)
            return G

        elif expression[0] == 'TC':
            G = self._get_graph(expression[1])
            size = len(G.nodes())
            G.nodes[size]['tc'] = (expression[2], expression[3])
            return G


class PanguGrailQAEvaluator(PanguEvaluator):
    """Pangu evaluator for **GrailQA** (Freebase KB).

    Uses graph isomorphism on logical forms (S-expressions) to compute
    Exact Match, plus answer-set overlap for F1.

    Original source:
    https://worksheets.codalab.org/rest/bundles/0x2d13989c17e44690ab62cc4edc0b900d/contents/blob/

    Corresponding function in the legacy script:
    ``Pangu_GrailQA_main()``
    """

    _dataset = Dataset.GRAIL

    # Hardcoded ontology paths relative to the project root
    _FB_ONTOLOGY_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "FB_ontology",
    )

    def __init__(self,
                 fb_roles: Optional[str] = None,
                 fb_types: Optional[str] = None,
                 reverse_properties: Optional[str] = None):
        """Load ontology and initialise the semantic matcher.

        Args:
            fb_roles: Path to the ``fb_roles`` ontology file.
                Defaults to ``data/FB_ontology/fb_roles`` under the project root.
            fb_types: Path to the ``fb_types`` ontology file.
                Defaults to ``data/FB_ontology/fb_types`` under the project root.
            reverse_properties: Path to the ``reverse_properties`` file.
                Defaults to ``data/FB_ontology/reverse_properties``.
        """
        fb_roles = fb_roles or os.path.join(self._FB_ONTOLOGY_DIR, "fb_roles")
        fb_types = fb_types or os.path.join(self._FB_ONTOLOGY_DIR, "fb_types")
        reverse_properties = (reverse_properties or
                              os.path.join(self._FB_ONTOLOGY_DIR, "reverse_properties"))

        (reverse_props, relation_dr, relations,
         upper_types, types) = process_ontology(fb_roles, fb_types,
                                                 reverse_properties)
        self._matcher = SemanticMatcher(reverse_props, relation_dr,
                                        relations, upper_types, types)

    # ------------------------------------------------------------------
    #  Format conversion utility
    # ------------------------------------------------------------------

    @staticmethod
    def convert_prediction_format(folder: str) -> str:
        """Convert line-based GrailQA prediction file.

        Reads ``predictions/{folder}/predictions.txt`` and writes
        ``predictions/{folder}/predictions_for_evaluation.json``.

        Args:
            folder: sub-directory name under ``predictions/``.

        Returns:
            Path to the generated JSON file.
        """
        # Original source: convert_prediction_format() (GrailQA section)
        old_file = f"predictions/{folder}/predictions.txt"
        data: Dict[str, Dict[str, Any]] = {}
        with open(old_file, 'r') as f:
            for line in f:
                line_data = json.loads(line)
                data[line_data["qid"]] = {
                    "logical_form": line_data["logical_form"],
                    "answer": line_data["answer"],
                }
        out_path = f"predictions/{folder}/predictions_for_evaluation.json"
        json.dump(data, open(out_path, 'w'))
        return out_path

    # ------------------------------------------------------------------
    #  Main evaluation
    # ------------------------------------------------------------------

    def evaluate(self, gold_file: str, pred_file: str) -> Dict[str, Any]:
        """Compute GrailQA evaluation metrics.

        Gold format: list of items with keys
        ``qid``, ``s_expression``, ``answer``, ``level``.

        Pred format: dict ``{qid: {"logical_form": str, "answer": list}}``.

        Returns:
            Dict with keys: ``em``, ``f1``, ``em_iid``, ``f1_iid``,
            ``em_comp``, ``f1_comp``, ``em_zero``, ``f1_zero``.
        """
        # Original source: Pangu_GrailQA_main()
        with open(gold_file) as f:
            data = json.load(f)
        with open(pred_file) as f:
            predict = json.load(f)

        em_sum, f1_sum = 0.0, 0.0
        level_count: Dict[str, int] = defaultdict(lambda: 0)
        level_em_sum: Dict[str, float] = defaultdict(lambda: 0.0)
        level_f1_sum: Dict[str, float] = defaultdict(lambda: 0.0)

        for item in data:
            level_count[item['level']] += 1

            answer = set()
            if item['answer'] != 'null':
                for a in item['answer']:
                    answer.add(a['answer_argument'])

            if str(item['qid']) in predict:
                pred_item = predict[str(item['qid'])]
                em = self._matcher.same_logical_form(
                    pred_item['logical_form'], item['s_expression'])
                em_sum += em
                level_em_sum[item['level']] += em

                if em:
                    f1_sum += 1.0
                    level_f1_sum[item['level']] += 1.0
                else:
                    predict_answer = set(pred_item['answer'])
                    if len(predict_answer.intersection(answer)) != 0:
                        precision = (len(predict_answer.intersection(answer))
                                     / len(predict_answer))
                        recall = (len(predict_answer.intersection(answer))
                                  / len(answer))
                        f1_val = (2 * recall * precision / (recall + precision))
                        f1_sum += f1_val
                        level_f1_sum[item['level']] += f1_val

        n = len(data)
        stats: Dict[str, Any] = {
            'em': em_sum / n,
            'f1': f1_sum / n,
            'em_iid': level_em_sum['i.i.d.'] / level_count['i.i.d.'],
            'f1_iid': level_f1_sum['i.i.d.'] / level_count['i.i.d.'],
            'em_comp': (level_em_sum['compositional']
                        / level_count['compositional']),
            'f1_comp': (level_f1_sum['compositional']
                        / level_count['compositional']),
            'em_zero': level_em_sum['zero-shot'] / level_count['zero-shot'],
            'f1_zero': level_f1_sum['zero-shot'] / level_count['zero-shot'],
        }
        print(stats)
        return stats


# ====================================================================
#  GMT-KBQA / CWQ  —  Dataset.CWQ
# ====================================================================
#
#  GMT-KBQA Evaluation Script.
#  https://github.com/HXX97/GMT-KBQA/tree/main/generation
#
#  @File    :   cwq_evaluate.py
#  @Time    :   2022/01/10 16:14:58
#  @Author  :   Xixin Hu
#  @Version :   1.0
#  @Contact :   xixinhu97@foxmail.com
# ====================================================================


class GMTKBQAEvaluator(Evaluator):
    """Evaluator for **GMT-KBQA** on the **CWQ** (Complex WebQuestions) dataset.

    Original source:
    https://github.com/HXX97/GMT-KBQA/tree/main/generation

    File: ``generation/cwq_evaluate.py``
    @Author: Xixin Hu
    @Contact: xixinhu97@foxmail.com

    Corresponding function in the legacy script:
    ``cwq_evaluate_valid_results()``

    Note:
        For ``split='test'``, if a gold item lacks a pre-computed
        ``answer`` field, a live SPARQL query is executed via
        :class:`SparqlOdbcQuerierNoSexpr` (Freebase ODBC).
    """

    _dataset = Dataset.CWQ

    # ------------------------------------------------------------------
    #  Lazy imports (deferred so src heavy ML deps are optional)
    # ------------------------------------------------------------------

    _utils_loaded = False

    @classmethod
    def _ensure_utils(cls):
        """Lazy-import ``load_json`` / ``dump_json`` from
        :mod:`src.core.utils` (stored as staticmethods to avoid
        Python descriptor binding)."""
        if not cls._utils_loaded:
            from src.core.utils import load_json, dump_json
            cls._load_json = staticmethod(load_json)
            cls._dump_json = staticmethod(dump_json)
            cls._utils_loaded = True

    # ------------------------------------------------------------------

    def __init__(self, split: str = 'dev',
                 sparql_wrapper_url: Optional[str] = None):
        """
        Args:
            split: One of ``'train'``, ``'dev'``, ``'test'``.  Determines
                how ground-truth answers are extracted from the gold file.
            sparql_wrapper_url: SPARQL HTTP endpoint URL for Freebase.
                Defaults to ``http://210.28.134.34:8890/sparql`` (official).
        """
        self._split = split
        self._sparql_wrapper_url = sparql_wrapper_url
        self._executor = None  # type: ignore[assignment]

    def _get_executor(self):
        """Lazy-init a minimal SPARQLWrapper executor.

        Uses :class:`SPARQLWrapper` directly to avoid the heavy
        ``src`` import chain (scipy / sklearn / NumPy).
        """
        if self._executor is None:
            from SPARQLWrapper import SPARQLWrapper, JSON

            url = (self._sparql_wrapper_url
                   or "http://210.28.134.34:8890/sparql")

            class _MinimalFreebaseExecutor:
                """Thin wrapper around SPARQLWrapper for Freebase queries."""
                PREFIX = "PREFIX ns: <http://rdf.freebase.com/ns/> "

                def __init__(self, endpoint_url):
                    self._sw = SPARQLWrapper(endpoint_url)
                    self._sw.setReturnFormat(JSON)
                    self._sw.setTimeout(30)

                def execute_query(self, query):
                    self._sw.setQuery(f"{self.PREFIX} {query}")
                    bindings = self._sw.query().convert()['results']['bindings']
                    # Strip Freebase NS prefix so results are bare MIDs
                    # matching the prediction answer format.
                    cleaned = []
                    for b in bindings:
                        cb = {}
                        for k, v in b.items():
                            val = v.get('value', '')
                            if val.startswith('http://rdf.freebase.com/ns/'):
                                val = val[len('http://rdf.freebase.com/ns/'):]
                            cb[k] = {**v, 'value': val}
                        cleaned.append(cb)
                    return cleaned

            self._executor = _MinimalFreebaseExecutor(url)
        return self._executor

    # ------------------------------------------------------------------
    #  Main evaluation
    # ------------------------------------------------------------------

    def evaluate(self, gold_file: str, pred_file: str,
                 max_queries: Optional[int] = None) -> Dict[str, Any]:
        """Compute CWQ evaluation metrics (P, R, F1, Accuracy).

        Gold format: list of ``{"ID": ..., "sparql": ...}`` (test split
        requires live SPARQL execution for each gold item).

        Pred format: list of ``{"qid": ..., "answer": [...]}``.

        Args:
            gold_file: Path to gold dataset.
            pred_file: Path to predictions.
            max_queries: If set, limit SPARQL queries to this many
                (for smoke testing).

        Returns:
            Dict with keys: ``total``, ``accuracy``, ``avg_precision``,
            ``avg_recall``, ``avg_f1``.
        """
        # Original source: cwq_evaluate_valid_results()
        self._ensure_utils()
        pred_data = self._load_json(pred_file)
        dataset_data = self._load_json(gold_file)

        if max_queries is not None:
            dataset_data = dataset_data[:max_queries]

        dataset_dict = {x["ID"]: x for x in dataset_data}

        p_list: List[float] = []
        r_list: List[float] = []
        f_list: List[float] = []
        p_dict: Dict[str, float] = {}
        r_dict: Dict[str, float] = {}
        f_dict: Dict[str, float] = {}
        acc_num = 0

        pred_dict: Dict[str, set] = {}
        acc_qid_list: List[str] = []
        for pred in pred_data:
            qid = pred['qid']
            pred_answer = set(pred['answer'])
            pred_dict[qid] = pred_answer

        gt_answer_record: Dict[str, List[str]] = {}

        for qid, example in tqdm(dataset_dict.items()):
            gt_sparql = example['sparql']
            if self._split == 'test':
                if 'answer' in example:
                    gt_answer = set(example['answer'])
                else:
                    executor = self._get_executor()
                    # execute_query returns SPARQL JSON bindings;
                    # extract answer values into a set of strings.
                    bindings = executor.execute_query(gt_sparql)
                    gt_answer = set()
                    for b in bindings:
                        for v in b.values():
                            gt_answer.add(v.get('value', str(v)))
            elif self._split in ('train', 'dev'):
                gt_answer = set(
                    [item["answer_id"] for item in example["answers"]]
                )

            # Store ground-truth answers for _with_answers.json output
            gt_answer_record[qid] = sorted(gt_answer)

            pred_answer = set(pred_dict.get(qid, {}))

            if pred_answer == gt_answer:
                acc_num += 1
                acc_qid_list.append(qid)

            if len(pred_answer) == 0:
                if len(gt_answer) == 0:
                    p = 1.0
                    r = 1.0
                    f = 1.0
                else:
                    p = 0.0
                    r = 0.0
                    f = 0.0
            elif len(gt_answer) == 0:
                p = 0.0
                r = 0.0
                f = 0.0
            else:
                p = len(pred_answer & gt_answer) / len(pred_answer)
                r = len(pred_answer & gt_answer) / len(gt_answer)
                f = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

            p_list.append(p)
            r_list.append(r)
            f_list.append(f)
            p_dict[qid] = p
            r_dict[qid] = r
            f_dict[qid] = f

        total = len(p_list)
        p_average = sum(p_list) / total
        r_average = sum(r_list) / total
        f_average = sum(f_list) / total
        acc = acc_num / total

        res = (f"Total: {total}, ACC:{acc}, "
               f"AVGP: {p_average}, AVGR: {r_average}, AVGF: {f_average}")
        print(res)

        # Write evaluation summary file (matching original behaviour)
        dirname = os.path.dirname(pred_file)
        filename = os.path.basename(pred_file)
        with open(os.path.join(dirname,
                               f'{filename}_final_eval_results.txt'), 'w') as f:
            f.write(res)
            f.flush()

        # Augment predictions with per-question metrics + ground-truth answers
        for pred in pred_data:
            qid = pred['qid']
            pred['answer_acc'] = qid in acc_qid_list
            pred['precision'] = p_dict.get(qid)
            pred['recall'] = r_dict.get(qid)
            pred['f1'] = f_dict.get(qid)
            if qid in gt_answer_record:
                pred['gt_answer'] = gt_answer_record[qid]

        self._dump_json(pred_data, os.path.join(dirname, f'{filename}_new.json'),
                       indent=4)
        # Also write a copy with _with_answers suffix (replace trailing .json)
        base_name = filename[:-5] if filename.endswith('.json') else filename
        self._dump_json(pred_data,
                        os.path.join(dirname, f'{base_name}_with_answers.json'),
                        indent=4)

        return {
            "total": total,
            "accuracy": acc,
            "avg_precision": p_average,
            "avg_recall": r_average,
            "avg_f1": f_average,
        }


# ====================================================================
#  Convenience: get the right evaluator by Dataset enum
# ====================================================================

def get_evaluator(dataset: Dataset, **kwargs) -> Evaluator:
    """Factory: return an :class:`Evaluator` instance for *dataset*.

    Args:
        dataset: A :class:`Dataset` enum value.
        **kwargs: Passed through to the evaluator constructor
            (e.g. ``split`` for CWQ; ``fb_roles``, ``fb_types``,
            ``reverse_properties`` for GrailQA).

    Returns:
        An :class:`Evaluator` subclass instance.

    Raises:
        ValueError: If *dataset* is not supported.
    """
    if dataset == Dataset.WEBQ:
        return PanguWebQSPEvaluator(**kwargs)
    elif dataset == Dataset.GRAIL:
        return PanguGrailQAEvaluator(**kwargs)
    elif dataset == Dataset.CWQ:
        return GMTKBQAEvaluator(**kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


# ====================================================================
#  Batch evaluation — run with:  python kbqa_performance_evaluation.py
# ====================================================================

# -- Paths ------------------------------------------------------------------
_BASE = "/home5/yhbao/git/SPARQL_query_annotating_task/data/output/kbqa_performance_experiment"
_GOLD_DIR = os.path.join(_BASE, "original_dataset")
_PRED_DIR = os.path.join(_BASE, "method_prediction_results")


def _write_result(out_path: str, title: str, metrics: Dict[str, Any]) -> None:
    """Write evaluation metrics to a simple text file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(f"{title}\n")
        f.write("=" * 50 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.flush()
    print(f"  -> wrote {out_path}\n")


# -------------------------------------------------------------------
#  Pangu / WebQSP
# -------------------------------------------------------------------

def eval_webqsp_pangu() -> None:
    """Evaluate Pangu predictions on WebQSP (quad & quadg variants)."""
    gold = os.path.join(_GOLD_DIR, "webqsp_test.json")
    evaluator = PanguWebQSPEvaluator()

    for variant in ["webqsp_pangu_with_quad", "webqsp_pangu_with_quadg"]:
        pred = os.path.join(_PRED_DIR, variant, "predictions_for_evaluation.json")
        out = os.path.join(_PRED_DIR, variant, "evaluation_result")
        if not os.path.exists(pred):
            print(f"[SKIP] {variant}: prediction file not found ({pred})")
            continue
        print(f"\n[Pangu WebQSP] {variant}")
        metrics = evaluator.evaluate(gold, pred)
        _write_result(out, f"Pangu WebQSP Evaluation — {variant}", metrics)


# -------------------------------------------------------------------
#  Pangu / GrailQA
# -------------------------------------------------------------------

def eval_grailqa_pangu() -> None:
    """Evaluate Pangu predictions on GrailQA (quad & quadg variants)."""
    gold = os.path.join(_GOLD_DIR, "grailqa_v1.0_dev.json")
    evaluator = PanguGrailQAEvaluator()  # uses hardcoded FB_ontology paths

    for variant in ["grailqa_pangu_with_quad", "grailqa_pangu_with_quadg"]:
        pred = os.path.join(_PRED_DIR, variant, "predictions_for_evaluation.json")
        out = os.path.join(_PRED_DIR, variant, "evaluation_result")
        if not os.path.exists(pred):
            print(f"[SKIP] {variant}: prediction file not found ({pred})")
            continue
        print(f"\n[Pangu GrailQA] {variant}")
        metrics = evaluator.evaluate(gold, pred)
        _write_result(out, f"Pangu GrailQA Evaluation — {variant}", metrics)


# -------------------------------------------------------------------
#  GMT-KBQA / CWQ
# -------------------------------------------------------------------
#
#  Two evaluation modes are supported:
#
#  1. **Standalone** — use :meth:`GMTKBQAEvaluator.evaluate(gold, pred)`.
#     This runs the full pipeline: loads gold + pred, executes live SPARQL
#     queries for each gold item (test split has no pre-computed answers),
#     computes P / R / F1 / Accuracy, and writes per-question metrics to a
#     ``*_new.json`` file.  Accurate but requires a reachable Freebase SPARQL
#     endpoint and takes ~1 hour for 3500 queries.
#
#  2. **Aggregation** (default below) — the ``*_new.json`` files already
#     contain per-question ``precision`` / ``recall`` / ``f1`` / ``answer_acc``
#     fields from a previous standalone run.  We simply aggregate them to
#     obtain the final metrics.  Instantaneous.
#
#  The default path is aggregation.  To recompute from scratch, call
#  ``GMTKBQAEvaluator(split="test").evaluate(gold, pred)`` directly.
# -------------------------------------------------------------------

def eval_cwq_gmtkbqa() -> None:
    """Evaluate GMT-KBQA predictions on CWQ — aggregation mode (default)."""
    import json

    for variant in ["cwq_gmtkbqa_with_quad", "cwq_gmtkbqa_with_quadg"]:
        pred = os.path.join(_PRED_DIR, variant,
                            "beam_50_test_4_top_k_predictions.json_gen_sexpr_results.json_new.json")
        out = os.path.join(_PRED_DIR, variant, "evaluation_result")
        if not os.path.exists(pred):
            print(f"[SKIP] {variant}: prediction file not found ({pred})")
            continue

        print(f"\n[GMT-KBQA CWQ] {variant}  (aggregation mode)")
        pred_data = json.load(open(pred))
        total = len(pred_data)
        acc_num = sum(1 for p in pred_data if p.get('answer_acc'))
        p_list = [p['precision'] for p in pred_data if p.get('precision') is not None]
        r_list = [p['recall'] for p in pred_data if p.get('recall') is not None]
        f_list = [p['f1'] for p in pred_data if p.get('f1') is not None]

        p_avg = sum(p_list) / len(p_list) if p_list else 0.0
        r_avg = sum(r_list) / len(r_list) if r_list else 0.0
        f_avg = sum(f_list) / len(f_list) if f_list else 0.0
        acc = acc_num / total if total else 0.0

        res = (f"Total: {total}, ACC: {acc:.4f}, "
               f"AVGP: {p_avg:.4f}, AVGR: {r_avg:.4f}, AVGF: {f_avg:.4f}")
        print(res)

        metrics = {
            "total": total, "accuracy": round(acc, 4),
            "avg_precision": round(p_avg, 4), "avg_recall": round(r_avg, 4),
            "avg_f1": round(f_avg, 4),
        }
        _write_result(out, f"GMT-KBQA CWQ Evaluation — {variant}", metrics)


# -------------------------------------------------------------------
#  GMT-KBQA / CWQ — standalone (live SPARQL)
# -------------------------------------------------------------------

def eval_cwq_gmtkbqa_standalone(max_queries: Optional[int] = None) -> None:
    """Evaluate GMT-KBQA predictions on CWQ with live SPARQL queries.

    Runs the full pipeline: executes SPARQL for each gold item to obtain
    ground-truth answers, then computes P / R / F1 / Accuracy.  Results
    are written to ``*_new.json``, ``*_with_answers.json``, and
    ``evaluation_result``.

    Args:
        max_queries: If set, process only this many gold items (for smoke
            testing).  Default ``None`` means full evaluation (~3500 queries,
            ~1-2 min at ~50 q/s).
    """
    gold = os.path.join(_GOLD_DIR, "ComplexWebQuestions_test.json")
    evaluator = GMTKBQAEvaluator(split="test")

    for variant in ["cwq_gmtkbqa_with_quad", "cwq_gmtkbqa_with_quadg"]:
        # Use the *_new.json which has the prediction data
        pred = os.path.join(_PRED_DIR, variant,
                            "beam_50_test_4_top_k_predictions.json_gen_sexpr_results.json_new.json")
        out = os.path.join(_PRED_DIR, variant, "evaluation_result")
        if not os.path.exists(pred):
            print(f"[SKIP] {variant}: prediction file not found ({pred})")
            continue
        print(f"\n[GMT-KBQA CWQ] {variant}  (standalone mode, max_queries={max_queries})")
        try:
            metrics = evaluator.evaluate(gold, pred, max_queries=max_queries)
            _write_result(out, f"GMT-KBQA CWQ Evaluation — {variant}", metrics)
        except Exception as ex:
            print(f"  [ERROR] {ex}")


# -------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("KBQA Performance Evaluation — Batch Run")
    print(f"Gold data : {_GOLD_DIR}")
    print(f"Pred data : {_PRED_DIR}")
    print("=" * 60)

    # print("\n>>> Pangu / WebQSP")
    # eval_webqsp_pangu()

    # print(">>> Pangu / GrailQA")
    # eval_grailqa_pangu()

    # Uncomment the line below to run full standalone CWQ evaluation
    # (live SPARQL queries, ~3500 gold items, ~1-2 min)
    eval_cwq_gmtkbqa_standalone()


    print("Done.")
