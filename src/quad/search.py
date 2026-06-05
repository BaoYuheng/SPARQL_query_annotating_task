import math
from src.logical_form.simple_graph import SimpleGraph, NodeType, EdgeType, Node, get_attached_graph, rename_node
from src.linking.semantic_sim import FBEmbeddingBasedSimilarity, PLMSimRanker
from src.sparql.executor import SparqlOdbcQuerierNoSexpr, SparqlOdbcQuerierNoSexprWikidata
import re
import time
from src.core.utils import (
    load_json, dump_json, setup_custom_logger, flatten_decomposition_tree
)
from src.sparql.sparql_utils import load_relation_dicts
from src.logical_form.s_expression_utils import sexp_to_sparql
from src.core.common import (SPARQL_wrapper_path_official,
                        KB_TYPE,
                        FREEBASE_CONSTANT_TYPE,
                        WIKIDATA_CONSTANT_TYPE,
                        Dataset, FB_DATASETS,
                        FreebaseConstantForConstruction,
                        WikidataConstantForConstruction,
                        SPARQL_wrapper_path_dkilab,
                        SPARQL_wrapper_path_wikidata_2023,
                        ODBC_CONFIG_OFFICIAL,
                        ODBC_CONFIG_WIKIDATA_2023,
                        ODBC_CONFIG_DKILAB,
                        DEFAULT_SPARQL_SERVICE)
import random 
from queue import Queue
import itertools
random.seed(1)
from copy import deepcopy
from func_timeout import func_set_timeout
import func_timeout
import traceback
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import faulthandler
import gc
import os
import datetime
from tqdm import tqdm


def has_token_overlap(sen_1, sen_2):
    """
    Check if two sentences have lexical overlap after stopword removal.
    判断两个句子在去除停用词后是否存在词汇重叠。

    Replaces punctuation with spaces, tokenizes, ignores stopwords,
    and returns True if any non-stopword token from sen_1 appears in sen_2.
    将标点替换为空格后分词，忽略停用词，若句1的某个非停用词出现在句2中则返回True。
    """
    nltk_stop_words = {i:True for i in ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", 
                "you", "your", "yours", "yourself", "yourselves", "he", "him", 
                "his", "himself", "she", "her", "hers", "herself", "it", "its", 
                "itself", "they", "them", "their", "theirs", "themselves", "what", 
                "which", "who", "whom", "this", "that", "these", "those", "am", "is", 
                "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", 
                "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because",
                    "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
                    "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
                        "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
                        "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", 
                        "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", 
                        "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]}

    delimiters = [',', '.', ';', ':', '#', "_", "-", "?"]
    sen_1 = sen_1.lower()
    sen_2 = sen_2.lower()
    for d in delimiters:
        sen_1 = sen_1.replace(d, " ")
        sen_2 = sen_2.replace(d, " ")
    tokens_1 = sen_1.lower().split()
    tokens_2 = sen_2.lower().split()
    for t in tokens_1:
        if t in nltk_stop_words:
            continue
        if t in tokens_2:
            return True
    else:
        return False


# 问题分解的树状结构 / Tree node for question decomposition (Sec 5.2 of paper)
class DecompositionTreeNode:
    def __init__(self, question, index):
        self.this_question = question
        self.this_LF_list = []
        self.sub_question_decomposition_nodes = []
        self.parent_question_decomposition_node = None
        self.index = index

        def __str__(self):
            return self.this_question.question_text
        
        def __repr__(self):
            return self.__str__()

class Question:
    def __init__(self, question_text, answers=None, linked_items = None) -> None:
        """
        question_text: 问题文本 / the question text
        answers: 问题答案（实体列表或数字） / answer entities or numeric values
        linked_items: 候选的关系或实体 / candidate entities or relations from entity linking
        """
        self.question_text = question_text
        self.candidate_LF = []
        self.decompostion_dependency = {}
        if answers is not None:
            self.answers = answers
        if linked_items is not None:
            self.linked_items = linked_items
    
    def __str__(self):
        return self.question_text
    
    def __repr__(self):
        return self.__str__()


    def build_subquestions_by_decomposition(self, use_golden_entity):
        '''
        Build the decomposition tree from {question: [dependent sub-questions]} dict.
        根据 {问题: [依赖的子问题]} 字典来构造问题的分解树。 (Sec 5.2)
        '''

        def has_overlap(item, question_text_wo_idx):
            has_overlap = False
            if item['type'] == 'entity':
                has_overlap = has_token_overlap(item['label'], subq_text_wo_idx)\
                    or "mention" in item and has_token_overlap(item['mention'], question_text_wo_idx)
            elif item['type'] == "class":
                has_overlap = has_token_overlap(item['label'], subq_text_wo_idx) \
                    or "mention" in item and has_token_overlap(item['mention'], question_text_wo_idx)
            elif item['type'] == "literal":
                has_overlap = has_token_overlap(item['label'], subq_text_wo_idx)
            else:
                assert(0)
            return has_overlap

        q_to_decom_node_dict = {}
        self.sub_questions = []
        for k in self.decompostion_dependency.keys():
            subq_text = k
            subq_text_wo_idx = subq_text.split("|")[1]
            subq_answers = self.answers if k.startswith("<OUTQ>") else []
            if len(self.decompostion_dependency) == 1:    # 只有一个子问题，使用全部 linked items / only one sub-q, use all linked items
                subq_linked_items = self.linked_items
            else:
                if use_golden_entity:
                    # 使用 golden entities 时不要求 mention 在 subq 内 / golden: don't require mention overlap for entities
                    subq_linked_items = [ent for ent in self.linked_items if ent['type'] != "literal"]
                    subq_linked_items += [ent for ent in self.linked_items if ent['type'] == 'literal' and has_overlap(ent, subq_text_wo_idx)]
                else:
                    subq_linked_items = [item for item in self.linked_items if has_overlap(item, subq_text_wo_idx)]
                    subq_linked_items = [item for item in self.linked_items]
            sub_question = Question(subq_text, subq_answers, subq_linked_items)
            sub_question_index = subq_text.split("|")[0]
            q_to_decom_node_dict[k]=DecompositionTreeNode(sub_question, sub_question_index)
        for k, v in self.decompostion_dependency.items():
            if len(v) > 0:
                for q in v:
                    q_to_decom_node_dict[q].parent_question_decomposition_node = q_to_decom_node_dict[k]
                    q_to_decom_node_dict[k].sub_question_decomposition_nodes.append(q_to_decom_node_dict[q])
        root_node = [node for node in q_to_decom_node_dict.values() if node.parent_question_decomposition_node is None][0]
        self.sub_question_structure = root_node
        return 0

    def subq_identification(self, subq_node_to_shrink:DecompositionTreeNode, use_golden_entity):
        def get_decomposition_dependency_from_tree(root:DecompositionTreeNode):
            def traverse(current:DecompositionTreeNode, res:dict):
                res[current.this_question.question_text] = []
                for subq in current.sub_question_decomposition_nodes:
                    res[current.this_question.question_text].append(subq.this_question.question_text)
                for subq in current.sub_question_decomposition_nodes:
                    traverse(subq, res)
                return res
            res = {}
            traverse(root, res)
            return res
        
        # 将某个子问题收缩到父问题内 / Shrink a sub-question into its parent question (Sec 5.2: top-down decomposition)
        # 步骤：先修改树→重算dict→按新dict建树 / Steps: modify tree → recompute dict → rebuild tree
        subq_index, subq_text_wo_idx = subq_node_to_shrink.this_question.question_text.split("|")[0], subq_node_to_shrink.this_question.question_text.split("|")[1]
        parent = subq_node_to_shrink.parent_question_decomposition_node
        index = parent.sub_question_decomposition_nodes.index(subq_node_to_shrink)
        parent.sub_question_decomposition_nodes = parent.sub_question_decomposition_nodes[:index] + subq_node_to_shrink.sub_question_decomposition_nodes \
                                                    + parent.sub_question_decomposition_nodes[index+1:]
        parent.this_question.question_text = parent.this_question.question_text.replace(subq_index, subq_text_wo_idx)
        #使用修改后的树结构重新算dict，最方便
        self.decompostion_dependency = get_decomposition_dependency_from_tree(self.sub_question_structure)
        self.build_subquestions_by_decomposition(use_golden_entity)
        return 0

class LFSearchAlgorithm:

    def __init__(self, dataset: Dataset, log_file_dir, use_golden_entity:bool, use_neighbor_entity:bool, sparql_cache_dir = None):
        self.sparql_logger = setup_custom_logger(log_file_dir)
        self.dataset = dataset
        if dataset in FB_DATASETS:
            self.kb = KB_TYPE.FREEBASE
            SUBQ_GRAPH_MAX_SIZE = 4
            if dataset == Dataset.GRAIL:
                self.sparql_executor = SparqlOdbcQuerierNoSexpr(
                    sparql_wrapper_path=SPARQL_wrapper_path_dkilab,
                    logger=self.sparql_logger,
                    service=DEFAULT_SPARQL_SERVICE,
                    odbc_config=ODBC_CONFIG_DKILAB,
                    timeout=6,
                    sparql_cache_dir=sparql_cache_dir,
                )
                # PR 计算专用 executor：ODBC 的 SELECT COUNT(...) AS ?cnt 返回损坏的整数值
                # (e.g. 125 万亿 vs 正确值 27388)，导致 F1 全为零。PR 必须走 SPARQLWrapper。
                self.counting_executor = SparqlOdbcQuerierNoSexpr(
                    sparql_wrapper_path=SPARQL_wrapper_path_dkilab,
                    logger=self.sparql_logger,
                    service="sparql_wrapper",
                    timeout=6,
                    sparql_cache_dir=sparql_cache_dir,
                )
            elif dataset == Dataset.WEBQ or dataset == Dataset.CWQ:
                # CWQ / WebQSP 使用 official (8890) 端点，因为 dkilab (8896) 中日期值格式为
                # "1975-04-30-08:00" (带时区 datetime)，与答案数据中的纯 date 格式
                # "1975-04-30"^^xsd:date 不匹配，导致时间类问题答案锚定搜索失败。
                # Official 端点存储纯日期格式，与答案数据一致。
                # CWQ / WebQSP use official (8890) endpoint because dkilab (8896) stores
                # dates as "1975-04-30-08:00" (datetime with timezone), which doesn't match
                # the pure-date answer format "1975-04-30"^^xsd:date, causing answer-anchored
                # search failures for time-related questions. Official uses pure dates.
                self.sparql_executor = SparqlOdbcQuerierNoSexpr(
                    sparql_wrapper_path=SPARQL_wrapper_path_official,
                    logger=self.sparql_logger,
                    service=DEFAULT_SPARQL_SERVICE,
                    odbc_config=ODBC_CONFIG_OFFICIAL,
                    timeout=6,
                    sparql_cache_dir=sparql_cache_dir,
                )
                # PR 计算专用 executor：ODBC COUNT 损坏，必须走 SPARQLWrapper. 详见 GrailQA 分支注释.
                self.counting_executor = SparqlOdbcQuerierNoSexpr(
                    sparql_wrapper_path=SPARQL_wrapper_path_official,
                    logger=self.sparql_logger,
                    service="sparql_wrapper",
                    timeout=6,
                    sparql_cache_dir=sparql_cache_dir,
                )
            self.relation_blacklists = ["base.yupgrade", "base.fbontology.", "common.webpage", "freebase.valuenotation"]
            self.FB_time_types = ['type.datetime']
            self.FB_quantity_types = ['type.int', "type.float"]
            self.reverse_relation_dict, self.relation_domain_range_dict = load_relation_dicts()

            # ============================================================
            # HYPERPARAMETERS (Freebase) / 超参数 (Freebase)
            # Tune these to trade off search quality vs. runtime.
            # ============================================================
            self.ANSWER_SAMPLE_SIZE = 3               # max answer samples for reachability check / 最多采样答案数
            self.CQV_TO_NQV_CANDIDATE_NUM = 10        # top-K relations from CQV to NQV / CQV→NQV 候选关系数
            self.CQV_QUANTITY_R_CANDIDATE_NUM = 5     # top-K quantity properties / 数量属性候选数
            self.CQV_TEMPORAL_R_CANDIDATE_NUM = 5     # top-K temporal properties / 时间属性候选数
            self.ENTITY_CANDIDATE_NUM = 3             # max neighbor entities to retain / 邻居实体保留数
            self.V_TO_ENTITY_CANDIDATE_NUM = 5        # top-K relations from variable to entity / 变量→实体关系 TOPK
            self.SUBQ_GRAPH_MAX_SIZE = 4              # max edges per sub-question subgraph / 每个子问题子图最大边数
            self.SUBQ_CANDIDATE_GRAPHS_NUM_NON_LEAF = 20  # beam size for non-leaf sub-questions / 非叶子子问题候选图数
            self.SUBQ_CANDIDATE_GRAPHS_NUM_LEAF = 10      # beam size for leaf sub-questions / 叶子子问题候选图数

            # 20260121: active timeout limits (see paper Sec 5.3 Implementation) / 启用超时限制
            self.SUBGRAPH_SEARCH_TIMEOUT = 20    # relation & entity search phase timeout (s) / 搜索阶段超时
            self.SEARCH_FAIL_TIMEOUT = 120       # whole question search timeout (s) / 整体搜索超时
            self.ENUMERATION_PHASE_TIMEOUT = 10  # enumeration phase timeout (s) / 枚举阶段超时
            # 20260121: relaxed limits (commented out) / 宽松限制（已禁用）
            # self.SUBGRAPH_SEARCH_TIMEOUT = 2000
            # self.SEARCH_FAIL_TIMEOUT = 1200
            # self.ENUMERATION_PHASE_TIMEOUT = 1000

            self.CQV_MAX_DEG = 2   # max degree of current question variable / CQV 最大度数
            self.SEARCH_TOPK = 10  # top-K final results retained / 最终保留结果数
        else:
            self.kb = KB_TYPE.WIKIDATA
            self.sparql_executor = SparqlOdbcQuerierNoSexprWikidata(
                sparql_wrapper_path=SPARQL_wrapper_path_wikidata_2023,
                logger=self.sparql_logger,
                service=DEFAULT_SPARQL_SERVICE,
                odbc_config=ODBC_CONFIG_WIKIDATA_2023,
                timeout=10,
                sparql_cache_dir=sparql_cache_dir,
            )
            # PR 计算专用 executor：ODBC COUNT 损坏，必须走 SPARQLWrapper. 详见 Freebase 分支注释.
            self.counting_executor = SparqlOdbcQuerierNoSexprWikidata(
                sparql_wrapper_path=SPARQL_wrapper_path_wikidata_2023,
                logger=self.sparql_logger,
                service="sparql_wrapper",
                timeout=10,
                sparql_cache_dir=sparql_cache_dir,
            )
            numeral_p_data = load_json("data/WD_ontology/wikidata_numeral_property_dict.json")
            self.WD_quantity_properties_dict = {item['pid']:item for item in numeral_p_data['quantity_properties']}
            self.WD_temporal_properties_dict = {item['pid']:item for item in numeral_p_data['temporal_properties']}
            self.WD_property2label_dict = load_json("data/WD_ontology/wikidata_property_id2label_dict.json")
            self.relation_blacklists = ['P31']
            #self.relation_blacklists = []

            # ============================================================
            # HYPERPARAMETERS (Wikidata) / 超参数 (Wikidata)
            # Tune these to trade off search quality vs. runtime.
            # ============================================================
            self.ANSWER_SAMPLE_SIZE = 3               # max answer samples for reachability check / 最多采样答案数
            self.CQV_TO_NQV_CANDIDATE_NUM = 10        # top-K relations from CQV to NQV / CQV→NQV 候选关系数
            self.CQV_QUANTITY_R_CANDIDATE_NUM = 3     # top-K quantity properties / 数量属性候选数
            self.CQV_TEMPORAL_R_CANDIDATE_NUM = 3     # top-K temporal properties / 时间属性候选数
            self.ENTITY_CANDIDATE_NUM = 3             # max neighbor entities to retain / 邻居实体保留数
            self.V_TO_ENTITY_CANDIDATE_NUM = 5        # top-K relations from variable to entity / 变量→实体关系 TOPK
            self.SUBQ_GRAPH_MAX_SIZE = 6              # max edges per sub-question subgraph / 每个子问题子图最大边数
            self.SUBQ_CANDIDATE_GRAPHS_NUM_NON_LEAF = 20  # beam size for non-leaf sub-questions / 非叶子子问题候选图数
            self.SUBQ_CANDIDATE_GRAPHS_NUM_LEAF = 10      # beam size for leaf sub-questions / 叶子子问题候选图数
            self.SUBGRAPH_SEARCH_TIMEOUT = 60    # relation & entity search phase timeout (s) / 搜索阶段超时
            self.SEARCH_FAIL_TIMEOUT = 180       # whole question search timeout (s) / 整体搜索超时
            self.ENUMERATION_PHASE_TIMEOUT = 60  # enumeration phase timeout (s) / 枚举阶段超时
            self.CQV_MAX_DEG = 2   # max degree of current question variable / CQV 最大度数
            self.SEARCH_TOPK = 10  # top-K final results retained / 最终保留结果数
        self.use_golden_entity = use_golden_entity
        self.use_neighbor_entity = use_neighbor_entity
        self.sentence_bert_ranker = PLMSimRanker(model="BAAI/bge-reranker-v2-m3")
        self.global_mid2item_dict = {}

    # def get_current_graph_PR_Sexpr(self, graph:SimpleGraph, answers):
    #     sexpr = graph.to_Sexpr()
    #     sparql = sexp_to_sparql(str(sexpr))
    #     results_set = set(self.sparql_executor.get_execution_result_one_variable(sparql))
    #     answer_mids = set([item['mid'] for item in answers])
    #     if len(results_set) == 0:
    #         precision = 0
    #         recall = 0
    #     else:
    #         recall = len(results_set.intersection(answer_mids)) / len(answer_mids)
    #         precision = len(results_set.intersection(answer_mids)) / len(results_set)
    #     return precision, recall
        
    def get_parameters_dict(self):
        parameter_dict = {key: value for key, value in vars(self).items() if isinstance(value, int) or isinstance(value, list) or isinstance(value, bool)}
        return parameter_dict

    def get_current_graph_PR(self, graph:SimpleGraph, answers):
        """
        Execute current graph as SPARQL and compute Precision & Recall against answer set.
        将当前图转为 SPARQL 执行，计算相对于答案集合的 Precision 与 Recall。(Sec 5.3: pruning via monotonic property)
        Uses self.counting_executor (SPARQLWrapper) because ODBC returns corrupted integer values
        for SELECT COUNT(...) AS ?cnt queries. / 使用 self.counting_executor，因 ODBC COUNT 返回损坏值.
        """
        ex = self.counting_executor  # always SPARQLWrapper for correct COUNT values

        if graph.count_query:
            sparql = graph.to_sparql_query()
            results_set = set(ex.get_execution_result_one_variable(sparql))
            assert len(answers) == 1 and "XMLSchema#integer" in answers[0]['mid']
            match = re.search(r'(\d+)', answers[0]['mid'])
            answers_number = int(match.group(1))
            result_number = int(list(results_set)[0])
            if result_number == 0:
                recall = 0
                precision = 0
            else:
                recall = 1
                precision = answers_number / result_number
        else:
            if graph.arg_function is None and len(answers) <= 10:
                p_sparql, r_sparql = graph.to_sparql_query_fast_PR_with_entity_answers(answers)
                recall = len(set(ex.get_execution_result_one_variable(r_sparql))) / len(answers)
                sparql_result = ex.get_execution_result_one_variable(p_sparql)
                if len(sparql_result) == 0:
                    recall = 0
                    precision = 0
                else:
                    sparql_result_size = int(list(ex.get_execution_result_one_variable(p_sparql))[0])
                    if sparql_result_size == 0:
                        precision = 0
                    else:
                        precision = len(answers) / sparql_result_size
            else:
                sparql = graph.to_sparql_query()
                results_set = set(ex.get_execution_result_one_variable(sparql))
                answer_mids = set([item['mid'] for item in answers])
                if len(results_set) == 0:
                    precision = 0
                    recall = 0
                else:
                    recall = len(results_set.intersection(answer_mids)) / len(answer_mids)
                    precision = len(results_set.intersection(answer_mids)) / len(results_set)
        return precision, recall

    def get_current_graph_PR_using_Sexpr(self, graph:SimpleGraph, answers):
        # 已废弃 / Deprecated: 使用 get_current_graph_PR 替代.
        # 保留以维持向后兼容 / Kept for backward compatibility.
        # Convert to S-expr then back to SPARQL for accurate check (adds implicit filters) / 转为 S-expr 再转回 SPARQL 增加隐式 Filter

        sparql = sexp_to_sparql(str(graph.to_Sexpr()))
        results_set = self.sparql_executor.get_execution_result_one_variable(sparql)        
        answer_mids = set([item['mid'] for item in answers])
        if len(results_set) == 0:
            precision = 0
            recall = 0
        else:
            recall = len(results_set.intersection(answer_mids)) / len(answer_mids)
            precision = len(results_set.intersection(answer_mids)) / len(results_set)
        return precision, recall

    def legal_candidate(self, graph:SimpleGraph, answers, prev_graph:SimpleGraph = None):
        if prev_graph is None:
            # if self.dataset == DATASET.GRAIL:
            #     p, r = self.get_current_graph_PR_Sexpr(graph, answers)
            # else:
            p, r = self.get_current_graph_PR(graph, answers)
        else:
            test_graph = get_attached_graph(prev_graph, graph)
            # if self.dataset == DATASET.GRAIL:
            #     p, r = self.get_current_graph_PR_Sexpr(test_graph, answers)
            # else:
            p, r = self.get_current_graph_PR(test_graph, answers)
        return (math.isclose(r, 1.0) and p <= 1.0)


    def accurate_query(self, graph:SimpleGraph, answers):
        # if self.dataset == DATASET.GRAIL:
        #     p, r = self.get_current_graph_PR_Sexpr(graph, answers)
        # else:
        p, r = self.get_current_graph_PR(graph, answers)
        #p, r = self.get_current_graph_PR_using_Sexpr(graph, answers)
        return (math.isclose(r, 1.0) and math.isclose(p, 1.0))

    def get_current_graph_F1(self, graph:SimpleGraph, answers):
        p, r = self.get_current_graph_PR(graph, answers)
        if p == 0 and r == 0:
            return 0
        else:
            return 2*p*r / (p+r)

    def get_special_property_type(self, r, KB:KB_TYPE):
        if KB == KB_TYPE.FREEBASE:
            if "/" in r:    # CVT relation type determined by the second part / CVT 关系类型由第二部分决定
                r = r.split("/")[1]
            if r in self.relation_domain_range_dict and 'range' in self.relation_domain_range_dict[r]:
                range = self.relation_domain_range_dict[r]['range']
            else:
                range = None
            if range is None:
                return "NORMAL"
            elif range in self.FB_time_types:
                return "TEMPORAL"
            elif range in self.FB_quantity_types:
                return "QUANTITY"
            else:
                return "NORMAL"
        elif KB == KB_TYPE.WIKIDATA:
            # Wikidata: 2-hop paths, type determined by r2 / Wikidata 两跳路径，类型由 r2 决定
            r = r.split("/")[1].split(":")[1]
            if r in self.WD_quantity_properties_dict:
                return "QUANTITY"
            elif r in self.WD_temporal_properties_dict:
                return "TEMPORAL"
            else:
                return "NORMAL"
        else:
            raise Exception("未实现")

    def remove_rebundant_tc(self, lf, answers):
        # Check if removing type constraints keeps results unchanged; if so, drop them / 检查移除 tc 后结果是否不变，是则删除
        tc = lf.type_constraints
        f1_1 = self.get_current_graph_F1(lf, answers)
        lf.type_constraints = []
        f1_2 = self.get_current_graph_F1(lf, answers)
        if f1_1 != f1_2:
             lf.type_constraints = tc
        return lf

    def detect_numeral_triggers_new(self, question_text):
        # Detect 3 trigger types: COUNT, COMPARE, ARG. Exact operator/arg type determined via enumeration.
        # 检测 3 种数值触发词：COUNT, COMPARE, ARG。具体比较符号与 ARG 类型通过穷举确定。(Sec 5.3: counting & superlatives)
        corpus_count = ['the count of', 'how many', 'how much', 'amount of', 'what number of', 'total number of', 'the number of', 'quantity of']
        corpus_arg = [' most ', ' first ', ' maximum ', ' minimum ' , ' max ', ' min ']
        corpus_compare = [' at most ', ' equal ', ' after ', ' under ', ' below ', ' within ', ' over ',' at least ', ' exceeds ', ' more ', " before ", " prior "]
        def detect_comparative_superlative(sentence):
            words = word_tokenize(sentence)
            tagged_words = pos_tag(words)

            comparatives = []
            superlatives = []

            for i in range(len(tagged_words) - 1):
                if tagged_words[i][1] == 'JJR':
                    comparatives.append(tagged_words[i][0])
                elif tagged_words[i][1] == 'JJS':
                    superlatives.append(tagged_words[i][0])

            return comparatives, superlatives
        triggers = []
        comparatives, superlatives = detect_comparative_superlative(question_text)
        count_triggers = []
        compare_triggers = []
        arg_triggers = []
        question_text = " " + question_text.lower().strip("?.")+ " "
        for c in corpus_count:
            if c in question_text:
                count_triggers.append(c)
        for c in corpus_compare:
            if c in question_text:
                compare_triggers.append(c)
        for c in corpus_arg:
            if c in question_text:
                arg_triggers.append(c)
        #对于WIKIDATA，需要考虑年份，而YEAR(X) = a这种结构需要探测in xxx, in year xx, in the year xxx, in xxxx year
        # if self.kb == KB_TYPE.WIKIDATA:
        #     has_year = len(re.findall(r"in \d+", question_text)) > 0 or len(re.findall(r"in year \d+", question_text)) > 0 \
        #         or len(re.findall(r"in the year \d+", question_text)) > 0 or len(re.findall(r"in \d+ year", question_text)) > 0
        # else:
        has_year = False
        if len(comparatives) > 0 or len(compare_triggers) or has_year:
            triggers.append("COMPARE")
        if len(count_triggers) > 0:
            triggers.append("COUNT")
        if len(superlatives) > 0 or len(arg_triggers) > 0:
            triggers.append("ARG")
        # Handle special case: 'least' vs 'at least' / 判断特殊情况：least vs at least
            if len(arg_triggers) == 1 and arg_triggers[0] == 'least' and 'at least' in compare_triggers:
                triggers.remove("ARG")
        return triggers

    
    def detect_numeral_triggers(self, question_text, linked_items):
            corpus_count = ['the count of', 'how many', 'how much', 'amount of', 'what number of', 'total number of', 'the number of', 'quantity of']
            corpus_argmax = [' last ', 'biggest', 'tallest', 'latest', 'maximum', ' most', 'heaviest', 'highest', 'max', 'largest', 'greatest', 'newest', 'fastest', 'longest', 'ultimate', 'brightest']
            corpus_argmin = ['lowest', 'least', 'first', 'smallest', 'shortest', 'minimum', 'earliest', 'nearest', 'skinniest', 'skinnest', 'closest', 'fewest', 'oldest', 'tiniest', 'lightest', 'newest', 'weakest', 'darkest']
            corpus_l = ['at most', 'less than', 'equal to or smaller', 'less than or equal to', 'less or equal to', 'smaller than', 'lower than', 'same as or less than', 'after', ' under ', 'below', 'fewer than', 'or less', 'lesser value than', 'shorter than', 'less that', 'within']
            corpus_g = ['greater or equal to', ' over ', 'or greater', 'at least', 'larger than', 'not sooner than', 'more than', 'more then', 'no less than', 'not less than','greater than', 'exceeds', 'more that', 'or more', 'fatter than', 'longer than', 'bigger than', 'no less', 'heavier than', 'no earlier than', 'larger amount than']
            corpus_le_final = ['not exceed', 'no more than', 'no later than', 'no greater than']
            
            
            def process_mention(mention):
                delimiters = [',', '.', ';', ':', '#', "_", "-"]
                for delimiter in delimiters:
                    parts = mention.split(delimiter)
                    parts = [p.strip() for p in parts]
                    if delimiter == '#':
                        mention = f' {delimiter}'.join(parts)
                    else:
                        mention = f'{delimiter} '.join(parts)
                
                return mention.lower()
            
            q = question_text.lower()
            func = []
            # 使用触发词的方式来检测各类特殊trigger
            for c in corpus_count:
                if c in q:
                    func.append("COUNT")
                    break
            for c in corpus_argmax:
                if c in q:
                    func.append("ARGMAX")
                    break
            for c in corpus_argmin:
                if c in q:
                    func.append("ARGMIN")
                    break
            for c in corpus_l:
                if c in q:
                    func.append("<=")
                    func.append("<")
                    break
            for c in corpus_g:
                if c in q:
                    func.append(">=")
                    func.append(">")
                    break
            # # compare和argmin/max可能覆盖应该是count的情况
            # if q.startswith('number of') or 'how many' in q or 'what is the number' in q:
            #     func = 'count'
            
            return func


    def LF_search_main(self, data_item, dataset: Dataset, only_keep_accurate = False):
        '''
        Main search entry point — top-down beam search over the decomposition tree.
        搜索主函数 — 基于分解树的自顶向下 Beam Search。(Sec 5.3: Query Graph Construction)
        '''
        def init_search_array(question_main):
            nonlocal previous_LFs
            nonlocal search_array
            search_array.clear()
            previous_LFs.clear()
            search_array.append(question_main.sub_question_structure)
            # Common search starting point / 公共搜索起点
            numeral_triggers = self.detect_numeral_triggers_new(question_main.question_text)
            # ARG and COUNT are mutually exclusive / ARG 与 COUNT 互斥
            assert("ARG" not in numeral_triggers or "COUNT" not in numeral_triggers)
            # COUNT: add search starting points with type constraints / COUNT: 为每个可能的 type 添加类型限定的搜索起点
            if "COUNT" in numeral_triggers and len(question_main.answers) == 1 and "integer" in question_main.answers[0]['mid']:
                # Add a type-constrained query for each possible class / 为每个可能的 type 添加类型限定的查询
                possible_types = [item['mid'] for item in question_main.linked_items if item['type'] == "class"]
                outq_node = Node("?OUTQ", NodeType.VARIABLE)
                for type in possible_types:
                    graph = SimpleGraph(outq_node, ground_kb=self.kb)
                    graph.count_query = True
                    graph.add_type_constaint(outq_node, type)
                    previous_LFs.append(graph)
                # Freebase lacks "number of" properties, but Wikidata has them / Freebase 没有 number of 属性，Wikidata 有
                if self.kb == KB_TYPE.WIKIDATA:
                    previous_LFs.append(None)
            else:
                previous_LFs.append(None)
        question_main = self.built_question_structure(data_item, dataset)
        # Drop randomness: take the first ANSWER_SAMPLE_SIZE answers / 抛弃随机性：取前 ANSWER_SAMPLE_SIZE 个答案
        self.selected_answers = question_main.answers if len(question_main.answers) <= self.ANSWER_SAMPLE_SIZE else question_main.answers[0:self.ANSWER_SAMPLE_SIZE]
        #self.selected_answers = question_main.answers if len(question_main.answers) <= ANSWER_SAMPLE_SIZE else random.sample(question_main.answers, ANSWER_SAMPLE_SIZE)
        self.global_mid2item_dict = {}
        # Separate accurate (F1=1) vs. non-accurate results. Prefer accurate; fallback to highest-F1 non-accurate.
        # 分离精确结果 (F1=1) 与非精确结果。优先精确结果，否则返回 F1 最高的非精确结果。
        accurate_final_results = []
        nonaccurate_final_results = []
        previous_LFs = []
        search_array = []
        # Solve sub-questions in topological order (BFS-like, top-down) / 按拓扑序自顶向下逐步求解（类 BFS）
        init_search_array(question_main)
        has_arg = "ARG" in self.detect_numeral_triggers_new(question_main.question_text)
        search_terminated = False
        search_begin_time = time.time()
        # BFS-like traversal; shrink sub-questions at the head of the queue when backtracking / 类 BFS 遍历，回溯时收缩队首子问题
        while not search_terminated:
            current_subq_node = search_array[0]
            sub_question_id = current_subq_node.index
            cqv = Node("?"+sub_question_id.strip("<>"), NodeType.VARIABLE)
            nqvs = [Node("?"+nqv, NodeType.VARIABLE) for nqv in [node.index.strip("<>") for node in current_subq_node.sub_question_decomposition_nodes]]
            for next_node in current_subq_node.sub_question_decomposition_nodes:
                search_array.append(next_node)
            subq = current_subq_node.this_question
            temp = []
            for lf1 in previous_LFs:
                if time.time() - search_begin_time > self.SEARCH_FAIL_TIMEOUT:
                    self.sparql_logger.info(f"Search timeout exceeded {str(self.SEARCH_FAIL_TIMEOUT)}s, aborting / 总搜索超时，放弃")
                    return [], 0
                try:
                    subq_LF=self.subquestion_graph_enumeration_with_timeout(subq, lf1, question_main.answers, cqv, nqvs)
                except Exception as e:
                    self.sparql_logger.info(traceback.format_exc())
                    subq_LF = []
                for lf2 in subq_LF:
                    if lf1 is None:
                        combined_lf = lf2
                    else:
                        lf1 = deepcopy(lf1)
                        combined_lf = get_attached_graph(lf1, lf2)
                    temp.append(combined_lf)
            previous_LFs = temp
            if len(previous_LFs) == 0:
                if current_subq_node.parent_question_decomposition_node is not None:
                    self.sparql_logger.info("Shrinking sub-question / 收缩子问题 =======================================================================")
                    question_main.subq_identification(current_subq_node, use_golden_entity=self.use_golden_entity)
                    init_search_array(question_main)
                    search_terminated = False
                    continue
            search_array = search_array[1:]
            if len(search_array) == 0: 
                self.sparql_logger.info("-------------------Search attempt finished / 搜索尝试完成----------------------------------------")
                # Enumerate ARG for each LF / 为每个 LF 枚举 ARG (Sec 5.3: superlatives)
                if has_arg:
                    numeral_neighbor_relations_of_outq = [r for r in self.get_neighbor_relations_with_previous_LF(answers=self.selected_answers, end_point=None, 
                                                                                                      expand_point=Node("?OUTQ", NodeType.VARIABLE), previous_LF=None, question=question_main.question_text)
                                                        if r['type'] == 'TEMPORAL' or r['type'] == 'QUANTITY'
                    ]
                    last_results = []
                    for LF in previous_LFs:
                        last_results.append(LF)
                        last_results+=(self.enumerate_arg(LF=LF, question_text=question_main.question_text, answers=self.selected_answers, candidate_relations=numeral_neighbor_relations_of_outq))
                else:
                    last_results = previous_LFs
                for lf in last_results:
                    # Final SPARQL must have at least one constant anchor; exceptions: ARG or comparisons without triple constants
                    # 最终 SPARQL 需要至少一个常量锚点；例外：有 ARG 或比较但无三元组常量
                    if not (lf.no_constant() and lf.arg_function is None and len(lf.comparisons) == 0):
                        #self.sparql_logger.info(str(lf))
                        f1 = self.get_current_graph_F1(lf, question_main.answers)
                        if math.isclose(f1, 1):
                            # Drop redundant type constraints if result stays accurate / 若删除 class 后仍精确则删除
                            if len(lf.edges) > 0:
                                    accurate_final_results.append(self.remove_rebundant_tc(lf, question_main.answers))
                            else:
                                # No edge but has TC: cannot remove TC (would leave no triple) / 没有 edge 而有 tc，不能删除
                                accurate_final_results.append(lf)
                        else:
                            if not only_keep_accurate:# Only track non-accurate when not requiring accuracy / 仅在不要求精确时才记录非精确结果
                                # Keep only the highest-F1 results / 只保存当前 F1 最高的结果
                                if len(nonaccurate_final_results) == 0:
                                    nonaccurate_final_results.append((self.remove_rebundant_tc(lf, question_main.answers), f1))
                                else:
                                    current_highest_f1 = nonaccurate_final_results[0][1]
                                    if f1 == current_highest_f1:
                                        nonaccurate_final_results.append((self.remove_rebundant_tc(lf, question_main.answers), f1))
                                    elif f1 > current_highest_f1:
                                        nonaccurate_final_results.clear()
                                        nonaccurate_final_results.append((self.remove_rebundant_tc(lf, question_main.answers), f1))
                if len(accurate_final_results) == 0 and current_subq_node.parent_question_decomposition_node is not None:
                    self.sparql_logger.info("Shrinking sub-question / 收缩子问题 =======================================================================")
                    question_main.subq_identification(current_subq_node, self.use_golden_entity)
                    init_search_array(question_main)
                    search_terminated = False
                else:
                    search_terminated = True
        final_results = []
        highest_f1 = 0
        if len(accurate_final_results) > 0:
            # Accurate results: keep all, rank by semantic similarity / 精确结果：保留并基于语义排序 (Sec 5.3: final selection)
            highest_f1 = 1
            accurate_final_results = self.get_current_graph_sim_top_k(accurate_final_results, question_main.question_text, self.global_mid2item_dict, top_k = self.SEARCH_TOPK)
            final_results = accurate_final_results
        else:
            # Fallback: rank highest-F1 non-accurate results by semantic similarity / 回退：对最高 F1 非精确结果基于语义排序
            if not only_keep_accurate and len(nonaccurate_final_results) > 0:
                highest_f1 = nonaccurate_final_results[0][1]
                results_with_highest_f1 = [item[0] for item in nonaccurate_final_results]
                results_with_highest_f1 = self.get_current_graph_sim_top_k(results_with_highest_f1, question_main.question_text, self.global_mid2item_dict, top_k=self.SEARCH_TOPK)
                final_results = results_with_highest_f1
        return final_results, highest_f1


    def get_current_graph_sim_top_k(self, graphs:list[SimpleGraph], sen, mid2constant_dict, top_k):
        #按照语义相似度，返回最可能的子图topk

        def get_naturalized_subgraph_representation(graph:SimpleGraph, mid2constant_dict):
            graph_str = ""
            if graph.count_query:
                graph_str += f"COUNT ({graph.center_node.value}) "
            else:
                graph_str += f"SELECT {graph.center_node.value} "
            if graph.arg_function is not None:
                if graph.arg_function['type'] == "ARGMAX":
                    graph_str += f" that has maximum {graph.arg_function['property']}, "
                else:
                    graph_str += f" that has minimum {graph.arg_function['property']}, "
            for type_constraint in graph.type_constraints:
                graph_str += f"{type_constraint['node'].value} is a {type_constraint['type']}, "
            for edge in graph.edges:
                if edge['from'].type == NodeType.ENTITY or edge['from'].type == NodeType.CLASS:
                    from_str = mid2constant_dict[edge['from'].value]['label']
                else:
                    from_str = edge['from'].value
                if edge['to'].type == NodeType.ENTITY or edge['to'].type == NodeType.CLASS:
                    to_str = mid2constant_dict[edge['to'].value]['label']
                else:
                    to_str = edge['to'].value
                edge_str = ""
                edge_str += " ( "
                edge_str += from_str + ", "
                #edge_str += edge['edge'].split(".")[-1] + ", "
                edge_str += edge['edge'] + ", "
                edge_str += to_str + " ), "
                graph_str += edge_str
            for comparison in graph.comparisons:
                operator_to_nl = {">=": "greater or equal", ">":"greater", "<":"smaller", "<=": "less or equal"}
                constraint_variable = comparison['node'].value
                comparison_str = f"{constraint_variable} has {comparison['property']} that {operator_to_nl[comparison['operator']]} than {comparison['value']}, " 
                graph_str += comparison_str
            return graph_str
        
        str2graph_dict = {get_naturalized_subgraph_representation(g, mid2constant_dict): g for g in graphs}
        naturalized_graphs = list(str2graph_dict.keys())
        sim_topk = self.sentence_bert_ranker.get_semantic_sim_topk(sen, naturalized_graphs, top_k)
        res = [str2graph_dict[s] for s in sim_topk]
        return res


    def get_ranked_relations(self, relations, text, top_k, subject:str, object:str):
        def get_natural_relation_rep_freebase(r, subject, object):
            if "/" in r:
                r1, r2 = r.split("/")[0], r.split("/")[1]
                return f"{subject} {r1} {r2} {object}"
            else:
                return f"{subject} {r} {object}"
            # else:
            #     if r in self.relation_domain_range_dict and 'label' in self.relation_domain_range_dict[r] :
            #         r_label = self.relation_domain_range_dict[r]['label']
            #         if reversed:
            #             return f"{object} {r} {subject}"
            #         else:
            #             return f"{subject} {r} {object}"
                # else:
                #     return None

        def get_natural_relation_rep_wikidata(r):
            # Subject and object labels not used here / 主语宾语标签在此不适用
            if "/" in r:
                r1, r2 = r.split("/")[0].split(":")[1], r.split("/")[1].split(":")[1]
                r1_label = self.WD_property2label_dict[r1]
                r2_label = self.WD_property2label_dict[r2]
                return f"{r1_label} / {r2_label}"
                
        results = []
        relation_rep_to_relation_dict = {}
        for r in relations:
            if self.kb == KB_TYPE.FREEBASE:
                r_rep = get_natural_relation_rep_freebase(r['relation'], subject, object)
            elif self.kb == KB_TYPE.WIKIDATA:
                r_rep = get_natural_relation_rep_wikidata(r['relation'])
            if r_rep is not None:
                if r_rep not in relation_rep_to_relation_dict:
                    relation_rep_to_relation_dict[r_rep] = [r]
                else:
                    relation_rep_to_relation_dict[r_rep].append(r)
        top_relation_labels = self.sentence_bert_ranker.get_semantic_sim_topk(text, list(relation_rep_to_relation_dict.keys()), top_k= top_k)
        for label in top_relation_labels:
            results += relation_rep_to_relation_dict[label]
        return results


    def get_neighbor_relations_with_previous_LF(self, answers, end_point, expand_point, previous_LF:SimpleGraph, question):
        if self.kb == KB_TYPE.FREEBASE:
            relations = set()
            relations_reversed = set()
            if answers is None:
                neighbor_relations = self.sparql_executor.expand_next_hop_path_with_LF(LF = previous_LF, expand_point = expand_point, \
                                                                                end_point = end_point, answer=None, semantic_sim_ranker=self.sentence_bert_ranker, 
                                                                                question=question, get_end_points=False)
                #cvt关系与非cvt关系分开处理
                temp = []
                for item in neighbor_relations:
                    if "/" in item:
                        r1, r2 = item.split("/")[0], item.split("/")[1]
                        # Exclude when r1 and r2 are mutual inverses / r1和r2互为反转的情况排除
                        if r1 == "^" + r2 or r2 == "^" + r1:
                            continue 
                        in_black_list = False
                        for black_r in self.relation_blacklists:
                            if black_r in r1.strip("^") or black_r in r2.strip("^") :
                                in_black_list = True
                                break
                        if not in_black_list:
                            #处理freebase上的逆关系：总是尝试将逆向的关系（包含^）的替换为正向关系
                            if r1.startswith("^"):
                                if r1.strip("^") in self.reverse_relation_dict:
                                    r1 = self.reverse_relation_dict[r1.strip("^")]
                            if r2.startswith("^"):
                                if r2.strip("^") in self.reverse_relation_dict:
                                    r2 = self.reverse_relation_dict[r2.strip("^")]
                            temp.append(r1+"/"+r2)
                    else:
                        r = item.strip("^")
                        in_black_list = False
                        for black_r in self.relation_blacklists:
                            if black_r in r.strip("^"):
                                in_black_list = True
                                break
                        if not in_black_list:
                            #处理freebase上的逆关系：总是尝试将逆向的关系（包含^）的替换为正向关系
                            if r.startswith("^"):
                                if r.strip("^") in self.reverse_relation_dict:
                                    r = self.reverse_relation_dict[r.strip("^")]
                            temp.append(r)
                neighbor_relations = temp
                relations = set(neighbor_relations)
            else:
                for ans in answers:
                    #优化1：候选关系应当被所有selected answers所满足
                        neighbor_relations = self.sparql_executor.expand_next_hop_path_with_LF(LF = previous_LF, expand_point = expand_point, \
                                                                                        end_point = end_point, answer=ans, semantic_sim_ranker=self.sentence_bert_ranker, 
                                                                                        question=question, get_end_points=False)       
                        temp = []
                        for item in neighbor_relations:
                            if "/" in item:
                                r1, r2 = item.split("/")[0], item.split("/")[1]
                                in_black_list = False
                                for black_r in self.relation_blacklists:
                                    if black_r in r1.strip("^") or black_r in r2.strip("^") :
                                        in_black_list = True
                                        break
                                if not in_black_list:
                                    #处理freebase上的逆关系：总是尝试将逆向的关系（包含^）的替换为正向关系
                                    if r1.startswith("^"):
                                        if r1.strip("^") in self.reverse_relation_dict:
                                            r1 = self.reverse_relation_dict[r1.strip("^")]
                                    if r2.startswith("^"):
                                        if r2.strip("^") in self.reverse_relation_dict:
                                            r2 = self.reverse_relation_dict[r2.strip("^")]
                                    temp.append(r1+"/"+r2)
                            else:
                                r = item
                                in_black_list = False
                                for black_r in self.relation_blacklists:
                                    if black_r in item.strip("^"):
                                        in_black_list = True
                                        break
                                if not in_black_list:
                                    #处理freebase上的逆关系：总是尝试将逆向的关系（包含^）的替换为正向关系
                                    if r.startswith("^"):
                                        if r.strip("^") in self.reverse_relation_dict:
                                            r = self.reverse_relation_dict[r.strip("^")]
                                    temp.append(r)
                        if len(relations) == 0:
                            relations = set(temp)
                        elif len(temp) > 0:
                            relations= relations.intersection(temp)
            #如果正向和逆向的邻接关系都为空，那么我们认为end_point到answers此时不可达，返回[]
            if len(relations) == 0:
                return []
            # 在这里分类 / Classify candidate relations here
            candidate_relation_list = []
            for r in list(relations):
                candidate_relation_list.append({"relation":r,  "type":self.get_special_property_type(r, self.kb)})
            for r in list(relations_reversed):
                candidate_relation_list.append({"relation":r,  "type":self.get_special_property_type(r, self.kb)})
            return candidate_relation_list
        
        elif self.kb == KB_TYPE.WIKIDATA:
            # WD relations are 2-hop without inverse consideration / WD 关系按两跳考虑，不考虑逆关系
            relations = set()
            if answers is None:
                neighbor_relations = self.sparql_executor.expand_next_hop_path_with_LF(LF = previous_LF, expand_point = expand_point, \
                                                                                end_point = end_point, answer=None)
                temp = []
                for item in neighbor_relations:
                    if "/" in item:
                        r1, r2 = item.split("/")[0].strip("^").split(":")[1], item.split("/")[1].strip("^").split(":")[1]
                    else:
                        r1, r2 = item.strip("^"), ""
                    in_black_list = False
                    for black_r in self.relation_blacklists:
                        if black_r in r1 or black_r in r2:
                            in_black_list = True
                            break
                    if not in_black_list:
                        temp.append(item)
                neighbor_relations = temp
                relations = set(neighbor_relations)
            else:
                for ans in answers:
                    #优化1：候选关系应当被所有selected answers所满足
                    neighbor_relations = self.sparql_executor.expand_next_hop_path_with_LF(LF = previous_LF, expand_point = expand_point, \
                                                                                        end_point = end_point, answer=ans)             
                    temp = []
                    for item in neighbor_relations:
                        if "/" in item:
                            r1, r2 = item.split("/")[0].strip("^").split(":")[1], item.split("/")[1].strip("^").split(":")[1]
                        else:
                            r1, r2 = item.strip("^"), ""
                        in_black_list = False
                        for black_r in self.relation_blacklists:
                            if black_r in r1 or black_r in r2:
                                in_black_list = True
                                break
                        if not in_black_list:
                            temp.append(item)
                    neighbor_relations = temp
                    if len(relations) == 0:
                        relations = set(neighbor_relations)
                    elif len(neighbor_relations) > 0 :
                        relations = relations.intersection(neighbor_relations)

            #如果邻接关系都为空，那么我们认为end_point到answers此时不可达，返回[]
            if len(relations) == 0:
                return []
            # 在这里分类 / Classify candidate relations here
            candidate_relation_list = []
            for r in list(relations):
                candidate_relation_list.append({"relation":r, "type":self.get_special_property_type(r, self.kb)})
            return candidate_relation_list


    def enumerate_arg(self, LF:SimpleGraph, question_text, candidate_relations, answers):
        #arg的枚举是当搜索完毕后进行的，此时LF是一系列由各子问题对应LF的list
        #我们只考虑：约束OUTQ的arg
        cqv = LF.center_node
        legal_arg_LFs = []
        possible_args = []
        top_quantity_relation_labels = self.sentence_bert_ranker.get_semantic_sim_topk(question_text, 
                                                                            [r['relation'] for r in candidate_relations if r['type'] == "QUANTITY"],
                                                                            top_k=self.CQV_QUANTITY_R_CANDIDATE_NUM)
        top_temporal_relation_labels = self.sentence_bert_ranker.get_semantic_sim_topk(question_text, 
                                                                            [r['relation'] for r in candidate_relations if r['type'] == "TEMPORAL"],
                                                                            top_k=self.CQV_TEMPORAL_R_CANDIDATE_NUM)
        for arg_operator in ['ARGMIN', 'ARGMAX']:
            for r in top_quantity_relation_labels:
                possible_args.append({"property":r, "type":arg_operator})
            for r in top_temporal_relation_labels:
                possible_args.append({"property":r, "type":arg_operator})
        #3、添加ARGMIN、ARGMAX类约束
        self.sparql_logger.info("枚举ARGMIN ARGMAX约束=======================================================================")
        for arg in possible_args:   #形式是：{'relation', 'type'}
            if "/" in arg['property']: 
                arg_r1, arg_r2 = arg['property'].split("/")[0], arg['property'].split("/")[1]
                arg_r1_reversed = arg_r1.startswith("^")
                arg_r1 = arg_r1.strip("^")
                #一种情况是，当前Skeleton中有可以连接上去的cvt；此时将r2接上去就可以
                linked_to_cvt = False
                for cvt_node in LF.get_cvt_nodes():
                    star_in_struct = LF.get_cvt_star(cvt_node)
                    #讨论skeleton中的cvt结构，需要考虑cvt的方向问题
                    if len(star_in_struct) > 0:
                        cqv2cvt_edge_in_struct = [item for item in star_in_struct if item['to'] == cqv or item['from'] == cqv]
                        if len(cqv2cvt_edge_in_struct) > 0:
                            cqv2cvt_edge_in_struct = cqv2cvt_edge_in_struct[0]
                            cqv2cvt_reversed_in_struct = cqv2cvt_edge_in_struct['to'] == cqv
                            cvt2nxt_edgeitems_in_struct = [item['edge'] for item in star_in_struct if item['to'] != cqv and item['from'] != cqv]
                            if cqv2cvt_edge_in_struct['edge'] == arg_r1 and arg_r1_reversed == cqv2cvt_reversed_in_struct:
                                #要求：此时被arg的数值属性不能已经被确定
                                if arg_r2.strip("^") in cvt2nxt_edgeitems_in_struct:
                                    continue
                                else:
                                    test_graph = deepcopy(LF)
                                    #对于arg函数中关系的方向，我们在转为SPARQL时再处理
                                    test_graph.arg_function={"node":cvt_node, "property":arg_r2, "type":arg['type']}
                                    if self.legal_candidate(test_graph, answers):
                                        linked_to_cvt = True
                                        self.sparql_logger.info(f"一个合法结构: {str(test_graph)}")
                                        legal_arg_LFs.append(test_graph)
                #第二种情况是：没有这个cvt；那么，我们需要创造一个两跳的arg，并链接到cqv上
                if not linked_to_cvt:
                    test_graph = deepcopy(LF)
                    test_graph.arg_function={"node":cqv, "property":arg['property'], "type":arg['type']}
                    if self.legal_candidate(test_graph, answers, None):
                        self.sparql_logger.info(f"一个合法结构: {str(test_graph)}")
                        legal_arg_LFs.append(test_graph)
            else:
                if arg['property'] not in [edge['edge'] for edge in LF.edges]:
                    test_graph = deepcopy(LF)
                    test_graph.arg_function={"node":cqv, "property":arg['property'], "type":arg['type']}
                    if self.legal_candidate(test_graph, answers, None):
                        self.sparql_logger.info(f"一个合法结构: {str(test_graph)}")
                        legal_arg_LFs.append(test_graph)
        return legal_arg_LFs


    def subquestion_graph_enumeration_with_timeout(self, current_sub_question, previous_LF:SimpleGraph, answers, cqv:Node, nqvs:list[Node]):
        candidate_sub_question_LFs = []
        mid2item_dict = {}

        def subquestion_graph_enumeration(current_sub_question, previous_LF:SimpleGraph, answers, cqv:Node, nqvs:list[Node]):
            def is_wdt(p):
                # Check if relation follows Wikidata truthy pattern: p:Pxxx/ps:Pxxx or ^ps:Pxxx/^p:Pxxx
                # 判断关系是否符合 Wikidata truthy 模式
                p_1, p_2 = p.split("/")[0], p.split("/")[1]
                prefix_1, pid_1 = p_1.split(":")[0], p_1.split(":")[1]
                prefix_2, pid_2 = p_2.split(":")[0], p_2.split(":")[1]
                if (pid_1 == pid_2) and (prefix_1 == "p" and prefix_2 == "ps") or (prefix_1 == "^ps" and prefix_2 == "^p"):
                    return True
                else:
                    return False

            # Enumerate subgraphs for the current sub-question. / 穷举当前子问题对应的子图。
            # Each sub-question centers on a Current Question Variable (CQV). For the root sub-q,
            # CQV values are known from answers (non-COUNT). Non-leaf CQVs connect to Next Question
            # Variables (NQVs). / CQV 为当前子问题中心变量，根子问题的 CQV 取值由答案已知。非叶 CQV 需连接到 NQV。
            # Timeout via manual checks (func_timeout causes segfault with multi-threading). / 手动超时检查（func_timeout 导致多线程 segfault）。
            # Shared from outer scope: candidate_sub_question_LFs and mid2item_dict / 共享外层变量。
            nonlocal candidate_sub_question_LFs
            nonlocal mid2item_dict
            start_time = time.time()
            sub_question_index = sub_question_text = current_sub_question.question_text.split("|")[0]
            sub_question_text = "|".join(current_sub_question.question_text.split("|")[1:])
            subquestion_text_with_idx = sub_question_index + " is " + sub_question_text
            leaf_question = len(nqvs) == 0
            # Numeral trigger detection for comparisons and ARG during enumeration / 数值触发检测，用于枚举比较和 ARG
            #numeral_triggers = self.detect_numeral_triggers(sub_question_text, current_sub_question.linked_items)
            numeral_triggers = self.detect_numeral_triggers_new(sub_question_text)
            # COUNT is handled at the outer search level / COUNT 已在外层搜索处理
            #arg_triggers = [trigger for trigger in numeral_triggers if trigger == "ARG"]
            comparison_triggers = [trigger for trigger in numeral_triggers if trigger == "COMPARE"]
            # arg_triggers = [trigger for trigger in numeral_triggers if trigger == "ARGMIN" or trigger == "ARGMAX"]
            # comparison_triggers = [trigger for trigger in numeral_triggers if trigger == ">=" or trigger == "<=" or trigger == "<" or trigger == ">"]
            #assert(len(arg_triggers) <= 1 and len(comparison_triggers) <= 2)
            # Explore CQV neighborhood when there is a next sub-question or numeral trigger / 有下一个子问题或数值 trigger 时探测 CQV 邻居
            if not leaf_question or (len(comparison_triggers) > 0):
                if previous_LF is not None and previous_LF.count_query:
                    # COUNT queries have no entity-type answer constraint / COUNT 查询无实体类型答案约束
                    cqv_neighbor_relations = self.get_neighbor_relations_with_previous_LF(answers=None, previous_LF=previous_LF, end_point=None,
                                                                                    expand_point=cqv, question=sub_question_text)
                else:
                    cqv_neighbor_relations = self.get_neighbor_relations_with_previous_LF(answers=self.selected_answers, previous_LF=previous_LF, end_point=None,
                                                                                    expand_point=cqv, question=sub_question_text)
            if not leaf_question:
                assert(len(nqvs)) <= 2
                # Enumerate skeleton structures from CQV to NQV / 穷举从 CQV 到 NQV 的骨架结构
                # Default: only answer variable can be numeric; QVs are entities. Only allow numeric properties when answer is numeric.
                # 默认：仅答案变量可为数量；其他 QV 为实体。仅当答案是数量时允许数值属性。
                if previous_LF is None:
                    if self.kb == KB_TYPE.FREEBASE:
                        if FreebaseConstantForConstruction.get_constant_type(answers[0]['mid']) == FREEBASE_CONSTANT_TYPE.TIME:
                            cqv_neighbor_relations_possible_cqv2nqv = [item for item in cqv_neighbor_relations if item['type'] == "TEMPORAL"]
                        elif FreebaseConstantForConstruction.get_constant_type(answers[0]['mid']) == FREEBASE_CONSTANT_TYPE.QUANTITY:
                            cqv_neighbor_relations_possible_cqv2nqv = [item for item in cqv_neighbor_relations if item['type'] == "QUANTITY"]
                        else:
                            cqv_neighbor_relations_possible_cqv2nqv = [item for item in cqv_neighbor_relations if item['type'] == "NORMAL"]
                    elif self.kb == KB_TYPE.WIKIDATA:
                        if WikidataConstantForConstruction.get_constant_type(answers[0]['mid']) == WIKIDATA_CONSTANT_TYPE.TIME:
                            cqv_neighbor_relations_possible_cqv2nqv = [item for item in cqv_neighbor_relations if item['type'] == "TEMPORAL"]
                        elif WikidataConstantForConstruction.get_constant_type(answers[0]['mid']) == WIKIDATA_CONSTANT_TYPE.QUANTITY:
                            cqv_neighbor_relations_possible_cqv2nqv = [item for item in cqv_neighbor_relations if item['type'] == "QUANTITY"]
                        else:
                            cqv_neighbor_relations_possible_cqv2nqv = [item for item in cqv_neighbor_relations if item['type'] == "NORMAL"]
                else:
                    cqv_neighbor_relations_possible_cqv2nqv = [item for item in cqv_neighbor_relations if item['type'] == "NORMAL"]
                if self.kb == KB_TYPE.WIKIDATA:
                    # Wikidata: CQV→NQV must follow truthy pattern cqv p:x/ps:x nqv or reverse / Wikidata 必须是 p:x/ps:x 模式
                     cqv_neighbor_relations_possible_cqv2nqv = [item for item in cqv_neighbor_relations_possible_cqv2nqv if is_wdt(item['relation'])]
                cqv_to_nqv_candidate_relations = []
                for nqv in nqvs:
                    top_candidate_relation_of_this_nqv = self.get_ranked_relations(cqv_neighbor_relations_possible_cqv2nqv, subquestion_text_with_idx, self.CQV_TO_NQV_CANDIDATE_NUM,
                                                                            str(cqv), str(nqv))
                    cqv_to_nqv_candidate_relations.append([(nqv, item) for item in top_candidate_relation_of_this_nqv])
                    # Gather into num_nqv lists, enumerate with product / 收集为多个 list，使用 product 穷举
                self.sparql_logger.info(f"Candidate CQV→NQV relations / 候选 CQV→NQV 关系: {str(cqv_to_nqv_candidate_relations)}")
                possible_cqv_to_nqv_structures = []
                for r_combination in itertools.product(*cqv_to_nqv_candidate_relations):
                    graph = SimpleGraph(cqv, ground_kb=self.kb)
                    for nqv_r in r_combination:
                        nqv = nqv_r[0]
                        r = nqv_r[1]
                        if "/" in r['relation']:
                            #注意，两条边都是从CVT出来的
                            (cvt_r1, cvt_r2) = r['relation'].split("/")[0], r['relation'].split("/")[1]
                            (direction_r1, direction_r2) = "in" if cvt_r1.startswith("^") else "out", "in" if cvt_r2.startswith("^") else "out"
                            (cvt_r1, cvt_r2) = cvt_r1.strip("^"), cvt_r2.strip("^")
                            new_cvt_node = graph.get_new_cvt_node()
                            graph.attach_node(cqv, cvt_r1, new_cvt_node, direction_r1)
                            graph.attach_node(new_cvt_node, cvt_r2, nqv, direction_r2)
                        else:
                            direction_r = "in" if r['relation'].startswith("^") else "out"
                            r = r['relation'].strip("^")
                            graph.attach_node(cqv, r, nqv, direction_r)
                        #同时，我们假定：两个NQV不能作为CVT的两个节点，即不会出现CQV-CVT-NQV1
                        #                                                         |_NQV2      
                    possible_cqv_to_nqv_structures.append(graph)
            else:
                graph_only_cqv = SimpleGraph(cqv, ground_kb=self.kb)
                possible_cqv_to_nqv_structures = [graph_only_cqv]
            # Checkpoint 1: timeout during CQV neighborhood search / 断点1：搜索 CQV 邻居超时
            if time.time() - start_time > self.SUBGRAPH_SEARCH_TIMEOUT:
                self.sparql_logger.info("Subgraph search timeout: CQV neighbor search / 子图搜索超时：CQV 邻居关系搜索")
                return
            # Attach constants (entities, literals, classes) to the skeleton / 将常量连接到骨架上
            candidate_classes = [item for item in current_sub_question.linked_items if item['type'] == "class"]
            candidate_literals = [item for item in current_sub_question.linked_items if item['type'] == "literal"]
            candidate_entities = [item for item in current_sub_question.linked_items if item['type'] == "entity"]
            # Determine usable class types for CQV constraint (currently only for ?OUTQ) / 确定可约束 CQV 的类型（目前仅 ?OUTQ）
            usable_classes = []
            if previous_LF is None: 
                for cls in candidate_classes:
                    test_graph = SimpleGraph(cqv, ground_kb=self.kb)
                    test_graph.add_type_constaint(cqv, cls['mid'])
                    if self.legal_candidate(test_graph, answers, previous_LF):
                        usable_classes.append(cls['mid'])
            if len(usable_classes) > 0:
                class_used = usable_classes[0]
                temp = []
                for struct in possible_cqv_to_nqv_structures:
                    struct.add_type_constaint(cqv, class_used)
                    temp.append(struct)
                possible_cqv_to_nqv_structures = temp
            # Filter literals and entities reachable from CQV within one hop / 筛选一跳内可达 CQV 的字面量和实体
            candidate_constant_dict = {}
            for item in candidate_entities + candidate_literals:
                mid2item_dict[item['mid']] = item
                if previous_LF is not None and previous_LF.count_query:
                    cqv_to_item_relations = self.get_neighbor_relations_with_previous_LF(answers=None, end_point=item, expand_point=cqv, \
                                                                                    question=subquestion_text_with_idx, previous_LF=previous_LF)
                else:
                    cqv_to_item_relations = self.get_neighbor_relations_with_previous_LF(answers=self.selected_answers, end_point=item, expand_point=cqv, \
                                                                                    question=subquestion_text_with_idx, previous_LF=previous_LF)
                if len(cqv_to_item_relations) > 0:
                    candidate_constant_dict[item['mid']] = self.get_ranked_relations(cqv_to_item_relations, subquestion_text_with_idx, self.V_TO_ENTITY_CANDIDATE_NUM, str(cqv), item['label'])
                # Checkpoint 2: timeout during entity neighbor search / 断点2：搜索实体邻居超时
                if time.time() - start_time > self.SUBGRAPH_SEARCH_TIMEOUT:
                    self.sparql_logger.info("Subgraph search timeout: entity neighbor search / 子图搜索超时：实体邻居关系搜索")
                    return
            # Leaf node: augment entity linking with neighbor entities of CQV / 叶子节点：加入 CQV 的邻居实体增强链接 (Sec 5.3: Exploration)
            if leaf_question and self.use_neighbor_entity:
                # Neighbor retrieval is impractical with type constraints / 使用类型约束时检索邻居节点不现实
                if previous_LF is not None and previous_LF.count_query:
                    pass
                else:
                    neighbor_entities_to_label = {}
                    neighbor_mids = None
                    #返回的是mid, label的二元组。因而，我们需要用mid来做并集，最后再加上label
                    for ans in self.selected_answers:
                        neighbor_entities_of_this_answer = self.sparql_executor.get_next_hop_items_with_LF(previous_LF, cqv, ans, \
                                                                                    semantic_sim_ranker=self.sentence_bert_ranker, question=subquestion_text_with_idx)
                        neighbor_entities_to_label.update({item['mid']:item['label'] for item in neighbor_entities_of_this_answer})
                        if neighbor_mids is None:
                            neighbor_mids = [item['mid'] for item in neighbor_entities_of_this_answer]
                        else:
                            neighbor_mids = list(set(neighbor_mids).intersection([item['mid'] for item in neighbor_entities_of_this_answer]))
                
                    neighbor_entities =[{'mid':mid, 'label':neighbor_entities_to_label[mid]} for mid in neighbor_mids if \
                                        neighbor_entities_to_label[mid] is not None and has_token_overlap(neighbor_entities_to_label[mid], sub_question_text)]
                    if len(neighbor_entities) > self.ENTITY_CANDIDATE_NUM:
                        entity_sim_topk = self.sentence_bert_ranker.get_semantic_sim_topk(sub_question_text, [item['label'] for item in neighbor_entities], self.ENTITY_CANDIDATE_NUM)
                        neighbor_entities = [item for item in neighbor_entities if item['label'] in entity_sim_topk]
                    # Checkpoint: timeout during neighbor entity search / 断点：搜索邻居实体超时
                    if time.time() - start_time > self.SUBGRAPH_SEARCH_TIMEOUT:
                        self.sparql_logger.info("Subgraph search timeout: neighbor entity search / 图搜索超时：邻居实体搜索")
                        pass
                    else:
                        neighbor_entity_dict = {}
                        relation_all = []
                        for item in neighbor_entities:
                            item['type'] = 'entity'
                            if previous_LF is not None and previous_LF.count_query:
                                cqv_to_ent_relations = self.get_neighbor_relations_with_previous_LF(answers=None, end_point=item,expand_point=cqv, \
                                                                                            previous_LF=previous_LF, question=subquestion_text_with_idx)
                            else:
                                cqv_to_ent_relations = self.get_neighbor_relations_with_previous_LF(answers=self.selected_answers, end_point=item,expand_point=cqv, \
                                                                                            previous_LF=previous_LF, question=subquestion_text_with_idx)
                            if len(cqv_to_ent_relations) > 0:
                                cqv_to_ent_relations = self.get_ranked_relations(cqv_to_ent_relations, subquestion_text_with_idx, \
                                                                                        self.V_TO_ENTITY_CANDIDATE_NUM, str(cqv), item['label'])
                                neighbor_entity_dict[item['mid']] = cqv_to_ent_relations
                                mid2item_dict[item['mid']] = item
                            if time.time() - start_time > self.SUBGRAPH_SEARCH_TIMEOUT:
                                self.sparql_logger.info("Subgraph search timeout: enumerating neighbor entity relations / 子图搜索超时：枚举邻居实体关系")
                                break
                        candidate_constant_dict.update(neighbor_entity_dict)
            self.sparql_logger.info(f"Candidate neighbor entities & relations / 候选邻接实体与关系: {str(candidate_constant_dict)}")

            # Enumerate constant (entity/literal) structures attachable to skeleton / 枚举可链接到骨架的常量结构
            cvt_constant_structures = []    # CVT structure with two entities / 由两个实体组成的 CVT 结构
            single_constant_stuctures = []  # Single-entity structure (may include CVT) / 单实体结构（可能含 CVT）
            # Structures: CQV--entity and CQV--CVT--entity / 结构类型：CQV-实体 与 CQV-CVT-实体
            for mid in candidate_constant_dict.keys():
                if mid2item_dict[mid]['type'] == 'literal':
                    const_node = Node(mid, NodeType.LITERAL)
                elif mid2item_dict[mid]['type'] == 'entity':
                    const_node = Node(mid, NodeType.ENTITY)
                else:
                    raise Exception("Unknown type")
                for rel in candidate_constant_dict[mid]:
                    # Wikidata: only allow truthy pattern (p/ps or ^ps/^p) for single-constant structures / Wikidata 仅允许 p/ps 或 ^ps/^p 模式
                    if self.kb == KB_TYPE.WIKIDATA:
                        if not is_wdt(rel['relation']):
                            continue
                    graph = SimpleGraph(cqv, ground_kb=self.kb)
                    if "/" in rel['relation']:
                        (cvt_r1, cvt_r2) = rel['relation'].split("/")[0], rel['relation'].split("/")[1]
                        (direction_r1, direction_r2) = "in" if cvt_r1.startswith("^") else "out", "in" if cvt_r2.startswith("^") else "out"
                        (cvt_r1, cvt_r2) = cvt_r1.strip("^"), cvt_r2.strip("^")
                        temp_cvt = Node("?cvt_temp_0", NodeType.VARIABLE)
                        graph.attach_node(cqv, cvt_r1, temp_cvt, direction_r1)
                        graph.attach_node(temp_cvt, cvt_r2, const_node, direction_r2)
                    else:
                    # Wikidata: split into 2-hop; ?cvt here is the statement node / WD 拆分为两跳，?cvt 为 statement node
                        direction_r = "in" if rel['relation'].startswith("^") else "out"
                        r = rel['relation'].strip("^")
                        graph.attach_node(cqv, r, const_node, direction_r)
                    single_constant_stuctures.append(graph)
            cvt_single_constant_structures = [item for item in single_constant_stuctures if len(item.get_cvt_nodes()) > 0]
            # Possible multi-entity CVT: CQV--CVT, CVT---ent1, ..., CVT---entX (X>1). Only consider pairs (X=2).
            # 多实体 CVT 情况：只考虑两个实体组成的 CVT。
            for mid1, mid2 in itertools.combinations(candidate_constant_dict.keys(), 2):
                for r1 in candidate_constant_dict[mid1]:
                    if "/" in r1['relation']:
                        r11, r12 = r1['relation'].split("/")[0], r1['relation'].split("/")[1]
                        for r2 in candidate_constant_dict[mid2]:
                            if "/" in r2['relation']:
                                r21, r22 = r2['relation'].split("/")[0], r2['relation'].split("/")[1]
                                if r11 == r21:
                                    r1_reversed = r11[0] == "^"
                                    r12_reversed, r22_reversed = r12[0] == "^", r22[0] == "^"
                                    r11 = r11.strip("^")
                                    r12 = r12.strip("^")
                                    r22 = r22.strip("^")
                                    struct = SimpleGraph(cqv, ground_kb=self.kb)
                                    node_type1 = NodeType.ENTITY if mid2item_dict[mid1]['type'] == "entity" else NodeType.LITERAL
                                    node_type2 = NodeType.ENTITY if mid2item_dict[mid2]['type'] == "entity" else NodeType.LITERAL
                                    temp_cvt = Node(f"?cvt_temp_0", NodeType.VARIABLE)
                                    node1 = Node(mid1, node_type1)
                                    node2 = Node(mid2, node_type2)
                                    if r1_reversed:
                                        struct.attach_node(cqv, r11, temp_cvt, "in")
                                    else:
                                        struct.attach_node(cqv, r11, temp_cvt, "out")
                                    if r12_reversed:
                                        struct.attach_node(temp_cvt, r12, node1, "in")
                                    else:
                                        struct.attach_node(temp_cvt, r12, node1, "out")
                                    if r22_reversed:
                                        struct.attach_node(temp_cvt, r22, node2, "in")
                                    else:
                                        struct.attach_node(temp_cvt, r22, node2, "out")
                                    cvt_constant_structures.append(struct)
            # Enumerate possible COMPARISON structures / 枚举可能的比较结构 (Sec 5.3: Enumeration)
            possible_comparisons = []
            if len(comparison_triggers) > 0:
                top_quantity_relation_labels = self.sentence_bert_ranker.get_semantic_sim_topk(subquestion_text_with_idx, 
                                                                                    [r['relation'] for r in cqv_neighbor_relations if r['type'] == "QUANTITY"],
                                                                                    top_k=self.CQV_QUANTITY_R_CANDIDATE_NUM)
                top_temporal_relation_labels = self.sentence_bert_ranker.get_semantic_sim_topk(subquestion_text_with_idx, 
                                                                                    [r['relation'] for r in cqv_neighbor_relations if r['type'] == "TEMPORAL"],
                                                                                    top_k=self.CQV_TEMPORAL_R_CANDIDATE_NUM)
                if self.kb == KB_TYPE.FREEBASE:
                    for r in top_quantity_relation_labels:
                        for l in candidate_literals:
                            if FreebaseConstantForConstruction.get_constant_type(l['mid']) == FREEBASE_CONSTANT_TYPE.QUANTITY:
                                for comparison_operator in ['>=', ">", "<=", "<"]:
                                    possible_comparisons.append({"property":r, "operator":comparison_operator, "value": l['mid']})
                    for r in top_temporal_relation_labels:
                        for l in candidate_literals:
                            if FreebaseConstantForConstruction.get_constant_type(l['mid']) == FREEBASE_CONSTANT_TYPE.TIME:
                                for comparison_operator in ['>=', ">", "<=", "<"]:
                                    possible_comparisons.append({"property":r, "operator":comparison_operator, "value": l['mid']})
                elif self.kb == KB_TYPE.WIKIDATA:
                    for r in top_quantity_relation_labels:
                        for l in candidate_literals:
                            if WikidataConstantForConstruction.get_constant_type(l['mid']) == WIKIDATA_CONSTANT_TYPE.QUANTITY:
                                for comparison_operator in ['>=', ">", "<=", "<"]:
                                    possible_comparisons.append({"property":r, "operator":comparison_operator, "value": l['mid']})
                    for r in top_temporal_relation_labels:
                        for l in candidate_literals:
                            if WikidataConstantForConstruction.get_constant_type(l['mid']) == WIKIDATA_CONSTANT_TYPE.TIME:
                                for comparison_operator in ['>=', ">", "<=", "<"]:
                                    possible_comparisons.append({"property":r, "operator":comparison_operator, "value": l['mid'],"convert_to_year":False})
                            # Wikidata annotation experiment: year comparison handling / Wikidata 标注实验需要处理年份比较
                            # if WikidataConstantForConstruction.is_legal_year(l['mid']):
                            #     for comparison_operator in ['>=', ">", "=", "<=", "<"]:
                            #         possible_comparisons.append({"property":r, "operator":comparison_operator, "value": l['mid'], "convert_to_year":True})
            # Begin subgraph enumeration / 开始子图穷举
            enumeration_start_time = time.time()
            for skeleton in possible_cqv_to_nqv_structures:
                possible_structures_phase1 = []
                possible_structures_phase2 = []
                possible_structures_phase3 = []
                self.sparql_logger.info(f"Enumerating / 穷举 {str(skeleton)}")
                # Phase 1: attach constant structures to CQV / 阶段1：常量结构链接到 CQV
                self.sparql_logger.info("Phase 1: enumerating constant attachments / 枚举常量连接情况 ==============================================================")
                illegal_stucture_sets = []
                possible_structures_phase1.append(skeleton)
                if skeleton.get_deg(cqv) < self.CQV_MAX_DEG:
                    entities_set_max_size = self.CQV_MAX_DEG - skeleton.get_deg(cqv) + 1
                    for size in range(1, entities_set_max_size):
                        # Pruning: if a constant set S makes the skeleton illegal, any superset of S is also illegal.
                        # 剪枝：若常量集合 S 使骨架不合法，则包含 S 的超集也不合法。
                        for relation_set in itertools.combinations(cvt_constant_structures + single_constant_stuctures, size):
                            legal = True
                            for item in illegal_stucture_sets:
                                if item.issubset(set(relation_set)):
                                    legal = False 
                                    break
                            if not legal:
                                continue
                            test_graph = deepcopy(skeleton)
                            for r in relation_set:
                                test_graph = get_attached_graph(test_graph, r)
                            if not self.legal_candidate(test_graph, answers, prev_graph=previous_LF):
                                illegal_stucture_sets.append(set(relation_set))
                            else:
                                if len(test_graph.edges) <= self.SUBQ_GRAPH_MAX_SIZE:
                                    possible_structures_phase1.append(test_graph)
                                    self.sparql_logger.info(f"Legal structure / 合法结构: {str(test_graph)}")
                            if time.time() - enumeration_start_time > self.ENUMERATION_PHASE_TIMEOUT:
                                self.sparql_logger.info("Subgraph search timeout: phase 1 enumeration / 子图搜索超时：阶段1穷举")
                                # Try to preserve phase 1 results / 尝试保留第一阶段结果
                                # candidate_sub_question_LFs += possible_structures_phase1
                                return
                # Phase 2: attach constant structures to CVT nodes in the skeleton / 阶段2：常量结构链接到骨架中的 CVT 节点
                # Enumerate: single-entity CVT structures that can attach to skeleton CVT nodes / 穷举可链接到骨架 CVT 的单实体 CVT 结构
                possible_cvt_entity_attachment = {}
                for cvt_node in skeleton.get_cvt_nodes():
                    star_in_skeleton = skeleton.get_cvt_star(cvt_node)
                    assert(len(star_in_skeleton) == 2)
                    # Analyze CVT direction in skeleton / 分析骨架中 CVT 方向
                    cqv2cvt_edge_in_skeleton = [item for item in star_in_skeleton if item['to'] == cqv or item['from'] == cqv][0]
                    cqv2cvt_reversed_in_skeleton = cqv2cvt_edge_in_skeleton['to'] == cqv
                    for cvt_struct in cvt_single_constant_structures:
                        star_in_struct = cvt_struct.get_cvt_star(cvt_struct.get_cvt_nodes()[0])
                        assert(len(star_in_struct) == 2)
                        # Analyze CVT direction in entity CVT struct / 分析实体 CVT 结构方向
                        cqv2cvt_edge_in_struct = [item for item in star_in_struct if item['to'] == cqv or item['from'] == cqv][0]
                        cqv2cvt_reversed_in_struct = cqv2cvt_edge_in_struct['to'] == cqv
                        cvt2nqv_edge_in_struct = [item for item in star_in_struct if item['to'] != cqv and item['from'] != cqv][0]
                        cvt2nqv_reversed_in_struct = cvt2nqv_edge_in_struct['to'].value.startswith("?cvt")
                        # Require: same edge and same direction / 要求：相同边且方向一致
                        if cqv2cvt_edge_in_skeleton['edge'] == cqv2cvt_edge_in_struct['edge'] and cqv2cvt_reversed_in_skeleton == cqv2cvt_reversed_in_struct:
                            # Test: is the attachment legal? / 测试连接后是否合法
                            if not cvt2nqv_reversed_in_struct:
                                # Check: does this edge already exist? / 检查边是否已存在
                                exist = False
                                for edgeitem in test_graph.edges:
                                    if edgeitem['edge'] == cvt2nqv_edge_in_struct['edge'] and edgeitem['from'] == cvt_node and edgeitem['to'] == cvt2nqv_edge_in_struct['to']:
                                        exist = True
                                        break
                                if not exist:
                                    test_graph.attach_node(cvt_node, cvt2nqv_edge_in_struct['edge'], cvt2nqv_edge_in_struct['to'], 'out')
                                    if self.legal_candidate(test_graph, answers, prev_graph=previous_LF):
                                        possible_cvt_entity_attachment[cvt_node]=((cvt_node, cvt2nqv_edge_in_struct['edge'], cvt2nqv_edge_in_struct['to'], 'out'))
                            else:
                                exist = False
                                for edgeitem in test_graph.edges:
                                    if edgeitem['edge'] == cvt2nqv_edge_in_struct['edge'] and edgeitem['to'] == cvt_node and edgeitem['from'] == cvt2nqv_edge_in_struct['from']:
                                        exist = True
                                        break
                                if not exist:
                                    test_graph.attach_node(cvt_node, cvt2nqv_edge_in_struct['edge'], cvt2nqv_edge_in_struct['from'], 'in')
                                    if self.legal_candidate(test_graph, answers, prev_graph=previous_LF):
                                        possible_cvt_entity_attachment[cvt_node]=((cvt_node, cvt2nqv_edge_in_struct['edge'], cvt2nqv_edge_in_struct['from'], 'in'))
                        if time.time() - enumeration_start_time > self.ENUMERATION_PHASE_TIMEOUT:
                                self.sparql_logger.info("Subgraph search timeout: phase 2 enumeration / 子图搜索超时：阶段2穷举")
                                # Try to preserve phase 1 results / 尝试保留第一阶段结果
                                # candidate_sub_question_LFs += possible_structures_phase1
                                return
                for struct in possible_structures_phase1:
                    possible_structures_phase2.append(struct)
                    for k, v in possible_cvt_entity_attachment.items():
                        node, edge, ent, direction = v[0], v[1], v[2], v[3]
                        if ent in struct.nodes:   # An entity already in the graph cannot constrain CVT again / 已在图中的实体不能再次约束 CVT
                            continue
                        else:                   # Temporarily assume only one can be placed / 暂时假定只能放一个
                            test_graph = deepcopy(struct)
                            test_graph.attach_node(node, edge, ent, direction)
                            if len(test_graph.edges) <= self.SUBQ_GRAPH_MAX_SIZE and self.legal_candidate(test_graph, answers, prev_graph=previous_LF):
                                self.sparql_logger.info(f"Legal structure / 合法结构: {str(test_graph)}")
                            possible_structures_phase2.append(test_graph)
                # If no comparison trigger detected, end here / 未探测到比较 trigger，在此结束
                if len(comparison_triggers) == 0:
                    candidate_sub_question_LFs += possible_structures_phase2
                    continue
                # Phase 3: add numeric comparison constraints / 阶段3：添加数值比较约束 (Sec 5.3: comparisons)
                self.sparql_logger.info("Phase 3: enumerating comparison constraints / 枚举比较类型约束 ==============================================================")
                phase_ended = False
                for struct in possible_structures_phase2:
                    # Skip graphs with no type constraints and no edges / 跳过无类型约束且无边图
                    if len(struct.nodes) == 1 and len(struct.type_constraints) == 0:
                        continue 
                    possible_structures_phase3.append(struct)
                    for cmp in possible_comparisons:
                        if "/" in cmp['property']: 
                            cmp_r1, cmp_r2 = cmp['property'].split("/")[0], cmp['property'].split("/")[1]
                            cmp_r1_reversed = cmp_r1.startswith("^")
                            cmp_r1 = cmp_r1.strip("^")
                            # Case 1: skeleton has a matching CVT; attach r2 to it / 情况1：骨架中有匹配的 CVT，将 r2 接上
                            linked_to_cvt = False
                            for cvt_node in struct.get_cvt_nodes():
                                star_in_struct = struct.get_cvt_star(cvt_node)
                                #assert(len(star_in_struct) == 2)
                                # Analyze CVT direction in skeleton / 分析骨架中 CVT 方向
                                cqv2cvt_edge_in_struct = [item for item in star_in_struct if item['to'] == cqv or item['from'] == cqv][0]
                                cqv2cvt_reversed_in_struct = cqv2cvt_edge_in_struct['to'] == cqv
                                cvt2nxt_edgeitems_in_struct = [item['edge'] for item in star_in_struct if item['to'] != cqv and item['from'] != cqv]
                                if cqv2cvt_edge_in_struct['edge'] == cmp_r1 and cmp_r1_reversed == cqv2cvt_reversed_in_struct:
                                    # Requirement: the compared numeric property must not be already determined / 要求：被比较的数值属性不能已确定
                                    if cmp_r2.strip("^") in cvt2nxt_edgeitems_in_struct:
                                        continue
                                    else:
                                        test_graph = deepcopy(struct)
                                        # Direction of the property in ARG function is handled during SPARQL conversion / ARG 关系中方向在转 SPARQL 时处理
                                        test_graph.comparisons.append({"node":cvt_node, "property":cmp_r2, "operator":cmp['operator'], "value":cmp["value"]})
                                        if self.legal_candidate(test_graph, answers, previous_LF):
                                            linked_to_cvt = True
                                            self.sparql_logger.info(f"Legal structure / 合法结构: {str(test_graph)}")
                                            possible_structures_phase3.append(test_graph)
                            # Case 2: no matching CVT; create a 2-hop comparison linked to CQV / 情况2：无匹配 CVT，创建链接到 CQV 的两跳比较
                            if not linked_to_cvt:
                                test_graph = deepcopy(struct)
                                test_graph.comparisons.append({"node":cqv, "property":cmp['property'], "operator":cmp['operator'], "value":cmp["value"]})
                                if self.legal_candidate(test_graph, answers, previous_LF):
                                    self.sparql_logger.info(f"Legal structure / 合法结构: {str(test_graph)}")
                                    possible_structures_phase3.append(test_graph)
                        else:
                            if cmp['property'] not in [edge['edge'] for edge in struct.edges]:
                                test_graph = deepcopy(struct)
                                test_graph.comparisons.append({"node":cqv, "property":cmp['property'], "operator":cmp['operator'], "value":cmp["value"]})
                                if self.legal_candidate(test_graph, answers, previous_LF):
                                    self.sparql_logger.info(f"Legal structure / 合法结构: {str(test_graph)}")
                                    possible_structures_phase3.append(test_graph)
                        if time.time() - enumeration_start_time > self.ENUMERATION_PHASE_TIMEOUT:
                            # Save current results, skip this phase / 记录此时结果，跳过此阶段
                            possible_structures_phase3 = list(set(possible_structures_phase3).union(possible_structures_phase2))
                            phase_ended = True
                            break
                    if phase_ended:
                        break
                candidate_sub_question_LFs += possible_structures_phase3
        
        self.sparql_logger.info(f"Processing from / 从 {str(previous_LF)} 出发，处理 {current_sub_question.question_text}")
        try:
            subquestion_graph_enumeration(current_sub_question, previous_LF, answers, cqv, nqvs)
        except Exception as e:
            self.sparql_logger.info(str(e))
            self.sparql_logger.info(traceback.format_exc())
            pass
        leaf_question = len(nqvs) == 0
        sub_question_index = sub_question_text = current_sub_question.question_text.split("|")[0]
        sub_question_text = "|".join(current_sub_question.question_text.split("|")[1:])
        subquestion_text_with_idx = sub_question_index + " is " + sub_question_text
        if leaf_question:   # Leaf nodes must have at least some content attached to CQV / 叶子节点必须在 CQV 上有内容
            for lf in candidate_sub_question_LFs:
                if len(lf.nodes) == 1 and lf.arg_function is None and len(lf.comparisons) == 0 and len(lf.type_constraints) == 0:
                    candidate_sub_question_LFs.remove(lf)
        subquestion_text_with_idx = sub_question_index + " is " + sub_question_text
        if len(candidate_sub_question_LFs) > 0:
            if leaf_question:
                top_k = self.SUBQ_CANDIDATE_GRAPHS_NUM_LEAF
            else:
                top_k = self.SUBQ_CANDIDATE_GRAPHS_NUM_NON_LEAF
            self.global_mid2item_dict.update(mid2item_dict)
            candidate_sub_question_LFs = self.get_current_graph_sim_top_k(candidate_sub_question_LFs, subquestion_text_with_idx, mid2item_dict, top_k)
        return candidate_sub_question_LFs
        

    def built_question_structure(self, question_item, dataset):
        def get_decomposition_dependency(decomposition):
            '''
            Build sub-question topological structure from decomposition. Output is a dict of in-edges.
            按照分解建立子问题的拓扑结构。输出 dict 记录每个子问题的入边。

            The decomposition may contain conjunctive descriptions alongside sub-questions;
            our framework only handles sub-questions, so conjunctive descriptions are merged.
            Recursive: if current list length > 1, concatenate descriptions.
            分解可能同时含并列描述与子问题；框架仅处理子问题，因此合并并列描述。递归合并。
            '''
            def join_conjunctive_descriptions(current):
                if len(current) > 1:
                    conjunctive_desc = ", ".join(item['description'] for item in current)
                    subquestions_of_conjunctive_desc = {}
                    for item in current:
                        for k, v in item.items():
                            if k != "description":
                                subquestions_of_conjunctive_desc[k] = v
                    joined_current = {"description": conjunctive_desc}
                    joined_current.update(subquestions_of_conjunctive_desc)
                    return joined_current
                else:
                    return current[0]
            
            def recursively_rebuild_decompostion(current, k):
                """
                Merge conjunctive descriptions and prepend sub-question index for downstream use.
                调用 join_conjunctive_descriptions 合并并列描述，并为每个子问题描述前添加问题编号。
                """
                current = join_conjunctive_descriptions(current)
                current['description'] = k+"|"+current['description']
                for k in current.keys():
                    if k != "description":
                        current[k] = recursively_rebuild_decompostion(current[k], k)
                return current
            
            dependency_dict = {}
            def recursively_built_dependency_dict(current):
                nonlocal dependency_dict
                dependency_dict[current['description']] = []
                if len(current) == 1:   # Leaf sub-question: only description, no dependent sub-questions / 最底层子问题，仅有 description
                    return
                else:
                    for k, v in current.items():
                        if k != "description":
                            dependency_dict[current['description']].append(v['description'])
                            recursively_built_dependency_dict(v)


            rebuilt_decompostion = recursively_rebuild_decompostion(decomposition, "<OUTQ>")    # Root level index is <OUTQ> / 最外层编号为 <OUTQ>
            recursively_built_dependency_dict(rebuilt_decompostion)
            return dependency_dict

        text = question_item['question']
        answers = question_item['answer']
        linked_items = []
        if self.use_golden_entity:
            # Use golden classes and entities / 使用 golden 的 class 与 entity
            for item in question_item['golden_grounded_items']:
                if item['type'] == "class":
                    if self.kb == KB_TYPE.FREEBASE:
                        # Freebase class label is the mid / Freebase class label 就是 mid
                        item['label'] = item['mid']
                    else:
                        item['label'] = question_item['golden_item_to_label'][item['mid']]
                else:
                    item['label'] = question_item['golden_item_to_label'][item['mid']]
                # if self.kb == KB_TYPE.WIKIDATA:
                #     item['mid'] = "wd:" + item['mid'] if not item['mid'].startswith("wd:") else item['mid']
                linked_items.append(item)
            # Still use detected literals / 但仍然使用探测到的 literal
            if len(question_item['linked_item_list']) > 0:
                if isinstance(question_item['linked_item_list'][0], list):
                    for source in question_item['linked_item_list']:
                        for item in source:
                            if item['type'] == "literal":
                                item['label'] = question_item['linked_item_to_label'][item['mid']]
                                linked_items.append(item)
                else:
                    for item in question_item['linked_item_list']:
                        if item['type'] == "literal":
                            item['label'] = question_item['linked_item_to_label'][item['mid']]
                            linked_items.append(item)
        else:
            # Attach labels to linked_items / 为 linked_items 附加上 label
            if len(question_item['linked_item_list']) > 0:
                if isinstance(question_item['linked_item_list'][0], list):
                    for source in question_item['linked_item_list']:
                        for item in source:
                            if item['type'] == "class":
                                if self.kb == KB_TYPE.FREEBASE:
                                    item['label'] = item['mid']
                                else:
                                    item['label'] = question_item['linked_item_to_label'][item['mid']]
                            else:
                                item['label'] = question_item['linked_item_to_label'][item['mid']]
                            # if self.kb == KB_TYPE.WIKIDATA:
                            #     item['mid'] = "wd:" + item['mid'] if not item['mid'].startswith("wd:") else item['mid']
                            linked_items.append(item)
                else:
                    for item in question_item['linked_item_list']:
                        if item['type'] == "class":
                            if self.kb == KB_TYPE.FREEBASE:
                                item['label'] = item['mid']
                            else:
                                item['label'] = question_item['linked_item_to_label'][item['mid']]
                        else:
                            item['label'] = question_item['linked_item_to_label'][item['mid']]
                        linked_items.append(item)
        if dataset == Dataset.GRAIL:
            temp = []
            # Handle year, month, and day literal formats / 需要处理年份、月份与日份格式
            for item in linked_items:
                if "#dateTime" in item['mid'] or "#date" in item['mid']:
                    match_y = re.search(r'(\d{4})', item['mid'])
                    match_ym = re.search(r'(\d{4})-(\d{2})', item['mid'])
                    match_ymd = re.search(r'(\d{4})-(\d{2})-(\d{2})', item['mid'])
                    if match_ymd:
                        y, m, d = match_ymd.group(1), match_ymd.group(2), match_ymd.group(3)
                        item['mid'] = f'\"{y}-{m}-{d}-08:00\"^^<http://www.w3.org/2001/XMLSchema#date>'
                    elif match_ym:
                        y, m = match_ym.group(1), match_ym.group(2)
                        item['mid'] = f'\"{y}-{m}-08:00\"^^<http://www.w3.org/2001/XMLSchema#gYearMonth>'
                    elif match_y:
                        y = match_y.group(1)
                        item['mid'] = f'\"{y}-08:00\"^^<http://www.w3.org/2001/XMLSchema#gYear>'
                    else:
                        assert(0)
                temp.append(item)
            linked_items = temp
        question = Question(text, answers, linked_items)
        decomposition = deepcopy(question_item['decomposition'])
        try:
            question.decompostion_dependency = get_decomposition_dependency(decomposition)
        except:
            # Decomposition failed; fallback to treating the whole question as a single unit / 分解出错，回退为不分解整个问题
            self.sparql_logger.error(traceback.format_exc())
            question.decompostion_dependency = {f"<OUTQ>|{question_item['question']}":[]}
            pass
        question.build_subquestions_by_decomposition(self.use_golden_entity)
        return question


    def begin_search(self, input_data, return_sexpr = True, only_keep_accurate = True):
        results = []
        find_cnt = 0
        fail_questions = []
        for idx, item in enumerate(tqdm(input_data)):
            if idx > 0 and idx %100 == 0: 
                #objgraph.show_most_common_types(limit=50)  
                gc.collect()
            self.sparql_logger.info(f"----------------------------------------{str(idx)}/{str(len(input_data))}----------------------------------------------------------------")
            self.sparql_logger.info(item['question'])
            try:
                start_time = time.time()
                res, f1 = self.LF_search_main(item, self.dataset, only_keep_accurate = only_keep_accurate)
                if len(res) > 0:
                    find_cnt += 1
                    self.sparql_logger.info("Search results / 搜索结果")
                    for r in res:
                        self.sparql_logger.info(str(r))
                    self.sparql_logger.info(f"Search result F1 / 搜索结果 F1: {str(f1)}")
                else:
                    fail_questions.append(item)
                result_list = []
                if return_sexpr:
                    for graph in res:
                        try:
                            sparql = graph.to_sparql_query()
                            sexpr = graph.to_Sexpr()
                            result_list.append({"sparql": sparql, "sexpr":sexpr})
                        except:
                            pass
                    results.append({"qid": item['qid'], "idx":idx, "time":time.time()-start_time, "question": item['question'], "search_results": result_list, "F1": f1})
                else:
                    for graph in res:
                        try:
                            sparql = graph.to_sparql_query_easy_to_read()
                            result_list.append({"sparql": sparql})
                        except:
                            pass
                    results.append({"qid": item['qid'], "idx":idx, "time":time.time()-start_time, "question": item['question'], "search_results": result_list, "F1": f1})
            except Exception as e:
                f1 = 0
                self.sparql_logger.error(traceback.format_exc())
                results.append({"qid": item['qid'], "idx":idx, "question": item['question'], "search_results": [], "F1": f1, "error":traceback.format_exc()})
            self.sparql_logger.info(f"in {str(idx+1)} questions, found {str(find_cnt)} LFs")
        if self.sparql_executor.cached_results is not None:
            self.sparql_executor.write_current_cache_to_file()
        return results, fail_questions