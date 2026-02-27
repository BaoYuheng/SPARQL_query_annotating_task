import sys
sys.path.append("/home5/yhbao/SPARQLannotation")
import math
from src.simple_graph import SimpleGraph, NodeType, EdgeType, Node, get_attached_graph, rename_node
from src.semantic_sim_utils import FBEmbeddingBasedSimilarity, PLMSimRanker
from src.sparql_executor import SparqlOdbcQuerierNoSexpr, SparqlOdbcQuerierNoSexprWikidata
import re 
import time
from src.utils import (
    load_json, dump_json, setup_custom_logger, flatten_decomposition_tree
)
from src.sparql_utils import load_relation_dicts
from src.s_expression_utils import sexp_to_sparql
from src.common import (SPARQL_wrapper_path_official, 
                        KB_TYPE, 
                        FREEBASE_CONSTANT_TYPE,
                        WIKIDATA_CONSTANT_TYPE,
                        DATASET,
                        FreebaseConstantForConstruction,
                        WikidataConstantForConstruction,
                        SPARQL_wrapper_path_dkilab, 
                        SPARQL_wrapper_path_wikidata_2019, 
                        SPARQL_wrapper_path_wikidata_2023, 
                        ODBC_CONFIG_OFFICIAL, 
                        ODBC_CONFIG_WIKIDATA_2019,
                        ODBC_CONFIG_WIKIDATA_2023,
                        ODBC_CONFIG_DKILAB)
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


#记录问题分解的树状结构
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
        question_text：问题文本
        answer：问题答案，暂时只能是一个实体list，或是数字
        candidate_entities, relations：候选的关系或实体
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
        根据{问题：[依赖的子问题]}来构造问题的分解树
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
            if len(self.decompostion_dependency) == 1:    #只有一个子问题，此时应当使用全部的linked items
                subq_linked_items = self.linked_items
            else:
                if use_golden_entity:
                    #使用golden entities时，我们不考虑golden entity的mention必须在subq内
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
        
        #将某个子问题收缩到上个问题内
        #先修改树，再重算dict，再按照dict建新树
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

    def __init__(self, dataset:DATASET, log_file_dir, use_golden_entity:bool, use_neighbor_entity:bool, sparql_cache_dir = None):
        self.sparql_logger = setup_custom_logger(log_file_dir)
        self.dataset = dataset
        if dataset == DATASET.GRAIL or dataset == DATASET.CWQ or dataset == DATASET.WEBQ:
            self.kb = KB_TYPE.FREEBASE
            SUBQ_GRAPH_MAX_SIZE = 4
            if dataset == DATASET.GRAIL:
                self.sparql_executor = SparqlOdbcQuerierNoSexpr(odbc_config=ODBC_CONFIG_DKILAB, sparql_wrapper_path=SPARQL_wrapper_path_dkilab, logger=self.sparql_logger, timeout=6, sparql_cache_dir= sparql_cache_dir)
            elif dataset == DATASET.WEBQ or dataset == DATASET.CWQ:
                #20260120 补充实验
                self.sparql_executor = SparqlOdbcQuerierNoSexpr(odbc_config=ODBC_CONFIG_DKILAB, sparql_wrapper_path=SPARQL_wrapper_path_dkilab, logger=self.sparql_logger, timeout=6, sparql_cache_dir = sparql_cache_dir)
                # self.sparql_executor = SparqlOdbcQuerierNoSexpr(odbc_config=ODBC_CONFIG_OFFICIAL, sparql_wrapper_path=SPARQL_wrapper_path_official, logger=self.sparql_logger, timeout=6, sparql_cache_dir = sparql_cache_dir)
            self.relation_blacklists = ["base.yupgrade", "base.fbontology.", "common.webpage", "freebase.valuenotation"]
            self.FB_time_types = ['type.datetime']
            self.FB_quantity_types = ['type.int', "type.float"]
            self.reverse_relation_dict, self.relation_domain_range_dict = load_relation_dicts()
            #超参数
            self.ANSWER_SAMPLE_SIZE = 3 #最多采样多少个答案，用于判断linked item到entity的可达性
            self.CQV_TO_NQV_CANDIDATE_NUM = 10 #每次从CQV到NQV，筛选最可能的关系topK
            self.CQV_QUANTITY_R_CANDIDATE_NUM = 5 #保留多少个可能的数量属性
            self.CQV_TEMPORAL_R_CANDIDATE_NUM = 5 #保留多少个可能的时间属性
            self.ENTITY_CANDIDATE_NUM = 3
            self.V_TO_ENTITY_CANDIDATE_NUM = 5 #从变量到候选实体的关系TOPK数量
            self.SUBQ_GRAPH_MAX_SIZE = 4 #每个子问题的子图最大可能是多大，这里指边数
            self.SUBQ_CANDIDATE_GRAPHS_NUM_NON_LEAF = 20 #每个非叶子问题的一跳子图的穷举至多返回多少个结果
            self.SUBQ_CANDIDATE_GRAPHS_NUM_LEAF = 10 #每个叶子问题的一跳子图的穷举至多返回多少个结果
            #20260121：加以限制
            self.SUBGRAPH_SEARCH_TIMEOUT = 20 #关系和实体搜索阶段超时限制
            self.SEARCH_FAIL_TIMEOUT = 120  #整个问题搜索的超时限制
            self.ENUMERATION_PHASE_TIMEOUT = 10 #枚举阶段时间限制
            #20260121：不加以限制
            # self.SUBGRAPH_SEARCH_TIMEOUT = 2000 #关系和实体搜索阶段超时限制
            # self.SEARCH_FAIL_TIMEOUT = 1200  #整个问题搜索的超时限制
            # self.ENUMERATION_PHASE_TIMEOUT = 1000 #枚举阶段时间限制
            self.CQV_MAX_DEG = 2  #CQV最大度数
            self.SEARCH_TOPK = 10    #最后返回结果TOPK数
        else:
            self.kb = KB_TYPE.WIKIDATA
            self.sparql_executor = SparqlOdbcQuerierNoSexprWikidata(odbc_config=ODBC_CONFIG_WIKIDATA_2023, sparql_wrapper_path=SPARQL_wrapper_path_wikidata_2023, logger=self.sparql_logger, timeout=10, sparql_cache_dir=sparql_cache_dir)
            numeral_p_data = load_json("/home5/yhbao/SPARQLannotation/data/WD_ontology/wikidata_numeral_property_dict.json")
            self.WD_quantity_properties_dict = {item['pid']:item for item in numeral_p_data['quantity_properties']}
            self.WD_temporal_properties_dict = {item['pid']:item for item in numeral_p_data['temporal_properties']}
            self.WD_property2label_dict = load_json("/home5/yhbao/SPARQLannotation/data/WD_ontology/wikidata_property_id2label_dict.json")
            self.relation_blacklists = ['P31']
            #self.relation_blacklists = []
            #超参数
            self.ANSWER_SAMPLE_SIZE = 3 #最多采样多少个答案，用于判断linked item到entity的可达性
            self.CQV_TO_NQV_CANDIDATE_NUM = 10 #每次从CQV到NQV，筛选最可能的关系topK
            self.CQV_QUANTITY_R_CANDIDATE_NUM = 3 #保留多少个可能的数量属性
            self.CQV_TEMPORAL_R_CANDIDATE_NUM = 3 #保留多少个可能的时间属性
            self.ENTITY_CANDIDATE_NUM = 3
            self.V_TO_ENTITY_CANDIDATE_NUM = 5 #从变量到候选实体的关系TOPK数量
            self.SUBQ_GRAPH_MAX_SIZE = 6#每个子问题的子图最大可能是多大，这里指边数
            self.SUBQ_CANDIDATE_GRAPHS_NUM_NON_LEAF = 20 #每个非叶子问题的一跳子图的穷举至多返回多少个结果
            self.SUBQ_CANDIDATE_GRAPHS_NUM_LEAF = 10 #每个叶子问题的一跳子图的穷举至多返回多少个结果
            self.SUBGRAPH_SEARCH_TIMEOUT = 60 #关系和实体搜索阶段超时限制
            self.SEARCH_FAIL_TIMEOUT = 180  #整个问题搜索的超时限制
            self.ENUMERATION_PHASE_TIMEOUT = 60 #枚举阶段时间限制
            self.CQV_MAX_DEG = 2  #CQV最大度数
            self.SEARCH_TOPK = 10    #最后返回结果TOPK数
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
        if self.kb == KB_TYPE.WIKIDATA:
            service = "sparql_wrapper"
        elif self.kb == KB_TYPE.FREEBASE:
            #20260120
            # service = "odbc"
            service = "sparql_wrapper"
        """
        将当前的graph转化为SPARQL，并求其执行结果相对于答案集合的Precision与Recall
        """
        if graph.count_query:
            sparql = graph.to_sparql_query()
            results_set = set(self.sparql_executor.get_execution_result_one_variable(sparql, service))
            assert len(answers) == 1 and "XMLSchema#integer" in answers[0]['mid']
            match = re.search(r'(\d+)', answers[0]['mid'])
            answers_number = int(match.group(1))
            result_number = int(list(results_set)[0])
            #无法定义recall；只能考虑precision是不断上升的
            if result_number == 0:
                recall = 0
                precision = 0
            else:
                recall = 1              
                precision = answers_number / result_number
        else:
            if graph.arg_function is None and len(answers) <= 10:      #优化：此时使用技巧来重写SPARQL，能够加快一些
                p_sparql, r_sparql = graph.to_sparql_query_fast_PR_with_entity_answers(answers)
                recall = len(set(self.sparql_executor.get_execution_result_one_variable(r_sparql, service))) / len(answers)    #这个值是精确的
                sparql_result = self.sparql_executor.get_execution_result_one_variable(p_sparql, service)
                if len(sparql_result) == 0: #这里，只可能是P_SPARQL执行超时
                    #我们认为这种不合法
                    recall = 0 
                    precision = 0
                else:
                    sparql_result_size = int(list(self.sparql_executor.get_execution_result_one_variable(p_sparql, service))[0])
                    if sparql_result_size == 0:
                        precision = 0
                    else:
                        precision = len(answers) / sparql_result_size #这个值可能偏小，但sparql精确时，P的值也是1；而我们只要求P=1时候是对的就行了
            else:
                sparql = graph.to_sparql_query()
                results_set = set(self.sparql_executor.get_execution_result_one_variable(sparql, service))
                answer_mids = set([item['mid'] for item in answers])
                if len(results_set) == 0:
                    precision = 0
                    recall = 0
                else:
                    recall = len(results_set.intersection(answer_mids)) / len(answer_mids)
                    precision = len(results_set.intersection(answer_mids)) / len(results_set)
        return precision, recall

    def get_current_graph_PR_using_Sexpr(self, graph:SimpleGraph, answers):
        #对于最后的accurate检验，转成SEXPR再翻译回SPRAQL，增加一些Filter
        if self.kb == KB_TYPE.WIKIDATA:
            service = "sparql_wrapper"
        elif self.kb == KB_TYPE.FREEBASE:
            service = "odbc"

        sparql = sexp_to_sparql(str(graph.to_Sexpr()))
        results_set = self.sparql_executor.get_execution_result_one_variable(sparql, service)        
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
            if "/" in r:    #cvt关系的类型由第二部分决定
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
            #注意，这里都是两跳，同样的，由距离较远的r2决定
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
        #检查某个lf在移除tc后，是否结果不变，如果是，移除该lf的tc
        tc = lf.type_constraints
        f1_1 = self.get_current_graph_F1(lf, answers)
        lf.type_constraints = []
        f1_2 = self.get_current_graph_F1(lf, answers)
        if f1_1 != f1_2:
             lf.type_constraints = tc
        return lf

    def detect_numeral_triggers_new(self, question_text):
        #我们只返回3种：COUNT，COMPARE和ARG，具体的比较符号与ARG类型通过穷举来进行
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
        #判断一种特殊情况：
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


    def LF_search_main(self, data_item, dataset:DATASET, only_keep_accurate = False):
        '''
        搜索主函数
        '''
        def init_search_array(question_main):
            nonlocal previous_LFs
            nonlocal search_array
            search_array.clear()
            previous_LFs.clear()
            search_array.append(question_main.sub_question_structure)
            #所有情况下，公共的搜索起点
            numeral_triggers = self.detect_numeral_triggers_new(question_main.question_text)
            #不可能同时存在argmin和argmax
            assert("ARG" not in numeral_triggers or "COUNT" not in numeral_triggers)
            #对于COUNT，添加一系列搜索起点
            if "COUNT" in numeral_triggers and len(question_main.answers) == 1 and "integer" in question_main.answers[0]['mid']:
                #为每个可能的type，添加一个用类型限定的查询
                possible_types = [item['mid'] for item in question_main.linked_items if item['type'] == "class"]
                outq_node = Node("?OUTQ", NodeType.VARIABLE)
                for type in possible_types:
                    graph = SimpleGraph(outq_node, ground_kb=self.kb)
                    graph.count_query = True
                    graph.add_type_constaint(outq_node, type)
                    previous_LFs.append(graph)
                #freebase没有number of...属性，但wikidata显然有很多，因此要额外考虑
                if self.kb == KB_TYPE.WIKIDATA:
                    previous_LFs.append(None)
            else:
                previous_LFs.append(None)
        question_main = self.built_question_structure(data_item, dataset)
        #扔掉随机性
        self.selected_answers = question_main.answers if len(question_main.answers) <= self.ANSWER_SAMPLE_SIZE else question_main.answers[0:self.ANSWER_SAMPLE_SIZE]
        #self.selected_answers = question_main.answers if len(question_main.answers) <= ANSWER_SAMPLE_SIZE else random.sample(question_main.answers, ANSWER_SAMPLE_SIZE)
        self.global_mid2item_dict = {}
        #分为精确的results(f1 =1 )与不精确的results。优先搜索精确的results，但如果一直搜索不到，当Only_keep_accurate=false的时候，返回F1最高的不精确results中，语义最接近的那个
        accurate_final_results = []
        nonaccurate_final_results = []
        previous_LFs = []
        search_array = []
        #按照拓扑序，逐步地求解问题
        init_search_array(question_main)
        has_arg = "ARG" in self.detect_numeral_triggers_new(question_main.question_text)
        search_terminated = False
        search_begin_time = time.time()
        #求解差不多是一个BFS，但将子问题收缩时，需要对序列的头（当前要收缩的子问题）进行操作
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
                    self.sparql_logger.info(f"整体搜索用时超过了阈值{str(self.SEARCH_FAIL_TIMEOUT)}，放弃")
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
                    self.sparql_logger.info("收缩子问题=======================================================================")
                    question_main.subq_identification(current_subq_node, use_golden_entity=self.use_golden_entity)
                    init_search_array(question_main)
                    search_terminated = False
                    continue
            search_array = search_array[1:]
            if len(search_array) == 0: 
                self.sparql_logger.info("-------------------目前对问题搜索尝试完成----------------------------------------")
                #为每个lf枚举ARG
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
                    #注意到，最后的sparql中，显然是需要有某个常量作为锚点的，一种额外情况是，没有三元组中的常量，但有arg或比较
                    if not (lf.no_constant() and lf.arg_function is None and len(lf.comparisons) == 0):  
                        #self.sparql_logger.info(str(lf))
                        f1 = self.get_current_graph_F1(lf, question_main.answers)
                        if math.isclose(f1, 1):
                            #考虑删除class：如果有三元组，且删除class后，结果仍然是精确的，那么删除class
                            if len(lf.edges) > 0:
                                    accurate_final_results.append(self.remove_rebundant_tc(lf, question_main.answers))
                            else:
                                #没有edge而有tc，此时不能删除tc，否则就没有三元组了
                                accurate_final_results.append(lf)
                        else:
                            if not only_keep_accurate:#只有在不仅仅保存精确结果的情况下，才需要记录非精确的结果
                                #我们只保存当前f1最高的结果
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
                    self.sparql_logger.info("收缩子问题=======================================================================")
                    question_main.subq_identification(current_subq_node, self.use_golden_entity)
                    init_search_array(question_main) 
                    search_terminated = False
                else:
                    search_terminated = True
        final_results = []
        highest_f1 = 0
        if len(accurate_final_results) > 0:
            #如果有精确的结果，那么自然只保存精确的结果，且基于语义排序
            highest_f1 = 1
            accurate_final_results = self.get_current_graph_sim_top_k(accurate_final_results, question_main.question_text, self.global_mid2item_dict, top_k = self.SEARCH_TOPK)
            final_results = accurate_final_results
        else:
            #否则，考虑F1最高的结果，也基于语义排序
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
            #主语宾语扔了吧没用
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
                        #有一种情况需要排除：r1和r2是反转的
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
            #在这里，进行分类
            candidate_relation_list = []
            for r in list(relations):
                candidate_relation_list.append({"relation":r,  "type":self.get_special_property_type(r, self.kb)})
            for r in list(relations_reversed):
                candidate_relation_list.append({"relation":r,  "type":self.get_special_property_type(r, self.kb)})
            return candidate_relation_list
        
        elif self.kb == KB_TYPE.WIKIDATA:
            #最大的区别是，WD关系都是按照两跳考虑；同时不考虑逆关系
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
            #在这里，进行分类
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
                #判断关系是不是形如p:Pxxxx/ps:Pxxxx 或 ^ps:Pxxxx/^p:Pxxxx
                p_1, p_2 = p.split("/")[0], p.split("/")[1]
                prefix_1, pid_1 = p_1.split(":")[0], p_1.split(":")[1]
                prefix_2, pid_2 = p_2.split(":")[0], p_2.split(":")[1]
                if (pid_1 == pid_2) and (prefix_1 == "p" and prefix_2 == "ps") or (prefix_1 == "^ps" and prefix_2 == "^p"):
                    return True
                else:
                    return False
            #穷举子问题对应的子图
            #每个问题均有一个中心变量，称为current question variable（CQV)，注意到，答案的利用就是第一个子问题的CQV的取值是已知的（非count问题）
            #除了最后一个子问题，CQV都要连到下一个问题对应的中心变量上，称下一个问题的中心变量为next question variable(NQV)
            #需要引入超时；经过实验，使用func_timeout会导致segmentation fault（多线程）。因此用time简单手写
            #从外层函数共享的搜索结果，和mid到item的dict
            nonlocal candidate_sub_question_LFs
            nonlocal mid2item_dict
            start_time = time.time()
            sub_question_index = sub_question_text = current_sub_question.question_text.split("|")[0]
            sub_question_text = "|".join(current_sub_question.question_text.split("|")[1:])
            subquestion_text_with_idx = sub_question_index + " is " + sub_question_text
            leaf_question = len(nqvs) == 0
            #数值触发检测，在枚举子图时加入比较与ARG
            #numeral_triggers = self.detect_numeral_triggers(sub_question_text, current_sub_question.linked_items)
            numeral_triggers = self.detect_numeral_triggers_new(sub_question_text)
            #COUNT我们已放在搜索的外层处理
            #arg_triggers = [trigger for trigger in numeral_triggers if trigger == "ARG"]
            comparison_triggers = [trigger for trigger in numeral_triggers if trigger == "COMPARE"]
            # arg_triggers = [trigger for trigger in numeral_triggers if trigger == "ARGMIN" or trigger == "ARGMAX"]
            # comparison_triggers = [trigger for trigger in numeral_triggers if trigger == ">=" or trigger == "<=" or trigger == "<" or trigger == ">"]
            #assert(len(arg_triggers) <= 1 and len(comparison_triggers) <= 2)
            #在包含下一个子问题，或者是包含数值trigger时，需要探测CQV旁边的属性信息
            if not leaf_question or (len(comparison_triggers) > 0):
                if previous_LF is not None and previous_LF.count_query:
                    #如果是COUNT类查询，不存在实体类型的答案约束
                    cqv_neighbor_relations = self.get_neighbor_relations_with_previous_LF(answers=None, previous_LF=previous_LF, end_point=None,
                                                                                    expand_point=cqv, question=sub_question_text)
                else:
                    cqv_neighbor_relations = self.get_neighbor_relations_with_previous_LF(answers=self.selected_answers, previous_LF=previous_LF, end_point=None,
                                                                                    expand_point=cqv, question=sub_question_text)
            if not leaf_question:
                assert(len(nqvs)) <= 2
                #穷举从CQV到NQV的可能结构，称为骨架
                #我们默认，只有答案变量可能是数量，其他qv都是实体；那么，仅当答案是数量，且该Neighbor为逆向时，允许数值属性
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
                    #我们对WIKIDATA，有以下判断：必须是cqv p:x/ps:x nqv 或反之
                     cqv_neighbor_relations_possible_cqv2nqv = [item for item in cqv_neighbor_relations_possible_cqv2nqv if is_wdt(item['relation'])]
                cqv_to_nqv_candidate_relations = []
                for nqv in nqvs:
                    top_candidate_relation_of_this_nqv = self.get_ranked_relations(cqv_neighbor_relations_possible_cqv2nqv, subquestion_text_with_idx, self.CQV_TO_NQV_CANDIDATE_NUM,
                                                                            str(cqv), str(nqv))
                    cqv_to_nqv_candidate_relations.append([(nqv, item) for item in top_candidate_relation_of_this_nqv])
                    #整理为num_nqv个list，使用product穷举
                self.sparql_logger.info(f"可能的CQV到NQV关系: {str(cqv_to_nqv_candidate_relations)}")
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
            #断点1：如果搜索CQV旁边的内容时超时
            if time.time() - start_time > self.SUBGRAPH_SEARCH_TIMEOUT: 
                self.sparql_logger.info("子图搜索超时：在搜索CQV周围关系时超时")
                return
            #然后，将常量连接到这个骨架上
            candidate_classes = [item for item in current_sub_question.linked_items if item['type'] == "class"]
            candidate_literals = [item for item in current_sub_question.linked_items if item['type'] == "literal"]
            candidate_entities = [item for item in current_sub_question.linked_items if item['type'] == "entity"]
            #首先，需要确定哪些类型是可以用于约束CQV的
            #我们目前只考虑用类型来限制?OUTQ
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
            ##首先在候选字面量和实体中筛选出一跳内可以到达cqv的
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
                #断点2：如果搜索实体相关内容时超时
                if time.time() - start_time > self.SUBGRAPH_SEARCH_TIMEOUT: 
                    self.sparql_logger.info("子图搜索超时：在检索实体链接结果的邻居关系时超时")
                    return
            #如果是叶子节点，加入当前CQV的邻居实体来增强实体链接
            if leaf_question and self.use_neighbor_entity:
                #如果使用类型，这里检索邻居节点是不现实的！
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
                    #断点1：如果搜索CQV旁边的内容时超时
                    if time.time() - start_time > self.SUBGRAPH_SEARCH_TIMEOUT: 
                        self.sparql_logger.info("图搜索超时：在搜索邻居实体时超时")
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
                                self.sparql_logger.info("子图搜索超时：在枚举邻居实体的可能关系时超时")
                                break
                        candidate_constant_dict.update(neighbor_entity_dict)
            self.sparql_logger.info(f"可能的邻接实体与关系: {str(candidate_constant_dict)}")
            #枚举可能链接到骨架上的实体结构
            cvt_constant_structures = []    #由两个实体组成的cvt结构
            single_constant_stuctures = []  #由一个实体组成的结构（包含CVT）
            #记录单个实体的结构，包含CQV-ent与CQV--CVT--ent
            for mid in candidate_constant_dict.keys():
                if mid2item_dict[mid]['type'] == 'literal':
                    const_node = Node(mid, NodeType.LITERAL)
                elif mid2item_dict[mid]['type'] == 'entity':
                    const_node = Node(mid, NodeType.ENTITY)
                else:
                    raise Exception("Unknown type")
                for rel in candidate_constant_dict[mid]:
                    #对于wikidata，我们只考虑这种方式的single_constant_stucture：p/ps 或^ps/^p
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
                    #WD都拆分成两跳；这里的?cvt其实是statement node
                        direction_r = "in" if rel['relation'].startswith("^") else "out"
                        r = rel['relation'].strip("^")
                        graph.attach_node(cqv, r, const_node, direction_r)
                    single_constant_stuctures.append(graph)
            cvt_single_constant_structures = [item for item in single_constant_stuctures if len(item.get_cvt_nodes()) > 0]
            #有一种可能是，CQV--CVT, CVT---ent1, ..., CVT---entx(x>1)。
            #这种情况，我们只考虑2个组成的。
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
            #枚举可能的COMPARISION结构。
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
                            #对于Wikidata标注实验，需要处理年份的比较
                            # if WikidataConstantForConstruction.is_legal_year(l['mid']):
                            #     for comparison_operator in ['>=', ">", "=", "<=", "<"]:
                            #         possible_comparisons.append({"property":r, "operator":comparison_operator, "value": l['mid'], "convert_to_year":True})
            #对子图进行穷举
            enumeration_start_time = time.time()
            for skeleton in possible_cqv_to_nqv_structures:
                possible_structures_phase1 = []
                possible_structures_phase2 = []
                possible_structures_phase3 = []
                self.sparql_logger.info(f"穷举 {str(skeleton)}")
                #1、常量结构链接到CQV上
                self.sparql_logger.info("枚举常量连接情况=======================================================================")
                illegal_stucture_sets = []
                possible_structures_phase1.append(skeleton)
                if skeleton.get_deg(cqv) < self.CQV_MAX_DEG:
                    entities_set_max_size = self.CQV_MAX_DEG - skeleton.get_deg(cqv) + 1
                    for size in range(1, entities_set_max_size):
                        #注意到一个事实：如果某个常量结构S的集合使得骨架加上S后不是合法的候选，那么包含S的其他集合也不可能是合法的候选，利用这个进行部分的剪枝
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
                                    self.sparql_logger.info(f"一个合法结构: {str(test_graph)}")
                            if time.time() - enumeration_start_time > self.ENUMERATION_PHASE_TIMEOUT: 
                                self.sparql_logger.info("子图搜索超时：在穷举可能子图phase1时超时")
                                #尝试保留第一阶段的结果
                                # candidate_sub_question_LFs += possible_structures_phase1
                                return
                #2、常量结构链接到SKELETON中的某个CVT上
                #穷举目前，单个实体的CVT结构能够链接到skeleton上的情况
                possible_cvt_entity_attachment = {}
                for cvt_node in skeleton.get_cvt_nodes():
                    star_in_skeleton = skeleton.get_cvt_star(cvt_node)
                    assert(len(star_in_skeleton) == 2)
                    #讨论skeleton中的cvt结构，需要考虑cvt的方向问题
                    cqv2cvt_edge_in_skeleton = [item for item in star_in_skeleton if item['to'] == cqv or item['from'] == cqv][0]
                    cqv2cvt_reversed_in_skeleton = cqv2cvt_edge_in_skeleton['to'] == cqv
                    for cvt_struct in cvt_single_constant_structures:
                        star_in_struct = cvt_struct.get_cvt_star(cvt_struct.get_cvt_nodes()[0])
                        assert(len(star_in_struct) == 2)
                        #讨论entity cvt struct中的cvt结构，需要考虑cvt的方向问题
                        cqv2cvt_edge_in_struct = [item for item in star_in_struct if item['to'] == cqv or item['from'] == cqv][0]
                        cqv2cvt_reversed_in_struct = cqv2cvt_edge_in_struct['to'] == cqv
                        cvt2nqv_edge_in_struct = [item for item in star_in_struct if item['to'] != cqv and item['from'] != cqv][0]
                        cvt2nqv_reversed_in_struct = cvt2nqv_edge_in_struct['to'].value.startswith("?cvt")
                        #要求：有相同的边，而且方向相同
                        if cqv2cvt_edge_in_skeleton['edge'] == cqv2cvt_edge_in_struct['edge'] and cqv2cvt_reversed_in_skeleton == cqv2cvt_reversed_in_struct:
                            #测试：连接上之后是否合法？
                            if not cvt2nqv_reversed_in_struct:
                                #检查：这条边是不是已经存在
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
                                self.sparql_logger.info("子图搜索超时：在穷举可能子图phase2时超时")
                                #尝试保留第一阶段的结果
                                # candidate_sub_question_LFs += possible_structures_phase1
                                return
                for struct in possible_structures_phase1:
                    possible_structures_phase2.append(struct)
                    for k, v in possible_cvt_entity_attachment.items():
                        node, edge, ent, direction = v[0], v[1], v[2], v[3]
                        if ent in struct.nodes:   #规定：已经在图中的实体不能再次被用来约束CVT
                            continue
                        else:                   #暂时假定只能放一个
                            test_graph = deepcopy(struct)
                            test_graph.attach_node(node, edge, ent, direction)
                            if len(test_graph.edges) <= self.SUBQ_GRAPH_MAX_SIZE and self.legal_candidate(test_graph, answers, prev_graph=previous_LF):
                                self.sparql_logger.info(f"一个合法结构: {str(test_graph)}")
                            possible_structures_phase2.append(test_graph)
                #如果没有探测到比较trigger，在这里已经结束了
                if len(comparison_triggers) == 0:
                    candidate_sub_question_LFs += possible_structures_phase2
                    continue
                #3、添加数值比较约束
                self.sparql_logger.info("枚举比较类型约束=======================================================================")
                phase_ended = False
                for struct in possible_structures_phase2:
                    #这里添加一个要求，即：不考虑那些没有类型约束，也没有边的图
                    if len(struct.nodes) == 1 and len(struct.type_constraints) == 0:
                        continue 
                    possible_structures_phase3.append(struct)
                    for cmp in possible_comparisons:
                        if "/" in cmp['property']: 
                            cmp_r1, cmp_r2 = cmp['property'].split("/")[0], cmp['property'].split("/")[1]
                            cmp_r1_reversed = cmp_r1.startswith("^")
                            cmp_r1 = cmp_r1.strip("^")
                            #一种情况是，当前Skeleton中有可以连接上去的cvt；此时将r2接上去就可以
                            linked_to_cvt = False
                            for cvt_node in struct.get_cvt_nodes():
                                star_in_struct = struct.get_cvt_star(cvt_node)
                                #assert(len(star_in_struct) == 2)
                                #讨论skeleton中的cvt结构，需要考虑cvt的方向问题
                                cqv2cvt_edge_in_struct = [item for item in star_in_struct if item['to'] == cqv or item['from'] == cqv][0]
                                cqv2cvt_reversed_in_struct = cqv2cvt_edge_in_struct['to'] == cqv
                                cvt2nxt_edgeitems_in_struct = [item['edge'] for item in star_in_struct if item['to'] != cqv and item['from'] != cqv]
                                if cqv2cvt_edge_in_struct['edge'] == cmp_r1 and cmp_r1_reversed == cqv2cvt_reversed_in_struct:
                                    #要求：此时被cmp的数值属性不能已经被确定
                                    if cmp_r2.strip("^") in cvt2nxt_edgeitems_in_struct:
                                        continue
                                    else:
                                        test_graph = deepcopy(struct)
                                        #对于arg函数中关系的方向，我们在转为SPARQL时再处理
                                        #有问题
                                        test_graph.comparisons.append({"node":cvt_node, "property":cmp_r2, "operator":cmp['operator'], "value":cmp["value"]})
                                        if self.legal_candidate(test_graph, answers, previous_LF):
                                            linked_to_cvt = True
                                            self.sparql_logger.info(f"一个合法结构: {str(test_graph)}")
                                            possible_structures_phase3.append(test_graph)
                            #第二种情况是：没有这个cvt；那么，我们需要创造一个两跳的arg，并链接到cqv上
                            if not linked_to_cvt:
                                test_graph = deepcopy(struct)
                                test_graph.comparisons.append({"node":cqv, "property":cmp['property'], "operator":cmp['operator'], "value":cmp["value"]})
                                if self.legal_candidate(test_graph, answers, previous_LF):
                                    self.sparql_logger.info(f"一个合法结构: {str(test_graph)}")
                                    possible_structures_phase3.append(test_graph)
                        else:
                            if cmp['property'] not in [edge['edge'] for edge in struct.edges]:
                                test_graph = deepcopy(struct)
                                test_graph.comparisons.append({"node":cqv, "property":cmp['property'], "operator":cmp['operator'], "value":cmp["value"]})
                                if self.legal_candidate(test_graph, answers, previous_LF):
                                    self.sparql_logger.info(f"一个合法结构: {str(test_graph)}")
                                    possible_structures_phase3.append(test_graph)
                        if time.time() - enumeration_start_time > self.ENUMERATION_PHASE_TIMEOUT: 
                            #记录此时结果，跳过此阶段
                            possible_structures_phase3 = list(set(possible_structures_phase3).union(possible_structures_phase2))
                            phase_ended = True
                            break
                    if phase_ended:
                        break
                candidate_sub_question_LFs += possible_structures_phase3
        
        self.sparql_logger.info(f"从 {str(previous_LF)} 出发，处理 {current_sub_question.question_text}")
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
        if leaf_question:   #如果是叶子节点，至少需要CQV上链接有某些内容
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
            按照分解，建立sub_questions的拓扑结构
            输出是一个dict，记录每个子问题问题的入边

            现有的分解考虑了并列描述与子问题，但我们的框架只处理子问题，因而对并列描述进行合并
            采用递归的方法：如果当前处理的List长度大于1，那么拼接起来
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
                调用join_conjunctive_descriptions来合并并列的描述。同时，为每个子问题的描述前面添加问题编号，便于后续使用
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
                if len(current) == 1:   #最下层的子问题，只有descrition而没有依赖的子问题
                    return
                else:
                    for k, v in current.items():
                        if k != "description":
                            dependency_dict[current['description']].append(v['description'])
                            recursively_built_dependency_dict(v)


            rebuilt_decompostion = recursively_rebuild_decompostion(decomposition, "<OUTQ>")    #最外层的编号为<OUTQ>
            recursively_built_dependency_dict(rebuilt_decompostion)
            return dependency_dict

        text = question_item['question']
        answers = question_item['answer']
        linked_items = []
        if self.use_golden_entity:
            #使用golden的class与entity
            for item in question_item['golden_grounded_items']:
                if item['type'] == "class":
                    if self.kb == KB_TYPE.FREEBASE:
                        #Freebase class label就是mid
                        item['label'] = item['mid']
                    else:
                        item['label'] = question_item['golden_item_to_label'][item['mid']]
                else:
                    item['label'] = question_item['golden_item_to_label'][item['mid']]
                # if self.kb == KB_TYPE.WIKIDATA:
                #     item['mid'] = "wd:" + item['mid'] if not item['mid'].startswith("wd:") else item['mid']
                linked_items.append(item)
            #但仍然使用探测到的Literal
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
            #为linked_items附加上label
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
        if dataset == DATASET.GRAIL:
            temp = []
            #需要处理年份、月份与日份
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
            #此时分解出错；我们只能采取将整个问题不分解的方法
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
                    self.sparql_logger.info("搜索结果")
                    for r in res:
                        self.sparql_logger.info(str(r))
                    self.sparql_logger.info(f"搜索结果F1: {str(f1)}")
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

