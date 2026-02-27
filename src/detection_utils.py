from .utils import (
    get_PRF1, flatten_list, PRF1_for_count, load_json
)
from .sparql_executor import (
    SparqlOdbcQuerier, SparqlOdbcQuerierWikidata
)
from .s_expression_utils import (
    sexp_to_sparql,
    sexp_to_sparql_wikidata,
    AND, ARG, COUNT,
    extract_value_from_literal
)
from .common import (
   DECOMPOSITION_TAGS, DELIMETER, PATTERN_DELIMETER, KB_TYPE, 
   FreebaseConstantForConstruction, WikidataConstantForConstruction, WIKIDATA_CONSTANT_TYPE,
   convert_wikidata_type, convert_freebase_type
)
from .sentence_bert_utils import (
    SentenceSimilarityPredictor
)
# from .bi_encoder_utils import BiEncoderInferencor
from tqdm import tqdm
from queue import PriorityQueue
import copy
import math
import itertools
import random
import re
from typing import List, Dict, Union


class SearchItem:
    '''能够作为 SearchItem 的前提: recall = 1'''
    def __init__(self, sexp, execution_result, recall, precision, depth=1, atom_visited_index=list(), arg_visited_index=list(), qid=None):
        """
        @param visited_index: 已经加入 sexp 中的 atom sexp 下标
        """
        self.sexp = sexp
        self.execution_result = execution_result
        self.recall = recall
        self.precision = precision
        self.depth = depth
        self.atom_visited_index = atom_visited_index
        self.arg_visited_index = arg_visited_index
        self.qid = qid

    def __str__(self):
        # print(num)
        return f"S-expression: {self.sexp}; Precision: {self.precision}"
    
    def __repr__(self):
        return self.__str__()

class BestFirstSearch:
    '''Recall 为 1 的前提下 Precision 增大'''
    def __init__(self, atom_paths, arg_paths, answers, dataset, sparql_querier:SparqlOdbcQuerier, logger, depth_upper_bound=3, top_k=1):
        self.atom_paths = atom_paths
        self.arg_paths = arg_paths
        self.atom_sexps = list() # 保持顺序

        self.answer_mids = set([
            ans["mid"] for ans in answers
        ])
        self.dataset = dataset
        self.logger = logger
        self.sparql_querier = sparql_querier
        self.sexp_priority_queue = PriorityQueue(0)
        self.depth_upper_bound = depth_upper_bound # 当前 depth > depth_upper_bound, 不加入优先级队列中
        assert self.depth_upper_bound >= 1
        self.top_k = top_k
        self.searched_queries = list()
    
    def start(self):
        self.prepare_atom_sexps()
        self.search()
    
    def is_completed(self):
        return len(self.searched_queries) >= self.top_k
    
    def prepare_atom_sexps(self):
        for atom_path in self.atom_paths:
            search_item = self.path_to_original_search_item(atom_path)
            if search_item is not None:
                self.atom_sexps.append(search_item)
        
        for (idx, atom_sexp) in enumerate(self.atom_sexps):
            self.sexp_priority_queue.put((
                - atom_sexp.precision,
                self.sexp_priority_queue.qsize(), # precision 相同时，按照插入队列的顺序来排序
                SearchItem(
                    atom_sexp.sexp,
                    atom_sexp.execution_result,
                    atom_sexp.recall,
                    atom_sexp.precision,
                    depth=1, # Original Search Item
                    atom_visited_index=[idx]
                )
            ))
        # 当前的实现: arg_sexps 不能够作为原子 S-expression, ARGMAX / ARGMIN 操作必须作用于一个集合上

    def search(self):
        if self.sexp_priority_queue.qsize() == 0:
            return
        if self.is_completed():
            return

        while self.sexp_priority_queue.qsize() > 0:
            # 出队时判断 Prec. 是否为 1.0
            current_search_item:SearchItem = self.sexp_priority_queue.get()[2]
            if math.isclose(current_search_item.precision, 1.0):
                self.searched_queries.append(
                    copy.deepcopy(current_search_item)
                )
                if self.is_completed():
                    return
                continue # Prec. = 1, 不允许继续拓展

            print(f"queue size: {self.sexp_priority_queue.qsize()}")
            if current_search_item.depth >= self.depth_upper_bound: # 超出深度约束，不再进行拓展
                continue
            assert math.isclose(current_search_item.recall, 1.0)
            prev_precision = current_search_item.precision

            for (idx, atom_sexp) in enumerate(self.atom_sexps):
                if idx in current_search_item.atom_visited_index:
                    continue
                tmp_sexp = AND(current_search_item.sexp, atom_sexp.sexp)
                new_search_item = self.execute_tmp_sexp(
                    tmp_sexp, prev_precision, current_search_item,
                    new_atom_index=[idx],
                    new_arg_index=None
                )
                if new_search_item:
                    self.sexp_priority_queue.put((
                        - new_search_item.precision,
                        self.sexp_priority_queue.qsize(),
                        new_search_item
                    ))
            
            for (idx, arg_path) in enumerate(self.arg_paths):
                if idx in current_search_item.arg_visited_index:
                    continue
                
                tmp_sexp = ARG(arg_path["function"], current_search_item.sexp, arg_path["relation"])
                new_search_item = self.execute_tmp_sexp(
                    tmp_sexp, prev_precision, current_search_item,
                    new_atom_index=None,
                    new_arg_index=[idx]
                )

                if new_search_item:
                    self.sexp_priority_queue.put((
                        - new_search_item.precision,
                        self.sexp_priority_queue.qsize(),
                        new_search_item
                    ))
            

    def path_to_original_search_item(self, path):
        """
        original search item, 我们将他们的深度(depth 定义为 1)
        """
        sexp = path["s_expression"]
        sparql = sexp_to_sparql(sexp)
        tmp_res = self.sparql_querier.get_execution_result_one_variable(sparql)

        precision, recall, f1 = get_PRF1(tmp_res, self.answer_mids)
        if not (math.isclose(recall, 1.0)): # Recall = 1 是加入搜索的前提
            return None
        
        return SearchItem(
            sexp,
            tmp_res,
            recall,
            precision
        )
    
    def execute_tmp_sexp(
        self, 
        tmp_sexp, 
        prev_precision, 
        current_search_item:SearchItem,
        new_atom_index,
        new_arg_index,
    ):
        """
        执行组合得到的 tmp S-expression, 返回 new SearchItem, 以及是否停止探测的标志位
        @param new_atom_index: list or None
        @param new_arg_index: list or None
        @return SearchItem
        """
        sparql = sexp_to_sparql(tmp_sexp)
        tmp_execution_result = self.sparql_querier.get_execution_result_one_variable(
            sparql
        )
        precision, recall, f1 = get_PRF1(tmp_execution_result, self.answer_mids)
        if not math.isclose(recall, 1.0):
            return None
        if precision <= prev_precision:
            return None
    
        new_atom_visited_index = copy.deepcopy(current_search_item.atom_visited_index)
        if new_atom_index is not None:
            new_atom_visited_index.extend(new_atom_index)
        
        new_arg_visited_index = copy.deepcopy(current_search_item.arg_visited_index)
        if new_arg_index is not None:
            new_arg_visited_index.extend(new_arg_index)
        
        new_search_item = SearchItem(
            tmp_sexp, 
            tmp_execution_result, 
            recall,
            precision, 
            depth=current_search_item.depth + 1,
            atom_visited_index=new_atom_visited_index,
            arg_visited_index=new_arg_visited_index
        )
        if math.isclose(precision, 1.0):
            return new_search_item
        else:
            return new_search_item

    def get_searched_queries(self):
        # 我们暂时不会使用 Precision < 1 的 Simulated Query
        return self.searched_queries

class _State:
    def __init__(self, current_sexp=None, current_entity=None, tmp_sexp=None):
        '''从 grounded_item 出发，经过 current_sexp，即可到达 current_entity(但可能不止 current_entity)'''
        self.current_sexp = current_sexp
        self.current_entity = current_entity
        '''暂存，例如最后一跳的 SPARQL 查询结果就是个 S-expression'''
        self.tmp_sexp = tmp_sexp

class DetectionInstance:
    def __init__(
        self, question_id, question, sub_question_list:List, grounded_item_list:List,
        answer, item_to_label, logger, sparql_querier:SparqlOdbcQuerier, 
        multivariate_pattern_bi_encoder_path=None, multivariate_pattern_list_path=None, 
        multivariate_pattern_serialized_list_path=None, top_k_pattern=None,
        depth_upper_bound=4, top_k=3, enumeration_beam_size=5, use_question_info=True
    ):
        """
        @param sub_question_list 和 grounded_item_list 长度应该对应，按照下标对齐
        @param enumeration_beam_size: 存在嵌套的子问题，Atomic Query 需要 Beam Search, 这里给出了 Beam Size
        @param multivariate_pattern_list_path: 原始的 pattern list
        @param multivariate_pattern_list_serialized_path: 序列化后的 pattern list, 关系被替换成其 label
        消融实验:
        @param multivariate_pattern_bi_encoder_path:
            - freebase_2024_03_07: 远程监督数据微调后的检索模型
            - distilbert-base-uncased: 未经微调的检索模型
            - None: 不做检索
        @param use_question_info:
            - False: 不使用问题信息，且不做分解
        """
        self.kb_type = KB_TYPE.FREEBASE
        self.question_id = question_id
        self.question = question
        self.sub_question_list = sub_question_list
        self.grounded_item_list = grounded_item_list
        self.searched_queries = list()
        if len(answer) == 0: # 无能为力
            self.skip_flag = True
            return
        self.skip_flag = False
        self.answer = list(answer)
        self.answer_mid = set([
            ans["mid"] for ans in self.answer
        ])
        self.anchor_answer = self.answer[0]
        self.item_to_label = item_to_label
        self.logger = logger
        self.sparql_querier = sparql_querier
        self.depth_upper_bound = depth_upper_bound
        self.top_k = top_k
        self.enumeration_beam_size = enumeration_beam_size
        self.sentence_bert_ranker = SentenceSimilarityPredictor.instance()
        self.count_flag = False
        if len(self.answer_mid) == 1:
            try:
                answer_mid = list(self.answer_mid)[0]
                answer_mid = answer_mid.replace('^^<http://www.w3.org/2001/XMLSchema#decimal>', '').replace('^^<http://www.w3.org/2001/XMLSchema#integer>', '').replace('^^<http://www.w3.org/2001/XMLSchema#float>', '')
                if (answer_mid.startswith('"')) and (answer_mid.endswith('"')):
                    answer_mid = answer_mid[1:-1]
                self.answer_num = int(answer_mid)
                self.count_flag = True
                self.answer_types = [
                    item for item_list in self.grounded_item_list for item in item_list
                    if item['type'] == 'class'
                ] # 链接到的所有 class, 都作为候选答案类型
                self.answer_types = set([
                    item['mid'] for item in self.answer_types
                ]) # 去重
                self.answer_types = [{
                    "mid": mid,
                    'type': 'class'
                } for mid in self.answer_types]
            except:
                self.count_flag = False
        if multivariate_pattern_bi_encoder_path is not None:
            self.bi_encoder_inferencor = BiEncoderInferencor.instance(
                model_path=multivariate_pattern_bi_encoder_path,
                pattern_serialized_list_path=multivariate_pattern_serialized_list_path
            ) # 传入序列化后的 pattern list
            self.pattern_list = load_json(multivariate_pattern_list_path) # 未序列化的 pattern list
            self.top_k_pattern = top_k_pattern
        else:
            self.bi_encoder_inferencor = None
        self.use_question_info = use_question_info
        if not self.use_question_info: # 不做分解和排序
            self.sub_question_list = [
                [question]
            ]
            self.grounded_item_list = combine_grounded_items(self.grounded_item_list)

    
    def get_searched_queries(self):
        return self.searched_queries
    
    def is_complete(self):
        return len(self.searched_queries) >= self.top_k
    
    def filter_empty_paths(self, path_nested_list):
        '''Empty path: 对于某个子句，我们并没有找到对应的路径，那么也只能忽略这个子句了'''
        return [
            p_list for p_list in path_nested_list
            if len(p_list) > 0
        ]

    def search(self):
        if self.skip_flag:
            return
        atomic_query_list:List[List] = list()
        for _ in range(len(self.sub_question_list)):
            atomic_query_list.append(list())
        
        if self.count_flag:
            for (sub_question_idx, (sub_question, _grounded_item_list)) in enumerate(zip(self.sub_question_list, self.grounded_item_list)):
                '''子问题可能由多个 clause 组成'''
                serialized_sub_question = DELIMETER.join(sub_question)
                if len(sub_question) == 1:
                    for grounded_item in tqdm(_grounded_item_list, desc=f"sub_question_idx: {sub_question_idx}; Enumerating grounded items"):
                        for ans_type in self.answer_types:
                            atomic_query_list[sub_question_idx].extend(
                                self.sparql_querier.query_one_hop_paths(
                                    grounded_item, answer_type=ans_type
                                )
                            )
                            atomic_query_list[sub_question_idx].extend(
                                self.sparql_querier.query_one_hop_paths_reversed(
                                    grounded_item, answer_type=ans_type
                                )
                            )
                    if self.bi_encoder_inferencor is not None:
                        # 先做Pattern检索
                        _, index_list = self.bi_encoder_inferencor.retrieve(serialized_sub_question, self.top_k_pattern)
                        _pattern_list = [self.pattern_list[idx] for idx in index_list.tolist()[0]] # 检索得到未序列化的 pattern
                        # 将 Pattern 根据答案类型和候选实体，实例化成原子查询
                        for _pattern in _pattern_list:
                            relation_list = _pattern.split(PATTERN_DELIMETER)
                            for ans_type in self.answer_types:
                                for (grounded_item_0, grounded_item_1) in itertools.combinations(_grounded_item_list, 2):
                                    atomic_queries = self.sparql_querier.query_multivariate_atomic_queries(
                                        grounded_item_0, grounded_item_1, relation_list, answer_type=ans_type
                                    )
                                    for query in atomic_queries:
                                        query["retrieval_flag"] = True
                                    atomic_query_list[sub_question_idx].extend(atomic_queries)
                    '''Arg paths 不需要遍历 grounded item'''
                    for ans_type in self.answer_types:
                        atomic_query_list[sub_question_idx].extend(
                            self.sparql_querier.get_one_hop_arg_relation(
                                answer_type=ans_type
                            )
                        )
                    if self.use_question_info:
                        '''SentenceBert 做个排序'''
                        sorted_index = self.sentence_bert_ranker.sort_atomic_list_return_index(
                            serialized_sub_question, 
                            [
                                self.encode_s_expression(q)
                                for q in atomic_query_list[sub_question_idx]
                            ], 
                            self.item_to_label
                        )
                        atomic_query_list[sub_question_idx] = [
                            atomic_query_list[sub_question_idx][idx] for idx in sorted_index
                        ]
                    
                    atomic_query_list[sub_question_idx] = sort_atomic_query_list_with_retrieval_flag(atomic_query_list[sub_question_idx])
                else: # 子问题由多个 clause 组成，即存在嵌套现象
                    for grounded_item in tqdm(_grounded_item_list, desc=f"sub_question_idx: {sub_question_idx}; Enumerating grounded items"):
                        multi_hop_path_enumerator = MultiHopPathEnumerator(
                            sub_question, self.kb_type, self.sparql_querier, self.enumeration_beam_size,
                            self.sentence_bert_ranker, self.anchor_answer, self.item_to_label, self.logger,
                            count_flag=True, answer_types=self.answer_types, use_question_info=self.use_question_info
                        )
                        atomic_query_list[sub_question_idx].extend(
                            [{"s_expression": atomic_query} for atomic_query in multi_hop_path_enumerator.enumerate_multi_hop_path(grounded_item)]
                        )
                    
                    if self.bi_encoder_inferencor is not None:
                        # 多跳的情况下，检索的依据是最后一个子问题
                        _, index_list = self.bi_encoder_inferencor.retrieve(sub_question[-1], self.top_k_pattern)
                        _pattern_list = [self.pattern_list[idx] for idx in index_list.tolist()[0]]
                        # 重新声明一个 MultiHopPathEnumerator; 每个 MultiHopPathEnumerator 有一些内部状态，尽量还是分开
                        for grounded_item_pair in tqdm(itertools.combinations(_grounded_item_list, 2)):
                            multi_hop_path_enumerator = MultiHopPathEnumerator(
                                sub_question, self.kb_type, self.sparql_querier, self.enumeration_beam_size,
                                self.sentence_bert_ranker, self.anchor_answer, self.item_to_label, self.logger,
                                count_flag=True, answer_types=self.answer_types, use_question_info=self.use_question_info
                            )
                            atomic_query_list[sub_question_idx].extend(
                                [{"s_expression": atomic_query} for atomic_query in multi_hop_path_enumerator.enumerate_multi_hop_path_with_multiproperty_pattern(grounded_item_pair, _pattern_list)]
                            )
                    
                    if self.use_question_info:
                        '''SentenceBert 做个排序'''
                        sorted_index = self.sentence_bert_ranker.sort_atomic_list_return_index(
                            serialized_sub_question, 
                            [self.encode_s_expression(q) for q in atomic_query_list[sub_question_idx]], 
                            self.item_to_label
                        )
                        atomic_query_list[sub_question_idx] = [
                            atomic_query_list[sub_question_idx][idx] for idx in sorted_index
                        ]

                    atomic_query_list[sub_question_idx] = sort_atomic_query_list_with_retrieval_flag(atomic_query_list[sub_question_idx])
            
            all_atomic_queries = self.filter_empty_paths(atomic_query_list)
            depth_first_searcher = DepthFirstSearchForCount(
                self.question, self.item_to_label, all_atomic_queries,
                self.answer_num, self.sparql_querier, self.logger, self.sentence_bert_ranker,
                self.kb_type, self.depth_upper_bound, self.top_k, self.use_question_info
            )
            depth_first_searcher.start()
            searched_queries = depth_first_searcher.get_searched_queries()
            # 去重有点困难，如果这一次搜索得到的查询数量更多，我们就更新一下
            if len(searched_queries) > len(self.searched_queries):
                self.searched_queries = searched_queries
        
        '''如果视作 count 之后的搜索结果为空，则将答案视作一个 literal, 再做一次搜索'''
        if len(self.searched_queries) == 0:
            for (sub_question_idx, (sub_question, _grounded_item_list)) in enumerate(zip(self.sub_question_list, self.grounded_item_list)):
                '''子问题可能由多个 clause 组成'''
                serialized_sub_question = DELIMETER.join(sub_question)
                if len(sub_question) == 1:
                    for grounded_item in tqdm(_grounded_item_list, desc=f"sub_question_idx: {sub_question_idx}; Enumerating grounded items"):
                        atomic_query_list[sub_question_idx].extend(
                            self.sparql_querier.query_one_hop_paths(
                                grounded_item, answer_entity=self.anchor_answer
                            )
                        )
                        atomic_query_list[sub_question_idx].extend(
                            self.sparql_querier.query_one_hop_paths_reversed(
                                grounded_item, answer_entity=self.anchor_answer
                            )
                        )
                    if self.bi_encoder_inferencor is not None:
                        # 先做Pattern检索
                        _, index_list = self.bi_encoder_inferencor.retrieve(serialized_sub_question, self.top_k_pattern)
                        _pattern_list = [self.pattern_list[idx] for idx in index_list.tolist()[0]]
                        # 将 Pattern 根据答案和候选实体，实例化成原子查询
                        for _pattern in _pattern_list:
                            relation_list = _pattern.split(PATTERN_DELIMETER)
                            for (grounded_item_0, grounded_item_1) in tqdm(itertools.combinations(_grounded_item_list, 2), desc="遍历 grounded item 的组合，做多元关系查询"):
                                atomic_queries = self.sparql_querier.query_multivariate_atomic_queries(
                                    grounded_item_0, grounded_item_1, relation_list, answer_entity=self.anchor_answer
                                )
                                for query in atomic_queries:
                                    query["retrieval_flag"] = True
                                atomic_query_list[sub_question_idx].extend(atomic_queries)
                    '''Arg paths 不需要遍历 grounded item'''
                    atomic_query_list[sub_question_idx].extend(
                        self.sparql_querier.get_one_hop_arg_relation(
                            answer_entity=self.anchor_answer
                        )
                    )
                    if self.use_question_info:
                        '''SentenceBert 做个排序'''
                        sorted_index = self.sentence_bert_ranker.sort_atomic_list_return_index(
                            serialized_sub_question, 
                            [
                                self.encode_s_expression(q)
                                for q in atomic_query_list[sub_question_idx]
                            ], 
                            self.item_to_label
                        )
                        atomic_query_list[sub_question_idx] = [
                            atomic_query_list[sub_question_idx][idx] for idx in sorted_index
                        ]
                    
                    atomic_query_list[sub_question_idx] = sort_atomic_query_list_with_retrieval_flag(atomic_query_list[sub_question_idx])
                else: # 子问题由多个 clause 组成，即存在嵌套现象
                    for grounded_item in tqdm(_grounded_item_list, desc=f"sub_question_idx: {sub_question_idx}; Enumerating grounded items"):
                        multi_hop_path_enumerator = MultiHopPathEnumerator(
                            sub_question, self.kb_type, self.sparql_querier, self.enumeration_beam_size,
                            self.sentence_bert_ranker, self.anchor_answer, self.item_to_label, self.logger,
                            count_flag=False, answer_types=None, use_question_info=self.use_question_info
                        ) # 这里是做非 COUNT 类型的搜索，故 count_flag 传 False
                        atomic_query_list[sub_question_idx].extend(
                            [{"s_expression": atomic_query} for atomic_query in multi_hop_path_enumerator.enumerate_multi_hop_path(grounded_item)]
                        ) 
                    
                    if self.bi_encoder_inferencor is not None:
                        # 多跳的情况下，检索的依据是最后一个子问题
                        _, index_list = self.bi_encoder_inferencor.retrieve(sub_question[-1], self.top_k_pattern)
                        _pattern_list = [self.pattern_list[idx] for idx in index_list.tolist()[0]]
                        # 重新声明一个 MultiHopPathEnumerator; 每个 MultiHopPathEnumerator 有一些内部状态，尽量还是分开
                        for grounded_item_pair in tqdm(itertools.combinations(_grounded_item_list, 2)):
                            multi_hop_path_enumerator = MultiHopPathEnumerator(
                                sub_question, self.kb_type, self.sparql_querier, self.enumeration_beam_size,
                                self.sentence_bert_ranker, self.anchor_answer, self.item_to_label, self.logger,
                                count_flag=False, answer_types=None, use_question_info=self.use_question_info
                            ) # 这里是做非 COUNT 类型的搜索，故 count_flag 传 False
                            atomic_query_list[sub_question_idx].extend(
                                [{"s_expression": atomic_query} for atomic_query in multi_hop_path_enumerator.enumerate_multi_hop_path_with_multiproperty_pattern(grounded_item_pair, _pattern_list)]
                            )

                    if self.use_question_info:   
                        '''SentenceBert 做个排序'''
                        sorted_index = self.sentence_bert_ranker.sort_atomic_list_return_index(
                            serialized_sub_question, 
                            [self.encode_s_expression(q) for q in atomic_query_list[sub_question_idx]], 
                            self.item_to_label
                        )
                        atomic_query_list[sub_question_idx] = [
                            atomic_query_list[sub_question_idx][idx] for idx in sorted_index
                        ]

                    atomic_query_list[sub_question_idx] = sort_atomic_query_list_with_retrieval_flag(atomic_query_list[sub_question_idx])
            
            all_atomic_queries = self.filter_empty_paths(atomic_query_list)
            depth_first_searcher = DepthFirstSearch(
                self.question, self.item_to_label, all_atomic_queries,
                self.answer, self.sparql_querier, self.logger, self.sentence_bert_ranker,
                self.kb_type, self.depth_upper_bound, self.top_k, self.use_question_info
            )
            depth_first_searcher.start()
            searched_queries = depth_first_searcher.get_searched_queries()
            # 去重有点困难，如果这一次搜索得到的查询数量更多，我们就更新一下
            if len(searched_queries) > len(self.searched_queries):
                self.searched_queries = searched_queries
    
    def supply_label(self, item_mid):
        if item_mid not in self.item_to_label:
            if self._get_item_type(item_mid) == "entity":
                label = self.sparql_querier.get_friendly_name({
                    "mid": item_mid, "type": "entity"
                })
                self.item_to_label[item_mid] = label
            elif self._get_item_type(item_mid) == "literal":
                self.item_to_label[item_mid] = extract_value_from_literal(item_mid)
    
    def _get_item_type(self, item_mid):
        if self.kb_type is KB_TYPE.WIKIDATA:
            return convert_wikidata_type(WikidataConstantForConstruction.get_constant_type(item_mid))
        elif self.kb_type is KB_TYPE.FREEBASE:
            return convert_freebase_type(FreebaseConstantForConstruction.get_constant_type(item_mid))
        else:
            raise NotImplementedError(f"kb_type: {self.kb_type}")
    
    def encode_s_expression(self, sexp):
        """
        形式 1: {"s_expression": }
        形式 2: {"function", "relation}
        """
        if "s_expression" in sexp:
            return sexp["s_expression"]
        else:
            return f"{sexp['function']}{DELIMETER}{sexp['relation']}"

class DepthFirstSearch:
    def __init__(
        self, question, item_to_label, all_atomic_queries,
        answer, sparql_querier:Union[SparqlOdbcQuerier, SparqlOdbcQuerierWikidata], logger, sentence_bert_ranker:SentenceSimilarityPredictor, kb_type,
        depth_upper_bound=4, top_k=3, use_question_info=True
    ):
        self.question = question
        self.item_to_label = item_to_label
        self.all_atomic_queries = all_atomic_queries

        self.answer_mids = set([
            ans["mid"] for ans in answer
        ])
        self.sparql_querier = sparql_querier
        self.logger = logger
        self.sentence_bert_ranker = sentence_bert_ranker
        self.kb_type = kb_type
        self.depth_upper_bound = depth_upper_bound
        self.top_k = top_k
        self.searched_queries = list()
        self.atomic_queries_for_search:List = list() # 展平后
        self.use_question_info = use_question_info
    
    def start(self):
        self.search()
    
    def is_completed(self):
        return len(self.searched_queries) >= self.top_k

    def get_searched_queries(self):
        # 我们暂时不会使用 Precision < 1 的 Simulated Query
        return self.searched_queries
    
    def get_dummy_search_item(self):
        '''各项属性都赋默认值的 SearchItem'''
        return SearchItem(
            sexp=None,
            execution_result=list(),
            recall=0.0,
            precision=0.0,
            depth=0
        )
    
    def encode_s_expression(self, sexp):
        """
        形式 1: {"s_expression": }
        形式 2: {"function", "relation}
        """
        if "s_expression" in sexp:
            return sexp["s_expression"]
        else:
            return f"{sexp['function']}{DELIMETER}{sexp['relation']}"

    def get_queries_for_search(self, selected_queries, all_queries):
        '''Sexp 作为判断路径是否相等的依据；应该不存在 Sexp 相同的两条路径吧'''
        selected_sexp_list = [
            self.encode_s_expression(q)
            for q in selected_queries
        ] # 使用 list, 维持顺序，all_queries 之前可能是排好序的
        queries_for_search = [
            q for q in all_queries
            if (self.encode_s_expression(q) not in selected_sexp_list)
        ]
        return queries_for_search
    
    def search(self):
        """
        Step 1: 要求搜索得到的查询，对于每个子句，至少包含一个 Atomic Query; 通过搜索的起点设置来完成
        Step 2: Step 1 没搜满，则执行暴力搜索；把所有 Atomic Query 构成一个 list, 暴力搜索
        """
        if self.is_completed():
            return
        for query_tup in itertools.product(*self.all_atomic_queries): # 必须覆盖的子句，遍历这些子句下的路径组合
            if self.is_completed():
                return
            search_item = self.get_dummy_search_item()
            skip_flag = False
            for query in query_tup:
                if search_item.sexp is None:
                    if "function" in query:
                        '''ARGMAX / ARGMIN 无法作为初始 query, 跳过'''
                        skip_flag = True
                        break
                    tmp_sexp = query["s_expression"]
                else:
                    if "s_expression" in query:
                        tmp_sexp = AND(search_item.sexp, query["s_expression"])
                    else:
                        tmp_sexp = ARG(query["function"], search_item.sexp, query["relation"])
                search_item = self.execute_tmp_sexp(
                    tmp_sexp, 0.0, search_item
                )
                if search_item is None:
                    skip_flag = True
                    break
            if skip_flag:
                continue
            self.atomic_queries_for_search = self.get_queries_for_search(
                query_tup, flatten_list(copy.deepcopy(self.all_atomic_queries))
            )
            if self.use_question_info:
                sorted_index = self.sentence_bert_ranker.sort_atomic_list_return_index(
                    self.question,
                    [self.encode_s_expression(q) for q in self.atomic_queries_for_search],
                    self.item_to_label
                )
                self.atomic_queries_for_search = [
                    self.atomic_queries_for_search[idx] for idx in sorted_index
                ]
            self.atomic_queries_for_search = sort_atomic_query_list_with_retrieval_flag(self.atomic_queries_for_search)
            self.dfs(search_item, 0)
        
        '''Step 2'''
        if self.is_completed():
            return
        self.atomic_queries_for_search = flatten_list(copy.deepcopy(self.all_atomic_queries))
        if self.use_question_info:
            sorted_index = self.sentence_bert_ranker.sort_atomic_list_return_index(
                self.question,
                [self.encode_s_expression(q) for q in self.atomic_queries_for_search],
                self.item_to_label
            )
            self.atomic_queries_for_search = [
                self.atomic_queries_for_search[idx] for idx in sorted_index
            ]

        self.atomic_queries_for_search = sort_atomic_query_list_with_retrieval_flag(self.atomic_queries_for_search)
        self.dfs(self.get_dummy_search_item(), 0)
    
    def convert_sexp_to_sparql(self, sexp):
        if self.kb_type is KB_TYPE.FREEBASE:
            return sexp_to_sparql(sexp)
        elif self.kb_type is KB_TYPE.WIKIDATA:
            return sexp_to_sparql_wikidata(sexp)
        else:
            raise NotImplementedError(f"kb_type: {self.kb_type}")
    
    def get_sparql_execution_results(self, sparql):
        if self.kb_type is KB_TYPE.WIKIDATA:
            return self.sparql_querier.get_execution_result_one_variable_sparql_wrapper(sparql)
        elif self.kb_type is KB_TYPE.FREEBASE:
            return self.sparql_querier.get_execution_result_one_variable_sparql_wrapper(sparql)
        else:
            raise Exception(f"kb_type: {self.kb_type}")

    def execute_tmp_sexp(
        self, 
        tmp_sexp, 
        prev_precision, 
        current_search_item:SearchItem,
    ):
        """
        执行组合得到的 tmp S-expression, 返回 new SearchItem, 以及是否停止探测的标志位
        @param new_atom_index: list or None
        @param new_arg_index: list or None
        @return SearchItem
        """
        sparql = self.convert_sexp_to_sparql(tmp_sexp)
        tmp_execution_result = self.get_sparql_execution_results(sparql)
        precision, recall, f1 = get_PRF1(tmp_execution_result, self.answer_mids)
        if not math.isclose(recall, 1.0):
            return None
        if precision <= prev_precision:
            return None
        
        new_search_item = SearchItem(
            tmp_sexp, 
            tmp_execution_result, 
            recall,
            precision, 
            depth=current_search_item.depth + 1,
        )
        return new_search_item
    
    def add_searched_query(self, current_search_item):
        """补充去重逻辑"""
        existed_s_expression = [
            item.sexp for item in self.searched_queries
        ]
        if current_search_item.sexp in existed_s_expression:
            return # S-expression 重复
        else:
            self.searched_queries.append(
                copy.deepcopy(current_search_item)
            )
    
    def dfs(self, current_search_item:SearchItem, atom_path_index):
        """
        @param atom_sexp_index: 可访问的下标起始位置(闭区间)
        @param arg_path_index: 可访问的下标起始位置(闭区间)
        """
        if math.isclose(current_search_item.precision, 1.0): # Prec. = 1.0, 不会继续拓展这个 sexp
            self.add_searched_query(current_search_item)
            return
        if current_search_item.depth >= self.depth_upper_bound: # 超出深度限制，不再拓展
            return
        if self.is_completed():
            return
        prev_precision = current_search_item.precision
        
        for idx in range(atom_path_index, len(self.atomic_queries_for_search)):
            if self.is_completed():
                return
            if "s_expression" in self.atomic_queries_for_search[idx]:
                if current_search_item.sexp is None:
                    tmp_sexp = self.atomic_queries_for_search[idx]["s_expression"]
                else:
                    tmp_sexp = AND(current_search_item.sexp, self.atomic_queries_for_search[idx]["s_expression"])
            else:
                if current_search_item.sexp is None:
                    '''arg path 无法作为初始 S-expression'''
                    continue
                else:
                    tmp_sexp = ARG(
                        self.atomic_queries_for_search[idx]["function"], 
                        current_search_item.sexp, 
                        self.atomic_queries_for_search[idx]["relation"]
                    )
            new_search_item = self.execute_tmp_sexp(
                tmp_sexp, prev_precision, current_search_item
            )
            if new_search_item: # Prec. 上升 且 Recall 为 1
                self.dfs(new_search_item, idx + 1)

class MultiHopPathEnumerator:
    def __init__(
        self, sub_question, kb_type, sparql_querier: Union[SparqlOdbcQuerier,SparqlOdbcQuerierWikidata], enumeration_beam_size,
        sentence_bert_ranker:SentenceSimilarityPredictor, anchor_answer, item_to_label, logger,
        count_flag, answer_types=None, use_question_info=False
    ):
        self.sub_question = sub_question
        self.kb_type = kb_type
        self.serialized_sub_question = DELIMETER.join(sub_question)
        
        self.sparql_querier = sparql_querier
        self.enumeration_beam_size = enumeration_beam_size
        self.sentence_bert_ranker = sentence_bert_ranker
        self.anchor_answer = anchor_answer
        self.item_to_label = item_to_label
        self.logger = logger
        self.count_flag = count_flag
        if self.count_flag:
            self.answer_types = answer_types
        self.use_question_info = use_question_info
        
        self.results = set()
    
    def convert_sexp_to_sparql(self, sexp):
        if self.kb_type is KB_TYPE.FREEBASE:
            return sexp_to_sparql(sexp)
        elif self.kb_type is KB_TYPE.WIKIDATA:
            return sexp_to_sparql_wikidata(sexp)
        else:
            raise NotImplementedError(f"kb_type: {self.kb_type}")
    
    def enumerate_multi_hop_path(self, grounded_item):
        initial_state_list = [
            _State(current_sexp=grounded_item['mid'], current_entity=grounded_item)
        ]
        '''因为从 grounded item 出发，因此 sub_question 应该倒过来'''  
        self._rank_one_hop_relations(initial_state_list, self.sub_question[::-1], 0)
        return self.results
    
    def enumerate_multi_hop_path_with_multiproperty_pattern(self, grounded_item_pair, pattern_list):
        """
        针对多元关系图模式的多跳搜索
        这个函数中需要首先完成第一跳的搜索，然后再调用 _rank_one_hop_relations (因为第一跳涉及两个 grounded_item, 后面几跳都只涉及一个 grounded_item)
        """
        if len(grounded_item_pair) != 2:
            raise Exception(f"grounded_item_pair: {grounded_item_pair}")
        first_sub_question = self.sub_question[-1] # 从 grounded item 出发倒推，故把子问题的顺序倒过来
        self.supply_label(grounded_item_pair[0]['mid'])
        self.supply_label(grounded_item_pair[1]['mid'])
        sexp_list = list()
        # 这边要求第一跳必须是一个 CVT pattern; 不是 CVT pattern 的情况在之前的搜索中（输入单个 grounded item）已经覆盖了
        next_state_list_validated:List[_State] = list() 
        for _pattern in pattern_list:
            if self.kb_type is KB_TYPE.WIKIDATA:
                [p_prop, pq_prop] = _pattern.split(PATTERN_DELIMETER)
                sexp_list.extend(
                    self.sparql_querier.query_multivariate_relations(
                        grounded_item_pair[0], grounded_item_pair[1], p_prop, pq_prop
                    )
                )
            elif self.kb_type is KB_TYPE.FREEBASE:
                relation_list = _pattern.split(PATTERN_DELIMETER)
                sexp_list.extend(
                    self.sparql_querier.query_multivariate_relations(
                        grounded_item_pair[0], grounded_item_pair[1], relation_list
                    )
                )
        
        for sexp in sexp_list:
            try:
                sparql = self.convert_sexp_to_sparql(sexp['s_expression'])
                execution_result = self.get_sparql_execution_results(sparql)
            except Exception as e:
                self.logger.error(e)
                execution_result = []
            if (not execution_result) or len(execution_result) == 0: # 执行结果为空的，可以忽略
                continue
            if len(execution_result) > self.enumeration_beam_size:
                execution_result = random.sample(execution_result, self.enumeration_beam_size)
            
            for next_entity in execution_result:
                item_type = self._get_item_type(next_entity)
                if item_type is None:
                    continue
                next_state_list_validated.append(_State(
                    current_sexp=sexp["s_expression"],
                    current_entity={"mid": next_entity, "type": item_type},
                    tmp_sexp=None
                ))
        if self.use_question_info:
            '''对于所有 state 排序，选择 top enumeration_beam_size'''
            state_sorted_index = self.sentence_bert_ranker.sort_atomic_list_return_index(
                first_sub_question, [state.current_sexp for state in next_state_list_validated], self.item_to_label
            )[:self.enumeration_beam_size] # 只根据第一个问题进行排序
            sorted_state_list:List[_State] = [
                next_state_list_validated[idx] for idx in state_sorted_index
            ]
        else:
            sorted_state_list:List[_State] = next_state_list_validated[:self.enumeration_beam_size] # 不排序，直接取前 5 (SPARQL 查询结果的前 5)
        # 1. 把子问题倒过来
        clause_idx = 0
        clause_list = self.sub_question[::-1]
        if (clause_idx + 1) == len(clause_list) - 1:
            if self.count_flag: # count_flag = True, 则最终目标为类型
                for answer_class in self.answer_types:
                    self._rank_one_hop_relations(sorted_state_list, clause_list, clause_idx + 1, end_class=answer_class)
            else: # count_flag = False, 则最终目标为实体
                self._rank_one_hop_relations(sorted_state_list, clause_list, clause_idx + 1, end_entity=self.anchor_answer)
        else:
            self._rank_one_hop_relations(sorted_state_list, clause_list, clause_idx + 1)
        
        return self.results

    def get_sparql_execution_results(self, sparql):
        if self.kb_type is KB_TYPE.WIKIDATA:
            return self.sparql_querier.get_execution_result_one_variable_sparql_wrapper(sparql)
        elif self.kb_type is KB_TYPE.FREEBASE:
            return self.sparql_querier.get_execution_result_one_variable_sparql_wrapper(sparql)
        else:
            raise Exception(f"kb_type: {self.kb_type}")

    def supply_label(self, item_mid):
        if item_mid not in self.item_to_label:
            if self._get_item_type(item_mid) == "entity":
                label = self.sparql_querier.get_friendly_name({
                    "mid": item_mid, "type": "entity"
                })
                self.item_to_label[item_mid] = label
            elif self._get_item_type(item_mid) == "literal":
                self.item_to_label[item_mid] = extract_value_from_literal(item_mid)
    
    def _rank_one_hop_relations(self, current_state_list: List[_State], clause_list, clause_idx, end_entity=None, end_class=None):
        """
        顺序倒过来，从 grounded item 出发，需要到达 anchor answer
        """
        def combine_sexp(tmp_sexp, current_sexp, grounded_item_rep):
            '''
            从 anchor answer 出发，经过 relation_tuple 构成的路径，即可到达 current_entity
            同理可构造 anchor answer 相对于 current_entity 的 S-expression

            (r1, r2, ARGMAX r3/r4, r5) 格式
            '''
            if tmp_sexp is None:
                return current_sexp
            elif "function" in tmp_sexp:
                current_sexp = ARG(
                    tmp_sexp["function"], 
                    current_sexp,
                    tmp_sexp["relation"]
                )
            else:
                current_sexp = tmp_sexp["s_expression"].replace(grounded_item_rep, current_sexp)
            return current_sexp
        
        if end_entity is not None:
            '''最后一跳'''
            next_state_list:List[_State] = copy.deepcopy(current_state_list) # 上一轮次保留的状态，也参加排序
            for current_state in current_state_list: 
                self.supply_label(current_state.current_entity['mid'])
                sexp_list = self.sparql_querier.query_one_hop_paths(
                    grounded_item=current_state.current_entity,
                    answer_entity=end_entity
                )
                sexp_list_reversed = self.sparql_querier.query_one_hop_paths_reversed(
                    grounded_item=current_state.current_entity,
                    answer_entity=end_entity
                )

                arg_sexp_list = self.sparql_querier.get_one_hop_arg_relation(answer_entity=current_state.current_entity)
                for lst in [sexp_list, sexp_list_reversed]:
                    for sexp in lst:
                        next_state_list.append(_State(
                            current_sexp=current_state.current_sexp,
                            current_entity=current_state.current_entity,
                            tmp_sexp=sexp
                        ))
                for sexp in arg_sexp_list:
                    next_state_list.append(_State(
                        current_sexp=current_state.current_sexp,
                        current_entity=current_state.current_entity,
                        tmp_sexp=sexp
                    ))
                
            next_state_list_validated:List[_State] = list()
            for state in next_state_list:
                try: # 查询的执行结果，需要包含 end_entity
                    sexp = combine_sexp(state.tmp_sexp, state.current_sexp, state.current_entity['mid'])
                    sparql = self.convert_sexp_to_sparql(sexp)
                    execution_result = self.get_sparql_execution_results(sparql)
                    if end_entity['mid'] in execution_result:
                        next_state_list_validated.append(state)
                except Exception as e: # 可能出现无法执行的查询
                    self.logger.error(f"S-expression combination error: {e}")
            
            if self.use_question_info:
                '''使用完整的 S-expression (anchor answer 到 grounded item) 来做排序; 对于所有 state 统一做排序'''
                state_sorted_index = self.sentence_bert_ranker.sort_atomic_list_return_index(
                    self.serialized_sub_question, [combine_sexp(state.tmp_sexp, state.current_sexp, state.current_entity['mid']) for state in next_state_list_validated], self.item_to_label
                )[:self.enumeration_beam_size]
                sorted_state_list:List[_State] = [
                    next_state_list_validated[idx] for idx in state_sorted_index
                ]
            else:
                sorted_state_list = next_state_list_validated[:self.enumeration_beam_size]
            for current_state in sorted_state_list:
                self.results.add(combine_sexp(current_state.tmp_sexp, current_state.current_sexp, current_state.current_entity['mid']))
        elif end_class is not None:
            '''最后一跳'''
            next_state_list:List[_State] = copy.deepcopy(current_state_list) # 上一轮次保留的状态，也参加排序
            for current_state in current_state_list:  
                self.supply_label(current_state.current_entity['mid'])
                sexp_list = self.sparql_querier.query_one_hop_paths(
                    grounded_item=current_state.current_entity,
                    answer_type=end_class
                )
                sexp_list_reversed = self.sparql_querier.query_one_hop_paths_reversed(
                    grounded_item=current_state.current_entity,
                    answer_type=end_class
                )

                arg_sexp_list = self.sparql_querier.get_one_hop_arg_relation(answer_entity=current_state.current_entity)
                for lst in [sexp_list, sexp_list_reversed]:
                    for sexp in lst:
                        next_state_list.append(_State(
                            current_sexp=current_state.current_sexp,
                            current_entity=current_state.current_entity,
                            tmp_sexp=sexp
                        ))
                for sexp in arg_sexp_list:
                    next_state_list.append(_State(
                        current_sexp=current_state.current_sexp,
                        current_entity=current_state.current_entity,
                        tmp_sexp=sexp
                    ))
            '''答案只有类型，无法做验证; 问题不大，Combine Atomic Query 的时候，根据 Precision, Recall 再做筛选'''
            next_state_list_validated:List[_State] = list()
            for state in next_state_list:
                try: # 查询的执行结果，需要包含 end_entity
                    sexp = combine_sexp(state.tmp_sexp, state.current_sexp, state.current_entity['mid'])
                    sparql = self.convert_sexp_to_sparql(sexp)
                    execution_result = self.get_sparql_execution_results(sparql)
                    '''执行结果数量可能非常多，也只能截取一部分做类型的验证了'''
                    execution_result = list(execution_result)[:self.enumeration_beam_size] 
                    skip_flag = False
                    for exec_res in execution_result:
                        if not self.sparql_querier.check_class(
                            {'mid': exec_res, "type": self._get_item_type(exec_res)},
                            end_class
                        ):
                            skip_flag = True
                            break
                    if not skip_flag:
                        next_state_list_validated.append(state)
                    
                except Exception as e: # 可能出现无法执行的查询
                    self.logger.error(f"S-expression combination error: {e}")
            if self.use_question_info:
                '''使用完整的 S-expression (anchor answer 到 grounded item) 来做排序; 对于所有 state 统一做排序'''
                state_sorted_index = self.sentence_bert_ranker.sort_atomic_list_return_index(
                    self.serialized_sub_question, [combine_sexp(state.tmp_sexp, state.current_sexp, state.current_entity['mid']) for state in next_state_list_validated], self.item_to_label
                )[:self.enumeration_beam_size]
                sorted_state_list:List[_State] = [
                    next_state_list_validated[idx] for idx in state_sorted_index
                ]
            else:
                sorted_state_list = next_state_list_validated[:self.enumeration_beam_size]
            for current_state in sorted_state_list:
                self.results.add(combine_sexp(current_state.tmp_sexp, current_state.current_sexp, current_state.current_entity['mid']))
        
        elif (end_entity is None) and (end_class is None):
            next_state_list = copy.deepcopy(current_state_list) # 上一轮次的状态，也参与排序
            for current_state in current_state_list:
                self.supply_label(current_state.current_entity['mid'])
                sexp_list = list()
                sexp_list.extend(self.sparql_querier.get_one_hop_relations(
                    item=current_state.current_entity
                ))
                sexp_list.extend(self.sparql_querier.get_one_hop_relations_reversed(
                    item=current_state.current_entity
                ))
                sexp_list.extend(self.sparql_querier.get_one_hop_arg_relation(
                    answer_entity=current_state.current_entity
                ))

                for sexp in sexp_list:
                    next_state_list.append(_State(
                        current_sexp=current_state.current_sexp,
                        current_entity=current_state.current_entity,
                        tmp_sexp=sexp
                    ))

            next_state_list_validated:List[_State] = list()
            for state in next_state_list:
                try:
                    sexp = combine_sexp(state.tmp_sexp, state.current_sexp, state.current_entity['mid'])
                    sparql = self.convert_sexp_to_sparql(sexp)
                    execution_result = self.get_sparql_execution_results(sparql)
                except Exception as e: # 可能出现执行错误，跳过即可
                    # self.logger.error(f"S-expression execution error: {e}")
                    execution_result = []
                if (not execution_result) or len(execution_result) == 0: # 执行结果为空的，可以忽略
                    continue
                if self.kb_type is KB_TYPE.WIKIDATA:
                    execution_result = [execution_item for execution_item in execution_result if WikidataConstantForConstruction.get_constant_type(execution_item) is not None]
                elif self.kb_type is KB_TYPE.FREEBASE:
                    execution_result = [execution_item for execution_item in execution_result if FreebaseConstantForConstruction.get_constant_type(execution_item) is not None]
                if len(execution_result) > self.enumeration_beam_size:
                    execution_result = random.sample(execution_result, self.enumeration_beam_size) 
                for next_entity in execution_result:
                    item_type = self._get_item_type(next_entity)
                    if item_type is None:
                        # self.logger.error(f"next_entity: {next_entity}, undefined type")
                        continue
                    next_state_list_validated.append(_State(
                        current_sexp=sexp,
                        current_entity={"mid": next_entity, "type": item_type},
                        tmp_sexp=None
                    ))
            if self.use_question_info:
                '''对于所有 state 排序，选择 top 5'''
                state_sorted_index = self.sentence_bert_ranker.sort_atomic_list_return_index(
                    self.serialized_sub_question, [state.current_sexp for state in next_state_list_validated], self.item_to_label
                )[:self.enumeration_beam_size]
                sorted_state_list:List[_State] = [
                    next_state_list_validated[idx] for idx in state_sorted_index
                ]
            else:
                sorted_state_list = next_state_list_validated[:self.enumeration_beam_size]
            if (clause_idx + 1) == len(clause_list) - 1:
                if self.count_flag:
                    for answer_class in self.answer_types:
                        self._rank_one_hop_relations(sorted_state_list, clause_list, clause_idx + 1, end_class=answer_class)
                else:
                    self._rank_one_hop_relations(sorted_state_list, clause_list, clause_idx + 1, end_entity=self.anchor_answer)
            else:
                self._rank_one_hop_relations(sorted_state_list, clause_list, clause_idx + 1)

    def _get_item_type(self, item_mid):
        if self.kb_type is KB_TYPE.WIKIDATA:
            return convert_wikidata_type(WikidataConstantForConstruction.get_constant_type(item_mid))
        elif self.kb_type is KB_TYPE.FREEBASE:
            return convert_freebase_type(FreebaseConstantForConstruction.get_constant_type(item_mid))
        else:
            raise NotImplementedError(f"kb_type: {self.kb_type}")


class DepthFirstSearchForCount:
    '''
    这里假设传入的 paths 都已经排好序了
    对于 COUNT 类型的问题，只能利用集合的大小信息
    - 仍然用 recall, precision 这样的说法，但是 recall, precision 只根据数量来计算
    后处理阶段，需要把 COUNT 操作符加上去
    '''
    def __init__(
        self, question, item_to_label, all_atomic_queries,
        answer_num, sparql_querier:Union[SparqlOdbcQuerier, SparqlOdbcQuerierWikidata], logger, sentence_bert_ranker:SentenceSimilarityPredictor, kb_type,
        depth_upper_bound=4, top_k=3, use_question_info=True
    ):
        self.question = question
        self.item_to_label = item_to_label
        self.all_atomic_queries = all_atomic_queries
        
        self.answer_num = answer_num
        self.logger = logger
        self.sparql_querier = sparql_querier
        self.sentence_bert_ranker = sentence_bert_ranker
        self.kb_type = kb_type
        self.depth_upper_bound = depth_upper_bound # 当前 depth > depth_upper_bound, 不加入优先级队列中
        assert self.depth_upper_bound >= 1
        self.top_k = top_k

        self.searched_queries = list()
        self.atomic_queries_for_search:List = list() # 展平后
        self.use_question_info = use_question_info
    
    def convert_sexp_to_sparql(self, sexp):
        if self.kb_type is KB_TYPE.FREEBASE:
            return sexp_to_sparql(sexp)
        elif self.kb_type is KB_TYPE.WIKIDATA:
            return sexp_to_sparql_wikidata(sexp)
        else:
            raise NotImplementedError(f"kb_type: {self.kb_type}")
    
    def start(self):
        self.search()
    
    def is_completed(self):
        return len(self.searched_queries) >= self.top_k

    def get_searched_queries(self):
        # 我们暂时不会使用 Precision < 1 的 Simulated Query
        return self.searched_queries
    
    def get_dummy_search_item(self):
        '''各项属性都赋默认值的 SearchItem'''
        return SearchItem(
            sexp=None,
            execution_result=list(),
            recall=0.0,
            precision=0.0,
            depth=0
        )
    
    def encode_s_expression(self, sexp):
        """
        形式 1: {"s_expression": }
        形式 2: {"function", "relation}
        """
        if "s_expression" in sexp:
            return sexp["s_expression"]
        else:
            return f"{sexp['function']}{DELIMETER}{sexp['relation']}"

    def get_queries_for_search(self, selected_queries, all_queries):
        '''Sexp 作为判断路径是否相等的依据；应该不存在 Sexp 相同的两条路径吧'''
        selected_sexp_list = [
            self.encode_s_expression(q)
            for q in selected_queries
        ] # 使用 list, 维持顺序，all_queries 之前可能是排好序的
        queries_for_search = [
            q for q in all_queries
            if (self.encode_s_expression(q) not in selected_sexp_list)
        ]
        return queries_for_search

    def search(self):
        """
        Step 1: 要求搜索得到的查询，对于每个子句，至少包含一个 Atomic Query; 通过搜索的起点设置来完成
        Step 2: Step 1 没搜满，则执行暴力搜索；把所有 Atomic Query 构成一个 list, 暴力搜索
        """
        if self.is_completed():
            return
        '''Step 1'''
        for query_tup in itertools.product(*self.all_atomic_queries): # 必须覆盖的子句，遍历这些子句下的路径组合
            if self.is_completed():
                return
            '''path_tup: 每个子句里面抽取一个路径，构成的路径 tup'''
            search_item = self.get_dummy_search_item()
            skip_flag = False
            for query in query_tup:
                if search_item.sexp is None:
                    if "function" in query:
                        '''ARGMAX / ARGMIN 无法作为初始 query, 跳过'''
                        skip_flag = True
                        break
                    tmp_sexp = query["s_expression"]
                else:
                    if "s_expression" in query:
                        tmp_sexp = AND(search_item.sexp, query["s_expression"])
                    else:
                        tmp_sexp = ARG(query["function"], search_item.sexp, query["relation"])
                # 需要检查 Recall = 1, Prec. > 0
                search_item = self.execute_tmp_sexp(
                    tmp_sexp, 0.0, search_item
                )
                if search_item is None:
                    skip_flag = True
                    break
                # 没有加入初始路径中的，就作为其他可用于搜索的路径
            if skip_flag:
                continue
            '''已经设置了来自每个子句的 atomic query 作为搜索起点；再次基础上，还可以允许更多的 atomic query'''
            self.atomic_queries_for_search = self.get_queries_for_search(
                query_tup, flatten_list(copy.deepcopy(self.all_atomic_queries))
            )
            if self.use_question_info:
                sorted_index = self.sentence_bert_ranker.sort_atomic_list_return_index(
                    self.question,
                    [self.encode_s_expression(q) for q in self.atomic_queries_for_search],
                    self.item_to_label
                )
                self.atomic_queries_for_search = [
                    self.atomic_queries_for_search[idx] for idx in sorted_index
                ]
            self.atomic_queries_for_search = sort_atomic_query_list_with_retrieval_flag(self.atomic_queries_for_search)
            self.dfs(search_item, 0)
        '''Step 2'''
        if self.is_completed():
            return
        self.atomic_queries_for_search = flatten_list(copy.deepcopy(self.all_atomic_queries))
        if self.use_question_info:
            sorted_index = self.sentence_bert_ranker.sort_atomic_list_return_index(
                self.question,
                [self.encode_s_expression(q) for q in self.atomic_queries_for_search],
                self.item_to_label
            )
            self.atomic_queries_for_search = [
                self.atomic_queries_for_search[idx] for idx in sorted_index
            ]

        self.atomic_queries_for_search = sort_atomic_query_list_with_retrieval_flag(self.atomic_queries_for_search)
        self.dfs(self.get_dummy_search_item(), 0)

    def get_sparql_execution_results(self, sparql):
        if self.kb_type is KB_TYPE.WIKIDATA:
            return self.sparql_querier.get_execution_result_one_variable_sparql_wrapper(sparql)
        elif self.kb_type is KB_TYPE.FREEBASE:
            return self.sparql_querier.get_execution_result_one_variable_sparql_wrapper(sparql)
        else:
            raise Exception(f"kb_type: {self.kb_type}")

    def execute_tmp_sexp(
        self, 
        tmp_sexp, 
        prev_precision, 
        current_search_item:SearchItem,
    ):
        """
        执行组合得到的 tmp S-expression, 返回 new SearchItem, 以及是否停止探测的标志位
        @param new_atom_index: list or None
        @param new_arg_index: list or None
        @return SearchItem
        """
        sparql = self.convert_sexp_to_sparql(tmp_sexp)
        tmp_execution_result = self.get_sparql_execution_results(sparql)
        precision, recall, f1 = PRF1_for_count(tmp_execution_result, self.answer_num)
        if not math.isclose(recall, 1.0):
            return None
        if precision <= prev_precision:
            return None
        
        new_search_item = SearchItem(
            tmp_sexp, 
            tmp_execution_result, 
            recall,
            precision, 
            depth=current_search_item.depth + 1,
        )
        return new_search_item
    
    def add_count_operation(
        self, 
        current_search_item:SearchItem,
    ):
        """
        执行组合得到的 tmp S-expression, 返回 new SearchItem, 以及是否停止探测的标志位
        @param new_atom_index: list or None
        @param new_arg_index: list or None
        @return SearchItem
        """
        tmp_sexp = COUNT(current_search_item.sexp)
        sparql = self.convert_sexp_to_sparql(tmp_sexp)
        tmp_execution_result = self.get_sparql_execution_results(sparql)
        
        new_search_item = SearchItem(
            tmp_sexp, 
            tmp_execution_result, 
            1.0,
            1.0, # 执行到这里，默认 Precision 和 Recall 都为 1
            depth=current_search_item.depth + 1,
        )
        return new_search_item

    def add_searched_query(self, current_search_item):
        """补充去重逻辑"""
        current_search_item = self.add_count_operation(current_search_item)
        existed_s_expression = [
            item.sexp for item in self.searched_queries
        ]
        if current_search_item.sexp in existed_s_expression:
            return # S-expression 重复
        else:
            self.searched_queries.append(
                copy.deepcopy(current_search_item)
            )

    def dfs(self, current_search_item:SearchItem, atom_path_index):
        """
        @param atom_sexp_index: 可访问的下标起始位置(闭区间)
        """
        if math.isclose(current_search_item.precision, 1.0): # Prec. = 1.0, 不会继续拓展这个 sexp
            '''后处理，加上 COUNT 操作符'''
            self.add_searched_query(current_search_item)
            return
        if current_search_item.depth >= self.depth_upper_bound: # 超出深度限制，不再拓展
            return
        if self.is_completed():
            return
        prev_precision = current_search_item.precision
        
        for idx in range(atom_path_index, len(self.atomic_queries_for_search)):
            if self.is_completed():
                return
            if "s_expression" in self.atomic_queries_for_search[idx]:
                if current_search_item.sexp is None:
                    tmp_sexp = self.atomic_queries_for_search[idx]["s_expression"]
                else:
                    tmp_sexp = AND(current_search_item.sexp, self.atomic_queries_for_search[idx]["s_expression"])
            else:
                if current_search_item.sexp is None:
                    '''arg path 无法作为初始 S-expression'''
                    continue
                else:
                    tmp_sexp = ARG(
                        self.atomic_queries_for_search[idx]["function"], 
                        current_search_item.sexp, 
                        self.atomic_queries_for_search[idx]["relation"]
                    )
            new_search_item = self.execute_tmp_sexp(
                tmp_sexp, prev_precision, current_search_item
            )
            if new_search_item: # Prec. 上升 且 Recall 为 1
                self.dfs(new_search_item, idx + 1)

class DetectionInstanceWikidata:
    def __init__(
        self, question_id, question, sub_question_list:List, grounded_item_list:List,
        answer, item_to_label, logger, sparql_querier:SparqlOdbcQuerierWikidata, 
        multivariate_pattern_bi_encoder_path=None, multivariate_pattern_list_path=None, 
        multivariate_pattern_list_serialized_path=None, top_k_pattern=None,
        depth_upper_bound=4, top_k=3, enumeration_beam_size=5, use_question_info=True
    ):
        """
        @param sub_question_list 和 grounded_item_list 长度应该对应，按照下标对齐
        @param enumeration_beam_size: 存在嵌套的子问题，Atomic Query 需要 Beam Search, 这里给出了 Beam Size
        """
        self.kb_type = KB_TYPE.WIKIDATA
        self.question_id = question_id
        self.question = question
        self.sub_question_list = sub_question_list
        self.grounded_item_list = grounded_item_list
        self.searched_queries = list()
        if len(answer) == 0: # 无能为力
            self.skip_flag = True
            return
        self.skip_flag = False
        self.answer = list(answer)
        self.answer_mid = set([
            ans["mid"] for ans in self.answer
        ])
        self.anchor_answer = self.answer[0]
        self.item_to_label = item_to_label
        self.logger = logger
        self.sparql_querier = sparql_querier
        self.depth_upper_bound = depth_upper_bound
        self.top_k = top_k
        self.enumeration_beam_size = enumeration_beam_size
        self.sentence_bert_ranker = SentenceSimilarityPredictor.instance(kb_type=KB_TYPE.WIKIDATA)
        self.count_flag = False
        if len(self.answer_mid) == 1:
            try:
                answer_mid = list(self.answer_mid)[0]
                answer_mid = answer_mid.replace('^^<http://www.w3.org/2001/XMLSchema#decimal>', '').replace('^^<http://www.w3.org/2001/XMLSchema#integer>', '')
                if (answer_mid.startswith('"')) and (answer_mid.endswith('"')):
                    answer_mid = answer_mid[1:-1]
                self.answer_num = int(answer_mid)
                self.count_flag = True
                self.answer_types = [
                    item for item_list in self.grounded_item_list for item in item_list
                    if item['type'] in ["entity", "class"]
                ] # Wikidata 无法从 id 上区分 entity 和 class, 故都作为候选答案类型
                self.answer_types = set([
                    item['mid'] for item in self.answer_types
                ]) # 去重
                self.answer_types = [{
                    "mid": mid,
                    'type': 'class'
                } for mid in self.answer_types]
            except:
                self.count_flag = False
        if multivariate_pattern_bi_encoder_path is not None:
            self.bi_encoder_inferencor = BiEncoderInferencor.instance(
                model_path=multivariate_pattern_bi_encoder_path,
                pattern_serialized_list_path=multivariate_pattern_list_serialized_path
            )
            self.pattern_list = load_json(multivariate_pattern_list_path)
            self.top_k_pattern = top_k_pattern
        else:
            self.bi_encoder_inferencor = None
        self.use_question_info = use_question_info
        if not self.use_question_info: # 不做分解和排序
            self.sub_question_list = [
                [question]
            ]
            self.grounded_item_list = combine_grounded_items(self.grounded_item_list)
    
    def get_searched_queries(self):
        return self.searched_queries
    
    def is_complete(self):
        return len(self.searched_queries) >= self.top_k
    
    def filter_empty_paths(self, path_nested_list):
        '''Empty path: 对于某个子句，我们并没有找到对应的路径，那么也只能忽略这个子句了'''
        return [
            p_list for p_list in path_nested_list
            if len(p_list) > 0
        ]
    
    def search(self):
        if self.skip_flag:
            return
        atomic_query_list:List[List] = list()
        for _ in range(len(self.sub_question_list)):
            atomic_query_list.append(list())
        if self.count_flag:
            for (sub_question_idx, (sub_question, _grounded_item_list)) in enumerate(zip(self.sub_question_list, self.grounded_item_list)):
                '''子问题可能由多个 clause 组成'''
                serialized_sub_question = DELIMETER.join(sub_question)
                if len(sub_question) == 1:
                    for grounded_item in tqdm(_grounded_item_list, desc=f"sub_question_idx: {sub_question_idx}; Enumerating grounded items"):
                        for ans_type in self.answer_types:
                            atomic_query_list[sub_question_idx].extend(
                                self.sparql_querier.query_one_hop_paths(
                                    grounded_item, answer_type=ans_type
                                )
                            )
                            atomic_query_list[sub_question_idx].extend(
                                self.sparql_querier.query_one_hop_paths_reversed(
                                    grounded_item, answer_type=ans_type
                                )
                            )
                    if self.bi_encoder_inferencor is not None:
                        # 先做Pattern检索
                        _, index_list = self.bi_encoder_inferencor.retrieve(serialized_sub_question, self.top_k_pattern)
                        _pattern_list = [self.pattern_list[idx] for idx in index_list.tolist()[0]]
                        # 将 Pattern 根据答案和候选实体，实例化成原子查询
                        for _pattern in _pattern_list:
                            [p_prop, pq_prop] = _pattern.split(PATTERN_DELIMETER)
                            for (grounded_item_0, grounded_item_1) in tqdm(itertools.combinations(_grounded_item_list, 2), desc="遍历 grounded item 的组合，做多元关系查询"):
                                for ans_type in self.answer_types:
                                    atomic_queries = self.sparql_querier.query_multivariate_atomic_queries(
                                        grounded_item_0, grounded_item_1, p_prop, pq_prop, answer_type=ans_type
                                    )
                                    for query in atomic_queries:
                                        query["retrieval_flag"] = True
                                    atomic_query_list[sub_question_idx].extend(atomic_queries)
                    '''Arg paths 不需要遍历 grounded item'''
                    for ans_type in self.answer_types:
                        atomic_query_list[sub_question_idx].extend(
                            self.sparql_querier.get_one_hop_arg_relation(
                                answer_type=ans_type
                            )
                        )
                    
                    if self.use_question_info:
                        '''SentenceBert 做个排序'''
                        sorted_index = self.sentence_bert_ranker.sort_atomic_list_return_index(
                            serialized_sub_question, 
                            [
                                self.encode_s_expression(q)
                                for q in atomic_query_list[sub_question_idx]
                            ], 
                            self.item_to_label
                        )
                        atomic_query_list[sub_question_idx] = [
                            atomic_query_list[sub_question_idx][idx] for idx in sorted_index
                        ]
                    atomic_query_list[sub_question_idx] = sort_atomic_query_list_with_retrieval_flag(atomic_query_list[sub_question_idx])
                else: # 子问题由多个 clause 组成，即存在嵌套现象
                    for grounded_item in tqdm(_grounded_item_list, desc=f"sub_question_idx: {sub_question_idx}; Enumerating grounded items"):
                        multi_hop_path_enumerator = MultiHopPathEnumerator(
                            sub_question, self.kb_type, self.sparql_querier, self.enumeration_beam_size,
                            self.sentence_bert_ranker, self.anchor_answer, self.item_to_label, self.logger,
                            count_flag=True, answer_types=self.answer_types, use_question_info=self.use_question_info
                        )
                        atomic_query_list[sub_question_idx].extend(
                            [{"s_expression": atomic_query} for atomic_query in multi_hop_path_enumerator.enumerate_multi_hop_path(grounded_item)]
                        )
                    
                    if self.bi_encoder_inferencor is not None:
                        # 多跳的情况下，检索的依据是最后一个子问题
                        _, index_list = self.bi_encoder_inferencor.retrieve(sub_question[-1], self.top_k_pattern)
                        _pattern_list = [self.pattern_list[idx] for idx in index_list.tolist()[0]]
                        # 重新声明一个 MultiHopPathEnumerator; 每个 MultiHopPathEnumerator 有一些内部状态，尽量还是分开
                        for grounded_item_pair in tqdm(itertools.combinations(_grounded_item_list, 2)):
                            multi_hop_path_enumerator = MultiHopPathEnumerator(
                                sub_question, self.kb_type, self.sparql_querier, self.enumeration_beam_size,
                                self.sentence_bert_ranker, self.anchor_answer, self.item_to_label, self.logger,
                                count_flag=True, answer_types=self.answer_types, use_question_info=self.use_question_info
                            )
                            atomic_query_list[sub_question_idx].extend(
                                [{"s_expression": atomic_query} for atomic_query in multi_hop_path_enumerator.enumerate_multi_hop_path_with_multiproperty_pattern(grounded_item_pair, _pattern_list)]
                            )

                    if self.use_question_info:
                        '''SentenceBert 做个排序'''
                        sorted_index = self.sentence_bert_ranker.sort_atomic_list_return_index(
                            serialized_sub_question, 
                            [self.encode_s_expression(q) for q in atomic_query_list[sub_question_idx]], 
                            self.item_to_label
                        )
                        atomic_query_list[sub_question_idx] = [
                            atomic_query_list[sub_question_idx][idx] for idx in sorted_index
                        ]
                    
                    atomic_query_list[sub_question_idx] = sort_atomic_query_list_with_retrieval_flag(atomic_query_list[sub_question_idx])
            
            all_atomic_queries = self.filter_empty_paths(atomic_query_list)
            depth_first_searcher = DepthFirstSearchForCount(
                self.question, self.item_to_label, all_atomic_queries,
                self.answer_num, self.sparql_querier, self.logger, self.sentence_bert_ranker,
                self.kb_type, self.depth_upper_bound, self.top_k, self.use_question_info
            )
            depth_first_searcher.start()
            searched_queries = depth_first_searcher.get_searched_queries()
            # 去重有点困难，如果这一次搜索得到的查询数量更多，我们就更新一下
            if len(searched_queries) > len(self.searched_queries):
                self.searched_queries = searched_queries
        
        '''这里作非 COUNT 类型问题的探测；COUNT 类型问题如果没有探测到正确查询，则将答案视作一个普通 literal 处理'''
        if len(self.searched_queries) == 0:
            for (sub_question_idx, (sub_question, _grounded_item_list)) in enumerate(zip(self.sub_question_list, self.grounded_item_list)):
                '''子问题可能由多个 clause 组成'''
                serialized_sub_question = DELIMETER.join(sub_question)
                if len(sub_question) == 1:
                    for grounded_item in tqdm(_grounded_item_list, desc=f"sub_question_idx: {sub_question_idx}; Enumerating grounded items"):
                        atomic_query_list[sub_question_idx].extend(
                            self.sparql_querier.query_one_hop_paths(
                                grounded_item, answer_entity=self.anchor_answer
                            )
                        )
                        atomic_query_list[sub_question_idx].extend(
                            self.sparql_querier.query_one_hop_paths_reversed(
                                grounded_item, answer_entity=self.anchor_answer
                            )
                        )
                    if self.bi_encoder_inferencor is not None:
                        # 先做Pattern检索
                        _, index_list = self.bi_encoder_inferencor.retrieve(serialized_sub_question, self.top_k_pattern)
                        _pattern_list = [self.pattern_list[idx] for idx in index_list.tolist()[0]]
                        # 将 Pattern 根据答案和候选实体，实例化成原子查询
                        for _pattern in _pattern_list:
                            [p_prop, pq_prop] = _pattern.split(PATTERN_DELIMETER)
                            for (grounded_item_0, grounded_item_1) in tqdm(itertools.combinations(_grounded_item_list, 2), desc="遍历 grounded item 的组合，做多元关系查询"):
                                atomic_queries = self.sparql_querier.query_multivariate_atomic_queries(
                                    grounded_item_0, grounded_item_1, p_prop, pq_prop, answer_entity=self.anchor_answer
                                )
                                for query in atomic_queries:
                                    query["retrieval_flag"] = True
                                atomic_query_list[sub_question_idx].extend(atomic_queries)
                    '''Arg paths 不需要遍历 grounded item'''
                    atomic_query_list[sub_question_idx].extend(
                        self.sparql_querier.get_one_hop_arg_relation(
                            answer_entity=self.anchor_answer
                        )
                    )
                    if self.use_question_info:
                        '''SentenceBert 做个排序'''
                        sorted_index = self.sentence_bert_ranker.sort_atomic_list_return_index(
                            serialized_sub_question, 
                            [
                                self.encode_s_expression(q)
                                for q in atomic_query_list[sub_question_idx]
                            ], 
                            self.item_to_label
                        )
                        atomic_query_list[sub_question_idx] = [
                            atomic_query_list[sub_question_idx][idx] for idx in sorted_index
                        ]
                   
                    atomic_query_list[sub_question_idx] = sort_atomic_query_list_with_retrieval_flag(atomic_query_list[sub_question_idx])
                else: # 子问题由多个 clause 组成，即存在嵌套现象
                    for grounded_item in tqdm(_grounded_item_list, desc=f"sub_question_idx: {sub_question_idx}; Enumerating grounded items"):
                        multi_hop_path_enumerator = MultiHopPathEnumerator(
                            sub_question, self.kb_type, self.sparql_querier, self.enumeration_beam_size,
                            self.sentence_bert_ranker, self.anchor_answer, self.item_to_label, self.logger,
                            count_flag=False, answer_types=None, use_question_info=self.use_question_info
                        ) # 视作非 COUNT 类型处理
                        atomic_query_list[sub_question_idx].extend(
                            [{"s_expression": atomic_query} for atomic_query in multi_hop_path_enumerator.enumerate_multi_hop_path(grounded_item)]
                        )
                    
                    if self.bi_encoder_inferencor is not None:
                        # 多跳的情况下，检索的依据是最后一个子问题
                        _, index_list = self.bi_encoder_inferencor.retrieve(sub_question[-1], self.top_k_pattern)
                        _pattern_list = [self.pattern_list[idx] for idx in index_list.tolist()[0]]
                        # 重新声明一个 MultiHopPathEnumerator; 每个 MultiHopPathEnumerator 有一些内部状态，尽量还是分开
                        for grounded_item_pair in tqdm(itertools.combinations(_grounded_item_list, 2)):
                            multi_hop_path_enumerator = MultiHopPathEnumerator(
                                sub_question, self.kb_type, self.sparql_querier, self.enumeration_beam_size,
                                self.sentence_bert_ranker, self.anchor_answer, self.item_to_label, self.logger,
                                count_flag=False, answer_types=None, use_question_info=self.use_question_info
                            ) # 视作非 COUNT 类型处理
                            atomic_query_list[sub_question_idx].extend(
                                [{"s_expression": atomic_query} for atomic_query in multi_hop_path_enumerator.enumerate_multi_hop_path_with_multiproperty_pattern(grounded_item_pair, _pattern_list)]
                            )
                    if self.use_question_info:  
                        '''SentenceBert 做个排序'''
                        sorted_index = self.sentence_bert_ranker.sort_atomic_list_return_index(
                            serialized_sub_question, 
                            [self.encode_s_expression(q) for q in atomic_query_list[sub_question_idx]], 
                            self.item_to_label
                        )
                        atomic_query_list[sub_question_idx] = [
                            atomic_query_list[sub_question_idx][idx] for idx in sorted_index
                        ]

                    atomic_query_list[sub_question_idx] = sort_atomic_query_list_with_retrieval_flag(atomic_query_list[sub_question_idx])
            
            all_atomic_queries = self.filter_empty_paths(atomic_query_list)
            depth_first_searcher = DepthFirstSearch(
                self.question, self.item_to_label, all_atomic_queries,
                self.answer, self.sparql_querier, self.logger, self.sentence_bert_ranker,
                self.kb_type, self.depth_upper_bound, self.top_k, self.use_question_info
            )
            depth_first_searcher.start()
            searched_queries = depth_first_searcher.get_searched_queries()
            # 去重有点困难，如果这一次搜索得到的查询数量更多，我们就更新一下
            if len(searched_queries) > len(self.searched_queries):
                self.searched_queries = searched_queries
    
    def supply_label(self, item_mid):
        if item_mid not in self.item_to_label:
            if self._get_item_type(item_mid) == "entity":
                label = self.sparql_querier.get_friendly_name(item_mid)
                self.item_to_label[item_mid] = label
            elif self._get_item_type(item_mid) == "literal":
                self.item_to_label[item_mid] = extract_value_from_literal(item_mid)
    
    def _get_item_type(self, item_mid):
        if self.kb_type is KB_TYPE.WIKIDATA:
            return convert_wikidata_type(WikidataConstantForConstruction.get_constant_type(item_mid))
        elif self.kb_type is KB_TYPE.FREEBASE:
            # constant_type = FreebaseConstantSerializer.get_constant_type(item_mid)
            # if constant_type is FREEBASE_CONSTANT_TYPE.ENTITY:
            #     return "entity"
            # elif constant_type in [FREEBASE_CONSTANT_TYPE.STRING, FREEBASE_CONSTANT_TYPE.QUANTITY, FREEBASE_CONSTANT_TYPE.TIME]:
            #     return "literal"
            # else:
            #     return None
            if item_mid.startswith('m.') or item_mid.startswith('g.'):
                return "entity"
            elif re.fullmatch("[a-zA-Z_]+\.[a-zA-Z_]+", item_mid):
                return "class"
            elif 'http://www.w3.org/2001/XMLSchema' in item_mid:
                return "literal"
            elif item_mid.startswith('"') and item_mid.endswith('"'):
                return "literal"
            elif re.fullmatch('".+"@.+', item_mid): # 形如 "Andrew"@en
                return "literal"
            else:
                try: # 尝试转成数值，针对 CWQ
                    value = float(item_mid)
                    return "literal"
                except:
                    pass
            return None
        else:
            raise NotImplementedError(f"kb_type: {self.kb_type}")
    
    def encode_s_expression(self, sexp):
        """
        形式 1: {"s_expression": }
        形式 2: {"function", "relation}
        """
        if "s_expression" in sexp:
            return sexp["s_expression"]
        else:
            return f"{sexp['function']}{DELIMETER}{sexp['relation']}"

def sort_atomic_query_list_with_retrieval_flag(_original_list):
    retrieval_items = [item for item in _original_list if ("retrieval_flag" in item) and item["retrieval_flag"]]
    other_items = [item for item in _original_list if ("retrieval_flag" not in item) or (not item["retrieval_flag"])]
    return retrieval_items + other_items

def combine_grounded_items(grounded_item_list):
    mid_set = set()
    combine_result = list()
    for nested_list in grounded_item_list:
        for item in nested_list:
            if item['mid'] not in mid_set:
                mid_set.add(item['mid'])
                combine_result.append(item)
    return [combine_result] # 和只有一个子问题对应