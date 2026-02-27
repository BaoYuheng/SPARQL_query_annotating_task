import string
import random
import re
import requests
import json
from src.sparql_executor import (
    SparqlOdbcQuerier
)
from src.common import (
    DECOMPOSITION_TAGS, PRESENT_DATE, FreebaseConstantForConstruction
)
from src.utils import load_json, get_n_gram, post_process_timex_obj
from src.facc1_surface_index_memory import EntitySurfaceIndexMemory
from nltk.corpus import stopwords
from nltk import word_tokenize
from collections import defaultdict
from rapidfuzz import process as fuzz_process, fuzz

class FreebaseClassLinker(object):
    @classmethod
    def instance(cls, *args, **kwargs):
        if not hasattr(FreebaseClassLinker, "_instance"):
            FreebaseClassLinker._instance = FreebaseClassLinker(*args, **kwargs)
        return FreebaseClassLinker._instance
    
    def __init__(self):
        self.lemmatized_label_to_class = load_json("/home5/yhbao/data/input/common/lemmatized_label_to_category_class.json")
        from nltk.stem import WordNetLemmatizer
        self.wlem = WordNetLemmatizer()
    
    def lemmatize_input(self, name):
        name = name.lower()
        token_list = word_tokenize(name)
        token_list = [self.wlem.lemmatize(tok) for tok in token_list]
        lemmatized_name = ' '.join(token_list)
        lemmatized_name = lemmatized_name.replace('``', '"').replace("''", '"')

        return lemmatized_name
    
    def accurate_matching(self, text):
        text = self.lemmatize_input(text)
        if text in self.lemmatized_label_to_class:
            return self.lemmatized_label_to_class[text]
        return None
    
    def fuzzy_matching(self, text):
        '''
        所有 class 里面模糊匹配，返回得分最高的
        '''
        text = self.lemmatize_input(text)
        choices = list(self.lemmatized_label_to_class.keys())
        (label, _, _ )  = fuzz_process.extractOne(text, choices, scorer=fuzz.QRatio)
        return self.lemmatized_label_to_class[label]
    
    def class_linking_main(self, text):
        linked_class = self.accurate_matching(text)
        if linked_class is None:
            linked_class = self.fuzzy_matching(text)
        return linked_class

class EntityLinker(object):
    @classmethod
    def instance(cls, *args, **kwargs):
        if not hasattr(EntityLinker, "_instance"):
            EntityLinker._instance = EntityLinker(*args, **kwargs)
        return EntityLinker._instance
    
    def __init__(self, logger, sparql_querier:SparqlOdbcQuerier, top_k_per_mention=5):
        from sutime import SUTime
        self.sutime = SUTime(mark_time_ranges=False, include_range=False)
        self.stopwords = set(stopwords.words('english'))
        # self.facc1_surface_index = EntitySurfaceIndexMemory.instance(
        #     "data/facc1/entity_list_file_freebase_complete_all_mention", 
        #     "data/facc1/surface_map_file_freebase_complete_all_mention",
        #     "data/facc1/freebase_complete_all_mention"
        # )
        self.facc1_surface_index = EntitySurfaceIndexMemory.instance(
            "/home5/yhbao/data/facc1/entity_list_file_freebase_complete_all_mention", 
            "/home5/yhbao/data/facc1/surface_map_file_freebase_complete_all_mention",
            "/home5/yhbao/data/facc1/freebase_complete_all_mention"
        )
        self.facc1_service_url = "http://114.212.190.19:5688/entity_linking"
        self.logger = logger
        self.sparql_querier = sparql_querier
        self.top_k_per_mention = top_k_per_mention
        self.class_linker = FreebaseClassLinker.instance()
    
    def _filter_ngrams(self, n_gram_set):
        n_gram_set = {
            n_gram for n_gram in n_gram_set
            if (n_gram.lower() not in string.punctuation) and (n_gram.lower() not in self.stopwords) and (n_gram.lower() not in DECOMPOSITION_TAGS)
        }
        return n_gram_set  
    
    def mention_detection_and_candidate_generation(self, question):
        tokens = word_tokenize(question) 
        n_gram_set = set()
        for i in range(1, len(tokens)):
            n_gram_set.update(get_n_gram(tokens, i))
        n_gram_set = self._filter_ngrams(n_gram_set)
        entity_results = dict()
        class_results = dict()
        for n_gram in n_gram_set:
            rtn_value = self.facc1_surface_index.get_indexrange_entity_el_pro_one_mention(
                n_gram, self.top_k_per_mention
            ) # collections.OrderedDict(), 按照流行度排好序了
            # res = requests.post(
            #     url=self.facc1_service_url,
            #     data=json.dumps({'mention':n_gram, 'top_k': self.top_k_per_mention})
            # )
            # rtn_value = json.loads(res.text)['result']
            entity_results[n_gram] = [
                (key, value) for (key, value) in rtn_value.items()
            ] 
            # 这边不做排序，candidate_entity_filtering 会做排序

            linked_class = self.class_linker.accurate_matching(n_gram)
            if linked_class is not None:
                class_results[n_gram] = linked_class
        return entity_results, class_results
    
    def candidate_entity_filtering(self, candidate_entity_by_mention, top_k):
        '''
        总共保留 max(mention_num, top_k) 数量个候选
        每个 List[Tuple(str, float)] 存放了这个 mention 对应的所有候选，按照流行度排序(我们自己再做一遍排序吧)
        策略: 拉链法，依次遍历每个 mention, 选择当前候选实体中流行度最高的那个
        '''
        '''首先过滤候选实体为空的 mention'''
        candidate_entity_by_mention = {
            key: sorted(value, key=lambda item:item[1], reverse=True) for (key, value) in candidate_entity_by_mention.items()
            if len(value) > 0
        }
        total_candidate_entities = sum([len(cand_entity_list) for cand_entity_list in candidate_entity_by_mention.values()])
        mention_num = len(candidate_entity_by_mention)

        target_entity_num = min(max(top_k, mention_num), total_candidate_entities)

        target_entity_dict = defaultdict(list) # 仍然保留 mention 到 候选实体的映射关系
        target_entity_count = 0
        list_idx = 0

        while target_entity_count < target_entity_num:
            for (mention, entity_tup_list) in candidate_entity_by_mention.items():
                if list_idx < len(entity_tup_list):
                    target_entity_dict[mention].append(entity_tup_list[list_idx])
                    target_entity_count += 1
                    if target_entity_count >= target_entity_num:
                        break
            list_idx += 1
        
        return target_entity_dict
    
    def extract_answer_classes(self, answer_list):
        '''所有答案实体的共同 class'''
        types = set()
        for (idx, answer) in enumerate(answer_list):
            answer_mid = answer["mid"]
            answer_types = self.sparql_querier.get_category_classes(answer_mid)
            if idx == 0:
                types.update(answer_types)
            else:
                types = types & set(answer_types)
        return list(types)
    
    def extract_string_literals(self, question):
        #增加了mention的记录
        '''
        Literal 字面量的格式比较固定，可以直接按照规则从问题中抽取
        自然语言问题中的 Literal, 不太可能有 @en tag
        '''
        return ((w, w) for w in re.findall(r"\".+?\"", question))
    
    def extract_number_literals(self, question):
        #增加了mention的记录
        """
        很难想象一个数字里面会有空格
        哪怕是位数很多，一般也通过 ',' 或者 '.' 来连接
        因此，我觉得把 question 按照空格分开，然后所有 1-gram 尝试是否可以转成数字即可

        格式转换，统一加上 #float 的后缀
        """
        words = word_tokenize(question)
        number_set = set()
        for w in words:
            value = None
            w_origin = w
            w = w.replace(',', '')
            try:
                value = int(w)
                number_set.add((w_origin, f"\"{value}\"^^<http://www.w3.org/2001/XMLSchema#integer>"))
            except:
                pass
            if value is None:
                try:
                    value = float(w)
                    number_set.add((w_origin,f"\"{value}\"^^<http://www.w3.org/2001/XMLSchema#float>"))
                except:
                    pass
        return number_set
    
    def extract_time_literals(self, question):
        try:
            rtn_value = self.sutime.parse(question, PRESENT_DATE)
            rtn_list = list()
            for value_item in rtn_value:
                if (value_item["type"] in ["DATE", "TIME"]) and ("timex-value" in value_item):
                    processed_v_list = post_process_timex_obj(value_item['timex-value'])
                    if len(processed_v_list) == 0:
                        self.logger.info(f"conversion failed: {value_item}")
                    else:
                        for processed_v in processed_v_list:
                            rtn_list.append((value_item['text'], f"\"{processed_v}\"^^<http://www.w3.org/2001/XMLSchema#dateTime>"))
            return rtn_list
        except Exception as e:
            self.logger.error(f"extract_time_literals; question: {question}; error: {e}")
            return list()

    def entity_linking_clause(self, clause, top_k_entities=10, top_k_classes=5):
        '''
        @params top_k_entities: 对于这个 clause, 返回 top_k 个实体
        @params top_k_classes: 对于这个 clause, 返回 top_k 个 class
        @return linked_items: List[List]
        '''
        random.seed(374)
        entities_by_mention, classes_by_mention = self.mention_detection_and_candidate_generation(
            clause
        )
        filtered_entities_by_mention = self.candidate_entity_filtering(
            entities_by_mention, top_k_entities
        ) # Dict[Str, List[Tuple]]
        filtered_classes_by_mention = self.candidate_entity_filtering(
            classes_by_mention, top_k_classes
        ) # Dict[Str, List[Tuple]]
        detected_entities = {
            tup[0]
            for tup_list in filtered_entities_by_mention.values()
            for tup in tup_list
        }
        detected_classes = {
            tup[0]
            for tup_list in filtered_classes_by_mention.values()
            for tup in tup_list
        }
        detected_literals = set()
        literal_mention_dict = {}
        for item in self.extract_string_literals(clause):
            literal_mention_dict[item[1]] = item[0]
        for item in self.extract_number_literals(clause):
            literal_mention_dict[item[1]] = item[0]
        for item in self.extract_time_literals(clause):
            literal_mention_dict[item[1]] = item[0]
        if len(literal_mention_dict) > 0:
            pass
        detected_literals.update([k for k in literal_mention_dict.keys()])
        # detected_literals.update(self.extract_string_literals(clause))
        # detected_literals.update(self.extract_number_literals(clause))
        # detected_literals.update(self.extract_time_literals(clause))
        items_to_mentions = {}
        for k, v in filtered_entities_by_mention.items():
            for v2 in v:
                items_to_mentions[v2[0]] = k
        for k, v in filtered_classes_by_mention.items():
            for v2 in v:
                items_to_mentions[v2[0]] = k
        #在这里，需要保留
        rtn = [
            {
                "type": "entity", 
                "mid": FreebaseConstantForConstruction(
                    "uri", f"http://rdf.freebase.com/ns/{ent}"
                ).__repr__(),
                "mention": items_to_mentions[ent]
            }
            for ent in detected_entities
        ] + [
            {
                "type": "class", 
                "mid": FreebaseConstantForConstruction(
                    "uri", f"http://rdf.freebase.com/ns/{class_item}"
                ).__repr__(),
                "mention": items_to_mentions[class_item]
            }
            for class_item in detected_classes
        ] + [
            {
                "type": "literal", 
                "mid": lit, # 观察了三个来源函数，格式上应该是处理好了
                "mention": literal_mention_dict[lit]
            }
            for lit in detected_literals
        ]

        return rtn
