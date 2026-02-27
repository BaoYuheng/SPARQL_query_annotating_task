from src.logic_form_util import lisp_to_sparql
from src.utils import load_json, dump_json, setup_custom_logger
from src.semantic_sim_utils import FBEmbeddingBasedSimilarity, calculate_question_rel_topk
from models.gpt_rerank import get_top1_sparql_by_gpt
from models.search import LFSearchAlgorithm
from src.common import DATASET, KB_TYPE
import random 
from tqdm import tqdm
import datetime
import os
import math
import itertools
from copy import deepcopy
from src.sparql_executor import SparqlOdbcQuerierWikidata, SparqlOdbcQuerierNoSexpr
from src.common import ODBC_CONFIG_WIKIDATA_2023, SPARQL_wrapper_path_wikidata_2023, ODBC_CONFIG_DKILAB, ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_dkilab, SPARQL_wrapper_path_official
random.seed(1)
import re


def calculate_relation_recall():
    data = load_json("/home5/yhbao/data/cwq/cwq_test_linking.json")
    cand_rels = load_json("/home5/yhbao/SPARQLannotation/data/linking_data/CWQ_cand_rel_cache/CWQ_cand_rels_test.json")
    total_recall = 0
    for item in data:
        tokens =  item['golden_sparql_query'].replace(")", " ").replace("(", " ").split()
        true_relations = [r.replace("ns:", "") for r in tokens if r.count(".") >= 2 and "http" not in r]
        true_relations = set(true_relations)
        cand_relations = cand_rels[item['question']]
        hit = 0
        for r in true_relations:
            if r in cand_relations[0:100]:
                hit += 1
        recall = float(hit)/float(len(true_relations))
        total_recall += recall
    print(total_recall/len(data))


def get_FB_range_type():
    d = {}
    data = load_json("/home5/yhbao/data/fb_relations_domain_range_label.json")
    for item in data.values():
        if 'range' in item and item['range'].startswith("type."):
            if item['range'] not in d:
                d[item['range']] = 1
    print(d.keys())

def get_grailQA_wo_nc():
    data = load_json("/home5/yhbao/data/grailqa/grailqa_v1.0_dev_linking.json")
    new_data = []
    for item in data:
        if "(gt " not in item['golden_s_expression'] and "(lt " not in item['golden_s_expression'] and "ARGM" not in item['golden_s_expression'] \
            and "(le " not in item['golden_s_expression']  and "(ge " not in item['golden_s_expression'] and "(COUNT" not in item['golden_s_expression']:
            new_data.append(item)
    dump_json(new_data, "/home5/yhbao/SPARQLannotation/data/grailqa_dev_wo_nc.json")

def get_grailQA_nc():
    data = load_json("/home5/yhbao/data/grailqa/grailqa_v1.0_dev_linking.json")
    new_data = []
    for item in data:
        if "(gt " in item['golden_s_expression'] or "(lt " in item['golden_s_expression'] or "ARGM" in item['golden_s_expression'] \
            or "(le " in item['golden_s_expression']  or "(ge " in item['golden_s_expression'] or "(COUNT" in item['golden_s_expression']:
            new_data.append(item)
    dump_json(new_data, "/home5/yhbao/SPARQLannotation/data/grailqa_dev_nc.json")


def fix_fail_splits(search_result_files, begin, end):
    #原理是，检查失败的区域，然后对失败的区域进行split_search，然后拼接
    return 0 


def get_split(search_result_file, begin, end):
    return load_json(search_result_file)[begin:end]


def research_file(search_algorithm:LFSearchAlgorithm, input_data, current_output_file_dir, new_result_output_dir, start, end):
    #对于某些搜索有问题的问题，重新开始查询
    current_result = load_json(current_output_file_dir)
    qids_need_to_research = []
    data_need_to_research = []
    for result in current_result[start:end]:
        if len(result['search_results']) == 0:
            qids_need_to_research.append(result['qid'])
    for item in input_data:
        if item['qid'] in qids_need_to_research:
            data_need_to_research.append(item)
    researched_results, _ = search_algorithm.begin_search(data_need_to_research)
    researched_results_dict = {item['qid']:item for item in researched_results}
    new_results = []
    for item in current_result:
        if item['qid'] in researched_results_dict:
            new_results.append(researched_results_dict[item['qid']])
        else:
            new_results.append(item)
    dump_json(new_results, new_result_output_dir)

def split_search(search_algorithm:LFSearchAlgorithm, input_data, output_dir, split_size = 1000, return_Sexpr = True, only_keep_accurate = True):
    #每split_size存储一次，避免中途卡死
    #每次开始之前，查看是否有temp_file；如果有，读取并从此开始；如果没有，创建新的temp_file，并从0开始
    finised_data = []
    temp_data_file_dir = output_dir.replace(".json", "_temp.json")
    if os.path.isfile(temp_data_file_dir):
        finised_data = load_json(temp_data_file_dir)
    else:
        dump_json(finised_data, temp_data_file_dir)
    remain_data = input_data[len(finised_data):]
    print("总共还要搜索：", len(remain_data))
    while(len(remain_data) > 0):
        this_split = remain_data[0:min(split_size, len(remain_data))]
        results, _ = search_algorithm.begin_search(this_split, return_Sexpr, only_keep_accurate)
        finised_data += results
        dump_json(finised_data, temp_data_file_dir)
        finised_data = load_json(temp_data_file_dir)
        remain_data = input_data[len(finised_data):]
    #此时最终完成了
    dump_json(finised_data, output_dir)
    os.remove(temp_data_file_dir)
    

def GrailQA_search_experiment(split, use_golden_entity, experiment_tag):
    assert(split == "train" or split == "dev")
    dataset = DATASET.GRAIL
    use_neighbor_entity = False
    entity_status = "golden" if use_golden_entity else "linked"
    neighbor_entity_status = "use_neighbor_entity" if use_neighbor_entity else "not_use_neighbor_entity"
    logger_dir = f"/home5/yhbao/SPARQLannotation/output/log/grailqa_{split}_sparql_search_{entity_status}_{neighbor_entity_status}_{experiment_tag}.log"
    output_dir = f"/home5/yhbao/SPARQLannotation/output/grailqa/grailqa_{split}_{entity_status}_{neighbor_entity_status}_{experiment_tag}.json"
    param_dir =  f"/home5/yhbao/SPARQLannotation/output/grailqa/grailqa_{split}_{entity_status}_{neighbor_entity_status}_{experiment_tag}_params.json"
    input_data = load_json(f"/home5/yhbao/data/grailqa/grailqa_v1.0_{split}_linking_new.json")
    search_algorithm = LFSearchAlgorithm(dataset, log_file_dir=logger_dir, use_golden_entity=use_golden_entity, use_neighbor_entity = use_neighbor_entity,
                                         sparql_cache_dir="/home5/yhbao/SPARQLannotation/cache/sparql_result_cache/grailqa_freebase_cache.json")
    dump_json(search_algorithm.get_parameters_dict(), param_dir)
    split_search(search_algorithm, input_data, output_dir, split_size=100)


def GrailQA_search_experiment_dev_1000(use_golden_entity, experiment_tag, use_cache=False, size=1000):
    dataset = DATASET.GRAIL
    use_neighbor_entity = False
    entity_status = "golden" if use_golden_entity else "linked"
    neighbor_entity_status = "use_neighbor_entity" if use_neighbor_entity else "not_use_neighbor_entity"
    logger_dir = f"/home5/yhbao/SPARQLannotation/output/log/grailqa_dev_1000_sparql_search_{entity_status}_{neighbor_entity_status}_{experiment_tag}.log"
    output_dir = f"/home5/yhbao/SPARQLannotation/output/grailqa_final/grailqa_dev_0_1000_{entity_status}_{neighbor_entity_status}_{experiment_tag}.json"
    param_dir =  f"/home5/yhbao/SPARQLannotation/output/grailqa_final/grailqa_dev_0_1000_{entity_status}_{neighbor_entity_status}_{experiment_tag}_params.json"
    input_data = load_json(f"/home5/yhbao/data/grailqa/grailqa_dev_1000_facc1_linking.json")[0:size]
    if use_cache:
        sparql_cache_dir="/home5/yhbao/SPARQLannotation/cache/sparql_result_cache/grailqa_freebase_cache.json"
    else:
        sparql_cache_dir=None
    search_algorithm = LFSearchAlgorithm(dataset, log_file_dir=logger_dir, use_golden_entity=use_golden_entity, use_neighbor_entity = use_neighbor_entity,
                                         sparql_cache_dir=sparql_cache_dir)
    search_algorithm.sparql_executor.direct_manage_2hop = True
    dump_json(search_algorithm.get_parameters_dict(), param_dir)
    split_search(search_algorithm, input_data, output_dir, split_size=100, only_keep_accurate=False)


def CWQ_research(split, link_setting, experiment_tag):
    id_list = load_json("/home5/yhbao/data/cwq/wrong.json")
    assert split == "dev" or split == "train"
    assert link_setting == "golden" or link_setting == "linked"
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    dataset = DATASET.CWQ
    use_golden_entity = True if link_setting == "golden" else False
    use_neighbor_entity = False
    entity_status = link_setting
    neighbor_entity_status = "use_neighbor_entity" if use_neighbor_entity else "not_use_neighbor_entity"
    logger_dir = f"/home5/yhbao/SPARQLannotation/output/log/cwq_sparql_search_{entity_status}_{neighbor_entity_status}_{experiment_tag}_{formatted_time}_research_v2.log"
    output_dir = f"/home5/yhbao/SPARQLannotation/output/cwq/cwq_{split}_{entity_status}_{neighbor_entity_status}_{experiment_tag}_research_v2.json"
    param_dir =  f"/home5/yhbao/SPARQLannotation/output/cwq/cwq_{split}_{entity_status}_{neighbor_entity_status}_{experiment_tag}_research_v2_params.json"
    input_data = load_json(f"/home5/yhbao/data/cwq/cwq_{split}_linking.json")
    input_data = [item for item in input_data if item['qid'] in id_list]
    search_algorithm = LFSearchAlgorithm(dataset, log_file_dir=logger_dir, use_golden_entity=use_golden_entity, use_neighbor_entity = use_neighbor_entity,
                                         sparql_cache_dir="/home5/yhbao/SPARQLannotation/cache/sparql_result_cache/cwq_freebase_cache.json")
    #search_algorithm.begin_search(input_data)
    dump_json(search_algorithm.get_parameters_dict(), param_dir)
    split_search(search_algorithm, input_data, output_dir, split_size=100)


def CWQ_search_experiment_debug(split, link_setting, experiment_tag):
    assert split == "dev" or split == "train"
    assert link_setting == "golden" or link_setting == "linked"
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    dataset = DATASET.CWQ
    use_golden_entity = True if link_setting == "golden" else False
    use_neighbor_entity = False
    entity_status = link_setting
    data_need_to_research = []
    neighbor_entity_status = "use_neighbor_entity" if use_neighbor_entity else "not_use_neighbor_entity"
    logger_dir = f"/home5/yhbao/SPARQLannotation/debug/cwq_{split}_sparql_search_{entity_status}_{neighbor_entity_status}_{experiment_tag}.log"
    output_dir = f"/home5/yhbao/SPARQLannotation/debug/cwq_{split}_{entity_status}_{neighbor_entity_status}_{experiment_tag}.json"
    original_data = {item['qid']:item for item in load_json(f"/home5/yhbao/data/cwq/cwq_{split}_linking.json")}
    original_result_dir = f"/home5/yhbao/SPARQLannotation/output/cwq/cwq_{split}_{link_setting}_not_use_neighbor_entity_nov18_sexpr_gpt_rerank_updated_v2.json"
    for k, v in load_json(original_result_dir).items():
        if len(v['search_results']) == 0:
            data_need_to_research.append(original_data[k])
    #input_data = [item for item in input_data if item['question'] == "Which of the politicians involved in the Israeli-Palestinian conflict ended his/her government positon prior to April 14,2006?"]
    search_algorithm = LFSearchAlgorithm(dataset, log_file_dir=logger_dir, use_golden_entity=use_golden_entity, use_neighbor_entity = use_neighbor_entity,
                                         sparql_cache_dir="/home5/yhbao/SPARQLannotation/cache/sparql_result_cache/cwq_freebase_cache.json")
    search_algorithm.sparql_executor.direct_manage_2hop = True
    # search_algorithm.SEARCH_FAIL_TIMEOUT = 100000
    # search_algorithm.   SUBGRAPH_SEARCH_TIMEOUT = 10000
    # search_algorithm.ENUMERATION_PHASE_TIMEOUT = 1000000
    # search_algorithm.sparql_executor.timeout = 10
    #search_algorithm.begin_search(input_data)
    split_search(search_algorithm, data_need_to_research, output_dir, split_size=200)


def CWQ_search_experiment_test_0_1000(link_setting, experiment_tag, source_file, use_cache=False, size=1000):
    assert link_setting == "golden" or link_setting == "linked"
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    dataset = DATASET.CWQ
    use_golden_entity = True if link_setting == "golden" else False
    use_neighbor_entity = False
    entity_status = link_setting
    neighbor_entity_status = "use_neighbor_entity" if use_neighbor_entity else "not_use_neighbor_entity"
    logger_dir = f"/home5/yhbao/SPARQLannotation/output/log/cwq_sparql_search_{entity_status}_{neighbor_entity_status}_{experiment_tag}_{formatted_time}.log"
    output_dir = f"/home5/yhbao/SPARQLannotation/output/cwq/cwq_test_{entity_status}_{neighbor_entity_status}_{experiment_tag}.json"
    param_dir =  f"/home5/yhbao/SPARQLannotation/output/cwq/cwq_test_{entity_status}_{neighbor_entity_status}_{experiment_tag}_params.json"
    input_data = load_json(source_file)[0:size]
    if use_cache:
        sparql_cache_dir="/home5/yhbao/SPARQLannotation/cache/sparql_result_cache/cwq_freebase_cache.json"
    else:
        sparql_cache_dir = None
    search_algorithm = LFSearchAlgorithm(dataset, log_file_dir=logger_dir, use_golden_entity=use_golden_entity, use_neighbor_entity = use_neighbor_entity,
                                         sparql_cache_dir=sparql_cache_dir)
    search_algorithm.sparql_executor.direct_manage_2hop = True
    # search_algorithm.SEARCH_FAIL_TIMEOUT = 100000
    # search_algorithm.   SUBGRAPH_SEARCH_TIMEOUT = 10000
    # search_algorithm.ENUMERATION_PHASE_TIMEOUT = 1000000
    # search_algorithm.sparql_executor.timeout = 10
    #search_algorithm.begin_search(input_data)
    dump_json(search_algorithm.get_parameters_dict(), param_dir)
    split_search(search_algorithm, input_data, output_dir, split_size=200, only_keep_accurate=False)


def CWQ_search_experiment(split, link_setting, experiment_tag):
    assert split == "dev" or split == "train"
    assert link_setting == "golden" or link_setting == "linked"
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    dataset = DATASET.CWQ
    use_golden_entity = True if link_setting == "golden" else False
    use_neighbor_entity = False
    entity_status = link_setting
    neighbor_entity_status = "use_neighbor_entity" if use_neighbor_entity else "not_use_neighbor_entity"
    logger_dir = f"/home5/yhbao/SPARQLannotation/output/log/cwq_sparql_search_{entity_status}_{neighbor_entity_status}_{experiment_tag}_{formatted_time}.log"
    output_dir = f"/home5/yhbao/SPARQLannotation/output/cwq/cwq_{split}_{entity_status}_{neighbor_entity_status}_{experiment_tag}.json"
    param_dir =  f"/home5/yhbao/SPARQLannotation/output/cwq/cwq_{split}_{entity_status}_{neighbor_entity_status}_{experiment_tag}_params.json"
    input_data = load_json(f"/home5/yhbao/data/cwq/cwq_{split}_linking.json")
    search_algorithm = LFSearchAlgorithm(dataset, log_file_dir=logger_dir, use_golden_entity=use_golden_entity, use_neighbor_entity = use_neighbor_entity,
                                         sparql_cache_dir="/home5/yhbao/SPARQLannotation/cache/sparql_result_cache/cwq_freebase_cache.json")
    #search_algorithm.begin_search(input_data)
    dump_json(search_algorithm.get_parameters_dict(), param_dir)
    split_search(search_algorithm, input_data, output_dir, split_size=100)

def CWQ_search_experiment_test(split, link_setting):
    assert split == "dev" or split == "train" or split == "test"
    assert link_setting == "golden" or link_setting == "linked"
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    dataset = DATASET.CWQ
    use_golden_entity = True if link_setting == "golden" else False
    use_neighbor_entity = False
    entity_status = link_setting
    neighbor_entity_status = "use_neighbor_entity" if use_neighbor_entity else "not_use_neighbor_entity"
    #logger_dir = f"/home5/yhbao/SPARQLannotation/output/log/cwq_sparql_search_{entity_status}_{neighbor_entity_status}_{experiment_tag}_{formatted_time}.log"
    logger_dir = f"/home5/yhbao/SPARQLannotation/output/log/example.log"
    input_data = load_json(f"/home5/yhbao/data/cwq/examples.json")
    search_algorithm = LFSearchAlgorithm(dataset, log_file_dir=logger_dir, use_golden_entity=use_golden_entity, use_neighbor_entity = use_neighbor_entity,
                                         sparql_cache_dir="/home5/yhbao/SPARQLannotation/cache/sparql_result_cache/cwq_freebase_cache.json")
    search_algorithm.begin_search(input_data)
    #dump_json(search_algorithm.get_parameters_dict(), param_dir)
    #split_search(search_algorithm, input_data, output_dir, split_size=100)


def WebQSP_search_experiment(split, use_golden_entity, experiment_tag):
    assert split == "test" or split == "train"
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    dataset = DATASET.WEBQ
    use_neighbor_entity = False
    entity_status = "golden" if use_golden_entity else "linked"
    neighbor_entity_status = "use_neighbor_entity" if use_neighbor_entity else "not_use_neighbor_entity"
    logger_dir = f"/home5/yhbao/SPARQLannotation/output/log/webqsp_{split}_sparql_search_{entity_status}_{neighbor_entity_status}_{experiment_tag}_{formatted_time}.log"
    output_dir = f"/home5/yhbao/SPARQLannotation/output/webqsp/webqsp_{split}_{entity_status}_{neighbor_entity_status}_{experiment_tag}.json"
    #output_dir = f"/home5/yhbao/SPARQLannotation/output/output_temp_debug"
    input_data = load_json(f"/home5/yhbao/data/webqsp/webqsp_{split}_linking.json")
    search_algorithm = LFSearchAlgorithm(dataset, log_file_dir=logger_dir, use_golden_entity=use_golden_entity, use_neighbor_entity = use_neighbor_entity,
                                         sparql_cache_dir="/home5/yhbao/SPARQLannotation/cache/sparql_result_cache/webqsp_freebase_cache.json")
    param_dir =  f"/home5/yhbao/SPARQLannotation/output/webqsp/webqsp_{split}_{entity_status}_{neighbor_entity_status}_{experiment_tag}_params.json"
    dump_json(search_algorithm.get_parameters_dict(), param_dir)
    split_search(search_algorithm, input_data, output_dir, split_size=100)

def WebQSP_search_experiment_test1000(use_golden_entity, experiment_tag, use_cache=False, size=1000):
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    dataset = DATASET.WEBQ
    use_neighbor_entity = False
    entity_status = "golden" if use_golden_entity else "linked"
    neighbor_entity_status = "use_neighbor_entity" if use_neighbor_entity else "not_use_neighbor_entity"
    logger_dir = f"/home5/yhbao/SPARQLannotation/output/log/webqsp_test1000_sparql_search_{entity_status}_{neighbor_entity_status}_{experiment_tag}_{formatted_time}.log"
    output_dir = f"/home5/yhbao/SPARQLannotation/output/webqsp_final/webqsp_test_1000_{entity_status}_{neighbor_entity_status}_{experiment_tag}.json"
    #output_dir = f"/home5/yhbao/SPARQLannotation/output/output_temp_debug"
    input_data = load_json(f"/home5/yhbao/data/webqsp/webqsp_0_1000_linking.json")[0:size]
    if use_cache:
        sparql_cache_dir="/home5/yhbao/SPARQLannotation/cache/sparql_result_cache/webqsp_freebase_cache.json"
    else:
        sparql_cache_dir=None
    search_algorithm = LFSearchAlgorithm(dataset, log_file_dir=logger_dir, use_golden_entity=use_golden_entity, use_neighbor_entity = use_neighbor_entity,
                                         sparql_cache_dir=sparql_cache_dir)
    param_dir =  f"/home5/yhbao/SPARQLannotation/output/webqsp/webqsp_test1000_{entity_status}_{neighbor_entity_status}_{experiment_tag}_params.json"
    search_algorithm.sparql_executor.direct_manage_2hop = True
    dump_json(search_algorithm.get_parameters_dict(), param_dir)
    split_search(search_algorithm, input_data, output_dir, split_size=200, only_keep_accurate=False)


def CWQ_research_experiment_1116(): #对于一些cache造成的问题，对14000-20000中的结果为空的问题重新搜索
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    dataset = DATASET.CWQ
    use_golden_entity = False
    use_neighbor_entity = False
    entity_status = "golden" if use_golden_entity else "linked"
    neighbor_entity_status = "use_neighbor_entity" if use_neighbor_entity else "not_use_neighbor_entity"
    experiment_tag = "nov10_sexpr"
    logger_dir = f"/home5/yhbao/SPARQLannotation/output/log/cwq_sparql_search_{entity_status}_{neighbor_entity_status}_{experiment_tag}_{formatted_time}.log"
    #logger_dir = f"/home5/yhbao/SPARQLannotation/output/log/temp.log"
    old_result_dir = f"/home5/yhbao/SPARQLannotation/output/cwq/cwq_train_{entity_status}_{neighbor_entity_status}_{experiment_tag}.json"
    new_result_dir = f"/home5/yhbao/SPARQLannotation/output/cwq/cwq_train_{entity_status}_{neighbor_entity_status}_{experiment_tag}_researched.json"
    #output_dir = f"/home5/yhbao/SPARQLannotation/output/output_temp_debug"
    input_data = load_json("/home5/yhbao/data/cwq/cwq_train_linking.json")
    search_algorithm = LFSearchAlgorithm(dataset, log_file_dir=logger_dir, use_golden_entity=use_golden_entity, use_neighbor_entity = use_neighbor_entity,
                                         sparql_cache_dir="/home5/yhbao/SPARQLannotation/cache/sparql_result_cache/cwq_freebase_cache.json")
    #search_algorithm.begin_search(input_data)
    research_file(search_algorithm, input_data, old_result_dir, new_result_dir, 14400,19800)


def combine_wikidata_annotation_results_with_possible_answers(data_dir, output_dir):
    #由于答案采用链接，因而需要对每个问题，首先选取其不同答案链接情况的搜索结果，然后取f1最高的
    data = load_json(data_dir)
    original_qids = []
    original_qid2question = {}
    original_qid2results = {}
    combined_data = []
    for item in data:
        original_qid = item['qid'].split("_")[0]
        if original_qid not in original_qids:
            original_qids.append(original_qid)
        if original_qid not in original_qid2question:
            original_qid2question[original_qid] = item['question']
        if original_qid not in original_qid2results:
            original_qid2results[original_qid] = []
        for result in item['search_results']:
            original_qid2results[original_qid].append((result, item['F1']))
    for original_qid in original_qids:
        results = original_qid2results[original_qid]
        if len(results) == 0:
            highest_f1 = 0
            results_with_highest_f1 = []
        else:
            highest_f1 = max([item[1] for item in results])
            results_with_highest_f1 = [item[0] for item in results if item[1] == highest_f1]
        combined_data.append({"qid":original_qid, 'question':original_qid2question[original_qid], "search_results": results_with_highest_f1, "F1":highest_f1})
    dump_json(combined_data, output_dir)
        
        

def Wikidata_annotation_experiment(use_golden_entity, experiment_tag):
    def manage_linked_answers(input_data):
        #每个答案有若干个链接结果；需要每个链接结果进行组合，拼接出若干个答案集合
        new_input_data = []
        for item in input_data:
            item['qid'] = str(item['qid'])
            print(item['question'])
            if len(item['answer_mention']) > 0: #这种情况下，答案都是entity
                answer_mention_to_item_dict = {}
                #首先，倒置answer_linked_item_to_label
                for k, v in item['answer_linked_item_to_label'].items():
                    if v not in answer_mention_to_item_dict:
                        answer_mention_to_item_dict[v] = [{'mid':k, 'type':"entity"}]
                    else:
                        answer_mention_to_item_dict[v].append({'mid':k, 'type':"entity"})
                for idx, possible_answer_set in enumerate(list(itertools.product(*answer_mention_to_item_dict.values()))):
                    print("一个可能的答案集合：", possible_answer_set)
                    new_item = deepcopy(item)
                    new_item['qid'] = new_item['qid'] + "_" + str(idx)
                    new_item['answer'] = possible_answer_set
                    new_input_data.append(new_item)
            else:           #这种情况下，答案是数值
                item['qid'] = item['qid'] + "_" + str(0)
                new_input_data.append(item)
        return new_input_data

    dataset = DATASET.SIMULATED_WIKIDATA
    use_neighbor_entity = False 
    entity_status = "golden" if use_golden_entity else "linked"
    neighbor_entity_status = "use_neighbor_entity" if use_neighbor_entity else "not_use_neighbor_entity"
    logger_dir = f"/home5/yhbao/SPARQLannotation/output/log/wd_annotation/sparql_search_wd_annotation_{entity_status}_{neighbor_entity_status}_{experiment_tag}.log"
    #logger_dir = f"/home5/yhbao/SPARQLannotation/output/log/temp.log"
    output_dir = f"/home5/yhbao/SPARQLannotation/output/wd_annotation/WD_annotation_{entity_status}_{neighbor_entity_status}_{experiment_tag}.json"
    params_dir = f"/home5/yhbao/SPARQLannotation/output/wd_annotation/WD_annotation_{entity_status}_{neighbor_entity_status}_{experiment_tag}_params.json"
    input_data = load_json("/home5/yhbao/data/annotated_test/annotation_dataset.json")
    processed_input_data = []
    for item in input_data:
        if len(item['answer']) == 0:
            continue
        processed_input_data.append(item)
    processed_input_data = manage_linked_answers(processed_input_data)
    #processed_input_data = [item for item in processed_input_data if item['question'] == "who won the super heavyweight gold medal at the 2000 olympics"]
    #input_data = [item for item in input_data if item['question'] == "what are all the engine types that have an engine horsepower under 429.0?"]
    #dump_json(data, "/home5/yhbao/SPARQLannotation/output/grailqa/GrailQA_train_golden_not_use_neighbor_entity_train10000_temp.json")
    search_algorithm = LFSearchAlgorithm(dataset, log_file_dir=logger_dir, use_golden_entity=use_golden_entity, use_neighbor_entity = use_neighbor_entity, 
                                         sparql_cache_dir="/home5/yhbao/SPARQLannotation/cache/sparql_result_cache/annotation_wikidata_cache.json")
    dump_json(search_algorithm.get_parameters_dict(), params_dir)
    #search_algorithm.begin_search(processed_input_data, return_sexpr= False, only_keep_accurate=False)
    split_search(search_algorithm, processed_input_data, output_dir=output_dir, split_size=100, return_Sexpr = False, only_keep_accurate = False)


def Wikidata_search_test():
    dataset = DATASET.QALD
    use_golden_entity = True
    use_neighbor_entity = False
    entity_status = "golden" if use_golden_entity else "linked"
    neighbor_entity_status = "use_neighbor_entity" if use_neighbor_entity else "not_use_neighbor_entity"
    experiment_tag = "test"
    logger_dir = f"/home5/yhbao/SPARQLannotation/output/log/sparql_search_{entity_status}_{neighbor_entity_status}_{experiment_tag}.log"
    logger_dir = f"/home5/yhbao/SPARQLannotation/output/log/temp.log"
    output_dir = f"/home5/yhbao/SPARQLannotation/output/qald/QALD_{entity_status}_{neighbor_entity_status}_{experiment_tag}.json"
    output_dir = f"/home5/yhbao/SPARQLannotation/output/qald/qald_{entity_status}_{neighbor_entity_status}_{experiment_tag}.json"
    input_data = load_json("/home5/yhbao/data/qald/qald_golden.json")[0:100]
    processed_input_data = []
    for item in input_data:
        item['linked_item_list'] = []
        item['linked_item_to_label'] = {}
        processed_input_data.append(item)
    #QALD没有linked_items域
    #input_data = [item for item in input_data if item['question'] == "which award piers anthony short listed for that's work is the Crewel Lye: A Caustic Yarn"]
    #input_data = [item for item in input_data if item['question'] == "what are all the engine types that have an engine horsepower under 429.0?"]
    #dump_json(data, "/home5/yhbao/SPARQLannotation/output/grailqa/GrailQA_train_golden_not_use_neighbor_entity_train10000_temp.json")
    search_algorithm = LFSearchAlgorithm(dataset, log_file_dir=logger_dir, use_golden_entity=use_golden_entity, use_neighbor_entity = use_neighbor_entity)
    #search_algorithm.begin_search(input_data, return_sexpr= False)
    split_search(search_algorithm, processed_input_data, output_dir=output_dir, split_size= 100)



def find_questions_with_different_answers_cwq():
    logger = setup_custom_logger("/home5/yhbao/SPARQLannotation/output/log/cwq_KB_test.log")
    sparql_executor = SparqlOdbcQuerierNoSexpr(ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger)
    notcorrect = 0
    not_correct_list = []
    sparql_data = load_json("/home5/yhbao/SPARQLannotation/output/cwq/cwq_dev_golden_not_use_neighbor_entity_nov18_sexpr_gpt_rerank_updated.json")\
        .update(load_json("/home5/yhbao/SPARQLannotation/output/cwq/cwq_train_golden_not_use_neighbor_entity_nov18_sexpr_gpt_rerank_updated.json"))
    data = load_json("/home5/yhbao/data/cwq/cwq_dev_linking.json") + load_json("/home5/yhbao/data/cwq/cwq_train_linking.json")
    for idx, item in tqdm(enumerate(data)):
        if len(sparql_data[item['qid']]['search_results']) > 0:
            if "gpt_select_top1" in sparql_data[item['qid']]['search_results']:
                sparql = sparql_data[item['qid']]['search_results']["gpt_select_top1"]['sparql']
            else:
                sparql = sparql_data[item['qid']]['search_results'][0]['sparql']
            results = sparql_executor.get_execution_result_one_variable_sparql_wrapper(sparql)
            answer_set = set([ans['mid'] for ans in item['answer']])
            if results != answer_set:
                notcorrect += 1
                not_correct_list.append(item['qid'])
            if idx % 100 == 0:
                print(notcorrect)
    dump_json(not_correct_list, ("/home5/yhbao/data/cwq/wrongondkilab_traindev.json"))


def update_data(old_data_dir, update_data_dir, output_data_dir):
    #假设有两个文件，利用update_data中的结果，覆盖old_data_dir中的相应结果
    updated_data = {}
    old_data = load_json(old_data_dir)
    update_dict = load_json(update_data_dir)
    for qid, item in tqdm(old_data.items()):
        if qid in update_dict:
            updated_data[qid] = update_dict[item['qid']]
        else:
            updated_data[qid] = item
    dump_json(updated_data, output_data_dir)

def combine_annotation_results(output_dir="/home5/yhbao/SPARQLannotation/output/wd_annotation/Annotation_combined_nov25.json"):
    def get_labelized_sparql_wikidata(sparql, qid2label_dict, pid2label_dict):
        labelized_sparql = sparql
        qids = list(set(re.findall(r"Q\d+", sparql)))
        qids.sort(key = lambda i:len(i), reverse=True)
        pids = list(set(re.findall(r"P\d+", sparql)))
        pids.sort(key = lambda i:len(i), reverse=True)
        for pid in pids:
            if pid in pid2label_dict:
                labelized_sparql = labelized_sparql.replace(pid, pid2label_dict[pid])
        for qid in qids:
            if "wd:"+qid in qid2label_dict:
                labelized_sparql = labelized_sparql.replace("wd:"+qid, qid2label_dict["wd:"+qid])
        return labelized_sparql
    combined_data = []
    #整理标注数据集
    logger = setup_custom_logger("/home5/yhbao/SPARQLannotation/output/log/wd_annotation_combine.log")
    wikidata_sparql_executor = SparqlOdbcQuerierWikidata(odbc_config=ODBC_CONFIG_WIKIDATA_2023, sparql_wrapper_path=SPARQL_wrapper_path_wikidata_2023, logger=logger, timeout=10)
    #应当提供：问题、实体结果（链接或golden)， 答案（mention）,SPARQL标注模型输出
    pid2label_dict = load_json("/home5/yhbao/SPARQLannotation/data/WD_ontology/wikidata_property_id2label_dict.json")
    quad_golden_path = "/home5/yhbao/SPARQLannotation/output/wd_annotation/WD_annotation_golden_not_use_neighbor_entity_dec03_gpt_rerank.json"
    quad_linked_path = "/home5/yhbao/SPARQLannotation/output/wd_annotation/WD_annotation_linked_not_use_neighbor_entity_dec03_gpt_rerank.json"
    query_agent_golden_path = "/home5/whzhou/STAR-QC/data/final_test/queryagent/queryagent_golden.json"
    query_agent_linked_path = "/home5/whzhou/STAR-QC/data/final_test/queryagent/queryagent_linked.json"
    data_file = "/home5/yhbao/data/annotated_test/annotation_dataset.json"
    quad_golden_results = load_json(quad_golden_path)
    quad_linked_results = load_json(quad_linked_path)
    queryagent_golden_results = {str(item['qid']):item for item in load_json(query_agent_golden_path)}
    queryagent_linked_results = {str(item['qid']):item for item in load_json(query_agent_linked_path)}
    for item in tqdm(load_json(data_file)):
        qid = item['qid']
        qid2label_dict = item["linked_item_to_label"]
        if "golden_item_to_label" in item:
            qid2label_dict.update(item["golden_item_to_label"])
        if len(quad_golden_results[qid]["search_results"]) >= 1:
            if len(quad_golden_results[qid]["search_results"]) == 1:
                quad_golden_sparql = quad_golden_results[qid]["search_results"][0]['sparql']
            else:
                quad_golden_sparql = quad_golden_results[qid]['gpt_select_top1']['sparql']
        else:
            quad_golden_sparql = ""
        if quad_golden_sparql == "":
            quad_golden_execution_result = []
        else:
            quad_golden_execution_result = list(wikidata_sparql_executor.get_execution_result_one_variable_sparql_wrapper(quad_golden_sparql))
        if len(quad_linked_results[qid]["search_results"]) >= 1:
            if len(quad_linked_results[qid]["search_results"]) == 1:
                quad_linked_sparql = quad_linked_results[qid]["search_results"][0]['sparql']
            else:
                quad_linked_sparql = quad_linked_results[qid]['gpt_select_top1']['sparql']
        else:
            quad_linked_sparql = ""
        if quad_linked_sparql == "":
            quad_linked_execution_result = []
        else:
            quad_linked_execution_result = list(wikidata_sparql_executor.get_execution_result_one_variable_sparql_wrapper(quad_linked_sparql))
        agent_golden_sparql = queryagent_golden_results[qid]['gen_sparql']
        agent_golden_execution_result = [item.replace("http://www.wikidata.org/entity/", "wd:") for item in queryagent_golden_results[qid]['pred']]
        agent_linked_sparql = queryagent_linked_results[qid]['gen_sparql']
        agent_linked_execution_result = [item.replace("http://www.wikidata.org/entity/", "wd:") for item in queryagent_linked_results[qid]['pred']]
        if len(agent_golden_execution_result) > 5:
            agent_golden_execution_result = agent_golden_execution_result[0:5]+ ["and more........"]
        if len(agent_linked_execution_result) > 5:
            agent_linked_execution_result = agent_linked_execution_result[0:5]+ ["and more........"]
        if len(quad_golden_execution_result) > 5:
            quad_golden_execution_result = quad_golden_execution_result[0:5]+ ["and more........"]
        if len(quad_linked_execution_result) > 5:
            quad_linked_execution_result = quad_linked_execution_result[0:5]+ ["and more........"]
        item['quad_golden_result'] = {'labelized':get_labelized_sparql_wikidata(quad_golden_sparql, qid2label_dict, pid2label_dict), 
                                      'sparql': quad_golden_sparql,
                                      'result': quad_golden_execution_result}
        item['quad_linked_result'] = {'labelized':get_labelized_sparql_wikidata(quad_linked_sparql, qid2label_dict, pid2label_dict), 
                                      'sparql': quad_linked_sparql,
                                      'result':quad_linked_execution_result}
        item['queryagent_golden_result'] = {'labelized':get_labelized_sparql_wikidata(agent_golden_sparql, qid2label_dict, pid2label_dict), 
                                            'sparql': agent_golden_sparql,
                                            'result': agent_golden_execution_result}
        item['queryagent_linked_result'] = {'labelized':get_labelized_sparql_wikidata(agent_linked_sparql, qid2label_dict, pid2label_dict), 
                                            'sparql': agent_linked_sparql,
                                            'result': agent_linked_execution_result}
        combined_data.append(item)
    dump_json(combined_data, output_dir)

def distribute_annotation_result(seed, data):
    def get_answer_rep(item):
        if len(item['answer_mention']) == 0:
            answer_mention = list(item['answer_linked_item_to_label'].values())[0]
        else:
            answer_mention = item['answer_mention']
        answer_linkings = item['answer_linked_item_to_label']
        return {"mention":answer_mention, "linkings":answer_linkings}
    
    random.seed(seed)
    nq_questions = data[0:100]
    hotpot_questions = data[100:]
    random.shuffle(nq_questions)
    random.shuffle(hotpot_questions)
    nq_random_splits = [nq_questions[i * 20:(i+1) * 20] for i in range(0, 5)]
    hotpot_random_splits = [hotpot_questions[i * 20:(i+1) * 20] for i in range(0, 5)]
    final_splits = [nq_random_splits[i] + hotpot_random_splits[i] for i in range(0,5)]
    for i in range(0,5):
        random.shuffle(final_splits[i])
    #A:什么都不给
    A_split = []
    for item in final_splits[0]:
        if 'golden_item_to_label' not in item:
            item["golden_item_to_label"] = []
        A_split.append({"qid":item['qid'], "question":item['question'], "answer":get_answer_rep(item),
                        "linked_items":item["linked_item_to_label"], "simulated_LF": {"labelized":"", "sparql":"", "result":[]}})
    #B:QUERYAGENT LINKED
    B_split = []
    for item in final_splits[1]:
        if 'golden_item_to_label' not in item:
            item["golden_item_to_label"] = []
        B_split.append({"qid":item['qid'], "question":item['question'], "answer":get_answer_rep(item), 
                        "linked_items":item["linked_item_to_label"], "simulated_LF":item['queryagent_linked_result']})
    #C:QUERYAGENT GOLDEN
    C_split = []
    if 'golden_item_to_label' not in item:
        item["golden_item_to_label"] = []
    for item in final_splits[2]:
        C_split.append({"qid":item['qid'], "question":item['question'], "answer":get_answer_rep(item), 
                        "linked_items":item["golden_item_to_label"], "simulated_LF":item['queryagent_golden_result']})
    #D:QUAD LINKED
    D_split = []
    for item in final_splits[3]:
        if 'golden_item_to_label' not in item:
            item["golden_item_to_label"] = []
        D_split.append({"qid":item['qid'], "question":item['question'], "answer":get_answer_rep(item), 
                        "linked_items":item["linked_item_to_label"], "simulated_LF":item['quad_linked_result']})
    #E:QueryAgent Golden
    E_split = []
    for item in final_splits[4]:
        if 'golden_item_to_label' not in item:
            item["golden_item_to_label"] = []
        E_split.append({"qid":item['qid'], "question":item['question'], "answer":get_answer_rep(item), 
                        "linked_items":item["golden_item_to_label"], "simulated_LF":item['quad_golden_result']})
    shuffled_data = A_split + B_split + C_split + D_split + E_split
    random.shuffle(shuffled_data)
    ordered_data = {"A":A_split, "B":B_split, "C":C_split, "D":D_split, "E":E_split}
    return ordered_data, shuffled_data

if __name__ == "__main__":
    #获取当前时间
    current_time = datetime.datetime.now()
    # 格式化时间为易读且带下划线
    os.environ['TRANSFORMERSW_OFFLINE'] = '1'  # 模型
    os.environ['HF_DATASETS_OFFLINE'] = '1'  # 数据
    #==============================================一些文件调整================================================================================
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    # WebQSP_search_experiment_test1000(use_golden_entity=True, experiment_tag="20260120_golden_elwith_timelimit", size=200)
    # WebQSP_search_experiment_test1000(use_golden_entity=False, experiment_tag="20260120_facc1_el_with_timelimit", size=200)
    # GrailQA_search_experiment_dev_1000(use_golden_entity=True, experiment_tag="20260120_golden_elwith_timelimit", size=200)
    # GrailQA_search_experiment_dev_1000(use_golden_entity=False, experiment_tag="20260120_facc1_el_with_timelimit", size=200)
    # CWQ_search_experiment_test_0_1000(link_setting="golden", experiment_tag = "20260120_golden_elwith_timelimit", source_file = "/home5/yhbao/data/cwq/cwq_test_1000_facc1_linking.json", size=200)
    # CWQ_search_experiment_test_0_1000(link_setting="linked", experiment_tag = "20260120_qgg_linking_with_timelimit", source_file = "/home5/yhbao/data/cwq/cwq_test_1000_qgg_linking.json", size=200)
    # CWQ_search_experiment_test_0_1000(link_setting="linked", experiment_tag = "dec03", source_file = "/home5/yhbao/data/cwq/cwq_test_1000_facc1_linking.json")

    get_top1_sparql_by_gpt(search_result_file="/home5/yhbao/SPARQLannotation/output/grailqa_final/grailqa_dev_0_1000_linked_not_use_neighbor_entity_20260120_facc1_el.json",
                        data_file="/home5/yhbao/data/grailqa/grailqa_dev_1000_facc1_linking.json",
                        dataset=DATASET.GRAIL,
                        output_dir="/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/grailqa_dev_0_1000_linked_not_use_neighbor_entity_20260120_facc1_el_gpt4omini_rerank.json",
                        model='gpt-4o-mini',
                        )
    get_top1_sparql_by_gpt(search_result_file="/home5/yhbao/SPARQLannotation/output/webqsp_final/webqsp_test_1000_linked_not_use_neighbor_entity_20260120_facc1_el.json",
                        data_file="/home5/yhbao/data/webqsp/webqsp_0_1000_linking.json",
                        dataset=DATASET.WEBQ,
                        output_dir="/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/webqsp_test_1000_linked_not_use_neighbor_entity_20260120_facc1_el_gpt4omini_rerank.json",
                        model='gpt-4o-mini',
                        )
    get_top1_sparql_by_gpt(search_result_file="/home5/yhbao/SPARQLannotation/output/cwq/cwq_test_linked_not_use_neighbor_entity_20260120_qgg_linking.json",
                        data_file="/home5/yhbao/data/cwq/cwq_test_1000_qgg_linking.json",
                        dataset=DATASET.CWQ,
                        output_dir="/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/cwq_test_linked_not_use_neighbor_entity_20260120_qgg_linking_gpt4omini_rerank.json",
                        model='gpt-4o-mini',
                        )


    # get_top1_sparql_by_gpt(search_result_file="/home5/yhbao/SPARQLannotation/output/webqsp_final/webqsp_test_1000_golden_not_use_neighbor_entity_dec03.json",
    #                     data_file="/home5/yhbao/data/webqsp/webqsp_0_1000_linking.json",
    #                     dataset=DATASET.WEBQ,
    #                     output_dir="/home5/yhbao/SPARQLannotation/output/webqsp_final/webqsp_test_1000_golden_not_use_neighbor_entity_dec03_gpt_rerank.json",
    #                     model='gpt-3.5-turbo-0125',
    #                     )
    # get_top1_sparql_by_gpt(search_result_file="/home5/yhbao/SPARQLannotation/output/webqsp_final/webqsp_test_1000_linked_not_use_neighbor_entity_dec03.json",
    #                     data_file="/home5/yhbao/data/webqsp/webqsp_0_1000_linking.json",
    #                     dataset=DATASET.WEBQ,
    #                     output_dir="/home5/yhbao/SPARQLannotation/output/webqsp_final/webqsp_test_1000_linked_not_use_neighbor_entity_dec_gpt_rerank.json",
    #                     model='gpt-3.5-turbo-0125',
    #                     )
    # get_top1_sparql_by_gpt(search_result_file="/home5/yhbao/SPARQLannotation/output/cwq_final/cwq_test_linked_not_use_neighbor_entity_dec03.json",
    #                     data_file="/home5/yhbao/data/cwq/cwq_test_1000_facc1_linking.json",
    #                     dataset=DATASET.CWQ,
    #                     output_dir="/home5/yhbao/SPARQLannotation/output/cwq_final/cwq_test_linked_not_use_neighbor_entity_dec03_facc1_gpt_rerank.json",
    #                     model='gpt-3.5-turbo-0125',
    #                     )
    #find_questions_with_different_answers_cwq()
    # CWQ_search_experiment('train', 'golden', "nov18_sexpr")
    # CWQ_search_experiment("dev", 'golden', "nov18_sexpr")
    # CWQ_search_experiment('train', 'linked', "nov18_sexpr")
    # CWQ_search_experiment("dev", 'linked', "nov18_sexpr")
    # GrailQA_search_experiment("dev", True, "nov28_sexpr")
    # GrailQA_search_experiment("train", True, "nov28_sexpr")
    # GrailQA_search_experiment("dev", False, "nov28_sexpr")
    # GrailQA_search_experiment("train", False, "nov28_sexpr")
    # WebQSP_search_experiment("test", True, "nov28_sexpr")
    # WebQSP_search_experiment("train", True, "nov28_sexpr")
    # WebQSP_search_experiment("test")
    # WebQSP_search_experiment("train")
    #------------------------------------------------------        golden rerank ---------------------------------------------------------------
    # get_top1_sparql_by_gpt(search_result_file="/home5/yhbao/SPARQLannotation/output/grailqa/grailqa_train_golden_not_use_neighbor_entity_nov28_sexpr.json",
    #                        data_file="/home5/yhbao/data/grailqa/grailqa_v1.0_train_linking.json",
    #                        dataset=DATASET.GRAIL,
    #                        output_dir="/home5/yhbao/SPARQLannotation/output/grailqa/grailqa_train_golden_not_use_neighbor_entity_nov28_sexpr_gpt_rerank.json",
    #                        model='gpt-3.5-turbo-0125',
    #                        )
    
    # get_top1_sparql_by_gpt(search_result_file="/home5/yhbao/SPARQLannotation/output/grailqa/grailqa_dev_golden_not_use_neighbor_entity_nov28_sexpr.json",
    #                        data_file="/home5/yhbao/data/grailqa/grailqa_v1.0_dev_linking.json",
    #                        dataset=DATASET.GRAIL,
    #                        output_dir="/home5/yhbao/SPARQLannotation/output/grailqa/grailqa_dev_golden_not_use_neighbor_entity_nov28_sexpr_gpt_rerank.json",
    #                        model='gpt-3.5-turbo-0125',
    #                        )
    # get_top1_sparql_by_gpt(search_result_file="/home5/yhbao/SPARQLannotation/output/webqsp/webqsp_test_golden_not_use_neighbor_entity_nov28_sexpr.json",
    #                        data_file="/home5/yhbao/data/webqsp/webqsp_test_linking.json",
    #                        dataset=DATASET.WEBQ,
    #                        output_dir="/home5/yhbao/SPARQLannotation/output/webqsp/webqsp_test_golden_not_use_neighbor_entity_nov28_sexpr_gpt_rerank.json",
    #                        model='gpt-3.5-turbo-0125',
    #                        )
    # get_top1_sparql_by_gpt(search_result_file="/home5/yhbao/SPARQLannotation/output/webqsp/webqsp_train_golden_not_use_neighbor_entity_nov28_sexpr.json",
    #                     data_file="/home5/yhbao/data/webqsp/webqsp_train_linking.json",
    #                     dataset=DATASET.WEBQ,
    #                     output_dir="/home5/yhbao/SPARQLannotation/output/webqsp/webqsp_train_golden_not_use_neighbor_entity_nov28_sexpr_gpt_rerank.json",
    #                     model='gpt-3.5-turbo-0125',
    #                     )
    # get_top1_sparql_by_gpt(search_result_file="/home5/yhbao/SPARQLannotation/output/cwq/cwq_dev_golden_not_use_neighbor_entity_nov18_sexpr.json",
    #                     data_file="/home5/yhbao/data/cwq/cwq_dev_linking.json",
    #                     dataset=DATASET.CWQ,
    #                     output_dir="/home5/yhbao/SPARQLannotation/output/cwq/cwq_dev_golden_not_use_neighbor_entity_nov18_sexpr_gpt_rerank.json",
    #                     model='gpt-3.5-turbo-0125',
    #                     )
    # get_top1_sparql_by_gpt(search_result_file="/home5/yhbao/SPARQLannotation/output/cwq/cwq_train_golden_not_use_neighbor_entity_nov18_sexpr.json",
    #                     data_file="/home5/yhbao/data/cwq/cwq_train_linking.json",
    #                     dataset=DATASET.CWQ,
    #                     output_dir="/home5/yhbao/SPARQLannotation/output/cwq/cwq_train_golden_not_use_neighbor_entity_nov18_sexpr_gpt_rerank.json",
    #                     model='gpt-3.5-turbo-0125',
    #                     )
    #------------------------------------------ linked rerank -----------------------------------------------------------------------------------------
    # get_top1_sparql_by_gpt(search_result_file="/home5/yhbao/SPARQLannotation/output/grailqa/GrailQA_train_linked_not_use_neighbor_entity_nov6_sexpr_10000.json",
    #                        data_file="/home5/yhbao/data/grailqa/grailqa_v1.0_train_linking.json",
    #                        dataset=DATASET.GRAIL,
    #                        output_dir="/home5/yhbao/SPARQLannotation/output/grailqa/GrailQA_train_linked_not_use_neighbor_entity_nov6_sexpr_10000_gpt4mini_rerank.json",
    #                        model='gpt-3.5-turbo-0125',
    #                        )
    
    # get_top1_sparql_by_gpt(search_result_file="/home5/yhbao/SPARQLannotation/output/grailqa/grailqa_train_linked_not_use_neighbor_entity_nov18_sexpr.json",
    #                        data_file="/home5/yhbao/data/grailqa/grailqa_v1.0_train_linking.json",
    #                        dataset=DATASET.GRAIL,
    #                        output_dir="/home5/yhbao/SPARQLannotation/output/grailqa/grailqa_train_linked_not_use_neighbor_entity_nov18_sexpr_gpt3.5_rerank.json",
    #                        model='gpt-3.5-turbo-0125',
    #                        )
    # get_top1_sparql_by_gpt(search_result_file="/home5/yhbao/SPARQLannotation/output/webqsp/webqsp_test_linked_not_use_neighbor_entity_nov18_sexpr.json",
    #                        data_file="/home5/yhbao/data/webqsp/webqsp_test_linking.json",
    #                        dataset=DATASET.WEBQ,
    #                        output_dir="/home5/yhbao/SPARQLannotation/output/webqsp/webqsp_test_linked_not_use_neighbor_entity_nov18_sexpr_gpt3.5_rerank.json",
    #                        model='gpt-3.5-turbo-0125',
    #                        )
    # get_top1_sparql_by_gpt(search_result_file="/home5/yhbao/SPARQLannotation/output/webqsp/webqsp_train_linked_not_use_neighbor_entity_nov18_sexpr.json",
    #                     data_file="/home5/yhbao/data/webqsp/webqsp_train_linking.json",
    #                     dataset=DATASET.WEBQ,
    #                     output_dir="/home5/yhbao/SPARQLannotation/output/webqsp/webqsp_train_linked_not_use_neighbor_entity_nov18_sexpr_gpt3.5_rerank.json",
    #                     model='gpt-3.5-turbo-0125',
    #                     )
    # get_top1_sparql_by_gpt(search_result_file="/home5/yhbao/SPARQLannotation/output/cwq/cwq_dev_linked_not_use_neighbor_entity_nov18_sexpr.json",
    #                     data_file="/home5/yhbao/data/cwq/cwq_dev_linking.json",
    #                     dataset=DATASET.CWQ,
    #                     output_dir="/home5/yhbao/SPARQLannotation/output/cwq/cwq_dev_linked_not_use_neighbor_entity_nov18_sexpr_gpt_3.5_rerank.json",
    #                     model='gpt-3.5-turbo-0125',
    #                     )
    # get_top1_sparql_by_gpt(search_result_file="/home5/yhbao/SPARQLannotation/output/cwq/cwq_train_linked_not_use_neighbor_entity_nov18_sexpr.json",
    #                     data_file="/home5/yhbao/data/cwq/cwq_train_linking.json",
    #                     dataset=DATASET.CWQ,
    #                     output_dir="/home5/yhbao/SPARQLannotation/output/cwq/cwq_train_linked_not_use_neighbor_entity_nov18_sexpr_gpt3.5_rerank.json",
    #                     model='gpt-3.5-turbo-0125',
    #                     )
    #=================================================Wikidata Annotation Experiment========================================================================
    #--------------------------------------search-------------------------------------------
    # Wikidata_annotation_experiment(use_golden_entity=True, experiment_tag = "dec03")
    # Wikidata_annotation_experiment(use_golden_entity=False, experiment_tag = "dec03")
    # #-------------------------------------ranking results-----------------------------------
    # combine_wikidata_annotation_results_with_possible_answers("/home5/yhbao/SPARQLannotation/output/wd_annotation/WD_annotation_golden_not_use_neighbor_entity_dec03.json",
    #                                                           "/home5/yhbao/SPARQLannotation/output/wd_annotation/WD_annotation_golden_not_use_neighbor_entity_dec03_combined.json")
    # combine_wikidata_annotation_results_with_possible_answers("/home5/yhbao/SPARQLannotation/output/wd_annotation/WD_annotation_linked_not_use_neighbor_entity_dec03.json",
    #                                                           "/home5/yhbao/SPARQLannotation/output/wd_annotation/WD_annotation_linked_not_use_neighbor_entity_dec03_combined.json")
    # #-----------------------------------ranking search results-------------------------------------------------------------------------------------------
    # get_top1_sparql_by_gpt("/home5/yhbao/SPARQLannotation/output/wd_annotation/WD_annotation_golden_not_use_neighbor_entity_dec03_combined.json",
    #                        "/home5/yhbao/data/annotated_test/annotation_dataset.json",
    #                        "/home5/yhbao/SPARQLannotation/output/wd_annotation/WD_annotation_golden_not_use_neighbor_entity_dec03_gpt_rerank.json",
    #                        model='gpt-3.5-turbo-0125', dataset=DATASET.SIMULATED_WIKIDATA)
    # get_top1_sparql_by_gpt("/home5/yhbao/SPARQLannotation/output/wd_annotation/WD_annotation_linked_not_use_neighbor_entity_dec03_combined.json",
    #                        "/home5/yhbao/data/annotated_test/annotation_dataset.json",
    #                        "/home5/yhbao/SPARQLannotation/output/wd_annotation/WD_annotation_linked_not_use_neighbor_entity_dec03_gpt_rerank.json",
    #                        model='gpt-3.5-turbo-0125', dataset=DATASET.SIMULATED_WIKIDATA)
    # #------------------------------------combining results---------------------------------
    # combine_annotation_results("/home5/yhbao/SPARQLannotation/output/wd_annotation/Annotation_combined_dec03.json")
    #------------------------------------distribute results---------------------------------
    # data = load_json("/home5/yhbao/SPARQLannotation/output/wd_annotation/Annotation_combined_nov25.json")
    # for i in range(0, 5):
    #     order_data, shuffle_data = distribute_annotation_result(i, data)
    #     dump_json(shuffle_data, f"/home5/yhbao/SPARQLannotation/output/annotation_splits_final/split_{str(i)}_shuffled.json")
    #     dump_json(order_data, f"/home5/yhbao/SPARQLannotation/output/annotation_splits_final/split_{str(i)}_ordered.json")
    # data = load_json("/home5/yhbao/SPARQLannotation/output/wd_annotation/Annotation_combined_nov28.json")
    # for i in range(0, 5):
    #     order_data, shuffle_data = distribute_annotation_result(i, data)
    #     dump_json(shuffle_data, f"/home5/yhbao/SPARQLannotation/output/annotation_splits_final/split_{str(i)}_shuffled_nov28.json")
    #     dump_json(order_data, f"/home5/yhbao/SPARQLannotation/output/annotation_splits_final/split_{str(i)}_ordered_nov28.json")
    #-------------------------------------------------------------------------------------------------------------