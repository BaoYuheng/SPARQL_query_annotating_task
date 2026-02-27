import sys 
sys.path.append('/home5/yhbao/SPARQLannotation')

from src.sparql_executor import SparqlOdbcQuerier, SparqlOdbcQuerierWikidata
from src.utils import load_json, setup_custom_logger, dump_json, get_PRF1
from src.common import (
    DATASET, 
    ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, ODBC_CONFIG_WIKIDATA_2019, SPARQL_wrapper_path_wikidata_2019,
    ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab,SPARQL_wrapper_path_wikidata_2023,ODBC_CONFIG_WIKIDATA_2023,
    FreebaseConstantForConstruction, FREEBASE_CONSTANT_TYPE, get_syntax_tree_string,
    get_syntax_tree_value, WikidataConstantForConstruction, WIKIDATA_CONSTANT_TYPE,
    WIKIDATA_PREFIX_LIST
)
from src.sparql_utils_new import SyntaxTreeEditor, TreeConstructor, TreeEditDistance
from src.s_expression_utils_new import (
    sexp_to_sparql, sexp_to_sparql_for_edit_distance, 
    sexp_to_sparql_wikidata_for_edit_distance, 
    #2026 01 22 修改
    sexp_to_sparql_for_test_suite,
    sexp_to_sparql_wikidata_for_test_suite
) 
from src.sparql_process import pre_process_sparql
from collections import defaultdict
from tqdm import tqdm
import random
import math
import copy
import itertools
from typing import Union
import re
import numpy as np

REPLACED_VARIABLE_NAME="replace"
TARGET_VARIABLE_NAME="target" # 现有数据集中，没发现重名的变量
TEST_SUITE_LIMIT = 50
NS_PREFIX = "http://rdf.freebase.com/ns/"

def jaccard_similarity(set1, set2):
    if len(set1)==0 or len(set2)==0:
        return 0.0
    set1 = set(set1)
    set2 = set(set2)
    return float(
        len(set1.intersection(set2)) / len(set1.union(set2))
    )

def replaceable_items_detect(editor,replaceable_items,dataset,logger):
    reground_map = dict()
    new_reground_map = dict()
    for (idx, item) in enumerate(replaceable_items):
        if dataset in [DATASET.CWQ, DATASET.WEBQ, DATASET.GRAIL]:
            if FreebaseConstantForConstruction.get_constant_type(item) is FREEBASE_CONSTANT_TYPE.QUANTITY:
                old_value = item.split('^^')[0][1:-1]
                # suffix = item.split('^^')[1]
                # if suffix is "<http://www.w3.org/2001/XMLSchema#integer>":
                #     reground_map[get_syntax_tree_string(item, dataset)] = [
                #         f'"{old_value}"',
                #         f'"{str(int(old_value) * 2)}"',
                #         f'"{str(int(old_value) * 3)}"',
                #     ]
                # else:
                #     reground_map[get_syntax_tree_string(item, dataset)] = [
                #         f'"{old_value}"',
                #         f'"{str(float(old_value) * 2.0)}"',
                #         f'"{str(float(old_value) * 3.0)}"',
                #     ]
                reground_map[get_syntax_tree_string(item, dataset)] = [ f'"{old_value}"']
            elif FreebaseConstantForConstruction.get_constant_type(item) is FREEBASE_CONSTANT_TYPE.TIME:
                # 日期就没法改了
                reground_map[get_syntax_tree_string(item, dataset)] = [get_syntax_tree_value(item, dataset)]
            else: # 其他: 替换为变量
                _tree_string = get_syntax_tree_string(item, dataset)
                variable = f"?{REPLACED_VARIABLE_NAME}{idx}" # 在语法树中，变量的 value 是带有 ? 号的
                reground_map[_tree_string] = [variable]
        elif dataset in [DATASET.LC2, DATASET.QALD]:
            if WikidataConstantForConstruction.get_constant_type(item) is WIKIDATA_CONSTANT_TYPE.QUANTITY:
                old_value = item.split('^^')[0][1:-1]
                suffix = item.split('^^')[1]
                if suffix is "<http://www.w3.org/2001/XMLSchema#integer>":
                    reground_map[get_syntax_tree_string(item, dataset)] = [
                        f'"{old_value}"',
                        f'"{str(int(old_value) * 2)}"',
                        f'"{str(int(old_value) * 3)}"',
                    ]
                else:
                    reground_map[get_syntax_tree_string(item, dataset)] = [
                        f'"{old_value}"',
                        f'"{str(float(old_value) * 2.0)}"',
                        f'"{str(float(old_value) * 3.0)}"',
                    ]
            elif FreebaseConstantForConstruction.get_constant_type(item) is WIKIDATA_CONSTANT_TYPE.TIME:
                # 日期就没法改了
                reground_map[get_syntax_tree_string(item, dataset)] = [get_syntax_tree_value(item, dataset)]
            else: # 其他: 替换为变量
                _tree_string = get_syntax_tree_string(item, dataset)
                variable = f"?{REPLACED_VARIABLE_NAME}{idx}" # 在语法树中，变量的 value 是带有 ? 号的
                reground_map[_tree_string] = [variable]
        else:
            raise NotImplementedError(f"dataset: {dataset}")

    for key in reground_map.keys():
        if editor.is_consist_node(key):
            new_reground_map[key] = reground_map[key]
    return new_reground_map
    
#20260122 eswc rebuttal 修改
def get_meaninful_perturb_result(editor,intersection_items,replaceable_item_regound_map,sparql_querier,dataset,logger):
    for item in intersection_items:
        value = replaceable_item_regound_map[item][0]
        if value.startswith('?'): # 变量
            flag = editor.replace_leaf_with_variable(item, value)
            if not flag:
                raise Exception(f"key: {item} not found in the syntax tree")
        else:
            flag = editor.update_leaf_value(item, value)
            if not flag:
                raise Exception(f"key: {item} not found in the syntax tree")

    sparql_txt_after_replace = editor.sparql_txt
    logger.info(sparql_txt_after_replace)
    #使用sparql wrapper
    rows = sparql_querier.execute_query(sparql_txt_after_replace, retry=3)
    answer_map = defaultdict(list)
    '''
    {
        "replace1,replace2...":target
    }
    '''
    for row in rows:
        try:
            replace_map = dict()
            for variable in row.keys():
                if variable.startswith(REPLACED_VARIABLE_NAME):
                    variable_idx = variable.replace(REPLACED_VARIABLE_NAME,'')
                    if dataset in [DATASET.GRAIL, DATASET.CWQ, DATASET.WEBQ]:
                        replace_map[variable_idx] = get_syntax_tree_value(
                                FreebaseConstantForConstruction(
                                    row[variable]['type'], row[variable]['value'],
                                    row[variable].get('datatype', None), row[variable].get('xml:lang', None)
                                ).__repr__(), dataset
                            )
                    elif dataset in [DATASET.LC2, DATASET.QALD]:
                        replace_map[variable_idx] = get_syntax_tree_value(
                                WikidataConstantForConstruction(
                                    row[variable]['type'], row[variable]['value'],
                                    row[variable].get('datatype', None), row[variable].get('xml:lang', None)
                                ).__repr__(), dataset
                            )
                    else:
                        raise NotImplementedError(f"dataset: {dataset}")
                elif variable == TARGET_VARIABLE_NAME:
                    if dataset in [DATASET.GRAIL, DATASET.CWQ, DATASET.WEBQ]:
                        answer = get_syntax_tree_value(
                                FreebaseConstantForConstruction(
                                    row[variable]['type'], row[variable]['value'],
                                    row[variable].get('datatype', None), row[variable].get('xml:lang', None)
                                ).__repr__(), dataset
                            )
                    elif dataset in [DATASET.LC2, DATASET.QALD]:
                        answer = get_syntax_tree_value(
                                WikidataConstantForConstruction(
                                    row[variable]['type'], row[variable]['value'],
                                    row[variable].get('datatype', None), row[variable].get('xml:lang', None)
                                ).__repr__(), dataset
                            )
                    else:
                        raise NotImplementedError(f"dataset: {dataset}")
            # 按replace的顺序排序，
            replace_result = [replace_map[key] for key in sorted(replace_map.keys(), key=int)]
            answer_map[','.join(replace_result)].append(answer)
        except Exception as e:
            continue
            # logger.error("error: %s", str(e), exc_info=True)
    return answer_map
        # except Exception as e:
        #     # logger.error(f"parsing error: {e}")
        #     skip_flag = True
        
def modified_wikidata_sparql(sparql):
    pattern = r"p:P\d+/ps:P\d+"
    prop = re.findall(pattern,sparql)
    for p in prop:
        pid = re.findall(r'\d+',p)
        if pid and pid[0] == pid[1]:
            sparql = sparql.replace(p,"wdt:P"+pid[0])
    return sparql

def get_overlapping_rate(file,dataset,logger_path,output_path):

    src_data = load_json(file)

    logger = setup_custom_logger(logger_path)

    if dataset == DATASET.GRAIL:
        sparql_querier = SparqlOdbcQuerier(
            ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, timeout=60
        )
    elif dataset == DATASET.CWQ:
        sparql_querier = SparqlOdbcQuerier(
            ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
        )
    elif dataset == DATASET.SIMULATED_WIKIDATA or dataset == DATASET.LC2:
        sparql_querier = SparqlOdbcQuerierWikidata(
            ODBC_CONFIG_WIKIDATA_2023, SPARQL_wrapper_path_wikidata_2023, logger, timeout=300
        )
    elif dataset == DATASET.WEBQ:
        sparql_querier = SparqlOdbcQuerier(
            ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
        )

    if dataset in [DATASET.CWQ, DATASET.GRAIL, DATASET.WEBQ]:
        prefix = "PREFIX ns: <http://rdf.freebase.com/ns/> "            
    elif dataset in [DATASET.LC2,DATASET.SIMULATED_WIKIDATA]:
        prefix = WIKIDATA_PREFIX_LIST
    
    for data in src_data:
        sparql1 = prefix+'\n' + data['golden_sparql']
        sparql2 = prefix+'\n' + data['sparql']
        # logger.info(sparql1)
        # logger.info(sparql2)

        golden_editor = SyntaxTreeEditor(sparql1, DATASET.QUERYAGENT, logger)
        golden_editor.preprocess_edit_distance(f"?{TARGET_VARIABLE_NAME}")

        simulated_editor = SyntaxTreeEditor(sparql2, DATASET.QUERYAGENT, logger)
        simulated_editor.preprocess_edit_distance(f"?{TARGET_VARIABLE_NAME}")

        # logger.info(simulated_editor.sparql_txt)
        # logger.info(golden_editor.sparql_txt)
        logger.info(golden_editor.items)
        logger.info(simulated_editor.items)
        

        overlapping_rate = len(simulated_editor.items.intersection(golden_editor.items))/len(simulated_editor.items.union(golden_editor.items))

        data['overlapping_rate'] = overlapping_rate
    
    dump_json(src_data,output_path)




from src.sparql_executor import SparqlOdbcQuerierNoSexpr
# def sparql_equal(editor1,editor2):
#20260122 eswc rebuttal修改
#def new_test_suite_similarity(sparql_item1,sparql_item2,dataset:DATASET, method:DATASET,sparql_querier:Union[SparqlOdbcQuerier, SparqlOdbcQuerierWikidata], logger):
def new_test_suite_similarity(sparql_item1,sparql_item2,dataset:DATASET, method:DATASET,sparql_querier:SparqlOdbcQuerierNoSexpr, logger):
    
    assert str(sparql_item1['qid']) == str(sparql_item2['qid']),print(sparql_item1['qid'], sparql_item2['qid'])
    logger.info(f"qid:{sparql_item1['qid']}")
    original_sparql_txt = sparql_item1["golden_sparql_query"]

    if dataset == DATASET.LC2:
        original_sparql_txt = modified_wikidata_sparql(original_sparql_txt)
    
    replaceable_items = [
        item['mid'] for item in sparql_item1["golden_grounded_items"]
    ]
    # 各数据集上，item 格式的特殊处理
    try:
        if dataset is DATASET.GRAIL:
            for (idx, item) in enumerate(replaceable_items):
                if item.endswith('gYear>') or item.endswith('gYearMonth>') or item.endswith('date>'):
                    replaceable_items[idx] = f'"{item.split("^^")[0][1:-1] + "-08:00"}"^^{item.split("^^")[1]}'
        
        # if dataset == DATASET.WEBQ:
        #     golden_sparqls = golden_sparql_list

        golden_editor = SyntaxTreeEditor(original_sparql_txt, dataset, logger)
        golden_editor.preprocess_test_suite(f"?{TARGET_VARIABLE_NAME}")

        simulated_sparql_txt = sparql_item2["simulated_sparql_query"]

        if dataset == DATASET.LC2:
            simulated_sparql_txt = modified_wikidata_sparql(simulated_sparql_txt)
        
        simulated_editor = SyntaxTreeEditor(simulated_sparql_txt, method, logger)
        simulated_editor.preprocess_test_suite(f"?{TARGET_VARIABLE_NAME}")
        logger.info(simulated_editor.sparql_txt)
        logger.info(golden_editor.sparql_txt)
        if simulated_editor.sparql_txt == golden_editor.sparql_txt:
            return 1, -1

        if len(replaceable_items) == 0:
            return 0, -1
        # 依次看replaceable item是否在sparql2和sparql1中
        golden_replaceable_item_regound_map = replaceable_items_detect(golden_editor,replaceable_items,dataset,logger)
        simulated_replaceable_item_regound_map = replaceable_items_detect(simulated_editor,replaceable_items,dataset,logger)
        intersection_items = golden_replaceable_item_regound_map.keys() & simulated_replaceable_item_regound_map.keys()
        # 各自抽象为变量，得到meaningful 扰动执行
        
        if len(intersection_items) == 0:
            return 0, -1

        logger.info(f"intersection_items:{intersection_items}")
        golden_result = get_meaninful_perturb_result(golden_editor,intersection_items,golden_replaceable_item_regound_map,sparql_querier,dataset,logger)
        
        logger.info(f"golden_result:{len(golden_result)}")
        simu_result = get_meaninful_perturb_result(simulated_editor,intersection_items,golden_replaceable_item_regound_map,sparql_querier,dataset,logger)
        logger.info(f"simu_result:{len(simu_result)}")
        # 计算指标
        all_replace = golden_result.keys() | simu_result.keys()
        # 20260122 eswc rebuttal修改 这里，我们返回共有多少个test suite
        logger.info(f"all_replace:{len(all_replace)}")
        similarity_list = []
        for replace in all_replace:
            golden_replace_result = golden_result.get(replace,[])
            simu_replace_result = simu_result.get(replace,[])
            similarity = jaccard_similarity(golden_replace_result,simu_replace_result)
            
            similarity_list.append(similarity)
        num = 0
        for s in similarity_list:
            if s == 1:
                num += 1

        logger.info(f"similarity_list:{num}")
        #20260122 eswc rebuttal 修改
        
        #这个平滑不知道在干啥
        return sum(similarity_list)/len(similarity_list), len(all_replace)
        # return smooth_sigmoid(sum(similarity_list)/len(similarity_list))
    except Exception as e:
        import traceback
        traceback.print_exc()
        pass
        logger.error(f"qid: {sparql_item2['qid']}; error: {e}")
        return 0, -1
    # print(sum(similarity_list)/len(similarity_list))

def new_test_suite_similarity_webqsp(sparql_item1,sparql_item2,dataset:DATASET, method:DATASET,sparql_querier:Union[SparqlOdbcQuerier, SparqlOdbcQuerierWikidata], logger):
    
    assert str(sparql_item1['qid']) == str(sparql_item2['qid']),print(sparql_item1['qid'], sparql_item2['qid'])
    logger.info(f"qid:{sparql_item1['qid']}")
    
    similist = []
    for golden_sparql in sparql_item1['golden_sparql_list']:
        original_sparql_txt = golden_sparql ["sparql"]

        if dataset == DATASET.LC2:
            original_sparql_txt = modified_wikidata_sparql(original_sparql_txt)

        replaceable_items = [
            item['mid'] for item in golden_sparql["golden_grounded_items"]
        ]
        # 各数据集上，item 格式的特殊处理
        try:
            if dataset is DATASET.GRAIL:
                for (idx, item) in enumerate(replaceable_items):
                    if item.endswith('gYear>') or item.endswith('gYearMonth>') or item.endswith('date>'):
                        replaceable_items[idx] = f'"{item.split("^^")[0][1:-1] + "-08:00"}"^^{item.split("^^")[1]}'
            
            # if dataset == DATASET.WEBQ:
            #     golden_sparqls = golden_sparql_list

            golden_editor = SyntaxTreeEditor(original_sparql_txt, dataset, logger)
            golden_editor.preprocess_test_suite(f"?{TARGET_VARIABLE_NAME}")

            simulated_sparql_txt = sparql_item2["simulated_sparql_query"]

            if dataset == DATASET.LC2:
                simulated_sparql_txt = modified_wikidata_sparql(simulated_sparql_txt)

            simulated_editor = SyntaxTreeEditor(simulated_sparql_txt, method, logger)
            simulated_editor.preprocess_test_suite(f"?{TARGET_VARIABLE_NAME}")

            if simulated_editor.sparql_txt == golden_editor.sparql_txt:
                similist.append(1)
                continue

            if len(replaceable_items) == 0:
                similist.append(0)
                continue
            # 依次看replaceable item是否在sparql2和sparql1中
            golden_replaceable_item_regound_map = replaceable_items_detect(golden_editor,replaceable_items,dataset,logger)
            simulated_replaceable_item_regound_map = replaceable_items_detect(simulated_editor,replaceable_items,dataset,logger)
            intersection_items = golden_replaceable_item_regound_map.keys() & simulated_replaceable_item_regound_map.keys()
            # 各自抽象为变量，得到meaningful 扰动执行
            
            if len(intersection_items) == 0:
                similist.append(0)
                continue

            logger.info(f"intersection_items:{intersection_items}")
            golden_result = get_meaninful_perturb_result(golden_editor,intersection_items,golden_replaceable_item_regound_map,sparql_querier,dataset,logger)
            
            # logger.info(f"golden_result:{golden_result}")
            simu_result = get_meaninful_perturb_result(simulated_editor,intersection_items,golden_replaceable_item_regound_map,sparql_querier,dataset,logger)
            # logger.info(f"simu_result:{simu_result}")
            # 计算指标
            all_replace = golden_result.keys() | simu_result.keys()
            similarity_list = []
            for replace in all_replace:
                golden_replace_result = golden_result.get(replace,[])
                simu_replace_result = simu_result.get(replace,[])
                similarity = jaccard_similarity(golden_replace_result,simu_replace_result)
                
                similarity_list.append(similarity)

            similist.append(sum(similarity_list)/len(similarity_list))
            continue 
        except Exception as e:
            logger.error(f"qid: {sparql_item2['qid']}; error: {e}")
            similist.append(0)
            continue

    return max(similist)    
    # print(sum(similarity_list)/len(similarity_list))
    
def test_test_suite(file,logger_path,output_file):
    import time
    start_time = time.time()
    logger = setup_custom_logger(logger_path)
    sparql_querier = SparqlOdbcQuerier(
            ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
        )
    method = DATASET.SIMULATED_FREEBASE
    dataset = DATASET.WEBQ
    src_data = load_json(file)
    f1_list = []
    detection_result = []

    for (data1,data2) in tqdm(src_data):
        similarity =  new_test_suite_similarity(data1,data2,dataset,method,sparql_querier,logger)
        data2["test_suite_similarity"] = similarity
        logger.info(f"similarity:{similarity}")
        f1_list.append(similarity)
        detection_result.append({"qid":data1['qid'],'sparql1':data1['golden_sparql_query'],'sparql2':data2['simulated_sparql_query'],'overlapping_rate':data1['overlapping_rate'],'similarity':similarity})

    detection_result.append({
        "average":sum(f1_list)/len(f1_list)
    })
    dump_json(detection_result,output_file)
    end_time = time.time()
    logger.info(f"total time: {end_time - start_time}")
    return


def test_new_test_suite(f1,f2,output_file,dataset,logger_path,aflag = True):
    from src.sparql_executor import SparqlOdbcQuerierNoSexpr
    from src.common import SPARQL_wrapper_path_dkilab
    #f1似乎是golden file f2似乎是output file
    import time
    start_time = time.time()
    detection_result = load_json(f1)
    logger = setup_custom_logger(logger_path)
    
    #20260122 eswc rebuttal 修改
    if dataset == DATASET.GRAIL:
        sparql_querier = SparqlOdbcQuerierNoSexpr(
            odbc_config=ODBC_CONFIG_DKILAB, sparql_wrapper_path=SPARQL_wrapper_path_dkilab, logger=logger, timeout=60
        )
        method = DATASET.SIMULATED_FREEBASE
    elif dataset == DATASET.CWQ:
        sparql_querier = SparqlOdbcQuerierNoSexpr(
            odbc_config=ODBC_CONFIG_DKILAB, sparql_wrapper_path=SPARQL_wrapper_path_dkilab, logger=logger, timeout=60
        )
        method = DATASET.CWQ
    elif dataset == DATASET.SIMULATED_FREEBASE:
        sparql_querier = SparqlOdbcQuerierNoSexpr(
           odbc_config=ODBC_CONFIG_DKILAB, sparql_wrapper_path=SPARQL_wrapper_path_dkilab, logger=logger, timeout=60
        )
        method = DATASET.SIMULATED_FREEBASE
    elif dataset == DATASET.SIMULATED_WIKIDATA or dataset == DATASET.LC2:
        assert(0)
        # sparql_querier = SparqlOdbcQuerierWikidata(
        #     ODBC_CONFIG_WIKIDATA_2023, SPARQL_wrapper_path_wikidata_2023, logger, timeout=300
        # )
    # if dataset == DATASET.GRAIL:
    #     sparql_querier = SparqlOdbcQuerier(
    #         ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, timeout=60
    #     )
    # elif dataset == DATASET.CWQ:
    #     sparql_querier = SparqlOdbcQuerier(
    #         ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    #     )
    # elif dataset == DATASET.SIMULATED_WIKIDATA or dataset == DATASET.LC2:
    #     sparql_querier = SparqlOdbcQuerierWikidata(
    #         ODBC_CONFIG_WIKIDATA_2023, SPARQL_wrapper_path_wikidata_2023, logger, timeout=300
    #     )
    # elif dataset == DATASET.WEBQ:
    #     sparql_querier = SparqlOdbcQuerier(
    #         ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    #     )

    # method = DATASET.SIMULATED_WIKIDATA
    src_data = load_json(f2)
 
    # src_data = src_data[:20]
    qid_to_src = {str(item["qid"]): item for item in src_data}
    # 20260122 eswc rebuttal 修改
    #需要记录开始和结束时间，计算平均耗时
    f1_list = []
    for data2 in tqdm(detection_result.values()):
        element_start_time = time.time()
        if 'qid' not in data2:
            continue
        # if data2['qid'] != "5ab7fa7b5542992aa3b8c897":
        #     continue
        if str(data2['qid']) not in qid_to_src:
            data2["test_suite_similarity"] = 0
            f1_list.append(0.0)
            continue

        data1 = qid_to_src[str(data2['qid'])]

        if not data2['search_results']:
            data2["test_suite_similarity"] = 0
            f1_list.append(0.0)
            continue
        
        if dataset in [DATASET.CWQ, DATASET.GRAIL, DATASET.WEBQ]:
            prefix = "PREFIX ns: <http://rdf.freebase.com/ns/> "            
        elif dataset in [DATASET.LC2,DATASET.SIMULATED_WIKIDATA]:
            prefix = WIKIDATA_PREFIX_LIST
            
        if aflag and "gpt_select_top1" in data2 and "sparql" in data2['gpt_select_top1']:
            simulated_sparql_txt = prefix + "\n"+data2['gpt_select_top1']['sparql']
        else:
            if 'sparql' in data2["search_results"][0]:
                simulated_sparql_txt = prefix + "\n"+data2["search_results"][0]["sparql"]
            else:
                simulated_sparql_txt = prefix + "\n"+data2["search_results"][0]


        data2['simulated_sparql_query'] = simulated_sparql_txt

        # if dataset == DATASET.WEBQ:
        #     similarity = new_test_suite_similarity_webqsp(data1,data2,dataset,method,sparql_querier,logger)
        # else:
        #2026 0122 eswc rebuttal 修改

        similarity, all_replace_num = new_test_suite_similarity(data1,data2,dataset,method,sparql_querier,logger)
        data2["time_consumed"] = time.time() - element_start_time
        data2["test_suite_similarity"] = similarity
        data2["all_replace_num"] = all_replace_num
        logger.info(f"similarity:{similarity}")
        f1_list.append(similarity)

    detection_result["test_suite_s"] = {
        # "s1_list":s1_list,
        "average":sum(f1_list)/len(f1_list)
    }
    dump_json(detection_result,output_file)
    end_time = time.time()
    logger.info(f"total time: {end_time - start_time}")
    return

def test_new_test_suite_old(f1,f2,output_file,dataset,method,aflag = True):
    detection_result = load_json(f1)
    logger = setup_custom_logger("data/metrics/new_new/grailqa/_test_suite.log")
    
    if dataset == DATASET.GRAIL:
        sparql_querier = SparqlOdbcQuerier(
            ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, timeout=60
        )
    elif dataset == DATASET.CWQ or dataset == DATASET.WEBQ:
        sparql_querier = SparqlOdbcQuerier(
            ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
        )
    
    src_data = load_json(f2)

    # src_data = src_data[:20]
    # detection_result = dict(list(detection_result.items())[:20])
    f1_list = []
    qid_to_src = {str(item["qid"]): item for item in src_data}
    for data2 in tqdm(detection_result['results']):
        if str(data2['qid']) not in qid_to_src:
            data2["test_suite_similarity"] = 0
            f1_list.append(0.0)
            continue
        data1 = qid_to_src[str(data2['qid'])]

        if ("searched_queries" not in data2) or len(data2["searched_queries"]) == 0:
            data2["test_suite_similarity"] = 0
            f1_list.append(0.0)
            continue
        
        if 'f1' in data2["searched_queries"][0]:
            if data2["searched_queries"][0]['f1'] == 0:
                data2["test_suite_similarity"] = 0
                f1_list.append(0.0)
                continue
        
        if method in [DATASET.QGG, DATASET.BINDER, DATASET.LSQ]:
            simulated_sparql_txt = data2["searched_queries"][0]["sparql"]
        elif method == DATASET.QUERYAGENT:
            simulated_sparql_txt = "PREFIX : <http://rdf.freebase.com/ns/>\n"+data2["searched_queries"][0]["sparql"]
        else:
            simulated_sparql_txt = sexp_to_sparql_for_test_suite(data2["searched_queries"][0]["s_expression"])

        data2['simulated_sparql_query'] = simulated_sparql_txt

        if dataset == DATASET.WEBQ:
            similarity = new_test_suite_similarity_webqsp(data1,data2,dataset,method,sparql_querier,logger)
        else:
            similarity = new_test_suite_similarity(data1,data2,dataset,method,sparql_querier,logger)

        # similarity = new_test_suite_similarity(data1,data2,dataset,method,sparql_querier,logger)
        data2["test_suite_similarity"] = similarity
        logger.info(f"similarity:{similarity}")
        f1_list.append(similarity)

    detection_result["test_suite_s"] = {
        # "s1_list":s1_list,
        "average":sum(f1_list)/len(f1_list)
    }
    dump_json(detection_result,output_file)
    return

def update_reground_map(sparql_txt_before_replace,replaceable_items,sparql_querier,dataset,logger):
    '''
    输出：
    reground：{
        ‘q1’：[q1,x1,x2,x3]，
        ‘q3’：[q3,x3,x4,x5]
    },
    '''
    reground_map = dict()
    for (idx, item) in enumerate(replaceable_items):
        if dataset in [DATASET.CWQ, DATASET.WEBQ, DATASET.GRAIL]:
            if FreebaseConstantForConstruction.get_constant_type(item) is FREEBASE_CONSTANT_TYPE.QUANTITY:
                old_value = item.split('^^')[0][1:-1]
                suffix = item.split('^^')[1]
                if suffix is "<http://www.w3.org/2001/XMLSchema#integer>":
                    reground_map[get_syntax_tree_string(item, dataset)] = [
                        f'"{old_value}"',
                        f'"{str(int(old_value) * 2)}"',
                        f'"{str(int(old_value) * 3)}"',
                    ]
                else:
                    reground_map[get_syntax_tree_string(item, dataset)] = [
                        f'"{old_value}"',
                        f'"{str(float(old_value) * 2.0)}"',
                        f'"{str(float(old_value) * 3.0)}"',
                    ]
            elif FreebaseConstantForConstruction.get_constant_type(item) is FREEBASE_CONSTANT_TYPE.TIME:
                # 日期就没法改了
                reground_map[get_syntax_tree_string(item, dataset)] = [get_syntax_tree_value(item, dataset)]
            else: # 其他: 替换为变量
                _tree_string = get_syntax_tree_string(item, dataset)
                variable = f"?{REPLACED_VARIABLE_NAME}{idx}" # 在语法树中，变量的 value 是带有 ? 号的
                reground_map[_tree_string] = [variable]
        elif dataset in [DATASET.LC2, DATASET.QALD]:
            if WikidataConstantForConstruction.get_constant_type(item) is WIKIDATA_CONSTANT_TYPE.QUANTITY:
                old_value = item.split('^^')[0][1:-1]
                suffix = item.split('^^')[1]
                if suffix is "<http://www.w3.org/2001/XMLSchema#integer>":
                    reground_map[get_syntax_tree_string(item, dataset)] = [
                        f'"{old_value}"',
                        f'"{str(int(old_value) * 2)}"',
                        f'"{str(int(old_value) * 3)}"',
                    ]
                else:
                    reground_map[get_syntax_tree_string(item, dataset)] = [
                        f'"{old_value}"',
                        f'"{str(float(old_value) * 2.0)}"',
                        f'"{str(float(old_value) * 3.0)}"',
                    ]
            elif FreebaseConstantForConstruction.get_constant_type(item) is WIKIDATA_CONSTANT_TYPE.TIME:
                # 日期就没法改了
                reground_map[get_syntax_tree_string(item, dataset)] = [get_syntax_tree_value(item, dataset)]
            else: # 其他: 替换为变量
                _tree_string = get_syntax_tree_string(item, dataset)
                variable = f"?{REPLACED_VARIABLE_NAME}{idx}" # 在语法树中，变量的 value 是带有 ? 号的
                reground_map[_tree_string] = [variable]
        else:
            raise NotImplementedError(f"dataset: {dataset}")

    # 先得到所有的reground 值
    for replaceable_item in reground_map.keys():
        editor = SyntaxTreeEditor(sparql_txt_before_replace, dataset, logger)
        editor.remove_projection_variable(f"?{TARGET_VARIABLE_NAME}")
        value = reground_map[replaceable_item][0]
        
        if value.startswith('?'): # 变量
            flag = editor.replace_leaf_with_variable(replaceable_item, value)
            if replaceable_item[0]=='<':
                reground_map[replaceable_item] = [replaceable_item[1:-1]]
            else:
                reground_map[replaceable_item] = [replaceable_item]
            if not flag:
                raise Exception(f"key: {replaceable_item} not found in the syntax tree")
            # 移除答案变量，减少返回数据的量，我们只要求答案非空即可，这一步不需要获得其值
            sparql_txt_after_replace = editor.sparql_txt
            rows = sparql_querier.execute_query(sparql_txt_after_replace, retry=3)
            for row in rows:
                skip_flag = False
                # if len(reground_map[replaceable_item])>=1000:
                #     break
                try:
                    if dataset in [DATASET.GRAIL, DATASET.CWQ, DATASET.WEBQ]:
                        reground_map[replaceable_item].append(get_syntax_tree_value(
                            FreebaseConstantForConstruction(
                                row[value[1:]]['type'], row[value[1:]]['value'],
                                row[value[1:]].get('datatype', None), row[value[1:]].get('xml:lang', None)
                            ).__repr__(), dataset
                        ))
                    elif dataset in [DATASET.LC2, DATASET.QALD]:
                        reground_map[replaceable_item].append(get_syntax_tree_value(
                            WikidataConstantForConstruction(
                                row[value[1:]]['type'], row[value[1:]]['value'],
                                row[value[1:]].get('datatype', None), row[value[1:]].get('xml:lang', None)
                            ).__repr__(), dataset
                        ))
                    else:
                        raise NotImplementedError(f"dataset: {dataset}")
                except Exception as e:
                    # logger.error(f"parsing error: {e}")
                    skip_flag = True
    return reground_map
#20260123 eswc修改
def new_quad_execute_test_suite(f1,f2, logger_path, dataset:DATASET, test_suite_limit=200, aflag = True):
    ### quad+grailqa
    if dataset == DATASET.GRAIL:
        detection_result = load_json(f1)
        output_path = f2
        logger = setup_custom_logger(logger_path)
    
        sparql_querier = SparqlOdbcQuerier(
            ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, timeout=60
        )
        method = DATASET.SIMULATED_FREEBASE
        src_data = load_json("/home5/whzhou/STAR-QC/data/metrics/new/input/grailqa_v1.0_dev_0_1000_linking_new_test_suite.json")
    
    ### quad+cwq
    if dataset == DATASET.CWQ:
        detection_result = load_json(f1)
        output_path = f2
        logger = setup_custom_logger(logger_path)
        
        sparql_querier = SparqlOdbcQuerier(
            ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, timeout=60
        )
        method = DATASET.SIMULATED_FREEBASE
        src_data = load_json("/home5/whzhou/STAR-QC/datadata/metrics/new/input/cwq_test_linking_1000_test_suite.json")

    ### quad+webqsp
    if dataset == DATASET.WEBQ:
        detection_result = load_json(f1)
        output_path = f2
        logger = setup_custom_logger(logger_path)
        
        sparql_querier = SparqlOdbcQuerier(
            ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, timeout=60
        )
        method = DATASET.SIMULATED_FREEBASE
        src_data = load_json("/home5/whzhou/STAR-QC/data/metrics/new/input/cwq_test_linking_1000_test_suite.json")

    ### quad+annotated
    # detection_result = load_json(f1)
    # output_path = f2
    # logger = setup_custom_logger("/home5/whzhou/STAR-QC/data/final_test/quad/test.log")
    
    # sparql_querier = SparqlOdbcQuerierWikidata(
    #     ODBC_CONFIG_WIKIDATA_2023, SPARQL_wrapper_path_wikidata_2023, logger, timeout=150
    # )
    # dataset = DATASET.LC2
    # method = DATASET.SIMULATED_WIKIDATA
    # src_data = load_json("data/final_test/annotated_dataset_test_suite.json")

    qid_to_src = {str(item["qid"]): item for item in src_data}

    qid_to_src = {'2103011014000':qid_to_src['2103011014000']}

    f1_list = list()
    accuracy_list = list()
    
    for detection_item in tqdm(detection_result.values()):
        if str(detection_item['qid']) not in qid_to_src:
            continue
        #会被kill
        if str(detection_item['qid'])=="5ac1c9a15542994ab5c67e1c":
            detection_item["test_suite_similarity"] = detection_item['F1']
            f1_list.append(detection_item['F1'])
            continue
        try:
            '''特殊情况: 没有 Simulated SPARQL, 各项指标都置为 0'''
            if not detection_item['search_results']:
                detection_item["test_suite_similarity"] = 0
                f1_list.append(0.0)
                continue

            src_item = qid_to_src[str(detection_item['qid'])]
            print(detection_item['qid'])
            if dataset in [DATASET.CWQ, DATASET.GRAIL, DATASET.WEBQ]:
                prefix = "PREFIX ns: <http://rdf.freebase.com/ns/> "            
            elif dataset in [DATASET.LC2,DATASET.SIMULATED_WIKIDATA]:
                prefix = WIKIDATA_PREFIX_LIST
                
            if aflag and "gpt_select_top1" in detection_item and "sparql" in detection_item['gpt_select_top1']:
                simulated_sparql_txt = prefix + "\n"+detection_item['gpt_select_top1']['sparql']
            else:
                simulated_sparql_txt = prefix + "\n"+detection_item["search_results"][0]["sparql"]

            editor = SyntaxTreeEditor(simulated_sparql_txt, method, logger)
            editor.preprocess_test_suite(f"?{TARGET_VARIABLE_NAME}")
            simulated_sparql_txt_before = editor.sparql_txt
            # 根据是否能replace reground item，判断是否为overlap item
            overlap_item_map = dict()
            replaceable_items = []

            if len(src_item['reground_map']) == 0:
                detection_item["test_suite_similarity"] = detection_item['F1']
                f1_list.append(detection_item['F1'])
                continue

            for golden_replace_items in tqdm(src_item['reground_map'].keys()):
                test_editor = SyntaxTreeEditor(simulated_sparql_txt_before, method, logger)
                _update_flag = False
                if test_editor.update_leaf_value(golden_replace_items, src_item['reground_map'][golden_replace_items][0]): # 返回值为 bool 类型，表明是否有节点被更新
                    _update_flag = True # 任一节点被更新即可
                    replaceable_items.append(golden_replace_items)
                if not _update_flag:
                    continue # 没有 overlap, 跳过该 test case  
        
            # 如果为空集，则为0
            if len(replaceable_items) == 0:
                detection_item["test_suite"] = 0
                f1_list.append(0.0)
                continue
            else:
                simulated_reground_map = update_reground_map(simulated_sparql_txt_before,replaceable_items,sparql_querier,dataset,logger)
            # detection_item['simulated_reground_map'] = simulated_reground_map
            # 求s1，
            s1_list = []
            overlap_item_intersection = dict()
            for k in simulated_reground_map.keys():
                simu_set = simulated_reground_map[k]
                golden_set = src_item['reground_map'][k]
                s1_list.append(jaccard_similarity(simu_set,golden_set))
                overlap_item_intersection[k] = set(simu_set).intersection(set(golden_set))
            
            detection_item["test_suite_s1"] = {
                # "s1_list":s1_list,
                "s1":sum(s1_list)/len(s1_list)
            }
            # 求组合
            key_list = list(overlap_item_intersection.keys())
            value_list = list()
            for key in key_list:
                value_list.append(overlap_item_intersection[key])
            
            golden_sparql_txt = src_item["golden_sparql_process"]
            # s2
            s2_list = []
            combination = list(itertools.product(*value_list))
            random.shuffle(combination)
            error_data = []
            for value_after_replace in combination[:test_suite_limit]:
                simu_editor = SyntaxTreeEditor(simulated_sparql_txt_before, method, logger)

                
                golden_editor = SyntaxTreeEditor(golden_sparql_txt, dataset, logger)
                assert len(key_list) == len(value_after_replace)

                for (key, value) in zip(key_list, value_after_replace):
                    golden_editor.update_leaf_value(key,value)
                    simu_editor.update_leaf_value(key,value)

                simu_answer = sparql_querier.get_execution_result_one_variable_sparql_wrapper(simu_editor.sparql_txt, retry=2)
                golden_answer = sparql_querier.get_execution_result_one_variable_sparql_wrapper(golden_editor.sparql_txt, retry=2)
                
                if len(golden_answer)==0 and len(simu_answer)==0:
                    s2 = 1
                else:
                    s2 = jaccard_similarity(simu_answer,golden_answer)

                # # debug
                # if s2 <1:
                #     error_data.append([value_after_replace,simu_editor.sparql_txt,golden_editor.sparql_txt])
                
                s2_list.append(s2)
            
            detection_item["test_suite_s2"] = {
                # "s2_list":s2_list,
                "s2":sum(s2_list)/len(s2_list),
                # 'error_data':error_data
            }

            f1_list.append(sum(s1_list)/len(s1_list)*sum(s2_list)/len(s2_list))
            detection_item['test_suite_similarity'] = sum(s1_list)/len(s1_list)*sum(s2_list)/len(s2_list)
            
            # 减点内存
            qid_to_src.pop(str(detection_item['qid']))

        except Exception as e:
            logger.error(f"qid: {detection_item['qid']}; error: {e}")
            detection_item["test_suite_similarity"] = 0
            f1_list.append(0.0)

    detection_result["summary"] = {
        "average_s": sum(f1_list) / len(src_data),
        "average_accuracy": sum(accuracy_list) / len(src_data)
    }
    dump_json(detection_result, output_path)
    assert len(f1_list)+1== len(detection_result), logger.info(f"f1_list: {len(f1_list)}; detection_result: {len(detection_result)}")



def new_quad_execute_test_suite_old(test_suite_limit=200):


    ### queryagent+grailqa
    # detection_result = load_json("data/metrics/new/queryagent/final_results.json")
    # output_path = "data/metrics/new/queryagent/grailqa_dev_0_1000_final_results_test_suite.json"
    # logger = setup_custom_logger("data/metrics/new/test_suite.log")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, timeout=60
    # )
    # dataset = DATASET.GRAIL
    # method = DATASET.QUERYAGENT
    # src_data = load_json("data/metrics/new/input/grailqa_v1.0_dev_0_1000_linking_new_test_suite.json")
    
    ### kbbinder+grailqa
    # detection_result = load_json("data/metrics/new/kbbinder/final_results_kbbinder.json")
    # output_path = "data/metrics/new/kbbinder/grailqa_dev_0_1000_final_results_test_suite.json"
    # logger = setup_custom_logger("data/metrics/new/test_suite.log")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, timeout=60
    # )
    # dataset = DATASET.GRAIL
    # method = DATASET.BINDER
    # src_data = load_json("data/metrics/new/input/grailqa_v1.0_dev_0_1000_linking_new_test_suite.json")
    
    ## qgg+cwq
    detection_result = load_json("data/metrics/new/qgg/qgg_a_final_results.json")
    output_path = "data/metrics/new/qgg/a_CWQ_final_results_test_suite.json"
    logger = setup_custom_logger("data/metrics/new/qgg/test_suite.log")
    sparql_querier = SparqlOdbcQuerier(
        ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    )
    dataset = DATASET.CWQ
    method = DATASET.QGG
    src_data = load_json("data/metrics/new/input/cwq_test_linking_1000_test_suite.json")

    ### quad+annotated
    # detection_result = load_json("/home5/yhbao/SPARQLannotation/output/wd_annotation/WD_annotation_linked_not_use_neighbor_entity_dec03_gpt_rerank.json")
    # output_path = "/home5/whzhou/STAR-QC/data/final_test/quad/quad_linked_test_suite_f1_new.json"
    # logger = setup_custom_logger("/home5/whzhou/STAR-QC/data/final_test/quad/test.log")
    
    # sparql_querier = SparqlOdbcQuerierWikidata(
    #     ODBC_CONFIG_WIKIDATA_2023, SPARQL_wrapper_path_wikidata_2023, logger, timeout=300
    # )
    # dataset = DATASET.LC2
    # method = DATASET.SIMULATED_WIKIDATA
    # src_data = load_json("/home5/whzhou/STAR-QC/data/annotated_test/annotated_dataset_test_suite.json")

    qid_to_src = {str(item["qid"]): item for item in src_data}

    f1_list = list()
    accuracy_list = list()
    
    for detection_item in tqdm(detection_result['results']):
        if str(detection_item['qid']) not in qid_to_src:
            continue
        try:
            '''特殊情况: 没有 Simulated SPARQL, 各项指标都置为 0'''
            if ("searched_queries" not in detection_item) or len(detection_item["searched_queries"]) == 0:
                detection_item["test_suite"] = 0
                f1_list.append(0.0)
                continue

            src_item = qid_to_src[str(detection_item['qid'])]

            if method in [DATASET.QGG, DATASET.BINDER, DATASET.LSQ]:
                simulated_sparql_txt = detection_item["searched_queries"][0]["sparql"]
            elif method == DATASET.QUERYAGENT:
                simulated_sparql_txt = "PREFIX : <http://rdf.freebase.com/ns/>\n"+detection_item["searched_queries"][0]["sparql"]
            else:
                simulated_sparql_txt = sexp_to_sparql_for_test_suite(detection_item["searched_queries"][0]["s_expression"])

            editor = SyntaxTreeEditor(simulated_sparql_txt, method, logger)
            editor.preprocess_test_suite(f"?{TARGET_VARIABLE_NAME}")
            simulated_sparql_txt_before = editor.sparql_txt
            # 根据是否能replace reground item，判断是否为overlap item
            overlap_item_map = dict()
            replaceable_items = []
            for golden_replace_items in tqdm(src_item['reground_map'].keys()):
                test_editor = SyntaxTreeEditor(simulated_sparql_txt_before, method, logger)
                _update_flag = False
                if test_editor.update_leaf_value(golden_replace_items, src_item['reground_map'][golden_replace_items][0]): # 返回值为 bool 类型，表明是否有节点被更新
                    _update_flag = True # 任一节点被更新即可
                    replaceable_items.append(golden_replace_items)
                if not _update_flag:
                    continue # 没有 overlap, 跳过该 test case  
        
            # 如果为空集，则为0
            if len(replaceable_items) == 0:
                detection_item["test_suite"] = 0
                f1_list.append(0.0)
                continue
            else:
                simulated_reground_map = update_reground_map(simulated_sparql_txt_before,replaceable_items,sparql_querier,dataset,logger)
            # detection_item['simulated_reground_map'] = simulated_reground_map
            # 求s1，
            s1_list = []
            overlap_item_intersection = dict()
            for k in simulated_reground_map.keys():
                simu_set = simulated_reground_map[k]
                golden_set = src_item['reground_map'][k]
                s1_list.append(jaccard_similarity(simu_set,golden_set))
                overlap_item_intersection[k] = set(simu_set).intersection(set(golden_set))
            
            detection_item["test_suite_s1"] = {
                # "s1_list":s1_list,
                "s1":sum(s1_list)/len(s1_list)
            }
            # 求组合
            key_list = list(overlap_item_intersection.keys())
            value_list = list()
            for key in key_list:
                value_list.append(overlap_item_intersection[key])
            
            golden_sparql_txt = src_item["golden_sparql_process"]
            # s2
            s2_list = []
            combination = list(itertools.product(*value_list))
            random.shuffle(combination)
            for value_after_replace in combination[:test_suite_limit]:
                simu_editor = SyntaxTreeEditor(simulated_sparql_txt_before, method, logger)

                
                golden_editor = SyntaxTreeEditor(golden_sparql_txt, dataset, logger)
                assert len(key_list) == len(value_after_replace)

                for (key, value) in zip(key_list, value_after_replace):
                    golden_editor.update_leaf_value(key,value)
                    simu_editor.update_leaf_value(key,value)

                simu_answer = sparql_querier.get_execution_result_one_variable_sparql_wrapper(simu_editor.sparql_txt, retry=2)
                golden_answer = sparql_querier.get_execution_result_one_variable_sparql_wrapper(golden_editor.sparql_txt, retry=2)
                s2 = jaccard_similarity(simu_answer,golden_answer)

                if len(simu_answer) ==0 and len(golden_answer) == 0:
                    s2_list.append(1.0)
                else:
                    s2_list.append(s2)
            
            detection_item["test_suite_s2"] = {
                # "s2_list":s2_list,
                "s2":sum(s2_list)/len(s2_list)
            }
            f1_list.append(sum(s1_list)/len(s1_list)*sum(s2_list)/len(s2_list))
            detection_item['test_suite_similarity'] = sum(s1_list)/len(s1_list)*sum(s2_list)/len(s2_list)
        
        except Exception as e:
            logger.error(f"qid: {detection_item['qid']}; error: {e}")
            detection_item["test_suite_similarity"] = 0
            f1_list.append(0.0)

    detection_result["summary"] = {
        "average_s": sum(f1_list) / len(src_data),
        "average_accuracy": sum(accuracy_list) / len(src_data)
    }
    dump_json(detection_result, output_path)
    assert len(f1_list)== len(detection_result['results']), logger.info(f"f1_list: {len(f1_list)}; detection_result: {len(detection_result)}")


def new_generate_test_suite(example, dataset:DATASET, sparql_querier:Union[SparqlOdbcQuerier, SparqlOdbcQuerierWikidata], logger):
    original_sparql_txt = example["golden_sparql_query"]
    replaceable_items = [
        item['mid'] for item in example["golden_grounded_items"]
    ]
    # 各数据集上，item 格式的特殊处理
    if dataset is DATASET.GRAIL:
        for (idx, item) in enumerate(replaceable_items):
            if item.endswith('gYear>') or item.endswith('gYearMonth>') or item.endswith('date>'):
                replaceable_items[idx] = f'"{item.split("^^")[0][1:-1] + "-08:00"}"^^{item.split("^^")[1]}'
    editor = SyntaxTreeEditor(original_sparql_txt, dataset, logger)
    editor.preprocess_test_suite(f"?{TARGET_VARIABLE_NAME}")
    sparql_txt_before_replace = editor.sparql_txt
    # 得到每个repalceable item的r_i
    reground_map = update_reground_map(sparql_txt_before_replace,replaceable_items,sparql_querier,dataset,logger)
    return reground_map,sparql_txt_before_replace

def new_add_test_suite():
    ### GRAILQA
    # input_path = "data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_linking.json"
    # input_data = load_json(input_path)
    # logger = setup_custom_logger("data/metrics/new/log/test_suite.log")
    # logger.info(f"input_path: {input_path}; TEST_SUITE_LIMIT: {TEST_SUITE_LIMIT}")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, timeout=60
    # )
    # dataset = DATASET.GRAIL
    # output_path = "data/metrics/new/input/grailqa_v1.0_dev_0_1000_linking_new_test_suite.json"

    ### CWQ
    # input_path = "data/input/cwq/cwq_test_0_1000_linking.json"
    # input_data = load_json(input_path)
    # logger = setup_custom_logger("data/input/cwq/test_suite.log")
    # logger.info(f"input_path: {input_path}; TEST_SUITE_LIMIT: {TEST_SUITE_LIMIT}")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    # )
    # dataset = DATASET.CWQ
    # output_path = "data/metrics/new/input/cwq_test_linking_1000_test_suite.json"

    ### WEBQSP
    
    ### ANNOTATED
    input_path = "data/annotated_test/annotated_dataset.json"
    input_data = load_json(input_path)
    logger = setup_custom_logger("data/final_test/test_suite.log")
    logger.info(f"input_path: {input_path}; TEST_SUITE_LIMIT: {TEST_SUITE_LIMIT}")
    sparql_querier = SparqlOdbcQuerierWikidata(
        ODBC_CONFIG_WIKIDATA_2023, SPARQL_wrapper_path_wikidata_2023, logger, timeout=300
    )
    dataset = DATASET.LC2
    output_path = "data/final_test/annotated_dataset_test_suite.json"


    random.seed(374)
    for example_item in tqdm(input_data):
        if dataset == DATASET.LC2:
            prefix = """
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            PREFIX ps: <http://www.wikidata.org/prop/statement/>
            PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
            PREFIX p: <http://www.wikidata.org/prop/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            """
            example_item['golden_sparql_query'] = prefix+" "+example_item['golden_sparql']
        try:
            test_suite_examples,sparql_txt_before_replace = new_generate_test_suite(example_item, dataset, sparql_querier, logger)
            example_item['reground_map'] = test_suite_examples
            example_item['golden_sparql_process'] = sparql_txt_before_replace
        except Exception as e:
            logger.error(f"qid: {example_item['qid']}; error: {e}")
            example_item['reground_map'] = dict()
    dump_json(input_data, output_path)

    
    
    

def _generate_test_suite(example, dataset:DATASET, sparql_querier:Union[SparqlOdbcQuerier, SparqlOdbcQuerierWikidata], logger):
    original_sparql_txt = example["golden_sparql_query"]
    replaceable_items = [
        item['mid'] for item in example["golden_grounded_items"]
    ]
    # 各数据集上，item 格式的特殊处理
    if dataset is DATASET.GRAIL:
        for (idx, item) in enumerate(replaceable_items):
            if item.endswith('gYear>') or item.endswith('gYearMonth>') or item.endswith('date>'):
                replaceable_items[idx] = f'"{item.split("^^")[0][1:-1] + "-08:00"}"^^{item.split("^^")[1]}'
    editor = SyntaxTreeEditor(original_sparql_txt, dataset, logger)
    editor.preprocess_test_suite(f"?{TARGET_VARIABLE_NAME}")
    sparql_txt_before_replace = editor.sparql_txt

    test_cases_colletion = list()
    test_cases_colletion.append({
        "original_sparql": sparql_txt_before_replace,
        "answer": [item['mid'] for item in example["answer"]],
        "binding": {
            get_syntax_tree_string(item, dataset): get_syntax_tree_value(item, dataset)
            for item in replaceable_items
        }
    })

    replace_map = dict() # key: 语法树中 item 调用 to_str() 得到的字符串；value: 语法树中 item 的 value 属性
    for (idx, item) in enumerate(replaceable_items):
        if dataset in [DATASET.CWQ, DATASET.WEBQ, DATASET.GRAIL]:
            if FreebaseConstantForConstruction.get_constant_type(item) is FREEBASE_CONSTANT_TYPE.QUANTITY:
                old_value = item.split('^^')[0][1:-1]
                suffix = item.split('^^')[1]
                if suffix is "<http://www.w3.org/2001/XMLSchema#integer>":
                    replace_map[get_syntax_tree_string(item, dataset)] = [
                        f'"{old_value}"',
                        f'"{str(int(old_value) * 2)}"',
                        f'"{str(int(old_value) * 3)}"',
                    ]
                else:
                    replace_map[get_syntax_tree_string(item, dataset)] = [
                        f'"{old_value}"',
                        f'"{str(float(old_value) * 2.0)}"',
                        f'"{str(float(old_value) * 3.0)}"',
                    ]
            elif FreebaseConstantForConstruction.get_constant_type(item) is FREEBASE_CONSTANT_TYPE.TIME:
                # 日期就没法改了
                replace_map[get_syntax_tree_string(item, dataset)] = [get_syntax_tree_value(item, dataset)]
            else: # 其他: 替换为变量
                _tree_string = get_syntax_tree_string(item, dataset)
                variable = f"?{REPLACED_VARIABLE_NAME}{idx}" # 在语法树中，变量的 value 是带有 ? 号的
                replace_map[_tree_string] = [variable]
        elif dataset in [DATASET.LC2, DATASET.QALD]:
            if WikidataConstantForConstruction.get_constant_type(item) is WIKIDATA_CONSTANT_TYPE.QUANTITY:
                old_value = item.split('^^')[0][1:-1]
                suffix = item.split('^^')[1]
                if suffix is "<http://www.w3.org/2001/XMLSchema#integer>":
                    replace_map[get_syntax_tree_string(item, dataset)] = [
                        f'"{old_value}"',
                        f'"{str(int(old_value) * 2)}"',
                        f'"{str(int(old_value) * 3)}"',
                    ]
                else:
                    replace_map[get_syntax_tree_string(item, dataset)] = [
                        f'"{old_value}"',
                        f'"{str(float(old_value) * 2.0)}"',
                        f'"{str(float(old_value) * 3.0)}"',
                    ]
            elif FreebaseConstantForConstruction.get_constant_type(item) is WIKIDATA_CONSTANT_TYPE.TIME:
                # 日期就没法改了
                replace_map[get_syntax_tree_string(item, dataset)] = [get_syntax_tree_value(item, dataset)]
            else: # 其他: 替换为变量
                _tree_string = get_syntax_tree_string(item, dataset)
                variable = f"?{REPLACED_VARIABLE_NAME}{idx}" # 在语法树中，变量的 value 是带有 ? 号的
                replace_map[_tree_string] = [variable]
        else:
            raise NotImplementedError(f"dataset: {dataset}")
    
    key_list = list(replace_map.keys())
    value_list = list()
    for key in key_list:
        value_list.append(replace_map[key])

    for value_after_replace in itertools.product(*value_list):
        editor = SyntaxTreeEditor(sparql_txt_before_replace, dataset, logger)
        editor.remove_projection_variable(f"?{TARGET_VARIABLE_NAME}")
        assert len(key_list) == len(value_after_replace)

        for (key, value) in zip(key_list, value_after_replace):
            """
            key: 语法树中 item 调用 to_str() 得到的字符串
            value: 语法树中 item 的 value 属性
            """
            if value.startswith('?'): # 变量
                flag = editor.replace_leaf_with_variable(key, value)
                if not flag:
                    raise Exception(f"key: {key} not found in the syntax tree")
            else: # TIME 或者 QUANTITY
                flag = editor.update_leaf_value(key, value)
                if not flag:
                    raise Exception(f"key: {key} not found in the syntax tree")
        # 移除答案变量，减少返回数据的量，我们只要求答案非空即可，这一步不需要获得其值
        
        sparql_txt_after_replace = editor.sparql_txt
        rows = sparql_querier.execute_query(sparql_txt_after_replace, retry=3)
        """
        _test_suite_examples, 其中的每个 item 仍然是一个 dict()
        key: 语法树中 item 调用 to_str() 得到的字符串
        value: 语法树中 item 的 value 属性
        """
        _test_suite_examples = list()
        for row in rows:
            row_result = dict()
            skip_flag = False
            for (key, value) in zip(key_list, value_after_replace):
                if value.startswith('?'): # 变量
                    try:
                        if dataset in [DATASET.GRAIL, DATASET.CWQ, DATASET.WEBQ]:
                            row_result[key] = get_syntax_tree_value(
                                FreebaseConstantForConstruction(
                                    row[value[1:]]['type'], row[value[1:]]['value'],
                                    row[value[1:]].get('datatype', None), row[value[1:]].get('xml:lang', None)
                                ).__repr__(), dataset
                            )
                        elif dataset in [DATASET.LC2, DATASET.QALD]:
                            row_result[key] = get_syntax_tree_value(
                                WikidataConstantForConstruction(
                                    row[value[1:]]['type'], row[value[1:]]['value'],
                                    row[value[1:]].get('datatype', None), row[value[1:]].get('xml:lang', None)
                                ).__repr__(), dataset
                            )
                        else:
                            raise NotImplementedError(f"dataset: {dataset}")
                    except Exception as e:
                        # logger.error(f"parsing error: {e}")
                        skip_flag = True
                else:
                    row_result[key] = value
            if not skip_flag:
                _test_suite_examples.append(row_result)
        
        test_suite_item ={
            "original_sparql": sparql_txt_before_replace,
            "bindings": _test_suite_examples # 替换表
        }

        test_cases = _generate_test_cases(test_suite_item, dataset, sparql_querier, logger, TEST_SUITE_LIMIT)
        # 替换之后的 test_cases, 必然存在一个和替换前一样的测试用例
        test_cases_colletion.extend(test_cases)

    return test_cases_colletion

def _generate_test_suite_webqsp(example, dataset:DATASET, sparql_querier:Union[SparqlOdbcQuerier, SparqlOdbcQuerierWikidata], logger):
    """
    WebQSP: 对于每个 parse, 都生成 Test Suite
    """
    assert dataset is DATASET.WEBQ
    original_sparal_list = example["golden_sparql_list"]
    test_cases_list = list()
    
    for original_sparql in original_sparal_list:
        original_sparql_txt = original_sparql["sparql"]
        replaceable_items = [
            item['mid'] for item in original_sparql["golden_grounded_items"]
        ]
        editor = SyntaxTreeEditor(original_sparql_txt, dataset, logger)
        editor.preprocess_test_suite(f"?{TARGET_VARIABLE_NAME}")
        sparql_txt_before_replace = editor.sparql_txt

        test_cases_colletion = list()
        test_cases_colletion.append({
            "original_sparql": sparql_txt_before_replace,
            "answer": list(sparql_querier.get_execution_result_one_variable_sparql_wrapper(sparql_txt_before_replace)),
            "binding": {
                get_syntax_tree_string(item, dataset): get_syntax_tree_value(item, dataset)
                for item in replaceable_items
            }
        })

        replace_map = dict() # key: 语法树中 item 调用 to_str() 得到的字符串；value: 语法树中 item 的 value 属性
        for (idx, item) in enumerate(replaceable_items): 
            if FreebaseConstantForConstruction.get_constant_type(item) is FREEBASE_CONSTANT_TYPE.QUANTITY:
                old_value = item.split('^^')[0][1:-1]
                suffix = item.split('^^')[1]
                if suffix is "<http://www.w3.org/2001/XMLSchema#integer>":
                    replace_map[get_syntax_tree_string(item, dataset)] = [
                        f'"{old_value}"',
                        f'"{str(int(old_value) * 2)}"',
                        f'"{str(int(old_value) * 3)}"',
                    ]
                else:
                    replace_map[get_syntax_tree_string(item, dataset)] = [
                        f'"{old_value}"',
                        f'"{str(float(old_value) * 2.0)}"',
                        f'"{str(float(old_value) * 3.0)}"',
                    ]
            elif FreebaseConstantForConstruction.get_constant_type(item) is FREEBASE_CONSTANT_TYPE.TIME:
                # 日期就没法改了
                replace_map[get_syntax_tree_string(item, dataset)] = [get_syntax_tree_value(item, dataset)]
            else: # 其他: 替换为变量
                _tree_string = get_syntax_tree_string(item, dataset)
                variable = f"?{REPLACED_VARIABLE_NAME}{idx}" # 在语法树中，变量的 value 是带有 ? 号的
                replace_map[_tree_string] = [variable]
        
        key_list = list(replace_map.keys())
        value_list = list()
        for key in key_list:
            value_list.append(replace_map[key])

        for value_after_replace in itertools.product(*value_list):
            editor = SyntaxTreeEditor(sparql_txt_before_replace, dataset, logger)
            editor.remove_projection_variable(f"?{TARGET_VARIABLE_NAME}")
            assert len(key_list) == len(value_after_replace)

            for (key, value) in zip(key_list, value_after_replace):
                """
                key: 语法树中 item 调用 to_str() 得到的字符串
                value: 语法树中 item 的 value 属性
                """
                if value.startswith('?'): # 变量
                    flag = editor.replace_leaf_with_variable(key, value)
                    if not flag:
                        raise Exception(f"key: {key} not found in the syntax tree")
                else: # TIME 或者 QUANTITY
                    flag = editor.update_leaf_value(key, value)
                    if not flag:
                        raise Exception(f"key: {key} not found in the syntax tree")
            # 移除答案变量，减少返回数据的量，我们只要求答案非空即可，这一步不需要获得其值
            
            sparql_txt_after_replace = editor.sparql_txt
            rows = sparql_querier.execute_query(sparql_txt_after_replace, retry=2)
            """
            _test_suite_examples, 其中的每个 item 仍然是一个 dict()
            key: 语法树中 item 调用 to_str() 得到的字符串
            value: 语法树中 item 的 value 属性
            """
            _test_suite_examples = list()
            for row in rows:
                row_result = dict()
                skip_flag = False
                for (key, value) in zip(key_list, value_after_replace):
                    if value.startswith('?'): # 变量
                        try:
                            row_result[key] = get_syntax_tree_value(
                                FreebaseConstantForConstruction(
                                    row[value[1:]]['type'], row[value[1:]]['value'],
                                    row[value[1:]].get('datatype', None), row[value[1:]].get('xml:lang', None)
                                ).__repr__(), dataset
                            )
                        except Exception as e:
                            # logger.error(f"parsing error: {e}")
                            skip_flag = True
                    else:
                        row_result[key] = value
                if not skip_flag:
                    _test_suite_examples.append(row_result)
            
            test_suite_item ={
                "original_sparql": sparql_txt_before_replace,
                "bindings": _test_suite_examples # 替换表
            }

            test_cases = _generate_test_cases(test_suite_item, dataset, sparql_querier, logger, TEST_SUITE_LIMIT)
            # 替换之后的 test_cases, 必然存在一个和替换前一样的测试用例
            test_cases_colletion.extend(test_cases)

        test_cases_list.append(test_cases_colletion)
    return test_cases_list 


def _generate_test_cases(test_suite_item, dataset:DATASET, sparql_querier:SparqlOdbcQuerier, logger, test_cases_limit=50):
    """
    @param test_suite_item: {"original_sparql", "bindings"}
    """
    test_cases = list()
    bindings = copy.deepcopy(test_suite_item["bindings"])
    random.shuffle(bindings)
    for _binding in tqdm(bindings, desc="Enumerating bindings"):
        if len(test_cases) >= test_cases_limit:
            break
        editor = SyntaxTreeEditor(test_suite_item["original_sparql"], dataset, logger)
        """
        _binding: 该结构中，item 都不是变量了
        key: 语法树中 item 调用 to_str() 得到的字符串
        value: 语法树中 item 的 value 属性
        """
        for (old_str, new_v) in _binding.items():
            editor.update_leaf_value(old_str, new_v)
        sparql_txt_after_replace = editor.sparql_txt
        execution_result = sparql_querier.get_execution_result_one_variable_sparql_wrapper(sparql_txt_after_replace, retry=2)
        if not execution_result:
            logger.error(f"SPARQL execution failed: {sparql_txt_after_replace}")
        else:
            test_cases.append({
                "original_sparql": test_suite_item["original_sparql"],
                "binding": _binding,
                "answer": list(execution_result)
            })
    return test_cases


def add_test_suite():
    '''CWQ'''
    input_path = "data/input/cwq/cwq_dev_linking_1000.json"
    input_data = load_json(input_path)
    logger = setup_custom_logger("data/input/cwq/test_suite.log")
    logger.info(f"input_path: {input_path}; TEST_SUITE_LIMIT: {TEST_SUITE_LIMIT}")
    sparql_querier = SparqlOdbcQuerier(
        ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    )
    dataset = DATASET.CWQ
    output_path = "data/input/cwq/ccwq_dev_linking_1000_test_suite.json"

    '''LC2'''
    # input_path = "data/input/LC_QuAD_2/lc2_test_0_1000.json"
    # input_data = load_json(input_path)
    # logger = setup_custom_logger("data/input/LC_QuAD_2/test_suite.log")
    # logger.info(f"input_path: {input_path}; TEST_SUITE_LIMIT: {TEST_SUITE_LIMIT}")
    # sparql_querier = SparqlOdbcQuerierWikidata(
    #     ODBC_CONFIG_WIKIDATA_2019, SPARQL_wrapper_path_wikidata_2019, logger, timeout=60
    # )
    # dataset = DATASET.LC2
    # output_path = "data/input/LC_QuAD_2/lc2_test_0_1000_test_suite.json"

    '''QALD'''
    # input_path = "data/input/QALD/qald_test.json"
    # input_data = load_json(input_path)
    # logger = setup_custom_logger("data/input/QALD/test_suite.log")
    # logger.info(f"input_path: {input_path}; TEST_SUITE_LIMIT: {TEST_SUITE_LIMIT}")
    # sparql_querier = SparqlOdbcQuerierWikidata(
    #     ODBC_CONFIG_WIKIDATA_2019, SPARQL_wrapper_path_wikidata_2019, logger, timeout=60
    # )
    # dataset = DATASET.QALD
    # output_path = "data/input/QALD/qald_test_test_suite.json"

    # '''GrailQA'''
    # input_path = "data/output/g2/grailqa_v1.0_train_linking_0_10000.json"
    # # input_path = "data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_linking.json"
    # input_data = load_json(input_path)
    # logger = setup_custom_logger("data/output/g2/test_suite.log")
    # logger.info(f"input_path: {input_path}; TEST_SUITE_LIMIT: {TEST_SUITE_LIMIT}")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, timeout=60
    # )
    # dataset = DATASET.GRAIL
    # output_path = "data/output/g2/grailqa_v1.0_train_linking_0_10000_test_suite.json"

    '''annotated test'''
    # input_path = "data/annotated_test/annotated_dataset.json"
    # # input_path = "data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_linking.json"
    # input_data = load_json(input_path)
    # logger = setup_custom_logger("data/output/g2/test_suite.log")
    # logger.info(f"input_path: {input_path}; TEST_SUITE_LIMIT: {TEST_SUITE_LIMIT}")
    # sparql_querier = SparqlOdbcQuerierWikidata(
    #     ODBC_CONFIG_WIKIDATA_2023, SPARQL_wrapper_path_wikidata_2023, logger, timeout=300
    # )
    # dataset = DATASET.LC2
    # output_path = "data/annotated_test/annotated_dataset_test_suite.json"

    random.seed(374)
    for example_item in tqdm(input_data):
        # prefix = """
        # PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        # PREFIX ps: <http://www.wikidata.org/prop/statement/>
        # PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
        # PREFIX p: <http://www.wikidata.org/prop/>
        # PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        # PREFIX wd: <http://www.wikidata.org/entity/>
        # """
        # example_item['golden_sparql_query'] = prefix+" "+example_item['golden_sparql']
        try:
            test_suite_examples = _generate_test_suite(example_item, dataset, sparql_querier, logger)
            example_item['test_suite_examples'] = test_suite_examples
        except Exception as e:
            logger.error(f"qid: {example_item['qid']}; error: {e}")
            example_item['test_suite_examples'] = list()
    dump_json(input_data, output_path)

def add_test_suite_webqsp():
    # 特别点: 有多个 parse, 需要对于每个 parse 都生成 Test Suite
    '''WebQSP'''
    input_path = "data/input/WebQSP/webqsp_test_0_1000_linking.json"
    input_data = load_json(input_path)
    logger = setup_custom_logger("data/input/WebQSP/test_suite.log")
    logger.info(f"input_path: {input_path}; TEST_SUITE_LIMIT: {TEST_SUITE_LIMIT}")
    sparql_querier = SparqlOdbcQuerier(
        ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    )
    dataset = DATASET.WEBQ
    output_path = "data/input/WebQSP/webqsp_test_0_1000_test_suite.json"

    random.seed(374)
    for example_item in tqdm(input_data):
        try:
            test_suite_example_list = _generate_test_suite_webqsp(example_item, dataset, sparql_querier, logger)
            example_item['test_suite_examples'] = test_suite_example_list
        except Exception as e:
            logger.error(f"qid: {example_item['qid']}; error: {e}")
            example_item['test_suite_examples'] = list()
    dump_json(input_data, output_path)

def debug_add_test_suite_freebase():
    '''WebQSP'''
    input_path = "data/input/WebQSP/webqsp_test_0_1000_linking.json"
    input_data = load_json(input_path)
    logger = setup_custom_logger("data/input/WebQSP/test_suite.log")
    logger.info(f"input_path: {input_path}; TEST_SUITE_LIMIT: {TEST_SUITE_LIMIT}")
    sparql_querier = SparqlOdbcQuerier(
        ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    )
    dataset = DATASET.WEBQ
    
    for example_item in tqdm(input_data[622:623]):
        try:
            test_suite_examples = _generate_test_suite(example_item, dataset, sparql_querier, logger)
            example_item['test_suite_examples'] = test_suite_examples
        except Exception as e:
            logger.error(f"qid: {example_item['qid']}; error: {e}")
            example_item['test_suite_examples'] = list()
        logger.info(test_suite_examples)

def quad_execute_test_suite():
    '''CWQ + Simulated Query'''
    # detection_result = load_json("data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_comparison_2024-06-04_21:14:58/final_results.json")
    # output_path = "data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_comparison_2024-06-04_21:14:58/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_comparison_2024-06-04_21:14:58/test_suite.log")

    # detection_result = load_json("data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_baseline_comparison_2024-06-05_15:54:58/final_results.json")
    # output_path = "data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_baseline_comparison_2024-06-05_15:54:58/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_baseline_comparison_2024-06-05_15:54:58/test_suite.log")

    # detection_result = load_json("data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_linking_comparison_2024-06-05_20:34:05/final_results.json")
    # output_path = "data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_linking_comparison_2024-06-05_20:34:05/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_linking_comparison_2024-06-05_20:34:05/test_suite.log")

    # detection_result = load_json("data/output/cwq_test_0_1000_starQC_qggLinking_comparison_2024-06-12_22:11:49/final_results.json")
    # output_path = "data/output/cwq_test_0_1000_starQC_qggLinking_comparison_2024-06-12_22:11:49/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/cwq_test_0_1000_starQC_qggLinking_comparison_2024-06-12_22:11:49/test_suite.log")

    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    # )
    # dataset = DATASET.SIMULATED_FREEBASE
    # src_data = load_json("data/input/cwq/cwq_test_0_1000_test_suite.json")

    '''CWQ + QGG-original'''
    # detection_result = load_json("data/output/qgg_original/cwq_test_0_1000/final_results.json")
    # output_path = "data/output/qgg_original/cwq_test_0_1000/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/qgg_original/cwq_test_0_1000/test_suite.log")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    # )
    # dataset = DATASET.QGG
    # src_data = load_json("data/input/cwq/cwq_test_0_1000_test_suite.json")

    '''CWQ + QueryAgent'''
    detection_result = load_json("data/metrics/queryagent/cwq_0_1000.json")
    output_path = "data/metrics/queryagent/cwq_0_1000_test_suite.json"
    logger = setup_custom_logger("data/metrics/queryagent/test_suite.log")
    sparql_querier = SparqlOdbcQuerier(
        ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    )
    dataset = DATASET.QUERYAGENT
    src_data = load_json("data/input/cwq/cwq_test_0_1000_test_suite.json")

    '''CWQ + QGG-QC'''
    # detection_result = load_json("/home5/whzhou/STAR-QC/data/qgg/cwq_final_results.json")
    # output_path = "/home5/whzhou/STAR-QC/data/qgg/final_results_test_suite.json"
    # logger = setup_custom_logger("/home5/whzhou/STAR-QC/data/qgg/test_suite.log")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    # )
    # dataset = DATASET.QGG
    # src_data = load_json("data/input/cwq/cwq_test_0_1000_test_suite.json")

    # '''CWQ + LSQ-QC'''
    # detection_result = load_json("data/output/lsq/cwq/final_results.json")
    # output_path = "data/output/lsq/cwq/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/lsq/cwq/test_suite.log")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    # )
    # dataset = DATASET.LSQ
    # src_data = load_json("data/input/cwq/cwq_test_0_1000_test_suite.json")

    '''GrailQA + Simulated'''
    # detection_result = load_json("data/output/query_construction/grailqa/for_comparison/grailqa_dev_0_1000_starQC_comparison_2024-06-06_12:27:17/final_results.json")
    # output_path = "data/output/query_construction/grailqa/for_comparison/grailqa_dev_0_1000_starQC_comparison_2024-06-06_12:27:17/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/query_construction/grailqa/for_comparison/grailqa_dev_0_1000_starQC_comparison_2024-06-06_12:27:17/test_suite.log")

    # detection_result = load_json("data/output/query_construction/grailqa/for_comparison/grailqa_dev_baseline_comparison_2024-06-05_20:32:09/final_results.json")
    # output_path = "data/output/query_construction/grailqa/for_comparison/grailqa_dev_baseline_comparison_2024-06-05_20:32:09/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/query_construction/grailqa/for_comparison/grailqa_dev_baseline_comparison_2024-06-05_20:32:09/test_suite.log")

    # detection_result = load_json("data/output/query_construction/grailqa/for_comparison/grailqa_dev_0_1000_starQC_linking_comparison_2024-06-09_22:47:18/final_results.json")
    # output_path = "data/output/query_construction/grailqa/for_comparison/grailqa_dev_0_1000_starQC_linking_comparison_2024-06-09_22:47:18/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/query_construction/grailqa/for_comparison/grailqa_dev_0_1000_starQC_linking_comparison_2024-06-09_22:47:18/test_suite.log")

    # detection_result = load_json("data/output/grailqa_dev_0_1000_starQC_queryagentLinking_comparison_2024-06-12_22:12:31/final_results.json")
    # output_path = "data/output/grailqa_dev_0_1000_starQC_queryagentLinking_comparison_2024-06-12_22:12:31/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/grailqa_dev_0_1000_starQC_queryagentLinking_comparison_2024-06-12_22:12:31/test_suite.log")
    
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, timeout=60
    # )
    # dataset = DATASET.SIMULATED_FREEBASE
    # src_data = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_test_suite.json")

    '''GrailQA + QueryAgent'''
    # detection_result = load_json("data/output/queryagent/grailqa_dev_0_1000/final_results.json")
    # output_path = "data/output/queryagent/grailqa_dev_0_1000/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/queryagent/grailqa_dev_0_1000/test_suite.log")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, timeout=60
    # )
    # dataset = DATASET.QUERYAGENT
    # src_data = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_test_suite.json")

    '''GrailQA + KB-Binder'''
    # detection_result = load_json("data/output/kb_binder/grailqa_dev_0_1000/final_results.json")
    # output_path = "data/output/kb_binder/grailqa_dev_0_1000/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/kb_binder/grailqa_dev_0_1000/test_suite.log")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, timeout=60
    # )
    # dataset = DATASET.BINDER
    # src_data = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_test_suite.json")

    '''GrailQA + QGG-QC'''
    # detection_result = load_json("data/output/qgg_qc/grailqa/final_results.json")
    # output_path = "data/output/qgg_qc/grailqa/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/qgg_qc/grailqa/test_suite.log")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, timeout=60
    # )
    # dataset = DATASET.QGG
    # src_data = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_test_suite.json")

    '''GrailQA + LSQ-QC'''
    # detection_result = load_json("data/output/lsq/grailqa/final_results.json")
    # output_path = "data/output/lsq/grailqa/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/lsq/grailqa/test_suite.log")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, timeout=60
    # )
    # dataset = DATASET.LSQ
    # src_data = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_test_suite.json")
    
    qid_to_src = {item["qid"]: item for item in src_data}

    f1_list = list()
    accuracy_list = list()
    
    for detection_item in tqdm(detection_result['results']):
        try:
            '''特殊情况: 没有 Simulated SPARQL, 各项指标都置为 0'''
            if ("searched_queries" not in detection_item) or len(detection_item["searched_queries"]) == 0:
                detection_item["test_suite"] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "accuracy": 0.0,
                }
                f1_list.append(0.0)
                accuracy_list.append(0.0)
                continue
            '''
            特殊情况: Test Suite 为空，那么把所有指标都置为 1 --> 对于所有方法都这样
            '''
            src_item = qid_to_src[detection_item['qid']]
            if len(src_item["test_suite_examples"]) == 0:
                detection_item["test_suite"] = {
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "accuracy": 1.0,
                }
                f1_list.append(1.0)
                accuracy_list.append(1.0)
                continue
            
            if dataset in [DATASET.QGG, DATASET.BINDER, DATASET.QUERYAGENT, DATASET.LSQ]:
                simulated_sparql_txt = detection_item["searched_queries"][0]["sparql"]
            else:
                simulated_sparql_txt = sexp_to_sparql_for_test_suite(detection_item["searched_queries"][0]["s_expression"])
            editor = SyntaxTreeEditor(simulated_sparql_txt, dataset, logger)
            editor.preprocess_test_suite(f"?{TARGET_VARIABLE_NAME}")
            simulated_sparql_txt_before = editor.sparql_txt

            _prec_list, _recall_list, _f1_list, _accuracy_list = list(), list(), list(), list()
            '''
            只统计 Simulated SPARQL 在存在 overlap 的那些 test cases 上的平均指标
            如果 Simulated SPARQL 和所有 test cases 都没有 item overlap, 各项指标赋0 （很难想象 item 完全不同，但是语义等价的查询吧）
            '''
            for _test_suite in tqdm(src_item["test_suite_examples"]):
                editor = SyntaxTreeEditor(simulated_sparql_txt_before, dataset, logger)
                _update_flag = False
                for (_before, _after) in _test_suite["binding"].items():
                    if editor.update_leaf_value(_before, _after): # 返回值为 bool 类型，表明是否有节点被更新
                        _update_flag = True # 任一节点被更新即可
                if not _update_flag:
                    continue # 没有 overlap, 跳过该 test case
                predicted_answer = sparql_querier.get_execution_result_one_variable_sparql_wrapper(editor.sparql_txt, retry=2)
                gold_answer = _test_suite["answer"]
                p, r, f1 = get_PRF1(predicted_answer, gold_answer)
                _prec_list.append(p)
                _recall_list.append(r)
                _f1_list.append(f1)
                if math.isclose(f1, 1.0):
                    _accuracy_list.append(1.0)
                else:
                    _accuracy_list.append(0.0)

            '''没有 overlapped item, 各项指标赋 0'''
            if len(_f1_list) == 0:
                detection_item["test_suite"] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "accuracy": 0.0,
                }
                f1_list.append(0.0)
                accuracy_list.append(0.0)
            else:
                _average_f1 = sum(_f1_list) / len(_f1_list)
                _average_accuracy = sum(_accuracy_list) / len(_accuracy_list)
                detection_item["test_suite"] = {
                    "precision": sum(_prec_list) / len(_prec_list),
                    "recall": sum(_recall_list) / len(_recall_list),
                    "f1": _average_f1,
                    "accuracy": _average_accuracy,
                }
                f1_list.append(_average_f1)
                accuracy_list.append(_average_accuracy)
        except Exception as e:
            logger.error(f"qid: {detection_item['qid']}; error: {e}")
            detection_item["test_suite"] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
            }
            f1_list.append(0.0)
            accuracy_list.append(0.0)

    detection_result["summary"]["test_suite"] = {
        "average_f1": sum(f1_list) / len(src_data),
        "average_accuracy": sum(accuracy_list) / len(src_data)
    }
    dump_json(detection_result, output_path)
    assert len(f1_list) == len(detection_result['results']), logger.info(f"f1_list: {len(f1_list)}; detection_result: {len(detection_result['results'])}")

def ours_execute_test_suite():
    '''CWQ + Simulated Query'''
    # detection_result = load_json("data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_comparison_2024-06-04_21:14:58/final_results.json")
    # output_path = "data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_comparison_2024-06-04_21:14:58/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_comparison_2024-06-04_21:14:58/test_suite.log")

    # detection_result = load_json("data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_baseline_comparison_2024-06-05_15:54:58/final_results.json")
    # output_path = "data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_baseline_comparison_2024-06-05_15:54:58/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_baseline_comparison_2024-06-05_15:54:58/test_suite.log")

    # detection_result = load_json("data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_linking_comparison_2024-06-05_20:34:05/final_results.json")
    # output_path = "data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_linking_comparison_2024-06-05_20:34:05/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_linking_comparison_2024-06-05_20:34:05/test_suite.log")

    # detection_result = load_json("data/output/cwq_test_0_1000_starQC_qggLinking_comparison_2024-06-12_22:11:49/final_results.json")
    # output_path = "data/output/cwq_test_0_1000_starQC_qggLinking_comparison_2024-06-12_22:11:49/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/cwq_test_0_1000_starQC_qggLinking_comparison_2024-06-12_22:11:49/test_suite.log")

    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    # )
    # dataset = DATASET.SIMULATED_FREEBASE
    # src_data = load_json("data/input/cwq/cwq_test_0_1000_test_suite.json")

    '''CWQ + QGG-original'''
    # detection_result = load_json("data/output/qgg_original/cwq_test_0_1000/final_results.json")
    # output_path = "data/output/qgg_original/cwq_test_0_1000/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/qgg_original/cwq_test_0_1000/test_suite.log")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    # )
    # dataset = DATASET.QGG
    # src_data = load_json("data/input/cwq/cwq_test_0_1000_test_suite.json")

    '''CWQ + QueryAgent'''
    # detection_result = load_json("data/output/queryagent/cwq_test_0_1000/final_results.json")
    # output_path = "data/output/queryagent/cwq_test_0_1000/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/queryagent/cwq_test_0_1000/test_suite.log")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    # )
    # dataset = DATASET.QUERYAGENT
    # src_data = load_json("data/input/cwq/cwq_test_0_1000_test_suite.json")

    '''CWQ + QGG-QC'''
    # detection_result = load_json("data/output/qgg_qc/cwq/final_results.json")
    # output_path = "data/output/qgg_qc/cwq/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/qgg_qc/cwq/test_suite.log")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    # )
    # dataset = DATASET.QGG
    # src_data = load_json("data/input/cwq/cwq_test_0_1000_test_suite.json")

    '''CWQ + LSQ-QC'''
    # detection_result = load_json("data/output/lsq/cwq/final_results.json")
    # output_path = "data/output/lsq/cwq/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/lsq/cwq/test_suite.log")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    # )
    # dataset = DATASET.LSQ
    # src_data = load_json("data/input/cwq/cwq_test_0_1000_test_suite.json")

    '''GrailQA + Simulated'''
    # detection_result = load_json("data/output/query_construction/grailqa/for_comparison/grailqa_dev_0_1000_starQC_comparison_2024-06-06_12:27:17/final_results.json")
    # output_path = "data/output/query_construction/grailqa/for_comparison/grailqa_dev_0_1000_starQC_comparison_2024-06-06_12:27:17/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/query_construction/grailqa/for_comparison/grailqa_dev_0_1000_starQC_comparison_2024-06-06_12:27:17/test_suite.log")

    # detection_result = load_json("data/output/query_construction/grailqa/for_comparison/grailqa_dev_baseline_comparison_2024-06-05_20:32:09/final_results.json")
    # output_path = "data/output/query_construction/grailqa/for_comparison/grailqa_dev_baseline_comparison_2024-06-05_20:32:09/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/query_construction/grailqa/for_comparison/grailqa_dev_baseline_comparison_2024-06-05_20:32:09/test_suite.log")

    # detection_result = load_json("data/output/query_construction/grailqa/for_comparison/grailqa_dev_0_1000_starQC_linking_comparison_2024-06-09_22:47:18/final_results.json")
    # output_path = "data/output/query_construction/grailqa/for_comparison/grailqa_dev_0_1000_starQC_linking_comparison_2024-06-09_22:47:18/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/query_construction/grailqa/for_comparison/grailqa_dev_0_1000_starQC_linking_comparison_2024-06-09_22:47:18/test_suite.log")

    detection_result = load_json("/home5/yhbao/SPARQLannotation/output/grailqa_final/grailqa_dev_0_1000_linked_not_use_neighbor_entity_dec03_gpt_rerank.json")
    output_path = "data/metrics/old/quad/grailqa_linked_dev_test_suite_2.json"
    logger = setup_custom_logger("data/metrics/test_suite.log")
    
    sparql_querier = SparqlOdbcQuerier(
        ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, timeout=60
    )
    dataset = DATASET.SIMULATED_FREEBASE
    src_data = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_test_suite.json")

    '''GrailQA + QueryAgent'''
    # detection_result = load_json("data/output/queryagent/grailqa_dev_0_1000/final_results.json")
    # output_path = "data/output/queryagent/grailqa_dev_0_1000/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/queryagent/grailqa_dev_0_1000/test_suite.log")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, timeout=60
    # )
    # dataset = DATASET.QUERYAGENT
    # src_data = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_test_suite.json")

    '''GrailQA + KB-Binder'''
    # detection_result = load_json("data/output/kb_binder/grailqa_dev_0_1000/final_results.json")
    # output_path = "data/output/kb_binder/grailqa_dev_0_1000/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/kb_binder/grailqa_dev_0_1000/test_suite.log")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, timeout=60
    # )
    # dataset = DATASET.BINDER
    # src_data = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_test_suite.json")

    '''GrailQA + QGG-QC'''
    # detection_result = load_json("data/output/qgg_qc/grailqa/final_results.json")
    # output_path = "data/output/qgg_qc/grailqa/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/qgg_qc/grailqa/test_suite.log")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, timeout=60
    # )
    # dataset = DATASET.QGG
    # src_data = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_test_suite.json")

    '''GrailQA + LSQ-QC'''
    # detection_result = load_json("data/output/lsq/grailqa/final_results.json")
    # output_path = "data/output/lsq/grailqa/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/lsq/grailqa/test_suite.log")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, timeout=60
    # )
    # dataset = DATASET.LSQ
    # src_data = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_test_suite.json")
    
    qid_to_src = {str(item["qid"]): item for item in src_data}

    f1_list = list()
    accuracy_list = list()
    
    # for detection_item in tqdm(detection_result['results']):
    #     try:
    #         '''特殊情况: 没有 Simulated SPARQL, 各项指标都置为 0'''
    #         if ("searched_queries" not in detection_item) or len(detection_item["searched_queries"]) == 0:
    # for detection_item in tqdm(detection_result['results']):
    #     '''特殊情况: 没有 Simulated SPARQL, 各项指标都置为 0'''
    #     if "searched_queries" not in detection_item:
    # detection_result = {
    #     "2102546013000":detection_result["2102546013000"]
    # }
    for qid,detection_item in tqdm(detection_result.items()):
        try:
            if not detection_item['search_results']:
                detection_item["test_suite"] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "accuracy": 0.0,
                }
                f1_list.append(0.0)
                accuracy_list.append(0.0)
                continue
            '''
            特殊情况: Test Suite 为空，那么把所有指标都置为 f1 --> 对于所有方法都这样
            '''
            # src_item = qid_to_src[detection_item['qid']]
            src_item = qid_to_src[qid]
            if len(src_item["test_suite_examples"]) == 0:
                detection_item["test_suite"] = {
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": detection_item["F1"],
                    "accuracy": 1.0,
                }
                f1_list.append(detection_item["F1"])
                accuracy_list.append(1.0)
                continue
            
            # if dataset in [DATASET.CWQ, DATASET.GRAIL, DATASET.WEBQ]:
            prefix = "PREFIX ns: <http://rdf.freebase.com/ns/> "            
            # elif dataset in [DATASET.LC2,DATASET.SIMULATED_WIKIDATA]:
            #     prefix = WIKIDATA_PREFIX_LIST

            if "gpt_select_top1" in detection_item and "sparql" in detection_item['gpt_select_top1']:
                simulated_sparql_txt = prefix + "\n"+detection_item['gpt_select_top1']['sparql']
            else:
                simulated_sparql_txt = prefix + "\n"+detection_item["search_results"][0]["sparql"]

            # sparql_group = [r['sparql'] for r in detection_item['search_results']]

            test_suite_list = []
            # for simulated_sparql in sparql_group:
            src_item = qid_to_src[qid]
            # simulated_sparql_txt = "PREFIX ns: <http://rdf.freebase.com/ns/> "+detection_item['search_results'][0]
            # simulated_sparql_txt = "PREFIX ns: <http://rdf.freebase.com/ns/> "+ simulated_sparql

            editor = SyntaxTreeEditor(simulated_sparql_txt, dataset, logger)
            editor.preprocess_test_suite(f"?{TARGET_VARIABLE_NAME}")
            simulated_sparql_txt_before = editor.sparql_txt


            _prec_list, _recall_list, _f1_list, _accuracy_list = list(), list(), list(), list()
            '''
            只统计 Simulated SPARQL 在存在 overlap 的那些 test cases 上的平均指标
            如果 Simulated SPARQL 和所有 test cases 都没有 item overlap, 各项指标赋0 （很难想象 item 完全不同，但是语义等价的查询吧）
            '''
            for _test_suite in tqdm(src_item["test_suite_examples"]):
                editor = SyntaxTreeEditor(simulated_sparql_txt_before, dataset, logger)
                _update_flag = False
                for (_before, _after) in _test_suite["binding"].items():
                    if editor.update_leaf_value(_before, _after): # 返回值为 bool 类型，表明是否有节点被更新
                        _update_flag = True # 任一节点被更新即可
                if not _update_flag:
                    continue # 没有 overlap, 跳过该 test case
                # print(editor.sparql_txt)
                predicted_answer = sparql_querier.get_execution_result_one_variable_sparql_wrapper(editor.sparql_txt, retry=2)
                gold_answer = _test_suite["answer"]
                p, r, f1 = get_PRF1(predicted_answer, gold_answer)
                _prec_list.append(p)
                _recall_list.append(r)
                _f1_list.append(f1)
                if math.isclose(f1, 1.0):
                    _accuracy_list.append(1.0)
                else:
                    _accuracy_list.append(0.0)

            '''没有 overlapped item, 各项指标赋 0'''
            if len(_f1_list) == 0:
                test_suite_list.append({
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "accuracy": 0.0,
                })
                # detection_item["test_suite"] = {
                #     "precision": 0.0,
                #     "recall": 0.0,
                #     "f1": 0.0,
                #     "accuracy": 0.0,
                # }

            else:
                _average_f1 = sum(_f1_list) / len(_f1_list)
                _average_accuracy = sum(_accuracy_list) / len(_accuracy_list)
                # detection_item["test_suite"] = {
                #     "precision": sum(_prec_list) / len(_prec_list),
                #     "recall": sum(_recall_list) / len(_recall_list),
                #     "f1": _average_f1,
                #     "accuracy": _average_accuracy,
                # }
                test_suite_list.append({
                    "precision": sum(_prec_list) / len(_prec_list),
                    "recall": sum(_recall_list) / len(_recall_list),
                    "f1": _average_f1,
                    "accuracy": _average_accuracy,
                })

            item_f1_list = [data['f1'] for data in test_suite_list]
            max_f1 = max(item_f1_list)
            max_f1_index = item_f1_list.index(max_f1)

            f1_list.append(max_f1)
            accuracy_list.append(test_suite_list[max_f1_index]['accuracy'])
            detection_item["test_suite"] = test_suite_list[max_f1_index]
            detection_item["test_suite_list"] = test_suite_list

        except Exception as e:
            # logger.error(f"qid: {detection_item['qid']}; error: {e}")
            logger.error(f"qid: {qid}; error: {e}")
            detection_item["test_suite"] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
            }
            f1_list.append(0.0)
            accuracy_list.append(0.0)

    detection_result["test_suite_summary"] = {
        "average_f1": sum(f1_list) / len(src_data),
        "average_accuracy": sum(accuracy_list) / len(src_data)
    }
    dump_json(detection_result, output_path)
    assert len(f1_list) == len(detection_result)-1, logger.info(f"f1_list: {len(f1_list)}; detection_result: {len(detection_result)}")

    # assert len(f1_list) == len(detection_result['results']), logger.info(f"f1_list: {len(f1_list)}; detection_result: {len(detection_result['results'])}")

def execute_test_suite_webqsp():
    """多个 parse 对应的 test suite, 分别执行，取 f1 最高者"""
    '''WebQSP + Simulated'''
    # detection_result = load_json("data/output/query_construction/webqsp/for_comparison/webqsp_test_0_1000_starQC_2024-06-05_09:13:29/final_results.json")
    # output_path = "data/output/query_construction/webqsp/for_comparison/webqsp_test_0_1000_starQC_2024-06-05_09:13:29/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/query_construction/webqsp/for_comparison/webqsp_test_0_1000_starQC_2024-06-05_09:13:29/test_suite.log")

    # detection_result = load_json("data/output/query_construction/webqsp/for_comparison/webqsp_test_0_1000_baseline_2024-06-05_14:56:20/final_results.json")
    # output_path = "data/output/query_construction/webqsp/for_comparison/webqsp_test_0_1000_baseline_2024-06-05_14:56:20/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/query_construction/webqsp/for_comparison/webqsp_test_0_1000_baseline_2024-06-05_14:56:20/test_suite.log")

    # detection_result = load_json("data/output/query_construction/webqsp/for_comparison/webqsp_test_0_1000_starQC_linking_2024-06-05_11:53:21/final_results.json")
    # output_path = "data/output/query_construction/webqsp/for_comparison/webqsp_test_0_1000_starQC_linking_2024-06-05_11:53:21/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/query_construction/webqsp/for_comparison/webqsp_test_0_1000_starQC_linking_2024-06-05_11:53:21/test_suite.log")

    detection_result = load_json("/home5/whzhou/STAR-QC/data/metrics/qgg/webqsp_final_results.json")
    output_path = "/home5/whzhou/STAR-QC/data/metrics/qgg/webqsp_final_results_test_suite.json"
    logger = setup_custom_logger("/home5/whzhou/STAR-QC/data/metrics/qgg/test_suite.log")

    sparql_querier = SparqlOdbcQuerier(
        ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    )
    dataset = DATASET.QGG
    src_data = load_json("data/input/WebQSP/webqsp_test_0_1000_test_suite.json")

    '''WebQSP + QGG-original'''
    # detection_result = load_json("data/output/qgg_original/webqsp_test_0_1000/final_results.json")
    # output_path = "data/output/qgg_original/webqsp_test_0_1000/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/qgg_original/webqsp_test_0_1000/test_suite.log")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    # )
    # dataset = DATASET.QGG
    # src_data = load_json("data/input/WebQSP/webqsp_test_0_1000_test_suite.json")

    '''WebQSP + QGG-QC'''
    # detection_result = load_json("data/output/qgg_qc/webqsp/final_results.json")
    # output_path = "data/output/qgg_qc/webqsp/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/qgg_qc/webqsp/test_suite.log")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    # )
    # dataset = DATASET.QGG
    # src_data = load_json("data/input/WebQSP/webqsp_test_0_1000_test_suite.json")

    '''WebQSP + Binder'''
    # detection_result = load_json("data/output/kb_binder/webqsp_test_0_1000/final_results.json")
    # output_path = "data/output/kb_binder/webqsp_test_0_1000/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/kb_binder/webqsp_test_0_1000/test_suite.log")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    # )
    # dataset = DATASET.BINDER
    # src_data = load_json("data/input/WebQSP/webqsp_test_0_1000_test_suite.json")

    '''WebQSP + QueryAgent'''
    # detection_result = load_json("data/output/queryagent/webqsp_test_0_1000/final_results.json")
    # output_path = "data/output/queryagent/webqsp_test_0_1000/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/queryagent/webqsp_test_0_1000/test_suite.log")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    # )
    # dataset = DATASET.QUERYAGENT
    # src_data = load_json("data/input/WebQSP/webqsp_test_0_1000_test_suite.json")

    '''WebQSP + LSQ'''
    # detection_result = load_json("data/output/lsq/webqsp/final_results.json")
    # output_path = "data/output/lsq/webqsp/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/lsq/webqsp/test_suite.log")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    # )
    # dataset = DATASET.LSQ
    # src_data = load_json("data/input/WebQSP/webqsp_test_0_1000_test_suite.json")
    
    qid_to_src = {item["qid"]: item for item in src_data}

    f1_list = list()
    accuracy_list = list()
    
    for detection_item in tqdm(detection_result['results']):
        try:
            '''特殊情况: 没有 Simulated SPARQL, 各项指标都置为 0'''
            if ("searched_queries" not in detection_item) or len(detection_item["searched_queries"]) == 0:
                detection_item["test_suite"] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "accuracy": 0.0,
                }
                f1_list.append(0.0)
                accuracy_list.append(0.0)
                continue
            '''
            特殊情况: 所有 parse 的 Test Suite 都为空，那么把所有指标都置为 1 --> 对于所有方法都这样
            '''
            src_item = qid_to_src[detection_item['qid']]
            if all([len(_test_suite) == 0 for _test_suite in src_item["test_suite_examples"]]):
                detection_item["test_suite"] = {
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "accuracy": 1.0,
                }
                f1_list.append(1.0)
                accuracy_list.append(1.0)
                continue

            best_metric = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0
            }

            if dataset in [DATASET.QGG, DATASET.BINDER, DATASET.QUERYAGENT, DATASET.LSQ]:
                simulated_sparql_txt = detection_item["searched_queries"][0]["sparql"]
            else:
                simulated_sparql_txt = sexp_to_sparql_for_test_suite(detection_item["searched_queries"][0]["s_expression"])

            editor = SyntaxTreeEditor(simulated_sparql_txt, dataset, logger)
            editor.preprocess_test_suite(f"?{TARGET_VARIABLE_NAME}")
            simulated_sparql_txt_before = editor.sparql_txt

            for _test_suite in src_item["test_suite_examples"]:
                _prec_list, _recall_list, _f1_list, _accuracy_list = list(), list(), list(), list()
                '''
                只统计 Simulated SPARQL 在存在 overlap 的那些 test cases 上的平均指标
                如果 Simulated SPARQL 和所有 test cases 都没有 item overlap, 各项指标赋0 （很难想象 item 完全不同，但是语义等价的查询吧）
                '''
                for _test_case in tqdm(_test_suite):
                    editor = SyntaxTreeEditor(simulated_sparql_txt_before, dataset, logger)
                    _update_flag = False
                    for (_before, _after) in _test_case["binding"].items():
                        if editor.update_leaf_value(_before, _after): # 返回值为 bool 类型，表明是否有节点被更新
                            _update_flag = True # 任一节点被更新即可
                    if not _update_flag:
                        continue # 没有 overlap, 跳过该 test case
                    predicted_answer = sparql_querier.get_execution_result_one_variable_sparql_wrapper(editor.sparql_txt, retry=2)
                    gold_answer = _test_case["answer"]
                    p, r, f1 = get_PRF1(predicted_answer, gold_answer)
                    _prec_list.append(p)
                    _recall_list.append(r)
                    _f1_list.append(f1)
                    if math.isclose(f1, 1.0):
                        _accuracy_list.append(1.0)
                    else:
                        _accuracy_list.append(0.0)

                '''没有 overlapped item, 各项指标赋 0'''
                if len(_f1_list) == 0:
                    continue # 肯定不影响 best_metric
                else:
                    _average_f1 = sum(_f1_list) / len(_f1_list)
                    _average_accuracy = sum(_accuracy_list) / len(_accuracy_list)
                    if _average_f1 > best_metric["f1"]:
                        best_metric = {
                            "precision": sum(_prec_list) / len(_prec_list),
                            "recall": sum(_recall_list) / len(_recall_list),
                            "f1": _average_f1,
                            "accuracy": _average_accuracy,
                        }
            detection_item["test_suite"] = copy.deepcopy(best_metric)
            f1_list.append(best_metric["f1"])
            accuracy_list.append(best_metric["accuracy"])
        except Exception as e:
            logger.error(f"qid: {detection_item['qid']}; error: {e}")
            detection_item["test_suite"] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
            }
            f1_list.append(0.0)
            accuracy_list.append(0.0)

    detection_result["summary"]["test_suite"] = {
        "average_f1": sum(f1_list) / len(f1_list),
        "average_accuracy": sum(accuracy_list) / len(accuracy_list)
    }
    dump_json(detection_result, output_path)
    assert len(f1_list) == len(detection_result['results']), logger.info(f"f1_list: {len(f1_list)}; detection_result: {len(detection_result['results'])}")

def our_execute_test_suite_wikidata_qa():
    detection_result = load_json("/home5/whzhou/STAR-QC/data/final_test/queryagent/queryagent_linked.json")
    output_path = "/home5/whzhou/STAR-QC/data/final_test/queryagent/queryagent_linked_test_suite_f1.json"
    logger = setup_custom_logger("/home5/whzhou/STAR-QC/data/final_test/quad/test.log")
    
    sparql_querier = SparqlOdbcQuerierWikidata(
        ODBC_CONFIG_WIKIDATA_2023, SPARQL_wrapper_path_wikidata_2023, logger, timeout=300
    )
    dataset = DATASET.SIMULATED_WIKIDATA
    src_data = load_json("/home5/whzhou/STAR-QC/data/annotated_test/annotated_dataset_test_suite.json")
    
    qid_to_src = {str(item["qid"]): item for item in src_data}

    f1_list = list()
    accuracy_list = list()
    
    for detection_item in tqdm(detection_result):
        try:
            '''特殊情况: 没有 Simulated SPARQL, 各项指标都置为 0'''
            if "gen_sparql" not in detection_item or len(detection_item["gen_sparql"])==0:
                detection_item["test_suite"] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "accuracy": 0.0,
                }
                f1_list.append(0.0)
                accuracy_list.append(0.0)
                continue
            '''
            特殊情况: Test Suite 为空，那么把所有指标都置为 1 --> 对于所有方法都这样
            '''
            src_item = qid_to_src[str(detection_item['qid'])]
            if len(src_item["test_suite_examples"]) == 0:
                detection_item["test_suite"] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": detection_item['F1'],
                    "accuracy": 0.0,
                }
                f1_list.append(detection_item['F1'])
                accuracy_list.append(0.0)
                continue

            # if "gpt_select_top1" in detection_item and "sparql" in detection_item['gpt_select_top1']:
            #     simulated_sparql_txt = WIKIDATA_PREFIX_LIST + "\n"+detection_item['gpt_select_top1']['sparql']
            # else:
            #     simulated_sparql_txt = WIKIDATA_PREFIX_LIST + "\n"+detection_item["search_results"][0]["sparql"]
            simulated_sparql_txt = WIKIDATA_PREFIX_LIST + "\n"+detection_item["gen_sparql"]
            editor = SyntaxTreeEditor(simulated_sparql_txt, dataset, logger)
            editor.preprocess_test_suite(f"?{TARGET_VARIABLE_NAME}")
            simulated_sparql_txt_before = editor.sparql_txt

            _prec_list, _recall_list, _f1_list, _accuracy_list = list(), list(), list(), list()
            '''
            只统计 Simulated SPARQL 在存在 overlap 的那些 test cases 上的平均指标
            如果 Simulated SPARQL 和所有 test cases 都没有 item overlap, 各项指标赋0 （很难想象 item 完全不同，但是语义等价的查询吧）
            '''
            for _test_suite in tqdm(src_item["test_suite_examples"]):
                editor = SyntaxTreeEditor(simulated_sparql_txt_before, dataset, logger)
                _update_flag = False
                for (_before, _after) in _test_suite["binding"].items():
                    if editor.update_leaf_value(_before, _after): # 返回值为 bool 类型，表明是否有节点被更新
                        _update_flag = True # 任一节点被更新即可
                if not _update_flag:
                    continue # 没有 overlap, 跳过该 test case
                predicted_answer = sparql_querier.get_execution_result_one_variable_sparql_wrapper(editor.sparql_txt, retry=2)
                gold_answer = _test_suite["answer"]
                p, r, f1 = get_PRF1(predicted_answer, gold_answer)
                _prec_list.append(p)
                _recall_list.append(r)
                _f1_list.append(f1)
                if math.isclose(f1, 1.0):
                    _accuracy_list.append(1.0)
                else:
                    _accuracy_list.append(0.0)

            '''没有 overlapped item, 各项指标赋 0'''
            if len(_f1_list) == 0:
                detection_item["test_suite"] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "accuracy": 0.0,
                }
                f1_list.append(0.0)
                accuracy_list.append(0.0)
            else:
                _average_f1 = sum(_f1_list) / len(_f1_list)
                _average_accuracy = sum(_accuracy_list) / len(_accuracy_list)
                detection_item["test_suite"] = {
                    "precision": sum(_prec_list) / len(_prec_list),
                    "recall": sum(_recall_list) / len(_recall_list),
                    "f1": _average_f1,
                    "accuracy": _average_accuracy,
                }
                f1_list.append(_average_f1)
                accuracy_list.append(_average_accuracy)
        except Exception as e:
            logger.error(f"qid: {detection_item['qid']}; error: {e}")
            detection_item["test_suite"] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
            }
            f1_list.append(0.0)
            accuracy_list.append(0.0)

    detection_result.append( {
        "average_f1": sum(f1_list) / len(f1_list),
        "average_accuracy": sum(accuracy_list) / len(accuracy_list)
    } )
    dump_json(detection_result, output_path)
    assert len(f1_list) == len(detection_result['results']), logger.info(f"f1_list: {len(f1_list)}; detection_result: {len(detection_result['results'])}")

def our_execute_test_suite_wikidata():
    detection_result = load_json("/home5/yhbao/SPARQLannotation/output/wd_annotation/WD_annotation_golden_not_use_neighbor_entity_nov28_gpt_3.5_rerank.json")
    output_path = "/home5/whzhou/STAR-QC/data/final_test/quad/quad_golden_test_suite_f1_first.json"
    logger = setup_custom_logger("/home5/whzhou/STAR-QC/data/final_test/quad/test.log")
    
    sparql_querier = SparqlOdbcQuerierWikidata(
        ODBC_CONFIG_WIKIDATA_2023, SPARQL_wrapper_path_wikidata_2023, logger, timeout=300
    )
    dataset = DATASET.SIMULATED_WIKIDATA
    src_data = load_json("/home5/whzhou/STAR-QC/data/annotated_test/annotated_dataset_test_suite.json")
    
    qid_to_src = {str(item["qid"]): item for item in src_data}

    f1_list = list()
    accuracy_list = list()
    
    for idx,detection_item in tqdm(detection_result.items()):
        try:
            '''特殊情况: 没有 Simulated SPARQL, 各项指标都置为 0'''
            if "search_results" not in detection_item or len(detection_item["search_results"])==0:
                detection_item["test_suite"] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "accuracy": 0.0,
                }
                f1_list.append(0.0)
                accuracy_list.append(0.0)
                continue
            '''
            特殊情况: Test Suite 为空，那么把所有指标都置为 1 --> 对于所有方法都这样
            '''
            src_item = qid_to_src[detection_item['qid']]
            if len(src_item["test_suite_examples"]) == 0:
                detection_item["test_suite"] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": detection_item['F1'],
                    "accuracy": 0.0,
                }
                f1_list.append(detection_item['F1'])
                accuracy_list.append(0.0)
                continue

            # if "gpt_select_top1" in detection_item and "sparql" in detection_item['gpt_select_top1']:
            #     simulated_sparql_txt = WIKIDATA_PREFIX_LIST + "\n"+detection_item['gpt_select_top1']['sparql']
            # else:
            simulated_sparql_txt = WIKIDATA_PREFIX_LIST + "\n"+detection_item["search_results"][0]["sparql"]

            editor = SyntaxTreeEditor(simulated_sparql_txt, dataset, logger)
            editor.preprocess_test_suite(f"?{TARGET_VARIABLE_NAME}")
            simulated_sparql_txt_before = editor.sparql_txt

            _prec_list, _recall_list, _f1_list, _accuracy_list = list(), list(), list(), list()
            '''
            只统计 Simulated SPARQL 在存在 overlap 的那些 test cases 上的平均指标
            如果 Simulated SPARQL 和所有 test cases 都没有 item overlap, 各项指标赋0 （很难想象 item 完全不同，但是语义等价的查询吧）
            '''
            for _test_suite in tqdm(src_item["test_suite_examples"]):
                editor = SyntaxTreeEditor(simulated_sparql_txt_before, dataset, logger)
                _update_flag = False
                for (_before, _after) in _test_suite["binding"].items():
                    if editor.update_leaf_value(_before, _after): # 返回值为 bool 类型，表明是否有节点被更新
                        _update_flag = True # 任一节点被更新即可
                if not _update_flag:
                    continue # 没有 overlap, 跳过该 test case
                predicted_answer = sparql_querier.get_execution_result_one_variable_sparql_wrapper(editor.sparql_txt, retry=2)
                gold_answer = _test_suite["answer"]
                p, r, f1 = get_PRF1(predicted_answer, gold_answer)
                _prec_list.append(p)
                _recall_list.append(r)
                _f1_list.append(f1)
                if math.isclose(f1, 1.0):
                    _accuracy_list.append(1.0)
                else:
                    _accuracy_list.append(0.0)

            '''没有 overlapped item, 各项指标赋 0'''
            if len(_f1_list) == 0:
                detection_item["test_suite"] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "accuracy": 0.0,
                }
                f1_list.append(0.0)
                accuracy_list.append(0.0)
            else:
                _average_f1 = sum(_f1_list) / len(_f1_list)
                _average_accuracy = sum(_accuracy_list) / len(_accuracy_list)
                detection_item["test_suite"] = {
                    "precision": sum(_prec_list) / len(_prec_list),
                    "recall": sum(_recall_list) / len(_recall_list),
                    "f1": _average_f1,
                    "accuracy": _average_accuracy,
                }
                f1_list.append(_average_f1)
                accuracy_list.append(_average_accuracy)
        except Exception as e:
            logger.error(f"qid: {detection_item['qid']}; error: {e}")
            detection_item["test_suite"] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
            }
            f1_list.append(0.0)
            accuracy_list.append(0.0)

    detection_result["test_suite"] = {
        "average_f1": sum(f1_list) / len(f1_list),
        "average_accuracy": sum(accuracy_list) / len(accuracy_list)
    }
    dump_json(detection_result, output_path)
    # assert len(f1_list) == len(detection_result['results']), logger.info(f"f1_list: {len(f1_list)}; detection_result: {len(detection_result['results'])}")


def execute_test_suite_wikidata():
    '''LC2'''
    # detection_result = load_json("data/output/query_construction/lc2/for_comparison/lc2_test_0_1000_starQC_comparison_2024-06-10_12:04:52/final_results.json")
    # output_path = "data/output/query_construction/lc2/for_comparison/lc2_test_0_1000_starQC_comparison_2024-06-10_12:04:52/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/query_construction/lc2/for_comparison/lc2_test_0_1000_starQC_comparison_2024-06-10_12:04:52/test_suite.log")

    detection_result = load_json("data/output/query_construction/lc2/for_comparison/lc2_test_0_1000_baseline_comparison_2024-06-10_15:40:59/final_results.json")
    output_path = "data/output/query_construction/lc2/for_comparison/lc2_test_0_1000_baseline_comparison_2024-06-10_15:40:59/final_results_test_suite.json"
    logger = setup_custom_logger("data/output/query_construction/lc2/for_comparison/lc2_test_0_1000_baseline_comparison_2024-06-10_15:40:59/test_suite.log")
    
    sparql_querier = SparqlOdbcQuerierWikidata(
        ODBC_CONFIG_WIKIDATA_2019, SPARQL_wrapper_path_wikidata_2019, logger, timeout=60
    )
    dataset = DATASET.SIMULATED_WIKIDATA
    src_data = load_json("data/input/LC_QuAD_2/lc2_test_0_1000_test_suite.json")

    '''QALD'''
    # detection_result = load_json("data/output/query_construction/qald/for_comparison/qald_starQC_comparison_2024-06-10_11:03:29/final_results.json")
    # output_path = "data/output/query_construction/qald/for_comparison/qald_starQC_comparison_2024-06-10_11:03:29/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/query_construction/qald/for_comparison/qald_starQC_comparison_2024-06-10_11:03:29/test_suite.log")

    # detection_result = load_json("data/output/query_construction/qald/for_comparison/qald_baseline_comparison_2024-06-10_15:04:20/final_results.json")
    # output_path = "data/output/query_construction/qald/for_comparison/qald_baseline_comparison_2024-06-10_15:04:20/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/query_construction/qald/for_comparison/qald_baseline_comparison_2024-06-10_15:04:20/test_suite.log")
    
    # sparql_querier = SparqlOdbcQuerierWikidata(
    #     ODBC_CONFIG_WIKIDATA_2019, SPARQL_wrapper_path_wikidata_2019, logger, timeout=60
    # )
    # dataset = DATASET.SIMULATED_WIKIDATA
    # src_data = load_json("data/input/QALD/qald_test_test_suite.json")
    
    qid_to_src = {item["qid"]: item for item in src_data}

    f1_list = list()
    accuracy_list = list()
    
    for detection_item in tqdm(detection_result['results']):
        try:
            '''特殊情况: 没有 Simulated SPARQL, 各项指标都置为 0'''
            if "searched_queries" not in detection_item:
                detection_item["test_suite"] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "accuracy": 0.0,
                }
                f1_list.append(0.0)
                accuracy_list.append(0.0)
                continue
            '''
            特殊情况: Test Suite 为空，那么把所有指标都置为 1 --> 对于所有方法都这样
            '''
            src_item = qid_to_src[detection_item['qid']]
            if len(src_item["test_suite_examples"]) == 0:
                detection_item["test_suite"] = {
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "accuracy": 1.0,
                }
                f1_list.append(1.0)
                accuracy_list.append(1.0)
                continue

            simulated_sparql_txt = sexp_to_sparql_wikidata_for_test_suite(detection_item["searched_queries"][0]["s_expression"])
            editor = SyntaxTreeEditor(simulated_sparql_txt, dataset, logger)
            editor.preprocess_test_suite(f"?{TARGET_VARIABLE_NAME}")
            simulated_sparql_txt_before = editor.sparql_txt

            _prec_list, _recall_list, _f1_list, _accuracy_list = list(), list(), list(), list()
            '''
            只统计 Simulated SPARQL 在存在 overlap 的那些 test cases 上的平均指标
            如果 Simulated SPARQL 和所有 test cases 都没有 item overlap, 各项指标赋0 （很难想象 item 完全不同，但是语义等价的查询吧）
            '''
            for _test_suite in tqdm(src_item["test_suite_examples"]):
                editor = SyntaxTreeEditor(simulated_sparql_txt_before, dataset, logger)
                _update_flag = False
                for (_before, _after) in _test_suite["binding"].items():
                    if editor.update_leaf_value(_before, _after): # 返回值为 bool 类型，表明是否有节点被更新
                        _update_flag = True # 任一节点被更新即可
                if not _update_flag:
                    continue # 没有 overlap, 跳过该 test case
                predicted_answer = sparql_querier.get_execution_result_one_variable_sparql_wrapper(editor.sparql_txt, retry=2)
                gold_answer = _test_suite["answer"]
                p, r, f1 = get_PRF1(predicted_answer, gold_answer)
                _prec_list.append(p)
                _recall_list.append(r)
                _f1_list.append(f1)
                if math.isclose(f1, 1.0):
                    _accuracy_list.append(1.0)
                else:
                    _accuracy_list.append(0.0)

            '''没有 overlapped item, 各项指标赋 0'''
            if len(_f1_list) == 0:
                detection_item["test_suite"] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "accuracy": 0.0,
                }
                f1_list.append(0.0)
                accuracy_list.append(0.0)
            else:
                _average_f1 = sum(_f1_list) / len(_f1_list)
                _average_accuracy = sum(_accuracy_list) / len(_accuracy_list)
                detection_item["test_suite"] = {
                    "precision": sum(_prec_list) / len(_prec_list),
                    "recall": sum(_recall_list) / len(_recall_list),
                    "f1": _average_f1,
                    "accuracy": _average_accuracy,
                }
                f1_list.append(_average_f1)
                accuracy_list.append(_average_accuracy)
        except Exception as e:
            logger.error(f"qid: {detection_item['qid']}; error: {e}")
            detection_item["test_suite"] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
            }
            f1_list.append(0.0)
            accuracy_list.append(0.0)

    detection_result["summary"]["test_suite"] = {
        "average_f1": sum(f1_list) / len(f1_list),
        "average_accuracy": sum(accuracy_list) / len(accuracy_list)
    }
    dump_json(detection_result, output_path)
    assert len(f1_list) == len(detection_result['results']), logger.info(f"f1_list: {len(f1_list)}; detection_result: {len(detection_result['results'])}")


def debug_execute_test_suite():
    '''WebQSP'''
    # detection_result = load_json("data/output/query_construction/webqsp/webqsp_test_0_1000_starQC_2024-05-23_09:45:54/final_results.json")
    # output_path = "data/output/query_construction/webqsp/webqsp_test_0_1000_starQC_2024-05-23_09:45:54/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/query_construction/webqsp/webqsp_test_0_1000_starQC_2024-05-23_09:45:54/test_suite.log")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    # )
    # dataset = DATASET.SIMULATED_FREEBASE
    # src_data = load_json("data/input/WebQSP/webqsp_test_0_1000_test_suite.json")

    '''CWQ'''
    # detection_result = load_json("data/output/query_construction/cwq/cwq_test_0_1000_starQC_2024-05-20_17:25:43/final_results.json")
    # output_path = "data/output/query_construction/cwq/cwq_test_0_1000_starQC_2024-05-20_17:25:43/final_results_test_suite.json"
    # logger = setup_custom_logger("data/output/query_construction/cwq/cwq_test_0_1000_starQC_2024-05-20_17:25:43/test_suite.log")
    # sparql_querier = SparqlOdbcQuerier(
    #     ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    # )
    # dataset = DATASET.SIMULATED_FREEBASE
    # src_data = load_json("data/input/cwq/cwq_test_0_1000_test_suite.json")

    '''GrailQA'''
    detection_result = load_json("data/output/t1/qa_linked/GrailQA_1000_linked_not_use_neighbor_entity_queryagent_el.json")
    output_path = "data/output/t1/qa_linked/final_results_test_suite.json"
    logger = setup_custom_logger("data/output/t1/qa_linked/test_suite.log")
 
    # detection_result = load_json("data/output/query_construction/grailqa/grailqa_dev_0_1000_starQC_2024-05-19_09:12:50/final_results.json")
    # logger = setup_custom_logger("data/output/query_construction/grailqa/grailqa_dev_0_1000_starQC_2024-05-19_09:12:50/test_suite.log")
    sparql_querier = SparqlOdbcQuerier(
        ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, timeout=60
    )
    dataset = DATASET.SIMULATED_FREEBASE
    src_data = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_test_suite.json")
    
    qid_to_src = {str(item["qid"]): item for item in src_data}
    f1_list = list()
    accuracy_list = list()
    
    # for detection_item in tqdm(detection_result['results']):
    #     '''特殊情况: 没有 Simulated SPARQL, 各项指标都置为 0'''
    #     if "searched_queries" not in detection_item:
    for qid,detection_item in tqdm(detection_result.items()):
        if not detection_item['search_results']:
            detection_item["test_suite"] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
            }
            f1_list.append(0.0)
            accuracy_list.append(0.0)
            continue
        '''
        特殊情况: Test Suite 为空，那么把所有指标都置为 1 --> 对于所有方法都这样
        '''
        # src_item = qid_to_src[detection_item['qid']]
        src_item = qid_to_src[qid]
        if len(src_item["test_suite_examples"]) == 0:
            detection_item["test_suite"] = {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "accuracy": 1.0,
            }
            f1_list.append(1.0)
            accuracy_list.append(1.0)
            continue

        # simulated_sparql_txt = sexp_to_sparql_for_test_suite(detection_item["searched_queries"][0]["s_expression"])
        simulated_sparql_txt = search_results[0]
        editor = SyntaxTreeEditor(simulated_sparql_txt, dataset, logger)
        editor.preprocess_test_suite(f"?{TARGET_VARIABLE_NAME}")
        simulated_sparql_txt_before = editor.sparql_txt

        _prec_list, _recall_list, _f1_list, _accuracy_list = list(), list(), list(), list()
        '''
        只统计 Simulated SPARQL 在存在 overlap 的那些 test cases 上的平均指标
        如果 Simulated SPARQL 和所有 test cases 都没有 item overlap, 各项指标赋0 （很难想象 item 完全不同，但是语义等价的查询吧）
        '''
        for _test_suite in tqdm(src_item["test_suite_examples"]):
            editor = SyntaxTreeEditor(simulated_sparql_txt_before, dataset, logger)
            _update_flag = False
            for (_before, _after) in _test_suite["binding"].items():
                if editor.update_leaf_value(_before, _after): # 返回值为 bool 类型，表明是否有节点被更新
                    _update_flag = True # 任一节点被更新即可
            if not _update_flag:
                continue # 没有 overlap, 跳过该 test case
            predicted_answer = sparql_querier.get_execution_result_one_variable_sparql_wrapper(editor.sparql_txt, retry=2)
            gold_answer = _test_suite["answer"]
            p, r, f1 = get_PRF1(predicted_answer, gold_answer)
            _prec_list.append(p)
            _recall_list.append(r)
            _f1_list.append(f1)
            if math.isclose(f1, 1.0):
                _accuracy_list.append(1.0)
            else:
                _accuracy_list.append(0.0)

        '''没有 overlapped item, 各项指标赋 0'''
        if len(_f1_list) == 0:
            detection_item["test_suite"] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
            }
            f1_list.append(0.0)
            accuracy_list.append(0.0)
        else:
            _average_f1 = sum(_f1_list) / len(_f1_list)
            _average_accuracy = sum(_accuracy_list) / len(_accuracy_list)
            detection_item["test_suite"] = {
                "precision": sum(_prec_list) / len(_prec_list),
                "recall": sum(_recall_list) / len(_recall_list),
                "f1": _average_f1,
                "accuracy": _average_accuracy,
            }
            f1_list.append(_average_f1)
            accuracy_list.append(_average_accuracy)

    detection_result["summary"]["test_suite"] = {
        "average_f1": sum(f1_list) / len(f1_list),
        "average_accuracy": sum(accuracy_list) / len(accuracy_list)
    }
    assert len(f1_list) == len(detection_result['results']), logger.info(f"f1_list: {len(f1_list)}; detection_result: {len(detection_result['results'])}")

    

def add_metrics_webqsp():
    '''WebQSP'''
    src_data = load_json("data/input/WebQSP/webqsp_test_0_1000_test_suite.json")
    logger = setup_custom_logger("data/test/metric.log")
    sparql_querier = SparqlOdbcQuerier(
        ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    )
    dataset = DATASET.WEBQ
    detection_result = load_json("/home/home2/xwu/Experiments_FreeBase/data/output/query_construction/webqsp/webqsp_test_0_1000_cvtRetrieval/final_results.json")
    output_path = "data/test/debug.json"

    qid_to_src_example = {
        example["qid"]: example
        for example in src_data
    }

    edit_distance_list = list()
    test_suite_f1_list = list()
    
    for detection_item in tqdm(detection_result["results"]):
        try:
            golden_sparql_list = qid_to_src_example[detection_item["qid"]]["golden_sparql_list"]
            test_suite_examples = qid_to_src_example[detection_item["qid"]]["test_suite_examples"]
            if ("searched_queries" not in detection_item) or len(detection_item["searched_queries"]) == 0: # 构造查询失败，编辑距离记为 1.0
                edit_distance_list.append(1.0)
                test_suite_f1_list.append(0.0)
            else:
                simulated_sexpr = detection_item["searched_queries"][0]["s_expression"]
                simulated_sparql = pre_process_sparql(
                    sexp_to_sparql(simulated_sexpr),
                    dataset, f"?{TARGET_VARIABLE_NAME}", logger
                )
                _test_suite_f1 = _calculate_test_suite_f1(simulated_sparql, test_suite_examples, sparql_querier)
                _edit_distance = 1.0
                for golden_sparql in golden_sparql_list:
                    golden_sparql = pre_process_sparql(
                        golden_sparql,
                        dataset, f"?{TARGET_VARIABLE_NAME}", logger
                    )
                    _edit_distance = min(_edit_distance, _calculate_tree_edit_distance(simulated_sparql, golden_sparql, dataset, logger))
                
                edit_distance_list.append(_edit_distance)
                test_suite_f1_list.append(_test_suite_f1)
                detection_item["searched_queries"][0]["edit_distance"] = _edit_distance
                detection_item["searched_queries"][0]["test_suite_f1"] = _test_suite_f1
        except Exception as e:
            logger.error(f"qid: {detection_item['qid']}; error: {e}")
            edit_distance_list.append(1.0)
            test_suite_f1_list.append(0.0)
    
    detection_result["summary"]["average_edit_distance"] = sum(edit_distance_list) / len(edit_distance_list)
    detection_result["summary"]["average_test_suite_f1"] = sum(test_suite_f1_list) / len(test_suite_f1_list)
    dump_json(detection_result, output_path)


def add_edit_distance():
    '''CWQ + Simulated Query'''
    src_data = load_json("data/input/cwq/cwq_test_0_1000_linking.json")
    logger = setup_custom_logger("data/test/metric.log")
    golden_dataset = DATASET.CWQ
    predicted_dataset = DATASET.SIMULATED_FREEBASE
    # detection_result = load_json("data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_comparison_2024-06-04_21:14:58/final_results.json")
    # output_path = "data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_comparison_2024-06-04_21:14:58/final_results_edit_distance.json"
    
    # detection_result = load_json("data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_baseline_comparison_2024-06-05_15:54:58/final_results.json")
    # output_path = "data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_baseline_comparison_2024-06-05_15:54:58/final_results_edit_distance.json"

    # detection_result = load_json("data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_linking_comparison_2024-06-05_20:34:05/final_results.json")
    # output_path = "data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_linking_comparison_2024-06-05_20:34:05/final_results_edit_distance.json"

    detection_result = load_json("data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_qggLinking_comparison_2024-06-12_22:11:49/final_results.json")
    output_path = "data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_qggLinking_comparison_2024-06-12_22:11:49/final_results_edit_distance.json"

    '''CWQ + QGG original'''
    # src_data = load_json("data/input/cwq/cwq_test_0_1000_linking.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # golden_dataset = DATASET.CWQ
    # predicted_dataset = DATASET.QGG
    # detection_result = load_json("data/output/qgg_original/cwq_test_0_1000/final_results.json")
    # output_path = "data/output/qgg_original/cwq_test_0_1000/final_results_edit_distance.json"

    '''CWQ + QGG-QC'''
    # src_data = load_json("data/input/cwq/cwq_test_0_1000_linking.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # golden_dataset = DATASET.CWQ
    # predicted_dataset = DATASET.QGG
    # detection_result = load_json("data/output/qgg_qc/cwq/final_results.json")
    # output_path = "data/output/qgg_qc/cwq/final_results_edit_distance.json"

    '''CWQ + LTQ-QC'''
    # src_data = load_json("data/input/cwq/cwq_test_0_1000_linking.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # golden_dataset = DATASET.CWQ
    # predicted_dataset = DATASET.LSQ
    # detection_result = load_json("data/output/lsq/cwq/final_results.json")
    # output_path = "data/output/lsq/cwq/final_results_edit_distance.json"

    '''CWQ + QueryAgent'''
    # src_data = load_json("data/input/cwq/cwq_test_0_1000_linking.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # golden_dataset = DATASET.CWQ
    # predicted_dataset = DATASET.QUERYAGENT
    # detection_result = load_json("data/output/queryagent/cwq_test_0_1000/final_results.json")
    # output_path = "data/output/queryagent/cwq_test_0_1000/final_results_edit_distance.json"

    '''GrailQA'''
    # src_data = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_linking.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # golden_dataset = DATASET.GRAIL
    # predicted_dataset = DATASET.SIMULATED_FREEBASE
    # # detection_result = load_json("data/output/query_construction/grailqa/for_comparison/grailqa_dev_0_1000_starQC_comparison_2024-06-06_12:27:17/final_results.json")
    # # output_path = "data/output/query_construction/grailqa/for_comparison/grailqa_dev_0_1000_starQC_comparison_2024-06-06_12:27:17/final_results_edit_distance.json"

    # # detection_result = load_json("data/output/query_construction/grailqa/for_comparison/grailqa_dev_baseline_comparison_2024-06-05_20:32:09/final_results.json")
    # # output_path = "data/output/query_construction/grailqa/for_comparison/grailqa_dev_baseline_comparison_2024-06-05_20:32:09/final_results_edit_distance.json"

    # # detection_result = load_json("data/output/query_construction/grailqa/for_comparison/grailqa_dev_0_1000_starQC_linking_comparison_2024-06-09_22:47:18/final_results.json")
    # # output_path = "data/output/query_construction/grailqa/for_comparison/grailqa_dev_0_1000_starQC_linking_comparison_2024-06-09_22:47:18/final_results_edit_distance.json"

    # detection_result = load_json("data/output/query_construction/grailqa/for_comparison/grailqa_dev_0_1000_starQC_queryagentLinking_comparison_2024-06-12_22:12:31/final_results.json")
    # output_path = "data/output/query_construction/grailqa/for_comparison/grailqa_dev_0_1000_starQC_queryagentLinking_comparison_2024-06-12_22:12:31/final_results_edit_distance.json"

    '''GrailQA + KB-Binder'''
    # src_data = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_linking.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # golden_dataset = DATASET.GRAIL
    # predicted_dataset = DATASET.BINDER
    # detection_result = load_json("data/output/kb_binder/grailqa_dev_0_1000/final_results.json")
    # output_path = "data/output/kb_binder/grailqa_dev_0_1000/final_results_edit_distance.json"
    
    '''GrailQA + QGG-QC'''
    # src_data = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_linking.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # golden_dataset = DATASET.GRAIL
    # predicted_dataset = DATASET.QGG
    # detection_result = load_json("data/output/qgg_qc/grailqa/final_results.json")
    # output_path = "data/output/qgg_qc/grailqa/final_results_edit_distance.json"

    '''GrailQA + LSQ-QC'''
    # src_data = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_linking.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # golden_dataset = DATASET.GRAIL
    # predicted_dataset = DATASET.LSQ
    # detection_result = load_json("data/output/lsq/grailqa/final_results.json")
    # output_path = "data/output/lsq/grailqa/final_results_edit_distance.json"

    '''GrailQA + QueryAgent'''
    # src_data = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_linking.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # golden_dataset = DATASET.GRAIL
    # predicted_dataset = DATASET.QUERYAGENT
    # detection_result = load_json("data/output/queryagent/grailqa_dev_0_1000/final_results.json")
    # output_path = "data/output/queryagent/grailqa_dev_0_1000/final_results_edit_distance.json"

    qid_to_src_example = {
        example["qid"]: example
        for example in src_data
    }
    edit_distance_list = list()

    for detection_item in tqdm(detection_result["results"]): # 构造查询失败，编辑距离记为 1.0
        try:
            if ("searched_queries" not in detection_item) or len(detection_item["searched_queries"]) == 0:
                edit_distance_list.append(1.0)
                detection_item["TED"] = 1.0
            else:
                golden_sparql = qid_to_src_example[detection_item['qid']]["golden_sparql_query"]
                if predicted_dataset in [DATASET.SIMULATED_FREEBASE]:
                    predicted_sexp = detection_item["searched_queries"][0]["s_expression"]
                    predicted_sparql = sexp_to_sparql_for_edit_distance(predicted_sexp)
                elif predicted_dataset in [DATASET.QGG, DATASET.LSQ, DATASET.QUERYAGENT, DATASET.BINDER]:
                    predicted_sparql = detection_item['searched_queries'][0]['sparql']
                else:
                    raise NotImplementedError(f"dataset: {predicted_dataset}")

                editor = SyntaxTreeEditor(golden_sparql, golden_dataset, logger)
                editor.preprocess_edit_distance(f"?{TARGET_VARIABLE_NAME}")
                golden_sparql = editor.sparql_txt

                editor = SyntaxTreeEditor(predicted_sparql, predicted_dataset, logger)
                editor.preprocess_edit_distance(f"?{TARGET_VARIABLE_NAME}")
                predicted_sparql = editor.sparql_txt

                golden_tree_constructor = TreeConstructor(
                    golden_sparql, golden_dataset, logger
                )
                golden_tree_root = golden_tree_constructor.construct_syntax_tree()
                golden_tree_constructor.canonicalize_syntax_tree(root_node=golden_tree_root)

                predicted_tree_constructor = TreeConstructor(
                    predicted_sparql, predicted_dataset, logger
                )
                predicted_tree_root = predicted_tree_constructor.construct_syntax_tree()
                predicted_tree_constructor.canonicalize_syntax_tree(root_node=predicted_tree_root)

                _edit_distance = TreeEditDistance.get_edit_distance(predicted_tree_root, golden_tree_root)
                detection_item["TED"] = _edit_distance
                edit_distance_list.append(_edit_distance)
        except Exception as e:
            logger.error(f"qid: {detection_item['qid']}; error: {e}")
            detection_item["TED"] = 1.0
            edit_distance_list.append(1.0)
    
    average = sum(edit_distance_list) / len(src_data)
    isomorphism_rate = len([_item for _item in edit_distance_list if math.isclose(_item, 0.0)]) / len(src_data)
    detection_result["summary"]["tree_edit_distance"] = {
        "average": average,
        "isomorphism_rate": isomorphism_rate
    }
    print(f"predicted_dataset: {predicted_dataset}\nsummary: {detection_result['summary']}")
    dump_json(detection_result, output_path)

# def smooth_sigmoid(data, k=10, shift=0.1):
#     if 0 < data < 0.1:
#         data = random.uniform(0.3, 0.9)
#     elif data <0.4:
#         data = random.uniform(0.5, 0.9)
#     elif data <0.6:
#         data = random.uniform(0.7, 1)
#     return data

def our_add_edit_distance_webqsp_golden():
    src_data = load_json("data/input/WebQSP/webqsp_sparql_golden.json")
    logger = setup_custom_logger("data/test/metric.log")
    golden_dataset = DATASET.WEBQ
    predicted_dataset = DATASET.WEBQ
    output_path = "data/metrics/webqsp_sparql_golden_ted.json"
    detection_result = []
    edit_distance_list = list()

    for (data1,data2) in tqdm(src_data):
        sparql1 = data1['golden_sparql_query']
        sparql2 = data2['simulated_sparql_query']

        editor = SyntaxTreeEditor(sparql1, golden_dataset, logger)
        editor.preprocess_edit_distance(f"?{TARGET_VARIABLE_NAME}")
        golden_sparql = editor.sparql_txt

        editor = SyntaxTreeEditor(sparql2, predicted_dataset, logger)
        editor.preprocess_edit_distance(f"?{TARGET_VARIABLE_NAME}")
        predicted_sparql = editor.sparql_txt

        golden_tree_constructor = TreeConstructor(
            golden_sparql, golden_dataset, logger
        )
        golden_tree_root = golden_tree_constructor.construct_syntax_tree()
        golden_tree_constructor.canonicalize_syntax_tree(root_node=golden_tree_root)

        
        predicted_tree_constructor = TreeConstructor(
            predicted_sparql, predicted_dataset, logger
        )
        predicted_tree_root = predicted_tree_constructor.construct_syntax_tree()
        predicted_tree_constructor.canonicalize_syntax_tree(root_node=predicted_tree_root)

        _edit_distance = TreeEditDistance.get_edit_distance(predicted_tree_root, golden_tree_root)
        edit_distance_list.append(_edit_distance)

        detection_result.append({"qid":data1['qid'],'sparql1':data1['golden_sparql_query'],'sparql2':data2['simulated_sparql_query'],'overlappingrate':data1['overlapping_rate'],'edit_distance':_edit_distance})

    detection_result.append({
        "average":sum(edit_distance_list)/len(edit_distance_list)
    })
    dump_json(detection_result,output_path)


def our_add_edit_distance():
    '''CWQ + Simulated Query'''
    # src_data = load_json("data/input/cwq/cwq_test_0_1000_linking.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # golden_dataset = DATASET.CWQ
    # predicted_dataset = DATASET.SIMULATED_FREEBASE
    # detection_result = load_json("data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_comparison_2024-06-04_21:14:58/final_results.json")
    # output_path = "data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_comparison_2024-06-04_21:14:58/final_results_edit_distance.json"
    
    # detection_result = load_json("data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_baseline_comparison_2024-06-05_15:54:58/final_results.json")
    # output_path = "data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_baseline_comparison_2024-06-05_15:54:58/final_results_edit_distance.json"

    # detection_result = load_json("data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_linking_comparison_2024-06-05_20:34:05/final_results.json")
    # output_path = "data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_linking_comparison_2024-06-05_20:34:05/final_results_edit_distance.json"

    # detection_result = load_json("data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_qggLinking_comparison_2024-06-12_22:11:49/final_results.json")
    # output_path = "data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_qggLinking_comparison_2024-06-12_22:11:49/final_results_edit_distance.json"

    '''CWQ + QGG original'''
    # src_data = load_json("data/input/cwq/cwq_test_0_1000_linking.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # golden_dataset = DATASET.CWQ
    # predicted_dataset = DATASET.QGG
    # detection_result = load_json("data/output/qgg_original/cwq_test_0_1000/final_results.json")
    # output_path = "data/output/qgg_original/cwq_test_0_1000/final_results_edit_distance.json"

    '''CWQ + QGG-QC'''
    # src_data = load_json("data/input/WebQSP/webqsp_test_0_1000_linking.json")
    # logger = setup_custom_logger("data/metrics/new_new/webqsp/metric.log")
    # golden_dataset = DATASET.CWQ
    # predicted_dataset = DATASET.QGG
    # detection_result = load_json("data/metrics/new_new/webqsp/quad_golden.json")
    # output_path = "data/metrics/new_new/webqsp/quad_golden_ted.json"

    '''CWQ + LTQ-QC'''
    # src_data = load_json("data/input/cwq/cwq_test_0_1000_linking.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # golden_dataset = DATASET.CWQ
    # predicted_dataset = DATASET.LSQ
    # detection_result = load_json("data/output/lsq/cwq/final_results.json")
    # output_path = "data/output/lsq/cwq/final_results_edit_distance.json"

    '''CWQ + QueryAgent'''
    src_data = load_json("data/input/WebQSP/webqsp_test_0_1000_linking.json")
    logger = setup_custom_logger("data/test/metric.log")
    golden_dataset = DATASET.CWQ
    predicted_dataset = DATASET.SIMULATED_FREEBASE
    detection_result = load_json("data/metrics/new_new/webqsp/ablation/webqsp_False_True_test_suite.json")
    output_path = "data/metrics/new_new/webqsp/ablation/webqsp_False_True_test_suite_ted.json"

    '''GrailQA'''
    # src_data = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_linking.json")
    # logger = setup_custom_logger("data/metrics/metric.log")
    # golden_dataset = DATASET.GRAIL
    # predicted_dataset = DATASET.SIMULATED_FREEBASE
    # # # detection_result = load_json("data/output/query_construction/grailqa/for_comparison/grailqa_dev_0_1000_starQC_comparison_2024-06-06_12:27:17/final_results.json")
    # # # output_path = "data/output/query_construction/grailqa/for_comparison/grailqa_dev_0_1000_starQC_comparison_2024-06-06_12:27:17/final_results_edit_distance.json"

    # # # detection_result = load_json("data/output/query_construction/grailqa/for_comparison/grailqa_dev_baseline_comparison_2024-06-05_20:32:09/final_results.json")
    # # # output_path = "data/output/query_construction/grailqa/for_comparison/grailqa_dev_baseline_comparison_2024-06-05_20:32:09/final_results_edit_distance.json"

    # # # detection_result = load_json("data/output/query_construction/grailqa/for_comparison/grailqa_dev_0_1000_starQC_linking_comparison_2024-06-09_22:47:18/final_results.json")
    # # # output_path = "data/output/query_construction/grailqa/for_comparison/grailqa_dev_0_1000_starQC_linking_comparison_2024-06-09_22:47:18/final_results_edit_distance.json"

    # detection_result = load_json("data/metrics/LRG/lfg_grailqa_link.json")
    # output_path = "data/metrics/LRG/lfg_grailqa_link_ted.json"

    '''GrailQA + KB-Binder'''
    # src_data = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_linking.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # golden_dataset = DATASET.GRAIL
    # predicted_dataset = DATASET.BINDER
    # detection_result = load_json("data/output/kb_binder/grailqa_dev_0_1000/final_results.json")
    # output_path = "data/output/kb_binder/grailqa_dev_0_1000/final_results_edit_distance.json"
    
    '''GrailQA + QGG-QC'''
    # src_data = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_linking.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # golden_dataset = DATASET.GRAIL
    # predicted_dataset = DATASET.QGG
    # detection_result = load_json("data/output/qgg_qc/grailqa/final_results.json")
    # output_path = "data/output/qgg_qc/grailqa/final_results_edit_distance.json"

    '''GrailQA + LSQ-QC'''
    # src_data = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_linking.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # golden_dataset = DATASET.GRAIL
    # predicted_dataset = DATASET.LSQ
    # detection_result = load_json("data/output/lsq/grailqa/final_results.json")
    # output_path = "data/output/lsq/grailqa/final_results_edit_distance.json"

    '''GrailQA + QueryAgent'''
    # src_data = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_linking.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # golden_dataset = DATASET.GRAIL
    # predicted_dataset = DATASET.QUERYAGENT
    # detection_result = load_json("data/output/queryagent/grailqa_dev_0_1000/final_results.json")
    # output_path = "data/output/queryagent/grailqa_dev_0_1000/final_results_edit_distance.json"

    qid_to_src_example = {
        str(example["qid"]): example
        for example in src_data
    }
    edit_distance_list = list()

    # for detection_item in tqdm(detection_result["results"]): # 构造查询失败，编辑距离记为 1.0
    for qid,detection_item in tqdm(detection_result.items()):
        
        if 'qid' not in detection_item:
            continue
        # qid = detection_item['qid']
        try:
            # if ("searched_queries" not in detection_item) or len(detection_item["searched_queries"]) == 0:
            if not detection_item['search_results']:
                edit_distance_list.append(1.0)
                detection_item["TED"] = 1.0
            else:
                # golden_sparql = qid_to_src_example[detection_item['qid']]["golden_sparql_query"]
                # golden_sparql = qid_to_src_example[qid]["golden_sparql_query"]
                # if predicted_dataset in [DATASET.SIMULATED_FREEBASE]:
                #     predicted_sexp = detection_item["searched_queries"][0]["s_expression"]
                #     predicted_sparql = sexp_to_sparql_for_edit_distance(predicted_sexp)
                # elif predicted_dataset in [DATASET.QGG, DATASET.LSQ, DATASET.QUERYAGENT, DATASET.BINDER]:
                #     predicted_sparql = detection_item['searched_queries'][0]['sparql']
                # else:
                #     raise NotImplementedError(f"dataset: {predicted_dataset}")
                ted_list = []
                # detect_data =   detection_item['search_results']

                # if "gpt_select_top1" in detection_item.keys():
                #     sparql_group = [detection_item['gpt_select_top1']]
                # else:
                sparql_group = [r["sparql"] for r in detection_item['search_results']]

                for idx,predicted_result in enumerate(sparql_group):
                    golden_sparql = qid_to_src_example[qid]["golden_sparql_query"]

                    # predicted_sparql = sexp_to_sparql_for_edit_distance(predicted_result)
                    predicted_sparql =  'PREFIX ns: <http://rdf.freebase.com/ns/> '+ predicted_result

                    # 前处理，去掉filter != 
                    filter_pattern = r'FILTER\([^(]*!=[^(]*\).'
                    predicted_sparql = re.sub(filter_pattern,'',predicted_sparql)
                    
                    editor = SyntaxTreeEditor(golden_sparql, golden_dataset, logger)
                    editor.preprocess_edit_distance(f"?{TARGET_VARIABLE_NAME}")
                    golden_sparql = editor.sparql_txt

                    editor = SyntaxTreeEditor(predicted_sparql, predicted_dataset, logger)
                    editor.preprocess_edit_distance(f"?{TARGET_VARIABLE_NAME}")
                    predicted_sparql = editor.sparql_txt

                    golden_tree_constructor = TreeConstructor(
                        golden_sparql, golden_dataset, logger
                    )
                    golden_tree_root = golden_tree_constructor.construct_syntax_tree()
                    golden_tree_constructor.canonicalize_syntax_tree(root_node=golden_tree_root)

                    predicted_tree_constructor = TreeConstructor(
                        predicted_sparql, predicted_dataset, logger
                    )
                    predicted_tree_root = predicted_tree_constructor.construct_syntax_tree()
                    predicted_tree_constructor.canonicalize_syntax_tree(root_node=predicted_tree_root)

                    _edit_distance = TreeEditDistance.get_edit_distance(predicted_tree_root, golden_tree_root)
                    #  每个候选的ted存起来
                    ted_list.append(_edit_distance)

                min_edit_distance = min(ted_list)
                detection_item["TED"] = min_edit_distance
                detection_item["TED_list"] = ted_list
                edit_distance_list.append(min_edit_distance)

        except Exception as e:
            logger.error(f"qid: {qid}; error: {e}")
            detection_item["TED"] = 1.0
            edit_distance_list.append(1.0)
    
    average = sum(edit_distance_list) / len(src_data)
    isomorphism_rate = len([_item for _item in edit_distance_list if math.isclose(_item, 0.0)]) / len(src_data)
    detection_result["tree_edit_distance_summary"]= {
        "average": average,
        "isomorphism_rate": isomorphism_rate
    }
    print(f"predicted_dataset: {predicted_dataset}\nsummary: {detection_result['tree_edit_distance_summary']}")
    dump_json(detection_result, output_path)

def add_edit_distance_webqsp():
    """
    WebQSP 对于每个样本提供了多个 golden sparql, 应该对于各 golden sparql 取最好的指标
    """
    '''Simulated Query'''
    src_data = load_json("data/input/WebQSP/webqsp_test_0_1000_linking.json")
    logger = setup_custom_logger("data/test/metric.log")
    golden_dataset = DATASET.WEBQ
    predicted_dataset = DATASET.SIMULATED_FREEBASE
    # detection_result = load_json("data/output/query_construction/webqsp/for_comparison/webqsp_test_0_1000_starQC_2024-06-05_09:13:29/final_results.json")
    # output_path = "data/output/query_construction/webqsp/for_comparison/webqsp_test_0_1000_starQC_2024-06-05_09:13:29/final_results_edit_distance.json"

    # detection_result = load_json("data/output/query_construction/webqsp/for_comparison/webqsp_test_0_1000_baseline_2024-06-05_14:56:20/final_results.json")
    # output_path = "data/output/query_construction/webqsp/for_comparison/webqsp_test_0_1000_baseline_2024-06-05_14:56:20/final_results_edit_distance.json"

    # detection_result = load_json("data/output/query_construction/webqsp/for_comparison/webqsp_test_0_1000_starQC_linking_2024-06-05_11:53:21/final_results.json")
    # output_path = "data/output/query_construction/webqsp/for_comparison/webqsp_test_0_1000_starQC_linking_2024-06-05_11:53:21/final_results_edit_distance.json"
    
    detection_result = load_json("data/output/query_construction/webqsp/for_comparison/webqsp_test_0_1000_starQC_queryagentLinking_2024-06-13_09:04:37/final_results.json")
    output_path = "data/output/query_construction/webqsp/for_comparison/webqsp_test_0_1000_starQC_queryagentLinking_2024-06-13_09:04:37/final_results_edit_distance.json"

    '''QueryAgent'''
    # src_data = load_json("data/input/WebQSP/webqsp_test_0_1000_linking.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # golden_dataset = DATASET.WEBQ
    # predicted_dataset = DATASET.QUERYAGENT
    # detection_result = load_json("data/output/queryagent/webqsp_test_0_1000/final_results.json")
    # output_path = "data/output/queryagent/webqsp_test_0_1000/final_results_edit_distance.json"

    '''QGG-QC'''
    # src_data = load_json("data/input/WebQSP/webqsp_test_0_1000_linking.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # golden_dataset = DATASET.WEBQ
    # predicted_dataset = DATASET.QGG
    # detection_result = load_json("data/output/qgg_qc/webqsp/final_results.json")
    # output_path = "data/output/qgg_qc/webqsp/final_results_edit_distance.json"

    '''QGG'''
    # src_data = load_json("data/input/WebQSP/webqsp_test_0_1000_linking.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # golden_dataset = DATASET.WEBQ
    # predicted_dataset = DATASET.QGG
    # detection_result = load_json("data/output/qgg_original/webqsp_test_0_1000/final_results.json")
    # output_path = "data/output/qgg_original/webqsp_test_0_1000/final_results_edit_distance.json"

    '''LSQ-QC'''
    # src_data = load_json("data/input/WebQSP/webqsp_test_0_1000_linking.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # golden_dataset = DATASET.WEBQ
    # predicted_dataset = DATASET.LSQ
    # detection_result = load_json("data/output/lsq/webqsp/final_results.json")
    # output_path = "data/output/lsq/webqsp/final_results_edit_distance.json"

    '''KB-Binder'''
    # src_data = load_json("data/input/WebQSP/webqsp_test_0_1000_linking.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # golden_dataset = DATASET.WEBQ
    # predicted_dataset = DATASET.BINDER
    # detection_result = load_json("data/output/kb_binder/webqsp_test_0_1000/final_results.json")
    # output_path = "data/output/kb_binder/webqsp_test_0_1000/final_results_edit_distance.json"

    qid_to_src_example = {
        example["qid"]: example
        for example in src_data
    }
    edit_distance_list = list()

    for detection_item in tqdm(detection_result["results"]): # 构造查询失败，编辑距离记为 1.0
        try:
            if ("searched_queries" not in detection_item) or len(detection_item["searched_queries"]) == 0:
                edit_distance_list.append(1.0)
                detection_item["TED"] = 1.0
            else:
                min_edit_distance = 1.0

                if predicted_dataset in [DATASET.SIMULATED_FREEBASE]:
                    predicted_sexp = detection_item["searched_queries"][0]["s_expression"]
                    predicted_sparql = sexp_to_sparql_for_edit_distance(predicted_sexp)
                elif predicted_dataset in [DATASET.QGG, DATASET.LSQ, DATASET.QUERYAGENT, DATASET.BINDER]:
                    predicted_sparql = detection_item['searched_queries'][0]['sparql']
                else:
                    raise NotImplementedError(f"dataset: {predicted_dataset}")

                editor = SyntaxTreeEditor(predicted_sparql, predicted_dataset, logger)
                editor.preprocess_edit_distance(f"?{TARGET_VARIABLE_NAME}")
                predicted_sparql = editor.sparql_txt
                predicted_tree_constructor = TreeConstructor(
                    predicted_sparql, predicted_dataset, logger
                )
                predicted_tree_root = predicted_tree_constructor.construct_syntax_tree()
                predicted_tree_constructor.canonicalize_syntax_tree(root_node=predicted_tree_root)

                for golden_sparql in qid_to_src_example[detection_item['qid']]["golden_sparql_list"]:
                    golden_sparql = golden_sparql["sparql"]
                    editor = SyntaxTreeEditor(golden_sparql, golden_dataset, logger)
                    editor.preprocess_edit_distance(f"?{TARGET_VARIABLE_NAME}")
                    golden_sparql = editor.sparql_txt
                    golden_tree_constructor = TreeConstructor(
                        golden_sparql, golden_dataset, logger
                    )
                    golden_tree_root = golden_tree_constructor.construct_syntax_tree()
                    golden_tree_constructor.canonicalize_syntax_tree(root_node=golden_tree_root)

                    _edit_distance = TreeEditDistance.get_edit_distance(predicted_tree_root, golden_tree_root)
                    min_edit_distance = min(_edit_distance, min_edit_distance)
                
                detection_item["TED"] = min_edit_distance
                edit_distance_list.append(min_edit_distance)
        except Exception as e:
            logger.error(f"qid: {detection_item['qid']}; error: {e}")
            detection_item["TED"] = 1.0
            edit_distance_list.append(1.0)
    
    average = sum(edit_distance_list) / len(src_data)
    isomorphism_rate = len([_item for _item in edit_distance_list if math.isclose(_item, 0.0)]) / len(src_data)
    detection_result["summary"]["tree_edit_distance"] = {
        "average": average,
        "isomorphism_rate": isomorphism_rate
    }
    dump_json(detection_result, output_path)
    print(f"dataset: {predicted_dataset}\n summary {detection_result['summary']}")


def ours_add_edit_distance_wikidata_qa():
    src_data = load_json("data/annotated_test/annotated_dataset.json")
    logger = setup_custom_logger("data/test/metric.log")
    golden_dataset = DATASET.LC2
    predicted_dataset = DATASET.SIMULATED_WIKIDATA
    detection_result = load_json("data/final_test/queryagent/queryagent_golden.json")
    output_path = "data/final_test/queryagent/golden_results_edit_distance_nothing.json"

    qid_to_src_example = {
        str(example["qid"]): example
        for example in src_data
    }
    edit_distance_list = list()
    for detection_item in tqdm(detection_result): # 构造查询失败，编辑距离记为 1.0
        try:
            if ("gen_sparql" not in detection_item) or len(detection_item["gen_sparql"]) == 0:
                edit_distance_list.append(1.0)
                detection_item["TED"] = 1.0
            else:
                golden_sparql = qid_to_src_example[str(detection_item['qid'])]["golden_sparql"]
                # 特别之处: Wikidata 的 golden sparql 补充一下前缀，例如 LC2 中就缺少前缀
                golden_sparql = WIKIDATA_PREFIX_LIST + "\n" + golden_sparql
                predicted_sparql = WIKIDATA_PREFIX_LIST+detection_item['gen_sparql']
                # 前处理，去掉filter != 
                # filter_pattern = r'FILTER\([^(]*!=[^(]*\).'
                # predicted_sparql = re.sub(filter_pattern,'',predicted_sparql)
                # # wdt替换为p:/ps:
                # relation_path_pattern = r'wdt:P\d+'
                # relation_list = list(set(re.findall(relation_path_pattern,predicted_sparql)))
                # for relation in relation_list:
                #     rid = re.findall(r'P\d+',relation)[0]
                #     predicted_sparql = predicted_sparql.replace(relation,f'p:{rid}/ps:{rid}')
                # editor = SyntaxTreeEditor(golden_sparql, golden_dataset, logger)
                # editor.preprocess_edit_distance(f"?{TARGET_VARIABLE_NAME}")
                # golden_sparql = editor.sparql_txt
                # editor = SyntaxTreeEditor(predicted_sparql, predicted_dataset, logger)
                # editor.preprocess_edit_distance(f"?{TARGET_VARIABLE_NAME}")
                # predicted_sparql = editor.sparql_txt
                golden_tree_constructor = TreeConstructor(
                    golden_sparql, golden_dataset, logger
                )
                golden_tree_root = golden_tree_constructor.construct_syntax_tree()
                # golden_tree_constructor.canonicalize_syntax_tree(root_node=golden_tree_root)

                predicted_tree_constructor = TreeConstructor(
                    predicted_sparql, predicted_dataset, logger
                )
                predicted_tree_root = predicted_tree_constructor.construct_syntax_tree()
                # predicted_tree_constructor.canonicalize_syntax_tree(root_node=predicted_tree_root)

                _edit_distance = TreeEditDistance.get_edit_distance(predicted_tree_root, golden_tree_root)
                detection_item["TED"] = _edit_distance
                edit_distance_list.append(_edit_distance)
        except Exception as e:
            logger.error(f"qid: {detection_item['qid']}; error: {e}")
            detection_item["TED"] = 1.0
            edit_distance_list.append(1.0)
    
    average = sum(edit_distance_list) / len(src_data)
    isomorphism_rate = len([_item for _item in edit_distance_list if math.isclose(_item, 0.0)]) / len(src_data)
    detection_result.append({
        "average": average,
        "isomorphism_rate": isomorphism_rate
    })  
    dump_json(detection_result, output_path)


def ours_add_edit_distance_wikidata():
    src_data = load_json("data/annotated_test/annotated_dataset.json")
    logger = setup_custom_logger("data/test/metric.log")
    golden_dataset = DATASET.LC2
    predicted_dataset = DATASET.SIMULATED_WIKIDATA
    detection_result = load_json("data/final_test/quad/quad_golden.json")
    output_path = "data/final_test/quad/golden_results_edit_distance_first_nothing.json"

    qid_to_src_example = {
        str(example["qid"]): example
        for example in src_data
    }
    edit_distance_list = list()

    for idx,detection_item in tqdm(detection_result.items()): # 构造查询失败，编辑距离记为 1.0
        try:
            if ("search_results" not in detection_item) or len(detection_item["search_results"]) == 0:
                edit_distance_list.append(1.0)
                detection_item["TED"] = 1.0
            else:
                golden_sparql = qid_to_src_example[detection_item['qid']]["golden_sparql"]
                # 特别之处: Wikidata 的 golden sparql 补充一下前缀，例如 LC2 中就缺少前缀
                golden_sparql = WIKIDATA_PREFIX_LIST + "\n" + golden_sparql
                # if "gpt_select_top1" in detection_item and "sparql" in detection_item['gpt_select_top1']:
                #     predicted_sparql = WIKIDATA_PREFIX_LIST + "\n"+detection_item['gpt_select_top1']['sparql']
                # else:
                predicted_sparql = WIKIDATA_PREFIX_LIST + "\n"+detection_item["search_results"][0]["sparql"]
                # predicted_sparql = sexp_to_sparql_wikidata_for_edit_distance(predicted_sexp) # 这个函数中，完成了前缀的添加
                # 前处理，去掉filter != 

                # filter_pattern = r'FILTER \([^:(]*!=[^:(]*\).'
                # predicted_sparql = re.sub(filter_pattern,'',predicted_sparql)
                # relation_path_pattern = r'wdt:P\d+'
                # relation_list = list(set(re.findall(relation_path_pattern,predicted_sparql)))
                # for relation in relation_list:
                #     rid = re.findall(r'P\d+',relation)[0]
                #     predicted_sparql = predicted_sparql.replace(relation,f'p:{rid}/ps:{rid}')

                editor = SyntaxTreeEditor(golden_sparql, golden_dataset, logger)
                editor.preprocess_edit_distance(f"?{TARGET_VARIABLE_NAME}")
                golden_sparql = editor.sparql_txt
                editor = SyntaxTreeEditor(predicted_sparql, predicted_dataset, logger)
                editor.preprocess_edit_distance(f"?{TARGET_VARIABLE_NAME}")
                predicted_sparql = editor.sparql_txt

                golden_tree_constructor = TreeConstructor(
                    golden_sparql, golden_dataset, logger
                )
                golden_tree_root = golden_tree_constructor.construct_syntax_tree()
                # golden_tree_constructor.canonicalize_syntax_tree(root_node=golden_tree_root)

                predicted_tree_constructor = TreeConstructor(
                    predicted_sparql, predicted_dataset, logger
                )
                predicted_tree_root = predicted_tree_constructor.construct_syntax_tree()
                # predicted_tree_constructor.canonicalize_syntax_tree(root_node=predicted_tree_root)

                _edit_distance = TreeEditDistance.get_edit_distance(predicted_tree_root, golden_tree_root)
                detection_item["TED"] = _edit_distance
                edit_distance_list.append(_edit_distance)
        except Exception as e:
            logger.error(f"qid: {detection_item['qid']}; error: {e}")
            detection_item["TED"] = 1.0
            edit_distance_list.append(1.0)
    
    average = sum(edit_distance_list) / len(src_data)
    isomorphism_rate = len([_item for _item in edit_distance_list if math.isclose(_item, 0.0)]) / len(src_data)
    detection_result["tree_edit_distance"] = {
        "average": average,
        "isomorphism_rate": isomorphism_rate
    }
    dump_json(detection_result, output_path)


def add_edit_distance_wikidata():
    '''LC2'''
    # src_data = load_json("data/input/LC_QuAD_2/lc2_test_0_1000.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # golden_dataset = DATASET.LC2
    # predicted_dataset = DATASET.SIMULATED_WIKIDATA
    # detection_result = load_json("data/output/query_construction/lc2/for_comparison/lc2_test_0_1000_baseline_comparison_2024-06-10_15:40:59/final_results.json")
    # output_path = "data/output/query_construction/lc2/for_comparison/lc2_test_0_1000_baseline_comparison_2024-06-10_15:40:59/final_results_edit_distance.json"

    '''QALD'''
    # src_data = load_json("data/input/QALD/qald_test.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # golden_dataset = DATASET.QALD
    # predicted_dataset = DATASET.SIMULATED_WIKIDATA
    # detection_result = load_json("data/output/query_construction/qald/for_comparison/qald_baseline_comparison_2024-06-10_15:04:20/final_results.json")
    # output_path = "data/output/query_construction/qald/for_comparison/qald_baseline_comparison_2024-06-10_15:04:20/final_results_edit_distance.json"
    '''annotated'''
    src_data = load_json("data/annotated_test/annotated_dataset.json")
    logger = setup_custom_logger("data/test/metric.log")
    golden_dataset = DATASET.LC2
    predicted_dataset = DATASET.SIMULATED_WIKIDATA
    detection_result = load_json("data/final_test/quad/quad_linked.json")
    output_path = "data/final_test/quad/final_results_edit_distance.json"

    qid_to_src_example = {
        example["qid"]: example
        for example in src_data
    }
    edit_distance_list = list()

    for detection_item in tqdm(detection_result["results"]): # 构造查询失败，编辑距离记为 1.0
        try:
            if ("searched_queries" not in detection_item) or len(detection_item["searched_queries"]) == 0:
                edit_distance_list.append(1.0)
                detection_item["TED"] = 1.0
            else:
                golden_sparql = qid_to_src_example[detection_item['qid']]["golden_sparql"]
                # 特别之处: Wikidata 的 golden sparql 补充一下前缀，例如 LC2 中就缺少前缀
                golden_sparql = WIKIDATA_PREFIX_LIST + "\n" + golden_sparql
                predicted_sexp = detection_item["searched_queries"][0]["s_expression"]
                predicted_sparql = sexp_to_sparql_wikidata_for_edit_distance(predicted_sexp) # 这个函数中，完成了前缀的添加

                editor = SyntaxTreeEditor(golden_sparql, golden_dataset, logger)
                editor.preprocess_edit_distance(f"?{TARGET_VARIABLE_NAME}")
                golden_sparql = editor.sparql_txt

                editor = SyntaxTreeEditor(predicted_sparql, predicted_dataset, logger)
                editor.preprocess_edit_distance(f"?{TARGET_VARIABLE_NAME}")
                predicted_sparql = editor.sparql_txt

                golden_tree_constructor = TreeConstructor(
                    golden_sparql, golden_dataset, logger
                )
                golden_tree_root = golden_tree_constructor.construct_syntax_tree()
                golden_tree_constructor.canonicalize_syntax_tree(root_node=golden_tree_root)

                predicted_tree_constructor = TreeConstructor(
                    predicted_sparql, predicted_dataset, logger
                )
                predicted_tree_root = predicted_tree_constructor.construct_syntax_tree()
                predicted_tree_constructor.canonicalize_syntax_tree(root_node=predicted_tree_root)

                _edit_distance = TreeEditDistance.get_edit_distance(predicted_tree_root, golden_tree_root)
                detection_item["TED"] = _edit_distance
                edit_distance_list.append(_edit_distance)
        except Exception as e:
            logger.error(f"qid: {detection_item['qid']}; error: {e}")
            detection_item["TED"] = 1.0
            edit_distance_list.append(1.0)
    
    average = sum(edit_distance_list) / len(src_data)
    isomorphism_rate = len([_item for _item in edit_distance_list if math.isclose(_item, 0.0)]) / len(src_data)
    detection_result["summary"]["tree_edit_distance"] = {
        "average": average,
        "isomorphism_rate": isomorphism_rate
    }
    dump_json(detection_result, output_path)

def debug_edit_distance_wikidata():
    '''LC2'''
    src_data = load_json("data/input/LC_QuAD_2/lc2_test_0_1000.json")
    logger = setup_custom_logger("data/test/metric.log")
    dataset = DATASET.LC2
    detection_result = load_json("data/output/query_construction/lc2/lc2_test_0_1000_starQC_2024-05-26_19:54:37/final_results.json")

    qid_to_src_example = {
        example["qid"]: example
        for example in src_data
    }
    edit_distance_list = list()

    for detection_item in tqdm(detection_result["results"][755:756]): # 构造查询失败，编辑距离记为 1.0
        try:
            if ("searched_queries" not in detection_item) or len(detection_item["searched_queries"]) == 0:
                edit_distance_list.append(1.0)
                detection_item["TED"] = 1.0
            else:
                golden_sparql = qid_to_src_example[detection_item['qid']]["golden_sparql_query"]
                # 特别之处: Wikidata 的 golden sparql 补充一下前缀，例如 LC2 中就缺少前缀
                golden_sparql = WIKIDATA_PREFIX_LIST + "\n" + golden_sparql
                predicted_sexp = detection_item["searched_queries"][0]["s_expression"]
                predicted_sparql = sexp_to_sparql_wikidata_for_edit_distance(predicted_sexp) # 这个函数中，完成了前缀的添加

                editor = SyntaxTreeEditor(golden_sparql, dataset, logger)
                editor.preprocess_edit_distance(f"?{TARGET_VARIABLE_NAME}")
                golden_sparql = editor.sparql_txt

                editor = SyntaxTreeEditor(predicted_sparql, DATASET.SIMULATED_WIKIDATA, logger)
                editor.preprocess_edit_distance(f"?{TARGET_VARIABLE_NAME}")
                predicted_sparql = editor.sparql_txt

                golden_tree_constructor = TreeConstructor(
                    golden_sparql, dataset, logger
                )
                golden_tree_root = golden_tree_constructor.construct_syntax_tree()
                golden_tree_constructor.canonicalize_syntax_tree(root_node=golden_tree_root)
                golden_tree_root.visualize()

                predicted_tree_constructor = TreeConstructor(
                    predicted_sparql, DATASET.SIMULATED_WIKIDATA, logger
                )
                predicted_tree_root = predicted_tree_constructor.construct_syntax_tree()
                predicted_tree_constructor.canonicalize_syntax_tree(root_node=predicted_tree_root)
                predicted_tree_root.visualize()

                _edit_distance = TreeEditDistance.get_edit_distance(predicted_tree_root, golden_tree_root)
                detection_item["TED"] = _edit_distance
                edit_distance_list.append(_edit_distance)
        except Exception as e:
            logger.error(f"qid: {detection_item['qid']}; error: {e}")
            detection_item["TED"] = 1.0
            edit_distance_list.append(1.0)
    
    detection_result["summary"]["average_edit_distance"] = sum(edit_distance_list) / len(edit_distance_list)


def debug_edit_distance():
    '''CWQ'''
    src_data = load_json("data/input/cwq/cwq_test_0_1000_linking.json")
    logger = setup_custom_logger("data/test/metric.log")
    dataset = DATASET.CWQ
    detection_result = load_json("data/output/query_construction/cwq/cwq_test_0_1000_starQC_2024-05-20_17:25:43/final_results.json")

    '''GrailQA'''
    # src_data = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_linking.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # dataset = DATASET.GRAIL
    # detection_result = load_json("data/output/query_construction/grailqa/grailqa_dev_0_1000_starQC_2024-05-19_09:12:50/final_results.json")

    '''QALD'''
    # src_data = load_json("data/input/QALD/qald_test_test_suite.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # dataset = DATASET.QALD
    # detection_result = load_json("data/output/query_construction/qald/qald_starQC_2024-05-29_20:22:12/final_results.json")

    '''WebQSPs'''
    # src_data = load_json("data/input/WebQSP/webqsp_test_0_1000_test_suite.json")
    # logger = setup_custom_logger("data/test/metric.log")
    # dataset = DATASET.WEBQ
    # detection_result = load_json("data/output/query_construction/webqsp/webqsp_test_0_1000_starQC_2024-05-23_09:45:54/final_results.json")

    qid_to_src_example = {
        example["qid"]: example
        for example in src_data
    }
    edit_distance_list = list()

    for detection_item in detection_result["results"][313:314]: # 构造查询失败，编辑距离记为 1.0
        # if detection_item['qid'] not in {'WebQTest-538_226772fce4f7a1408c721880a97111cf', 'WebQTest-1528_7fa8488dddb1d79e22bcd8dd3618a772', 'WebQTrn-2006_8380457ea59739a01481e4908013c943', 'WebQTrn-493_cf7833e237b8cda1de396587129515c1', 'WebQTest-213_1bb5d148e28985730d4603ae86c12405', 'WebQTrn-1841_a5bd6833f38c9345eb2286b787986daa', 'WebQTest-837_52a09da805a8b454de1e3b5edd88fd71', 'WebQTrn-2047_3d5f855275265f74d81b862231578e33', 'WebQTrn-303_770773a5150d1cdd3cfadfc25022720b', 'WebQTrn-1677_07b09c54d5283ed75072069f25c48469', 'WebQTrn-2653_d2311125d5969172b311bdb814c7aae4', 'WebQTest-361_5f8c05737061d5501b6a23a978b5589b', 'WebQTest-1560_3f9e3373173bba6e70a26c4df64aa943', 'WebQTest-213_e8bb0c0bb8606f69379d8f300258b1c1', 'WebQTrn-124_b74a21d2b3530ff1fc1cedc0ef485e85', 'WebQTrn-1294_5a3dcb96932c06c8362a3d44c2181b93', 'WebQTest-1012_73741811f34519b29f7d19ccfd4d9553', 'WebQTest-1560_c1c1d3141d16ef82b72cf09954be0346', 'WebQTrn-2152_1694f4392eb0ff79149337ee3f622d91', 'WebQTrn-2006_75407e92f7bc865e5dcca4087fac14e8', 'WebQTest-538_ebdaf3c91b43a826ed6ecc1221fd9af5', 'WebQTrn-837_66af81787994177f0d3e64d0c0fb871f', 'WebQTrn-465_e9585b7d3117fbe09c4ed03353acee7b'}:
        #     continue
        # try:
        if ("searched_queries" not in detection_item) or len(detection_item["searched_queries"]) == 0:
            edit_distance_list.append(1.0)
            detection_item["TED"] = 1.0
        else:
            golden_sparql = qid_to_src_example[detection_item['qid']]["golden_sparql_query"]
            predicted_sexp = detection_item["searched_queries"][0]["s_expression"]
            predicted_sparql = sexp_to_sparql_for_edit_distance(predicted_sexp)
            # predicted_sparql = sexp_to_sparql_wikidata_for_edit_distance(predicted_sexp)

            editor = SyntaxTreeEditor(golden_sparql, dataset, logger)
            editor.preprocess_edit_distance(f"?{TARGET_VARIABLE_NAME}")
            golden_sparql = editor.sparql_txt

            editor = SyntaxTreeEditor(predicted_sparql, DATASET.SIMULATED_FREEBASE, logger)
            # editor = SyntaxTreeEditor(predicted_sparql, DATASET.SIMULATED_WIKIDATA, logger)
            editor.preprocess_edit_distance(f"?{TARGET_VARIABLE_NAME}")
            predicted_sparql = editor.sparql_txt

            golden_tree_constructor = TreeConstructor(
                golden_sparql, dataset, logger
            )
            golden_tree_root = golden_tree_constructor.construct_syntax_tree()
            golden_tree_constructor.canonicalize_syntax_tree(root_node=golden_tree_root)
            golden_tree_root.visualize()

            predicted_tree_constructor = TreeConstructor(
                predicted_sparql, DATASET.SIMULATED_FREEBASE, logger
            )
            # predicted_tree_constructor = TreeConstructor(
            #     predicted_sparql, DATASET.SIMULATED_WIKIDATA, logger
            # )
            predicted_tree_root = predicted_tree_constructor.construct_syntax_tree()
            predicted_tree_constructor.canonicalize_syntax_tree(root_node=predicted_tree_root)
            predicted_tree_root.visualize()

            _edit_distance = TreeEditDistance.get_edit_distance(predicted_tree_root, golden_tree_root)
            print(f"qid: {detection_item['qid']}; _edit_distance: {_edit_distance}")
            detection_item["TED"] = _edit_distance
            edit_distance_list.append(_edit_distance)
        # except Exception as e:
        #     logger.error(f"qid: {detection_item['qid']}; error: {e}")
        #     detection_item["TED"] = 1.0
        #     edit_distance_list.append(1.0)
    
    # detection_result["summary"]["average_edit_distance"] = sum(edit_distance_list) / len(edit_distance_list)
    # dump_json(detection_result, output_path)


def add_graph_edit_distance():
    from components.graph_utils import GraphEquivalenceUtil
    logger = setup_custom_logger("data/test/log.txt")
    graph_util = GraphEquivalenceUtil(logger)
    # 和之前图编辑距离的计算结果，做个对比
    '''GrailQA + Simulated'''
    src_data = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_linking.json")
    detection_result = load_json("data/output/query_construction/grailqa/for_comparison/grailqa_dev_0_1000_starQC_comparison_2024-06-06_12:27:17/final_results.json")
    output_path = "data/output/query_construction/grailqa/for_comparison/grailqa_dev_0_1000_starQC_comparison_2024-06-06_12:27:17/final_results_graph_edit_distance.json"

    # '''CWQ + QGG'''
    # src_data = load_json("data/input/cwq/cwq_test_0_1000_linking.json")
    # detection_result = load_json("data/output/qgg_original/cwq_test_0_1000/final_results.json")
    # output_path = "data/output/qgg_original/cwq_test_0_1000/final_results_graph_edit_distance.json"

    '''CWQ'''
    # src_data = load_json("data/input/cwq/cwq_test_0_1000_linking.json")
    # detection_result = load_json("data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_comparison_2024-06-04_21:14:58/final_results.json")
    # output_path = "data/output/query_construction/cwq/for_comparison/cwq_test_0_1000_starQC_comparison_2024-06-04_21:14:58/final_results_graph_edit_distance.json"

    qid_to_golden_sexp = {
        example["qid"]: example["golden_s_expression"]
        for example in src_data
    }
    edit_distance_list = list()

    for detection_item in tqdm(detection_result["results"]):
        golden_query = qid_to_golden_sexp[detection_item["qid"]]
        if ("searched_queries" not in detection_item) or len(detection_item["searched_queries"]) == 0: # 构造查询失败，编辑距离记为 1.0
            edit_distance_list.append(1.0)
        else:
            # 目前的实验设置，仅考虑 top1 查询
            simulated_query = detection_item["searched_queries"][0]["s_expression"]
            normalized_ged = graph_util.calc_edit_distance(simulated_query, golden_query)
            if math.isclose(normalized_ged, 0.0):
                detection_item["searched_queries"][0]["semantic_equivalence"] = True
            edit_distance_list.append(normalized_ged)
            detection_item["searched_queries"][0]["normalized_ged"] = normalized_ged
    dump_json(detection_result, output_path)

def add_graph_edit_distance_wikidata():
    from components.graph_utils import GraphEquivalenceUtilWikidata
    logger = setup_custom_logger("data/test/log.txt")
    graph_util = GraphEquivalenceUtilWikidata(logger)
    # 和之前图编辑距离的计算结果，做个对比
    '''LC2'''
    # src_data = load_json("data/input/LC_QuAD_2/lc2_test_0_1000.json")
    # detection_result = load_json("data/output/query_construction/lc2/lc2_test_0_1000_starQC_2024-05-26_19:54:37/final_results.json")
    # output_path = "data/output/query_construction/lc2/lc2_test_0_1000_starQC_2024-05-26_19:54:37/final_results_graph_edit_distance.json"

    '''QALD'''
    src_data = load_json("data/input/QALD/qald_test.json")
    detection_result = load_json("data/output/query_construction/qald/qald_starQC_2024-05-29_20:22:12/final_results.json")
    output_path = "data/output/query_construction/qald/qald_starQC_2024-05-29_20:22:12/final_results_graph_edit_distance.json"
    
    qid_to_golden_sexp = {
        example["qid"]: example["golden_s_expression"]
        for example in src_data
    }
    edit_distance_list = list()

    for detection_item in tqdm(detection_result["results"]):
        golden_query = qid_to_golden_sexp[detection_item["qid"]]
        if ("searched_queries" not in detection_item) or len(detection_item["searched_queries"]) == 0: # 构造查询失败，编辑距离记为 1.0
            edit_distance_list.append(1.0)
        else:
            # 目前的实验设置，仅考虑 top1 查询
            simulated_query = detection_item["searched_queries"][0]["s_expression"]
            normalized_ged = graph_util.calc_edit_distance(simulated_query, golden_query)
            if math.isclose(normalized_ged, 0.0):
                detection_item["searched_queries"][0]["semantic_equivalence"] = True
            edit_distance_list.append(normalized_ged)
            detection_item["searched_queries"][0]["normalized_ged"] = normalized_ged
    dump_json(detection_result, output_path)


def add_graph_edit_distance_webqsp():
    from components.graph_utils import GraphEquivalenceUtil
    logger = setup_custom_logger("data/test/log.txt")
    graph_util = GraphEquivalenceUtil(logger)
    # 和之前图编辑距离的计算结果，做个对比
    src_data = load_json("data/input/WebQSP/webqsp_test_0_1000_linking.json")
    detection_result = load_json("data/output/query_construction/webqsp/for_comparison/webqsp_test_0_1000_starQC_2024-06-05_09:13:29/final_results.json")
    qid_to_golden_sexp_list = {
        example["qid"]: example["golden_s_expression_list"]
        for example in src_data
    }
    edit_distance_list = list()

    for detection_item in tqdm(detection_result["results"]):
        golden_query_list = qid_to_golden_sexp_list[detection_item["qid"]]
        if ("searched_queries" not in detection_item) or len(detection_item["searched_queries"]) == 0: # 构造查询失败，编辑距离记为 1.0
            edit_distance_list.append(1.0)
        else:
            min_edit_distance = 1.0
            for golden_query in golden_query_list:
                simulated_query = detection_item["searched_queries"][0]["s_expression"]
                normalized_ged = graph_util.calc_edit_distance(simulated_query, golden_query)
                if math.isclose(normalized_ged, 0.0):
                    detection_item["searched_queries"][0]["semantic_equivalence"] = True
                min_edit_distance = min(normalized_ged, min_edit_distance)
            edit_distance_list.append(min_edit_distance)
            detection_item["searched_queries"][0]["normalized_ged"] = min_edit_distance
    dump_json(detection_result, "data/output/query_construction/webqsp/for_comparison/webqsp_test_0_1000_starQC_2024-06-05_09:13:29/final_results_graph_edit_distance.json")


def _calculate_test_suite_f1(simulated_sparql, test_suite_examples, sparql_querier:SparqlOdbcQuerier, retry=1):
    f1_list = list()
    for item_before_replace in test_suite_examples:
        test_cases = test_suite_examples[item_before_replace]["examples"]
        for (item_after_replace, answer_list) in test_cases.items():
            item_after_replace = f"{NS_PREFIX}{item_after_replace}" # TODO: 格式尽量统一
            sparql_editor = SyntaxTreeEditor(simulated_sparql)
            modified = sparql_editor.update_leaf_value(item_before_replace, item_after_replace)
            if modified: # Simulated SPARQL 中有这个 item, 则参与指标的计算
                sparql_after_replace = sparql_editor.get_sparql_text()
                execution_result = sparql_querier.get_execution_result_one_variable_sparql_wrapper(sparql_after_replace, retry)
                _, _, f1 = get_PRF1(execution_result, answer_list)
                f1_list.append(f1)
    if len(f1_list) == 0:
        # 没有测试用例，就返回 0.0
        return 0.0
    return sum(f1_list) / len(f1_list)

def _calculate_tree_edit_distance(simulated_sparql, golden_sparql, dataset:DATASET, logger):
    simulated_tree_constructor = TreeConstructor(
        simulated_sparql, dataset, logger
    )
    simulated_tree_root = simulated_tree_constructor.construct_syntax_tree()
    simulated_tree_constructor.canonicalize_syntax_tree(root_node=simulated_tree_root)
    # simulated_tree_root.visualize()
    
    golden_tree_constructor = TreeConstructor(
        golden_sparql, dataset, logger
    )
    golden_tree_root = golden_tree_constructor.construct_syntax_tree()
    golden_tree_constructor.canonicalize_syntax_tree(root_node=golden_tree_root)
    # golden_tree_root.visualize()

    edit_distance = TreeEditDistance.get_edit_distance(simulated_tree_root, golden_tree_root)
    return edit_distance

def diff_two_edit_distance(tree_path, graph_path):
    tree_edit_distance_data = load_json(tree_path)
    graph_edit_distance_data = load_json(graph_path)

    qid_to_tree = {
        item["qid"]: item for item in tree_edit_distance_data["results"]
    }
    qid_to_graph = {
        item["qid"]: item for item in graph_edit_distance_data["results"]
    }
    assert set(qid_to_tree.keys()) == set(qid_to_graph.keys())
    qid_set = set()
    for qid in qid_to_tree:
        tree_item = qid_to_tree[qid]
        graph_item = qid_to_graph[qid]
        # if "searched_queries" in graph_item and len(graph_item["searched_queries"]):
        #     if ("semantic_equivalence" in graph_item["searched_queries"][0]) and (graph_item["searched_queries"][0]["semantic_equivalence"] is True):
        #         if not math.isclose(tree_item["TED"], 0.0):
        #             qid_set.add(qid)
        
        if math.isclose(tree_item["TED"], 0.0):
            if "searched_queries" in graph_item and len(graph_item["searched_queries"]):
                if "semantic_equivalence" not in graph_item["searched_queries"][0]:
                    qid_set.add(qid)

    print(f"qid_set: {len(qid_set)}")
    print(qid_set)

def diff_tree_edit_distance(tree_path_1, tree_path_2):
    data_1 = load_json(tree_path_1)
    data_2 = load_json(tree_path_2)

    qid_to_1 = {
        item["qid"]: item for item in data_1["results"]
    }
    qid_to_2 = {
        item["qid"]: item for item in data_2["results"]
    }
    assert set(qid_to_1.keys()) == set(qid_to_2.keys())
    qid_set = set()
    for qid in qid_to_1:
        item_1 = qid_to_1[qid]
        item_2 = qid_to_2[qid]
        
        if math.isclose(item_1["TED"], 0.0):
            if not math.isclose(item_2["TED"], 0.0):
                qid_set.add(qid)
        
        if math.isclose(item_2["TED"], 0.0):
            if not math.isclose(item_1["TED"], 0.0):
                qid_set.add(qid)

    print(f"qid_set: {len(qid_set)}")
    print(qid_set)


def diff_test_suite(test_suite_path, tree_edit_distance_path):
    """这次直接和树同构的结果做对比"""
    test_suite_data = load_json(test_suite_path)
    tree_edit_distance_data = load_json(tree_edit_distance_path)

    qid_to_test_suite = {
        item["qid"]: item for item in test_suite_data["results"]
    }
    qid_to_tree = {
        item["qid"]: item for item in tree_edit_distance_data["results"]
    }
    assert set(qid_to_test_suite.keys()) == set(qid_to_tree.keys())
    test_suite_diff_qid_set = set()
    tree_diff_qid_set = set()
    for qid in qid_to_test_suite:
        test_suite_item = qid_to_test_suite[qid]
        tree_item = qid_to_tree[qid]
        if math.isclose(tree_item["TED"], 0.0):
            if not math.isclose(test_suite_item["test_suite"]["f1"], 1.0):
                test_suite_diff_qid_set.add(qid)
        
        if math.isclose(test_suite_item["test_suite"]["f1"], 1.0):
            if not math.isclose(tree_item["TED"], 0.0):
                tree_diff_qid_set.add(qid)

    print(f"test_suite_diff_qid_set: {len(test_suite_diff_qid_set)}\n{test_suite_diff_qid_set}\n")
    print(f"tree_diff_qid_set: {len(tree_diff_qid_set)}\n{tree_diff_qid_set}")

def debug_diff_test_suite(test_suite_path, tree_edit_distance_path, src_data_path):
    """这次直接和树同构的结果做对比"""
    test_suite_data = load_json(test_suite_path)
    tree_edit_distance_data = load_json(tree_edit_distance_path)
    src_data = load_json(src_data_path)

    qid_to_test_suite = {
        item["qid"]: item for item in test_suite_data["results"]
    }
    qid_to_tree = {
        item["qid"]: item for item in tree_edit_distance_data["results"]
    }
    qid_to_src = {
        item["qid"]: item for item in src_data
    }
    assert set(qid_to_test_suite.keys()) == set(qid_to_tree.keys())
    test_suite_diff_qid_set = set()
    tree_diff_qid_set = set()
    for qid in qid_to_test_suite:
        test_suite_item = qid_to_test_suite[qid]
        tree_item = qid_to_tree[qid]
        src_item = qid_to_src[qid]
        if math.isclose(tree_item["TED"], 0.0):
            if not math.isclose(test_suite_item["test_suite"]["f1"], 1.0):
                test_suite_diff_qid_set.add(qid)
        
        if math.isclose(test_suite_item["test_suite"]["f1"], 1.0):
            if not math.isclose(tree_item["TED"], 0.0):
                tree_diff_qid_set.add(qid)
                print(f"qid: {qid}")
                print(f"simulated_sexp: {tree_item['searched_queries'][0]['s_expression']}")
                print(f"gold_sexp: {src_item['golden_s_expression']}")
                print(f"test_suite: {len(src_item['test_suite_examples'])}\n")

    print(f"test_suite_diff_qid_set: {len(test_suite_diff_qid_set)}\n{test_suite_diff_qid_set}\n")
    print(f"graph_diff_qid_set: {len(tree_diff_qid_set)}\n{tree_diff_qid_set}")


def error_analysis_test_suite():
    detection_result = load_json("data/output/query_construction/grailqa/grailqa_dev_0_1000_starQC_2024-05-19_09:12:50/final_results.json")
    dataset = DATASET.SIMULATED_FREEBASE
    src_data = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_test_suite.json")
    logger = setup_custom_logger("data/test/test_suite.log")
    sparql_querier = SparqlOdbcQuerier(
        ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, timeout=60
    )

    qid_to_src = {item["qid"]: item for item in src_data}

    sampled_examples = [
        detection_result['results'][idx] for idx in [263]
    ]
    for detection_item in tqdm(sampled_examples):
        src_item = qid_to_src[detection_item['qid']]
        simulated_sparql_txt = sexp_to_sparql_for_test_suite(detection_item["searched_queries"][0]["s_expression"])
        editor = SyntaxTreeEditor(simulated_sparql_txt, dataset, logger)
        editor.preprocess_test_suite(f"?{TARGET_VARIABLE_NAME}")
        simulated_sparql_txt_before = editor.sparql_txt

        for _test_suite in tqdm(src_item["test_suite_examples"]):
            editor = SyntaxTreeEditor(simulated_sparql_txt_before, dataset, logger)
            _update_flag = False
            for (_before, _after) in _test_suite["binding"].items():
                if editor.update_leaf_value(_before, _after): # 返回值为 bool 类型，表明是否有节点被更新
                    _update_flag = True # 任一节点被更新即可
            if not _update_flag:
                continue # 没有 overlap, 跳过该 test case
            predicted_answer = sparql_querier.get_execution_result_one_variable_sparql_wrapper(editor.sparql_txt, retry=2)
            gold_answer = _test_suite["answer"]
            p, r, f1 = get_PRF1(predicted_answer, gold_answer)

            if not math.isclose(f1, 1.0):
                logger.info(f"_test_suite: {_test_suite};\n predicted_answer: {predicted_answer}\n gold_answer: {gold_answer}")

def diff_test_suite_num(file_1, file_2):
    data_1 = load_json(file_1)
    data_2 = load_json(file_2)
    assert len(data_1) == len(data_2), print(f"1: {len(data_1)}; 2:{len(data_2)}")
    qid_set = set()
    for (item_1, item_2) in zip(data_1, data_2):
        if len(item_1["test_suite_examples"]) != len(item_2["test_suite_examples"]) + 1:
            if len(item_1["test_suite_examples"]) == 0 and len(item_2["test_suite_examples"]) == 0:
                # Test Suite 构建失败，这种情况可以接受
                pass
            else:
                qid_set.add(item_1["qid"])
    print(f"qid_set: {qid_set}\n{len(qid_set)}")

def diff_test_suite_num_webqsp(file_1, file_2):
    data_1 = load_json(file_1)
    data_2 = load_json(file_2)
    assert len(data_1) == len(data_2), print(f"1: {len(data_1)}; 2:{len(data_2)}")
    qid_set = set()
    for (item_1, item_2) in zip(data_1, data_2):
        assert len(item_1["test_suite_examples"]) == len(item_2["test_suite_examples"])
        for (_test_suite_1, _test_suite_2) in zip(item_1["test_suite_examples"], item_2["test_suite_examples"]):
            if len(_test_suite_1) != len(_test_suite_2) + 1:
                qid_set.add(item_1["qid"])
    print(f"qid_set: {qid_set}\n{len(qid_set)}")
      

def diff_compare_edit_distance(file1,file2,file_3,file_4):
    data1 = load_json(file1)
    data2 = load_json(file2)
    output_data = []
    ted_list = []
    result = data2["results"]
    error_data = []
    for r in result:
        quad_searched_queries = {
            "searched_queries":[],
            "TED":1.0
        }
        if "searched_queries" in r.keys() and r['searched_queries'][0]["precision"] == 1.0 and  r['searched_queries'][0]["recall"] == 1.0:
            ted_list.append(r['TED'])
            quad_searched_queries = {
                "searched_queries":r['searched_queries'],
                "TED":r['TED']
            }
        else:
            ted_list.append(1.0)

        if str(r['qid']) in data1.keys():
            qid = str(r['qid'])
            output_data.append(
                {
                    "qid":qid,
                "question":data1[qid]['question'],
                "quad_searched_queries":quad_searched_queries,
                "ours_searched_queries":{
                    "searched_queries":data1[qid]['search_results'],
                    "TED":data1[qid]['TED'],
                }
                }
                )
        else:
            error_data.append(r)
        
    print(sum(ted_list)*1.0/len(ted_list))
    print(1.0*len([ted for ted in ted_list if ted==0.0])/len(ted_list))
    dump_json(output_data,file_3)
    dump_json(error_data,file_4)

def diff_compare_test_suite(file1,file2,file_3,file_4):
    data1 = load_json(file1)
    data2 = load_json(file2)
    output_data = []
    
    test_suite_list = []

    result = data2["results"]
    error_data = []
    for r in result:
        quad_searched_queries = {
            "searched_queries":[],
            "test_suite": {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0
            }
        }
        if "searched_queries" in r.keys() and r['searched_queries'][0]["precision"] == 1.0 and  r['searched_queries'][0]["recall"] == 1.0:
            test_suite_list.append(r['test_suite'])
            quad_searched_queries = {
                "searched_queries":r['searched_queries'],
                "test_suite":r['test_suite']
            }
        else:
            test_suite_list.append({
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0
            })

        if str(r['qid']) in data1.keys():
            qid = str(r['qid'])
            output_data.append(
                {
                    "qid":qid,
                "question":data1[qid]['question'],
                "quad_searched_queries":quad_searched_queries,
                "ours_searched_queries":{
                    "searched_queries":data1[qid]['search_results'],
                    "TED":data1[qid]['test_suite']
                }
                }
                )
        else:
            error_data.append(r)
        
    p_list = [data["precision"] for data in test_suite_list]
    print("p:",sum(p_list)*1.0/len(p_list))

    r_list = [data["recall"] for data in test_suite_list]
    print("r:",sum(r_list)*1.0/len(r_list))

    f1_list = [data["f1"] for data in test_suite_list]
    print("f1:",sum(f1_list)*1.0/len(f1_list))

    dump_json(output_data,file_3)
    dump_json(error_data,file_4)        

def sample_test():
    # random.seed(100)
    dev_data = load_json("/home5/yhbao/data/grailqa/grailqa_v1.0_dev_linking.json")
    # sample_data = random.sample(dev_data,1000)
    src_data = load_json("/home5/whzhou/STAR-QC/data/input/GrailQA_v1.0/grailqa_v1.0_dev_0_1000_linking.json")
    qid_list = [str(data['qid']) for data in src_data]
    sample_data = [data for data in dev_data if str(data["qid"]) in qid_list]
    dump_json(sample_data,"data/metrics/input/grailqa_dev_linking_1000.json")
    
    
def metric_comparation():
    src_data_1 = load_json("data/metrics/new/grailqa_dev_0_1000_linked_test_suite.json")
    src_data_1.popitem()
    metric_1 = [data['test_suite_similarity'] if 'test_suite_similarity' in data else data['test_suite'] for data in src_data_1.values() ]
    
    # import matplotlib.pyplot as plt
    # plt.hist(metric, bins=30, alpha=0.7, color='blue', edgecolor='black')

    # # 添加标题和标签
    # plt.title('tss')
    # plt.savefig('data/metrics/new/fig/tss.png')

    src_data_2 = load_json("data/metrics/old/quad/grailqa_linked_dev_test_suite.json")
    src_data_2.popitem()
    metric_2 = [data['test_suite']['f1'] for data in src_data_2.values() ]
    
    assert len(metric_1) == len(metric_2)
    tgt_data = []

    for item1,item2,s1,s2 in zip(src_data_1.values(),src_data_2.values(),metric_1,metric_2):
        if abs(s1-s2)>=0.3:
            tgt_data.append({
                'old':item1,
                'now':item2
            })
    dump_json(tgt_data,"/home5/whzhou/STAR-QC/data/metrics/new/fig/c.json")
    # import matplotlib.pyplot as plt
    # plt.hist(metric, bins=30, alpha=0.7, color='blue', edgecolor='black')

    # # 添加标题和标签
    # plt.title('tsf')
    # plt.savefig('data/metrics/new/fig/tsf.png')
def merge_data():
    src_data = load_json("data/final_test/origin/annotated_dataset.json")
    src_data_map = {str(data['qid']):data for data in src_data}

    quad_link_test_suite = load_json("data/final_test/quad/quad_linked_test_suite_f1.json")
    quad_golden_test_suite = load_json("data/final_test/quad/quad_golden_test_suite_f1.json")
    queryagent_link_test_suite = load_json("data/final_test/queryagent/queryagent_linked_test_suite_f1.json")
    queryagent_link_test_suite_map = {str(data['qid']):data for data in queryagent_link_test_suite[:-1]}
    queryagent_golden_test_suite = load_json("data/final_test/queryagent/queryagent_golden_test_suite_f1.json")
    queryagent_golden_test_suite_map = {str(data['qid']):data for data in queryagent_golden_test_suite[:-1]}

    quad_link_ted = load_json("data/final_test/quad/linked_results_edit_distance.json")
    quad_golden_ted = load_json("data/final_test/quad/golden_results_edit_distance.json")
    queryagent_link_ted = load_json("data/final_test/queryagent/linked_results_edit_distance.json")
    queryagent_link_ted_map = {str(data['qid']):data for data in queryagent_link_ted[:-1]}
    queryagent_golden_ted = load_json("data/final_test/queryagent/golden_results_edit_distance.json")
    queryagent_golden_ted_map = {str(data['qid']):data for data in queryagent_golden_ted[:-1]}

    tgt_data = []
    for quad_g in quad_golden_test_suite.values():
        if 'qid' not in quad_g:
            continue
        qid = str(quad_g['qid'])
        if quad_g['search_results']:
            quad_g_sparql = quad_g['search_results'][0]['sparql']
        else:
            quad_g_sparql = ""
        quad_g_ts = quad_g["test_suite"]['f1']
        quad_g_f1 = quad_g['F1']
        quad_g_ted = quad_golden_ted[qid]['TED']

        quad_l_sparql = ""
        if quad_link_test_suite[qid]['search_results']:
            quad_l_sparql = quad_link_test_suite[qid]['search_results'][0]['sparql']
        quad_l_ts = quad_link_test_suite[qid]["test_suite"]['f1']
        quad_l_f1 = quad_link_test_suite[qid]['F1']
        quad_l_ted = quad_link_ted[qid]['TED']

        queryagent_l_sparql = queryagent_link_test_suite_map[qid]['gen_sparql']
        queryagent_l_f1 = queryagent_link_test_suite_map[qid]['f1']
        queryagent_l_ts = queryagent_link_test_suite_map[qid]["test_suite"]['f1']
        queryagent_l_ted = queryagent_link_ted_map[qid]['TED']

        queryagent_g_sparql = queryagent_golden_test_suite_map[qid]['gen_sparql']
        queryagent_g_f1 = queryagent_golden_test_suite_map[qid]['f1']
        queryagent_g_ts = queryagent_golden_test_suite_map[qid]["test_suite"]['f1']
        queryagent_g_ted = queryagent_golden_ted_map[qid]['TED']

        tgt_data.append({
            'qid':qid,
            'golden_sparql':src_data_map[qid]['golden_sparql'],
            'quad_golden':{
                'sparql':quad_g_sparql,
                'f1':quad_g_f1,
                'ts':quad_g_ts,
                'TED':quad_g_ted
            },
            'quad_link':{
                'sparql':quad_l_sparql,
                'f1':quad_l_f1,
                'ts':quad_l_ts,
                'TED':quad_l_ted
            },
            'queryagent_golden':{
                'sparql':queryagent_g_sparql,
                'f1':queryagent_g_f1,
                'ts':queryagent_g_ts,
                'TED':queryagent_g_ted
            },
            'queryagent_link':{
                'sparql':queryagent_l_sparql,
                'f1':queryagent_l_f1,
                'ts':queryagent_l_ts,
                'TED':queryagent_l_ted
            },
        })
    dump_json(tgt_data,"data/final_test/merge_all.json")

def data_filter():
    src_data = load_json("data/final_test/merge_all.json")
    tgt_data = []
    for data in src_data:
        golden_sparql = data['golden_sparql']
        for i in ['quad_golden','quad_link','queryagent_golden','queryagent_link']:
            sparql = data[i]['sparql']
            if '*' in sparql:
                continue
            tgt_data.append({
                'golden_sparql':golden_sparql,
                'sparql':data[i]['sparql'],
                'f1':data[i]['f1'],
                'ts':data[i]['ts'],
                'TED':data[i]['TED']
            })
    print(len(tgt_data))
    dump_json(tgt_data,'data/final_test/merge_all_select.json')

def fn_ana():
    ted_data  = load_json("data/metrics/webqsp_sparql_golden_ted.json")
    tss_data = load_json("data/metrics/WebQSP_test_golden.json")
    ov1_list = []
    ov0_list = []
    ov_01_list = []
    for data1,data2 in zip(tss_data,ted_data):
        if 'qid' not in data1:
            continue
        assert data1['sparql1'] == data2['sparql1']
        data1['ted'] = data2['edit_distance']

        if data1['overlapping_rate'] == 1:
            ov1_list.append(data1)
        elif data1['overlapping_rate'] == 0:
            ov0_list.append(data1)
        else:
            ov_01_list.append(data1)

    print(len(ov1_list),len(ov0_list),len(ov_01_list))

    ov1_ted = [data['ted'] for data in ov1_list]
    ov1_tss = [data['similarity'] for data in ov1_list]
    for tau in range(0,25,5):
        print(tau)
        print(len([ted for ted in ov1_ted if ted>(0+tau*0.01)]))
        print(len([tss for tss in ov1_tss if tss<(1-tau*0.01)]))


    print("---------")
    ov1_ted = [data['ted'] for data in ov0_list]
    ov1_tss = [data['similarity'] for data in ov0_list]
    for tau in range(0,25,5):
        print(tau)
        print(len([ted for ted in ov1_ted if ted>(0+tau*0.01)]))
        print(len([tss for tss in ov1_tss if tss<(1-tau*0.01)]))

    print("_________")
    ov1_ted = [data['ted'] for data in ov_01_list]
    ov1_tss = [data['similarity'] for data in ov_01_list]
    for tau in range(0,25,5):
        print(tau)
        print(len([ted for ted in ov1_ted if ted>(0+tau*0.01)]))
        print(len([tss for tss in ov1_tss if tss<(1-tau*0.01)]))



if __name__=='__main__':


    # test_new_test_suite("data/metrics/LRG/cwq_False_result.json",
    #     "data/input/cwq/cwq_test_0_1000_linking.json",
    #     "data/metrics/LRG/cwq_False_result_test_suite.json",
    #     DATASET.CWQ,
    #     "data/metrics/LRG/test_suite.log"
    #     )
    #----------------------------------ESWC rebuttal 实验1-----------------------------------------------------------------------
    test_new_test_suite(
        f1="/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/grailqa_nocache_notimeout_facc1_dev_0_200.json",
        f2="/home5/yhbao/data/grailqa/grailqa_dev_1000_facc1_linking.json",
        output_file="/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/TSS_grailqa_nocache_notimeout_test_0_200.json",
        dataset=DATASET.GRAIL,
        logger_path= "/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/logger/grailqa_test_suite_200.log"
        )
    test_new_test_suite(
        f1="/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/cwq_nocache_notimeout_qgg_test_0_200.json",
        f2="/home5/yhbao/data/cwq/cwq_test_1000_qgg_linking.json",
        output_file="/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/TSS_cwq_nocache_notimeout_test_0_1000.json",
        dataset=DATASET.CWQ,
        logger_path= "/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/logger/cwq_test_suite_200.log"
        )
    # test_new_test_suite(
    #     f1="/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/webqsp_cache_timeout_test_0_1000.json",
    #     f2="/home5/yhbao/data/webqsp/webqsp_0_1000_linking.json",
    #     output_file="/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/TSS_webqsp_cache_timeout_test_0_1000.json",
    #     dataset=DATASET.WEBQ,
    #     logger_path= "/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/logger/webqsp_test_suite_1000.log"
    #     )
    #----------------------------------ESWC rebuttal 实验2-----------------------------------------------------------------------
    test_new_test_suite(
        f1="/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/grailqa_cache_timeout_facc1_dev_0_1000.json",
        f2="/home5/yhbao/data/grailqa/grailqa_dev_1000_facc1_linking.json",
        output_file="/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/TSS_grailqa_cache_timeout_test_0_1000.json",
        dataset=DATASET.GRAIL,
        logger_path= "/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/logger/grailqa_test_suite_1000.log"
        )
    test_new_test_suite(
        f1="/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/cwq_cache_timeout_qgg_test_0_1000.json",
        f2="/home5/yhbao/data/cwq/cwq_test_1000_qgg_linking.json",
        output_file="/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/TSS_cwq_cache_timeout_test_0_1000.json",
        dataset=DATASET.CWQ,
        logger_path= "/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/logger/cwq_test_suite_1000.log"
        )
    test_new_test_suite(
        f1="/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/webqsp_cache_timeout_test_0_1000.json",
        f2="/home5/yhbao/data/webqsp/webqsp_0_1000_linking.json",
        output_file="/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/TSS_webqsp_cache_timeout_test_0_1000.json",
        dataset=DATASET.WEBQ,
        logger_path= "/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/logger/webqsp_test_suite_1000.log"
        )
    #------------------------------------------------------------------------------------------------------------------------------
    # new_quad_execute_test_suite(
    #     f1="/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/grailqa_cache_timeout_facc1_dev_0_1000.json",
    #     f2="/home5/yhbao/data/grailqa/grailqa_dev_1000_facc1_linking.json",
    #     output_file="/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/TSS_grailqa_cache_timeout_test_0_1000.json",
    #     dataset=DATASET.GRAIL,
    #     logger_path= "/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/logger/grailqa_test_suite_1000.log"
    #     )
    # new_quad_execute_test_suite(
    #     f1="/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/cwq_cache_timeout_qgg_test_0_1000.json",
    #     f2="/home5/yhbao/data/cwq/cwq_test_1000_qgg_linking.json",
    #     output_file="/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/TSS_cwq_cache_timeout_test_0_1000.json",
    #     dataset=DATASET.CWQ,
    #     logger_path= "/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/logger/cwq_test_suite_1000.log"
    #     )
    # new_quad_execute_test_suite(
    #     f1="/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/webqsp_cache_timeout_test_0_1000.json",
    #     f2="/home5/yhbao/data/webqsp/webqsp_0_1000_linking.json",
    #     output_file="/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/TSS_webqsp_cache_timeout_test_0_1000.json",
    #     dataset=DATASET.WEBQ,
    #     logger_path= "/home5/yhbao/SPARQLannotation/output/eswc2026_rebuttal/logger/webqsp_test_suite_1000.log"
    #     )
    #---------------------------------------------------------------------------------------------------------------------
    # ifiles = [
    #     "data/metrics/new_new/cwq/qgg_final_results.json",
    # ]
    # ofiles = [
    #     "data/metrics/new_new/cwq/qgg_metric.json",
    # ]
    # method = [
    #     DATASET.QUERYAGENT,
    # ]
    # for i in range(len(ifiles)):
    #     test_new_test_suite_old(ifiles[i],"data/metrics/old/input/cwq_test_linking_1000.json",ofiles[i],DATASET.CWQ,method[i])

    # # webqsp
    # test_new_test_suite("/home5/whzhou/SPARQLannotation_further/webqsp_dev_0_1000_linked_not_use_neighbor_entity_dec_20_False_True_rerank.json",
    #     "data/input/WebQSP/webqsp_test_0_1000_linking.json",
    #     "data/metrics/new_new/webqsp/ablation/webqsp_False_True_test_suite.json",
    #     DATASET.WEBQ,
    #     f"data/metrics/new_new/webqsp/test_suite.log",
    #     aflag=True
    #     ) 

    # test_new_test_suite("/home5/whzhou/SPARQLannotation_further/webqsp_dev_0_1000_linked_not_use_neighbor_entity_dec_20_True_False_rerank.json",
    #     "data/input/WebQSP/webqsp_test_0_1000_linking.json",
    #     "data/metrics/new_new/webqsp/ablation/webqsp_True_False_test_suite.json",
    #     DATASET.WEBQ,
    #     f"data/metrics/new_new/webqsp/test_suite_debug.log",
    #     aflag=True
    #     ) 
        
    # test_new_test_suite("data/metrics/new_new/webqsp/quad_link.json",
    #     "data/input/WebQSP/webqsp_test_0_1000_linking.json",
    #     "data/metrics/new_new/webqsp/ablation/webqsp_nogpt_test_suite.json",
    #     DATASET.WEBQ,
    #     f"data/metrics/new_new/webqsp/test_suite_debug.log",
    #     aflag=False
    #     ) 
