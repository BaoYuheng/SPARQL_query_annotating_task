import random
import stanza
import json
import math
from src.utils import (
    load_json, dump_json, setup_custom_logger, flatten_decomposition_tree
)
from src.sparql_executor import (
    SparqlOdbcQuerier,
    SparqlOdbcQuerierWikidata
)
from src.s_expression_utils import (
    extract_value_from_literal,
    reverse_process_literal,
    post_process_sexp_grailqa,
    post_process_sexp
)
from src.sparql_to_sexp_utils import (
    SparqlParserLC2
)
from src.common import (
    ODBC_CONFIG_OFFICIAL,
    ODBC_CONFIG_DKILAB,
    ODBC_CONFIG_WIKIDATA_2019,
    SPARQL_wrapper_path_official,
    SPARQL_wrapper_path_dkilab,
    SPARQL_wrapper_path_wikidata_2019,
    DELIMETER,
    FreebaseConstantForConstruction,
    WikidataConstantForConstruction,
    convert_freebase_type,
    convert_wikidata_type,
    DATASET,
    WIKIDATA_PREFIX_LIST
)
from src.entity_utils import EntityLinker
from src.sparql_utils import SyntaxTreeEditor
from collections import defaultdict
from tqdm import tqdm, trange
from typing import List
import re
import os
import math
import copy
logger = setup_custom_logger('/home5/yhbao/SPARQLannotation/output/log/data_process.log')
from src.utils import load_json, dump_json
from src.llm_agent import QueryIntent, OpenAiRequestor
import random
random.seed(1)


api_keys = ["sk-k9rLQCwGc0y1CPVRCb27F261206c47C3A9E6Bc9a076f34B9", 
            "sk-f1iYyHwqzhKaCdgp3532B18b623c4009B3C862CbF1096012", 
            "sk-SSKcCg4VHzmUbQVU494d50565a0c48349933BdD98425D288",
            "sk-jgJeUkNqks5aouo58fFdA5B62d4e41A6A1C9DfD175DbB20f",
            "sk-h9RHx26FRXcPuqus84E0220e8d7746728885531dB4751cA4",
            "sk-B9AmdunaQfRZHMRn7eBf86753870425dA3FaCb2b72AeD20f",
            "sk-kd48MeZ8bSAOuchiAaB34aFcE3C248A0B9Ab3d1b53F2C2C8",
            "sk-mRtDuepsyGIrRsxcB1B00cDc072c4d42A54cF2D418B77aAe",
            "sk-lh8YYRHVnOFA0oH320C7C26bAfE04d13899eDc6041C32338",
            "sk-UfL3N6xtniByttjmA69c34101c4843BbA1D8A33aBdC84cBe"]

def get_CWQ_embedding():
    data = []
    data += [item['question'] for item in load_json("/home5/yhbao/data/cwq/cwq_dev_linking.json")]
    data += [item['question'] for item in load_json("/home5/yhbao/data/cwq/cwq_test_linking.json")]
    data += [item['question'] for item in load_json("/home5/yhbao/data/cwq/cwq_train_linking.json")]
    openai_asker = OpenAiRequestor(model="text-embedding-ada-002", url='https://api.gpt.ge/v1/', api_keys=api_keys)
    result = openai_asker.concurrent_query(data, 10, QueryIntent.EMBEDDING)
    #print([item[0] for item in result])
    result_to_dict = {}
    for item in result:
        result_to_dict[item[0]] = item[1]
    dump_json(result_to_dict, "/home5/yhbao/SPARQLannotation/data/linking_data/cwq_question_embed.json")



def get_Wikidata_class_embedding():
    data = load_json("/home5/yhbao/SPARQLannotation/data/WD_ontology/wikidata_classes_all.json")
    qid_to_input_dict = {}
    input_to_qid_dict = {}
    for item in data:
        qid_to_input_dict[item['class'].split("/")[-1]] = item['label']
        input_to_qid_dict[item['label']] = item['class'].split("/")[-1]
    input_data = list(qid_to_input_dict.values())
    openai_asker = OpenAiRequestor(model="text-embedding-ada-002", url='https://api.gpt.ge/v1/', api_keys=api_keys)
    result = openai_asker.concurrent_query(input_data, 10, QueryIntent.EMBEDDING)
    #print([item[0] for item in result])
    result_to_dict = {}
    for item in result:
        result_to_dict[item[0]] = item[1]
    dump_json(result_to_dict, "/home5/yhbao/SPARQLannotation/data/linking_data/wikidata_class_embed.json")


def process_grailqa(split):
    """
    @param do_process_literal: 默认为 True, 处理 S-expression 和 SPARQL 之间的一些异构
    改动: 收集 entity 和 class, 统一查询 friendly name
    """
    grailqa_train = load_json(f'data/input/GrailQA_v1.0/grailqa_v1.0_{split}.json')
    processed_examples = list()
    sparql_querier = SparqlOdbcQuerier(
        ODBC_CONFIG_DKILAB,
        SPARQL_wrapper_path_dkilab,
        logger,
        timeout=60
    )
    for example in tqdm(grailqa_train):
        # GrailQA 每个问题都提供了 S-expression
        grounded_entities, grounded_literals, grounded_classes = extract_grounded_items(
            example["s_expression"], "grailqa"
        )
        grounded_items = list()
        for ent in grounded_entities:
            grounded_items.append({
                "mid": ent,
                "type": "entity"
            })
        for cls in grounded_classes:
            grounded_items.append({
                "mid": cls,
                "type": "class"
            })
        for literal in grounded_literals:
            grounded_items.append({
                "mid": literal,
                "type": "literal"
            })

        answer = sparql_querier.get_execution_result_one_variable_sparql_wrapper(example["sparql_query"])
        answer = [{
            "mid": ans,
            "type": convert_freebase_type(FreebaseConstantForConstruction.get_constant_type(ans)),
        } for ans in answer]
        
        processed_examples.append({
            "qid": example["qid"],
            "question": example["question"],
            "answer": answer,
            "golden_grounded_items": grounded_items,
            "golden_s_expression": example["s_expression"],
            "golden_sparql_query": example["sparql_query"]
        })
    dump_json(
        processed_examples,
        f"data/input/GrailQA_v1.0/intermediate/grailqa_v1.0_{split}_processed.json"
    )

def grailqa_substitute_linking_items(n=200, do_process_literal=True):
    grailqa_data = load_json(f"data/input/GrailQA_v1.0/grailqa_v1.0_train_{n}.json")
    item_linking_result = load_json("data/entity_candidate_generation/GrailQA_200_2023-11-27_top10/final_results.json")
    sparql_querier = SparqlOdbcQuerier(
        ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, 10
    )
    processed_examples = list()
    qid_to_linking_results = {
        ex["qid"]: ex for ex in item_linking_result["results"]
    }
    # 数量不多，我们就直接查询实体的 label, 不做缓存了
    for example in tqdm(grailqa_data):
        qid = example["qid"]
        item_to_label = dict()
        grounded_items = list()
        if qid in qid_to_linking_results: # else: 链接结果为空
            for linked_item in qid_to_linking_results[qid]["detected_entities"]:
                item_type = get_item_type(linked_item)
                if item_type == "entity":
                    grounded_items.append({
                        "mid": linked_item,
                        "type": "entity"
                    })
                    item_to_label[linked_item] = sparql_querier.get_friendly_name({
                        "mid": linked_item,
                        "type": "entity"
                    })
                elif item_type == "class":
                    grounded_items.append({
                        "mid": linked_item,
                        "type": "class"
                    })
                elif item_type == "literal":
                    linked_item = process_linked_literal(linked_item) if do_process_literal else linked_item
                    # 链接这边已经完成了 Literal 格式的处理
                    grounded_items.append({
                        "mid": linked_item,
                        "type": "literal"
                    })
                    item_to_label[linked_item] = extract_value_from_literal(linked_item)
                elif item_type is None: # 会排除掉格式比较奇怪的 class
                    logger.error(f"linked_item: {linked_item}; item_type: {item_type}")
                else:
                    raise NotImplementedError(f"item_type: {item_type}; linked_item: {linked_item}")

        processed_examples.append({
            "qid": example["qid"],
            "question": example["question"],
            "answer": example["answer"],
            "grounded_items": grounded_items,
            "grounded_item_to_label": item_to_label,
            "golden_grounded_items": example["grounded_items"],
            "golden_s_expression": example["golden_s_expression"],
            "golden_sparql_query": example["golden_sparql_query"]
        })
    dump_json(
        processed_examples,
        f"data/input/GrailQA_v1.0/1128/grailqa_v1.0_train_{n}_linking.json"
    )

def cwq_substitute_linking_items(n=200, do_process_literal=True):
    cwq_data = load_json(f"data/input/cwq/cwq_train_{n}.json")
    item_linking_result = load_json("data/entity_candidate_generation/cwq_200_2023-11-27_top10/final_results.json")
    sparql_querier = SparqlOdbcQuerier(
        ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, 10
    )
    processed_examples = list()
    qid_to_linking_results = {
        ex["qid"]: ex for ex in item_linking_result["results"]
    }
    # 数量不多，我们就直接查询实体的 label, 不做缓存了
    for example in tqdm(cwq_data):
        qid = example["qid"]
        item_to_label = dict()
        grounded_items = list()
        if qid in qid_to_linking_results: # else: 链接结果为空
            for linked_item in qid_to_linking_results[qid]["detected_entities"]:
                item_type = get_item_type(linked_item)
                if item_type == "entity":
                    grounded_items.append({
                        "mid": linked_item,
                        "type": "entity"
                    })
                    item_to_label[linked_item] = sparql_querier.get_friendly_name({
                        "mid": linked_item,
                        "type": "entity"
                    })
                elif item_type == "class":
                    grounded_items.append({
                        "mid": linked_item,
                        "type": "class"
                    })
                elif item_type == "literal":
                    linked_item = process_linked_literal(linked_item) if do_process_literal else linked_item
                    grounded_items.append({
                        "mid": linked_item,
                        "type": "literal"
                    })
                    item_to_label[linked_item] = extract_value_from_literal(linked_item)
                elif item_type is None: # 会排除掉格式比较奇怪的 class
                    logger.error(f"linked_item: {linked_item}; item_type: {item_type}")
                else:
                    raise NotImplementedError(f"item_type: {item_type}; linked_item: {linked_item}")

        processed_examples.append({
            "qid": example["qid"],
            "question": example["question"],
            "answer": example["answer"],
            "grounded_items": grounded_items,
            "grounded_item_to_label": item_to_label,
            "golden_grounded_items": example["grounded_items"],
            "golden_s_expression": example["golden_s_expression"],
            "golden_sparql_query": example["golden_sparql_query"]
        })
    # TODO:
    dump_json(
        processed_examples,
        # f"data/input/cwq/cwq_train_{n}_linking.json"
        f"data/input/cwq/1128/cwq_train_{n}_linking.json"
    )


def webqsp_substitute_linking_items(n=200, do_process_literal=True):
    webqsp_data = load_json(f"data/input/WebQSP/webqsp_train_{n}.json")
    item_linking_result = load_json("data/entity_candidate_generation/webqsp_200_2023-11-27_top10/final_results.json")
    sparql_querier = SparqlOdbcQuerier(
        ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, 10
    )
    processed_examples = list()
    qid_to_linking_results = {
        ex["qid"]: ex for ex in item_linking_result["results"]
    }
    # 数量不多，我们就直接查询实体的 label, 不做缓存了
    for example in tqdm(webqsp_data):
        qid = example["qid"]
        item_to_label = dict()
        grounded_items = list()
        if qid in qid_to_linking_results: # else: 链接结果为空
            for linked_item in qid_to_linking_results[qid]["detected_entities"]:
                item_type = get_item_type(linked_item)
                if item_type == "entity":
                    grounded_items.append({
                        "mid": linked_item,
                        "type": "entity"
                    })
                    item_to_label[linked_item] = sparql_querier.get_friendly_name({
                        "mid": linked_item,
                        "type": "entity"
                    })
                elif item_type == "class":
                    grounded_items.append({
                        "mid": linked_item,
                        "type": "class"
                    })
                elif item_type == "literal":
                    linked_item = process_linked_literal(linked_item) if do_process_literal else linked_item
                    grounded_items.append({
                        "mid": linked_item,
                        "type": "literal"
                    })
                    item_to_label[linked_item] = extract_value_from_literal(linked_item)
                elif item_type is None: # 会排除掉格式比较奇怪的 class
                    logger.error(f"linked_item: {linked_item}; item_type: {item_type}")
                else:
                    raise NotImplementedError(f"item_type: {item_type}; linked_item: {linked_item}")

        processed_examples.append({
            "qid": example["qid"],
            "question": example["question"],
            "answer": example["answer"],
            "grounded_items": grounded_items,
            "grounded_item_to_label": item_to_label,
            "golden_grounded_items": example["grounded_items"],
            "golden_s_expression": example["golden_s_expression"],
            "golden_sparql_query": example["golden_sparql_query"],
            "golden_s_expression_list": example["golden_s_expression_list"]
        })
    dump_json(
        processed_examples,
        f"data/input/WebQSP/1128/webqsp_train_{n}_linking.json"
    )

def process_cwq(split='train'):
    cwq_data = load_json(f'data/input/cwq/sexpr/CWQ.{split}.expr.json')
    sparql_querier = SparqlOdbcQuerier(
        ODBC_CONFIG_OFFICIAL,
        SPARQL_wrapper_path_official,
        logger,
        timeout=60
    )

    processed_examples = list()

    for example in tqdm(cwq_data):
        grounded_items = list()
        answer = list()
        grounded_entities, grounded_literals, grounded_classes = extract_grounded_items(example["SExpr"], dataset="cwq")
        '''S-expression 中抽取失败，则改从 SPARQL 中抽取'''
        if (len(grounded_entities) == 0) and (len(grounded_literals) == 0) and (len(grounded_classes) == 0):
            grounded_entities, grounded_literals, grounded_classes = extract_grounded_items_from_sparql(
                example["sparql"]
            )
        for ent in grounded_entities:
            grounded_items.append({
                "mid": ent,
                "type": "entity"
            })
        for literal in grounded_literals:
            grounded_items.append({
                "mid": literal,
                "type": "literal"
            })
        for cls in grounded_classes:
            grounded_items.append({
                "mid": cls,
                "type": "class"
            })
        
        # 类型: ['literal', 'entity', 'class'], 暂时不区分 entity 和 class? (似乎就没有 class)        
        answer = sparql_querier.get_execution_result_one_variable_sparql_wrapper(example["sparql"])
        answer = [{
            "mid": ans,
            "type": convert_freebase_type(FreebaseConstantForConstruction.get_constant_type(ans))
        } for ans in answer]
        
        processed_examples.append({
            "qid": example["ID"],
            "question": example["question"],
            "answer": answer,
            "grounded_items": grounded_items,
            "golden_s_expression": example["SExpr"],
            "golden_sparql_query": example["sparql"]
        })

    dump_json(
        processed_examples,
        f"data/input/cwq/intermediate/cwq_{split}_processed.json"
    ) 

def sample_data(dataset, split, start=200, end=1200):
    random.seed(374)
    if dataset.lower() == 'cwq':
        '''CWQ'''
        data = load_json(f"data/input/cwq/cwq_{split}_linking.json")
    elif dataset.lower() == 'grailqa':
        '''GrailQA'''
        data = load_json(f"data/input/GrailQA_v1.0/grailqa_v1.0_{split}_linking.json")
    elif dataset.lower() == 'webqsp':
        '''WebQSP'''
        data = load_json(f"data/input/WebQSP/webqsp_{split}_linking.json")
    elif dataset.lower() == 'lc2':
        '''LC-QuAD2'''
        data = load_json(f"data/input/LC_QuAD_2/intermediate/lc2_{split}_processed.json")
    else:
        raise NotImplementedError(f"dataset: {dataset}")
    data_map = {
        item["qid"]: item for item in data
    }
    # 使用 list, 保证顺序
    qid_list = [item["qid"] for item in data]
    assert len(data_map) == len(data)
    ids = random.sample(qid_list, end)[start:end]
    sample = [
        data_map[qid] for qid in ids
    ]
    if dataset.lower() == 'cwq':
        '''CWQ'''
        dump_json(
            sample,
            f"data/input/cwq/cwq_{split}_{start}_{end}_linking.json"
        )
    elif dataset.lower() == 'grailqa':
        '''GrailQA'''
        dump_json(
            sample,
            f"data/input/GrailQA_v1.0/grailqa_v1.0_{split}_{start}_{end}_linking.json"
        )
    elif dataset.lower() == 'webqsp':
        '''WebQSP'''
        dump_json(
            sample,
            f"data/input/WebQSP/webqsp_{split}_{start}_{end}_linking.json"
        )
    elif dataset.lower() == 'lc2':
        '''LC-QuAD2'''
        dump_json(
            sample,
            f"data/input/LC_QuAD_2/intermediate/lc2_{split}_{start}_{end}.json"
        )
    else:
        raise NotImplementedError(f"dataset: {dataset}")

def parsing_and_add_decomposition(dataset, split, decomposition_dir, start=200, end=1200):
    if (start is not None) and (end is not None):
        if dataset.lower() == 'cwq':
            '''CWQ'''
            input_data = load_json(f"data/input/{dataset}/intermediate/cwq_{split}_{start}_{end}.json")
            decomposition_data = load_json(f"data/output/decomposition/cwq/{decomposition_dir}/results.json")
        elif dataset.lower() == 'grailqa':
            '''GrailQA'''
            input_data = load_json(f"data/input/GrailQA_v1.0/intermediate/grailqa_v1.0_{split}_{start}_{end}.json")
            decomposition_data = load_json(f"data/output/decomposition/grailqa/{decomposition_dir}/results.json")
        elif dataset.lower() == 'webqsp':
            '''WebQSP'''
            input_data = load_json(f"data/input/WebQSP/intermediate/webqsp_{split}_{start}_{end}.json")
            decomposition_data = load_json(f"data/output/{decomposition_dir}/results.json")
        elif dataset.lower() == 'lc2':
            '''LC-QuAD2'''
            input_data = load_json(f"data/input/LC_QuAD_2/intermediate/lc2_{split}_{start}_{end}.json")
            decomposition_data = load_json(f"data/output/decomposition/lc2/{decomposition_dir}/results.json")
        else:
            raise NotImplementedError(f"dataset: {dataset}")
    else:
        if dataset.lower() == 'cwq':
            '''CWQ'''
            input_data = load_json(f"data/input/{dataset}/intermediate/cwq_{split}_processed.json")
            decomposition_data = load_json(f"data/output/decomposition/cwq/{decomposition_dir}/results.json")
        elif dataset.lower() == 'grailqa':
            '''GrailQA'''
            input_data = load_json(f"data/input/GrailQA_v1.0/intermediate/grailqa_v1.0_{split}_processed.json")
            decomposition_data = load_json(f"data/output/decomposition/grailqa/{decomposition_dir}/results.json")
        elif dataset.lower() == 'webqsp':
            '''WebQSP'''
            input_data = load_json(f"data/input/WebQSP/intermediate/webqsp_{split}_processed.json")
            decomposition_data = load_json(f"data/output/decomposition/webqsp/{decomposition_dir}/results.json")
        elif dataset.lower() == 'qald':
            '''QALD'''
            input_data = load_json(f"data/input/QALD/intermediate/qald_wikidata_{split}_processed.json")
            decomposition_data = load_json(f"data/output/decomposition/qald/{decomposition_dir}/results.json")
        else:
            raise NotImplementedError(f"dataset: {dataset}")
    
    qid_to_decomposition = {
        example["qid"]: example for example in decomposition_data
    }
    failed_qids = list()
    for example in tqdm(input_data):
        '''parsing 出错，默认不做分解: processed_decomposition = [{"description": example["question"]}]'''
        qid = example["qid"]
        if qid not in qid_to_decomposition:
            logger.error(f"qid: {qid}")
        decomposition = qid_to_decomposition.get(qid, None)
        processed_decomposition = None
        
        if decomposition is None:
            failed_qids.append(qid)
            processed_decomposition = [{"description": example["question"]}]
        elif not decomposition["decomposition"]: # empty
            failed_qids.append(qid)
            processed_decomposition = [{"description": example["question"]}]
        else:
            processed_decomposition = decomposition["decomposition"]
            if type(processed_decomposition) is list: # 直接返回了 json 格式
                success_flag = True
            else:
                success_flag = False
                for delimeter in ['Output:\n', 'Output: \n', 'Output:']:
                    try:
                        '''选择最早出现的 delimeter: ChatGPT 有时候会自己补充几个回答'''
                        '''形如 Output: \n    [{\"description\": \"Who was the 1996 coach of the team owned by Jerry Jones\"}]\n\n    #\n    Question: What is the deepest lake in the world and where is it located?'''
                        proc = processed_decomposition.split(delimeter, 1)[-1].split('#', 1)[0].strip() # 同样选择最早出现的 '#', 取 Output: 和 # 之间的内容
                        processed_decomposition = json.loads(proc)
                        success_flag = True
                        break
                    except:
                        pass
                    
                    '''可能是多个问题的分解结果混在一起了, 然后可能没有 "#" '''
                    try:
                        '''形如 Thought:\n1. The question target is a vehicle.\n2. \"the color pink is an exterior color\" and \"which privately owned vehicle\" are conjunctive descriptions of the question target.\nOutput:\n[{\"description\": \"the color pink is an exterior color\"}, {\"description\": \"which privately owned vehicle\"}]\n\nQuestion:'''
                        proc = processed_decomposition.split(delimeter, 1)[-1].split('Question:', 1)[0].strip() # 同样选择最早出现的 'Question:', 取 Output: 和 Question: 之间的内容
                        processed_decomposition = json.loads(proc)
                        success_flag = True
                        break
                    except:
                        pass
            if not success_flag:
                '''示例: This question is not complex, therefore it does not need to be decomposed.'''
                failed_qids.append(qid)
                processed_decomposition = [{"description": example["question"]}]
        assert type(processed_decomposition) is list , print(f"qid: {qid}")
        example["decomposition"] = processed_decomposition # 兜底: 不做分解
        try:
            example["sub_question_list"] = flatten_decomposition_tree(processed_decomposition)
        except Exception as e:
            print(f"qid: {example['qid']} error: {e}; processed_decomposition: {processed_decomposition}")
            failed_qids.append(qid)
            example["sub_question_list"] = [[example['question']]]
    print(failed_qids)
    print(len(failed_qids))

    if (start is not None) and (end is not None):
        if dataset.lower() == 'cwq':
            '''CWQ'''
            dump_json(input_data, f"data/input/cwq/intermediate/cwq_{split}_{start}_{end}_decomposition.json")
        elif dataset.lower() == 'grailqa':
            '''GrailQA'''
            dump_json(input_data, f"data/input/GrailQA_v1.0/intermediate/grailqa_v1.0_{split}_{start}_{end}_decomposition.json")
        elif dataset.lower() == 'webqsp':
            '''WebQSP'''
            dump_json(input_data, f"data/input/WebQSP/intermediate/webqsp_{split}_{start}_{end}_decomposition.json")
        elif dataset.lower() == 'lc2':
            '''LC-QuAD2'''
            dump_json(input_data, f"data/input/LC_QuAD_2/intermediate/lc2_{split}_{start}_{end}_decomposition.json")
        elif dataset.lower() == 'qald':
            '''QALD'''
            dump_json(input_data, f"data/input/QALD/intermediate/qald_{split}_{start}_{end}_decomposition.json")
        else:
            raise NotImplementedError(f"dataset: {dataset}")
    else:
        if dataset.lower() == 'cwq':
            '''CWQ'''
            dump_json(input_data, f"data/input/cwq/intermediate/cwq_{split}_decomposition.json")
        elif dataset.lower() == 'grailqa':
            '''GrailQA'''
            dump_json(input_data, f"data/input/GrailQA_v1.0/intermediate/grailqa_v1.0_{split}_decomposition.json")
        elif dataset.lower() == 'webqsp':
            '''WebQSP'''
            dump_json(input_data, f"data/input/WebQSP/intermediate/webqsp_{split}_decomposition.json")
        elif dataset.lower() == 'qald':
            '''QALD'''
            dump_json(input_data, f"data/input/QALD/intermediate/qald_{split}_decomposition.json")
        else:
            raise NotImplementedError(f"dataset: {dataset}")

def parsing_and_add_decomposition_patch(input_data_path, decomposition_data_path, output_path):
    """补丁函数，哪里需要哪里搬"""
    input_data = load_json(input_data_path)
    decomposition_data = load_json(decomposition_data_path)
    
    qid_to_decomposition = {
        example["qid"]: example for example in decomposition_data
    }
    failed_qids = list()
    for example in tqdm(input_data):
        '''parsing 出错，默认不做分解: processed_decomposition = [{"description": example["question"]}]'''
        qid = example["qid"]
        if qid not in qid_to_decomposition:
            logger.error(f"qid: {qid}")
        decomposition = qid_to_decomposition.get(qid, None)
        processed_decomposition = None
        
        if decomposition is None:
            failed_qids.append(qid)
            processed_decomposition = [{"description": example["question"]}]
        elif not decomposition["decomposition"]: # empty
            failed_qids.append(qid)
            processed_decomposition = [{"description": example["question"]}]
        else:
            processed_decomposition = decomposition["decomposition"]
            if type(processed_decomposition) is list: # 直接返回了 json 格式
                success_flag = True
            else:
                success_flag = False
                for delimeter in ['Output:\n', 'Output: \n', 'Output:']:
                    try:
                        '''选择最早出现的 delimeter: ChatGPT 有时候会自己补充几个回答'''
                        '''形如 Output: \n    [{\"description\": \"Who was the 1996 coach of the team owned by Jerry Jones\"}]\n\n    #\n    Question: What is the deepest lake in the world and where is it located?'''
                        proc = processed_decomposition.split(delimeter, 1)[-1].split('#', 1)[0].strip() # 同样选择最早出现的 '#', 取 Output: 和 # 之间的内容
                        processed_decomposition = json.loads(proc)
                        success_flag = True
                        break
                    except:
                        pass
                    
                    '''可能是多个问题的分解结果混在一起了, 然后可能没有 "#" '''
                    try:
                        '''形如 Thought:\n1. The question target is a vehicle.\n2. \"the color pink is an exterior color\" and \"which privately owned vehicle\" are conjunctive descriptions of the question target.\nOutput:\n[{\"description\": \"the color pink is an exterior color\"}, {\"description\": \"which privately owned vehicle\"}]\n\nQuestion:'''
                        proc = processed_decomposition.split(delimeter, 1)[-1].split('Question:', 1)[0].strip() # 同样选择最早出现的 'Question:', 取 Output: 和 Question: 之间的内容
                        processed_decomposition = json.loads(proc)
                        success_flag = True
                        break
                    except:
                        pass
            if not success_flag:
                '''示例: This question is not complex, therefore it does not need to be decomposed.'''
                failed_qids.append(qid)
                processed_decomposition = [{"description": example["question"]}]
        assert type(processed_decomposition) is list , print(f"qid: {qid}")
        example["decomposition"] = processed_decomposition # 兜底: 不做分解
        try:
            example["sub_question_list"] = flatten_decomposition_tree(processed_decomposition)
        except Exception as e:
            print(f"qid: {example['qid']} error: {e}; processed_decomposition: {processed_decomposition}")
            failed_qids.append(qid)
            example["sub_question_list"] = [[example['question']]]
    logger.info(failed_qids)
    logger.info(len(failed_qids))

    dump_json(input_data, output_path)


def add_decomposition_dummy(dataset, split, start=200, end=1200):
    """
    消融实验，- 问题分解
    这里对原问题不做任何分解，修改相应字段
    """
    if (start is not None) and (end is not None):
        if dataset.lower() == 'cwq':
            '''CWQ'''
            input_data = load_json(f"data/input/{dataset}/intermediate/cwq_{split}_{start}_{end}.json")
        elif dataset.lower() == 'grailqa':
            '''GrailQA'''
            input_data = load_json(f"data/input/GrailQA_v1.0/intermediate/grailqa_v1.0_{split}_{start}_{end}.json")
        elif dataset.lower() == 'webqsp':
            '''WebQSP'''
            input_data = load_json(f"data/input/WebQSP/intermediate/webqsp_{split}_{start}_{end}.json")
        else:
            raise NotImplementedError(f"dataset: {dataset}")
    else:
        if dataset.lower() == 'cwq':
            '''CWQ'''
            input_data = load_json(f"data/input/{dataset}/intermediate/cwq_{split}_processed.json")
        elif dataset.lower() == 'grailqa':
            '''GrailQA'''
            input_data = load_json(f"data/input/GrailQA_v1.0/intermediate/grailqa_v1.0_{split}_processed.json")
        elif dataset.lower() == 'webqsp':
            '''WebQSP'''
            input_data = load_json(f"data/input/WebQSP/intermediate/webqsp_{split}_processed.json")
        else:
            raise NotImplementedError(f"dataset: {dataset}")
    
    failed_qids = list()
    for example in tqdm(input_data):
        qid = example['qid']
        example["decomposition"] = [{"description": example["question"]}] # 不做任何分解
        try:
            example["sub_question_list"] = flatten_decomposition_tree(example["decomposition"])
        except Exception as e:
            print(f"qid: {example['qid']} error: {e}; processed_decomposition: {example['decomposition']}")
            failed_qids.append(qid)
            example["sub_question_list"] = [[example['question']]]

    print(failed_qids)
    print(len(failed_qids))

    if (start is not None) and (end is not None):
        if dataset.lower() == 'cwq':
            '''CWQ'''
            dump_json(input_data, f"data/input/cwq/intermediate/cwq_{split}_{start}_{end}_decomposition_wo_decomp.json")
        elif dataset.lower() == 'grailqa':
            '''GrailQA'''
            dump_json(input_data, f"data/input/GrailQA_v1.0/intermediate/grailqa_v1.0_{split}_{start}_{end}_decomposition_wo_decomp.json")
        elif dataset.lower() == 'webqsp':
            '''WebQSP'''
            dump_json(input_data, f"data/input/WebQSP/intermediate/webqsp_{split}_{start}_{end}_decomposition_wo_decomp.json")
        else:
            raise NotImplementedError(f"dataset: {dataset}")
    else:
        if dataset.lower() == 'cwq':
            '''CWQ'''
            dump_json(input_data, f"data/input/cwq/intermediate/cwq_{split}_decomposition_wo_decomp.json")
        elif dataset.lower() == 'grailqa':
            '''GrailQA'''
            dump_json(input_data, f"data/input/GrailQA_v1.0/intermediate/grailqa_v1.0_{split}_decomposition_wo_decomp.json")
        elif dataset.lower() == 'webqsp':
            '''WebQSP'''
            dump_json(input_data, f"data/input/WebQSP/intermediate/webqsp_{split}_decomposition_wo_decomp.json")
        else:
            raise NotImplementedError(f"dataset: {dataset}")

def compare_file(
    file_1_path,
    file_2_path
):
    file_1 = load_json(file_1_path)
    file_2 = load_json(file_2_path)
    assert len(file_1) == len(file_2)
    for (example_1, example_2) in zip(file_1, file_2):
        grounded_1 = set([
            item["mid"] for item in example_1["grounded_items"]
        ])
        grounded_2 = set([
            item["mid"] for item in example_2["grounded_items"]
        ])
        if grounded_1 != grounded_2:
            print(f"grounded_1: {grounded_1}")
            print(f"grounded_2: {grounded_2}")

def process_grounded_item_literal_grailqa(literal_item):
    '''
    grounded_item 里面的 literal 格式和 SPARQL 也有所不同 处理起来也不太一样
    '''
    try:
        data_type = literal_item.split("^^")[1].split("#")[1]
    except: # 没有 data_type, LITERAL_TYPE.STRING; GrailQA 训练集中似乎没有这种情况？
        if literal_item.endswith("@en"):
            return literal_item
        else:
            return f"{literal_item}@en" 
    if data_type in ['date', 'gYearMonth', 'gYear']: # 观察 SPARQL 发现的结论, #dateTime 在 S-expression 中就处理好格式了
        literal_item = f'"{literal_item.split("^^")[0] + "-08:00"}"^^<{literal_item.split("^^")[1]}>'
    else:
        literal_item = f'"{literal_item.split("^^")[0]}"^^<{literal_item.split("^^")[1]}>'
    return literal_item

def process_linked_literal(literal_item):
    '''
    grounded_item 里面的 literal 格式和 SPARQL 也有所不同 处理起来也不太一样
    '''
    try:
        data_type = literal_item.split("^^")[1].split("#")[1]
    except: # 没有 data_type, LITERAL_TYPE.STRING 
        if literal_item.endswith("@en"):
            return literal_item
        else:
            return f"{literal_item}@en" 
    return literal_item


def process_literal_cwq(literal_item):
    '''
    literal 的格式和 SPARQL 保持一致
    #date, #gYear, #gYearMonth, #int, #integer, #float : 不存在
    #dateTime:
        - S-exp: "1943-03-20^^http://www.w3.org/2001/XMLSchema#dateTime"
        - SPARQL: \"1943-03-20\"^^xsd:dateTime 也就是 \"1943-03-20\"^^<http://www.w3.org/2001/XMLSchema#dateTime>
    普通字面量已经处理好了 内容外加了双引号 "\"Country\""

    CWQ 中只出现 xsd: dateTime 这一种后缀
    '''
    # 目前可能有的后缀: http://www.w3.org/2001/XMLSchema#dateTime, http://www.w3.org/2001/XMLSchema#float, 后者是我们补充的
    if 'http://www.w3.org/2001/XMLSchema#' in literal_item:
        value, literal_type = literal_item.split('^^')
        value = value.replace('-08:00', '')
        return f'"{value}"^^<{literal_type}>' # 加了一对双引号
    else:
        '''CWQ 的字面量 literal, 其格式应该和 S-expression 中抽取出来的保持一致，不应该额外添加 @en 标记'''
        return literal_item

def analyze_webqsp():
    """
    答案类型 grounded item 类型 等
    """
    examples = load_json('data/input/WebQSP/sexpr/WebQSP.train.expr.json')
    answer_types = set()
    keys = set()
    special_answers = set()
    special_topic_entity = set()
    special_constraints = set()
    value_constraints_type = set()
    for ex in tqdm(examples):
        # 只考虑 Parse 0
        parse = ex["Parses"][0]
        for ans in parse["Answers"]:
            answer_types.add(ans["AnswerType"])
            if ans["AnswerType"] == 'Value':
                special_answers.add(ans["AnswerArgument"])
        topic_entity = parse["TopicEntityMid"]
        if topic_entity:
            if not (topic_entity.startswith('m.') or topic_entity.startswith('g.')):
                special_topic_entity.add(topic_entity)
        if "Constraints" in parse:
            for cons in parse["Constraints"]:
                if cons["ArgumentType"] == "Value":
                    special_constraints.add(cons["Argument"])
                    value_constraints_type.add(cons["ValueType"])


        keys.update(parse.keys())
    # print(answer_types)
    # print(special_answers)
    # print(keys)
    print(special_topic_entity)
    print(special_constraints)
    print(value_constraints_type) # {'DateTime', 'String'}
    '''
    数据集中给出了答案类型，只有 {'Value', 'Entity'};
    我们观察了取值为 "Value" 的具体形式
    - 大部分是日期
    - 形如 'New Orleans Pelicans', 视作字面量处理即可 
        - 可能是 SPARQL 中对某个 item 的 type.object.name 做查询
        - 也可能 KB 上对应的位置就是一个字面量
    - 0030: 其实对应的是 "0030"^^<http://www.w3.org/2001/XMLSchema#gYear> 
        - 看来我们需要把 golden SPARQL 执行一遍，然后更新一下答案 (更新为查询的返回结果)
            - 执行结果返回的是一个字面量，也无法直接用于查询中
    - 我们对于执行结果还需要做个转换，一些已经观察到的结论是:
        - 1906-04-18 05:12:00 --> "1906-04-18T05:12:00"^^<http://www.w3.org/2001/XMLSchema#dateTime>
        - 1971-10-13 --> "1971-10-13"^^<http://www.w3.org/2001/XMLSchema#dateTime>
        - -1332-01-01 --> "-1332"^^<http://www.w3.org/2001/XMLSchema#gYear> ?
    {
        '2013-07-01', '1992-11-03', '1988', 'New Orleans Pelicans', '1984', '2012', '2004-05-01', 
        '1987', '1921-09', '2003-11-18', '2000-04-21', 'Animals', '1892-01-11', '1971-10-13', 
        '2014-06-23', '1897-03-04', '2008-11-04', '1883', '1973-01-01', '1948-05-15', '1957-03-25', 
        '1929-03-04', '2011', '1971-02-04', '2005', '2011-02-15', '0030', '-1332', '1997-10-28', 
        '2015-04-02', '2005-10-11', 'Australian Open', '2002', '2012-03-16', '2007-01-01', 
        '1789-04-30', '1988-11-08', '1961-01-20', '2001-07-07', '1992-11-04', '1964-08-27', 
        '1969-01-20', '1995-01-20', '2009-12-10', '1906-04-18', '2002-10-12', 
        'University of Vermont', '2001-08-03', '1990-01-12', '1914', 
        'Ferdinand Lewis Alcindor, Jr.', '1881', '1993-12-21', 
        '2012-02-07', '1996', '1901', '2011-09-12', '1955-11-01', '1962', '2003-03-20', 
        '2001', '2000', '1972', '1960-11-08', '1981-01-01', '2002-11-15', '1976-08-10', 
        '1998-12-30', '2007-08-02', '2001-04-19', '1860-11-06', '1864', 'Charles Philip Arthur George', 
        '1988-03-03', 'Seattle Pilots', '1977', '2004-05-23', '1977-07-13', "United Nations Children's Fund", '
        1995-01-01', '1986-01-01', '-479', '2007-07-21', '1991', '1980', '2006', '1996-11-05', '1991-05-17', 'jm', 
        '1949', '1996-08-28', '2001-01-20', '1940-05-15', '1992', '2008-12-18', '1656', '1968', '2009', '0343-12-06', '1864-11-08', 
        'The Enviroment', '2011-02-22', '2008', '0.01', 'Dwayne Michael Carter, Jr.', '2012-11-06', '2007'}
    '''
    '''
    和 grounded_item 可能相关的字段名
    - TopicEntityMid: 确认了全是实体
    - Constraints: "Entity" 和 "Value" 这两种类型(ArgumentType)
        - 其中 "Value" 类型的 "ValueType" 有 "String" 和 "DateTime"
    "Value": 
    '1945-01-01', '2007-01-01', '2011-12-31', 'Country', '2009-12-31', '1996-01-01', 
    '1819-01-01', 'City', 'Chief Executive Officer', '1980-01-01', '1958-12-31', 
    '1998-12-31', 'Young Forrest', '2003-12-31', '1998-01-01', '2003-01-01', 
    '5', '2005-01-01', '2012-12-31', '1980-12-31', '1948-12-31', '1971-01-01', 
    '2008-12-31', '2010-12-31', '2008-01-01', '2005-12-31', '1819-12-31', 'Danielle Rousseau', 
    '1899-12-31', '2013-01-01', '2013-12-31', '1971-12-31', 'State', '1800-01-01', '1948-01-01', 
    '2012-01-01', '2009-01-01', '1958-01-01', '2007-12-31', '2010-01-01', '2015-08-10', 
    '1945-12-31', '2011-01-01', '1996-12-31'
    - 1945-01-01: SPARQL 里面会这样写 `FILTER(xsd:datetime(?sk3) >= "1945-01-01"^^xsd:dateTime)`
        - 试了两个例子 ?sk3 >= "1945-01-01"^^xsd:dateTime 的结果也不变
    - Country: SPARQL 里面这样写 `FILTER (str(?sk0) = \"Country\")`


    看似有关，其实无关的:
    - Order: 对应 Order by 的谓词 但是我们不能使用关系的标注信息
    - InferentialChain: 关系标注，我们不需要
    - Time: 和 Constraints 重复了
    '''

def process_webqsp(split):
    """
    WebQSP 的 grounded items 是从数据集的标注信息中获得，而不是从 S-expression 中抽取；因此没有 process_webqsp_augmented
    """
    webqsp_train = load_json(f'data/input/WebQSP/sexpr/WebQSP.{split}.expr.json')
    processed_examples = list()
    sparql_querier = SparqlOdbcQuerier(
        ODBC_CONFIG_OFFICIAL,
        SPARQL_wrapper_path_official,
        logger,
        timeout=60
    )
    for example in tqdm(webqsp_train):
        '''GMT-KBQA 对于每个 parse 都提供了 S-expression, 因此这里的做法应该没问题'''
        parse = example["Parses"][0] # 选择首个 parse 作为 S-expression 标注和答案来源，清晰一些
        # 答案都重新获取一下
        answers = sparql_querier.get_execution_result_one_variable_sparql_wrapper(parse["Sparql"])
        answers = [{
            "mid": ans,
            "type": convert_freebase_type(FreebaseConstantForConstruction.get_constant_type(ans))
        } for ans in answers]
        
        grounded_items = list()
        if "TopicEntityMid" in parse and parse["TopicEntityMid"]:
            grounded_items.append({
                "mid": FreebaseConstantForConstruction(
                    "uri", f'http://rdf.freebase.com/ns/{parse["TopicEntityMid"]}'
                ).__repr__(),
                "type": "entity"
            })
        
        if "Constraints" in parse:
            for cons in parse["Constraints"]:
                if cons["ArgumentType"] == "Entity":
                    grounded_items.append({
                        "mid": FreebaseConstantForConstruction(
                            "uri", f'http://rdf.freebase.com/ns/{cons["Argument"]}'
                        ).__repr__(),
                        "type": "entity"
                    })
                elif cons["ArgumentType"] == "Value":
                    if cons["ValueType"] == "String":
                        grounded_items.append({
                            "mid": FreebaseConstantForConstruction(
                                "literal", cons["Argument"], None, "en"
                            ).__repr__(),
                            "type": "literal"
                        })
                    elif cons["ValueType"] == "DateTime":
                        grounded_items.append({
                            "mid": FreebaseConstantForConstruction(
                                "typed-literal", cons["Argument"], "http://www.w3.org/2001/XMLSchema#dateTime"
                            ).__repr__(),
                            "type": "literal"
                        })
                    else:
                        raise Exception(f'cons: {cons}; example id: {example["QuestionId"]}')
                else:
                    raise Exception(f'cons: {cons}; example id: {example["QuestionId"]}')
        # WebQSP 可能提供多个 Sexpr, 故补充一个字段
        gold_sexpr_list = [
            parse["SExpr"] for parse in example["Parses"]
        ]
        
        processed_examples.append({
            "qid": example["QuestionId"],
            "question": example["ProcessedQuestion"],
            "answer": answers,
            "golden_grounded_items": grounded_items,
            "golden_s_expression": parse["SExpr"],
            "golden_sparql_query": parse["Sparql"],
            "golden_s_expression_list": gold_sexpr_list
        })
    
    dump_json(processed_examples, f'data/input/WebQSP/intermediate/webqsp_{split}_processed.json')

def add_golden_sparql_list_webqsp(split):
    processed_data = load_json(f"data/input/WebQSP/webqsp_{split}_linking.json")
    src_data = load_json(f"data/input/WebQSP/sexpr/WebQSP.{split}.expr.json")
    qid_to_sparql_list = defaultdict(list)
    for item in src_data:
        qid = item["QuestionId"]
        for parse in item["Parses"]:
            qid_to_sparql_list[qid].append(parse["Sparql"])
    for processed_item in processed_data:
        qid = processed_item["qid"]
        sparql_list = qid_to_sparql_list[qid]
        processed_item["golden_sparql_list"] = sparql_list
    dump_json(processed_data, f"data/input/WebQSP/webqsp_{split}_linking.json")


def get_item_type(item):
    if item.startswith('m.') or item.startswith('g.'):
        return "entity"
    elif re.fullmatch("[a-zA-Z_]+\.[a-zA-Z_]+", item):
        return "class"
    elif 'http://www.w3.org/2001/XMLSchema' in item:
        return "literal"
    elif item.startswith('"') and item.endswith('"'):
        return "literal"
    elif re.fullmatch('".+"@.+', item): # 形如 "Andrew"@en
        return "literal"
    else:
        try: # 尝试转成数值，针对 CWQ
            value = float(item)
            return "literal"
        except:
            pass
    return None


def extract_grounded_items(expr, dataset):
    '''
    返回的 literal 基本保持了在 S-expression 中的原格式，除了 float 可能加个后缀
    '''
    grounded_entities = set()
    grounded_literals = set()
    grounded_classes = set()
    expr = expr.replace('(', ' ( ')
    expr = expr.replace(')', ' ) ')
    '''S-expression 这么写感觉没问题，带后缀的 literal, S-expression 里面是没有双引号的'''
    # if dataset == 'cwq': 
    #     # literal 里面可能含有空格，需要提前抽取一下
    #     literals = re.findall("(\".+?\"(@en)?)", expr)
    #     for (literal, _) in literals:
    #         grounded_literals.add(FreebaseConstantForConstruction(
    #             "literal", literal[:-3], lang_tag="en"
    #         ).__repr__())

    toks = expr.split(' ')
    toks = [x for x in toks if len(x)]
    for t in toks:
        t = t.strip()
        if t.startswith('m.') or t.startswith('g.'):
            grounded_entities.add(
                FreebaseConstantForConstruction(
                    "uri", f"http://rdf.freebase.com/ns/{t}"
                ).__repr__()
            )
        elif re.fullmatch("[a-zA-Z_]+\.[a-zA-Z_]+", t):
            grounded_classes.add(FreebaseConstantForConstruction(
                "uri", f"http://rdf.freebase.com/ns/{t}"
            ).__repr__())
        elif 'http://www.w3.org/2001/XMLSchema' in t: # 1998^^http://www.w3.org/2001/XMLSchema#dateTime
            grounded_literals.add(FreebaseConstantForConstruction(
                "typed-literal", t.split('^^')[0], t.split('^^')[1]
            ).__repr__())
        elif t.startswith('"') and t.endswith('"'):
            grounded_literals.add(FreebaseConstantForConstruction(
                "literal", t
            ).__repr__())
        elif re.fullmatch('".+?"@en', t): # 形如 "Andrew"@en
            grounded_literals.add(FreebaseConstantForConstruction(
                "literal", t[:-3], lang_tag="en"
            ).__repr__())
        else:
            try: # 尝试转成数值，针对 CWQ
                if t.startswith('"') and t.endswith('"'):
                    value = float(t[1:-1])
                else:
                    value = float(t)
                grounded_literals.add(FreebaseConstantForConstruction(
                    "typed-literal", str(value), "http://www.w3.org/2001/XMLSchema#float"
                ).__repr__())
            except:
                pass

    return grounded_entities, grounded_literals, grounded_classes


def extract_grounded_items_from_sparql(sparql):
    '''
    返回的 literal 基本保持了在 S-expression 中的原格式，除了 float 可能加个后缀
    '''
    grounded_entities = set()
    grounded_literals = set()
    grounded_classes = set()
    sparql = sparql.replace('(',' ( ').replace(')',' ) ')

    toks = sparql.split() # 空格, \n 等
    toks = [x for x in toks if len(x)]
    for t in toks:
        t = t.strip()
        if t.startswith('ns:m.') or t.startswith('ns:g.'):
            grounded_entities.add(
                FreebaseConstantForConstruction(
                    "uri", f"http://rdf.freebase.com/ns/{t[3:]}"
                ).__repr__()
            )
        elif re.fullmatch("ns:[a-zA-Z_]+\.[a-zA-Z_]+", t):
            grounded_classes.add(
                FreebaseConstantForConstruction(
                    "uri", f"http://rdf.freebase.com/ns/{t[3:]}"
                ).__repr__()
            )
        elif '^^xsd:' in t:
            split_res = t.split('^^xsd:')
            assert len(split_res) == 2
            value, suffix = t.split('^^xsd:') # "2015-08-10" dateTime
            grounded_literals.add(
                FreebaseConstantForConstruction(
                    "typed-literal", value, data_type=f"http://www.w3.org/2001/XMLSchema#{suffix}"
                ).__repr__()
            )
        elif t.startswith('"') and t.endswith('"'):
            grounded_literals.add(
                FreebaseConstantForConstruction(
                    "literal", t
                ).__repr__()
            )
        elif re.fullmatch('".+?"@en', t): # 形如 "Andrew"@en
            grounded_literals.add(
                FreebaseConstantForConstruction(
                    "literal", t[:-3], lang_tag="en"
                ).__repr__()
            )
        # else: # 可能会引入 LIMIT 1 之类的噪声
        #     try: # 尝试转成数值，针对 CWQ
        #         value = float(t)
        #         grounded_literals.add(f"{value}^^http://www.w3.org/2001/XMLSchema#float")
        #     except:
        #         pass


    return grounded_entities, grounded_literals, grounded_classes

def extract_grounded_items_from_sparql_wikidata(sparql):
    grounded_entities = set()
    grounded_literals = set()
    sparql = sparql.replace('(',' ( ').replace(')',' ) ').replace('{', ' { ').replace('}', ' } ')

    toks = sparql.split() # 空格, \n 等
    toks = [x for x in toks if len(x)]
    for t in toks:
        t = t.strip()
        if t.startswith('wd:Q'):
            grounded_entities.add(WikidataConstantForConstruction(
                "uri", f"http://www.wikidata.org/entity/{t[3:]}"
            ).__repr__())
        elif t.startswith("'") and t.endswith("'"):
            grounded_literals.add(
                WikidataConstantForConstruction("literal", f'"{t[1:-1]}"').__repr__()
            )
        else:
            try:
                v = int(t)
                grounded_literals.add(
                    WikidataConstantForConstruction("typed-literal", str(v), "http://www.w3.org/2001/XMLSchema#decimal").__repr__()
                )
            except:
                try:
                    v = float(t)
                    grounded_literals.add(
                        WikidataConstantForConstruction("typed-literal", str(v), "http://www.w3.org/2001/XMLSchema#decimal").__repr__()
                    )
                except:
                    continue
    return grounded_entities, grounded_literals

def extract_grounded_items_from_sparql_wikidata_qald(sparql):
    """
    QALD 中的 SPARQL 没有使用缩写，抽取规则又有所不同
    """
    grounded_entities = set()
    grounded_literals = set()
    sparql = sparql.replace('(',' ( ').replace(')',' ) ')

    toks = sparql.split() # 空格, \n 等
    toks = [x for x in toks if len(x)]
    for t in toks:
        t = t.strip()
        # 实体要带上前缀
        if t.startswith('<http://www.wikidata.org/entity/Q'): 
            grounded_entities.add(WikidataConstantForConstruction(
                "uri", t[1:-1]
            ).__repr__())
        elif t.startswith('wd:Q'): # 很不规整，两种写法都有
            grounded_entities.add(WikidataConstantForConstruction(
                "uri", f"http://www.wikidata.org/entity/{t[3:]}"
            ).__repr__())
        elif t.startswith('"') and t.endswith('"'):
            grounded_literals.add(
                WikidataConstantForConstruction("literal", t).__repr__()
            )
        elif re.fullmatch('".+?"@en', t): # 形如 "Andrew"@en
            grounded_literals.add(
                WikidataConstantForConstruction("literal", t[:-3], lang_tag="en").__repr__()
            )
        else:
            try:
                v = int(t)
                grounded_literals.add(
                    WikidataConstantForConstruction("typed-literal", str(v), "http://www.w3.org/2001/XMLSchema#decimal").__repr__()
                )
            except:
                try:
                    v = float(t)
                    grounded_literals.add(
                        WikidataConstantForConstruction("typed-literal", str(v), "http://www.w3.org/2001/XMLSchema#decimal").__repr__()
                    )
                except:
                    continue
    return grounded_entities, grounded_literals 

def extract_grounded_items_from_sexpr_wikidata(expr):
    grounded_entities = set()
    grounded_literals = set()
    grounded_classes = set()
    expr = expr.replace('(', ' ( ')
    expr = expr.replace(')', ' ) ')    
    toks = expr.split(' ')
    toks = [x for x in toks if len(x)]
    
    for t in toks:
        t = t.strip()
        if re.fullmatch(r"wd:Q\d+", t):
            grounded_entities.add(
                WikidataConstantForConstruction(
                    "uri", f"http://www.wikidata.org/entity/{t[3:]}"
                ).__repr__()
            )
        elif 'http://www.w3.org/2001/XMLSchema' in t:
            grounded_literals.add(WikidataConstantForConstruction(
                "typed-literal", t.split('^^')[0], t.split('^^')[1]
            ).__repr__())
        elif t.startswith('"') and t.endswith('"'):
            grounded_literals.add(WikidataConstantForConstruction(
                "literal", t
            ).__repr__())
        elif re.fullmatch('".+?"@en', t): # 形如 "Andrew"@en
            grounded_literals.add(WikidataConstantForConstruction(
                "literal", t[:-3], lang_tag="en"
            ).__repr__())
        else:
            try:
                v = int(t)
                grounded_literals.add(
                    WikidataConstantForConstruction("typed-literal", str(v), "http://www.w3.org/2001/XMLSchema#decimal").__repr__()
                )
            except:
                try:
                    v = float(t)
                    grounded_literals.add(
                        WikidataConstantForConstruction("typed-literal", str(v), "http://www.w3.org/2001/XMLSchema#decimal").__repr__()
                    )
                except:
                    continue

    return grounded_entities, grounded_literals, grounded_classes


def combine_tmp_results_and_summary(tmp_folder, start_idx, end_idx):
    """
    搜索过程中我们会保留多个 checkpoint, 避免程序终止之后代码白跑
    将一些 checkpoint 的结果组合起来，并且给出这部分结果的总结
    @param: start_idx: 闭区间
    @param: end_idx: 开区间
    """
    all_results = list()
    for idx in range(start_idx, end_idx):
        all_results.extend(
            load_json(os.path.join(tmp_folder, f"{idx}.json"))
        )
    best_precision_list = [
        item["searched_queries"][0]["precision"] 
        if "searched_queries" in item 
        else 0.0
        for item in all_results
    ]
    search_time_list = [
        item["time"] for item in all_results
    ]
    summary = {
        "总数": len(all_results),
        "best_precision=1的数量": len([prec for prec in best_precision_list if math.isclose(prec, 1.0)]),
        "平均搜索时间": sum(search_time_list) / len(search_time_list),
    }

    dump_json({
        "results": all_results,
        "summary": summary
    }, os.path.join(tmp_folder, f"{start_idx}_{end_idx}.json"))

def construct_dataset_with_simulated_queries_grailqa():
    logger = setup_custom_logger("data/test/data_process.log")
    sparql_odbc = SparqlOdbcQuerier(
        ODBC_CONFIG_DKILAB,
        SPARQL_wrapper_path_dkilab,
        logger,
        timeout=60
    )
    grailqa_train = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_train.json")
    simulated_query = load_json("data/output/GrailQA_processed_2023-10-22_15:29:16/_tmp/0_9.json")
    assert "summary" in simulated_query
    grailqa_train_map = {
        item["qid"]: item for item in grailqa_train
    }
    original_examples = list()
    new_examples = list()
    
    for res_item in tqdm(simulated_query["results"]):
        # 原数据集: 保留所有样本（无论是否探测到 Precision = 1 的 Simulated Query）
        assert res_item["qid"] in grailqa_train_map
        original_ex = grailqa_train_map[res_item["qid"]]
        original_examples.append(original_ex)

        new_ex = copy.deepcopy(original_ex)
        # 仅保留 Precision = 1.0 的
        if ("precision" not in res_item) or (not math.isclose(res_item["precision"], 1.0)):
            continue
        
        # S-expression: 需要转换成 GrailQA 原来的格式
        simulated_sexp = post_process_sexp_grailqa(res_item["s_expression"])

        # ["graph_query"]["nodes"] 的一些处理
        grounded_entities, grounded_literals, grounded_classes = extract_grounded_items(simulated_sexp, "grailqa")
        nodes = list()
        for g_class in grounded_classes:
            nodes.append({
                "node_type": "class",
                "id": g_class,
                "friendly_name": sparql_odbc.get_friendly_name({
                    "mid": g_class, "type": "class"
                }),
                "function": "none" # TODO: 暂时这么做，或许会有一些损失
            })
        for g_entity in grounded_entities:
            nodes.append({
                "node_type": "entity",
                "id": g_entity,
                "friendly_name": sparql_odbc.get_friendly_name({
                    "mid": g_entity, "type": "entity"
                }),
                "function": "none"
            })
        for g_literal in grounded_literals:
            g_literal = reverse_process_literal(g_literal)
            nodes.append({
                "node_type": "literal",
                "id": g_literal,
                "friendly_name": g_literal.split('^^')[0],
                "function": "none"
            })
        gold_answer_type = None # Pangu 训练时需要
        for node in original_ex["graph_query"]["nodes"]:
            if node["node_type"] == 'class':
                gold_answer_type = node['id']
                break
        assert gold_answer_type is not None
        
        new_ex["s_expression"] = simulated_sexp
        new_ex["graph_query"]["nodes"] = nodes
        new_ex["gold_answer_type"] = gold_answer_type
        new_examples.append(new_ex)
    
    dump_json(
        new_examples, 
        f"data/output/GrailQA_processed_2023-10-22_15:29:16/dataset/grailqa_v1.0_train_{len(new_examples)}_simulated.json"
    )
    dump_json(
        original_examples,
        f"data/output/GrailQA_processed_2023-10-22_15:29:16/dataset/grailqa_v1.0_train_{len(original_examples)}_original.json"
    )

def construct_dataset_with_simulated_queries_cwq(
    reserve_all_original_examples=True
):
    """
    @param reserve_all_original_examples: 无论探测到的 Simulated Query 是否 Precision = 1, 都保留原数据集的样本
    """
    from components.gmt.logic_form_utils import lisp_to_sparql as lisp_to_sparql_gmt
    logger = setup_custom_logger("data/test/data_process.log")
    sparql_odbc = SparqlOdbcQuerier(
        ODBC_CONFIG_OFFICIAL,
        SPARQL_wrapper_path_official,
        logger,
        timeout=60
    )
    cwq_train = load_json("data/input/cwq/sexpr/CWQ.train.expr.json")
    simulated_query = load_json("data/output/cwq_train_processed_2023-10-22_22:13:39/_tmp/0_9.json")
    assert "summary" in simulated_query
    cwq_train_map = {
        item["ID"]: item for item in cwq_train
    }
    original_examples = list()
    new_examples = list()
    
    for res_item in tqdm(simulated_query["results"]):
        # 原数据集: 保留所有样本（无论是否探测到 Precision = 1 的 Simulated Query）
        assert res_item["qid"] in cwq_train_map
        original_ex = cwq_train_map[res_item["qid"]]
        new_ex = copy.deepcopy(original_ex)
        if reserve_all_original_examples:
            original_examples.append(original_ex)
        else:
            if ("precision" in res_item) and (math.isclose(res_item["precision"], 1.0)):
                original_examples.append(original_ex)
        # 仅保留 Precision = 1.0 的
        if ("precision" not in res_item) or (not math.isclose(res_item["precision"], 1.0)):
            continue
        
        # S-expression: 需要转换成 GrailQA 原来的格式
        simulated_sexp = post_process_sexp(res_item["s_expression"])
        simulated_sparql = lisp_to_sparql_gmt(simulated_sexp)
        
        new_ex["SExpr"] = simulated_sexp
        new_ex["sparql"] = simulated_sparql
        new_examples.append(new_ex)

        golden_answer = set([ans["answer_id"] for ans in original_ex["answers"]])
        execution_answer = sparql_odbc.get_execution_result_one_variable(simulated_sparql)
        # 一共 39 个 SPARQL 的执行结果不同，这个损失可以接受
        if golden_answer != execution_answer:
            logger.error(f"qid: {res_item['qid']}; golden_answer: {golden_answer}; execution_answer: {execution_answer}")
    
    dump_json(
        new_examples, 
        f"data/output/cwq_train_processed_2023-10-22_22:13:39/dataset/cwq.train.expr.{len(new_examples)}_simulated.json"
    )
    dump_json(
        original_examples,
        f"data/output/cwq_train_processed_2023-10-22_22:13:39/dataset/cwq.train.expr.{len(original_examples)}_original.json"
    )

def analyze_qald():
    src_data = load_json("data/input/QALD/qald_9_plus_train_wikidata.json")
    for example in tqdm(src_data["questions"]):
        # 答案列表长度大于 1？没观察到
        if len(example["answers"]) != 1:
            pass
            # logger.info(f"qid: {example['id']}, answer length: {len(example['answers'])}")
        
        if "vars" not in example["answers"][0]["head"]: # 是否存在执行结果有多个变量的情况 --> 不存在
            # logger.info(f"qid: {example['id']}, var: {example['answers'][0]['head']}")
            pass
        elif len(example["answers"][0]["head"]["vars"]) != 1:
            logger.info(f"qid: {example['id']}")
        
        # 特殊的答案格式 -- 存在，如果是布尔型问题
        if 'results' not in example["answers"][0]:
            # logger.info(f"qid: {example['id']}, 'results' not in example['answers'][0]")
            continue
        for dic_item in example["answers"][0]["results"]["bindings"]:
            if len(dic_item) > 1: # 不存在这种情况
                # logger.info(f"qid: {example['id']} len(dic_item) > 1")
                pass
            if list(dic_item.values())[0]["type"] != "uri":
                pass
                # logger.info(f"qid: {example['id']}, {example['answers'][0]['results']['bindings']}")

    """
    统计结果:
    答案类型 即 type 字段: 'uri', 'literal' (dateTime, integer, decimal, 'xml:lang': 'de' (注意 parsing 方式))
    变量名可能变化，parsing 的时候应该忽视变量名

    特殊答案:
    - 布尔型问题，如 "104", 
    "answers": [
        {
            "head": {},
            "boolean": true
        }
    ]
    
    总结: 三种类型的答案
    - uri: 需要转换成 Freebase ID
    - literal: 参考其他数据集，格式处理好即可
    - 布尔值（新增）
    答案有可能为空
    """

def process_qald(do_process_literal=False):
    sparql_querier_wikidata = SparqlOdbcQuerierWikidata(
        ODBC_CONFIG_WIKIDATA_2019, logger, timeout=10
    )
    sparql_querier_freebase = SparqlOdbcQuerier(
        ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, timeout=10
    )
    qald_train = load_json("data/input/QALD/qald_9_plus_train_wikidata.json")["questions"]
    processed_examples = list()

    for example in tqdm(qald_train):
        # 特判，boolean 类型答案
        if "boolean" in example["answers"][0]:
            example_answers = [
                {"mid": example["answers"][0]["boolean"], "type": "boolean"}
            ]
        else:
            example_answers = list()
            skip_flag = False
            for dic_item in example["answers"][0]["results"]["bindings"]:
                answer_item = list(dic_item.values())[0]
                if answer_item["type"] == 'uri':
                    '''任一答案无法映射到 Freebase 上，则我们跳过这个问题'''
                    wikidata_id = answer_item["value"].split('/')[-1]
                    freebase_id = sparql_querier_wikidata.get_freebase_mid_from_wikidata_mid(wikidata_id)
                    if not freebase_id:
                        logger.info(f"wikidata_id: {wikidata_id}; freebase_id: {freebase_id}")
                        skip_flag = True
                        break
                    example_answers.append({
                        "mid": freebase_id,
                        "type": "entity"
                    })
                elif answer_item["type"] == 'literal':
                    if "datatype" in answer_item:
                        example_answers.append({
                            "mid": f"\"{answer_item['value']}\"^^<{answer_item['datatype']}>",
                            "type": "literal"
                        })
                    elif "xml:lang" in answer_item:
                        example_answers.append({
                            "mid": f"\"{answer_item['value']}\"@{answer_item['xml:lang']}",
                            "type": "literal"
                        })
                    else:
                        raise NotImplementedError(f"answer_item: {answer_item}")
            if skip_flag:
                logger.info(f"Skip qid: {example['id']}; answers cannot be mapped to freebase")
                continue
        english_question = [q["string"] for q in example["question"] if q["language"] == "en"]
        assert len(english_question) == 1, logger.error(f"qid: {example['id']}; english_question: {english_question}")
        english_question = english_question[0]
        processed_examples.append({
            "qid": example["id"],
            "question": english_question,
            "answer": example_answers,
        })
    
    dump_json(
        processed_examples,
        "data/input/QALD/qald_9_plus_train_from_wikidata_processed.json"
    )

def process_qald_wikidata(split='train'):
    qald_data = load_json(f'data/input/QALD/qald_9_plus_{split}_wikidata.json')["questions"]
    sparql_querier = SparqlOdbcQuerierWikidata(
        ODBC_CONFIG_WIKIDATA_2019,
        SPARQL_wrapper_path_wikidata_2019,
        logger,
        timeout=60
    )
    processed_examples = list()

    for example in tqdm(qald_data):
        grounded_items = list()
        sparql = example["query"]["sparql"]
        try:
            s_expression = SparqlParserQALD.parse_sparql(sparql)
        except Exception as e:
            logger.info(f"error: {e}")
            s_expression = 'null'
        grounded_entities, grounded_literals, _ = extract_grounded_items_from_sexpr_wikidata(s_expression)
        if len(grounded_entities) == 0 and len(grounded_literals) == 0:
            grounded_entities, grounded_literals = extract_grounded_items_from_sparql_wikidata_qald(sparql)
        
        for ent in grounded_entities:
            grounded_items.append({
                "mid": ent,
                "type": "entity"
            })
        for literal in grounded_literals:
            grounded_items.append({
                "mid": literal,
                "type": "literal"
            })
        '''使用的端口不一样，用我们的端口重新执行一遍把'''
        answer = sparql_querier.get_execution_result_one_variable_sparql_wrapper(sparql)
        answer = [{
            "mid": ans,
            "type": convert_wikidata_type(WikidataConstantForConstruction.get_constant_type(ans))
        } for ans in answer]

        english_question = [q["string"] for q in example["question"] if q["language"] == "en"]
        assert len(english_question) == 1, logger.error(f"qid: {example['id']}; english_question: {english_question}")
        english_question = english_question[0]
        processed_examples.append({
            "qid": example["id"],
            "question": english_question,
            "answer": answer,
            "grounded_items": grounded_items,
            "golden_sparql_query": sparql,
            "golden_s_expression": s_expression
        })
    
    dump_json(
        processed_examples,
        f"data/input/QALD/intermediate/qald_wikidata_{split}_processed.json"
    ) 

def qald_add_golden_items(split, start=None, end=None):
    sparql_querier = SparqlOdbcQuerierWikidata(
        ODBC_CONFIG_WIKIDATA_2019, SPARQL_wrapper_path_wikidata_2019, logger, timeout=10
    ) # 查 label 得用 2019 端口
    if (start is not None) and (end is not None):
        data = load_json(f"data/input/QALD/intermediate/qald_{split}_{start}_{end}_decomposition.json")
    else:
        data = load_json(f"data/input/QALD/intermediate/qald_{split}_decomposition.json")
    for example in tqdm(data):
        example["golden_grounded_items"] = copy.deepcopy(example["grounded_items"])
        example.pop("grounded_items", None)
        golden_item_to_label = dict()
        for item in example["golden_grounded_items"]:
            item_type = item["type"]
            item_mid = item["mid"]
            if item_type == "entity":
                golden_item_to_label[item_mid] = sparql_querier.query_entity_label(item_mid[3:])["label"] # 去掉 wd: 前缀
            elif item_type == "literal":
                golden_item_to_label[item_mid] = extract_value_from_literal(item_mid)
            else: # 目前没有区分 entity 和 class
                logger.error(f"{item_type} {item_mid}")
        example["golden_item_to_label"] = golden_item_to_label
    if (start is not None) and (end is not None):
        dump_json(data, f"data/input/QALD/qald_{split}_{start}_{end}.json")
    else:
        dump_json(data, f"data/input/QALD/qald_{split}.json")

def qald_substitute_linking_items():
    qald_data = load_json(f"data/input/QALD/qald_9_plus_train_from_wikidata_processed.json")
    item_linking_result = load_json("data/entity_candidate_generation/QALD_2023-11-23/final_results.json")
    sparql_querier = SparqlOdbcQuerier(
        ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, 10
    )
    processed_examples = list()
    qid_to_linking_results = {
        ex["qid"]: ex for ex in item_linking_result["results"]
    }
    # 数量不多，我们就直接查询实体的 label, 不做缓存了
    for example in tqdm(qald_data):
        qid = example["qid"]
        item_to_label = dict()
        grounded_items = list()
        if qid in qid_to_linking_results:
            for linked_item in qid_to_linking_results[qid]["detected_entities"]:
                item_type = get_item_type(linked_item)
                if item_type == "entity":
                    grounded_items.append({
                        "mid": linked_item,
                        "type": "entity"
                    })
                    item_to_label[linked_item] = sparql_querier.get_friendly_name({
                        "mid": linked_item,
                        "type": "entity"
                    })
                elif item_type == "class":
                    grounded_items.append({
                        "mid": linked_item,
                        "type": "class"
                    })
                elif item_type == "literal":
                    # literal 的格式上应该存在问题，后面再说吧
                    grounded_items.append({
                        "mid": linked_item,
                        "type": "literal"
                    })
                    item_to_label[linked_item] = extract_value_from_literal(linked_item)
                elif item_type is None: # 会排除掉格式比较奇怪的 class
                    logger.error(f"linked_item: {linked_item}; item_type: {item_type}")
                else:
                    raise NotImplementedError(f"item_type: {item_type}; linked_item: {linked_item}")
        
        processed_examples.append({
            "qid": example["qid"],
            "question": example["question"],
            "answer": example["answer"],
            "grounded_items": grounded_items,
            "grounded_item_to_label": item_to_label
        })
    dump_json(
        processed_examples,
        "data/input/QALD/qald_9_plus_train_from_wikidata_linking.json"
    )

# def collect_answer_item_pairs(pair_num=5000):
#     import itertools
#     cwq_train = load_json("data/input/cwq/cwq_train_200.json")
#     webqsp_train = load_json("data/input/WebQSP/webqsp_train_200.json")
#     grailqa_train = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_train_200.json")
#     answer_set = set()
#     grounded_entity_set = set()
#     pairs = set()

#     for dataset in [cwq_train, webqsp_train, grailqa_train]:
#         for item in dataset:
#             ans_list = [ans['mid'] for ans in item["answer"] if ans["type"] == 'entity']
#             grounded_list = [grouneded['mid'] for grouneded in item["grounded_items"] if grouneded["type"] == 'entity']
#             answer_set.update(ans_list)
#             grounded_entity_set.update(grounded_list)
#             for (ans, grounded) in itertools.product(ans_list, grounded_list):
#                 pairs.add((ans, grounded))
#     for (ans, entity) in zip(answer_set, grounded_entity_set):
#         pairs.add((ans, entity))
#         if len(pairs) > pair_num:
#             break
#     dump_json(list(pairs), f"data/test/pairs_{pair_num}.json")

def collect_answer_item_pairs():
    import itertools
    cwq_train = load_json("data/input/cwq/1128/cwq_train_200_linking.json")
    answer_set = set()
    grounded_entity_set = set()
    pairs = set()

    for dataset in [cwq_train]:
        for item in dataset:
            ans_list = [ans['mid'] for ans in item["answer"] if ans["type"] == 'entity']
            grounded_list = [grouneded['mid'] for grouneded in item["grounded_items"] if grouneded["type"] == 'entity']
            answer_set.update(ans_list)
            grounded_entity_set.update(grounded_list)
            for (ans, grounded) in itertools.product(ans_list, grounded_list):
                pairs.add((ans, grounded))
    for (ans, entity) in zip(answer_set, grounded_entity_set):
        pairs.add((ans, entity))
    dump_json(list(pairs), "data/input/cwq/pairs_linking_200.json")

def collect_degree():
    cwq_train = load_json("data/input/cwq/1128/cwq_train_200_linking.json")
    entity_set = set()
    for example in tqdm(cwq_train):
        ans_list = [ans['mid'] for ans in example["answer"] if ans["type"] == 'entity']
        grounded_list = [grouneded['mid'] for grouneded in example["grounded_items"] if grouneded["type"] == 'entity']
        entity_set.update(ans_list)
        entity_set.update(grounded_list)
    degree_map = dict()
    sparql_querier = SparqlOdbcQuerierExps(
        ODBC_CONFIG_OFFICIAL, 
        SPARQL_wrapper_path_official,
        logger, 
        timeout=60
    )
    for ent in tqdm(entity_set):
        entity_rep = {
            "type": "entity", "mid": ent
        }
        degree_map[ent] = {
            "in": sparql_querier.get_in_degree(entity_rep),
            "out": sparql_querier.get_out_degree(entity_rep)
        }
    dump_json(degree_map, "data/input/cwq/1128/degrees.json")

def filter_cwq_data_step1(n=200):
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', download_method=None)
    constituency_parser = ConstituencyParsing.instance(nlp)
    cwq_data = load_json(f"data/input/cwq/cwq_train_{n}_linking.json")
    filtered_examples = list()
    for example in tqdm(cwq_data):
        if len(example["answer"]) == 0:
            continue
        if any([ans['type'] != "entity" for ans in example["answer"]]):
            continue
        if any([grounded['type'] not in ["entity", "class"] for grounded in example["golden_grounded_items"]]):
            continue
        
        # filtered_grounded_items = list()
        # for grounded in example["grounded_items"]: # 链接得到的
        #     if grounded["type"] == "entity":
        #         filtered_grounded_items.append(grounded)
        # example["grounded_items"] = filtered_grounded_items
        intermediate_clauses = constituency_parser._a_rule(nlp(example["question"]))
        if len(intermediate_clauses) > 0:
            continue
        filtered_examples.append(example)
    
    dump_json(filtered_examples, f"data/input/cwq/cwq_train_{n}_linking_step1.json")

def filter_cwq_data_step2(n=200):
    '''只过滤涉及 literal 的问题'''
    cwq_data = load_json(f"data/input/cwq/cwq_train_{n}_linking_manual.json")
    filtered_examples = list()
    for example in tqdm(cwq_data):
        if len(example["answer"]) == 0:
            continue
        if any([ans['type'] != "entity" for ans in example["answer"]]):
            continue
        if any([grounded['type'] not in ["entity", "class"] for grounded in example["golden_grounded_items"]]):
            continue
        example.pop("grounded_items", None)
        filtered_examples.append(example)
    
    dump_json(filtered_examples, f"data/input/cwq/cwq_train_{n}_linking_manual_step2.json")

def filter_cwq_data_step3(n=200):
    '''只过滤 class 相关的'''
    cwq_data = load_json(f"data/input/cwq/cwq_train_{n}_linking.json")
    filtered_examples = list()
    for example in tqdm(cwq_data):
        if len(example["answer"]) == 0:
            continue
        if any([ans['type'] == "class" for ans in example["answer"]]):
            continue
        example.pop("grounded_items", None)
        filtered_examples.append(example)
    
    dump_json(filtered_examples, f"data/input/cwq/cwq_train_{n}_linking_step3.json")


def filter_grailqa_data_step1(n=200):
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', download_method=None)
    constituency_parser = ConstituencyParsing.instance(nlp)
    cwq_data = load_json(f"data/input/GrailQA_v1.0/grailqa_v1.0_train_{n}_linking.json")
    filtered_examples = list()
    for example in tqdm(cwq_data):
        if len(example["answer"]) == 0:
            continue
        if any([ans['type'] != "entity" for ans in example["answer"]]):
            continue
        if any([grounded['type'] not in ["entity", "class"] for grounded in example["golden_grounded_items"]]):
            continue
        
        # filtered_grounded_items = list()
        # for grounded in example["grounded_items"]: # 链接得到的
        #     if grounded["type"] == "entity":
        #         filtered_grounded_items.append(grounded)
                
        # example["grounded_items"] = filtered_grounded_items
        intermediate_clauses = constituency_parser._a_rule(nlp(example["question"]))
        if len(intermediate_clauses) > 0:
            continue
        filtered_examples.append(example)
    
    dump_json(filtered_examples, f"data/input/GrailQA_v1.0/grailqa_v1.0_train_{n}_linking_step1.json")

def filter_webqsp_data_step1(n=200):
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', download_method=None)
    constituency_parser = ConstituencyParsing.instance(nlp)
    data = load_json(f"data/input/WebQSP/webqsp_train_{n}_linking.json")
    filtered_examples = list()
    # 补充一下 golden 实体
    for example in tqdm(data):
        if len(example["answer"]) == 0:
            continue
        if any([ans['type'] != "entity" for ans in example["answer"]]):
            continue
        if any([grounded['type'] not in ["entity", "class"] for grounded in example["golden_grounded_items"]]):
            continue
        
        # filtered_grounded_items = list()
        # for grounded in example["grounded_items"]: # 链接得到的
        #     if grounded["type"] == "entity":
        #         filtered_grounded_items.append(grounded)
                
        # example["grounded_items"] = filtered_grounded_items
        intermediate_clauses = constituency_parser._a_rule(nlp(example["question"]))
        if len(intermediate_clauses) > 0:
            continue
        golden_items = {item['mid'] for item in example["golden_grounded_items"] if item["type"] == 'entity'}
        grounded_items_enhanced = list()
        for item_list in example["grounded_items"]:
            grounded_items_enhanced.append(list(set(item_list) | golden_items))
        example["grounded_items"] = grounded_items_enhanced
        filtered_examples.append(example)
    
    dump_json(filtered_examples, f"data/input/WebQSP/webqsp_train_{n}_linking_step1.json")



def filter_dataset_with_degree(degree_threshold=500): # 500 + 500 = 1000
    import itertools
    dataset = load_json("data/input/cwq/cwq_train_200_linking.json")
    degree_map = load_json("data/input/cwq/degrees_linking_200.json")
    high_degree_idx_list = list()
    for (idx, example) in tqdm(enumerate(dataset)):
        pairs = set()
        ans_list = [ans['mid'] for ans in example["answer"] if ans["type"] == 'entity']
        grounded_list = [grouneded['mid'] for grouneded in example["grounded_items"] if grouneded["type"] == 'entity']
        for (ans, grounded) in itertools.product(ans_list, grounded_list):
            pairs.add((ans, grounded))
        flag = False
        for (ans, grounded) in pairs:
            ans_in = degree_map[ans]["in"]
            ans_out = degree_map[ans]["out"]
            grounded_in = degree_map[grounded]["in"]
            grounded_out = degree_map[grounded]["out"]
            if ans_in > degree_threshold or ans_out > degree_threshold or grounded_in > degree_threshold or grounded_out > degree_threshold:
                flag = True
                break
        if flag:
            high_degree_idx_list.append(idx)
    print(f"high_degree_idx_list: {len(high_degree_idx_list)}; {high_degree_idx_list}")

def parse_and_entity_linking(dataset, split, start=200, end=1200):
    args = dict()
    args["top_k_per_mention"] = 5
    args["top_k_entities_per_question"] = 10
    args["top_k_classes_per_question"] = 5

    sparql_querier = SparqlOdbcQuerier(
        ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_official, logger, timeout=60
    )
    entity_linker = EntityLinker.instance(
        logger, sparql_querier, 
        top_k_per_mention=args["top_k_per_mention"],
    )

    if dataset.lower() == 'cwq':
        '''CWQ'''
        if (start is not None) and (end is not None):
            data = load_json(f"/home5/yhbao/data/cwq/intermidiate/cwq_{split}_{start}_{end}_decomposition.json")
        else:
            data = load_json(f"/home5/yhbao/data/cwq/intermidiate/cwq_{split}_decomposition.json")
    elif dataset.lower() == 'webqsp':
        '''WebQSP'''
        if (start is not None) and (end is not None):
            data = load_json(f"/home5/yhbao/data/webqsp/intermidiate/webqsp_{split}_{start}_{end}_decomposition.json")
        else:
            data = load_json(f"/home5/yhbao/data/webqsp/intermidiate/webqsp_{split}.json")
    else:
        raise NotImplementedError(f"dataset: {dataset}")

    for example in tqdm(data):
        sub_question_num = len(example["sub_question_list"])
        linked_item_list:List[List] = list()
        all_linked_items:List = list()
        '''sub_question_num = 0, 则链接结果为空'''
        if sub_question_num > 0:
            top_k_entities_per_sub_question = math.ceil(int(args["top_k_entities_per_question"]) / sub_question_num)
            top_k_classes_per_sub_question = math.ceil(int(args["top_k_classes_per_question"]) / sub_question_num)
            
            for sub_question in example["sub_question_list"]:
                serialized_sub_question = DELIMETER.join(sub_question)
                _linked = entity_linker.entity_linking_clause(
                    serialized_sub_question, 
                    top_k_entities_per_sub_question,
                    top_k_classes_per_sub_question
                )
                linked_item_list.append(_linked)
                all_linked_items.extend(_linked)
        else:
            logger.info(f"qid: {example['qid']}; sub_question_list: {example['sub_question_list']}")

        example["linked_item_list"] = linked_item_list
        if "grounded_items" in example:
            example["golden_grounded_items"] = copy.deepcopy(example["grounded_items"])
            example.pop("grounded_items", None)
        if "grounded_item_to_label" in example:
            example.pop("grounded_item_to_label", None)

        linked_item_to_label = dict()
        for item in all_linked_items:
            item_type = item["type"]
            item_mid = item["mid"]
            if item_type == "entity":
                linked_item_to_label[item_mid] = sparql_querier.get_friendly_name({
                    "mid": item_mid,
                    "type": "entity"
                })
            elif item_type == "literal":
                linked_item_to_label[item_mid] = extract_value_from_literal(item_mid)
            elif item_type == "class":
                if item_mid.startswith('m.') or item_mid.startswith('g.'):
                    linked_item_to_label[item_mid] = sparql_querier.get_friendly_name({
                        "mid": item_mid,
                        "type": "class"
                    })
                    # 目录类型的，就不查询 label 了（本身可读性就比较高）
            elif item_type is None: # 会排除掉格式比较奇怪的 class
                logger.error(f"linked_item: {item}")
            else:
                raise NotImplementedError(f"linked_item: {item}")
        example["linked_item_to_label"] = linked_item_to_label

        golden_item_to_label = dict()
        for item in example["golden_grounded_items"]:
            item_type = item["type"]
            item_mid = item["mid"]
            if item_type == "entity":
                golden_item_to_label[item_mid] = sparql_querier.get_friendly_name({
                    "mid": item_mid,
                    "type": "entity"
                })
            elif item_type == "literal":
                golden_item_to_label[item_mid] = extract_value_from_literal(item_mid)
            elif item_type == "class":
                if item_mid.startswith('m.') or item_mid.startswith('g.'):
                    golden_item_to_label[item_mid] = sparql_querier.get_friendly_name({
                        "mid": item_mid,
                        "type": "class"
                    })
            elif item_type is None: # 会排除掉格式比较奇怪的 class
                logger.error(f"linked_item: {item}")
            else:
                raise NotImplementedError(f"linked_item: {item}")
        example["golden_item_to_label"] = golden_item_to_label
    
    # if dataset.lower() == 'cwq':
    #     if (start is not None) and (end is not None):
    #         dump_json(data, f"data/input/cwq/cwq_{split}_{start}_{end}_linking.json")
    #     else:
    #         dump_json(data, f"data/input/cwq/cwq_{split}_linking.json")
    if dataset.lower() == 'cwq':
        if (start is not None) and (end is not None):
            dump_json(data, f"/home5/yhbao/data/cwq/cwq_{split}_{start}_{end}_linking.json")
        else:
            dump_json(data, f"/home5/yhbao/data/cwq/cwq_{split}_linking.json")
    elif dataset.lower() == 'webqsp':
        if (start is not None) and (end is not None):
            dump_json(data, f"/home5/yhbao/data/webqsp/webqsp_{split}_{start}_{end}_linking.json")
        else:
            dump_json(data, f"/home5/yhbao/data/webqsp/webqsp_{split}_linking.json")
    else:
        raise NotImplementedError(f"dataset: {dataset}")

def parse_and_entity_linking_wo_decomposition(dataset, split, start=200, end=1200):
    """
    功能实现上 和 parse_and_entity_linking 一致，只有文件名上的区别（不想放到同一个函数里面，怕整出 bug）
    """
    args = dict()
    args["top_k_per_mention"] = 5
    args["top_k_entities_per_question"] = 10
    args["top_k_classes_per_question"] = 5

    sparql_querier = SparqlOdbcQuerier(
        ODBC_CONFIG_OFFICIAL, SPARQL_wrapper_path_official, logger, timeout=60
    )
    entity_linker = EntityLinker.instance(
        logger, sparql_querier, 
        top_k_per_mention=args["top_k_per_mention"],
    )

    if dataset.lower() == 'cwq':
        '''CWQ'''
        if (start is not None) and (end is not None):
            data = load_json(f"data/input/cwq/intermediate/cwq_{split}_{start}_{end}_decomposition_wo_decomp.json")
        else:
            data = load_json(f"data/input/cwq/intermediate/cwq_{split}_decomposition_wo_decomp.json")
    elif dataset.lower() == 'webqsp':
        '''WebQSP'''
        if (start is not None) and (end is not None):
            data = load_json(f"data/input/WebQSP/intermediate/webqsp_{split}_{start}_{end}_decomposition_wo_decomp.json")
        else:
            data = load_json(f"data/input/WebQSP/intermediate/webqsp_{split}_decomposition_wo_decomp.json")
    else:
        raise NotImplementedError(f"dataset: {dataset}")

    for example in tqdm(data):
        sub_question_num = len(example["sub_question_list"])
        linked_item_list:List[List] = list()
        all_linked_items:List = list()
        '''sub_question_num = 0, 则链接结果为空'''
        if sub_question_num > 0:
            top_k_entities_per_sub_question = math.ceil(int(args["top_k_entities_per_question"]) / sub_question_num)
            top_k_classes_per_sub_question = math.ceil(int(args["top_k_classes_per_question"]) / sub_question_num)
            
            for sub_question in example["sub_question_list"]:
                serialized_sub_question = DELIMETER.join(sub_question)
                _linked = entity_linker.entity_linking_clause(
                    serialized_sub_question, 
                    top_k_entities_per_sub_question,
                    top_k_classes_per_sub_question
                )
                linked_item_list.append(_linked)
                all_linked_items.extend(_linked)
        else:
            logger.info(f"qid: {example['qid']}; sub_question_list: {example['sub_question_list']}")

        example["linked_item_list"] = linked_item_list
        if "grounded_items" in example: # 历史遗留问题，key 可能不一样
            example["golden_grounded_items"] = copy.deepcopy(example["grounded_items"])
            example.pop("grounded_items", None)
        if "grounded_item_to_label" in example:
            example.pop("grounded_item_to_label", None)

        linked_item_to_label = dict()
        for item in all_linked_items:
            item_type = item["type"]
            item_mid = item["mid"]
            if item_type == "entity":
                linked_item_to_label[item_mid] = sparql_querier.get_friendly_name({
                    "mid": item_mid,
                    "type": "entity"
                })
            elif item_type == "literal":
                # item_mid = process_linked_literal(item_mid) # 不应该有这个处理，处理之后可能 key 就对不上了
                linked_item_to_label[item_mid] = extract_value_from_literal(item_mid)
            elif item_type == "class":
                if item_mid.startswith('m.') or item_mid.startswith('g.'):
                    linked_item_to_label[item_mid] = sparql_querier.get_friendly_name({
                        "mid": item_mid,
                        "type": "class"
                    })
                    # 目录类型的，就不查询 label 了（本身可读性就比较高）
            elif item_type is None: # 会排除掉格式比较奇怪的 class
                logger.error(f"linked_item: {item}")
            else:
                raise NotImplementedError(f"linked_item: {item}")
        example["linked_item_to_label"] = linked_item_to_label

        golden_item_to_label = dict()
        for item in example["golden_grounded_items"]:
            item_type = item["type"]
            item_mid = item["mid"]
            if item_type == "entity":
                golden_item_to_label[item_mid] = sparql_querier.get_friendly_name({
                    "mid": item_mid,
                    "type": "entity"
                })
            elif item_type == "literal":
                # item_mid = process_linked_literal(item_mid) # 不应该有这个处理，处理之后可能 key 就对不上了
                golden_item_to_label[item_mid] = extract_value_from_literal(item_mid)
            elif item_type == "class":
                if item_mid.startswith('m.') or item_mid.startswith('g.'):
                    golden_item_to_label[item_mid] = sparql_querier.get_friendly_name({
                        "mid": item_mid,
                        "type": "class"
                    })
            elif item_type is None: # 会排除掉格式比较奇怪的 class
                logger.error(f"linked_item: {item}")
            else:
                raise NotImplementedError(f"linked_item: {item}")
        example["golden_item_to_label"] = golden_item_to_label
    
    if dataset.lower() == 'cwq':
        if (start is not None) and (end is not None):
            dump_json(data, f"data/input/cwq/cwq_{split}_{start}_{end}_linking_wo_decomp.json")
        else:
            dump_json(data, f"data/input/cwq/cwq_{split}_linking_wo_decomp.json")
    elif dataset.lower() == 'webqsp':
        if (start is not None) and (end is not None):
            dump_json(data, f"data/input/WebQSP/webqsp_{split}_{start}_{end}_linking_wo_decomp.json")
        else:
            dump_json(data, f"data/input/WebQSP/webqsp_{split}_linking_wo_decomp.json")
    else:
        raise NotImplementedError(f"dataset: {dataset}")


def add_manual():
    src_data = load_json("data/input/cwq/cwq_train_200_linking.json")
    annotated_data = load_json("data/input/cwq/cwq_train_200_linking_step1_manual.json")
    qid_to_annoated = {example["qid"]: example for example in annotated_data}
    for example in tqdm(src_data):
        qid = example["qid"]
        if qid in qid_to_annoated:
            clause_list_manual = qid_to_annoated[qid]["clause_list_manual"]
        else:
            clause_list_manual = []
        example["answer_centered_clause_list"] = clause_list_manual
        example["intermediate_entity_centered_clause_list"] = []
    dump_json(src_data, "data/input/cwq/cwq_train_200_linking_manual.json")

def parse_and_entity_linking_grailqa(split, start=200, end=1200):
    args = dict()
    args["top_k_per_mention"] = 5
    args["top_k_entities_per_question"] = 10
    args["top_k_classes_per_question"] = 5

    sparql_querier = SparqlOdbcQuerier(
        ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, timeout=60
    )
    entity_linker = EntityLinker.instance(
        logger, sparql_querier, 
        top_k_per_mention=args["top_k_per_mention"],
    )
    if (start is not None) and (end is not None):
        #/home5/yhbao/data/grailqa/grailqa_v1.0_train_linking.json
        data = load_json(f"/home5/yhbao/data/grailqa/grailqa_v1.0_{split}_{start}_{end}_linking.json")
    else:
        data = load_json(f"/home5/yhbao/data/grailqa/grailqa_v1.0_{split}_linking.json")

    for example in tqdm(data):
        sub_question_num = len(example["sub_question_list"])
        linked_item_list:List[List] = list()
        all_linked_items:List = list()
        '''sub_question_num = 0, 则链接结果为空'''
        if sub_question_num > 0:
            top_k_entities_per_sub_question = math.ceil(int(args["top_k_entities_per_question"]) / sub_question_num)
            top_k_classes_per_sub_question = math.ceil(int(args["top_k_classes_per_question"]) / sub_question_num)
            
            for sub_question in example["sub_question_list"]:
                serialized_sub_question = DELIMETER.join(sub_question)
                _linked = entity_linker.entity_linking_clause(
                    serialized_sub_question, 
                    top_k_entities_per_sub_question,
                    top_k_classes_per_sub_question
                )
                linked_item_list.append(_linked)
                all_linked_items.extend(_linked)
        else:
            logger.info(f"qid: {example['qid']}; sub_question_list: {example['sub_question_list']}")
        
        example["linked_item_list"] = linked_item_list
        if "grounded_items" in example:
            example["golden_grounded_items"] = copy.deepcopy(example["grounded_items"])
            example.pop("grounded_items", None)
        if "grounded_item_to_label" in example:
            example.pop("grounded_item_to_label", None)
        
        linked_item_to_label = dict()
        for item in all_linked_items:
            item_type = item["type"]
            item_mid = item["mid"]
            if item_type == "entity":
                linked_item_to_label[item_mid] = sparql_querier.get_friendly_name({
                    "mid": item_mid,
                    "type": "entity"
                })
            elif item_type == "literal":
                linked_item_to_label[item_mid] = extract_value_from_literal(item_mid)
            elif item_type == "class":
                if item_mid.startswith('m.') or item_mid.startswith('g.'):
                    linked_item_to_label[item_mid] = sparql_querier.get_friendly_name({
                        "mid": item_mid,
                        "type": "class"
                    })
            elif item_type is None: # 会排除掉格式比较奇怪的 class
                logger.error(f"linked_item: {item}")
            else:
                raise NotImplementedError(f"linked_item: {item}")
        example["linked_item_to_label"] = linked_item_to_label

        golden_item_to_label = dict()
        for item in example["golden_grounded_items"]:
            item_type = item["type"]
            item_mid = item["mid"]
            if item_type == "entity":
                golden_item_to_label[item_mid] = sparql_querier.get_friendly_name({
                    "mid": item_mid,
                    "type": "entity"
                })
            elif item_type == "literal":
                golden_item_to_label[item_mid] = extract_value_from_literal(item_mid)
            elif item_type == "class":
                if item_mid.startswith('m.') or item_mid.startswith('g.'):
                    golden_item_to_label[item_mid] = sparql_querier.get_friendly_name({
                        "mid": item_mid,
                        "type": "class"
                    })
            elif item_type is None: # 会排除掉格式比较奇怪的 class
                logger.error(f"linked_item: {item}")
            else:
                raise NotImplementedError(f"linked_item: {item}")
        example["golden_item_to_label"] = golden_item_to_label
    
    if (start is not None) and (end is not None):
        dump_json(data, f"/home5/yhbao/data/grailqa/grailqa_v1.0_{split}_{start}_{end}_linking_new.json")
    else:
        dump_json(data, f"/home5/yhbao/data/grailqa/grailqa_v1.0_{split}_linking_new.json")

def parse_and_entity_linking_grailqa_wo_decomposition(split, start=200, end=1200):
    args = dict()
    args["top_k_per_mention"] = 5
    args["top_k_entities_per_question"] = 10
    args["top_k_classes_per_question"] = 5

    sparql_querier = SparqlOdbcQuerier(
        ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab, logger, timeout=60
    )
    entity_linker = EntityLinker.instance(
        logger, sparql_querier, 
        top_k_per_mention=args["top_k_per_mention"],
    )
    if (start is not None) and (end is not None):
        data = load_json(f"data/input/GrailQA_v1.0/intermediate/grailqa_v1.0_{split}_{start}_{end}_decomposition_wo_decomp.json")
    else:
        data = load_json(f"data/input/GrailQA_v1.0/intermediate/grailqa_v1.0_{split}_decomposition_wo_decomp.json")

    for example in tqdm(data):
        sub_question_num = len(example["sub_question_list"])
        linked_item_list:List[List] = list()
        all_linked_items:List = list()
        '''sub_question_num = 0, 则链接结果为空'''
        if sub_question_num > 0:
            top_k_entities_per_sub_question = math.ceil(int(args["top_k_entities_per_question"]) / sub_question_num)
            top_k_classes_per_sub_question = math.ceil(int(args["top_k_classes_per_question"]) / sub_question_num)
            

            for sub_question in example["sub_question_list"]:
                serialized_sub_question = DELIMETER.join(sub_question)
                _linked = entity_linker.entity_linking_clause(
                    serialized_sub_question, 
                    top_k_entities_per_sub_question,
                    top_k_classes_per_sub_question
                )
                linked_item_list.append(_linked)
                all_linked_items.extend(_linked)
        else:
            logger.info(f"qid: {example['qid']}; sub_question_list: {example['sub_question_list']}")
        
        example["linked_item_list"] = linked_item_list
        if "grounded_items" in example:
            example["golden_grounded_items"] = copy.deepcopy(example["grounded_items"])
            example.pop("grounded_items", None)
        if "grounded_item_to_label" in example:
            example.pop("grounded_item_to_label", None)
        
        linked_item_to_label = dict()
        for item in all_linked_items:
            item_type = item["type"]
            item_mid = item["mid"]
            if item_type == "entity":
                linked_item_to_label[item_mid] = sparql_querier.get_friendly_name({
                    "mid": item_mid,
                    "type": "entity"
                })
            elif item_type == "literal":
                # item_mid = process_linked_literal(item_mid)
                linked_item_to_label[item_mid] = extract_value_from_literal(item_mid)
            elif item_type == "class":
                if item_mid.startswith('m.') or item_mid.startswith('g.'):
                    linked_item_to_label[item_mid] = sparql_querier.get_friendly_name({
                        "mid": item_mid,
                        "type": "class"
                    })
            elif item_type is None: # 会排除掉格式比较奇怪的 class
                logger.error(f"linked_item: {item}")
            else:
                raise NotImplementedError(f"linked_item: {item}")
        example["linked_item_to_label"] = linked_item_to_label

        golden_item_to_label = dict()
        for item in example["golden_grounded_items"]:
            item_type = item["type"]
            item_mid = item["mid"]
            if item_type == "entity":
                golden_item_to_label[item_mid] = sparql_querier.get_friendly_name({
                    "mid": item_mid,
                    "type": "entity"
                })
            elif item_type == "literal":
                # item_mid = process_linked_literal(item_mid)
                golden_item_to_label[item_mid] = extract_value_from_literal(item_mid)
            elif item_type == "class":
                if item_mid.startswith('m.') or item_mid.startswith('g.'):
                    golden_item_to_label[item_mid] = sparql_querier.get_friendly_name({
                        "mid": item_mid,
                        "type": "class"
                    })
            elif item_type is None: # 会排除掉格式比较奇怪的 class
                logger.error(f"linked_item: {item}")
            else:
                raise NotImplementedError(f"linked_item: {item}")
        example["golden_item_to_label"] = golden_item_to_label
    
    if (start is not None) and (end is not None):
        dump_json(data, f"data/input/GrailQA_v1.0/grailqa_v1.0_{split}_{start}_{end}_linking_wo_decomp.json")
    else:
        dump_json(data, f"data/input/GrailQA_v1.0/grailqa_v1.0_{split}_linking_wo_decomp.json")


def prepare_time_relations():
    sparql_querier = SparqlOdbcQuerierExps(
        ODBC_CONFIG_OFFICIAL,
        SPARQL_wrapper_path_official,
        logger, timeout=360
    )
    relations_by_domain = sparql_querier.get_time_related_relations()
    dump_json(relations_by_domain, "data/input/common/time_relations_by_domain.json")

def collect_time_relation_pairs():
    """
    每个 domain 下，我们遍历所有关系组合，记录每个关系的"对偶关系"构成的 dictionary
    """
    relations_by_domain = load_json("data/input/common/time_relations_by_domain.json")
    relation_pairs = defaultdict(set)
    for relation_list in relations_by_domain.values():
        for rel in relation_list:
            for dual_rel in relation_list:
                if rel != dual_rel:
                    relation_pairs[rel].add(dual_rel)
    relation_pairs = {k: list(v) for (k, v) in relation_pairs.items()}
    dump_json(relation_pairs, "data/input/common/time_relation_pairs.json")


def prepare_category_class():
    '''使用关系 type.object.type, 获取所有 目录 形式的 class'''
    sparql_querier = SparqlOdbcQuerier(
        ODBC_CONFIG_OFFICIAL,
        SPARQL_wrapper_path_official,
        logger, timeout=360
    )
    types = sparql_querier.get_all_category_types()
    dump_json(types, "data/input/common/category_types.json")

def prepare_class_label():
    sparql_querier = SparqlOdbcQuerier(
        ODBC_CONFIG_OFFICIAL,
        SPARQL_wrapper_path_official,
        logger, timeout=10
    )
    class_set = load_json("data/input/common/category_types.json")
    class_to_label = dict()
    for class_item in tqdm(class_set):
        label = sparql_querier.get_friendly_name({
            "mid": class_item, "type": "class"
        })
        if label: # 过滤空字符串
            class_to_label[class_item] = label.lower()
    dump_json(class_to_label, "data/input/common/category_class_map.json")

def prepare_label_class_map():
    """
    class 和 label 之间是一对一映射，并且 freebase 上应该不存在重复的 class
    label 和 class 之间应该是一对多映射，我们除了维护这个映射，还应该做一些简单的排序
    - user.* class, 得分 5
    - base.* class, 得分 8
    - 其他，得分 10
    """
    class_to_label = load_json("data/input/common/category_class_map.json")
    label_to_class = defaultdict(list)
    for (class_item, label) in class_to_label.items():
        if class_item.startswith('user.'):
            tup = (class_item, 5)
        elif class_item.startswith('base.'):
            tup = (class_item, 8)
        else:
            tup = (class_item, 10)
        label_to_class[label].append(tup)
    dump_json(label_to_class, "data/input/common/label_to_category_class.json")

def lemmatize_label_to_category_class():
    from nltk.stem import WordNetLemmatizer
    from nltk import word_tokenize
    wlem = WordNetLemmatizer()

    def get_lemmatized(name):
        name = name.lower()
        token_list = word_tokenize(name)
        token_list = [wlem.lemmatize(tok) for tok in token_list]
        lemmatized_name = ' '.join(token_list)
        lemmatized_name = lemmatized_name.replace('``', '"').replace("''", '"')
        return lemmatized_name
    
    label_to_class = load_json("data/input/common/label_to_category_class.json")
    print(f"label_to_class: {len(label_to_class)}")
    lemmatized_label_to_class = defaultdict(list)
    for (label, class_list) in label_to_class.items():
        lemmatized_label = get_lemmatized(label)
        # 两个不同的 key, lemmatized 之后可能想通了
        lemmatized_label_to_class[lemmatized_label].extend(class_list)
    print(f"lemmatized_label_to_class: {len(lemmatized_label_to_class)}")
    dump_json(lemmatized_label_to_class, "data/input/common/lemmatized_label_to_category_class.json")


def analyze_templates_lc_2():
    train_data = load_json("data/input/LC_QuAD_2/train.json")
    template_to_example = dict()
    count = 0
    for example in tqdm(train_data):
        try:
            template = example["template"].lower()
            if template not in template_to_example:
                template_to_example[template] = example
        except Exception as e:
            count += 1
            print(f"uid: {example['uid']}; error: {e}")
    print(f"count: {count}")
    dump_json(list(template_to_example.values()), "data/input/LC_QuAD_2/examples.json")

def analyze_entities_kqapro():
    train_data = load_json("data/input/kqapro/val.json")
    qid_list = set()
    entity_pattern = r'".+?"'
    print(f"total: {len(train_data)}")
    for (idx, example) in tqdm(enumerate(train_data)):
        for grounded_ent in re.findall(entity_pattern, example["sparql"]):
            if grounded_ent[1:-1].lower() not in example["question"].lower():
                qid_list.add(idx)
    print(f"qid_list, {sorted(qid_list, reverse=True)} {len(qid_list)}")

def split_openai_keys(n=5):
    '''
    推荐每个分片使用 20 个 key, 每个查询记作 3s (设置 2s 的 sleep)
    那么每个 key 的压力大约是 1 / min, 不会超限
    总能力是 20 * 200 = 4000 / 天，故分片大小(数据量) 建议设置为 4000
    '''
    key_total = load_json("config/exclusive/openai_keys.json")
    print(f"key_total: {len(key_total)}")
    for idx in range(n):
        start = int(len(key_total) / n * idx)
        if idx == n-1:
            end = len(key_total)
        else:
            end = int(len(key_total) / n * (idx + 1))
        print(f"start: {start}; end: {end}")
        key_subset = key_total[start:end]
        dump_json(key_subset, f"config/exclusive/openai_keys_{idx}.json")

def read_openai_keys():
    file = open("config/openai_keys.txt", "r")
    keys_json = list()
    for line in file.readlines():
        key = line.split(':')[1].strip()
        keys_json.append(key)
    dump_json(keys_json, "config/openai_keys.json")

def append_openai_keys():
    original_keys = set(load_json("config/origin/openai_keys.json")) | set(load_json("config/1226/openai_keys_new.json"))
    new_keys = open("config/1226/openai_keys_append.txt", "r")
    keys_json = original_keys
    for line in new_keys.readlines():
        key = line.strip()
        keys_json.add(key)
    dump_json(list(keys_json), "config/shared/openai_keys.json")

def compare_webqsp():
    file_ours = load_json("data/input/WebQSP/intermediate/webqsp_train_processed.json")
    file_pangu = load_json("data/pangu/webqsp_0107.train.json")
    ours_qid_set = {item["qid"] for item in file_ours}
    pangu_qid_set = {item["qid"] for item in file_pangu}
    assert ours_qid_set == pangu_qid_set

def combine_composition_result(parent_dir):
    assert os.path.isdir(parent_dir)
    combined_results = list()
    for sub_dir in os.listdir(parent_dir):
        combined_results.extend(
            load_json(os.path.join(parent_dir, sub_dir, 'results.json'))
        )
    unique_keys = set([
        item["qid"] for item in combined_results
    ])
    logger.info(f"unique_keys: {len(unique_keys)}")
    dump_json(combined_results, os.path.join(parent_dir, 'results.json'))

def combine_detection_result(parent_dir):
    assert os.path.isdir(parent_dir)
    combined_results = list()
    summary_result = {
        "总数": list(),
        "best_precision=1的数量": list(),
        "平均搜索时间": list()
    }
    for sub_dir in os.listdir(parent_dir):
        sub_result = load_json(os.path.join(parent_dir, sub_dir, 'final_results.json'))
        combined_results.extend(sub_result["results"])
        for key in sub_result["summary"]:
            if key == "平均搜索时间":
                summary_result[key].append(sub_result["summary"][key] * sub_result["summary"]["总数"])
            else:
                summary_result[key].append(sub_result["summary"][key])
    summary_result_final = {
        "total": sum(summary_result["总数"]),
        "Prec.=1_count": sum(summary_result["best_precision=1的数量"]),
        "average_search_time": sum(summary_result["平均搜索时间"]) / sum(summary_result["总数"])
    }
    unique_keys = set([
        item["qid"] for item in combined_results
    ])
    logger.info(f"unique_keys: {len(unique_keys)}")
    dump_json({
        "results": combined_results,
        "summary": summary_result_final
    }, os.path.join(parent_dir, 'final_results.json'))



def process_lc2(split='train'):
    lc_data = load_json(f"data/input/LC_QuAD_2/{split}.json")
    sparql_querier = SparqlOdbcQuerierWikidata(
        ODBC_CONFIG_WIKIDATA_2019,
        SPARQL_wrapper_path_wikidata_2019,
        logger,
        timeout=60
    )
    processed_examples = list()

    for example in tqdm(lc_data):
        grounded_items = list()
        answer = list()
        try:
            s_expression = SparqlParserLC2.parse_sparql(example["sparql_wikidata"])
        except Exception as e:
            logger.info(f"error: {e}")
            s_expression = 'null'
        grounded_entities, grounded_literals, _ = extract_grounded_items_from_sexpr_wikidata(s_expression)
        if len(grounded_entities) == 0 and len(grounded_literals) == 0:
            grounded_entities, grounded_literals = extract_grounded_items_from_sparql_wikidata(example["sparql_wikidata"])
        for ent in grounded_entities:
            grounded_items.append({
                "mid": ent,
                "type": "entity"
            })
        for literal in grounded_literals:
            grounded_items.append({
                "mid": literal,
                "type": "literal"
            })
        answer = sparql_querier.get_execution_result_one_variable_sparql_wrapper(example["sparql_wikidata"])
        answer = [{
            "mid": ans,
            "type": convert_wikidata_type(WikidataConstantForConstruction.get_constant_type(ans))
        } for ans in answer]
            
        question = example["paraphrased_question"] # LC-QuAD 的相关文档和论文中说，paraphrased_question 是众包改写之后的问题
        if not question:
            question = example['question']
        if not question:
            question = "null" # 执行起来不报错

        processed_examples.append({
            "qid": example["uid"],
            "question": question, 
            "answer": answer,
            "grounded_items": grounded_items,
            "golden_sparql_query": example["sparql_wikidata"],
            "golden_s_expression": s_expression
        })
    
    dump_json(
        processed_examples,
        f"data/input/LC_QuAD_2/intermediate/lc2_{split}_processed.json"
    ) 

def lc2_add_golden_items(split, start=None, end=None):
    from components.sparql_utils import SparqlParserLC2
    sparql_querier = SparqlOdbcQuerierWikidata(
        ODBC_CONFIG_WIKIDATA_2019, SPARQL_wrapper_path_wikidata_2019, logger, timeout=10
    )
    if (start is not None) and (end is not None):
        data = load_json(f"data/input/LC_QuAD_2/intermediate/lc2_{split}_{start}_{end}_decomposition.json")
    else:
        data = load_json(f"data/input/LC_QuAD_2/intermediate/lc2_{split}_decomposition.json")
    
    for example in tqdm(data):
        example["golden_grounded_items"] = copy.deepcopy(example["grounded_items"])
        example.pop("grounded_items", None)
        golden_item_to_label = dict()
        for item in example["golden_grounded_items"]:
            item_type = item["type"]
            item_mid = item["mid"]
            if item_type == "entity":
                golden_item_to_label[item_mid] = sparql_querier.query_entity_label(item_mid[3:])["label"] # 去掉 wd: 前缀
            elif item_type == "literal":
                golden_item_to_label[item_mid] = extract_value_from_literal(item_mid)
            else: # 目前没有区分 entity 和 class
                logger.error(f"{item_type} {item_mid}")
        example["golden_item_to_label"] = golden_item_to_label

    if (start is not None) and (end is not None):
        dump_json(data, f"data/input/LC_QuAD_2/lc2_{split}_{start}_{end}.json")
    else:
        dump_json(data, f"data/input/LC_QuAD_2/lc2_{split}.json")

def combine_tmp_results(tmp_dir, start_idx=0, end_idx=80):
    """代码执行过程中可能因各种原因异常终止，根据暂存的分片信息，组合得到当前的执行结果"""
    combined_results = list()
    total_count = 0
    precision_list = list()
    search_time_list = list()
    for tmp_idx in trange(start_idx, end_idx):
        sub_result = load_json(os.path.join(tmp_dir, f'{tmp_idx}.json'))
        combined_results.extend(sub_result)
        total_count += len(sub_result)
        for search_item in sub_result:
            precision_list.append(search_item["searched_queries"][0]["precision"] if ("searched_queries" in search_item) else 0.0)
            search_time_list.append(search_item["time"])
    dump_json({
        "results": combined_results, 
        "summary_result": {
            "总数": total_count,
            "best_precision=1的数量": len([prec for prec in precision_list if math.isclose(prec, 1.0)]),
            "平均搜索时间": sum(search_time_list) / len(search_time_list)
        }
    }, os.path.join(tmp_dir, f"results_{start_idx}_{end_idx}.json"))

def prepare_wikidata_property_label_mapping():
    sparql_querier = SparqlOdbcQuerierWikidata(
        ODBC_CONFIG_WIKIDATA_2019, SPARQL_wrapper_path_wikidata_2019, logger, timeout=10
    )
    property_list = load_json("data/input/common/candidate_properties.json")
    tmp_dir = os.path.join("data/input/common/prop_to_label/")
    os.makedirs(tmp_dir, exist_ok=True)
    args = list()
    for prop in property_list:
        args.append({
            "entity": prop
        })
    label_response = sparql_querier.run(
        data=args,
        func=sparql_querier.query_entity_label,
        output_dir=tmp_dir,
        return_format='list',
        batch_size=500,
        max_workers=2
    )
    prop_to_label = dict()
    for resp_item in label_response:
        if resp_item['label']:
            prop_to_label[resp_item['entity']] = resp_item['label']
    
    dump_json(prop_to_label, "data/input/common/property_to_label.json")

def get_detection_result_subset(detection_result_path, subset_data_path, output_path):
    detection_result = load_json(detection_result_path)["results"]
    subset_data = load_json(subset_data_path)
    subset_qid_list = [item['qid'] for item in subset_data]
    detection_result_subset = [
        detection_item for detection_item in detection_result
        if detection_item['qid'] in subset_qid_list
    ]
    logger.info(f"detection_result_subset: {len(detection_result_subset)}")
    precision_1_count = 0
    time_list = list()
    for detection_item in detection_result_subset:
        time_list.append(detection_item['time'])
        if ("searched_queries" in detection_item) and len(detection_item["searched_queries"]) > 0:
            precision_1_count += 1
    
    dump_json({
        "results": detection_result_subset,
        "summary": {
            "总数": len(detection_result_subset),
            "best_precision=1的数量": precision_1_count,
            "平均搜索时间": sum(time_list) / len(time_list)
        }
    }, output_path)

def get_wikidata_reverse_property():
    sparql_querier = SparqlOdbcQuerierWikidata(
        ODBC_CONFIG_WIKIDATA_2019, SPARQL_wrapper_path_wikidata_2019, logger, timeout=10
    )
    results = sparql_querier.get_reverse_property()
    with open("data/input/common/reverse_properties_wikidata", "w") as f:
        for item in results:
            f.write(f"{item[0]}\t{item[1]}\n")

def lc2_add_new_s_expression():
    '''
    之前生成的 Sexp 有不完善之处，重新生成一遍
    同时增加 S-expression 执行结果的检查；如果和 SPARQL 的执行结果不一致，那么还是把 s_expression 置为 null (这样语义等价始终为 false, 编辑距离始终为 1.0)
    '''
    from components.s_expression_utils import sexp_to_sparql_wikidata
    sparql_querier = SparqlOdbcQuerierWikidata(
        ODBC_CONFIG_WIKIDATA_2019,
        SPARQL_wrapper_path_wikidata_2019,
        logger,
        timeout=60
    )
    old_data = load_json("data/input/LC_QuAD_2/before_sparql_modification/lc2_test_0_1000.json")
    for example in tqdm(old_data):
        try:
            s_expression = SparqlParserLC2.parse_sparql(example["golden_sparql_query"])
            converted_sparql = sexp_to_sparql_wikidata(s_expression)
            execution_result = sparql_querier.get_execution_result_one_variable_sparql_wrapper(converted_sparql)
            original_answer = {ans['mid'] for ans in example["answer"]}
            if original_answer != execution_result:
                s_expression = 'null'
        except Exception as e:
            # logger.info(f"error: {e}")
            s_expression = 'null'
        if s_expression != example['golden_s_expression']:
            print(f"old: {example['golden_s_expression']}")
            print(f"new: {s_expression}")
            print("")
        example['golden_s_expression'] = s_expression
    dump_json(old_data, "data/input/LC_QuAD_2/lc2_test_0_1000.json")

def cwq_add_new_s_expression():
    '''
    进行评价的时候，使用正确的格式；前面抽取的时候，可能有一些误差，对本方法是负面影响，这应该没问题
    后面可能直接对 SPARQL 做评测，数据集中的 SPARQL 都是官方给出的，没有误差
    '''
    from components.sparql_utils import SparqlParserFreebase
    old_data = load_json("data/input/cwq/cwq_test_0_1000_linking.json")
    for example in tqdm(old_data):
        try:
            s_expression = SparqlParserFreebase.parse_sparql(example["golden_sparql_query"])
        except:
            s_expression = 'null'
        if s_expression != example['golden_s_expression']:
            logger.info(f"old: {example['golden_s_expression']}")
            logger.info(f"new: {s_expression}")
            example['golden_s_expression'] = s_expression
    dump_json(old_data, "data/input/cwq/cwq_test_0_1000_linking.json")

def webqsp_add_new_s_expression():
    from components.sparql_utils import SparqlParserFreebase
    old_data = load_json("data/input/WebQSP/webqsp_test_0_1000_linking.json")
    for example in tqdm(old_data):
        try:
            s_expression = SparqlParserFreebase.parse_sparql(example["golden_sparql_query"])
        except:
            s_expression = 'null'
        if s_expression != example['golden_s_expression']:
            logger.info(f"old: {example['golden_s_expression']}")
            logger.info(f"new: {s_expression}")
            example['golden_s_expression'] = s_expression
    dump_json(old_data, "data/input/WebQSP/webqsp_test_0_1000_linking.json")

def post_process_lc2():
    """
    0526: 例如将 SPARQL 中的 limit 5 替换为 limit 1 等操作
    """
    lc_data = load_json("/home/home2/xwu/Experiments_FreeBase/data/input/LC_QuAD_2/lc2_test_0_1000.json")
    sparql_querier = SparqlOdbcQuerierWikidata(
        ODBC_CONFIG_WIKIDATA_2019,
        SPARQL_wrapper_path_wikidata_2019,
        logger,
        timeout=60
    )
    processed_examples = list()
    
    for example in tqdm(lc_data):
        try:
            sparql_updated = f"{WIKIDATA_PREFIX_LIST}\n{example['golden_sparql_query']}"
            editor = SyntaxTreeEditor(sparql_updated, DATASET.LC2, logger)
            editor.manage_limit_lc2()
            sparql_updated = editor.sparql_txt # sparql 仅仅是把 limit 5 更新为 limit 1, 对于 grounded items 的抽取不会有影响
            
            answer = sparql_querier.get_execution_result_one_variable_sparql_wrapper(sparql_updated)
            answer = [{
                "mid": ans,
                "type": convert_wikidata_type(WikidataConstantForConstruction.get_constant_type(ans))
            } for ans in answer]
        except Exception as e:
            logger.error(f"qid: {example['qid']}")
            answer = []
            sparql_updated = None

        processed_examples.append({
            "qid": example["qid"],
            "question": example["question"], 
            "answer": answer,
            "golden_sparql_query": sparql_updated,
            "golden_s_expression": example["golden_s_expression"],
            "golden_grounded_items": example["golden_grounded_items"],
            "golden_item_to_label": example["golden_item_to_label"],
            "old_sparql": example["golden_sparql_query"]
        })
    
    dump_json(processed_examples, f"data/input/LC_QuAD_2/intermediate/lc2_test_0_1000.json")

def get_wikidata_numeral_properties():
    import csv
    def load_csv_as_dict(csv_file):
        data_dict = {}
        with open(csv_file, mode='r', encoding='utf-8-sig') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                key = row.pop('ID')  # 替换为你的键的列名
                data_dict[key] = row
        return data_dict
    quantity_properties = []
    temporal_properties = []
    data_dict = load_csv_as_dict("/home5/yhbao/SPARQLannotation/data/WD_ontology/all_wikidata_property.csv")
    for k, v in data_dict.items():
        if v['datatype'] == "Q":
            quantity_properties.append({"pid":k, "label": v['label']})
        if v['datatype'] == "T":
            temporal_properties.append({"pid":k, "label": v['label']})
    res_dict = {"temporal_properties":temporal_properties, "quantity_properties":quantity_properties}
    dump_json(res_dict, "/home5/yhbao/SPARQLannotation/data/WD_ontology/wikidata_numeral_property_dict.json")

def get_wikidata_property_id2label_dict():
    import csv
    def load_csv_as_dict(csv_file):
        data_dict = {}
        with open(csv_file, mode='r', encoding='utf-8-sig') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                key = row.pop('ID')  # 替换为你的键的列名
                data_dict[key] = row
        return data_dict
    
    res_dict = {}
    data_dict = load_csv_as_dict("/home5/yhbao/SPARQLannotation/data/WD_ontology/all_wikidata_property.csv")
    for k, v in data_dict.items():
        res_dict[k] = v['label']
    dump_json(res_dict, "/home5/yhbao/SPARQLannotation/data/WD_ontology/wikidata_property_id2label_dict.json")

def relace_el_results_by_queryagent_results_grailqa(data_dir, output_dir):
    #将原先格式的grailqa数据文件的实体连接部分替换为
    queryagent_results = load_json("/home5/yhbao/data/grailqa/grail_pangu_tiara.json")
    data = load_json(data_dir)
    new_data = []
    for item in tqdm(data):
        qid = item['qid']
        #要修改的域有linked_item_list与linked_item_to_label
        #我们只需要保留linked_classes
        new_linked_results = []
        new_linked_item_to_label = {}
        linked_classes = []
        for source in item['linked_item_list']:
            for r in source:
                if r['type'] == "class":
                    linked_classes.append(r)
        for label, mid_list in queryagent_results[str(qid)].items():
            mid = mid_list[0]
            ty = "literal" if "XMLSchema" in mid else "entity"
            new_linked_item_to_label[mid] = label
            new_linked_results.append({"mid":mid, "type":ty})
        new_linked_results += linked_classes
        item['linked_item_list']=[new_linked_results]
        item['linked_item_to_label'] = new_linked_item_to_label
        new_data.append(item)
    dump_json(new_data, output_dir)


if __name__=='__main__':
    # import json
    # def read_jsonl(file_path):    
    #     data = []    
    #     with open(file_path, 'r') as file:        
    #         for line in file:           
    #             obj = json.loads(line)         
    #             data.append(obj)    
    #     return data
    # data = read_jsonl("/home5/yhbao/data/SPARQL_annotation_dataset/NL_Open.jsonl")
    # dump_json(data, "/home5/yhbao/data/SPARQL_annotation_dataset/NQ_open.json")
    # 读取JSONL文件示例jsonl_data = read_jsonl('data.jsonl')for obj in jsonl_data:    print(obj)
    #get_wikidata_property_id2label_dict()
    #get_Wikidata_class_embedding()
    #选择CWQ没有数值约束的部分做测试
    #get_wikidata_numeral_properties()
    # data = load_json("/home5/yhbao/data/cwq/cwq_dev_linking.json")
    # new_data = []
    # for item in data:
    #     if "CMP" not in item['golden_s_expression'] and "dateTime" not in item['golden_sparql_query']:
    #         new_data.append(item)
    # dump_json(new_data, "/home5/yhbao/SPARQLannotation/data/cwq_dev_wo_nc.json")
    ###重跑CWQ的实体链接
    '''CWQ train'''
    # parse_and_entity_linking('cwq', 'dev', start=None, end=None)
    # parse_and_entity_linking('cwq', 'test', start=None, end=None)
    # parse_and_entity_linking('cwq', 'train', start=None, end=None)
    # process_cwq(split='train')
    # parsing_and_add_decomposition('cwq', 'train', 'cwq_decomposition_train_2023-12-27', start=None, end=None)
    # parse_and_entity_linking('cwq', 'train', start=None, end=None)
    #combine_detection_result('data/output/query_construction/cwq/cwq_train_starQC_2024-04-19')

    '''CWQ dev'''
    # process_cwq(split='dev')
    # parsing_and_add_decomposition('cwq', 'dev', 'cwq_decomposition_dev_2024-02-24', start=None, end=None)
    # parse_and_entity_linking('cwq', 'dev', start=None, end=None)

    '''CWQ test'''
    # process_cwq(split='test')
    # parsing_and_add_decomposition('cwq', 'test', 'cwq_decomposition_test_2024-03-05', start=None, end=None)
    # parse_and_entity_linking('cwq', 'test', start=None, end=None)
    
    '''CWQ test 1000 在 CWQ test 的基础上采样得到，但是使用新的分解结果（gpt-0613版本），并且实体链接也要重做'''
    # sample_data(dataset='cwq', split='test', start=0, end=1000) # 采样放到最后面
    # parsing_and_add_decomposition_patch(
    #     'data/input/cwq/cwq_test_0_1000_linking.json',
    #     'data/output/decomposition/cwq/cwq_test_0_1000_2024-05-19_09:44:17/results.json',
    #     'data/input/cwq/intermediate/cwq_test_0_1000_decomposition.json'
    # )
    # parse_and_entity_linking('cwq', 'test', start=0, end=1000)


    '''WebQSP train'''
    # process_webqsp(split='train')
    # parsing_and_add_decomposition('webqsp', 'train', 'webqsp_decomposition_train_2023-12-26', start=None, end=None)
    parse_and_entity_linking('webqsp', 'train', start=None, end=None)
    
    '''WebQSP test 和 WebQSP test 1000'''
    # process_webqsp(split='test')
    # parsing_and_add_decomposition('webqsp', 'test', 'webqsp_decomposition_test_2024-03-05', start=None, end=None)
    parse_and_entity_linking('webqsp', 'test', start=None, end=None)
    # add_golden_sparql_list_webqsp('test')
    '''WebQSP test 1000 在 WebQSP test 的基础上采样得到，但是使用新的分解结果（gpt-0613版本），并且实体链接也要重做'''
    # sample_data(dataset='webqsp', split='test', start=0, end=1000) # 放到最后
    # parsing_and_add_decomposition_patch(
    #     'data/input/WebQSP/webqsp_test_0_1000_linking.json',
    #     'data/output/decomposition/webqsp/webqsp_test_0_1000_2024-05-19_12:27:40/results.json',
    #     'data/input/WebQSP/intermediate/webqsp_test_0_1000_decomposition.json'
    # )
    #parse_and_entity_linking('webqsp', 'test', start=0, end=1000)


    '''GrailQA train'''
    # process_grailqa(split='train')
    # parsing_and_add_decomposition('grailqa', 'train', 'grailqa_decomposition_train_2023-12-29', start=None, end=None)
    # parse_and_entity_linking_grailqa('train', start=None, end=None)
    # parse_and_entity_linking_grailqa('dev', start=None, end=None)
    #combine_detection_result('data/output/query_construction/grailqa/grailqa_train_starQC_2024_05_05')

    '''GrailQA dev 和 GrailQA dev 1000'''
    # process_grailqa(split='dev')
    # parsing_and_add_decomposition('grailqa', 'dev', 'grailqa_decomposition_dev_2024-05-09_10:34:08', start=None, end=None)
    # parse_and_entity_linking_grailqa('dev', start=None, end=None)
    # sample_data(dataset='grailqa', split='dev', start=0, end=1000) # 放到最后

    '''LC-QuAD2 test 1000'''
    '''基于 Experiment_Freebase 下的数据，做进一步的处理'''
    # post_process_lc2()
    # parsing_and_add_decomposition_patch(
    #     'data/input/LC_QuAD_2/intermediate/lc2_test_0_1000.json',
    #     'data/output/decomposition/lc2/lc2_test_0_1000_2024-05-19_21:45:10/results.json',
    #     'data/input/LC_QuAD_2/lc2_test_0_1000.json'
    # )

    '''QALD test'''
    # process_qald_wikidata('test') # 没有实际执行，从 Experiment_Freebase 复制过来
    # parsing_and_add_decomposition_patch(
    #     'data/input/QALD/intermediate/qald_wikidata_test_processed.json',
    #     'data/output/decomposition/qald/qald_test_2024-05-24_15:41:42/results.json',
    #     'data/input/QALD/intermediate/qald_test_decomposition.json'
    # )
    # qald_add_golden_items('test', start=None, end=None)

    '''Wikidata'''
    # prepare_wikidata_property_label_mapping()


    '''GrailQA'''
    # combine_tmp_results_and_summary(
    #     "data/output/query_construction/grailqa/grailqa_train_starQC_2024_05_05/grailqa_train_11500_14600_starQC_2024-04-30_19:59:07/_tmp",
    #     start_idx=0, end_idx=62
    # )
    # construct_dataset_with_simulated_queries_grailqa()

    '''CWQ'''
    # combine_tmp_results_and_summary(
    #     "data/output/cwq_train_processed_2023-10-22_22:13:39/_tmp",
    #     start_idx=0, end_idx=12
    # )
    # construct_dataset_with_simulated_queries_cwq(
    #     reserve_all_original_examples=False
    # )

    '''KQA-Pro'''
    # analyze_entities_kqapro()

    '''公共数据准备'''
    # prepare_category_class()
    # prepare_class_label()
    # prepare_label_class_map()
    # lemmatize_label_to_category_class()

    # split_openai_keys(n=5)
    # compare_webqsp()

    # combine_tmp_results('data/output/grailqa_train_22000_26000_cvtRetrieval_2024-03-12_21:13:07/_tmp', 0, 35)
    # get_wikidata_reverse_property()