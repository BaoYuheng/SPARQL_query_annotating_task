from src.utils import load_json, dump_json
from src.s_expression_utils import sexp_to_sparql_for_edit_distance
from src.logic_form_util import lisp_to_sparql
import re
import math


import numpy as np
from scipy.stats import kendalltau, pointbiserialr



def manage_xsd_type(str):
    str = str.replace(")", " ) ").replace("(", " ( ")
    for xsd_type in ['decimal', 'integer', 'float', 'date', 'gYear', 'gYearMonth', 'dateTime']:
        str = str.replace(f"^^http://www.w3.org/2001/XMLSchema#{xsd_type} ", f"^^<http://www.w3.org/2001/XMLSchema#{xsd_type}> ") 
    str = str.replace(" ) ", ")").replace(" ( ", "(")
    return str


def calculate_metrics_false_positive_negative(thresold_tss = 1):
    data = load_json("/home5/whzhou/STAR-QC/data/final_test/merge_all_select_anno_ted0_last.json")
    tss_fn = 0
    tss_fp = 0
    f1_fp = 0
    ted_fn = 0
    ted_nonorm_fn = 0
    tss_fn_scores = []
    for item in data:
        if item['annotated'] == "negative":
            if item['ts'] >= thresold_tss or math.isclose(item['ts'], thresold_tss):
                tss_fp += 1
            if math.isclose(item['f1'], 1):
                f1_fp += 1
        elif item['annotated'] == "positive":
            if not (item['ts'] >= thresold_tss  or math.isclose(item['ts'], thresold_tss) ):
                tss_fn += 1
                tss_fn_scores.append(item['ts'])
            if not math.isclose(item['TED'], 0):
                ted_fn += 1
            if not math.isclose(item['TED_0'], 0):
                ted_nonorm_fn += 1
    print("TED_unnormed", "false negative count:", ted_nonorm_fn)
    print("TED", "false negative count:", ted_fn)
    print("F1", "false positive count:", f1_fp)
    print("Test Suite Score", "false positive count:", tss_fp, "false negative count:", tss_fn)

def calculate_metrics_pointbiserialr():
    data = load_json("/home5/whzhou/STAR-QC/data/final_test/merge_all_select_anno.json")
    oracles = []
    tss = []
    ted = []
    f1 = []
    for item in data:
        if item['annotated'] == 'positive':
            oracles.append(1)
        else:
            oracles.append(0)
        tss.append(item['ts'])
        ted.append(item['TED'])
        f1.append(item['f1'])
    oracles = np.array(oracles)
    tss = np.array(tss)
    ted = np.array(ted)
    f1 = np.array(f1)
    r_tss, p_value_tss = pointbiserialr(oracles, tss)
    r_ted, p_value_ted = pointbiserialr(oracles, ted)
    r_f1, p_value_f1 = pointbiserialr(oracles, f1)
    print("TED",r_ted, p_value_ted)
    print("F1", r_f1, p_value_f1)
    print("Test Suite Score", r_tss, p_value_tss)

def calculate_metrics_kendall():
    data = load_json("/home5/whzhou/STAR-QC/data/final_test/merge_all_select_anno.json")
    oracles = []
    tss = []
    ted = []
    f1 = []
    for item in data:
        if item['annotated'] == 'positive':
            oracles.append(1)
        else:
            oracles.append(0)
        tss.append(item['ts'])
        ted.append(item['TED'])
        f1.append(item['f1'])
    oracles = np.array(oracles)
    tss = np.array(tss)
    ted = np.array(ted)
    f1 = np.array(f1)
    tau_tss, p_value_tss = kendalltau(oracles, tss)
    tau_ted, p_value_ted = kendalltau(oracles, ted)
    tau_f1, p_value_f1 = kendalltau(oracles, f1)
    print("TED",tau_ted, p_value_ted)
    print("F1", tau_f1, p_value_f1)
    print("Test Suite Score", tau_tss, p_value_tss)

def analyze_webq_qgg_quad():
    qgg_old = {item['qid']:item for item in load_json("/home5/yhbao/SPARQLannotation/data/qgg_score_old.json")['results']}
    qgg_new = {item['qid']:item for item in load_json("/home5/yhbao/SPARQLannotation/data/qgg_score_new.json")['results']}
    quad_old = load_json("/home5/yhbao/SPARQLannotation/data/quad_score_old.json")
    quad_new = load_json("/home5/yhbao/SPARQLannotation/data/quad_score_new.json")
    quad_new.pop("summary")
    quad_new.pop("test_suite_s")
    quad_old.pop("summary")
    qgg_old_scores = {k:v['test_suite']['f1'] for k, v in qgg_old.items()}
    qgg_new_scores = {k:v['test_suite_similarity'] for k, v in qgg_new.items()}
    quad_old_scores = {k:v['test_suite']['f1'] for k, v in quad_old.items()}
    quad_new_scores = {k:v['test_suite_similarity'] for k, v in quad_new.items()}
    reverse_data = []
    no_reverse_data = []
    for qid in qgg_old_scores:
        if qgg_old_scores[qid] <= quad_old_scores[qid] and qgg_new_scores[qid] > quad_new_scores[qid] \
              or qgg_old_scores[qid] < quad_old_scores[qid] and qgg_new_scores[qid] >= quad_new_scores[qid]:
                new_item = {'qid': qid, 'quad_score_old':quad_old_scores[qid], 'quad_score_new':quad_new_scores[qid],
                            'qgg_output': qgg_old[qid]['searched_queries'][0],
                            'qgg_score_old':qgg_old_scores[qid], 'qgg_score_new':qgg_new_scores[qid],
                            'quad_output': quad_new[qid]["gpt_select_top1"] if 'gpt_select_top1' in quad_new[qid] else quad_new[qid]['search_results'][0],
                            'quad_score_old':quad_old_scores[qid], 'quad_score_new':quad_new_scores[qid]}
                reverse_data.append(new_item)
        if qgg_old_scores[qid] >= quad_old_scores[qid] and qgg_new_scores[qid] < quad_new_scores[qid] \
              or qgg_old_scores[qid] > quad_old_scores[qid] and qgg_new_scores[qid] <= quad_new_scores[qid]:
                new_item = {'qid': qid, 'quad_score_old':quad_old_scores[qid], 'quad_score_new':quad_new_scores[qid],
                            'qgg_output': qgg_old[qid]['searched_queries'][0],
                            'qgg_score_old':qgg_old_scores[qid], 'qgg_score_new':qgg_new_scores[qid],
                            'quad_score_old':quad_old_scores[qid], 'quad_score_new':quad_new_scores[qid]}
                if 'gpt_select_top1' in quad_new[qid]:
                    new_item['quad_output'] = quad_new[qid]['gpt_select_top1']
                elif len(quad_new[qid]['search_results']) > 0:
                   new_item['quad_output'] = quad_new[qid]['search_results'][0]
                else:
                    new_item['quad_output'] = None
                no_reverse_data.append(new_item)
    dump_json(reverse_data, "/home5/yhbao/SPARQLannotation/data/temp_analyze.json")
    dump_json(no_reverse_data, "/home5/yhbao/SPARQLannotation/data/temp_analyze_noreverse.json")

if __name__ == "__main__":
    # search_fails = [item for item in load_json("/home5/yhbao/SPARQLannotation/output/GrailQA_1000_golden_not_use_neighbor_entity_bge_rerank.json")]
    # search_fails_qids = [item['qid'] for item in search_fails if len(item['search_results']) == 0 and "error" not in item]
    # train_data = {item['qid']:item for item in \
    #               load_json("/home5/yhbao/data/grailqa/grailqa_v1.0_train_linking.json")}
    # search_fails = [train_data[k] for k in search_fails_qids]
    # onehops = [item for item in search_fails if len(re.findall("JOIN", item['golden_s_expression'])) == 0]
    # print(len(onehops))
    #dump_json(search_fails, "/home5/yhbao/SPARQLannotation/failures_debug_hop.json"
    # data1 = load_json("/home5/yhbao/SPARQLannotation/output/grailqa_final/grailqa_dev_0_1000_golden_not_use_neighbor_entity_dec03_gpt_rerank.json")
    # for item in data1.values():
    #     sexpr = None
    #     if "gpt_select_top1" in item:
    #         sexpr = item['gpt_select_top1']['sexpr']
    #     elif len(item['search_results']) > 0:
    #         sexpr = item['search_results'][0]['sexpr']
    #     if sexpr is not None:
    #         sparql = lisp_to_sparql(sexpr)
    # #         print(sparql)
    # # calculate_metrics_pointbiserialr()
    calculate_metrics_false_positive_negative(thresold_tss = 1)
    calculate_metrics_false_positive_negative(thresold_tss = 0.95)
    calculate_metrics_false_positive_negative(thresold_tss = 0.9)
    # analyze_webq_qgg_quad()