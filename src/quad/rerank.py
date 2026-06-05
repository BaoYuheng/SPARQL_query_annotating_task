from src.llm.agent import OpenAiRequestor, QueryIntent
from src.core.common import KB_TYPE, DATASET, FB_DATASETS, to_dataset
from src.core.utils import load_json, dump_json
import re

prompt_freebase = """
Please help me to select the index of the most relevant sparql Queries for the question.
Please learn from following examples.
!!! Strictly follow the required output: your output must look like Output: [index], where [index] is the SPARQL number you chose. !!!

#
Question:
give me some products that are made with tetramethrin in it.
SPARQLs:
1. SELECT DISTINCT ?OUTQ where { ?OUTQ ns:common.topic.subjects Tetramethrin. ?OUTQ ns:type.object.type ns:business.product_with_ingredients.}
2. SELECT DISTINCT ?OUTQ where { ?OUTQ ns:business.product_with_ingredients.ingredients Tetramethrin. ?OUTQ ns:type.object.type ns:business.product_with_ingredients.}
3. SELECT DISTINCT ?OUTQ where { ?OUTQ ns:common.topic.subjects Tetramethrin. ?OUTQ ns:business.product_with_ingredients.ingredients Tetramethrin. ?OUTQ ns:type.object.type ns:business.product_with_ingredients.}
Output: 1 

#
Question:
chiang kai shek college and sacred heart high school (roseville, michigan) are in what category of school?
SPARQLs:
1. SELECT DISTINCT ?OUTQ where { ?OUTQ ns:education.school_category.schools_of_this_kind Chiang_Kai_Shek_College. ?OUTQ ns:type.object.type ns:education.school_category.}
2. SELECT DISTINCT ?OUTQ where { ?OUTQ ns:education.school_category.schools_of_this_kind Chiang_Kai_Shek_College. ?OUTQ ns:education.school_category.schools_of_this_kind Sacred_Heart_High_School_(Roseville_Michigan). ?OUTQ ns:type.object.type ns:education.school_category.}
Output: 2

#
Question:
which musical artist stopped being active as musical artist on 1985-06?
SPARQLs:
1. SELECT DISTINCT ?OUTQ where { ?OUTQ ns:music.artist.active_end \"1985-06-01-08:00\"^^<http://www.w3.org/2001/XMLSchema#date>. ?OUTQ ns:type.object.type ns:music.artist.},
2. SELECT DISTINCT ?OUTQ where { ?OUTQ ns:music.artist.active_end \"1985-06-08:00\"^^<http://www.w3.org/2001/XMLSchema#gYearMonth>. ?OUTQ ns:type.object.type ns:music.artist.},
3. SELECT DISTINCT ?OUTQ where { ?OUTQ ns:music.artist.active_end \"1985-06-01-08:00\"^^<http://www.w3.org/2001/XMLSchema#date>. ?OUTQ ns:music.artist.active_end \"1985-06-08:00\"^^<http://www.w3.org/2001/XMLSchema#gYearMonth>.?OUTQ ns:type.object.type ns:music.artist.}"
Output: 2

#
Question:
which type of cricket match has the fewest innings per team?
SPARQLs:
1. SELECT DISTINCT ?OUTQ where { ?OUTQ ns:type.object.type ns:cricket.cricket_match_type. ?OUTQ ns:cricket.cricket_match_type.maximum_duration ?arg_value .} ORDER BY ?arg_value LIMIT 1",
2. SELECT DISTINCT ?OUTQ where { ?OUTQ ns:type.object.type ns:cricket.cricket_match_type. ?OUTQ ns:cricket.cricket_match_type.overs_per_inning ?arg_value .} ORDER BY DESC (?arg_value) LIMIT 1",
3. SELECT DISTINCT ?OUTQ where { ?OUTQ ns:type.object.type ns:cricket.cricket_match_type. ?OUTQ ns:cricket.cricket_match_type.innings_per_team ?arg_value .} ORDER BY ?arg_value LIMIT 1",
4. SELECT DISTINCT ?OUTQ where { ?OUTQ ns:type.object.type ns:cricket.cricket_match_type. ?OUTQ ns:cricket.cricket_match_type.overs_per_inning ?arg_value .} ORDER BY ?arg_value LIMIT 1",
5. SELECT DISTINCT ?OUTQ where { ?OUTQ ns:type.object.type ns:cricket.cricket_match_type. ?OUTQ ns:cricket.cricket_match_type.maximum_duration ?arg_value .} ORDER BY DESC (?arg_value) LIMIT 1"
Output: 3

#
Question:
what type of medical trial has a maximum age for eligibility less than 60?
SPARQLs:
1. SELECT DISTINCT ?OUTQ where { ?OUTQ ns:medicine.medical_trial_type.medical_trials ?C1. ?OUTQ ns:type.object.type ns:medicine.medical_trial_type. ?C1 ns:medicine.medical_trial.maximum_age_for_eligibility ?value_0 .FILTER (?value_0 >= \"60\"^^<http://www.w3.org/2001/XMLSchema#integer>) .}",
2. SELECT DISTINCT ?OUTQ where { ?OUTQ ns:medicine.medical_trial_type.medical_trials ?C1. ?OUTQ ns:type.object.type ns:medicine.medical_trial_type. ?C1 ns:medicine.medical_trial.maximum_age_for_eligibility ?value_0 .FILTER (?value_0 > \"60\"^^<http://www.w3.org/2001/XMLSchema#integer>) .}",
3. SELECT DISTINCT ?OUTQ where { ?OUTQ ns:medicine.medical_trial_type.medical_trials ?C1. ?OUTQ ns:type.object.type ns:medicine.medical_trial_type. ?C1 ns:medicine.medical_trial.maximum_age_for_eligibility ?value_0 .FILTER (?value_0 < \"60\"^^<http://www.w3.org/2001/XMLSchema#integer>) .}",
4. SELECT DISTINCT ?OUTQ where { ?OUTQ ns:medicine.medical_trial_type.medical_trials ?C1. ?OUTQ ns:type.object.type ns:medicine.medical_trial_type. ?C1 ns:medicine.medical_trial.maximum_age_for_eligibility ?value_0 .FILTER (?value_0 <= \"60\"^^<http://www.w3.org/2001/XMLSchema#integer>) .}",
5. SELECT DISTINCT ?OUTQ where { ?OUTQ ns:medicine.medical_trial_type.medical_trials ?C1. ?C1 ns:medicine.medical_trial.maximum_age_for_eligibility \"60\"^^<http://www.w3.org/2001/XMLSchema#integer>. ?OUTQ ns:type.object.type ns:medicine.medical_trial_type. \n\n}"
Output: 4


"""

prompt_wikidata = """
Please help me to select the index of the most relevant sparql Queries on Wikidata for the question.
The IDs of KB items in the SPARQL has been replaced into labels. 
Please learn from following examples.
!!! Strictly follow the required output: your output must look like Output: [index], where [index] is the SPARQL number you chose. !!!

#
Question:
What is the highest place of Karakoram?
SPARQLs:
1. SELECT DISTINCT ?OUTQ where { Karakoram wdt:highest_point ?OUTQ. }
2. SELECT DISTINCT ?OUTQ where { ?OUTQ wdt:named_after Karakoram. Karakoram wdt:highest_point ?OUTQ. } 
Output: 1 

#
Question:
When did Michael Jackson die?
SPARQLs:
1. SELECT DISTINCT ?OUTQ where { Michael_Jackson wdt:date_of_death ?OUTQ. FILTER (  ?OUTQ != ?cvt_0  ). }
2. SELECT DISTINCT ?OUTQ where { ?cvt_0 pq:point_in_time ?OUTQ. Michael_Jackson p:significant_event ?cvt_0. Michael_Jackson wdt:date_of_death ?OUTQ. }
Output: 1

#
Question:
which president of the united states was a boy scout
SPARQLs:
1. SELECT DISTINCT ?OUTQ where { ?OUTQ wdt:position_held President_of_the_United_States. ?OUTQ p:award_received ?cvt_1. ?cvt_1 pq:conferred_by Boy_Scouts_of_America. },
2. SELECT DISTINCT ?OUTQ where { ?cvt_0 pq:replaced_by ?OUTQ. ?cvt_0 ps:position_held President_of_the_United_States. ?OUTQ p:award_received ?cvt_1. ?cvt_1 pq:conferred_by Boy_Scouts_of_America. },
3. SELECT DISTINCT ?OUTQ where { ?OUTQ wdt:position_held President_of_the_United_States. ?OUTQ wdt:member_of Boy_Scouts_of_America. },
Output: 3

#
Question:
where was the movie 500 days of summer filmed
SPARQLs:
1. SELECT DISTINCT ?OUTQ where { (500)_Days_of_Summer wdt:narrative_location ?cvt_0. (500)_Days_of_Summer wdt::filming_location ?OUTQ. }
2. SELECT DISTINCT ?OUTQ where { (500)_Days_of_Summer wdt:filming_location ?OUTQ. }
3. SELECT DISTINCT ?OUTQ where { ?cvt_0 ps:narrative_location ?OUTQ. (500)_Days_of_Summer wdt:narrative_location ?OUTQ. }
Output: 2

#
Question:
actress who plays penelope garcia on criminal minds
SPARQLs:
1. SELECT DISTINCT ?OUTQ where { Penelope Garcia wdt:performer ?OUTQ. }
2. SELECT DISTINCT ?OUTQ where { Penelope Garcia wdt:performer ?OUTQ. Critical_Minds wdt:cast_member ?OUTQ. }
3. SELECT DISTINCT ?OUTQ where { ?cvt_0 ps:cast_member ?OUTQ. Critical_Minds p:cast_member ?cvt_0. ?cvt_0 pq:charactor_role Penelope Garcia.}
Output: 3
"""


def get_top1_sparql_by_gpt(search_result_file, data_file, output_dir, dataset:DATASET, model):
    # 首先选出包含多于1个搜索结果的问题 / First, filter questions that have more than 1 search result
    origin_data = {item['qid']:item for item in load_json(data_file)}
    search_results = {item['qid']:item for item in load_json(search_result_file)}
    search_results_need_to_rerank = {item['qid']:item for item in load_json(search_result_file) if len(item['search_results']) > 1}
    gpt_input_dict = {}
    if dataset == DATASET.SIMULATED_WIKIDATA or dataset == DATASET.QALD:
        # 需要额外替换 PID，因此需要载入 property label 列表 / Need to replace PIDs, so load the property label dict
        pid2label_dict = load_json("data/WD_ontology/wikidata_property_id2label_dict.json")
    # 准备 GPT 的输入，包括将 ID 替换为 Label / Prepare GPT input, including replacing IDs with labels
    for item in search_results_need_to_rerank.values():
        if dataset == DATASET.GRAIL or dataset == DATASET.WEBQ or dataset == DATASET.CWQ:
            gpt_input_dict[item['qid']] = prompt_freebase +"\n#\nQuestions:\n" + item['question'] + "\nSPARQLs:\n"
        elif dataset == DATASET.SIMULATED_WIKIDATA or dataset == DATASET.QALD: 
            gpt_input_dict[item['qid']] = prompt_wikidata +"\n#\nQuestions:\n" + item['question'] + "\nSPARQLs:\n"
        if dataset == DATASET.GRAIL or dataset == DATASET.WEBQ or dataset == DATASET.CWQ:
            id2label_dict = origin_data[item['qid']]['linked_item_to_label']
            id2label_dict.update(origin_data[item['qid']]['golden_item_to_label'])
            for idx, result in enumerate(item['search_results']):
                sparql = result['sparql']
                sparql = sparql.replace("\n", "")
                for id, label in id2label_dict.items():
                    sparql = sparql.replace("ns:"+id, label.replace(" ","_"))
                gpt_input_dict[item['qid']] += f"{str(idx+1)}. {sparql}\n"
            #print(gpt_input_dict[item['qid']])
        elif dataset == DATASET.SIMULATED_WIKIDATA or dataset == DATASET.QALD:
            id2label_dict = origin_data[item['qid']]['linked_item_to_label']
            id2label_dict.update(origin_data[item['qid']]['golden_item_to_label'])
            qids = list(id2label_dict.keys())
            qids.sort(key = lambda i: len(i), reverse=True)
            for idx, result in enumerate(item['search_results']):
                sparql = result['sparql']
                sparql = sparql.replace("\n", "")
                pids = list(set(re.findall(r"P\d+", sparql)))
                pids.sort(key = lambda i: len(i), reverse= True)
                for qid in qids:
                    sparql = sparql.replace(qid, id2label_dict[qid].replace(" ","_"))
                for pid in pids:
                    sparql = sparql.replace(pid, pid2label_dict[pid].replace(" ","_"))
                gpt_input_dict[item['qid']] += f"{str(idx+1)}. {sparql}\n"
        else:
            assert(0)
    gpt_input_dict_reversed = {v:k for k, v in gpt_input_dict.items()}
    gpt_messages = list(gpt_input_dict.values())
    gpt_asker = OpenAiRequestor(model)
    results = gpt_asker.concurrent_query(gpt_messages, 10, intent=QueryIntent.QUESTION)
    # 整理输出 / Organize output
    for input, output in results:
        qid = gpt_input_dict_reversed[input]
        try:
            select_idx = int(re.findall("\d+", output)[0]) - 1
            search_results[qid]['gpt_select_top1'] = search_results[qid]['search_results'][select_idx]
        except:
            search_results[qid]['gpt_select_top1'] = "error"
    dump_json(search_results, output_dir)
