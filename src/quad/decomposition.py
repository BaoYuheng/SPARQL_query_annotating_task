from src.core.utils import setup_custom_logger, load_json, dump_json
from datetime import datetime
from tqdm import tqdm
import time
import os
import json
import requests
from requests.adapters import HTTPAdapter
from src.llm.agent import OpenAiRequestor, QueryIntent


def get_decomposition_openai_complete_file_single_key(data_file, output_dir):
    """
    同样接受 start 和 end 两个参数，但是 start 和 end 直接作用在文件名上。
    Also accepts start and end parameters, applied directly to filenames.
    """
    current_time = datetime.now()
    args = dict()
    args["data_file"] = data_file
    args["output_dir"] = f"{output_dir}_{current_time.strftime('%Y-%m-%d_%H:%M:%S')}"
    
    args["max_retries"] = 3
    args["checkpoint_length"] = 100
    args["prompt"] = """
    Please help me to decompose complex questions into simpler sub-questions. Please learn from the provided examples and then decompose new complex questions.

    #
    Question: What state was host to the 2013 Twin Rivers Media Festival and borders Virginia?
    Thought: 
    1. Question target is a state.
    2."What state was host to the 2013 Twin Rivers Media Festival" and " What state borders Virginia" are two conjunctive descriptions of the question target.
    Output: 
    [{"description": "What state was host to the 2013 Twin Rivers Media Festival"}, {"description": "What state borders Virginia"}]

    #
    Question: What is the capital city of the country that imported from Greece?
    Thought: 
    1. Question target is a city.
    2. There is a inner question "the country that imported from Greece". We mark the answer to this inner question as <IQ1> .
    3. "What is the capital city of <IQ1>" is a description of the question target.
    Output: 
    [{"description": "What is the capital city of <IQ1>", "<IQ1>": [{"description": "the country that imported from Greece"}]}]

    #
    Question: Which author wrote the movie 'Flesh' and inspired John Steinbeck to start writing?
    Thought: 
    1. Question target is an author.
    2."Which author wrote the movie 'Flesh'" and "Which author inspired John Steinbeck to start writing" are two conjunctive descriptions of the question target.
    Output: 
    [{"description": "Which author wrote the movie 'Flesh'"}, {"description": "Which author inspired John Steinbeck to start writing"}]

    #
    Question: What home of the Florida Marlins is also the birthplace of a notable professional athlete who began their career in 1997?
    Thought:
    1. Question target is a location. 
    2. There is an inner question "a notable professional athlete who began their career in 1997". We mark the answer to this inner question as <IQ1>.
    3. "What home of the Florida Marlins" and "the birthplace of <IQ1>" are two conjunctive descriptions of the question target.
    Output: 
    [{"description": "What home of the Florida Marlins"}, {"description": "the birthplace of <IQ1>", "<IQ1>": [{"description": "a notable professional athlete who began their career in 1997"}]}]

    #
    Question: Where can I find lodging in the city where Citoyenne Henri was born?
    Thought: 
    1. Question target is a location. 
    2. There is an inner question "the city where Citoyenne Henri was born". We mark the answer to this inner question as <IQ1>.
    3. "Where can I find lodging in <IQ1>" describes the question target.
    Output: 
    [{"description": "Where can I find lodging in <IQ1>", "<IQ1>": [{"description": "the city where Citoyenne Henri was born"}]}]

    #
    Question: Who does Cristiano Ronaldo play for in 2010?
    Thought:
    1. The Question target are sports teams. 
    2. This question requires no further decomposition.
    Output:
    [{"description": "Who does Cristiano Ronaldo play for in 2010?"}]

    #
    """
    
    tmp_dir = os.path.join(args["output_dir"], "_tmp", "")
    os.makedirs(tmp_dir, exist_ok=True)
    logger = setup_custom_logger(os.path.join(tmp_dir, "log.txt"))
    openai_asker = OpenAiRequestor(model="gpt-3.5-turbo-0125", logger=logger)

    
    logger.info("arguments")
    for (key, value) in args.items():
        logger.info(f"{key}: {value}")
    results = list()
    data = load_json(args["data_file"])

    for (idx, example) in enumerate(tqdm(data)):
        '''只会用到数据集中的 question 一项 / Only the question field from the dataset is used'''
        question = example["question"].replace("\"", "'") 
        input = [args["prompt"] + question]

        reply = openai_asker.concurrent_query(input, 1, QueryIntent.QUESTION)

        results.append({
            "qid": example["qid"],
            "question": example["question"],
            "decomposition": reply
        })
        logger.info(f"reply: {reply}")

        if (idx+1) % args["checkpoint_length"] == 0:
            dump_json(
                results[(idx + 1 - args["checkpoint_length"]): (idx+1)], 
                os.path.join(tmp_dir, f"{int(idx / args['checkpoint_length'])}.json")
            )
    dump_json(results, os.path.join(
        args["output_dir"], "results.json"
    ))

if __name__=='__main__':
    # CWQ_test_0_1000 / CWQ 测试集 0-1000
    # get_decomposition_openai_complete_file_single_key(
    #     "data/input/cwq/cwq_test_0_1000_linking.json",
    #     "data/output/decomposition/cwq/cwq_test_0_1000"
    # )

    # WebQSP_test_0_1000 / WebQSP 测试集 0-1000
    # get_decomposition_openai_complete_file_single_key(
    #     "data/input/WebQSP/webqsp_test_0_1000_linking.json",
    #     "data/output/decomposition/webqsp/webqsp_test_0_1000"
    # )

    # LC-QuAD 2.0 test 0-1000 / lc2 测试集 0-1000
    # get_decomposition_openai_complete_file_single_key(
    #     "data/input/LC_QuAD_2/intermediate/lc2_test_0_1000.json",
    #     "data/output/decomposition/lc2/lc2_test_0_1000"
    # )

    # QALD test / QALD 测试集
    get_decomposition_openai_complete_file_single_key(
        "data/input/QALD/intermediate/qald_wikidata_test_processed.json",
        "data/output/decomposition/qald/qald_test"
    )

