from src.utils import setup_custom_logger, load_json, dump_json
from datetime import datetime
from tqdm import tqdm
import time
import os
import json
import requests
from requests.adapters import HTTPAdapter
from components.llm_agent import OpenaiAgentSingleKey

prompt = """
    You are a expert of question decomposition for Knowledge Base Question Answering task. 
    Understand the task definition and learn from the examples below, and decompose the following question.
    The Decompostion splits complex questions into possible DESCRIPTIONS using [DES],
    essentially splits the conjunctive facts inside a question.
    Moreover, the Decomposition tags SUBQUESTIONS inside descriptions using <INQ> </INQ>, 
    essentially tags the one-hop subgraphs inside the multi-hop KB graph.
    It is worth noting that some "simple" questions amy imply multi-hop subgraphs, 
    in which case you can rewrite the problem appropriately before perform decomposition.
    
    #
    Question: Who does Cristiano Ronaldo play for in 2010?
    Thought:
    1. The Question is about a sports team. There is neither conjunctive descriptions nor inner-questions and the decompostion ends.
    Output:
    Who does Cristiano Ronaldo play for in 2010?

    #
    Question: What state was host to the 2013 Twin Rivers Media Festival and borders Virginia?
    Thought: 
    1.  The question is about a state. 
        There are conjunctive descriptions "What state was host to the 2013 Twin Rivers Media Festival" and "borders Virginia" but no subquestions. 
        So it should be divided into "What state was host to the 2013 Twin Rivers Media Festival and [DES] borders Virginia".
        So the decompostion ends.
    Output:
    What state was host to the 2013 Twin Rivers Media Festival [DES] and borders Virginia ?

    #
    Question: What is the capital city of the country that imported from Greece?
    Thought: 
    1.  The question is about a city. 
        There is no counjunctive descriptions in question but a subquestion "the country that imported from Greece".
        So it should be tagged as "What is the capital city of <INQ> The country that imported from Greece </INQ>".
    2.  The mentioned sub-question "<INQ> The country that imported from Greece </INQ>" is about a country. There is no counjunctive descriptions.
        So the decomposition ends.
    Output: 
    What is the capital city of <INQ> the country that imported from Greece </INQ>?

    #
    Question: What home of the Florida Marlins is also the birthplace of a notable professional athlete who began their career in 1997?
    Thought:
    1.  The question is about a place. 
        There are descriptions "What home of the Florida Marlins" and "is also the birthplace of a notable professional athlete who began their career in 1997". 
        And there is a subquestion "a notable professional athlete who began their career in 1997".
        so it should be tagged as "What home of the Florida Marlins [DES] is also the birthplace of <INQ> a notable professional athlete who began their career in 1997 </INQ>?"
    2.  The mentioned sub-question "<INQ> a notable professional athlete who began their career in 1997 </INQ>" is about a professional athelete. There is neither conjunctive descriptions nor inner-questions.
    Output: 
    What home of the Florida Marlins [DES] is also the birthplace of <INQ> a notable professional athlete who began their career in 1997 </INQ>?

    #
    Question: Case of the Namesake Murders is the same genre as what comic book series?
    Thought: 
    1.  The "A is the same genre as B" implies "the genre of A is also the genre of B". so the questions needs to be rewritten as "The genre of Case of the Namesake Murders is also the genre as what comic book series?".
    2.  The question is about comic book series. 
        There is conjunctive descriptions "The genre of Case of the Namesake Murders" and "is also the genre as what comic book series?" but no subquestions.
    Output: 
        The genre of Case of the Namesake Murders [DES] is also the genre as what comic book series?


    Question: What other leagues are in the same football system as the conference premier?
    """
    
    # The Decompostion splits complex questions into possible conjunctive descriptions using [DES], and tags inner questions inside questions using <INQ>.
    # It is worth noting that some "simple" questions may contain inner questions, in which case you need to rewrite the problem appropriately first.
    
    """

    Please help me to decompose complex questions into simpler sub-questions. 
    Please learn from the provided examples and then decompose new complex questions. 
    You need to output the Thoughts and Output strictly following the format.

    #
    Question: What state was host to the 2013 Twin Rivers Media Festival and borders Virginia?
    Thought: 
    1. Question target is a state.
    2."What state was host to the 2013 Twin Rivers Media Festival" and " What state borders Virginia" are two conjunctive descriptions of the question target.
    Output: 
    What state was host to the 2013 Twin Rivers Media Festival [DES] and borders Virginia?

    #
    Question: What is the capital city of the country that imported from Greece?
    Thought: 
    1. Question target is a city.
    2. There is a inner question "the country that imported from Greece". We mark this inner question as <INQ1> .
    3. "What is the capital city of <INQ1>" is a description of the question target.
    Output: 
    What is the capital city of <INQ1> the country that imported from Greece </INQ1>?

    #
    Question: Which author wrote the movie 'Flesh' and inspired John Steinbeck to start writing?
    Thought: 
    1. Question target is an author.
    2."Which author wrote the movie 'Flesh'" and "Which author inspired John Steinbeck to start writing" are two conjunctive descriptions of the question target.
    Output: 
    Which author wrote the movie 'Flesh' [DES] and inspired John Steinbeck to start writing?

    #
    Question: What home of the Florida Marlins is also the birthplace of a notable professional athlete who began their career in 1997?
    Thought:
    1. Question target is a location. 
    2. There is an inner question "a notable professional athlete who began their career in 1997". We mark this inner question as <INQ1> .
    3. "What home of the Florida Marlins" and "the birthplace of <INQ1>" are two conjunctive descriptions of the question target.
    Output: 
    What home of the Florida Marlins [DES] is also the birthplace of <INQ1> a notable professional athlete who began their career in 1997 </INQ1>?

    #
    Question: Where can I find lodging in the city where Citoyenne Henri was born?
    Thought: 
    1. Question target is a location. 
    2. There is an inner question "the city where Citoyenne Henri was born".  We mark this inner question as <INQ1> .
    3. "Where can I find lodging in <INQ1>" describes the question target.
    Output: 
    Where can I find lodging in <INQ1> the city where Citoyenne Henri was born </INQ1>?

    #
    Question: Who does Cristiano Ronaldo play for in 2010?
    Thought:
    1. The Question target are sports teams. 
    2. This question requires no further decomposition.
    Output:
    Who does Cristiano Ronaldo play for in 2010?

    #
    Question: Case of the Namesake Murders is the same genre as what comic book series?
    Thought: 
    1.  The property "A is the same genre as B" implies a inner question "<INQ1> the genre of A <\INQ1> is the genre of B". 
        So the question is rewritten to "The genre of Case of the Namesake Murders is the genre as what comic book series?".
    2. Question target is a some comic book series . 
    3. There is an inner question ""The genre of Case of the Namesake Murders".  We mark this inner question as <INQ1> .
    4. "<INQ1> is the genre as what comic book series?" describes the question target.
    Output: 
        <INQ1> The genre of Case of the Namesake Murders <\INQ1> is the genre as what comic book series?
    
    
    
    """
    tmp_dir = os.path.join(args["output_dir"], "_tmp", "")
    os.makedirs(tmp_dir, exist_ok=True)
    logger = setup_custom_logger(os.path.join(tmp_dir, "log.txt"))
    openai_agent = OpenaiAgentSingleKey.instance(
        logger, max_retries=args["max_retries"]
    )
    logger.info("arguments")
    for (key, value) in args.items():
        logger.info(f"{key}: {value}")
    results = list()
    data = load_json(args["data_file"])

    for (idx, example) in enumerate(tqdm(data)):
        '''只会用到数据集中的 question 一项'''
        question = example["question"].replace("\"", "'") 
        input = args["prompt"] + question
        reply = openai_agent.query(input)
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


def combine_results():
    result_1 = load_json("data/output/grailqa_decomposition_ernie_2023-12-20/results1.json")
    result_2 = load_json("data/output/grailqa_decomposition_ernie_2023-12-21_08:56:33/results.json")
    qid_to_result_1 = {example["qid"]: example for example in result_1}
    qid_to_result_2 = {example["qid"]: example for example in result_2}
    combined_results = dict()
    for qid in qid_to_result_1:
        if qid_to_result_1[qid]["decomposition"]:
            combined_results[qid] = qid_to_result_1[qid]
        elif (qid in qid_to_result_2) and qid_to_result_2[qid]["decomposition"]:
            combined_results[qid] = qid_to_result_2[qid]
    for qid in qid_to_result_2:
        if qid_to_result_2[qid]["decomposition"]:
            combined_results[qid] = qid_to_result_2[qid]
        elif (qid in qid_to_result_1) and qid_to_result_1[qid]["decomposition"]:
            combined_results[qid] = qid_to_result_1[qid]
    print(f"{len(combined_results)}")
    dump_json(list(combined_results.values()), "data/output/grailqa_decomposition_ernie_2023-12-20/results.json")


if __name__=='__main__':
    '''CWQ_test_0_1000'''
    # get_decomposition_openai_complete_file_single_key(
    #     "data/input/cwq/cwq_test_0_1000_linking.json",
    #     "data/output/decomposition/cwq/cwq_test_0_1000"
    # )
    
    '''WebQSP_test_0_1000'''
    # get_decomposition_openai_complete_file_single_key(
    #     "data/input/WebQSP/webqsp_test_0_1000_linking.json",
    #     "data/output/decomposition/webqsp/webqsp_test_0_1000"
    # )

    '''lc2_test_0_1000'''
    # get_decomposition_openai_complete_file_single_key(
    #     "data/input/LC_QuAD_2/intermediate/lc2_test_0_1000.json",
    #     "data/output/decomposition/lc2/lc2_test_0_1000"
    # )

    '''qald_test'''
    get_decomposition_openai_complete_file_single_key(
        "data/input/QALD/intermediate/qald_wikidata_test_processed.json",
        "data/output/decomposition/qald/qald_test"
    )


    # 测试一下
    # logger = setup_custom_logger("data/test/log.txt")
    # openai_agent = OpenaiAgentSingleKey.instance(
    #     logger, max_retries=5
    # )
    # openai_agent.query("Please introduce yourself with less than 10 words")

