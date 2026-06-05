# 对于关系链接，使用 HX 的 Freebase embedding
# Relation linking using HX's Freebase embeddings

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sentence_transformers import util
from src.core.utils import load_json, dump_json
from transformers import AutoTokenizer, AutoModel
import torch
import sentence_transformers
import torch.nn.functional as F
from tqdm import tqdm

class FBEmbeddingBasedSimilarity:
    def __init__(self, dataset) -> None:
        """
        使用 TextAda 计算问题和关系 embedding，计算相似度获取。
        数据集是 GrailQA, CWQ, WebQSP。
        Compute question-relation similarity via TextAda embeddings.
        Supported datasets: GrailQA, CWQ, WebQSP.
        """
        # NOTE: embedding 文件体积较大（~1GB），未纳入本仓库，路径指向原始存储位置。
        # Embedding files (~1GB each) are not included in this repo; paths point to the original storage location.
        self.r_embedding = load_json("/home5/yhbao/SPARQLannotation/data/linking_data/fb_relation_embed.json")
        assert dataset in ['GrailQA', "CWQ", "WebQSP"]
        if dataset == "GrailQA":
            self.q_embedding = load_json("data/linking_data/grailqa_question_embed.json")
        elif dataset == "CWQ":
            # NOTE: embedding 文件体积较大（~1.5GB），未纳入本仓库，路径指向原始存储位置。
            # Embedding file (~1.5GB) not included in this repo; path points to the original storage location.
            self.q_embedding = load_json("/home5/yhbao/SPARQLannotation/data/linking_data/cwq_question_embed.json")
        elif dataset == "WebQSP":
            self.q_embedding = load_json("data/linking_data/gwebqsp_question_embed.json")
        else:
            assert(0)
        
    def retrieve_relation_topk(self, query, relation_list, top_k=30,):
        if relation_list == []:
            return []
        question_embedding = self.q_embedding[query]
        relation_embeddings = []
        for rel in relation_list:
            relation_embeddings.append(self.r_embedding[rel])
        similarities = util.pytorch_cos_sim(question_embedding, relation_embeddings)
        sorted_relations = [(relation, score) for relation, score in zip(relation_list, similarities.tolist()[0])]
        sorted_relations = sorted(sorted_relations, key=lambda x: x[1], reverse=True)
        sorted_relation_list = [relation[0] for relation in sorted_relations][:top_k]
        return sorted_relation_list
    
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class PLMSimRanker:
    """
    使用 SentenceBERT 等语义相似度模型，快速计算相似度。
    特点是不用预先跑 embedding，即算即用。
    On-the-fly semantic similarity via SentenceBERT-style models without pre-computed embeddings.
    """
    def __init__(self, model, cache_dir = "/home5/yhbao/SPARQLannotation/cache", batch_size = 8, max_length = 128) -> None:
        # NOTE: cache_dir 指向模型缓存的外部存储位置，不在本仓库内。
        # cache_dir points to an external model cache directory outside this repo.
        self.tokenizer = AutoTokenizer.from_pretrained(model, cache_dir="/".join([cache_dir, model]))
        self.model = AutoModel.from_pretrained(model, cache_dir="/".join([cache_dir, model]))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.batch_size = batch_size
        self.max_length = max_length

    def batch_encode(self, sentences, batch_size=8, max_length=128):
        """
        max_length=128: 该预训练语言模型训练时所使用的的配置
        但是 128 对于我们的数据肯定是不够的，建议对于三种筛选工具，统一将 max length 设置成 256
        不设置 max length? 会有一些太长的数据，超出模型限制（或者爆显存）
        筛选阶段，因为 max length 有一点损失，应该是允许的

        自带的 encode() 方法，好像不支持 max_length
        """
        results = None
        for start_idx in range(0, len(sentences), batch_size):
            batch_sentences = sentences[start_idx: start_idx+batch_size]
            encoded_input = self.tokenizer(
                batch_sentences,
                padding='longest',
                truncation='longest_first',
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            # print(sentence_embeddings.shape) # [batch_size, 768]
            if results is None:
                results = sentence_embeddings
            else:
                results = torch.cat((results, sentence_embeddings), dim=0)
        return results
    

    def get_semantic_sim_topk(self, sentence, targets, top_k = 10):
            if len(targets) <= 1:
                return targets
            gold_embedding = self.batch_encode(
                [sentence],
                batch_size=self.batch_size, 
                max_length=self.max_length
            )
            target_embedding = self.batch_encode(
                targets,
                batch_size=self.batch_size,
                max_length=self.max_length
            )
            similarity = sentence_transformers.util.cos_sim(gold_embedding, target_embedding)
            similarity = torch.squeeze(similarity, 0).cpu().numpy().tolist()
            target_sims = [(similarity[idx], t) for idx, t in enumerate(targets)]
            target_sims.sort(key = lambda i: i[0], reverse = True)
            if len(targets) < top_k:
                return [item[1] for item in target_sims]
            else:
                return [item[1] for item in target_sims[0:top_k]]
    


def calculate_question_rel_topk(dataset, split, output_dir):
    embedding_sim_calculator = FBEmbeddingBasedSimilarity(dataset)
    results = {}
    if dataset == "CWQ":
        # NOTE: embedding 文件体积较大（~1GB），未纳入本仓库，路径指向原始存储位置。
        # Embedding file (~1GB) not included in this repo; path points to the original storage location.
        fb_embeddings = load_json("/home5/yhbao/SPARQLannotation/data/linking_data/fb_relation_embed.json")
        full_relations = list(fb_embeddings.keys())
        if split == "dev":
            questions = [item['question'] for item in load_json("data/input/cwq/cwq_dev_linking.json")]
        elif split == "train":
            questions = [item['question'] for item in load_json("data/input/cwq/cwq_train_linking.json")]
        else:
            questions = [item['question'] for item in load_json("data/input/cwq/cwq_test_linking.json")]
        for q in tqdm(questions):
            top_relations = embedding_sim_calculator.retrieve_relation_topk(q, full_relations, 1000)
            results[q] = top_relations
        dump_json(results, output_dir)
    else:
        raise Exception("Not implymented")


if __name__ == "__main__":
    baai_reranker = PLMSimRanker(model = "BAAI/bge-reranker-v2-m3")