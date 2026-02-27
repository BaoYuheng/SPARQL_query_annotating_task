from transformers import AutoTokenizer, AutoModel
import torch
import sentence_transformers
import torch.nn.functional as F
from .common import (
    FUNCTION_TO_NAME, KB_TYPE
)
from .utils import (load_json)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

'''单例模式'''
class SentenceSimilarityPredictor(object):
    def __init__(
        self,
        hg_model_hub_name='sentence-transformers/all-mpnet-base-v2',
        batch_size=4,
        max_length=128,
        kb_type=KB_TYPE.FREEBASE
    ): 
        """
        把模型的初始化放在外面，加快速度，避免重复加载
        注意：很多地方是写死的，如果要尝试不同的预训练模型，需要修改
        该预训练模型相关信息: https://huggingface.co/sentence-transformers/all-mpnet-base-v2
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            hg_model_hub_name,
            cache_dir='/home/home2/xwu/Exploration/hfcache/sentence-transformers_all-mpnet-base-v2',
            local_files_only=True
        )

        self.model = AutoModel.from_pretrained(
            hg_model_hub_name,
            cache_dir='/home/home2/xwu/Exploration/hfcache/sentence-transformers_all-mpnet-base-v2',
            local_files_only=True
        )

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.batch_size = batch_size
        self.max_length = max_length

        self.kb_type = kb_type
        if self.kb_type is KB_TYPE.WIKIDATA:
            self.prop_to_label = load_json("data/input/common/property_to_label.json")
    
    @classmethod
    def instance(cls, *args, **kwargs):
        if not hasattr(SentenceSimilarityPredictor, "_instance"):
            SentenceSimilarityPredictor._instance = SentenceSimilarityPredictor(*args, **kwargs)
        return SentenceSimilarityPredictor._instance
    
    def batch_encode(
        self,
        sentences,
        batch_size=8,
        max_length=128
    ):
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
            encoded_input = self.tokenizer.batch_encode_plus(
                batch_sentences,
                padding='longest',
                truncation='longest_first',
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device) # 可以这么写，参考 https://github.com/huggingface/transformers/pull/9584/files
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
    
    def sort_atomic_list_return_index(self, question, sexp_list, item_to_label:dict):
        """
        @param: question: str
        @param: sexp_list, list of {"path", "s_expression"}
        @param: item_to_label: {'m.123': "label"} # 不含前缀信息，直接是 Sexp 中会出现的形式
        @return: sexp_sorted_list: sexp_list 按照相似度降序排列的结果
        文本上的处理，也包含在这个函数里面了
        """
        if not sexp_list: # None / []
            return []
        question_embedding = self.batch_encode(
            [question],
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        # 保持和 sexp_list 的顺序一致
        sexp_str_list = list()
        for expr in sexp_list:
            expr = expr.replace('(', ' ( ')
            expr = expr.replace(')', ' ) ')
            toks = expr.split(' ')
            toks = [x for x in toks if len(x)]
            new_toks = list()
            for t in toks:
                if t in item_to_label:
                    new_toks.append(item_to_label[t])
                elif t in FUNCTION_TO_NAME:
                    new_toks.append(FUNCTION_TO_NAME[t])
                elif (self.kb_type is KB_TYPE.WIKIDATA) and (replace_wikidata_property_prefix(t) in self.prop_to_label):
                    new_toks.append(self.prop_to_label[replace_wikidata_property_prefix(t)])
                else:
                    new_toks.append(t)
            sexp_str_list.append(" ".join(new_toks))
        
        sexp_embedding = self.batch_encode(
            sexp_str_list,
            batch_size=self.batch_size,
            max_length=self.max_length
        )
        similarity = sentence_transformers.util.cos_sim(question_embedding, sexp_embedding)
        similarity = torch.squeeze(similarity, 0)
        sorted_index = torch.argsort(similarity, descending=True).tolist()

        return sorted_index # TODO: 是否符合预期，返回相似度从高到低的下标

def replace_wikidata_property_prefix(prop):
    return prop.strip().replace('wdt:', '').replace('wd:', '').replace('pq:', '').replace('ps:', '').replace('p:', '')