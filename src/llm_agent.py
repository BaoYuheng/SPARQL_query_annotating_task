import random
from openai import OpenAI
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from tqdm import tqdm
import math

class QueryIntent(Enum):
    QUESTION = 1,
    EMBEDDING = 2

v3_api_keys = ["sk-k9rLQCwGc0y1CPVRCb27F261206c47C3A9E6Bc9a076f34B9", 
            "sk-f1iYyHwqzhKaCdgp3532B18b623c4009B3C862CbF1096012", 
            "sk-SSKcCg4VHzmUbQVU494d50565a0c48349933BdD98425D288",
            "sk-jgJeUkNqks5aouo58fFdA5B62d4e41A6A1C9DfD175DbB20f",
            "sk-h9RHx26FRXcPuqus84E0220e8d7746728885531dB4751cA4",
            "sk-B9AmdunaQfRZHMRn7eBf86753870425dA3FaCb2b72AeD20f",
            "sk-kd48MeZ8bSAOuchiAaB34aFcE3C248A0B9Ab3d1b53F2C2C8",
            "sk-mRtDuepsyGIrRsxcB1B00cDc072c4d42A54cF2D418B77aAe",
            "sk-lh8YYRHVnOFA0oH320C7C26bAfE04d13899eDc6041C32338",
            "sk-UfL3N6xtniByttjmA69c34101c4843BbA1D8A33aBdC84cBe"]

class OpenAiRequestor():
    def __init__(self, model, api_keys=v3_api_keys, url=None, logger=None, timeout=60, max_retries = 3) -> None:
        self.api_keys = api_keys
        self.model = model
        self.url = url
        self.logger = logger
        self.timeout = timeout
        self.max_retries = max_retries

    def _single_query(self, client, message, intent:QueryIntent, api_key = None, id = None):
        """
        单线程与OpenAI交互，意图可以是问答(QUESTION)或要求EMBEDDING(EMBEDDING)。
        如果需要记录k-v（用于并发，则填写id字段，会返回：(id, response)
        否则返回 response
        """
        for _ in range(self.max_retries): # 针对 Rate limit 或者 quota limit
            #time.sleep(1)
            try:
                if intent == QueryIntent.QUESTION:
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": message,
                            }
                        ],
                        model= self.model, 
                    )
                    reply = chat_completion.choices[0].message.content
                elif intent == QueryIntent.EMBEDDING:
                    reply = client.embeddings.create(input = [message], model=self.model).data[0].embedding
                else:
                    assert(0)
                return(message, reply)
            except Exception as e: # 重试
                if self.logger is not None:
                    self.logger.error(f"error: {e}") 
                time.sleep(1)

    def sequence_query(self, messages, intent:QueryIntent, api_key = None):
        """
        串行查询，用list的index当做id
        """        
        if api_key is None:
            api_key = self.api_keys[0]
        if self.url is not None:
            client = OpenAI(api_key = api_key, base_url = self.url, timeout = self.timeout)
        else:
            client = OpenAI(api_key = api_key, timeout = self.timeout)
        if api_key is None:
            api_key = self.api_keys[0]
        results = []
        for item in tqdm(messages):
            results.append(self._single_query(client, item, intent, api_key))
        return results
    
    def concurrent_query(self, messages, n_threads, intent:QueryIntent):
        results = []
        assert(n_threads <= len(self.api_keys))
        split_len = math.ceil(len(messages)/n_threads)
        pool = ThreadPoolExecutor(max_workers=n_threads)
        futures = []
        for i in range(0, n_threads):
            message_split = messages[i*split_len: min(len(messages), (i+1)*split_len)]
            future = pool.submit(self.sequence_query, message_split, intent, self.api_keys[i])
            futures.append(future)
        for future in futures:
            results+=(future.result())
        pool.shutdown()
        return results
