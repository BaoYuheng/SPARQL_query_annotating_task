import json
import logging
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

import time
import string
import calendar
import copy
from src.core.common import (
    TIME_SUFFIX, NUMBER_SUFFIX, EPS, DELIMETER, PRESENT_DATE, PATTERN_DELIMETER,
    DECOMPOSITION_TAGS
)
from datetime import datetime
from typing import List

def load_json(fname, mode="r", encoding="utf8"):
    if "b" in mode:
        encoding = None
    with open(fname, mode=mode, encoding=encoding) as f:
        return json.load(f)


def dump_json(obj, fname, indent=4, mode='w' ,encoding="utf8", ensure_ascii=False):
    """
    @param: ensure_ascii: `False`, 字符原样输出；`True`: 对于非 ASCII 字符进行转义
    """
    if "b" in mode:
        encoding = None
    with open(fname, "w", encoding=encoding) as f:
        return json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)

def setup_custom_logger(log_file_name):
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    fileHandler = logging.FileHandler(log_file_name, mode='a')
    fileHandler.setFormatter(formatter)

    # 根据日志文件名，创建 Logger 实例；可以从不同的地方写入相同的 Log 文件
    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(fileHandler)
    logger.addHandler(logging.StreamHandler()) # Write to stdout as well
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    logger.info(f"Start logging: {time_}")

    return logger

def convert_number(t):
    # 可能形如 "\"8542\""
    if t.startswith('"') and t.startswith('"'):
        t = t[1:-1]
    try:
        t = float(t)
        return t
    except ValueError:
        pass
    try:
        import unicodedata  # handle ascii
        t = unicodedata.numeric(t)  # string of number --> float
        return t
    except (TypeError, ValueError):
        pass
    return None

def compare_literal(literal1, literal2, operator):
    """
    理想状态，是可以做 Literal 类型的判断
    但是 ODBC 的返回结果，往往不带类型
    按照时间、数值的顺序，依次比较吧

    承认对于边界值的处理不完善；但仅仅是用于构造原子查询；原子查询在后面的过程中还会被验证
    """
    try: # 先做时间的比较
        kb_item_1 = KBTimeItem(literal1)
        kb_item_2 = KBTimeItem(literal2)
        if operator == '=':
            return kb_item_1 == kb_item_2
        elif operator == '<':
            return kb_item_1 < kb_item_2
        elif operator == '>':
            return kb_item_1 > kb_item_2
        elif operator == '<=':
            return kb_item_1 <= kb_item_2
        elif operator == '>=':
            return kb_item_1 >= kb_item_2
        else:
            raise NotImplementedError()
    except Exception as e: # Parsing 出问题
        # print(f"err: {e}") # 供调试
        pass
    
    try:
        kb_item_1 = KBNumberItem(literal1)
        kb_item_2 = KBNumberItem(literal2)
        if operator == '=':
            return kb_item_1 == kb_item_2
        elif operator == '<':
            return kb_item_1 < kb_item_2
        elif operator == '>':
            return kb_item_1 > kb_item_2
    except Exception as e: # Parsing 出问题
        # print(f" error: {e}") # 供调试
        return False
    # 不可能执行到这

def get_n_gram(token_list, N, delimeter=DELIMETER):
    return [_join_tokens(token_list[i: i+N], delimeter) for i in range(len(token_list) - N + 1)]

def _join_tokens(token_list, delimeter=DELIMETER):
    res = ""
    for (idx, tok) in enumerate(token_list):
        if idx == 0:
            res = tok
        elif (tok in string.punctuation) and (idx < len(token_list) - 1): # 序列中间的标点符号，不加空格
            res = f"{res}{tok}"
        elif (tok in string.punctuation) and (idx == len(token_list) - 1): # 序列末尾的标点符号，说明是句子里面所需的标点
            res = res # 不加上这个标点符号
        else:
            res = f"{res}{delimeter}{tok}"
    return res

def post_process_timex_obj(timex_obj):
    """
    有一些 timex 返回值是我们所不需要的，舍弃
    有一些返回值的格式和 KB 上的存储形式有点不同，需要转换

    针对 TC / 我们版本的 TC 实现，年 和 月级别的日期，我们会将其展开
    - 2012 -- (2012-01-01, 2012-12-31)
    - 2012-08 -- (2012-08-01, 2012-08-31)
    """
    # TIMEX 格式 到 Freebase 格式的映射，不在这个表内的，我们就不关心了
    TIMEX_to_kb = {
        "%Y": "%Y",
        "%Y-%m": "%Y-%m",
        "%Y-%m-%d": "%Y-%m-%d",
        "%Y-%m-%dT%H": "%Y-%m-%dT%HZ",
        "%Y-%m-%dT%H:%M": "%Y-%m-%dT%H:%MZ", 
        "%Y-%m-%dT%H:%M:%S": "%Y-%m-%dT%H:%M:%SZ"
    }
    SPECIAL_TIMEX_VALUE = {
        "PRESENT_REF": PRESENT_DATE
    }

    if timex_obj in SPECIAL_TIMEX_VALUE:
        return [SPECIAL_TIMEX_VALUE[timex_obj]]
    else:
        for (key, value) in TIMEX_to_kb.items():
            try:
                obj = datetime.strptime(timex_obj, key)
                if key == "%Y":
                    year = obj.year
                    obj_start = datetime(year=year, month=1, day=1)
                    obj_end = datetime(year=year, month=12, day=31)
                    return [
                        datetime.strftime(obj, value),
                        datetime.strftime(obj_start, "%Y-%m-%d"),
                        datetime.strftime(obj_end, "%Y-%m-%d")
                    ]
                elif key == "%Y-%m":
                    year = obj.year
                    month = obj.month
                    last_day_in_month = calendar.monthrange(year, month)[1]
                    obj_start = datetime(year=year, month=month, day=1)
                    obj_end = datetime(year=year, month=month, day=last_day_in_month)
                    return [
                        datetime.strftime(obj, value),
                        datetime.strftime(obj_start, "%Y-%m-%d"),
                        datetime.strftime(obj_end, "%Y-%m-%d")
                    ]
                else:
                    return [datetime.strftime(obj, value)]
            except:
                pass
        return list()

def flatten_decomposition_tree(decomposition):
    sub_question_list = list()

    def dfs(node, sub_question:list):
        sub_question.append(node["description"])
        has_child_flag = False
        for tag in DECOMPOSITION_TAGS:
            if tag in node:
                has_child_flag = True
                for child in node[tag]:
                    dfs(child, copy.deepcopy(sub_question))
        
        if not has_child_flag:
            sub_question_list.append(copy.deepcopy(sub_question))
    
    for head_node in decomposition:
        sub_question = list()
        dfs(head_node, sub_question)

    return sub_question_list

class KBNumberItem:
    def __init__(self, kb_item):
        self.value = float('-inf')
        self.kb_item = kb_item
        self.construct()
    
    def construct(self):
        if self.kb_item.endswith('<http://www.w3.org/2001/XMLSchema#integer>'):
            content = self.kb_item.split('^^')[0][1:-1]
        elif self.kb_item.endswith('<http://www.w3.org/2001/XMLSchema#float>'):
            content = self.kb_item.split('^^')[0][1:-1]
        else:
            content = self.kb_item
        value = float(content)
        self.value = value
    
    def __str__(self) -> str:
        return f"{self.value}"

    def __eq__(self, other):
        return abs(self.value - other.value) < EPS

    def __ne__(self, other):
        return abs(self.value - other.value) >= EPS

    def __lt__(self, other):
        return (abs(self.value - other.value) >= EPS) and (self.value < other.value)

    def __le__(self, other):
        return (self.value < other.value) or (abs(self.value - other.value) < EPS)

    def __gt__(self, other):
        return (abs(self.value - other.value) >= EPS) and (self.value > other.value)
    
    def __ge__(self, other):
        return (self.value > other.value) or (abs(self.value - other.value) < EPS)

class KBTimeItem:
    def __init__(self, kb_item):
        # 允许不含后缀
        self.year = float('-inf')
        self.month = 1
        self.day = 1
        self.hour = 0
        self.minute = 0
        self.second = 0
        self.kb_item = kb_item
        self.construct()
    
    def construct(self):
        '''
        TODO: 会抛出异常，外层处理
        '''
        if self.kb_item.endswith('<http://www.w3.org/2001/XMLSchema#dateTime>'):
            content = self.kb_item.split('^^')[0][1:-7] # "-08:00"
            format = "%Y-%m-%dT%H:%M:%S"
        elif self.kb_item.endswith('<http://www.w3.org/2001/XMLSchema#gYear>'):
            content = self.kb_item.split('^^')[0][1:-7]
            format = "%Y"
        elif self.kb_item.endswith('<http://www.w3.org/2001/XMLSchema#gYearMonth>'):
            content = self.kb_item.split('^^')[0][1:-7]
            format = "%Y-%m"
        elif self.kb_item.endswith('<http://www.w3.org/2001/XMLSchema#date>'):
            content = self.kb_item.split('^^')[0][1:-7]
            format = "%Y-%m-%d"
        try:
            date_time_obj = datetime.strptime(content, format)
        except: # Parsing 失败 / format undefined，就把前面的数字部分单独提取出来
            # TODO: 针对 CWQ 的特殊处理，有局限性
            if self.kb_item.endswith('^^<http://www.w3.org/2001/XMLSchema#dateTime>'):
                content = self.kb_item.replace("^^<http://www.w3.org/2001/XMLSchema#dateTime>", "")[1:-1]
            else:
                content = self.kb_item
            if content.count('-') == 2 and content.count(':') == 2: 
                if content.endswith('Z'):
                    if 'T' in content:
                        format = "%Y-%m-%dT%H:%M:%SZ"
                    else:
                        format = "%Y-%m-%d %H:%M:%SZ"
                else:
                    if 'T' in content:
                        format = "%Y-%m-%dT%H:%M:%S"
                    else:
                        format = "%Y-%m-%d %H:%M:%S"
            elif content.count('-') == 2:
                format = "%Y-%m-%d"
            elif content.count('-') == 1:
                format = "%Y-%m"
            else:
                format = "%Y"
            date_time_obj = datetime.strptime(content, format)

        self.year = date_time_obj.year
        self.month = date_time_obj.month
        self.day = date_time_obj.day
        self.hour = date_time_obj.hour
        self.minute = date_time_obj.minute
        self.second = date_time_obj.second
    
    def __str__(self) -> str:
        return f"{self.year}-{self.month}-{self.day} {self.hour}:{self.minute}:{self.second}"

    def __eq__(self, other):
        return (self.year, self.month, self.day, self.hour, self.minute, self.second) == (other.year, other.month, other.day, other.hour, other.minute, other.second)

    def __ne__(self, other):
        return (self.year, self.month, self.day, self.hour, self.minute, self.second) != (other.year, other.month, other.day, other.hour, other.minute, other.second)

    def __lt__(self, other):
        return (self.year, self.month, self.day, self.hour, self.minute, self.second) < (other.year, other.month, other.day, other.hour, other.minute, other.second)

    def __le__(self, other):
        return (self.year, self.month, self.day, self.hour, self.minute, self.second) <= (other.year, other.month, other.day, other.hour, other.minute, other.second)

    def __gt__(self, other):
        return (self.year, self.month, self.day, self.hour, self.minute, self.second) > (other.year, other.month, other.day, other.hour, other.minute, other.second)
    
    def __ge__(self, other):
        return (self.year, self.month, self.day, self.hour, self.minute, self.second) >= (other.year, other.month, other.day, other.hour, other.minute, other.second)

def get_PRF1(pred_answer, golden_answer):
    pred_answer = set(pred_answer)
    golden_answer = set(golden_answer)
    if len(pred_answer)== 0:
        if len(golden_answer)==0:
            p=1
            r=1
            f=1
        else:
            p=0
            r=0
            f=0
    elif len(golden_answer)==0:
        p=0
        r=0
        f=0
    else:
        p = len(pred_answer & golden_answer)/ len(pred_answer)
        r = len(pred_answer & golden_answer)/ len(golden_answer)
        f = 2*(p*r)/(p+r) if p+r>0 else 0
    return p, r, f

def flatten_list(nested_list:List[List]):
    flattened_list = list()
    for lst in nested_list: # 维护了每个 subList 内部的顺序；subList 之间，代表不同子问题的 atomic query, 本身就没有顺序
        flattened_list.extend(lst)
    return flattened_list

def PRF1_for_count(pred_answer, golden_answer_length):
    '''
    只考虑答案集合的大小关系，不考虑具体内容
    @param pred_answer: list or set
    @param golden_answer_length: integer
    '''
    pred_answer_length = len(set(pred_answer))

    if pred_answer_length == 0:
        if golden_answer_length == 0: # gold answer 为空的话，噪声也忒大了
            p=1
            r=1
            f=1
        else:
            p=0
            r=0
            f=0
    elif golden_answer_length == 0:
        p=0
        r=0 # 避免判定为 recall = 1, 被留下来
        f=0
    else:
        p = min(golden_answer_length, pred_answer_length) / pred_answer_length
        r = min(pred_answer_length, golden_answer_length) / golden_answer_length
        f = 2*(p*r)/(p+r) if p+r>0 else 0
    return p, r, f
