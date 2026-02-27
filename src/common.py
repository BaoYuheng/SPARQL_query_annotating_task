from enum import Enum
import re

class LITERAL_TYPE(Enum):
    TIME = 1
    NUMBER = 2
    STRING = 3 # 普通字面量

class WIKIDATA_CONSTANT_TYPE(Enum):
    ENTITY = 1
    QUANTITY = 2
    TIME = 3
    STRING = 4
    CLASS = 5

# class DATASET(Enum):
#     GRAIL = 1
#     CWQ = 2
#     WEBQ = 3
#     LC2 = 4
#     QALD = 5
#     SIMULATED_FREEBASE = 6 # 我们构造的模拟查询
#     SIMULATED_WIKIDATA = 7,
#     QUERYAGENT = 8 
#20260122 eswc rebuttal 修改
class DATASET(Enum):
    GRAIL = 1
    CWQ = 2
    WEBQ = 3
    LC2 = 4
    QALD = 5
    SIMULATED_FREEBASE = 6 # 我们构造的模拟查询
    SIMULATED_WIKIDATA = 7
    QGG = 8
    QUERYAGENT = 9
    BINDER = 10
    LSQ = 11


class KB_TYPE(Enum):
    FREEBASE = 1
    WIKIDATA = 2

OPERATOR_FUNCTION = {
    '=': ['GE', 'EQ', 'LE'],
    '>': ['GE', 'GT'],
    '<': ['LE', 'LT'],
}

FUNCTION_OPERATOR = {
    "GE": ">=",
    "GT": ">",
    "EQ": "=",
    "LE": "<=",
    "LT": "<",
    'le': '<=', 
    'ge': '>=', 
    'lt': '<', 
    'gt': '>' # 兼容旧版的 S-expression
}

COMPARISON_OPERATORS = ['=', '>', '<']
ARG_FUNCTIONS = ['ARGMAX', 'ARGMIN']

FUNCTION_TO_NAME = {
    "LT": "Less than",
    "LE": "Less than or Equal to",
    "GT": "Greater than",
    "GE": "Greater than or Equal to",
    "EQ": "Equal to",
    "R": "Reverse"
}

class FREEBASE_CONSTANT_TYPE(Enum):
    ENTITY = 1
    QUANTITY = 2
    TIME = 3
    STRING = 4
    CLASS = 5

SPARQL_wrapper_path_official = "http://210.28.134.34:8890/sparql"
SPARQL_wrapper_path_dkilab = "http://114.212.81.217:8896/sparql"
SPARQL_wrapper_path_wikidata_2019 = "http://114.212.81.217:8890/sparql/"
SPARQL_wrapper_path_wikidata_2023 = "http://114.212.81.217:8896/sparql/"
#ODBC_CONFIG_DKILAB = 'DRIVER=/data/virtuoso/virtuoso-opensource/lib/virtodbc.so;Host=114.212.81.217:18896;UID=dba;PWD=dba'
#ODBC_CONFIG_DKILAB = 'DRIVER=/home2/xxhu/virtuoso-opensource/lib/virtodbc.so;Host=114.212.81.217:18896;UID=dba;PWD=dba'
#sophia上的
ODBC_CONFIG_DKILAB = 'DRIVER=/home2/xxhu/virtuoso-opensource/lib/virtodbc.so;Host=210.28.134.34:18896;UID=dba;PWD=dba'
SPARQL_wrapper_path_dkilab = "http://210.28.134.34:8896/sparql"
ODBC_CONFIG_OFFICIAL = 'DRIVER=/home2/xxhu/virtuoso-opensource/lib/virtodbc.so;Host=210.28.134.34:1111;UID=dba;PWD=dba'
ODBC_CONFIG_WIKIDATA_2019 = 'DRIVER=/home2/xxhu/virtuoso-opensource/lib/virtodbc.so;Host=114.212.81.217:1111;UID=dba;PWD=dba'
ODBC_CONFIG_WIKIDATA_2023 = 'DRIVER=/home2/xxhu/virtuoso-opensource/lib/virtodbc.so;Host=114.212.81.217:1115;UID=dba;PWD=dba'

CLASS_RELATED_DOMAINS = [
    "type."
] # 我觉得不能忽略掉 common.
IGNORED_DOMAINS = [
    "kg.", "dataworld."
]
NS_PREFIX = "http://rdf.freebase.com/ns/"

TIME_SUFFIX = [
    '<http://www.w3.org/2001/XMLSchema#date>',
    '<http://www.w3.org/2001/XMLSchema#gYearMonth>', 
    '<http://www.w3.org/2001/XMLSchema#dateTime>', 
    '<http://www.w3.org/2001/XMLSchema#gYear>'
] # GrailQA 中观察得到的

NUMBER_SUFFIX = [
    '<http://www.w3.org/2001/XMLSchema#integer>',
    '<http://www.w3.org/2001/XMLSchema#float>',
    '<http://www.w3.org/2001/XMLSchema#decimal>'
] # GrailQA 中观察得到的

EPS = 0.001

DELIMETER = " "
INTERMEDIATE_ENTITY_PLACEHOLDER = "<ent>"
ENTITY_PLACEHOLDER_VARIANTS = [
    "<ent>", "<ent", "ent>", "ent"
]
DECOMPOSITION_TAGS = [
    "<IQ1>", "<IQ2>", "<IQ3>", "<IQ4>", "<IQ5>",
    "<iq1>", "<iq2>", "<iq3>", "<iq4>", "<iq5>"
] # 这些标记应该就够了

SUTIME_API = {
    "annotate": "http://114.212.190.19:5555/annotate"
}

TIMEX3_TYPES = [
    "DATE", "TIME", "DURATION", "SET"
]

PRESENT_DATE = "2015-08-10" # CWQ 和 WebQSP 中会出现一些和数据集当前时间相关的问题，观察得到 dump 的时间应该是 2015-08-10

EXP_PLACEHOLDER = "[EXP]"
REVERSE_RELATION_INDICATOR = "REV_"
PATTERN_DELIMETER = ","
FACT_DELIMETER = ","



class FreebaseConstantForConstruction:
    def __init__(self, type, value, data_type=None, lang_tag=None):
        self.type = type
        self.value = value
        self.data_type = data_type
        self.lang_tag = lang_tag
    
    def __repr__(self):
        if self.type == "uri":
            uri = self.value.replace('http://rdf.freebase.com/ns/', '') # Freebase 不带前缀
            return uri
        elif self.type == 'literal':
            value = self.value 
            if not (value.startswith('"') and value.endswith('"')):
                value = f'"{value}"'
            if self.lang_tag is not None:
                return f'{value}@{self.lang_tag}'
            else:
                return f'{value}'
        elif self.type == 'typed-literal':
            value = self.value 
            if not (value.startswith('"') and value.endswith('"')):
                value = f'"{value}"'
            return f"{value}^^<{self.data_type}>"
        else:
            return self.value
    
    @classmethod
    def get_constant_type(cls, constant):
        if constant.startswith('m.') or constant.startswith('g.'):
            return FREEBASE_CONSTANT_TYPE.ENTITY
        elif re.fullmatch("[a-zA-Z_]+\.[a-zA-Z_]+", constant):
            return FREEBASE_CONSTANT_TYPE.CLASS
        elif constant.endswith('^^<http://www.w3.org/2001/XMLSchema#decimal>') or constant.endswith('^^<http://www.w3.org/2001/XMLSchema#integer>') or constant.endswith('^^<http://www.w3.org/2001/XMLSchema#float>'):
            return FREEBASE_CONSTANT_TYPE.QUANTITY
        elif constant.endswith('^^<http://www.w3.org/2001/XMLSchema#date>') or constant.endswith('^^<http://www.w3.org/2001/XMLSchema#gYear>') or constant.endswith('^^<http://www.w3.org/2001/XMLSchema#gYearMonth>') or constant.endswith('^^<http://www.w3.org/2001/XMLSchema#dateTime>'):
            return FREEBASE_CONSTANT_TYPE.TIME
        elif constant.endswith('@en') or constant.endswith('"'):
            return FREEBASE_CONSTANT_TYPE.STRING
        else:
            return None

class WikidataConstantForConstruction:
    def __init__(self, type, value, data_type=None, lang_tag=None):
        self.type = type
        self.value = value
        self.data_type = data_type
        self.lang_tag = lang_tag
    

    def is_legal_year(constant):
        if constant.endswith('^^<http://www.w3.org/2001/XMLSchema#decimal>') or constant.endswith('^^<http://www.w3.org/2001/XMLSchema#integer>'):
            year_int = (constant.split('^'))[0][1:-1]
            if year_int.isdigit():
                return 0 <= int(year_int) and int(year_int) <= 2024
            else:
                return False
        else:
            return False

    def __repr__(self):
        if self.type == "uri":
            uri = self.value.replace('http://www.wikidata.org/entity/', 'wd:') 
            return uri
        elif self.type == 'literal':
            value = self.value 
            if not (value.startswith('"') and value.endswith('"')):
                value = f'"{value}"'
            if self.lang_tag is not None:
                return f'{value}@{self.lang_tag}'
            else:
                return f'{value}'
        elif self.type == 'typed-literal':
            value = self.value 
            if not (value.startswith('"') and value.endswith('"')):
                value = f'"{value}"'
            return f"{value}^^<{self.data_type}>"
        else:
            return self.value
    
    @classmethod
    def get_constant_type(cls, constant):
        if re.fullmatch(r"wd:Q\d+", constant):
            return WIKIDATA_CONSTANT_TYPE.ENTITY
        elif constant.endswith('^^<http://www.w3.org/2001/XMLSchema#decimal>') or constant.endswith('^^<http://www.w3.org/2001/XMLSchema#integer>') or constant.endswith('^^<http://www.w3.org/2001/XMLSchema#float>'):
            return WIKIDATA_CONSTANT_TYPE.QUANTITY
        elif constant.endswith('^^<http://www.w3.org/2001/XMLSchema#date>') or constant.endswith('^^<http://www.w3.org/2001/XMLSchema#gYear>') or constant.endswith('^^<http://www.w3.org/2001/XMLSchema#gYearMonth>') or constant.endswith('^^<http://www.w3.org/2001/XMLSchema#dateTime>'):
            return WIKIDATA_CONSTANT_TYPE.TIME
        elif constant.endswith('@en') or constant.endswith('"'):
            return WIKIDATA_CONSTANT_TYPE.STRING
        else:
            return None

def convert_freebase_type(freebase_type):
    if freebase_type in [FREEBASE_CONSTANT_TYPE.QUANTITY, FREEBASE_CONSTANT_TYPE.TIME, FREEBASE_CONSTANT_TYPE.STRING]:
        return "literal"
    elif freebase_type is FREEBASE_CONSTANT_TYPE.CLASS:
        return "class"
    elif freebase_type is FREEBASE_CONSTANT_TYPE.ENTITY:
        return "entity"
    else:
        return None

def convert_wikidata_type(wikidata_type):
    if wikidata_type in [WIKIDATA_CONSTANT_TYPE.QUANTITY, WIKIDATA_CONSTANT_TYPE.TIME, WIKIDATA_CONSTANT_TYPE.STRING]:
        return "literal"
    elif wikidata_type is WIKIDATA_CONSTANT_TYPE.CLASS:
        return "class"
    elif wikidata_type is WIKIDATA_CONSTANT_TYPE.ENTITY:
        return "entity"
    else:
        return None

WIKIDATA_PREFIX_LIST = "\n".join([
    "PREFIX p: <http://www.wikidata.org/prop/>",
    "PREFIX pq: <http://www.wikidata.org/prop/qualifier/>",
    "PREFIX ps: <http://www.wikidata.org/prop/statement/>",
    "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>",
    "PREFIX wd: <http://www.wikidata.org/entity/>",
    "PREFIX wdt: <http://www.wikidata.org/prop/direct/>",
    "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>"
])

def get_syntax_tree_string(item, dataset:DATASET):
    """
    在语法树中，各类 item 的 to_str() 方法的返回值
    """
    if dataset in [DATASET.CWQ, DATASET.WEBQ, DATASET.GRAIL]:
        if FreebaseConstantForConstruction.get_constant_type(item) in [FREEBASE_CONSTANT_TYPE.ENTITY, FREEBASE_CONSTANT_TYPE.CLASS]:
            return f"<{NS_PREFIX}{item}>"
        elif FreebaseConstantForConstruction.get_constant_type(item) in [FREEBASE_CONSTANT_TYPE.QUANTITY, FREEBASE_CONSTANT_TYPE.TIME, FREEBASE_CONSTANT_TYPE.STRING]:
            return item
        else:
            raise NotImplementedError(f"undefined item: {item}")
    elif dataset in [DATASET.LC2, DATASET.QALD]:
        if WikidataConstantForConstruction.get_constant_type(item) in [WIKIDATA_CONSTANT_TYPE.ENTITY, WIKIDATA_CONSTANT_TYPE.CLASS]:
            return f"<http://www.wikidata.org/entity/{item[3:]}>"
        elif WikidataConstantForConstruction.get_constant_type(item) in [WIKIDATA_CONSTANT_TYPE.QUANTITY, WIKIDATA_CONSTANT_TYPE.TIME]:
            return item
        elif WikidataConstantForConstruction.get_constant_type(item) in [WIKIDATA_CONSTANT_TYPE.STRING]:
            # Wikidata 的两个数据集里面，是单引号
            if item.endswith('@en'):
                return f"'{item.split('@en')[0][1:-1]}'@en"
            else:
                return f"'{item[1:-1]}'"
        else:
            raise NotImplementedError(f"undefined item: {item}")
    else:
        raise NotImplementedError(f"dataset: {dataset}")

def get_syntax_tree_value(item, dataset:DATASET):
    """
    各类 item 在语法树中的 value 属性
    """
    if dataset in [DATASET.CWQ, DATASET.WEBQ, DATASET.GRAIL]:
        if FreebaseConstantForConstruction.get_constant_type(item) in [FREEBASE_CONSTANT_TYPE.ENTITY, FREEBASE_CONSTANT_TYPE.CLASS]:
            return f"{NS_PREFIX}{item}" # 需要带上前缀
        elif FreebaseConstantForConstruction.get_constant_type(item) in [FREEBASE_CONSTANT_TYPE.QUANTITY, FREEBASE_CONSTANT_TYPE.TIME]:
            return item.split('^^')[0]
        elif FreebaseConstantForConstruction.get_constant_type(item) is FREEBASE_CONSTANT_TYPE.STRING:
            if item.endswith('@en'):
                return item[:-3]
            else:
                return item
        else:
            raise NotImplementedError(f"undefined item: {item}")
    elif dataset in [DATASET.LC2, DATASET.QALD]:
        if WikidataConstantForConstruction.get_constant_type(item) in [WIKIDATA_CONSTANT_TYPE.ENTITY, WIKIDATA_CONSTANT_TYPE.CLASS]:
            return f"http://www.wikidata.org/entity/{item[3:]}"
        elif WikidataConstantForConstruction.get_constant_type(item) in [WIKIDATA_CONSTANT_TYPE.QUANTITY, WIKIDATA_CONSTANT_TYPE.TIME]:
            return item.split('^^')[0]
        elif WikidataConstantForConstruction.get_constant_type(item) in [WIKIDATA_CONSTANT_TYPE.STRING]:
            # Wikidata 的两个数据集里面，是单引号
            if item.endswith('@en'):
                return f"'{item.split('@en')[0][1:-1]}'@en"
            else:
                return f"'{item[1:-1]}'"
        else:
            raise NotImplementedError(f"undefined item: {item}")
    else:
        raise NotImplementedError(f"dataset: {dataset}")
    