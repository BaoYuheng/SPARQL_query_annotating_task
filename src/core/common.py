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

# ============================================================
# 数据集枚举 / Dataset Enum — 真实 benchmark 数据集
# ============================================================
class Dataset(Enum):
    GRAIL = 1    # GrailQA, Freebase KB
    CWQ = 2      # Complex WebQuestions, Freebase KB
    WEBQ = 3     # WebQSP, Freebase KB
    LC2 = 4      # LC-QuAD 2.0, Wikidata KB
    QALD = 5     # QALD, Wikidata KB
    REQUMA = 6   # ReQUMA, Wikidata KB (our annotation dataset)


# ============================================================
# 方法枚举 / Method Enum — 构造查询的来源系统
# ============================================================
class Method(Enum):
    SIMULATED_FREEBASE = 6   # 我们自己构造的 Freebase 模拟查询
    SIMULATED_WIKIDATA = 7   # 我们自己构造的 Wikidata 模拟查询
    QGG = 8                  # QGG 方法
    QUERYAGENT = 9           # QueryAgent 方法
    BINDER = 10              # KB-BINDER 方法
    LSQ = 11                 # LSQ 方法


# ============================================================
# 旧枚举，向后兼容 / Legacy enum for backward compatibility
# 逐步迁移到 Dataset + Method 后删除 / Remove after full migration
# ============================================================
class DATASET(Enum):
    GRAIL = 1
    CWQ = 2
    WEBQ = 3
    LC2 = 4
    QALD = 5
    SIMULATED_FREEBASE = 6  # 我们构造的模拟查询
    SIMULATED_WIKIDATA = 7
    QGG = 8
    QUERYAGENT = 9
    BINDER = 10
    LSQ = 11
    REQUMA = 12  # ReQUMA, Wikidata KB


# ============================================================
# KB 类型辅助集合与函数 / KB-type helper sets & functions
# ============================================================
FB_DATASETS = frozenset({Dataset.CWQ, Dataset.WEBQ, Dataset.GRAIL})
WD_DATASETS = frozenset({Dataset.LC2, Dataset.QALD, Dataset.REQUMA})
FB_METHODS = frozenset({Method.SIMULATED_FREEBASE, Method.QGG,
                         Method.QUERYAGENT, Method.BINDER, Method.LSQ})
WD_METHODS = frozenset({Method.SIMULATED_WIKIDATA})

# 全局 SPARQL 后端默认值 / Global default SPARQL backend
# "odbc" = Virtuoso ODBC (binary protocol, faster for large results)
# "sparql_wrapper" = HTTP SPARQLWrapper (JSON, easier to debug)
DEFAULT_SPARQL_SERVICE = "odbc"


def dataset_to_kb_type(d: Dataset) -> "KB_TYPE":
    """真实数据集 → KB 类型 / Real dataset → KB type."""
    if d in FB_DATASETS:
        return KB_TYPE.FREEBASE
    elif d in WD_DATASETS:
        return KB_TYPE.WIKIDATA
    raise ValueError(f"Unknown dataset: {d}")


def method_to_kb_type(m: Method) -> "KB_TYPE":
    """构造方法 → KB 类型（按目标知识库） / Method → KB type (by target KB)."""
    if m in FB_METHODS:
        return KB_TYPE.FREEBASE
    elif m in WD_METHODS:
        return KB_TYPE.WIKIDATA
    raise ValueError(f"Unknown method: {m}")


def to_dataset(d: DATASET) -> Dataset:
    """DATASET → Dataset 安全转换 / Safe conversion from legacy DATASET to Dataset."""
    return Dataset[d.name]


def to_method(d: DATASET) -> Method:
    """DATASET → Method 安全转换 / Safe conversion from legacy DATASET to Method."""
    return Method[d.name]


def to_kb_type(val: "Union[Dataset, Method, DATASET]") -> "KB_TYPE":
    """统一转换 Dataset / Method / DATASET → KB_TYPE.
    / Unified KB-type resolution for any of the three enum types."""
    try:
        return method_to_kb_type(to_method(val))  # type: ignore
    except KeyError:
        return dataset_to_kb_type(to_dataset(val))  # type: ignore


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

# ============================================================
# KB 连接配置 / KB Connection Configuration
# ============================================================
# Driver: Virtuoso 7.2.15 ODBC (已验证兼容 7.2.5 服务)
# 注意 / Note: pyodbc 5.x + Python 3 需加 wideAsUTF16=Y
# Freebase 两个端点在本机 / Both Freebase endpoints are on localhost
# ============================================================

# ── SPARQL HTTP 端点 (默认后端 / Default backend) ──
# SPARQL_wrapper_path_official = "http://210.28.134.34:8890/sparql"
SPARQL_wrapper_path_official = "http://localhost:8890/sparql"
# SPARQL_wrapper_path_wikidata_2019 = "http://114.212.81.217:8890/sparql/"
# SPARQL_wrapper_path_wikidata_2023 = "http://114.212.86.175:8895/sparql/"
SPARQL_wrapper_path_wikidata_2023 = "http://114.212.86.175:8895/sparql/"
# SPARQL_wrapper_path_dkilab = "http://210.28.134.34:8896/sparql"
SPARQL_wrapper_path_dkilab = "http://localhost:8896/sparql"

# ── ODBC 连接串 (可选后端 / Optional backend) ──
# 旧配置 / Old configs:
#   ODBC_CONFIG_DKILAB  = 'DRIVER=/home5/yhbao/...virtuoso-7.2.5/lib/virtodbcu_r.so;Host=210.28.134.34:18896;UID=dba;PWD=dba'
#   ODBC_CONFIG_OFFICIAL = 'DRIVER=/home5/yhbao/...virtuoso-7.2.5/lib/virtodbcu_r.so;Host=210.28.134.34:18890;UID=dba;PWD=dba'
#   ODBC_CONFIG_WIKIDATA_2023 = 'DRIVER=/home5/yhbao/...virtuoso-7.2.5/lib/virtodbcu_r.so;Host=114.212.86.175:1115;UID=dba;PWD=dba'
# 旧机器上的 / On old machine:
#   #ODBC_CONFIG_DKILAB = 'DRIVER=/data/virtuoso/virtuoso-opensource/lib/virtodbc.so;Host=114.212.81.217:18896;UID=dba;PWD=dba'
#   #ODBC_CONFIG_DKILAB = 'DRIVER=/home2/xxhu/virtuoso-opensource/lib/virtodbc.so;Host=114.212.81.217:18896;UID=dba;PWD=dba'

VIRTUOSO_DRIVER = '/home5/yhbao/freebase_virtuoso_service/virtuoso_source/virtuoso-7.2.15/lib/virtodbcu_r.so'

# DKILAB (GrailQA Freebase, 服务版本 7.2.5, driver 7.2.15 后向兼容)
ODBC_CONFIG_DKILAB = (
    f'DRIVER={VIRTUOSO_DRIVER};Host=localhost:18896;UID=dba;PWD=dba;wideAsUTF16=Y'
)
# OFFICIAL Freebase (服务版本 7.2.15)
ODBC_CONFIG_OFFICIAL = (
    f'DRIVER={VIRTUOSO_DRIVER};Host=localhost:18890;UID=dba;PWD=dba;wideAsUTF16=Y'
)
# Wikidata 2023 (远程 / remote, 114.212.86.175)
ODBC_CONFIG_WIKIDATA_2023 = (
    f'DRIVER={VIRTUOSO_DRIVER};Host=114.212.86.175:1115;UID=dba;PWD=dba;wideAsUTF16=Y'
)
# 旧 Wikidata 2019 (已废弃 / deprecated)
# ODBC_CONFIG_WIKIDATA_2019 = 'DRIVER=/home5/yhbao/freebase_virtuoso_service/virtuoso_source/virtuoso-7.2.5/lib/virtodbcu_r.so;Host=114.212.81.217:1111;UID=dba;PWD=dba'

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


# ═══════════════════════════════════════════════════════════════
# TSS 参数 / TSS Parameters
# ═══════════════════════════════════════════════════════════════

TSS_QUERY_MAX_ROWS = 200000        # Method 1 all-at-once 扰动查询行数上限 / max rows for perturbation queries in Method 1
TSS_TEST_SUITE_LIMIT = 50          # Method 2 采样上限 / max sample number of perturbationstest cases sampled in Method 2
TSS_TEST_SUITE_GEN_LIMIT = 50      # Method 3 每题 test case 数量 / max test cases per question in Method 3
TSS_RANDOM_SEED = 374              # test suite 生成随机种子 / random seed for reproducibility



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
        suffix = constant.split('^^')[-1] if '^^' in constant else ''
        suffix = suffix.replace('<http://www.w3.org/2001/XMLSchema#', 'xsd:').rstrip('>')
        if suffix in ('xsd:decimal', 'xsd:integer'):
            year_int = (constant.split('^'))[0][1:-1]
            if year_int.isdigit():
                return 0 <= int(year_int) and int(year_int) <= 2024
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
        # Normalize full XSD URI → abbreviated xsd: form, then check uniformly
        suffix = constant.split('^^')[-1] if '^^' in constant else ''
        suffix = suffix.replace('<http://www.w3.org/2001/XMLSchema#', 'xsd:').rstrip('>')
        if suffix in ('xsd:decimal', 'xsd:integer', 'xsd:float'):
            return WIKIDATA_CONSTANT_TYPE.QUANTITY
        elif suffix in ('xsd:date', 'xsd:gYear', 'xsd:gYearMonth', 'xsd:dateTime'):
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

def get_syntax_tree_string(item, dataset: Dataset):
    """
    在语法树中，各类 item 的 to_str() 方法的返回值
    """
    if dataset in FB_DATASETS:
        if FreebaseConstantForConstruction.get_constant_type(item) in [FREEBASE_CONSTANT_TYPE.ENTITY, FREEBASE_CONSTANT_TYPE.CLASS]:
            return f"<{NS_PREFIX}{item}>"
        elif FreebaseConstantForConstruction.get_constant_type(item) in [FREEBASE_CONSTANT_TYPE.QUANTITY, FREEBASE_CONSTANT_TYPE.TIME, FREEBASE_CONSTANT_TYPE.STRING]:
            return item
        else:
            raise NotImplementedError(f"undefined item: {item}")
    elif dataset in WD_DATASETS:
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

def get_syntax_tree_value(item, dataset: Dataset):
    """
    各类 item 在语法树中的 value 属性
    """
    if dataset in FB_DATASETS:
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
    elif dataset in WD_DATASETS:
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
    