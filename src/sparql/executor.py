from __future__ import annotations
try:
    import pyodbc
    HAS_PYODBC = True
except ImportError:
    HAS_PYODBC = False
import itertools
from SPARQLWrapper import SPARQLWrapper, JSON
from src.logical_form.s_expression_utils import (
    JOIN, CMP, R, AND, sexp_to_sparql
)
from src.core.utils import (
    convert_number,
    compare_literal,
    load_json,
    dump_json
)
from src.concurrent.executor import ConcurrentExecutor
from src.logical_form.simple_graph import SimpleGraph, Node, NodeType
from src.core.common import (
    OPERATOR_FUNCTION, COMPARISON_OPERATORS, ARG_FUNCTIONS,
    SPARQL_wrapper_path_official,
    SPARQL_wrapper_path_dkilab,
    CLASS_RELATED_DOMAINS,
    IGNORED_DOMAINS,
    NS_PREFIX,
    EXP_PLACEHOLDER,
    FREEBASE_CONSTANT_TYPE,
    FreebaseConstantForConstruction,
    WikidataConstantForConstruction,
    WIKIDATA_CONSTANT_TYPE
)
import math
from collections import defaultdict
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.linking.semantic_sim import PLMSimRanker
import time
import os
import datetime as _dt


def _odbc_value_to_binding(val):
    """Convert an ODBC result value to SPARQLWrapper-compatible binding dict.

    Mimics SPARQLWrapper's JSON serialization:
      - type: 'uri' for URIs (values starting with http://)
      - type: 'literal' for everything else (incl. integers, booleans, dates)
      - datatype: XSD type URI for typed values (xsd:integer, xsd:double, etc.)

    Note: Virtuoso represents boolean as 0/1 (xsd:integer), NOT xsd:boolean.
    We match this behavior for ODBC↔SPARQLWrapper consistency.

    将 ODBC 查询结果值转换为 SPARQLWrapper 兼容的 binding dict 格式。
    模拟 SPARQLWrapper JSON 序列化行为：URI → type:uri，其他 → type:literal。
    """
    XSD_INTEGER = 'http://www.w3.org/2001/XMLSchema#integer'
    XSD_DOUBLE = 'http://www.w3.org/2001/XMLSchema#double'
    XSD_DATETIME = 'http://www.w3.org/2001/XMLSchema#dateTime'

    if val is None:
        return {'type': 'literal', 'value': ''}
    if isinstance(val, bool):
        # Virtuoso ODBC: boolean → 0/1 (xsd:integer), matching SPARQLWrapper
        return {'type': 'literal', 'value': '1' if val else '0',
                'datatype': XSD_INTEGER}
    if isinstance(val, int):
        return {'type': 'literal', 'value': str(val),
                'datatype': XSD_INTEGER}
    if isinstance(val, float):
        return {'type': 'literal', 'value': str(val),
                'datatype': XSD_DOUBLE}
    if isinstance(val, (_dt.date, _dt.datetime)):
        return {'type': 'literal', 'value': val.isoformat(),
                'datatype': XSD_DATETIME}
    if isinstance(val, _dt.time):
        return {'type': 'literal', 'value': val.isoformat(),
                'datatype': XSD_DATETIME}
    # str and fallback
    val_str = str(val)
    if val_str.startswith('http://'):
        return {'type': 'uri', 'value': val_str}
    # Virtuoso ODBC returns datetime strings as "YYYY-MM-DD HH:MM:SS" (space),
    # but SPARQLWrapper uses ISO 8601 "YYYY-MM-DDTHH:MM:SS".  Normalize.
    # 检测 Virtuoso ODBC 返回的日期时间字符串（空格分隔），转为 ISO 8601 (T分隔)
    if (isinstance(val, str) and len(val_str) >= 19
            and val_str[4] == '-' and val_str[7] == '-'
            and val_str[10] == ' ' and val_str[13] == ':' and val_str[16] == ':'):
        try:
            _parsed = _dt.datetime.strptime(val_str[:19], '%Y-%m-%d %H:%M:%S')
            _iso = _parsed.isoformat()
            # Preserve trailing timezone info (e.g. 'Z', '+00:00')
            _suffix = val_str[19:]
            return {'type': 'literal', 'value': _iso + _suffix,
                    'datatype': XSD_DATETIME}
        except ValueError:
            pass
    return {'type': 'literal', 'value': val_str}


class SparqlOdbcQuerierNoSexpr(ConcurrentExecutor):
    """
    SPARQL executor for Freebase via ODBC (Virtuoso).
    Freebase KB 后端：通过 ODBC (Virtuoso) 执行 SPARQL 查询。
    Paper: Sec 4.1 KB Backend, Sec 5.3 Query Graph Construction (Exploration).

    Supports: one-hop expansion with answer anchoring, CVT relation handling,
    relation blacklisting, caching, and two-hop CVT optimization.
    """
    def __init__(self, sparql_wrapper_path, logger, service="sparql_wrapper",
                 odbc_config=None, timeout=6, sparql_cache_dir=None,
                 direct_manage_2hop=False):
        super().__init__(logger)
        self.service = service
        if self.service == "odbc" and not HAS_PYODBC:
            raise ImportError(
                "pyodbc 未安装，无法使用 ODBC 后端。请安装: pip install pyodbc\n"
                "pyodbc not installed. Install: pip install pyodbc\n"
                "或使用默认的 SPARQLWrapper 后端: service='sparql_wrapper'"
            )
        if self.service == "odbc" and odbc_config is None:
            raise ValueError("service='odbc' 需要提供 odbc_config / requires odbc_config")
        self.CVT_R1_NUM = 10
        self.CVT_R1_BLACKLIST = ["common.", "base.fbontology.", "common.webpage", "freebase.valuenotation", ]
        self.odbc_config = odbc_config
        self.sparql_wrapper_path = sparql_wrapper_path
        self.timeout = timeout
        self.ODBC_PREFIX = "SPARQL PREFIX ns: <http://rdf.freebase.com/ns/> "
        self.SPARQL_PREFIX = "PREFIX ns: <http://rdf.freebase.com/ns/> "
        self.sparql_wrapper = SPARQLWrapper(self.sparql_wrapper_path)
        self.sparql_wrapper.setReturnFormat(JSON)
        self.sparql_wrapper.setTimeout(self.timeout)
        self.sparql_cache_dir = sparql_cache_dir
        # Optimization trick: when exploring a <-r1- ?cvt -r2->, first explore r1,
        # filter blacklist & rank, then find r2. Much faster but loses some recall.
        # 优化技巧：a <-r1- ?cvt -r2-> 时，先探索第一跳 r1 + blacklist 过滤 + 排序，再找 r2，更快但损失部分 recall
        self.direct_manage_2hop = direct_manage_2hop
        if self.sparql_cache_dir is not None:
            if not os.path.isfile(self.sparql_cache_dir):
                dump_json(dict(), self.sparql_cache_dir)
            self.cached_results = load_json(self.sparql_cache_dir)
            # Always save a backup to prevent cache corruption from file modification. // 每次保存备份，防止修改源文件损坏缓存
            dump_json(self.cached_results, sparql_cache_dir.split(".")[0]+"_backup.json")
            self.num_new_cache = 0
        else:
            self.cached_results = None
    
    def format_sparql(self, sparql):
        return sparql.replace("\n", " ").replace(" ", "")
    
    def write_current_cache_to_file(self):
        dump_json(self.cached_results, self.sparql_cache_dir)

    def update_cache_results(self, sparql, results, save_split = 1000):
        # Update cache; flush new entries to file when exceeding save_split. // 更新缓存，新增超过 save_split 条时写入文件
        self.cached_results[sparql] = results
        self.num_new_cache += 1
        if self.num_new_cache == save_split:
            self.write_current_cache_to_file()
            self.num_new_cache = 0
            self.cached_results = load_json(self.sparql_cache_dir)

    def get_ignored_relations_filter(self, variable):
        filter_list = [
            f"!regex({variable}, \"^{NS_PREFIX}{domain}\")"
            for domain in IGNORED_DOMAINS + CLASS_RELATED_DOMAINS
        ]
        return f"""
        FILTER ({" && ".join(filter_list)})
        """
    
    def connect(self):
        if not HAS_PYODBC:
            raise ImportError(
                "pyodbc 未安装，无法建立 ODBC 连接. "
                "请安装: pip install pyodbc, 或使用 service='sparql_wrapper'\n"
                "pyodbc not installed. Install: pip install pyodbc, "
                "or use service='sparql_wrapper'"
            )
        connection = pyodbc.connect(
            self.odbc_config
        )
        connection.setdecoding(pyodbc.SQL_CHAR, encoding='utf8')
        connection.setdecoding(pyodbc.SQL_WCHAR, encoding='utf8')
        connection.setencoding(encoding='utf8')
        connection.timeout = self.timeout       # SQL_ATTR_CONNECTION_TIMEOUT (login timeout)
        # SQL_ATTR_QUERY_TIMEOUT = 0: query execution timeout, inherited by all cursors
        connection.set_attr(0, self.timeout)
        return connection

    def execute_query_with_odbc(self, query):
        """GrailQA 使用 Dki-lab 修复了 Literal 类型问题之后的 Freebase dump"""
        try:
            query = f"{self.ODBC_PREFIX} {query}"
            with self.connect().cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
        except BaseException as err:
            self.logger.error(f"Query Execution Failed: {query}, error: {err}; error_str: {str(err)}")
            return []
        return rows

    def _execute_query_odbc_dict(self, query, retry=4):
        """Execute SPARQL via ODBC, return SPARQLWrapper-compatible dict bindings.

        Converts ODBC tuple rows to the dict format expected by
        execute_query() / get_execution_result_one_variable_sparql_wrapper().
        Uses cursor.description for column names and infers type info from
        Python value types.

        通过 ODBC 执行 SPARQL 查询，返回与 SPARQLWrapper 兼容的 dict bindings 格式。
        使用 cursor.description 获取列名，从 Python 类型推断 type/datatype 信息。
        """
        for idx in range(retry):
            if idx > 0:
                self.logger.info(f"Retrying ODBC execute_query_dict(); idx:{idx}")
            try:
                formatted = f"{self.ODBC_PREFIX} {query}"
                with self.connect().cursor() as cursor:
                    cursor.execute(formatted)
                    columns = [col[0] for col in (cursor.description or [])]
                    rows = cursor.fetchall()
                results = []
                for row in rows:
                    result = {}
                    for col_name, val in zip(columns, row):
                        result[col_name] = _odbc_value_to_binding(val)
                    results.append(result)
                return results
            except BaseException as err:
                self.logger.error(
                    f"ODBC Query Failed: {query}, error: {str(err)}")
        return []

    # ============================================================
    # 双后端归一化层 / Unified backend layer
    # 将 ODBC (tuple rows) 和 SPARQLWrapper (dict bindings) 归一化为统一格式
    # Normalize ODBC (tuple rows) and SPARQLWrapper (dict bindings) into a uniform format
    # ============================================================

    def _bindings_to_rows(self, bindings):
        """
        将 SPARQLWrapper dict bindings 转为 ODBC 兼容的 tuple 格式.
        / Convert SPARQLWrapper dict bindings to ODBC-compatible tuple format.

        SPARQLWrapper 返回 / returns:
          [{'var': {'type':'uri','value':'http://...'}}, ...]
        ODBC 返回 / returns:
          [('http://...', 1), ...]

        关键 / Key: 对 typed-literal 做类型还原 (boolean→int, integer→int 等)
        以模拟 Virtuoso ODBC 驱动对原生类型的返回行为.
        / Coerce typed-literals to native Python types to match Virtuoso ODBC driver behavior.
        """
        if not bindings:
            return []
        # Python 3.7+ dict 保序，列顺序与 SELECT 变量顺序一致
        # / Python 3.7+ dicts preserve insertion order → matches SELECT variable order
        columns = list(bindings[0].keys())
        rows = []
        for b in bindings:
            row = []
            for c in columns:
                binding = b.get(c, {})
                val = binding.get('value', '')
                dtype = binding.get('datatype', '')
                # 类型还原以匹配 ODBC Virtuoso 行为 / Type coercion to match ODBC Virtuoso
                if dtype.endswith('#boolean'):
                    val = 1 if val in ('true', '1') else 0
                elif dtype.endswith('#integer'):
                    val = int(val)
                elif dtype.endswith('#float') or dtype.endswith('#double') or dtype.endswith('#decimal'):
                    val = float(val)
                else:
                    # URI / literal / 其他 → 保持字符串
                    val = str(val)
                row.append(val)
            rows.append(tuple(row))
        return rows

    def _execute_sparql_wrapper_raw(self, query, retry=4):
        """
        通过 SPARQLWrapper 执行查询，返回原始 convert() 结果.
        / Execute query via SPARQLWrapper, return raw convert() result.

        与 execute_query() 不同：此方法返回完整 JSON (支持 ASK 查询的 {'boolean': ...}),
        而 execute_query() 直接提取 bindings 列表.
        / Unlike execute_query(), this returns the full JSON to support ASK queries.
        """
        for idx in range(retry):
            if idx > 0:
                self.logger.info(f"Retrying SPARQLWrapper query; idx:{idx}")
            try:
                formatted = f"{self.SPARQL_PREFIX} {query}"
                self.sparql_wrapper.setQuery(formatted)
                return self.sparql_wrapper.query().convert()
            except Exception as err:
                self.logger.error(f"SPARQLWrapper Query Failed: {query}, error: {str(err)}")
        return {}

    def _execute_query(self, query, retry=4):
        """
        统一查询入口：根据 self.service 选择后端，统一返回 ODBC tuple 格式.
        / Unified query dispatcher: selects backend via self.service, always returns ODBC tuple format.

        service='sparql_wrapper': SPARQLWrapper HTTP → dict → tuple (经由 _bindings_to_rows 做类型还原)
        service='odbc':            pyodbc Virtuoso → tuple (原生 / native)

        调用方无需关心后端差异，统一使用 row[0], row[1], ... 访问.
        / Callers are agnostic to backend; they always access row[0], row[1], ...
        """
        if self.service == "sparql_wrapper":
            result = self._execute_sparql_wrapper_raw(query, retry)
            # ASK 查询: SPARQLWrapper 返回 {'boolean': True/False}
            # ODBC 返回 [(1,)] / [(0,)]
            if 'boolean' in result:
                return [(1,)] if result['boolean'] else [(0,)]
            bindings = result.get('results', {}).get('bindings', [])
            return self._bindings_to_rows(bindings)
        else:
            # ODBC 后端 / ODBC backend
            if not HAS_PYODBC:
                raise ImportError(
                    "pyodbc 未安装，无法使用 ODBC 后端. "
                    "请安装: pip install pyodbc, 或切换后端: service='sparql_wrapper'\n"
                    "pyodbc not installed. Install: pip install pyodbc, "
                    "or use service='sparql_wrapper'"
                )
            return self.execute_query_with_odbc(query)

    def execute_query(self, query, retry=4):
        # 双后端分发 / Dual-backend dispatch
        if self.service == "odbc":
            return self._execute_query_odbc_dict(query, retry)
        for idx in range(retry): # Occasionally 502 errors occur; retry a few times as a workaround. // 偶尔出现 502 错误，重试几次作为 workaround
            if idx > 0:
                self.logger.info(f"Retrying execute_query(); idx:{idx}")
            try:
                complete_query = f"{self.SPARQL_PREFIX} {query}"
                self.sparql_wrapper.setQuery(complete_query)
                results = self.sparql_wrapper.query().convert()
                return results['results']['bindings']
            except Exception as err:
                self.logger.error(f"Query Execution Failed: {query}, error: {str(err)}")
        return []

    def get_execution_result_one_variable(self, query, service=None):
        """
        执行查询并提取单一变量的值集合.
        / Execute query and extract a set of single-variable values.

        Args:
            service: (废弃/Deprecated) 由 self.service 决定. 保留参数以兼容旧调用.
        """
        rows = self._execute_query(query, retry=1)
        results = set()
        for row in rows:
            results.add(str(row[0]).replace('http://rdf.freebase.com/ns/', ''))
        return results
    
    def get_domain(self, item):
        query = f"""
        SELECT DISTINCT ?domain WHERE {{
            ns:{item} rdfs:domain ?domain
        }}
        """
        rows = self._execute_query(query)
        rtn = set()
        for row in rows:
            rtn.add(row[0].replace('http://rdf.freebase.com/ns/', ''))
        return rtn
    
    def get_range(self, item):
        query = f"""
        SELECT DISTINCT ?range WHERE {{
            ns:{item} rdfs:range ?range
        }}
        """
        rows = self._execute_query(query)
        rtn = set()
        for row in rows:
            rtn.add(row[0].replace('http://rdf.freebase.com/ns/', ''))
        return rtn

    def get_execution_result_one_variable_sparql_wrapper(self, query, retry=4):
        """
        Special implementation for Literal-typed answers.
        Concatenates the Literal type information (datatype, language tag) into the result.
        Caller must ensure the query has exactly one target variable.

        答案类型为 Literal 的特殊实现。将 Literal 类型信息拼接到结果中。
        调用此函数时请确保查询目标变量只有一个。
        """
        rows = self.execute_query(query, retry)
        results = set()
        for row in rows:
            try:
                if len(row.keys()) != 1:
                    raise Exception(f"Multiple query target: {row.keys()}")
                variable_name = list(row.keys())[0]
            except:
                self.logger.error(f"row.keys(): {row.keys()}; query: {query}")
                return results # empty
            results.add(
                FreebaseConstantForConstruction(
                    row[variable_name]['type'], row[variable_name]['value'], 
                    row[variable_name].get('datatype', None), row[variable_name].get('xml:lang', None)
                ).__repr__()
            )
        return results
    
    def get_category_classes(self, mid):
        '''目录形式的 class'''
        query = f"""
        SELECT DISTINCT ?type WHERE {{
            ns:{mid} ns:type.object.type ?type .
        }}
        """  
        rows = self._execute_query(query)
        types = set()
        for row in rows:
            types.add(row[0].replace('http://rdf.freebase.com/ns/', ''))
        return list(types)  
    
    def get_friendly_name(self, kb_item):
        """
        Referring to the implementation of GrailQA: https://github.com/dki-lab/GrailQA/blob/0de52d18463e986047165dfeb085272513cad1ad/utils/sparql_executer.py#L146
        plus language requirement（"EN") and LIMIT 1
        @param kb_item: {"mid": , "type":}
        """
        if kb_item["type"].lower() in ['entity', 'class']:
            kb_item_rep = f"ns:{kb_item['mid']}"
        elif kb_item["type"].lower() in ['literal']:
            kb_item_rep = kb_item["mid"]
        
        query = f"""
        SELECT DISTINCT ?x WHERE {{
            {kb_item_rep} ns:type.object.name ?x .
            FILTER (langMatches( lang(?x), "EN" ) )
        }} LIMIT 1
        """
        rows = self._execute_query(query)
        name = rows[0][0] if len(rows) >= 1 else "" # Return empty string for consistent downstream handling. // 返回空串保证下游处理一致

        if not name:
            query2 = f"""
            SELECT DISTINCT ?x WHERE {{
                {kb_item_rep} ns:common.topic.alias ?x .
                FILTER (langMatches( lang(?x), "EN" ) )
            }} LIMIT 1
            """
            rows = self._execute_query(query2)
            name = rows[0][0] if len(rows) >= 1 else ""
        
        return name
    
    def get_all_category_types(self):
        """
        使用关系 type.object.type, 获取所有 目录 形式的 class，形如 `meteorology.tropical_cyclone`
        """
        query = """
        SELECT DISTINCT ?type WHERE {
            ?x ns:type.object.type ?type .
            ?x ns:type.object.type ns:common.topic .  # 实体 / literal
        }
        """
        rows = self._execute_query(query)
        types = set()
        for row in rows:
            types.add(row[0].replace('http://rdf.freebase.com/ns/', ''))
        return list(types)

    def check_class(self, item, type):
        if (item is None) or (type is None):
            return False
        if item['type'] == 'entity':
            item_rep = f"ns:{item['mid']}"
        else: # class or literal should not appear here; literals have no type // class 或 literal 不应出现，literal 无类型
            return False
        
        if type['type'] == 'class':
            type_rep = f"ns:{type['mid']}"
        else: 
            return False
        
        query = f"""
        ASK {{
            {item_rep} ns:type.object.type {type_rep} .
        }}
        """
        rows = self._execute_query(query)
        return (len(rows) > 0 and rows[0][0] == 1)
    
    def enumerate_multivariate_relations(self):
        """
        Enumerate multivariate (CVT) candidate relations in Freebase.
        Criteria: (1) prefixed with http://rdf.freebase.com/ns/,
        (2) connects to a CVT node (FILTER NOT EXISTS { ?cvt ns:type.object.name ?name . }).
        Direction is irrelevant since Freebase has reverse relations.

        枚举 Freebase 上的多元（CVT）候选关系。条件：(1) http://rdf.freebase.com/ns/ 开头，
        (2) 能指向 CVT 节点。Freebase 有逆关系，方向无需考虑。
        """
        query = f"""
        SELECT DISTINCT ?p WHERE {{
            ?s ?p ?cvt .
            FILTER regex(?p, "^http://rdf.freebase.com/ns/") .
            FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
        }}
        """
        rows = self._execute_query(query)
        results = set()
        for row in rows:
            results.add(row[0].replace('http://rdf.freebase.com/ns/', ''))
        return results
    
    def enumerate_freebase_relations(self):
        query = f"""
        SELECT DISTINCT ?p WHERE {{
            ?s ?p ?o .
            FILTER (strstarts(str(?p),"http://rdf.freebase.com/ns/")) .
        }}
        """
        rows = self._execute_query(query)
        results = set()
        for row in rows:
            results.add(row[0].replace('http://rdf.freebase.com/ns/', ''))
        return results
    
    def check_cvt_relation(self, relation):
        """
        Paper: Sec 5.3 — CVT handling.
        Check whether a relation can connect to a CVT (Compound Value Type) node.
        检查某个关系是否能连接到 CVT 节点。
        """
        query = f"""
        ASK {{
        {{?s ns:{relation} ?cvt .}} UNION {{?cvt ns:{relation} ?s .}}
        FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
        }}
        """
        rows = self._execute_query(query)
        return (len(rows) > 0 and rows[0][0] == 1)

    def check_two_relation_cvt_existence(self, relation_1, relation_2):
        """
        两个关系在 Freebase 上是否能连到同一个 CVT 节点上
        """
        query = f"""
        ASK {{
            {{?cvt ns:{relation_1} ?o1 . ?cvt ns:{relation_2} ?o2 .}} 
            UNION {{?o1 ns:{relation_1} ?cvt . ?cvt ns:{relation_2} ?o2 .}}
            UNION {{?cvt ns:{relation_1} ?o1 . ?o2 ns:{relation_2} ?cvt .}}
            UNION {{?o1 ns:{relation_1} ?cvt . ?o2 ns:{relation_2} ?cvt .}}
            FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
        }} 
        """
        rows = self._execute_query(query)
        return {
            "relation_1": relation_1,
            "relation_2": relation_2,
            "existence": len(rows) > 0 and rows[0][0] == 1
        }
    

    def get_third_cvt_connected_relation(self, conn_trip, cvt_var):
        if conn_trip["type"] == "S-S":
            conn_trip_serialized = f"?cvt ns:{conn_trip['r1']} ?o . ?cvt ns:{conn_trip['r2']} ?t ."
        elif conn_trip["type"] == "S-O":
            conn_trip_serialized = f"?cvt ns:{conn_trip['r1']} ?o . ?t ns:{conn_trip['r2']} ?cvt ."
        elif conn_trip["type"] == "O-S":
            conn_trip_serialized = f"?s ns:{conn_trip['r1']} ?cvt . ?cvt ns:{conn_trip['r2']} ?t ."
        elif conn_trip["type"] == "O-O":
            conn_trip_serialized = f"?s ns:{conn_trip['r1']} ?cvt . ?t ns:{conn_trip['r2']} ?cvt ."
        else:
            raise NotImplementedError(f"conn_trip type: {conn_trip['type']}")
        """
        前面两个关系确定，查询和 CVT 节点相连的第三个关系
        @param conn_trip: {"r1", "r2", "type"}
        @param cvt_var: 形如 ?o, 指示哪个位置是 CVT 节点
        """
        rel_conn_info = (
            lambda _conn_trip: f"""
            SELECT DISTINCT ?p WHERE {{
                {{ # 子查询先查到 CVT 节点的可能取值
                    SELECT DISTINCT {cvt_var} WHERE {{
                        {conn_trip_serialized}
                        FILTER NOT EXISTS {{ {cvt_var} ns:type.object.name ?name . }} . # 要求 var 是一个 CVT 节点
                    }}
                    LIMIT 10000000 # 数量级限制，超过这一限制的就只能忽略了
                }}
                {_conn_trip} # 新关系存在不同的相连方式（方向上的不同）
                FILTER (strstarts(str(?p),"http://rdf.freebase.com/ns/")) .
            }}
            """
        )
        s_trip = f"{cvt_var} ?p ?k"
        o_trip = f"?k ?p {cvt_var}"
        results = dict()
        results["S"] = list(self.get_execution_result_one_variable(
            rel_conn_info(s_trip)
        ))
        results["O"] = list(self.get_execution_result_one_variable(
            rel_conn_info(o_trip)
        ))
        results["conn_trip"] = conn_trip
        return results
 
    def get_reverse_property(self, rel):
        query = f"""
        SELECT ?o where {{
            ns:{rel} ns:type.property.reverse_property ?o .
            FILTER (strstarts(str(?o),"http://rdf.freebase.com/ns/")) .
        }}
        """
        results = list(self.get_execution_result_one_variable(query))
        return {
            "rel": rel,
            "results": results
        }
    
    def query_entity_label(self, entity):
        query = f"""
        SELECT DISTINCT ?label WHERE {{
            ns:{entity} ns:type.object.name ?label .
            FILTER(LANG(?label) = "en")
        }}
        """
        rows = self._execute_query(query)
        label = rows[0][0] if len(rows) >= 1 else None
        return {
            "entity": entity,
            "label": label
        }

    def query_entity_list_label(self, entity_list):
        entity_rep_list = ['ns:' + ent for ent in entity_list]
        print(len(entity_rep_list))
        entity_labels = {}
        #注意，如果ent太多，要进行切片机制，尝试每个查询查100个ent的label
        for i in range(0, math.ceil(len(entity_rep_list)/1000)):
            ent_rep_split = entity_rep_list[i*1000: min(len(entity_rep_list), (i+1)*1000)]
            query = f"""
            SELECT DISTINCT ?entity ?label WHERE {{
                {{ VALUES ?entity {{ {" ".join(ent_rep_split)} }} }}
                {{
                ?entity ns:type.object.name ?label .
                FILTER(LANG(?label) = "en")
                }}
            }}
            """
            for row in self._execute_query(query):
                entity_labels[row[0].replace('http://rdf.freebase.com/ns/', '')] = row[1]
        return entity_labels

    def query_entity_aliases(self, entity):
        query = f"""
        SELECT DISTINCT ?alias WHERE {{
            ns:{entity} ns:common.topic.alias ?alias .
            FILTER(LANG(?alias) = 'en')
        }}
        """
        rows = self._execute_query(query)
        aliases = set()
        for row in rows:
            aliases.add(row[0])
        return {
            "entity": entity,
            "aliases": list(aliases)
        } 

    def add_non_cvt_filter(self, term):
        if term.startswith("?"):
            return f"""FILTER (EXISTS {{ {term} ns:type.object.name ?name . }} || (isNumeric({term})) || (
                        datatype({term})
                        IN (<http://www.w3.org/2001/XMLSchema#date>, <http://www.w3.org/2001/XMLSchema#dateTime>, 
                        <http://www.w3.org/2001/XMLSchema#gYear>, <http://www.w3.org/2001/XMLSchema#gYearMonth>)
                    ) ).\n"""
        else:
            return ""

    def expand_next_hop_path_with_LF(self, LF:SimpleGraph, expand_point:Node, end_point = None, answer = None, get_end_points = False, semantic_sim_ranker:PLMSimRanker = None, question = None):
        '''
        Paper: Sec 5.3 Exploration — answer-anchored one-hop expansion.
        Given a logical form (LF), expand from `start_point` by one hop `p`
        to reach `end_point`, returning all relations `p` satisfying the condition.

        给定一个 LF，从 start_point 出发向前一跳 p 到达 end_point，返回满足条件的所有关系 p。
        If end_point is None, all possible next relations are returned.
        Merges one_hop_path and one_hop_reversed. CVT relations are handled with
        a two-phase optimization (r1 filtering + semantic ranking → r2 lookup).

        Two expansion modes: non-CVT (single-hop) and CVT (two-hop via ?cvt node).
        Uses caching to avoid repeated queries. Optimizes small expand-point sets
        by enumerating VALUES instead of joining with the LF graph pattern.
        '''
        def add_cvt_neq_filter(expand_point_rep, end_point_rep):
            # Tricky: FILTER (a1 != a2) can cause empty results, affecting Phase 2. // 很微妙：FILTER (a1 != a2) 会导致空结果影响 Phase 2
            if not expand_point_rep.startswith("?") and not end_point_rep.startswith("?"):
                return ""
            else:
                return f"FILTER ({expand_point_rep} != {end_point_rep}). "

        if LF is not None:
            if answer is not None:
                sparql_gp = LF.get_sparql_gp_with_answer(answer)
            else:
                sparql_gp = LF.to_sparql_gp()
        else:
            sparql_gp = ""
        if LF is None and answer is not None:
            #if (answer["type"].lower() == "entity") or (FreebaseConstantForConstruction.get_constant_type(answer['mid']) is FREEBASE_CONSTANT_TYPE.STRING):
            if answer["type"].lower() == "entity":
                expand_point_rep = f"ns:{answer['mid']}"
            else:
                expand_point_rep = answer['mid']
        else:
            expand_point_rep = expand_point.value
        if end_point is None:
            end_point_rep = "?end"
        #elif (end_point["type"].lower() == "entity") or (FreebaseConstantForConstruction.get_constant_type(end_point['mid']) is FREEBASE_CONSTANT_TYPE.STRING):
        else:
            if end_point["type"].lower() == "entity":
                end_point_rep = f"ns:{end_point['mid']}"
            else:
                end_point_rep = end_point['mid']
        # else:
        #     end_point_rep = end_point['mid']
            #raise Exception("not implemented")
        if get_end_points:
            query_variables_wo_cvt = "?x, ?end, ?reversed"
            query_variables_with_cvt = "?r1, ?r2, ?end"
        else:
            query_variables_wo_cvt = "?x, ?reversed"
            query_variables_with_cvt = "?r1, ?r2"
        if LF is not None:
            # Optimization: when expand_point has small cardinality, enumerate its VALUES instead of joining. // 优化：expand_point 度数小时直接穷举 VALUES
            # CVT two-hop queries frequently timeout due to combinatorial explosion; manually split into two phases. // CVT 两跳查询经常超时，拆为两阶段
            temp_q = f"SELECT DISTINCT COUNT(*) AS ?cnt WHERE {{ {sparql_gp} }}"
            expand_point_num = int(list(self.get_execution_result_one_variable(temp_q))[0])
        if LF is not None and expand_point_num <= 3:
            temp_q = f"SELECT DISTINCT {expand_point_rep} WHERE {{ {sparql_gp} }}"
            res_temp = self.get_execution_result_one_variable(temp_q)
            # Require all possible expand-point values to be entities. // 要求 expand point 的所有取值都是实体
            legal = True
            for r in res_temp:
                type_r = FreebaseConstantForConstruction.get_constant_type(r)
                if type_r != FREEBASE_CONSTANT_TYPE.CLASS and type_r != FREEBASE_CONSTANT_TYPE.ENTITY:
                    legal = False
                    break
            if legal:
                extend_point_values_rep = " ".join(["ns:" + v for v in self.get_execution_result_one_variable(temp_q)])
            else:
                extend_point_values_rep = None
        else:
            extend_point_values_rep = None
        #第一部分：非CVT关系=============================================================================================================
        if extend_point_values_rep is not None:
            query_wo_cvt = f"""
                SELECT DISTINCT {query_variables_wo_cvt} where {{
                    {{ VALUES {expand_point_rep} {{ {extend_point_values_rep} }}. 
                    {expand_point_rep} ?x {end_point_rep} .
                      BIND(False AS ?reversed) }}
                    UNION 
                    {{ VALUES {expand_point_rep} {{ {extend_point_values_rep} }}. 
                    {end_point_rep} ?x {expand_point_rep} .
                      BIND(True AS ?reversed) }}
                    FILTER regex(?x, "^http://rdf.freebase.com/ns/") .
                    {self.add_non_cvt_filter(end_point_rep)}
                    {self.get_ignored_relations_filter("?x")} .
                }}"""
        else:
            query_wo_cvt = f"""
            SELECT DISTINCT {query_variables_wo_cvt} where {{
                {{ {sparql_gp} {expand_point_rep} ?x {end_point_rep} . BIND(False AS ?reversed) }}
                UNION 
                {{ {sparql_gp} {end_point_rep} ?x {expand_point_rep} . BIND(True AS ?reversed) }}
                FILTER regex(?x, "^http://rdf.freebase.com/ns/") .
                {self.add_non_cvt_filter(end_point_rep)}
                {self.get_ignored_relations_filter("?x")} .
            }}"""   
        #非CVT查询执行，直接一次执行即可===========================================================================================================
        results_wo_cvt = []
        if not get_end_points:
            if self.cached_results is not None:
                formated_sparql_wo_cvt = self.format_sparql(query_wo_cvt)
                if formated_sparql_wo_cvt in self.cached_results:
                    results_wo_cvt = self.cached_results[formated_sparql_wo_cvt]
                else:
                    time_1 = time.time()
                    for row in self._execute_query(query_wo_cvt):
                        r = row[0].replace('http://rdf.freebase.com/ns/', '')
                        reversed = int(row[1]) == 1
                        if reversed:
                            results_wo_cvt.append("^"+r)
                        else:
                            results_wo_cvt.append(r)
                    # Only cache slow empty queries to avoid polluting cache with rapid empty results. // 仅缓存慢的空查询，防止快速空结果污染缓存
                    if time.time() - time_1 > 0.5 or len(results_wo_cvt) > 0:
                        self.update_cache_results(formated_sparql_wo_cvt, results_wo_cvt)
            else:
                for row in self._execute_query(query_wo_cvt):
                    r = row[0].replace('http://rdf.freebase.com/ns/', '')
                    reversed = int(row[1]) == 1
                    if reversed:
                        results_wo_cvt.append("^"+r)
                    else:
                        results_wo_cvt.append(r)
        else:
            assert(0)
        # else:
        #     for row in self._execute_query(query_wo_cvt):
        #         r = row[0].replace('http://rdf.freebase.com/ns/', '')
        #         end_point = row[1].replace('http://rdf.freebase.com/ns/', '')
        #         reversed = int(row[2]) == 1
        #         results.append({"relation":r, "end":end_point, "reversed":reversed})  
        # 第二部分：CVT关系================================================================================================================
        #实际上，需要处理正反
        if self.direct_manage_2hop:
        # # 原先的，超时太夸张所以不再使用
            if extend_point_values_rep is not None:
                query_with_cvt = f"""
                SELECT DISTINCT {query_variables_with_cvt} where {{
                    VALUES {expand_point_rep} {{ {extend_point_values_rep} }} .
                    {expand_point_rep} ?r1 ?cvt.
                    ?cvt ?r2 {end_point_rep}.
                    FILTER regex(?r1, "^http://rdf.freebase.com/ns/") .
                    FILTER regex(?r2, "^http://rdf.freebase.com/ns/") .
                    {add_cvt_neq_filter(expand_point_rep, end_point_rep)}
                    FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
                    FILTER (?r1 != ?r2) .               
                    {self.get_ignored_relations_filter("?r1")} .
                    {self.get_ignored_relations_filter("?r2")} .
                }}"""
                query_with_cvt_r = f"""
                SELECT DISTINCT {query_variables_with_cvt} where {{
                    VALUES {expand_point_rep} {{ {extend_point_values_rep} }} .
                    {end_point_rep} ?r2 ?cvt.
                    ?cvt ?r1 {expand_point_rep}.
                    FILTER regex(?r1, "^http://rdf.freebase.com/ns/") .
                    FILTER regex(?r2, "^http://rdf.freebase.com/ns/") .
                    {add_cvt_neq_filter(expand_point_rep, end_point_rep)}
                    FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
                    FILTER (?r1 != ?r2) .               
                    {self.get_ignored_relations_filter("?r1")} .
                    {self.get_ignored_relations_filter("?r2")} .
                }}"""
            else:
                query_with_cvt = f"""
                SELECT DISTINCT {query_variables_with_cvt} where {{
                    {sparql_gp} 
                    {expand_point_rep} ?r1 ?cvt.
                    ?cvt ?r2 {end_point_rep}.
                    FILTER regex(?r1, "^http://rdf.freebase.com/ns/") .
                    FILTER regex(?r2, "^http://rdf.freebase.com/ns/") .
                    {add_cvt_neq_filter(expand_point_rep, end_point_rep)}
                    FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
                    FILTER (?r1 != ?r2) .                  
                    {self.get_ignored_relations_filter("?r1")} .
                    {self.get_ignored_relations_filter("?r2")} .
                }}"""
                query_with_cvt_r = f"""
                SELECT DISTINCT {query_variables_with_cvt} where {{
                    {sparql_gp} 
                    {end_point_rep} ?r2 ?cvt.
                    ?cvt ?r1 {expand_point_rep}.
                    FILTER regex(?r1, "^http://rdf.freebase.com/ns/") .
                    FILTER regex(?r2, "^http://rdf.freebase.com/ns/") .
                    {add_cvt_neq_filter(expand_point_rep, end_point_rep)}
                    FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
                    FILTER (?r1 != ?r2) .                  
                    {self.get_ignored_relations_filter("?r1")} .
                    {self.get_ignored_relations_filter("?r2")} .
                }}"""
            # if extend_point_values_rep is not None:
            #     query_with_cvt = f"""
            #     SELECT DISTINCT {query_variables_with_cvt} where {{
            #         VALUES {expand_point_rep} {{ {extend_point_values_rep} }} .
            #         ?cvt ?r1 {expand_point_rep} .
            #         ?cvt ?r2 {end_point_rep} .
            #         FILTER regex(?r1, "^http://rdf.freebase.com/ns/") .
            #         FILTER regex(?r2, "^http://rdf.freebase.com/ns/") .
            #         {add_cvt_neq_filter(expand_point_rep, end_point_rep)}
            #         FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
            #         FILTER (?r1 != ?r2) .               
            #         {self.get_ignored_relations_filter("?r1")} .
            #         {self.get_ignored_relations_filter("?r2")} .
            #     }}"""
            # else:
            #     query_with_cvt = f"""
            #     SELECT DISTINCT {query_variables_with_cvt} where {{
            #         {sparql_gp} 
            #         ?cvt ?r1 {expand_point_rep} .
            #         ?cvt ?r2 {end_point_rep} .
            #         FILTER regex(?r1, "^http://rdf.freebase.com/ns/") .
            #         FILTER regex(?r2, "^http://rdf.freebase.com/ns/") .
            #         {add_cvt_neq_filter(expand_point_rep, end_point_rep)}
            #         FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
            #         FILTER (?r1 != ?r2) .                  
            #         {self.get_ignored_relations_filter("?r1")} .
            #         {self.get_ignored_relations_filter("?r2")} .
            #     }}"""
            results_w_cvt = []
            if not get_end_points:
                for idx, query in enumerate([query_with_cvt, query_with_cvt_r]):
                    if self.cached_results is not None:
                        formated_sparql = self.format_sparql(query)
                        if formated_sparql in self.cached_results:
                            query_result = self.cached_results[formated_sparql]
                        else:
                            query_result = []
                            time_1 =time.time()
                            for row in self._execute_query(query):
                                r1 = row[0].replace('http://rdf.freebase.com/ns/', '')
                                r2 = row[1].replace('http://rdf.freebase.com/ns/', '')
                                if idx == 0:
                                    query_result.append(r1 + "/" + r2)
                                else:
                                    query_result.append("^" + r1 + "/" + "^" + r2)
                            # Only cache slow empty queries to avoid polluting cache with rapid empty results. // 仅缓存慢的空查询，防止快速空结果污染缓存
                            if time.time() - time_1 > 0.2 or len(results_w_cvt) > 0:
                                self.update_cache_results(formated_sparql, query_result)
                    else:
                        query_result = []
                        for row in self._execute_query(query):
                            r1 = row[0].replace('http://rdf.freebase.com/ns/', '')
                            r2 = row[1].replace('http://rdf.freebase.com/ns/', '')
                            if idx == 0:
                                query_result.append(r1 + "/" + r2)
                            else:
                                query_result.append("^" + r1 + "/" + "^" + r2)
                    filtered_query_result = []
                    for r in query_result:
                        legal = True
                        for black_r in self.CVT_R1_BLACKLIST:
                            if black_r in r:
                                legal = False
                                break
                        if legal:
                            filtered_query_result.append(r)
                    results_w_cvt += filtered_query_result
        else:
            if extend_point_values_rep is not None:
                query_with_cvt_phase1 = f"""
                SELECT DISTINCT ?r1 where {{
                    VALUES {expand_point_rep} {{ {extend_point_values_rep} }} .
                    ?cvt ?r1 {expand_point_rep} .
                    FILTER regex(?r1, "^http://rdf.freebase.com/ns/") .
                    FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}.                
                    {self.get_ignored_relations_filter("?r1")} .
                    }}"""
                if self.cached_results is not None:
                    formated_sparql_cvt_phase1 = self.format_sparql(query_with_cvt_phase1)
                    if formated_sparql_cvt_phase1 in self.cached_results:
                        phase1_result = self.cached_results[formated_sparql_cvt_phase1]
                    else:
                        time_1 =time.time()
                        phase1_result = list(self.get_execution_result_one_variable(query_with_cvt_phase1))
                        # Only cache slow empty queries to avoid polluting cache with rapid empty results. // 仅缓存慢的空查询，防止快速空结果污染缓存
                        if time.time() - time_1 > 0.2 or len(phase1_result) > 0:
                            self.update_cache_results(formated_sparql_cvt_phase1, phase1_result)
                else:
                    phase1_result = list(self.get_execution_result_one_variable(query_with_cvt_phase1))
                temp = []
                for r1 in phase1_result:
                    throw = False
                    for blackr in self.CVT_R1_BLACKLIST:
                        if blackr in r1:
                            throw = True
                            break
                    if not throw:
                        temp.append(r1)
                r1_results = temp
                #这里，需要使用语义相似度对r1进行一次排序（避免太多超时）
                if semantic_sim_ranker is not None:
                    if len(r1_results) > self.CVT_R1_NUM:
                        top_k_results = semantic_sim_ranker.get_semantic_sim_topk(question, r1_results, self.CVT_R1_NUM)
                        r1_results = top_k_results
                r1_results.sort()   #排序，去掉随机性
                r1_results_rep = " ".join(["ns:" + item for item in r1_results])
                query_with_cvt_phase2 = f"""
                SELECT DISTINCT {query_variables_with_cvt} where {{
                    VALUES {expand_point_rep} {{ {extend_point_values_rep} }} .
                    VALUES ?r1 {{  {r1_results_rep}  }} .
                    ?cvt ?r1 {expand_point_rep}.
                    ?cvt ?r2 {end_point_rep}.
                    {add_cvt_neq_filter(expand_point_rep, end_point_rep)}
                    FILTER regex(?r2, "^http://rdf.freebase.com/ns/") .
                    {self.add_non_cvt_filter(end_point_rep)}
                    {self.get_ignored_relations_filter("?r2")} .
                }}"""
            else:
                query_with_cvt_phase1 = f"""
                SELECT DISTINCT ?r1 where {{
                    {sparql_gp}
                    ?cvt ?r1 {expand_point_rep} .
                    FILTER regex(?r1, "^http://rdf.freebase.com/ns/") .
                    FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}.                
                    {self.get_ignored_relations_filter("?r1")} .
                    }}"""
                temp = []
                if self.cached_results is not None:
                    formated_sparql_cvt_phase1 = self.format_sparql(query_with_cvt_phase1)
                    if formated_sparql_cvt_phase1 in self.cached_results:
                        phase1_result = self.cached_results[formated_sparql_cvt_phase1]
                    else:
                        time_1 = time.time()
                        phase1_result = list(self.get_execution_result_one_variable(query_with_cvt_phase1))
                        # Only cache slow empty queries to avoid polluting cache with rapid empty results. // 仅缓存慢的空查询，防止快速空结果污染缓存
                        if time.time() - time_1 > 0.5 or len(phase1_result) > 0:
                            self.update_cache_results(formated_sparql_cvt_phase1, phase1_result)
                else:
                    phase1_result = list(self.get_execution_result_one_variable(query_with_cvt_phase1))
                for r1 in phase1_result:
                    throw = False
                    for blackr in self.CVT_R1_BLACKLIST:
                        if blackr in r1:
                            throw = True
                            break
                    if not throw:
                        temp.append(r1)
                r1_results = temp
                #这里，需要使用语义相似度对r1进行一次排序（避免太多超时）
                if semantic_sim_ranker is not None:
                    if len(r1_results) > self.CVT_R1_NUM:
                        top_k_results = semantic_sim_ranker.get_semantic_sim_topk(question, r1_results, self.CVT_R1_NUM)
                        r1_results = top_k_results
                r1_results.sort()   #排序，去掉随机性
                r1_results_rep = " ".join(["ns:" + item for item in r1_results])
                query_with_cvt_phase2 = f"""
                SELECT DISTINCT {query_variables_with_cvt} where {{
                    {sparql_gp}
                    VALUES ?r1 {{  {r1_results_rep}  }} .
                    ?cvt ?r1 {expand_point_rep}.
                    ?cvt ?r2 {end_point_rep}.
                    FILTER regex(?r2, "^http://rdf.freebase.com/ns/") .
                    {add_cvt_neq_filter(expand_point_rep, end_point_rep)}
                    {self.add_non_cvt_filter(end_point_rep)}
                    {self.get_ignored_relations_filter("?r2")} .
                }}"""             
            #=============================================执行=========================================================================
            #----------------CVT的------------------------------------------------------------------------------------------------------------------
            results_w_cvt = []
            if not get_end_points:
                if self.cached_results is not None:
                    formated_sparql_w_cvt = self.format_sparql(query_with_cvt_phase2)
                    if formated_sparql_w_cvt in self.cached_results:
                        results_w_cvt = self.cached_results[formated_sparql_w_cvt]
                    else:
                        time_1 =time.time()
                        for row in self._execute_query(query_with_cvt_phase2):
                            r1 = row[0].replace('http://rdf.freebase.com/ns/', '')
                            r2 = row[1].replace('http://rdf.freebase.com/ns/', '')
                            #我们规定，CVT节点的两条边都是出度，因而是REVERSED的
                            results_w_cvt.append("^" + r1 + "/" + r2)
                        # Only cache slow empty queries to avoid polluting cache with rapid empty results. // 仅缓存慢的空查询，防止快速空结果污染缓存
                        if time.time() - time_1 > 0.2 or len(results_w_cvt) > 0:
                            self.update_cache_results(formated_sparql_w_cvt, results_w_cvt)
                else:
                    for row in self._execute_query(query_with_cvt_phase2):
                        r1 = row[0].replace('http://rdf.freebase.com/ns/', '')
                        r2 = row[1].replace('http://rdf.freebase.com/ns/', '')
                        results_w_cvt.append("^" + r1 + "/" + r2)
            else:
                assert(0)
            # else: 
            #     #if "?cvt" not in sparql_gp:
            #     for row in self._execute_query(query_with_cvt_phase2):
            #         r1 = row[0].replace('http://rdf.freebase.com/ns/', '')
            #         r2 = row[1].replace('http://rdf.freebase.com/ns/', '')
            #         end_point = row[2].replace('http://rdf.freebase.com/ns/', '')
            #         results.append({"relation":  "^" + r1 + "/" + r2, "end":end_point})
            #===============================================================================================================================
        results = results_wo_cvt + results_w_cvt
        return results     


    def get_next_hop_items_with_LF(self, LF:SimpleGraph, expand_point:Node, answer = None, semantic_sim_ranker = None, question = None):
        '''
        Paper: Sec 5.3 Exploration — enumerate neighbor entities (Freebase).
        Given a logical form (LF), expand from `start_point` by one hop to find
        all reachable entities (with labels), returning {mid, label} dicts.

        给定 LF，从 start_point 出发一跳到达的所有实体（含 label），返回 {mid, label} 字典。
        For CVT relations, uses the two-phase optimization: r1 enumeration + blacklist
        filtering + semantic ranking → r2 lookup to get reachable entities.
        '''
        results = []
        #为key path中的属性加上前缀，同时处理取反
        if LF is not None:
            if answer is not None:
                sparql_gp = LF.get_sparql_gp_with_answer(answer)
            else:
                sparql_gp = LF.to_sparql_gp()
        else:
            sparql_gp = ""
        if LF is None and answer is not None:
            #if (answer["type"].lower() == "entity") or (FreebaseConstantForConstruction.get_constant_type(answer['mid']) is FREEBASE_CONSTANT_TYPE.STRING):
            if answer["type"].lower() == "entity":
                expand_point_rep = f"ns:{answer['mid']}"
            else:
                expand_point_rep = answer['mid']
        else:
            expand_point_rep = expand_point.value
        if LF is not None:
        # Optimization: when expand_point has small cardinality, enumerate its VALUES instead of joining. // 优化：expand_point 度数小时直接穷举 VALUES
            temp_q = f"SELECT DISTINCT COUNT(*) AS ?cnt WHERE {{ {sparql_gp} }}"
            extend_point_num = int(list(self.get_execution_result_one_variable(temp_q))[0])
        if LF is not None and extend_point_num <= 3:
            temp_q = f"SELECT DISTINCT {expand_point_rep} WHERE {{ {sparql_gp} }}"
            extend_point_values_rep = " ".join(["ns:" + v for v in self.get_execution_result_one_variable(temp_q)])
        else:
            extend_point_values_rep = None
        #第一部分：非CVT查询========================================================================================================================
        if extend_point_values_rep is not None:
            query_wo_cvt = f"""
            SELECT DISTINCT ?end ?label where {{
                {{ VALUES {expand_point_rep} {{ {extend_point_values_rep} }}. 
                {expand_point_rep} ?x ?end . }}
                UNION 
                {{ VALUES {expand_point_rep} {{ {extend_point_values_rep} }}. 
                ?end ?x {expand_point_rep} . }}
                OPTIONAL
                {{?end rdfs:label ?label .  FILTER(LANG(?label) = "en")}}
                FILTER regex(?x, "^http://rdf.freebase.com/ns/") .
                {self.get_ignored_relations_filter("?x")} .
            }} LIMIT 10000"""
        else:
            query_wo_cvt = f"""
            SELECT DISTINCT ?end ?label where {{
                {{ {sparql_gp} {expand_point_rep} ?x ?end .}}
                UNION 
                {{ {sparql_gp} ?end ?x {expand_point_rep} .}}
                OPTIONAL
                {{?end rdfs:label ?label .  FILTER(LANG(?label) = "en")}}
                FILTER regex(?x, "^http://rdf.freebase.com/ns/") .
                {self.add_non_cvt_filter("?end")}
                {self.get_ignored_relations_filter("?x")} .
            }} LIMIT 10000"""
        #直接一次执行完成即可=====================================================================================================================
        for row in self._execute_query(query_wo_cvt):
            ent = row[0].replace('http://rdf.freebase.com/ns/', '')
            label = row[1]
            results.append({"mid": ent, "label":label})         
        #第二部分：CVT查询========================================================================================================================
        if extend_point_values_rep is not None:
        # CVT two-hop queries frequently timeout due to combinatorial explosion; manually split into two phases. // CVT 两跳查询经常超时，拆为两阶段
            query_with_cvt_phase1 = f"""
            SELECT DISTINCT ?r1 where {{
                VALUES {expand_point_rep} {{ {extend_point_values_rep} }}.
                ?cvt ?r1 {expand_point_rep} .
                FILTER regex(?r1, "^http://rdf.freebase.com/ns/") .
                FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
                {self.get_ignored_relations_filter("?r1")} .
            }}"""
            #这里，需要使用语义相似度对r1进行一次排序（避免太多超时）
            temp = []
            for r1 in list(self.get_execution_result_one_variable(query_with_cvt_phase1)):
                throw = False
                for blackr in self.CVT_R1_BLACKLIST:
                    if blackr in r1:
                        throw = True
                        break
                if not throw:
                    temp.append(r1)
            r1_results = temp
            if semantic_sim_ranker is not None:
                if len(r1_results) > self.CVT_R1_NUM:
                    top_k_results = semantic_sim_ranker.get_semantic_sim_topk(question, r1_results, self.CVT_R1_NUM)
                    r1_results = top_k_results
            r1_results_rep = " ".join(["ns:" + item for item in r1_results])
            query_with_cvt_phase2 = f"""
            SELECT DISTINCT ?end ?label where {{
                VALUES {expand_point_rep} {{ {extend_point_values_rep} }} .
                VALUES ?r1 {{  {r1_results_rep}  }} .
                {{ ?cvt ?r1 {expand_point_rep}. ?cvt ?r2 ?end. }}
                OPTIONAL
                {{?end rdfs:label ?label .  FILTER(LANG(?label) = "en")}}
                FILTER (?end != {expand_point_rep}) .
                FILTER regex(?r2, "^http://rdf.freebase.com/ns/") .
                {self.get_ignored_relations_filter("?r2")} .
            }} LIMIT 1000"""
        else:
        # 不含 CVT 节点的一跳路径，注意：不包含CVT关系，即end-point不能是CVT节点
            # #考虑CVT的一跳路径：更新：所有CVT结构，都是?a <---- ?cvt ----> ?b， 因而prev_path = 1时只有reversed = True
            query_with_cvt_phase1 = f"""
            SELECT DISTINCT ?r1 where {{
                {sparql_gp}
                ?cvt ?r1 {expand_point_rep} .
                FILTER regex(?r1, "^http://rdf.freebase.com/ns/") .
                FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
                {self.get_ignored_relations_filter("?r1")} .
            }}"""
            #这里，需要使用语义相似度对r1进行一次排序（避免太多超时）
            temp = []
            for r1 in list(self.get_execution_result_one_variable(query_with_cvt_phase1)):
                throw = False
                for blackr in self.CVT_R1_BLACKLIST:
                    if blackr in r1:
                        throw = True
                        break
                if not throw:
                    temp.append(r1)
            r1_results = temp
            if semantic_sim_ranker is not None:
                if len(r1_results) > self.CVT_R1_NUM:
                    top_k_results = semantic_sim_ranker.get_semantic_sim_topk(question, r1_results, self.CVT_R1_NUM)
                    r1_results = top_k_results
            r1_results_rep = " ".join(["ns:" + item for item in r1_results])
            query_with_cvt_phase2 = f"""
            SELECT DISTINCT ?end ?label where {{
                {{VALUES ?r1 {{  {r1_results_rep}  }} .
                {sparql_gp}
                ?cvt ?r1 {expand_point_rep}. ?cvt ?r2 ?end. }}
                OPTIONAL
                {{?end rdfs:label ?label .  FILTER(LANG(?label) = "en")}}
                FILTER (?end != {expand_point_rep}) .
                FILTER regex(?r2, "^http://rdf.freebase.com/ns/") .
                {self.get_ignored_relations_filter("?r2")} .
            }} LIMIT 100"""
        #执行：执行第二阶段获取?end=======================================================================================================
        for row in self._execute_query(query_with_cvt_phase2):
            ent = row[0].replace('http://rdf.freebase.com/ns/', '')
            label = row[1]
            results.append({"mid": ent, "label":label})         
        return results     


class SparqlOdbcQuerierNoSexprWikidata(ConcurrentExecutor):
    """
    SPARQL executor for Wikidata via ODBC (Virtuoso) + SPARQLWrapper HTTP.
    Wikidata KB 后端：通过 ODBC + SPARQLWrapper HTTP 执行 SPARQL 查询。
    Paper: Sec 4.1 KB Backend, Sec 5.3 Query Graph Construction (Exploration).

    Supports: one-hop path querying (ps/pq/wdt), CVT/Statement handling,
    multivariate relation instantiation, ARGMIN/ARGMAX relation enumeration,
    and Freebase↔Wikidata MID mapping.
    """
    def __init__(self, sparql_wrapper_path, logger, service="sparql_wrapper",
                 odbc_config=None, timeout=10, sparql_cache_dir=None):
        super().__init__(logger)
        self.service = service
        if self.service == "odbc" and not HAS_PYODBC:
            raise ImportError(
                "pyodbc 未安装，无法使用 ODBC 后端。请安装: pip install pyodbc\n"
                "pyodbc not installed. Install: pip install pyodbc\n"
                "或使用默认的 SPARQLWrapper 后端: service='sparql_wrapper'"
            )
        if self.service == "odbc" and odbc_config is None:
            raise ValueError("service='odbc' 需要提供 odbc_config / requires odbc_config")
        self.odbc_config = odbc_config
        self.sparql_wrapper_path = sparql_wrapper_path
        self.timeout = timeout
        self.ODBC_PREFIX = """SPARQL
        PREFIX p: <http://www.wikidata.org/prop/>
        PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
        PREFIX ps: <http://www.wikidata.org/prop/statement/>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX wds: <http://www.wikidata.org/entity/statement/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        """
        self.SPARQL_PREFIX = """
        PREFIX p: <http://www.wikidata.org/prop/>
        PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
        PREFIX ps: <http://www.wikidata.org/prop/statement/>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX wds: <http://www.wikidata.org/entity/statement/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        """
        self.sparql_wrapper = SPARQLWrapper(self.sparql_wrapper_path)
        self.sparql_wrapper.setReturnFormat(JSON)
        self.sparql_wrapper.setTimeout(self.timeout)
        self.connection = None
        self.sparql_cache_dir = sparql_cache_dir
        if self.sparql_cache_dir is not None:
            if not os.path.isfile(self.sparql_cache_dir):
                dump_json(dict(), self.sparql_cache_dir)
            self.cached_results = load_json(self.sparql_cache_dir)
            # Always save a backup to prevent cache corruption from file modification. // 每次保存备份，防止修改源文件损坏缓存
            dump_json(self.cached_results, sparql_cache_dir.split(".")[0]+"_backup.json")
            self.num_new_cache = 0
        else:
            self.cached_results = None
    
    def format_sparql(self, sparql):
        return sparql.replace("\n", " ").replace(" ", "")
    
    def write_current_cache_to_file(self):
        dump_json(self.cached_results, self.sparql_cache_dir)

    def update_cache_results(self, sparql, results, save_split = 1000):
        # Update cache; flush new entries to file when exceeding save_split. // 更新缓存，新增超过 save_split 条时写入文件
        self.cached_results[sparql] = results
        self.num_new_cache += 1
        if self.num_new_cache == save_split:
            self.write_current_cache_to_file()
            self.num_new_cache = 0
            self.cached_results = load_json(self.sparql_cache_dir)

    def connect(self):
        if not HAS_PYODBC:
            raise ImportError(
                "pyodbc 未安装，无法建立 ODBC 连接. "
                "请安装: pip install pyodbc, 或使用 service='sparql_wrapper'\n"
                "pyodbc not installed. Install: pip install pyodbc, "
                "or use service='sparql_wrapper'"
            )
        if self.connection is None:
            connection = pyodbc.connect(
                self.odbc_config
            )
            connection.setdecoding(pyodbc.SQL_CHAR, encoding='utf8')
            connection.setdecoding(pyodbc.SQL_WCHAR, encoding='utf8')
            connection.setencoding(encoding='utf8')
            connection.timeout = self.timeout       # SQL_ATTR_CONNECTION_TIMEOUT (login timeout)
            # SQL_ATTR_QUERY_TIMEOUT = 0: query execution timeout, inherited by all cursors
            connection.set_attr(0, self.timeout)
            self.connection = connection
        return self.connection

    def execute_query_with_odbc(self, query):
        try:
            query = f"{self.ODBC_PREFIX} {query}"
            with self.connect().cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
        except BaseException as err:
            self.logger.error(f"Query Execution Failed: {query}, error: {err}; error_str: {str(err)}")
            return []
        return rows

    def _execute_query_odbc_dict(self, query, retry=4):
        """Execute SPARQL via ODBC, return SPARQLWrapper-compatible dict bindings.

        Converts ODBC tuple rows to the dict format expected by
        execute_query() / get_execution_result_one_variable_sparql_wrapper().
        Uses cursor.description for column names and infers type info from
        Python value types.

        通过 ODBC 执行 SPARQL 查询，返回与 SPARQLWrapper 兼容的 dict bindings 格式。
        """
        for idx in range(retry):
            if idx > 0:
                self.logger.info(
                    f"Retrying ODBC execute_query_dict(); idx:{idx}")
            try:
                formatted = f"{self.ODBC_PREFIX} {query}"
                with self.connect().cursor() as cursor:
                    cursor.execute(formatted)
                    columns = [col[0] for col in (cursor.description or [])]
                    rows = cursor.fetchall()
                results = []
                for row in rows:
                    result = {}
                    for col_name, val in zip(columns, row):
                        result[col_name] = _odbc_value_to_binding(val)
                    results.append(result)
                return results
            except BaseException as err:
                self.logger.error(
                    f"ODBC Query Failed: {query}, error: {str(err)}")
        return []

    def execute_query_with_odbc_with_err(self, query):
        try:
            query = f"{self.ODBC_PREFIX} {query}"
            with self.connect().cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
        except BaseException as err:
            self.logger.error(f"Query Execution Failed: {query}, error: {err}; error_str: {str(err)}")
            return ["ERROR"]
        return rows
    
    # ============================================================
    # 双后端归一化层 / Unified backend layer
    # 与 Freebase 版本相同：将 ODBC tuple rows 和 SPARQLWrapper dict bindings 归一化
    # Identical to Freebase version: normalizes ODBC tuples and SPARQLWrapper dicts
    # ============================================================

    def _bindings_to_rows(self, bindings):
        """
        将 SPARQLWrapper dict bindings 转为 ODBC 兼容的 tuple 格式，含类型还原.
        / Convert SPARQLWrapper dict bindings to ODBC-compatible tuple format with type coercion.
        """
        if not bindings:
            return []
        columns = list(bindings[0].keys())
        rows = []
        for b in bindings:
            row = []
            for c in columns:
                binding = b.get(c, {})
                val = binding.get('value', '')
                dtype = binding.get('datatype', '')
                if dtype.endswith('#boolean'):
                    val = 1 if val in ('true', '1') else 0
                elif dtype.endswith('#integer'):
                    val = int(val)
                elif dtype.endswith('#float') or dtype.endswith('#double') or dtype.endswith('#decimal'):
                    val = float(val)
                else:
                    val = str(val)
                row.append(val)
            rows.append(tuple(row))
        return rows

    def _execute_sparql_wrapper_raw(self, query, retry=4):
        """通过 SPARQLWrapper 执行查询，返回原始 convert() 结果 (支持 ASK)."""
        for idx in range(retry):
            if idx > 0:
                self.logger.info(f"Retrying SPARQLWrapper query; idx:{idx}")
            try:
                formatted = f"{self.SPARQL_PREFIX} {query}"
                self.sparql_wrapper.setQuery(formatted)
                return self.sparql_wrapper.query().convert()
            except Exception as err:
                self.logger.error(f"SPARQLWrapper Query Failed: {query}, error: {str(err)}")
        return {}

    def _execute_query(self, query, retry=4):
        """
        统一查询入口 / Unified query dispatcher.
        service='sparql_wrapper' → dict bindings → tuple (含类型还原 / with type coercion)
        service='odbc' → pyodbc tuple (原生 / native)
        """
        if self.service == "sparql_wrapper":
            result = self._execute_sparql_wrapper_raw(query, retry)
            if 'boolean' in result:
                return [(1,)] if result['boolean'] else [(0,)]
            bindings = result.get('results', {}).get('bindings', [])
            return self._bindings_to_rows(bindings)
        else:
            if not HAS_PYODBC:
                raise ImportError(
                    "pyodbc 未安装，无法使用 ODBC 后端. "
                    "请安装: pip install pyodbc, 或切换后端: service='sparql_wrapper'\n"
                    "pyodbc not installed. Install: pip install pyodbc, "
                    "or use service='sparql_wrapper'"
                )
            return self.execute_query_with_odbc(query)

    def execute_query_with_err(self, query):
        try:
            complete_query = f"{self.SPARQL_PREFIX} {query}"
            self.sparql_wrapper.setQuery(complete_query)
            results = self.sparql_wrapper.query().convert()
            return results['results']['bindings'] # ASK 类型语句会报错，但是本身我们也处理不了 ASK
        except Exception as err:
            self.logger.error(f"Query Execution Failed: {query}, error: {str(err)}")
            return ["ERROR"]

    def execute_query(self, query, retry=4):
        # 双后端分发 / Dual-backend dispatch
        if self.service == "odbc":
            return self._execute_query_odbc_dict(query, retry)
        for idx in range(retry): # Occasionally 502 errors occur; retry a few times as a workaround. // 偶尔出现 502 错误，重试几次作为 workaround
            if idx > 0:
                self.logger.info(f"execute_query(); idx:{idx}")
            try:
                complete_query = f"{self.SPARQL_PREFIX} {query}"
                self.sparql_wrapper.setQuery(complete_query)
                results = self.sparql_wrapper.query().convert()
                return results['results']['bindings']
            except Exception as err:
                self.logger.error(f"Query Execution Failed: {query}, error: {str(err)}")
        return []

    def get_execution_result_one_variable(self, query, service=None):
        """
        执行查询并提取单一变量的值集合 (Wikidata 版).
        / Execute query and extract set of single-variable values (Wikidata).
        service: (废弃/Deprecated) 由 self.service 决定.
        """
        rows = self._execute_query(query, retry=1)
        results = set()
        for row in rows:
            results.add(str(row[0]))
        return results

    def get_execution_result_one_variable_sparql_wrapper(self, query, retry=4):
        rows = self.execute_query(query, retry)
        results = set()
        try:
            for row in rows:
                if len(list(row.keys())) != 1:
                    raise Exception(f"row.keys(): {row.keys()}")
                variable_name = list(row.keys())[0]
                variable_binding = row[variable_name]
                constant_item = WikidataConstantForConstruction(
                    variable_binding['type'], variable_binding['value'], variable_binding.get('datatype', None), variable_binding.get('xml:lang', None)
                ).__repr__()
                if WikidataConstantForConstruction.get_constant_type(constant_item) is None:
                    raise Exception(f"Unhandled constant: {constant_item}")
                else:
                    results.add(constant_item)
            return results
        except Exception as e:
            self.logger.error(f"query: {query}; error: {e}")
            return set()
    
    
    def get_candidate_properties(self, property_type_list):
        property_set = set()
        for prop_type in property_type_list:
            query = f"""
            SELECT ?prop WHERE {{
                ?prop wikibase:propertyType wikibase:{prop_type}
            }}
            """
            query_result = self._execute_query(query)
            for row in query_result:
                property_set.add(row[0].replace('http://www.wikidata.org/entity/', ''))
        return property_set
    
    def query_entity_label(self, entity):
        query = f"""
        SELECT DISTINCT ?label WHERE {{
            wd:{entity} rdfs:label ?label . 
            FILTER(LANG(?label) = "en")
        }} LIMIT 1
        """
        rows = self._execute_query(query)
        label = rows[0][0] if len(rows) >= 1 else None
        return {
            "entity": entity,
            "label": label
        }

    def get_friendly_name(self, kb_item):
        item_rep = kb_item['mid']
        query = f"""
        SELECT DISTINCT ?label WHERE {{
            {item_rep} rdfs:label ?label . 
            FILTER(LANG(?label) = "en")
        }} LIMIT 1
        """
        rows = self._execute_query(query)
        label = rows[0][0] if len(rows) >= 1 else ""
        return label

    def check_cvt_existence(self, prop_ps, prop_pq):
        """
        @return: True, KB 上存在这样的 CVT 组合
        """
        query = f"""
        ASK {{
            ?statement ps:{prop_ps} ?o1 .
            ?statement pq:{prop_pq} ?o2 .
            FILTER (strstarts(str(?statement), "http://www.wikidata.org/entity/statement/")).
        }} 
        """
        rows = self._execute_query(query)
        return {
            "prop_ps": prop_ps,
            "prop_pq": prop_pq,
            "existence": len(rows) > 0 and rows[0][0] == 1
        } 

    def query_entity_aliases(self, entity):
        query = f"""
        SELECT DISTINCT ?alias WHERE {{
            wd:{entity} skos:altLabel ?alias .
            FILTER(LANG(?alias) = 'en')
        }}
        """
        rows = self._execute_query(query)
        aliases = set()
        for row in rows:
            aliases.add(row[0])
        return {
            "entity": entity,
            "aliases": list(aliases)
        } 
    
    def get_freebase_mid_from_wikidata_mid(self, wikidata_mid):
        """查询结果有多个的情况，我们也只返回一个"""
        def post_process_mid(mid):
            mid = mid[1:] # m/06whf7
            mid = f"{mid[0]}.{mid[2:]}" # m.06whf7
            return mid
        query1 = f"""
        SELECT ?o WHERE {{
            wd:{wikidata_mid} wdt:P646 ?o .
        }}
        """
        rows = self._execute_query(query1)
        freebase_mid = post_process_mid(rows[0][0]) if len(rows) >= 1 else "" # Return empty string for consistent downstream handling. // 返回空串保证下游处理一致
        
        if not freebase_mid:
            '''Google KG id, 同样出现在 freebase 中，g.123'''
            query2 = f"""
            SELECT ?o WHERE {{
                wd:{wikidata_mid} wdt:P2671 ?o .
            }}
            """
            rows = self._execute_query(query2)
            freebase_mid = post_process_mid(rows[0][0]) if len(rows) >= 1 else ""
        
        return {
            "wikidata_mid": wikidata_mid,
            "freebase_mid": freebase_mid
        }

    def get_wikidata_mid_from_freebase_mid(self, freebase_mid):
        """查询结果有多个的情况，我们也只返回一个"""
        def post_process_freebase_mid(mid):
            """@param mid:m.0695j"""
            first, second = mid.split('.')
            mid = f'"/{first}/{second}"' # "/m/0695j"
            return mid
        processed_freebase_mid = post_process_freebase_mid(freebase_mid)
        query1 = f"""
        SELECT ?s WHERE {{
            ?s wdt:P646 {processed_freebase_mid}.
        }}
        """
        rows = self._execute_query(query1)
        wikidata_mid = rows[0][0].replace('http://www.wikidata.org/entity/', '') if len(rows) >= 1 else None
        
        if not wikidata_mid:
            '''Google KG id, 同样出现在 freebase 中，g.123'''
            query2 = f"""
            SELECT ?o WHERE {{
                ?s wdt:P2671 {processed_freebase_mid}.
            }}
            """
            rows = self._execute_query(query2)
            wikidata_mid = rows[0][0].replace('http://www.wikidata.org/entity/', '') if len(rows) >= 1 else None # Return empty string for consistent downstream handling. // 返回空串保证下游处理一致
        
        return {
            "freebase_mid": processed_freebase_mid,
            "wikidata_mid": wikidata_mid
        }
    
    def query_one_hop_paths(self, grounded_item, answer_entity=None, answer_type=None):
        """Paper: Sec 5.3 Exploration — one-hop path query (Wikidata).
        Query one-hop paths from `answer_entity` (or an entity of `answer_type`)
        to `grounded_item`, returning S-expression patterns.

        从 answer_entity（或 answer_type 对应的实体）出发，查询连接到 grounded_item
        的一跳路径，返回 S-expression 形式的图模式。
        Handles: entity→entity (ps/pq), entity→literal (ps/pq with comparison operators),
        and class constraints (P31/P279*).
        """
        results = list()
        if not grounded_item:
            return results
        if (not answer_entity) and (not answer_type):
            return results
        if answer_entity == grounded_item:
            return results
        if answer_type == grounded_item:
            return results

        if answer_entity is not None:
            if answer_entity["type"].lower() not in ["entity", "class"]:
                '''RDF 规范中，literal 不能位于 subject 位置'''
                return results
            answer_entity_rep = answer_entity['mid']
            answer_type_rep = None
        elif answer_type is not None:
            if answer_type['type'].lower() not in ['entity', 'class']:
                raise NotImplementedError(f"answer_type: {answer_type}")
            answer_type_rep = answer_type['mid']
            answer_entity_rep = None

        grounded_item_rep = grounded_item['mid']

        if (grounded_item["type"].lower() == "entity") or (WikidataConstantForConstruction.get_constant_type(grounded_item['mid']) is WIKIDATA_CONSTANT_TYPE.STRING):
            '''不含 CVT 的一跳: p+ps'''
            if answer_entity_rep is not None:
                query_wo_qualifier = f"""
                SELECT DISTINCT ?p, ?ps where {{
                    {answer_entity_rep} ?p [
                        ?ps {grounded_item_rep}
                    ] .
                    FILTER (strstarts(str(?p), "http://www.wikidata.org/prop/P")).
                    FILTER (strstarts(str(?ps), "http://www.wikidata.org/prop/statement/P")).
                }}
                """
                query_with_qualifier = f"""
                SELECT DISTINCT ?p, ?pq where {{
                    {answer_entity_rep} ?p [
                        ?pq {grounded_item_rep}
                    ] .
                    FILTER (strstarts(str(?p), "http://www.wikidata.org/prop/P")).
                    FILTER (strstarts(str(?pq), "http://www.wikidata.org/prop/qualifier/P")).
                }}
                """
            elif answer_type_rep is not None:
                query_wo_qualifier = f"""
                SELECT DISTINCT ?p, ?ps where {{
                    ?s ?p [
                        ?ps {grounded_item_rep}
                    ] .
                    ?s wdt:P31/wdt:P279* {answer_type_rep} .
                    FILTER (?s!={answer_type_rep} && ?s!={grounded_item_rep}) .
                    FILTER (strstarts(str(?p), "http://www.wikidata.org/prop/P")).
                    FILTER (strstarts(str(?ps), "http://www.wikidata.org/prop/statement/P")).
                }}
                """
                query_with_qualifier = f"""
                SELECT DISTINCT ?p, ?pq where {{
                    ?s ?p [
                        ?pq {grounded_item_rep}
                    ] .
                    ?s wdt:P31/wdt:P279* {answer_type_rep} .
                    FILTER (?s!={answer_type_rep} && ?s!={grounded_item_rep}) .
                    FILTER (strstarts(str(?p), "http://www.wikidata.org/prop/P")).
                    FILTER (strstarts(str(?pq), "http://www.wikidata.org/prop/qualifier/P")).
                }}
                """
            else:
                raise NotImplementedError()

            rows = self._execute_query(query_wo_qualifier)
            for row in rows:
                p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
                ps_prop = row[1].replace('http://www.wikidata.org/prop/statement/', 'ps:')
                results.append({
                    "s_expression": JOIN(p_prop, JOIN(ps_prop, grounded_item_rep)) 
                })
            
            rows = self._execute_query(query_with_qualifier)
            for row in rows:
                p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
                pq_prop = row[1].replace('http://www.wikidata.org/prop/qualifier/', 'pq:')
                results.append({
                    "s_expression": JOIN(p_prop, JOIN(pq_prop, grounded_item_rep)) 
                })
            
        elif WikidataConstantForConstruction.get_constant_type(grounded_item['mid']) in [WIKIDATA_CONSTANT_TYPE.TIME, WIKIDATA_CONSTANT_TYPE.QUANTITY]:
            '''不含 CVT 的一跳: p+ps'''
            if answer_entity_rep is not None:
                query_wo_qualifier = f"""
                SELECT DISTINCT ?p, ?ps, ?v where {{
                    {answer_entity_rep} ?p [
                        ?ps ?v
                    ] .
                    FILTER (strstarts(str(?p), "http://www.wikidata.org/prop/P")).
                    FILTER (strstarts(str(?ps), "http://www.wikidata.org/prop/statement/P")).
                    FILTER (isNumeric(?v) || (
                        datatype(?v)
                        IN (<http://www.w3.org/2001/XMLSchema#date>, <http://www.w3.org/2001/XMLSchema#dateTime>, <http://www.w3.org/2001/XMLSchema#gYear>, <http://www.w3.org/2001/XMLSchema#gYearMonth>)
                    )) .
                }} 
                """
                query_with_qualifier = f"""
                SELECT DISTINCT ?p, ?pq, ?v where {{
                    {answer_entity_rep} ?p [
                        ?pq ?v
                    ] .
                    FILTER (strstarts(str(?p), "http://www.wikidata.org/prop/P")).
                    FILTER (strstarts(str(?pq), "http://www.wikidata.org/prop/qualifier/P")).
                    FILTER (isNumeric(?v) || (
                        datatype(?v)
                        IN (<http://www.w3.org/2001/XMLSchema#date>, <http://www.w3.org/2001/XMLSchema#dateTime>, <http://www.w3.org/2001/XMLSchema#gYear>, <http://www.w3.org/2001/XMLSchema#gYearMonth>)
                    )) .
                }} 
                """
            elif answer_type_rep is not None:
                query_wo_qualifier = f"""
                SELECT DISTINCT ?p, ?ps, ?v where {{
                    ?s ?p [
                        ?ps ?v
                    ] .
                    ?s wdt:P31/wdt:P279* {answer_type_rep} .
                    FILTER (?s!={answer_type_rep} && ?s!=?v) .
                    FILTER (strstarts(str(?p), "http://www.wikidata.org/prop/P")).
                    FILTER (strstarts(str(?ps), "http://www.wikidata.org/prop/statement/P")).
                    FILTER (isNumeric(?v) || (
                        datatype(?v)
                        IN (<http://www.w3.org/2001/XMLSchema#date>, <http://www.w3.org/2001/XMLSchema#dateTime>, <http://www.w3.org/2001/XMLSchema#gYear>, <http://www.w3.org/2001/XMLSchema#gYearMonth>)
                    )) .
                }} 
                """
                query_with_qualifier = f"""
                SELECT DISTINCT ?p, ?pq, ?v where {{
                    ?s ?p [
                        ?pq ?v
                    ] .
                    ?s wdt:P31/wdt:P279* {answer_type_rep} .
                    FILTER (?s!={answer_type_rep} && ?s!=?v) .
                    FILTER (strstarts(str(?p), "http://www.wikidata.org/prop/P")).
                    FILTER (strstarts(str(?pq), "http://www.wikidata.org/prop/qualifier/P")).
                    FILTER (isNumeric(?v) || (
                        datatype(?v)
                        IN (<http://www.w3.org/2001/XMLSchema#date>, <http://www.w3.org/2001/XMLSchema#dateTime>, <http://www.w3.org/2001/XMLSchema#gYear>, <http://www.w3.org/2001/XMLSchema#gYearMonth>)
                    )) .
                }} 
                """
            else:
                raise NotImplementedError()
            
            rows = self._execute_query(query_wo_qualifier)
            for row in rows:
                p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
                ps_prop = row[1].replace('http://www.wikidata.org/prop/statement/', 'ps:')
                literal_str = row[2]
                for operator in COMPARISON_OPERATORS:
                    if compare_literal(literal_str, grounded_item_rep, operator):
                        for cmp_function in OPERATOR_FUNCTION[operator]:
                            results.append({
                                "s_expression": CMP(cmp_function, JOIN(p_prop, ps_prop), grounded_item_rep)
                            })

            rows = self._execute_query(query_with_qualifier)
            for row in rows:
                p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
                pq_prop = row[1].replace('http://www.wikidata.org/prop/qualifier/', 'pq:')
                literal_str = row[2]
                for operator in COMPARISON_OPERATORS:
                    if compare_literal(literal_str, grounded_item_rep, operator):
                        for cmp_function in OPERATOR_FUNCTION[operator]:
                            results.append({
                                "s_expression": CMP(cmp_function, JOIN(p_prop, pq_prop), grounded_item_rep)
                            })
        elif grounded_item["type"].lower() == "class": # TODO
            return [{
                "s_expression": JOIN("wdt:P31/wdt:P279*", grounded_item_rep)
            }]
        else:
            raise Exception(f"type: {grounded_item['type']} value: {grounded_item['mid']}")
        
        return results

    def get_one_hop_relations(self, item):
        '''从 {item} 出发的一跳关系'''
        results = list()
        if not item:
            return results
        
        if item["type"].lower() in ["entity"]:
            grounded_item_rep = item['mid']
        else:
            '''RDF 规范中，literal 不能位于 subject 位置；对于 class 我们只查 P31/P279* class 的情况, 这里方向不对'''
            return results

        '''不含 CVT 的一跳: p+ps'''
        query_wo_qualifier = f"""
        SELECT DISTINCT ?p, ?ps where {{
            {grounded_item_rep} ?p [
                ?ps ?o 
            ] .
            FILTER (strstarts(str(?p), "http://www.wikidata.org/prop/P")).
            FILTER (strstarts(str(?ps), "http://www.wikidata.org/prop/statement/P")).
        }}
        """
        query_with_qualifier = f"""
        SELECT DISTINCT ?p, ?pq where {{
            {grounded_item_rep} ?p [
                ?pq ?o
            ] .
            FILTER (strstarts(str(?p), "http://www.wikidata.org/prop/P")).
            FILTER (strstarts(str(?pq), "http://www.wikidata.org/prop/qualifier/P")).
        }}
        """
        
        rows = self._execute_query(query_wo_qualifier)
        for row in rows:
            p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
            ps_prop = row[1].replace('http://www.wikidata.org/prop/statement/', 'ps:')
            results.append({
                "s_expression": JOIN(R(ps_prop), JOIN(R(p_prop), grounded_item_rep))
            })
        
        rows = self._execute_query(query_with_qualifier)
        for row in rows:
            p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
            pq_prop = row[1].replace('http://www.wikidata.org/prop/qualifier/', 'pq:')
            results.append({
                "s_expression": JOIN(R(pq_prop), JOIN(R(p_prop), grounded_item_rep))
            })
            
        return results
    
    def query_one_hop_paths_reversed(self, grounded_item, answer_entity=None, answer_type=None):
        """
        Paper: Sec 5.3 Exploration — reversed one-hop path query (Wikidata).
        Query reversed one-hop paths from grounded_item to answer_entity, returning
        S-expression patterns (grounded_item → answer_entity direction).

        反向一跳路径查询：从 grounded_item 出发到达 answer_entity，返回 S-expression 图模式。
        """
        results = list()
        if not grounded_item:
            return results
        if (not answer_entity) and (not answer_type):
            return results
        if answer_entity == grounded_item:
            return results
        if answer_type == grounded_item: # TODO: 是否仅比较 mid?
            return results

        if answer_entity is not None:
            if answer_entity["type"].lower() not in ["entity", "class", "literal"]:
                return results
            answer_entity_rep = answer_entity['mid']
            answer_type_rep = None
        elif answer_type is not None:
            if answer_type['type'].lower() not in ['entity', 'class']:
                raise NotImplementedError(f"answer_entity is None")
            answer_type_rep = answer_type['mid']
            answer_entity_rep = None
        grounded_item_rep = grounded_item['mid']

        if (grounded_item["type"].lower() == "entity"): # Literal 不应该出现在主语位置
            '''不含 CVT 的一跳: p+ps'''
            if answer_entity_rep is not None:
                query_wo_qualifier = f"""
                SELECT DISTINCT ?p, ?ps where {{
                    {grounded_item_rep} ?p [
                        ?ps {answer_entity_rep}
                    ] .
                    FILTER (strstarts(str(?p), "http://www.wikidata.org/prop/P")).
                    FILTER (strstarts(str(?ps), "http://www.wikidata.org/prop/statement/P")).
                }}
                """
                query_with_qualifier = f"""
                SELECT DISTINCT ?p, ?pq where {{
                    {grounded_item_rep} ?p [
                        ?pq {answer_entity_rep}
                    ] .
                    FILTER (strstarts(str(?p), "http://www.wikidata.org/prop/P")).
                    FILTER (strstarts(str(?pq), "http://www.wikidata.org/prop/qualifier/P")).
                }}
                """
            elif answer_type_rep is not None:
                query_wo_qualifier = f"""
                SELECT DISTINCT ?p, ?ps where {{
                    {grounded_item_rep} ?p [
                        ?ps ?s
                    ] .
                    ?s wdt:P31/wdt:P279* {answer_type_rep} .
                    FILTER (?s!={answer_type_rep} && ?s!={grounded_item_rep}) .
                    FILTER (strstarts(str(?p), "http://www.wikidata.org/prop/P")).
                    FILTER (strstarts(str(?ps), "http://www.wikidata.org/prop/statement/P")).
                }}
                """
                query_with_qualifier = f"""
                SELECT DISTINCT ?p, ?pq where {{
                    {grounded_item_rep} ?p [
                        ?pq ?s
                    ] .
                    ?s wdt:P31/wdt:P279* {answer_type_rep} .
                    FILTER (?s!={answer_type_rep} && ?s!={grounded_item_rep}) .
                    FILTER (strstarts(str(?p), "http://www.wikidata.org/prop/P")).
                    FILTER (strstarts(str(?pq), "http://www.wikidata.org/prop/qualifier/P")).
                }}
                """
            else:
                raise NotImplementedError()

            rows = self._execute_query(query_wo_qualifier)
            for row in rows:
                p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
                ps_prop = row[1].replace('http://www.wikidata.org/prop/statement/', 'ps:')
                results.append({
                    "s_expression": JOIN(R(ps_prop), JOIN(R(p_prop), grounded_item_rep)) 
                })
            
            rows = self._execute_query(query_with_qualifier)
            for row in rows:
                p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
                pq_prop = row[1].replace('http://www.wikidata.org/prop/qualifier/', 'pq:')
                results.append({
                    "s_expression": JOIN(R(pq_prop), JOIN(R(p_prop), grounded_item_rep)) 
                })
            
            
        elif WikidataConstantForConstruction.get_constant_type(grounded_item['mid']) in [WIKIDATA_CONSTANT_TYPE.TIME, WIKIDATA_CONSTANT_TYPE.QUANTITY, WIKIDATA_CONSTANT_TYPE.STRING]:
            '''RDF 数据格式中，Literal 不可能作为 subject'''
            pass
        elif grounded_item["type"].lower() == "class":
            '''class 不能作为主语'''
            pass
        else:
            raise Exception(f"type: {grounded_item['type']} value: {grounded_item['mid']}")
        
        return results
    
    def get_one_hop_relations_reversed(self, item):
        results = list()
        if not item:
            return results

        grounded_item_rep = item['mid']

        if item["type"].lower() == "entity" or WikidataConstantForConstruction.get_constant_type(item['mid']) is WIKIDATA_CONSTANT_TYPE.STRING:
            query_wo_qualifier = f"""
            SELECT DISTINCT ?p, ?ps where {{
                ?s ?p [
                    ?ps {grounded_item_rep}
                ] .
                FILTER (strstarts(str(?p), "http://www.wikidata.org/prop/P")).
                FILTER (strstarts(str(?ps), "http://www.wikidata.org/prop/statement/P")).
            }}
            """
            query_with_qualifier = f"""
            SELECT DISTINCT ?p, ?pq where {{
                ?s ?p [
                    ?pq {grounded_item_rep}
                ] .
                FILTER (strstarts(str(?p), "http://www.wikidata.org/prop/P")).
                FILTER (strstarts(str(?pq), "http://www.wikidata.org/prop/qualifier/P")).
            }}
            """
            rows = self._execute_query(query_wo_qualifier)
            for row in rows:
                p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
                ps_prop = row[1].replace('http://www.wikidata.org/prop/statement/', 'ps:')
                results.append({
                    "s_expression": JOIN(p_prop, JOIN(ps_prop, grounded_item_rep)) 
                })
            
            rows = self._execute_query(query_with_qualifier)
            for row in rows:
                p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
                pq_prop = row[1].replace('http://www.wikidata.org/prop/qualifier/', 'pq:')
                results.append({
                    "s_expression": JOIN(p_prop, JOIN(pq_prop, grounded_item_rep)) 
                })
        
        elif WikidataConstantForConstruction.get_constant_type(item['mid']) in [WIKIDATA_CONSTANT_TYPE.TIME, WIKIDATA_CONSTANT_TYPE.QUANTITY]:
            '''
            理论上对于 Literal 应该支持范围查询的
            实践中发现两端都不确定的范围查询，搜索空间太大了，肯定超时
            '''
            query_wo_qualifier = f"""
            SELECT DISTINCT ?p, ?ps where {{
                ?s ?p [
                    ?ps {grounded_item_rep}
                ] .
                FILTER (strstarts(str(?p), "http://www.wikidata.org/prop/P")).
                FILTER (strstarts(str(?ps), "http://www.wikidata.org/prop/statement/P")).
            }} 
            """
            query_with_qualifier = f"""
            SELECT DISTINCT ?p, ?pq where {{
                ?s ?p [
                    ?pq {grounded_item_rep}
                ] .
                FILTER (strstarts(str(?p), "http://www.wikidata.org/prop/P")).
                FILTER (strstarts(str(?pq), "http://www.wikidata.org/prop/qualifier/P")).
            }} 
            """
            rows = self._execute_query(query_wo_qualifier)
            for row in rows:
                p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
                ps_prop = row[1].replace('http://www.wikidata.org/prop/statement/', 'ps:')
                operator = '='
                for cmp_function in OPERATOR_FUNCTION[operator]:
                    results.append({
                        "s_expression": CMP(
                            cmp_function, JOIN(p_prop, ps_prop),
                            grounded_item_rep
                        )
                    })
            
            rows = self._execute_query(query_with_qualifier)
            for row in rows:
                p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
                pq_prop = row[1].replace('http://www.wikidata.org/prop/qualifier/', 'pq:')
                operator = '='
                for cmp_function in OPERATOR_FUNCTION[operator]:
                    results.append({
                        "s_expression": CMP(
                            cmp_function, JOIN(p_prop, pq_prop),
                            grounded_item_rep
                        )
                    })
        elif item['type'].lower() == 'class':
            return [{
                "s_expression": JOIN("wdt:P31/wdt:P279*", grounded_item_rep)
            }]
        else:
            raise Exception(f"type: {item['type']} value: {item['mid']}")
        
        return results

    def get_one_hop_arg_relation(self, answer_entity=None, answer_type=None):
        """
        Paper: Sec 5.3 — ARGMIN/ARGMAX relation enumeration (Wikidata).
        Enumerate one-hop relations from answer_entity (or answer_type) that connect
        to numeric/temporal values, for ARGMIN/ARGMAX query construction.

        枚举从 answer_entity/answer_type 到数值/时间值的一跳关系，用于 ARGMIN/ARGMAX 查询构造。
        """
        arg_results = list() # ARGMIN / ARGMAX candidate relations / ARGMIN/ARGMAX 候选关系
        if (not answer_entity) and (not answer_type):
            return arg_results
        
        if answer_entity is not None:
            if answer_entity["type"].lower() != "entity":
                return arg_results
            answer_entity_rep = answer_entity['mid']
            answer_type_rep = None
        elif answer_type is not None:
            if answer_type["type"].lower() not in ["entity", "class"]:
                raise NotImplementedError(f"answer_type:{answer_type}")
            answer_type_rep = answer_type['mid']
            answer_entity_rep = None
        
        if answer_entity_rep is not None:
            query_wo_qualifier = f"""
            SELECT DISTINCT ?p, ?ps where {{
                {answer_entity_rep} ?p [
                    ?ps ?v 
                ] .
                FILTER (strstarts(str(?p), "http://www.wikidata.org/prop/P")).
                FILTER (strstarts(str(?ps), "http://www.wikidata.org/prop/statement/P")).
                FILTER (isNumeric(?v) || (
                    datatype(?v)
                    IN (<http://www.w3.org/2001/XMLSchema#date>, <http://www.w3.org/2001/XMLSchema#dateTime>, <http://www.w3.org/2001/XMLSchema#gYear>, <http://www.w3.org/2001/XMLSchema#gYearMonth>)
                ))
            }}
            """
            query_with_qualifier =  f"""
            SELECT DISTINCT ?p, ?pq where {{
                {answer_entity_rep} ?p [
                    ?pq ?v
                ] .
                FILTER (strstarts(str(?p), "http://www.wikidata.org/prop/P")).
                FILTER (strstarts(str(?pq), "http://www.wikidata.org/prop/qualifier/P")).
                FILTER (isNumeric(?v) || (
                    datatype(?v)
                    IN (<http://www.w3.org/2001/XMLSchema#date>, <http://www.w3.org/2001/XMLSchema#dateTime>, <http://www.w3.org/2001/XMLSchema#gYear>, <http://www.w3.org/2001/XMLSchema#gYearMonth>)
                ))
            }}
            """
        elif answer_type_rep is not None:
            query_wo_qualifier = f"""
            SELECT DISTINCT ?p, ?ps where {{
                ?s ?p [
                    ?ps ?v 
                ] .
                ?s wdt:P31/wdt:P279* {answer_type_rep} .
                FILTER (?s!={answer_type_rep} && ?s!=?v) .
                FILTER (strstarts(str(?p), "http://www.wikidata.org/prop/P")).
                FILTER (strstarts(str(?ps), "http://www.wikidata.org/prop/statement/P")).
                FILTER (isNumeric(?v) || (
                    datatype(?v)
                    IN (<http://www.w3.org/2001/XMLSchema#date>, <http://www.w3.org/2001/XMLSchema#dateTime>, <http://www.w3.org/2001/XMLSchema#gYear>, <http://www.w3.org/2001/XMLSchema#gYearMonth>)
                ))
            }}
            """
            query_with_qualifier =  f"""
            SELECT DISTINCT ?p, ?pq where {{
                ?s ?p [
                    ?pq ?v
                ] .
                ?s wdt:P31/wdt:P279* {answer_type_rep} .
                FILTER (?s!={answer_type_rep} && ?s!=?v) .
                FILTER (strstarts(str(?p), "http://www.wikidata.org/prop/P")).
                FILTER (strstarts(str(?pq), "http://www.wikidata.org/prop/qualifier/P")).
                FILTER (isNumeric(?v) || (
                    datatype(?v)
                    IN (<http://www.w3.org/2001/XMLSchema#date>, <http://www.w3.org/2001/XMLSchema#dateTime>, <http://www.w3.org/2001/XMLSchema#gYear>, <http://www.w3.org/2001/XMLSchema#gYearMonth>)
                ))
            }}
            """
        else:
            raise NotImplementedError()
        
        rows = self._execute_query(query_wo_qualifier)
        for row in rows:
            p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
            ps_prop = row[1].replace('http://www.wikidata.org/prop/statement/', 'ps:')
            for function in ARG_FUNCTIONS:
                arg_results.append({
                    "function": function,
                    "relation": JOIN(p_prop, ps_prop)
                })
        
        rows = self._execute_query(query_with_qualifier)
        for row in rows:
            p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
            pq_prop = row[1].replace('http://www.wikidata.org/prop/qualifier/', 'pq:')
            for function in ARG_FUNCTIONS:
                arg_results.append({
                    "function": function,
                    "relation": JOIN(p_prop, pq_prop)
                })
        
        return arg_results
        
    def query_multivariate_atomic_queries(
        self, 
        grounded_item_0, 
        grounded_item_1, 
        p_prop,
        pq_prop,
        answer_entity=None,
        answer_type=None
    ):    
        results = list()
        if (not grounded_item_0) or (not grounded_item_1):
            return results
        if (not answer_entity) and (not answer_type):
            return results
        if (answer_entity == grounded_item_0) or (answer_entity == grounded_item_1):
            return results
        if (answer_type == grounded_item_0) or (answer_type == grounded_item_1):
            return results
        
        if answer_entity is not None:
            if answer_entity["type"].lower() not in ["entity", "class", "literal"]:
                raise NotImplementedError()
            answer_entity_rep = answer_entity['mid']
            answer_type_rep = None
        elif answer_type is not None:
            if answer_type["type"].lower() not in ["entity", "class"]:
                raise NotImplementedError(f"answer_type: {answer_type}")
            answer_type_rep = answer_type['mid']
            answer_entity_rep = None

        if grounded_item_0["type"].lower() not in ["entity", "class", "literal"]:
            raise NotImplementedError()
        if grounded_item_1["type"].lower() not in ["entity", "class", "literal"]:
            raise NotImplementedError()
        grounded_item_0_rep = grounded_item_0['mid']
        grounded_item_1_rep = grounded_item_1['mid']
        
        if answer_entity_rep is not None:
            for (constant_0, constant_1, constant_2) in itertools.permutations([answer_entity_rep, grounded_item_0_rep, grounded_item_1_rep], 3):
                query = f"""
                ASK {{
                    {constant_0} p:{p_prop} [
                        ps:{p_prop} {constant_1};
                        pq:{pq_prop} {constant_2}
                    ] .
                }}
                """
                rows = self._execute_query(query)
                if len(rows) > 0 and rows[0][0] == 1: # pattern 能够被实例化
                    if constant_0 == answer_entity_rep:
                        results.append({
                            "s_expression": JOIN(
                                f"p:{p_prop}", AND(JOIN(f"ps:{p_prop}", constant_1), JOIN(f"pq:{pq_prop}", constant_2))
                            )
                        })
                    elif constant_1 == answer_entity_rep:
                        results.append({
                            "s_expression": JOIN(
                                R(f"ps:{p_prop}"), AND(JOIN(R(f"p:{p_prop}"), constant_0), JOIN(f"pq:{pq_prop}", constant_2))
                            )
                        })
                    elif constant_2 == answer_entity_rep:
                        results.append({
                            "s_expression": JOIN(
                                R(f"pq:{pq_prop}"), AND(JOIN(R(f"p:{p_prop}"), constant_0), JOIN(f"ps:{p_prop}", constant_1))
                            )
                        }) 
                    '''构造过程中得到的 S-expression, 格式上都和 SPARQL 保持一致，这里的格式包括实体的前缀，literal 的后缀等'''
        elif answer_type_rep is not None:
            for (constant_0, constant_1) in itertools.permutations([grounded_item_0_rep, grounded_item_1_rep], 2):
                query = f"""
                ASK {{
                    ?s p:{p_prop} [
                        ps:{p_prop} {constant_0};
                        pq:{pq_prop} {constant_1}
                    ] .
                    ?s wdt:P31/wdt:P279* {answer_type_rep} .
                    FILTER (?s!={constant_0} && ?s!={constant_1}) .
                }}
                """
                rows = self._execute_query(query)
                if len(rows) > 0 and rows[0][0] == 1:
                    results.append({
                        "s_expression": JOIN(
                            f"p:{p_prop}", AND(JOIN(f"ps:{p_prop}", constant_0), JOIN(f"pq:{pq_prop}", constant_1))
                        )
                    })
                
                query = f"""
                ASK {{
                    {constant_0} p:{p_prop} [
                        ps:{p_prop} ?s;
                        pq:{pq_prop} {constant_1}
                    ] .
                    ?s wdt:P31/wdt:P279* {answer_type_rep} .
                    FILTER (?s!={constant_0} && ?s!={constant_1}) .
                }}
                """
                rows = self._execute_query(query)
                if len(rows) > 0 and rows[0][0] == 1:
                    results.append({
                        "s_expression": JOIN(
                            R(f"ps:{p_prop}"), AND(JOIN(R(f"p:{p_prop}"), constant_0), JOIN(f"pq:{pq_prop}", constant_1))
                        )
                    })
                
                query = f"""
                ASK {{
                    {constant_0} p:{p_prop} [
                        ps:{p_prop} {constant_1};
                        pq:{pq_prop} ?s
                    ] .
                    ?s wdt:P31/wdt:P279* {answer_type_rep} .
                    FILTER (?s!={constant_0} && ?s!={constant_1}) .
                }}
                """
                rows = self._execute_query(query)
                if len(rows) > 0 and rows[0][0] == 1:
                    results.append({
                        "s_expression": JOIN(
                            R(f"pq:{pq_prop}"), AND(JOIN(R(f"p:{p_prop}"), constant_0), JOIN(f"ps:{p_prop}", constant_1))
                        )
                    })
        else:
            raise NotImplementedError()
        
        return results

    def query_multivariate_relations(
        self, 
        grounded_item_0, 
        grounded_item_1, 
        p_prop,
        pq_prop,
    ):    
        """
        与 query_multivariate_atomic_queries() 的差别: 本函数只有 一端是常量，另一端并不固定
        """
        results = list()
        if (not grounded_item_0) or (not grounded_item_1):
            return results

        if grounded_item_0["type"].lower() not in ["entity", "class", "literal"]:
            raise NotImplementedError()
        if grounded_item_1["type"].lower() not in ["entity", "class", "literal"]:
            raise NotImplementedError()

        query = f"""
        ASK {{
            ?x p:{p_prop} [
                ps:{p_prop} {grounded_item_0['mid']};
                pq:{pq_prop} {grounded_item_1['mid']}
            ]
        }}
        """
        rows = self._execute_query(query)
        if len(rows) > 0 and rows[0][0] == 1:
            results.append({
                "s_expression": JOIN(
                    f"p:{p_prop}", AND(JOIN(f"ps:{p_prop}", grounded_item_0['mid']), JOIN(f"pq:{pq_prop}", grounded_item_1['mid']))
                )
            })
        
        query = f"""
        ASK {{
            ?x p:{p_prop} [
                ps:{p_prop} {grounded_item_1['mid']};
                pq:{pq_prop} {grounded_item_0['mid']}
            ]
        }}
        """
        rows = self._execute_query(query)
        if len(rows) > 0 and rows[0][0] == 1:
            results.append({
                "s_expression": JOIN(
                    f"p:{p_prop}", AND(JOIN(f"ps:{p_prop}", grounded_item_1['mid']), JOIN(f"pq:{pq_prop}", grounded_item_0['mid']))
                )
            })
        
        if grounded_item_0["type"].lower() != "literal":
            # literal 不能作为 subject
            query = f"""
            ASK {{
                {grounded_item_0['mid']} p:{p_prop} [
                    ps:{p_prop} ?x;
                    pq:{pq_prop} {grounded_item_1['mid']}
                ]
            }}
            """
            rows = self._execute_query(query)
            if len(rows) > 0 and rows[0][0] == 1:
                results.append({
                    "s_expression": JOIN(
                        R(f"ps:{p_prop}"), AND(JOIN(R(f"p:{p_prop}"), grounded_item_0['mid']), JOIN(f"pq:{pq_prop}", grounded_item_1['mid']))
                    )
                })
            
            query = f"""
            ASK {{
                {grounded_item_0['mid']} p:{p_prop} [
                    ps:{p_prop} {grounded_item_1['mid']};
                    pq:{pq_prop} ?x
                ]
            }}
            """
            rows = self._execute_query(query)
            if len(rows) > 0 and rows[0][0] == 1:
                results.append({
                    "s_expression": JOIN(
                        R(f"pq:{pq_prop}"), AND(JOIN(f"ps:{p_prop}", grounded_item_1['mid']), JOIN(R(f"p:{p_prop}"), grounded_item_0['mid']))
                    )
                })
        
        if grounded_item_1["type"].lower() != "literal":
            # literal 不能作为 subject
            query = f"""
            ASK {{
                {grounded_item_1['mid']} p:{p_prop} [
                    ps:{p_prop} ?x;
                    pq:{pq_prop} {grounded_item_0['mid']}
                ]
            }}
            """
            rows = self._execute_query(query)
            if len(rows) > 0 and rows[0][0] == 1:
                results.append({
                    "s_expression": JOIN(
                        R(f"ps:{p_prop}"), AND(JOIN(R(f"p:{p_prop}"), grounded_item_1['mid']), JOIN(f"pq:{pq_prop}", grounded_item_0['mid']))
                    )
                })
            
            query = f"""
            ASK {{
                {grounded_item_1['mid']} p:{p_prop} [
                    ps:{p_prop} {grounded_item_0['mid']};
                    pq:{pq_prop} ?x
                ]
            }}
            """
            rows = self._execute_query(query)
            if len(rows) > 0 and rows[0][0] == 1:
                results.append({
                    "s_expression": JOIN(
                        R(f"pq:{pq_prop}"), AND(JOIN(f"ps:{p_prop}", grounded_item_0['mid']), JOIN(R(f"p:{p_prop}"), grounded_item_1['mid']))
                    )
                })
        
        return results

    def check_class(self, item, type):
        if (item is None) or (type is None):
            return False
        if item['type'] not in ['entity']:
            return False
        if type['type'] not in ['entity', 'class']: # Wikidata 上，class 和 entity 其实是难以区分的
            return False
        item_rep = item['mid']
        type_rep = type['mid']

        query = f"""
        ASK {{
            {item_rep} wdt:P31/wdt:P279* {type_rep} .
        }}
        """
        rows = self._execute_query(query)
        return (len(rows) > 0 and rows[0][0] == 1)

    def get_reverse_property(self):
        query = """
        SELECT DISTINCT ?p1 ?p2 where {
            ?p1 wdt:P1696 ?p2 .
            FILTER (strstarts(str(?p1), "http://www.wikidata.org/entity/P")) .
            FILTER (strstarts(str(?p2), "http://www.wikidata.org/entity/P")) .
        } 
        """
        results = set()
        rows = self._execute_query(query)
        for row in rows:
            results.add((
                row[0].replace("http://www.wikidata.org/entity/", ""),
                row[1].replace("http://www.wikidata.org/entity/", "")
            ))
        return results

    def expand_next_hop_path_with_LF(self, LF:SimpleGraph, expand_point:Node, end_point = None, answer = None):
        '''
        Paper: Sec 5.3 Exploration — answer-anchored one-hop expansion (Wikidata).
        Given a logical form (LF), expand from `expand_point` by one hop to reach
        `end_point`, returning all relation paths `p1/p2` satisfying the condition.

        给定一个 LF，从 expand_point 出发向前一跳到达 end_point，返回所有满足条件的 p1/p2 关系路径。
        Wikidata CVT is essentially two-hop: each hop is p+ps, with optional pq for qualifiers.
        Four direction combinations are explored: nn (expand→mid→end), rn (expand←mid→end),
        rr (expand←mid←end), and their wdt: direct fallbacks.

        In CVT/Statement representation, p (property) connects entity→statement,
        ps (property statement) connects statement→value, pq (property qualifier) adds qualifiers.
        Uses timeout fallback: if nn query times out, falls back to wdt: direct properties.
        '''
        results = []
        #为key path中的属性加上前缀，同时处理取反
        if LF is not None:
            if answer is not None:
                sparql_gp = LF.get_sparql_gp_with_answer(answer)
            else:
                sparql_gp = LF.to_sparql_gp()
        else:
            sparql_gp = ""
        if LF is None and answer is not None:
                expand_point_rep = answer['mid']
        else:
            expand_point_rep = expand_point.value
        if end_point is None:
            end_point_rep = "?end"
        #elif (end_point["type"].lower() == "entity") or (FreebaseConstantForConstruction.get_constant_type(end_point['mid']) is FREEBASE_CONSTANT_TYPE.STRING):
        else:
            end_point_rep = end_point['mid']
        # else:
        #     end_point_rep = end_point['mid']
            #raise Exception("not implemented")
        if LF is not None:
            # Optimization: when expand_point has small cardinality, enumerate its VALUES instead of joining. // 优化：expand_point 度数小时直接穷举 VALUES
            temp_q = f"SELECT DISTINCT COUNT(*) AS ?cnt WHERE {{ {sparql_gp} }}"
            expand_point_num = int(list(self.get_execution_result_one_variable(temp_q))[0])
        if LF is not None and expand_point_num <= 3:
            temp_q = f"SELECT DISTINCT {expand_point_rep} WHERE {{ {sparql_gp}  }}"
            extend_point_values_rep = ""
            for v in self.get_execution_result_one_variable(temp_q) :
                if v.startswith("http://www.wikidata.org/entity/Q"):
                    v = v.replace("http://www.wikidata.org/entity/Q", "wd:Q")
                else:
                    v = '"' + v + '"'
                extend_point_values_rep += " " + v
            extend_point_values_rep = extend_point_values_rep.strip()
        else:
            extend_point_values_rep = None
        #需要用FILTER加以约束，否则必然超时
            #             FILTER (strstarts(str(?p), "http://www.wikidata.org/prop/P")).
            #             FILTER (strstarts(str(?ps), "http://www.wikidata.org/prop/statement/P")).
            #             FILTER (strstarts(str(?pq), "http://www.wikidata.org/prop/qualifier/P")).
        if extend_point_values_rep is not None:
            # expand -p1-> mid -p2-> end
            query_nn = f"""
                SELECT DISTINCT ?p1, ?p2 where {{
                    VALUES {expand_point_rep} {{ {extend_point_values_rep} }}.  
                    {expand_point_rep} ?p1 [ ?p2 {end_point_rep}] .
                    FILTER (strstarts(str(?p1), "http://www.wikidata.org/prop/P"))
                    FILTER (strstarts(str(?p2), "http://www.wikidata.org/prop/statement/P") || strstarts(str(?p2), "http://www.wikidata.org/prop/qualifier/P"))
                }}"""
            query_nn_direct = f"""
                SELECT DISTINCT ?p1 where {{
                    VALUES {expand_point_rep} {{ {extend_point_values_rep} }}.  
                    {expand_point_rep} ?p1 {end_point_rep} .
                    FILTER (strstarts(str(?p1), "http://www.wikidata.org/prop/direct/P"))
                }}"""
            # expand -p1-> mid <-p2 end
            #这种情况不可能：STATEMENT只有一个入度
            # query_nr = f"""
            #     SELECT DISTINCT ?p1, ?p2 where {{
            #         VALUES {expand_point_rep} {{ {extend_point_values_rep} }}.  
            #         {expand_point_rep} ?p1 _:mid. 
            #         {end_point_rep} ?p2 _:mid.
            #         FILTER (strstarts(str(?p1), "http://www.wikidata.org/prop/P"))
            #     }}"""
            # expand <-p1- mid -p2-> end
            query_rn = f"""
                SELECT DISTINCT ?p1, ?p2 where {{
                    VALUES {expand_point_rep} {{ {extend_point_values_rep} }}.  
                    {{
                        _:mid ?p1 {expand_point_rep}.
                        FILTER (strstarts(str(?p1), "http://www.wikidata.org/prop/statement/P") || strstarts(str(?p1), "http://www.wikidata.org/prop/qualifier/P"))
                    }}
                    {{
                        _:mid ?p2 {end_point_rep} .
                        FILTER (strstarts(str(?p2), "http://www.wikidata.org/prop/statement/P") || strstarts(str(?p2), "http://www.wikidata.org/prop/qualifier/P"))
                    }}
                    FILTER (?p1 != ?p2).
                }}"""
            # expand <-p1- mid <-p2- end
            query_rr = f"""
                SELECT DISTINCT ?p1, ?p2 where {{
                    VALUES {expand_point_rep} {{ {extend_point_values_rep} }}.  
                    {end_point_rep} ?p2 [ ?p1 {expand_point_rep} ].
                    FILTER (strstarts(str(?p2), "http://www.wikidata.org/prop/P"))
                    FILTER (strstarts(str(?p1), "http://www.wikidata.org/prop/statement/P") || strstarts(str(?p1), "http://www.wikidata.org/prop/qualifier/P"))
                }}"""
            query_rr_direct = f"""
                SELECT DISTINCT ?p1 where {{
                    VALUES {expand_point_rep} {{ {extend_point_values_rep} }}.  
                    {end_point_rep} ?p1 {expand_point_rep} .
                    FILTER (strstarts(str(?p1), "http://www.wikidata.org/prop/direct/P"))
                }}"""
        else:
            # expand -p1-> mid -p2-> end
            query_nn = f"""
                SELECT DISTINCT ?p1, ?p2 where {{
                    {sparql_gp} 
                    {expand_point_rep} ?p1 [ ?p2 {end_point_rep}] .
                    FILTER (strstarts(str(?p1), "http://www.wikidata.org/prop/P"))
                    FILTER (strstarts(str(?p2), "http://www.wikidata.org/prop/statement/P") || strstarts(str(?p2), "http://www.wikidata.org/prop/qualifier/P"))
                }}"""
            query_nn_direct = f"""
                SELECT DISTINCT ?p1 where {{
                    {sparql_gp} 
                    {expand_point_rep} ?p1 {end_point_rep} .
                    FILTER (strstarts(str(?p1), "http://www.wikidata.org/prop/direct/P"))
                }}"""
            # expand -p1-> mid <-p2 end
            #这种情况不可能：STATEMENT只有一个入度
            # query_nr = f"""
            #     SELECT DISTINCT ?p1, ?p2 where {{
            #         {sparql_gp} 
            #         {expand_point_rep} ?p1 _:mid. 
            #         {end_point_rep} ?p2 _:mid.
            #     }}"""
            # expand <-p1- mid -p2-> end
            query_rn = f"""
                SELECT DISTINCT ?p1, ?p2 where {{
                    {sparql_gp} 
                    {{
                        _:mid ?p1 {expand_point_rep}.
                        FILTER (strstarts(str(?p1), "http://www.wikidata.org/prop/statement/P") || strstarts(str(?p1), "http://www.wikidata.org/prop/qualifier/P"))
                    }}
                    {{
                        _:mid ?p2 {end_point_rep} .
                        FILTER (strstarts(str(?p2), "http://www.wikidata.org/prop/statement/P") || strstarts(str(?p2), "http://www.wikidata.org/prop/qualifier/P"))
                    }}
                    FILTER (?p1 != ?p2).
                }}"""
            # expand <-p1- mid <-p2- end
            query_rr = f"""
                SELECT DISTINCT ?p1, ?p2 where {{ 
                    {sparql_gp} 
                    {end_point_rep} ?p2 [ ?p1 {expand_point_rep} ].
                    FILTER (strstarts(str(?p2), "http://www.wikidata.org/prop/P"))
                    FILTER (strstarts(str(?p1), "http://www.wikidata.org/prop/statement/P") || strstarts(str(?p1), "http://www.wikidata.org/prop/qualifier/P"))
                }}"""
            query_rr_direct = f"""
                SELECT DISTINCT ?p1 where {{
                    {sparql_gp} 
                    {end_point_rep} ?p1 {expand_point_rep} .
                    FILTER (strstarts(str(?p1), "http://www.wikidata.org/prop/direct/P"))
                }}"""
        #===========================================================================================================
        """                   
        PREFIX p: <http://www.wikidata.org/prop/>
        PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
        PREFIX ps: <http://www.wikidata.org/prop/statement/>
        """
        uri2prefix = {"http://www.wikidata.org/prop/P":"p:P", "http://www.wikidata.org/prop/qualifier/P":"pq:P", 
                      "http://www.wikidata.org/prop/statement/P":"ps:P", "http://www.wikidata.org/prop/direct/P":"wdt:P"}
        #-----------------------nn -----------------------------------------------------------------------------------
        #对于nn，非常容易超时；如果超时（结果为空，且超过3秒），使用direct来补救
        #nn_result = self.execute_query_with_odbc_with_err(query_nn)
        if self.cached_results is not None and self.format_sparql(query_nn) in self.cached_results:
            nn_results = self.cached_results[self.format_sparql(query_nn)]
        else:
            nn_results = []
            start_time = time.time()
            sparql_result_nn = self.execute_query_with_err(query_nn)
            if len(sparql_result_nn) > 0 and sparql_result_nn[0] != 'ERROR':
                for rows in sparql_result_nn:
                    #p1, p2 = rows[0], rows[1]
                    p1, p2 = rows['p1']['value'], rows['p2']['value']
                    for uri, prefix in uri2prefix.items():
                        p1 = p1.replace(uri, prefix)
                        p2 = p2.replace(uri, prefix)
                    nn_results.append(p1+"/"+p2)
            else:
                sparql_result_nn = self.execute_query_with_err(query_nn_direct)
                if len(sparql_result_nn) > 0 and sparql_result_nn[0] != "ERROR":
                    for rows in sparql_result_nn:
                        #p1 = rows[0]
                        p1= rows['p1']['value']
                        for uri, prefix in uri2prefix.items():
                            p1 = p1.replace(uri, prefix)
                        #将wdt:改写为p:/ps:
                        p1 = p1.split(":")[1]
                        nn_results.append(f"p:{p1}/ps:{p1}")
            if self.cached_results is not None and self.format_sparql(query_nn) not in self.cached_results:
                if time.time() - start_time >= 0.2:
                    self.update_cache_results(self.format_sparql(query_nn), nn_results, 100)
        results += nn_results
        #---------------------rn-----------------------------------------------------------------------------------------
        #rn_result = self._execute_query(query_rn)
        if self.cached_results is not None and self.format_sparql(query_rn) in self.cached_results:
            rn_results = self.cached_results[self.format_sparql(query_rn)]
        else:       
            rn_results = []
            start_time = time.time()
            sparql_result_rn = self.execute_query_with_err(query_rn)
            if len(sparql_result_rn) >= 1 and sparql_result_rn[0] != "ERROR":
                for rows in sparql_result_rn:
                    #p1, p2 = rows[0], rows[1]
                    p1, p2 = rows['p1']['value'], rows['p2']['value']
                    for uri, prefix in uri2prefix.items():
                        p1 = p1.replace(uri, prefix)
                        p2 = p2.replace(uri, prefix)
                    rn_results.append(f"^{p1}/{p2}")      
            if self.cached_results is not None and self.format_sparql(query_rn) not in self.cached_results:
                if time.time() - start_time >= 0.2:
                    self.update_cache_results(self.format_sparql(query_rn), rn_results, 100)
        results += rn_results      
        #--------------------------rr------------------------------------------------------------------------------------
        #与nn同理
        #rr_result = self.execute_query_with_odbc_with_err(query_rr)
        if self.cached_results is not None and self.format_sparql(query_rr) in self.cached_results:
            rr_results = self.cached_results[self.format_sparql(query_rr)]
        else:
            rr_results = []
            start_time = time.time()
            sparql_result_rr = self.execute_query_with_err(query_rr)
            if len(sparql_result_rr) > 0 and sparql_result_rr[0] != "ERROR":
                for rows in sparql_result_rr:
                    #p1, p2 = rows[0], rows[1]
                    p1, p2 = rows['p1']['value'], rows['p2']['value']
                    for uri, prefix in uri2prefix.items():
                        p1 = p1.replace(uri, prefix)
                        p2 = p2.replace(uri, prefix)
                    rr_results.append(f"^{p1}/^{p2}")
            else:
                sparql_result_rr = self.execute_query_with_err(query_rr_direct)
                if len(sparql_result_rr) > 0 and sparql_result_rr[0] != "ERROR":
                    for rows in sparql_result_rr:
                        p1 = rows['p1']['value']
                        for uri, prefix in uri2prefix.items():
                            p1 = p1.replace(uri, prefix)
                        #将wdt:改写为p:/ps:
                        p1 = p1.split(":")[1]
                        rr_results.append(f"^ps:{p1}/^p:{p1}")
            if self.cached_results is not None and self.format_sparql(query_rr) not in self.cached_results:
                if time.time() - start_time >= 0.2:
                    self.update_cache_results(self.format_sparql(query_rr), rr_results, 100)
        results += rr_results      
        #===============================================================================================================================
        return results