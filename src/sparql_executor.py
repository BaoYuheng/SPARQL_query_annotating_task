import pyodbc
import itertools
from SPARQLWrapper import SPARQLWrapper, JSON
from .s_expression_utils import (
    JOIN, CMP, R, AND, sexp_to_sparql
)
from .utils import (
    convert_number,
    compare_literal
)
from .concurrent_executor import ConcurrentExecutor
from .simple_graph import SimpleGraph

from .common import (
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
from src.utils import load_json, dump_json
import math
from collections import defaultdict
from .simple_graph import SimpleGraph, Node, NodeType
from .semantic_sim_utils import PLMSimRanker
import time
import os


class SparqlOdbcQuerierWikidata(ConcurrentExecutor):
    def __init__(self, odbc_config, sparql_wrapper_path, logger, timeout=10):
        super().__init__(logger)
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
    
    def connect(self):
        if self.connection is None:
            connection = pyodbc.connect(
                self.odbc_config
            )
            connection.setdecoding(pyodbc.SQL_CHAR, encoding='utf8')
            connection.setdecoding(pyodbc.SQL_WCHAR, encoding='utf8')
            connection.setencoding(encoding='utf8')
            connection.timeout = self.timeout
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
    
    def execute_query(self, query, retry=4):
        for idx in range(retry): # 有时候出现 502 错误，难以定位原因，就重试几次吧
            if idx > 0:
                self.logger.info(f"execute_query(); idx:{idx}")
            try:
                complete_query = f"{self.SPARQL_PREFIX} {query}"
                self.sparql_wrapper.setQuery(complete_query)
                results = self.sparql_wrapper.query().convert()
                return results['results']['bindings'] # ASK 类型语句会报错，但是本身我们也处理不了 ASK
            except Exception as err:
                self.logger.error(f"Query Execution Failed: {query}, error: {str(err)}")
        return []
    
    
    def get_execution_result_one_variable(self, query):
        rows = self.execute_query_with_odbc(query)
        results = set()
        for row in rows:
            results.add(row[0])
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
            query_result = self.execute_query_with_odbc(query)
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
        rows = self.execute_query_with_odbc(query)
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
        rows = self.execute_query_with_odbc(query)
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
        rows = self.execute_query_with_odbc(query)
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
        rows = self.execute_query_with_odbc(query)
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
        rows = self.execute_query_with_odbc(query1)
        freebase_mid = post_process_mid(rows[0][0]) if len(rows) >= 1 else "" # 返回空串，处理起来一致
        
        if not freebase_mid:
            '''Google KG id, 同样出现在 freebase 中，g.123'''
            query2 = f"""
            SELECT ?o WHERE {{
                wd:{wikidata_mid} wdt:P2671 ?o .
            }}
            """
            rows = self.execute_query_with_odbc(query2)
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
        rows = self.execute_query_with_odbc(query1)
        wikidata_mid = rows[0][0].replace('http://www.wikidata.org/entity/', '') if len(rows) >= 1 else None
        
        if not wikidata_mid:
            '''Google KG id, 同样出现在 freebase 中，g.123'''
            query2 = f"""
            SELECT ?o WHERE {{
                ?s wdt:P2671 {processed_freebase_mid}.
            }}
            """
            rows = self.execute_query_with_odbc(query2)
            wikidata_mid = rows[0][0].replace('http://www.wikidata.org/entity/', '') if len(rows) >= 1 else None # 返回空串，处理起来一致
        
        return {
            "freebase_mid": processed_freebase_mid,
            "wikidata_mid": wikidata_mid
        }
    
    # def query_constant_connected_cvt_pattern(self, constant_0, constant_1, constant_2):
    #     constant_0 = ConstantSerializer.get_wikidata_constant(constant_0)
    #     constant_1 = ConstantSerializer.get_wikidata_constant(constant_1)
    #     constant_2 = ConstantSerializer.get_wikidata_constant(constant_2)
    #     results = list()

    #     if (constant_0 is None) or (constant_1 is None) or (constant_2 is None):
    #         return {
    #             "constant_list": (constant_0, constant_1, constant_2),
    #             "pattern_list": results
    #         }
        
    #     for (subj, obj_ps, obj_pq) in itertools.permutations([constant_0, constant_1, constant_2], 3):
    #         query = f"""
    #         SELECT DISTINCT ?p, ?ps, ?pq {{
    #             {subj} ?p [
    #                 ?ps {obj_ps};
    #                 ?pq {obj_pq}
    #             ] .
    #             FILTER (strstarts(str(?p), "http://www.wikidata.org/prop/P")).
    #             FILTER (strstarts(str(?ps), "http://www.wikidata.org/prop/statement/P")).
    #             FILTER (strstarts(str(?pq), "http://www.wikidata.org/prop/qualifier/P")).
    #         }}
    #         """
            
    #         rows = self.execute_query_with_odbc(query)
    #         for row in rows:
    #             results.append((
    #                 row[0].replace('http://www.wikidata.org/prop/', 'p:'),
    #                 row[1].replace('http://www.wikidata.org/prop/statement/', 'ps:'),
    #                 row[2].replace('http://www.wikidata.org/prop/qualifier/', 'pq:')
    #             ))
        
    #     return {
    #         "constant_list": (constant_0, constant_1, constant_2),
    #         "pattern_list": results
    #     }

    # def query_constant_connected_cvt_pattern(self, constant_0, constant_1, constant_2):
    #     constant_0_sparql_rep = ConstantSerializer.get_wikidata_constant(constant_0)
    #     constant_1_sparql_rep = ConstantSerializer.get_wikidata_constant(constant_1)
    #     constant_2_sparql_rep = ConstantSerializer.get_wikidata_constant(constant_2)
    #     results = list()

    #     if (constant_0 is None) or (constant_1 is None) or (constant_2 is None):
    #         return {
    #             "constant_list": (constant_0, constant_1, constant_2),
    #             "pattern_list": results
    #         }

    #     if ConstantSerializer.get_constant_type(constant_0) is WIKIDATA_CONSTANT_TYPE.ENTITY:
    #         # 按照 RDF 规范，Literal 不能出现在主语位置
    #         query = f"""
    #         SELECT DISTINCT ?p1, ?p2, ?p3 WHERE {{
    #             {constant_0_sparql_rep} ?p1 ?statement .
    #             ?statement ?p2 {constant_1_sparql_rep} .
    #             ?statement ?p3 {constant_2_sparql_rep} .
    #             FILTER (strstarts(str(?p1), "http://www.wikidata.org/prop/P")).
    #             FILTER (strstarts(str(?p2), "http://www.wikidata.org/prop/")).
    #             FILTER (strstarts(str(?p3), "http://www.wikidata.org/prop/")).
    #             FILTER (strstarts(str(?statement), "http://www.wikidata.org/entity/statement/")).
    #             FILTER (?p1 != ?p2 && ?p1 != ?p3 && ?p2 != ?p3 ) .
    #         }}
    #         """
    #         rows = self.execute_query_with_odbc(query)
    #         for row in rows:
    #             results.append((
    #                 constant_0, row[0].replace('http://www.wikidata.org/prop/', 'p:'),
    #                 row[1].replace('http://www.wikidata.org/prop/statement/', 'ps:').replace('http://www.wikidata.org/prop/qualifier/', 'pq:'), constant_1,
    #                 row[2].replace('http://www.wikidata.org/prop/statement/', 'ps:').replace('http://www.wikidata.org/prop/qualifier/', 'pq:'), constant_2
    #             ))
        
    #     if ConstantSerializer.get_constant_type(constant_1) is WIKIDATA_CONSTANT_TYPE.ENTITY:
    #         query = f"""
    #         SELECT DISTINCT ?p1, ?p2, ?p3 WHERE {{
    #             {constant_1_sparql_rep} ?p1 ?statement .
    #             ?statement ?p2 {constant_0_sparql_rep} .
    #             ?statement ?p3 {constant_2_sparql_rep} .
    #             FILTER (strstarts(str(?p1), "http://www.wikidata.org/prop/P")).
    #             FILTER (strstarts(str(?p2), "http://www.wikidata.org/prop/")).
    #             FILTER (strstarts(str(?p3), "http://www.wikidata.org/prop/")).
    #             FILTER (strstarts(str(?statement), "http://www.wikidata.org/entity/statement/")).
    #             FILTER (?p1 != ?p2 && ?p1 != ?p3 && ?p2 != ?p3 ) .
    #         }}
    #         """
    #         rows = self.execute_query_with_odbc(query)
    #         for row in rows:
    #             results.append((
    #                 constant_1, row[0].replace('http://www.wikidata.org/prop/', 'p:'),
    #                 row[1].replace('http://www.wikidata.org/prop/statement/', 'ps:').replace('http://www.wikidata.org/prop/qualifier/', 'pq:'), constant_0,
    #                 row[2].replace('http://www.wikidata.org/prop/statement/', 'ps:').replace('http://www.wikidata.org/prop/qualifier/', 'pq:'), constant_2
    #             ))
        
    #     if ConstantSerializer.get_constant_type(constant_2) is WIKIDATA_CONSTANT_TYPE.ENTITY:
    #         query = f"""
    #         SELECT DISTINCT ?p1, ?p2, ?p3 WHERE {{
    #             {constant_2_sparql_rep} ?p1 ?statement .
    #             ?statement ?p2 {constant_0_sparql_rep} .
    #             ?statement ?p3 {constant_1_sparql_rep} .
    #             FILTER (strstarts(str(?p1), "http://www.wikidata.org/prop/P")).
    #             FILTER (strstarts(str(?p2), "http://www.wikidata.org/prop/")).
    #             FILTER (strstarts(str(?p3), "http://www.wikidata.org/prop/")).
    #             FILTER (strstarts(str(?statement), "http://www.wikidata.org/entity/statement/")).
    #             FILTER (?p1 != ?p2 && ?p1 != ?p3 && ?p2 != ?p3 ) .
    #         }}
    #         """
    #         rows = self.execute_query_with_odbc(query)
    #         for row in rows:
    #             results.append((
    #                 constant_2, row[0].replace('http://www.wikidata.org/prop/', 'p:'),
    #                 row[1].replace('http://www.wikidata.org/prop/statement/', 'ps:').replace('http://www.wikidata.org/prop/qualifier/', 'pq:'), constant_0,
    #                 row[2].replace('http://www.wikidata.org/prop/statement/', 'ps:').replace('http://www.wikidata.org/prop/qualifier/', 'pq:'), constant_1
    #             ))

        
    #     return {
    #         "constant_list": (constant_0, constant_1, constant_2),
    #         "pattern_list": results
    #     }


    # def instantiate_multivariate_graph_pattern(self, prop_p, prop_ps, prop_pq, limit=500):
    #     """
    #     图模式在 KB 上实例化得到的事实
    #     """
    #     query = f"""
    #     SELECT DISTINCT ?s, ?o_ps, ?o_pq {{
    #         ?s p:{prop_p} [
    #         ps:{prop_ps} ?o_ps;
    #         pq:{prop_pq} ?o_pq
    #         ] .
    #     }} LIMIT {limit}
    #     """
    #     rows = self.execute_query(query)
    #     results = set()
    #     for row in rows:
    #         s_binding = row['s']
    #         s_serialized = ConstantSerializer(s_binding['type'], s_binding['value'], s_binding.get('datatype', None), s_binding.get('xml:lang', None))
    #         ops_binding = row['o_ps']
    #         ops_serialized = ConstantSerializer(ops_binding['type'], ops_binding['value'], ops_binding.get('datatype', None), ops_binding.get('xml:lang', None))
    #         opq_binding = row['o_pq']
    #         opq_serialized = ConstantSerializer(opq_binding['type'], opq_binding['value'], opq_binding.get('datatype', None), opq_binding.get('xml:lang', None))
    #         results.add((
    #             s_serialized.__repr__(), ops_serialized.__repr__(), opq_serialized.__repr__()
    #         ))
            
    #     return {
    #         "prop_p": prop_p,
    #         "prop_ps": prop_ps,
    #         "prop_pq": prop_pq,
    #         "results": list(results)
    #     }

    def query_one_hop_paths(self, grounded_item, answer_entity=None, answer_type=None):
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

            rows = self.execute_query_with_odbc(query_wo_qualifier)
            for row in rows:
                p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
                ps_prop = row[1].replace('http://www.wikidata.org/prop/statement/', 'ps:')
                results.append({
                    "s_expression": JOIN(p_prop, JOIN(ps_prop, grounded_item_rep)) 
                })
            
            rows = self.execute_query_with_odbc(query_with_qualifier)
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
            
            rows = self.execute_query_with_odbc(query_wo_qualifier)
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

            rows = self.execute_query_with_odbc(query_with_qualifier)
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
        
        rows = self.execute_query_with_odbc(query_wo_qualifier)
        for row in rows:
            p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
            ps_prop = row[1].replace('http://www.wikidata.org/prop/statement/', 'ps:')
            results.append({
                "s_expression": JOIN(R(ps_prop), JOIN(R(p_prop), grounded_item_rep))
            })
        
        rows = self.execute_query_with_odbc(query_with_qualifier)
        for row in rows:
            p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
            pq_prop = row[1].replace('http://www.wikidata.org/prop/qualifier/', 'pq:')
            results.append({
                "s_expression": JOIN(R(pq_prop), JOIN(R(p_prop), grounded_item_rep))
            })
            
        return results
    
    def query_one_hop_paths_reversed(self, grounded_item, answer_entity=None, answer_type=None):
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

            rows = self.execute_query_with_odbc(query_wo_qualifier)
            for row in rows:
                p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
                ps_prop = row[1].replace('http://www.wikidata.org/prop/statement/', 'ps:')
                results.append({
                    "s_expression": JOIN(R(ps_prop), JOIN(R(p_prop), grounded_item_rep)) 
                })
            
            rows = self.execute_query_with_odbc(query_with_qualifier)
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
            rows = self.execute_query_with_odbc(query_wo_qualifier)
            for row in rows:
                p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
                ps_prop = row[1].replace('http://www.wikidata.org/prop/statement/', 'ps:')
                results.append({
                    "s_expression": JOIN(p_prop, JOIN(ps_prop, grounded_item_rep)) 
                })
            
            rows = self.execute_query_with_odbc(query_with_qualifier)
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
            rows = self.execute_query_with_odbc(query_wo_qualifier)
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
            
            rows = self.execute_query_with_odbc(query_with_qualifier)
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
        arg_results = list() # ARGMIN / ARGMAX 相关结果
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
        
        rows = self.execute_query_with_odbc(query_wo_qualifier)
        for row in rows:
            p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
            ps_prop = row[1].replace('http://www.wikidata.org/prop/statement/', 'ps:')
            for function in ARG_FUNCTIONS:
                arg_results.append({
                    "function": function,
                    "relation": JOIN(p_prop, ps_prop)
                })
        
        rows = self.execute_query_with_odbc(query_with_qualifier)
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
                rows = self.execute_query_with_odbc(query)
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
                rows = self.execute_query_with_odbc(query)
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
                rows = self.execute_query_with_odbc(query)
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
                rows = self.execute_query_with_odbc(query)
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
        rows = self.execute_query_with_odbc(query)
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
        rows = self.execute_query_with_odbc(query)
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
            rows = self.execute_query_with_odbc(query)
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
            rows = self.execute_query_with_odbc(query)
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
            rows = self.execute_query_with_odbc(query)
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
            rows = self.execute_query_with_odbc(query)
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
        rows = self.execute_query_with_odbc(query)
        return (len(rows) > 0 and rows[0][0] == 1)

    # def check_all_fixed_statement_num(self, prop_p, prop_ps, prop_pq, s, o_ps, o_pq):
    #     """
    #     CVT / Statement 相关的三个常量都固定，获得此时 statement 节点的数量信息
    #     """
    #     from components.wikipedia_utils import ConstantSerializer
    #     s = ConstantSerializer.get_wikidata_constant(s)
    #     o_ps = ConstantSerializer.get_wikidata_constant(o_ps)
    #     o_pq = ConstantSerializer.get_wikidata_constant(o_pq)
    #     query = f"""
    #     SELECT COUNT(DISTINCT ?statement) {{
    #         {s} p:{prop_p} ?statement .
    #         ?statement ps:{prop_ps} {o_ps} .
    #         ?statement pq:{prop_pq} {o_pq} .
    #         FILTER (strstarts(str(?statement), "http://www.wikidata.org/entity/statement/")).
    #     }}
    #     """
    #     rows = self.execute_query_with_odbc(query)
    #     if len(rows) != 1:
    #         results = 0
    #     else:
    #         results = rows[0][0]
    #     return {
    #         "prop_p": prop_p,
    #         "prop_ps": prop_ps,
    #         "prop_pq": prop_pq,
    #         "s": s,
    #         "o_ps": o_ps,
    #         "o_pq": o_pq,
    #         "count": results
    #     }

    # def check_s_ops_fixed_statement_num(self, prop_p, prop_ps, prop_pq, s, o_ps):
    #     """
    #     CVT / Statement 相关s, o_ps 固定，获得此时 statement 节点的数量信息
    #     """
    #     s = ConstantSerializer.get_wikidata_constant(s)
    #     o_ps = ConstantSerializer.get_wikidata_constant(o_ps)
    #     query = f"""
    #     SELECT COUNT(DISTINCT ?statement) {{
    #         {s} p:{prop_p} ?statement .
    #         ?statement ps:{prop_ps} {o_ps} .
    #         FILTER (strstarts(str(?statement), "http://www.wikidata.org/entity/statement/")).
    #     }}
    #     """
    #     rows = self.execute_query_with_odbc(query)
    #     if len(rows) != 1:
    #         results = 0
    #     else:
    #         results = rows[0][0]
    #     return {
    #         "prop_p": prop_p,
    #         "prop_ps": prop_ps,
    #         "prop_pq": prop_pq,
    #         "s": s,
    #         "o_ps": o_ps,
    #         "count": results
    #     }
    
    # def check_s_opq_fixed_statement_num(self, prop_p, prop_ps, prop_pq, s, o_pq):
    #     """
    #     CVT / Statement 相关s, o_ps 固定，获得此时 statement 节点的数量信息
    #     """
    #     s = ConstantSerializer.get_wikidata_constant(s)
    #     o_pq = ConstantSerializer.get_wikidata_constant(o_pq)
    #     query = f"""
    #     SELECT COUNT(DISTINCT ?statement) {{
    #         {s} p:{prop_p} ?statement .
    #         ?statement pq:{prop_pq} {o_pq} .
    #         FILTER (strstarts(str(?statement), "http://www.wikidata.org/entity/statement/")).
    #     }}
    #     """
    #     rows = self.execute_query_with_odbc(query)
    #     if len(rows) != 1:
    #         results = 0
    #     else:
    #         results = rows[0][0]
    #     return {
    #         "prop_p": prop_p,
    #         "prop_ps": prop_ps,
    #         "prop_pq": prop_pq,
    #         "s": s,
    #         "o_pq": o_pq,
    #         "count": results
    #     }

    # def get_reverse_property(self):
    #     query = """
    #     SELECT DISTINCT ?p1 ?p2 where {
    #         ?p1 wdt:P1696 ?p2 .
    #         FILTER (strstarts(str(?p1), "http://www.wikidata.org/entity/P")) .
    #         FILTER (strstarts(str(?p2), "http://www.wikidata.org/entity/P")) .
    #     } 
    #     """
    #     results = set()
    #     rows = self.execute_query_with_odbc(query)
    #     for row in rows:
    #         results.add((
    #             row[0].replace("http://www.wikidata.org/entity/", ""),
    #             row[1].replace("http://www.wikidata.org/entity/", "")
    #         ))
    #     return results


class SparqlOdbcQuerier(ConcurrentExecutor):
    def __init__(self, odbc_config, sparql_wrapper_path, logger, timeout=5):
        super().__init__(logger)
        self.odbc_config = odbc_config
        self.sparql_wrapper_path = sparql_wrapper_path
        self.timeout = timeout
        self.ODBC_PREFIX = "SPARQL PREFIX ns: <http://rdf.freebase.com/ns/> "
        self.SPARQL_PREFIX = "PREFIX ns: <http://rdf.freebase.com/ns/> "
        self.sparql_wrapper = SPARQLWrapper(self.sparql_wrapper_path)
        self.sparql_wrapper.setReturnFormat(JSON)
        self.sparql_wrapper.setTimeout(self.timeout)
    
    def get_ignored_relations_filter(self, variable):
        filter_list = [
            f"!regex({variable}, \"^{NS_PREFIX}{domain}\")"
            for domain in IGNORED_DOMAINS + CLASS_RELATED_DOMAINS
        ]
        return f"""
        FILTER ({" && ".join(filter_list)})
        """
    
    def connect(self):
        connection = pyodbc.connect(
            self.odbc_config
        )
        connection.setdecoding(pyodbc.SQL_CHAR, encoding='utf8')
        connection.setdecoding(pyodbc.SQL_WCHAR, encoding='utf8')
        connection.setencoding(encoding='utf8')
        connection.timeout = self.timeout
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

    def execute_query(self, query, retry=4):
        for idx in range(retry): # 有时候出现 502 错误，难以定位原因，就重试几次吧
            if idx > 0:
                self.logger.info(f"Retrying execute_query(); idx:{idx}")
            try:
                query = f"{self.SPARQL_PREFIX} {query}"
                self.sparql_wrapper.setQuery(query)
                results = self.sparql_wrapper.query().convert()
                return results['results']['bindings']
            except Exception as err:
                self.logger.error(f"Query Execution Failed: {query}, error: {str(err)}")
        return []
    
    def get_execution_result_one_variable(self, query):
        rows = self.execute_query_with_odbc(query)
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
        rows = self.execute_query_with_odbc(query)
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
        rows = self.execute_query_with_odbc(query)
        rtn = set()
        for row in rows:
            rtn.add(row[0].replace('http://rdf.freebase.com/ns/', ''))
        return rtn

    def get_execution_result_one_variable_sparql_wrapper(self, query, retry=4):
        """
        答案类型是 Literal 的特殊实现
        会把 Literal 的类型拼接起来
        注意，调用此函数时，查询目标应该只有一个
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
        rows = self.execute_query_with_odbc(query)
        types = set()
        for row in rows:
            types.add(row[0].replace('http://rdf.freebase.com/ns/', ''))
        return list(types)  
    
    def query_one_hop_paths(self, grounded_item, answer_entity=None, answer_type=None):
        '''
        answer_entity 和 grounded_item 是一个相对的概念
        我们是从 answer_entity 的角度，写对应的 S-expression

        @param grounded_item: {"type": , "mid":}
        @param answer_entity: {"type": , "mid":}
        @param answer_type: {"type": , "mid":}
        规定: 本函数中，离 answer_entity 或者 answer_type 最近的关系记作 r1, 依次类推
        '''
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
            if answer_entity["type"].lower() in ["entity", "class"]:
                answer_entity_rep = f"ns:{answer_entity['mid']}"
            elif answer_entity["type"].lower() == "literal":
                '''RDF 规范中，literal 不能位于 subject 位置'''
                return results
            else:
                raise NotImplementedError()
            answer_type_rep = None
        elif answer_type is not None:
            if answer_type["type"].lower() == "class":
                answer_type_rep = f"ns:{answer_type['mid']}"
            else:
                raise NotImplementedError()
            answer_entity_rep = None
        
        if (grounded_item["type"].lower() == "entity") or (FreebaseConstantForConstruction.get_constant_type(grounded_item['mid']) is FREEBASE_CONSTANT_TYPE.STRING):
            if grounded_item["type"].lower() == "entity":
                grounded_item_rep = f"ns:{grounded_item['mid']}"
            else:
                grounded_item_rep = grounded_item['mid']
            # 不含 CVT 节点的一跳路径
            if answer_entity_rep is not None:
                query_wo_cvt = f"""
                SELECT DISTINCT ?x where {{
                    {answer_entity_rep} ?x {grounded_item_rep} .
                    FILTER regex(?x, "^http://rdf.freebase.com/ns/") .
                    {self.get_ignored_relations_filter("?x")} .
                }}
                """
            elif answer_type_rep is not None:
                query_wo_cvt = f"""
                SELECT DISTINCT ?x where {{
                    ?s ?x {grounded_item_rep} .
                    ?s ns:type.object.type {answer_type_rep} .
                    FILTER (?s != {grounded_item_rep} && ?s != {answer_type_rep})
                    FILTER regex(?x, "^http://rdf.freebase.com/ns/") .
                    {self.get_ignored_relations_filter("?x")} .
                }}
                """
            else:
                raise NotImplementedError()
            rows = self.execute_query_with_odbc(query_wo_cvt)
            for row in rows:
                relation = row[0].replace('http://rdf.freebase.com/ns/', '')
                results.append({
                    "s_expression": JOIN(relation, grounded_item['mid'])
                }) # answer_type 只是查询锚点，S-expression 到 answer_entity 即可
            
            if answer_entity_rep is not None:
                query_with_cvt = f"""
                SELECT DISTINCT ?r1, ?r2 where {{
                    {answer_entity_rep} ?r1 ?cvt .
                    ?cvt ?r2 {grounded_item_rep} .
                    FILTER regex(?r1, "^http://rdf.freebase.com/ns/") .
                    FILTER regex(?r2, "^http://rdf.freebase.com/ns/") .
                    FILTER (?cvt!={answer_entity_rep} && ?cvt!={grounded_item_rep}) . 
                    FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
                    {self.get_ignored_relations_filter("?r1")} .
                    {self.get_ignored_relations_filter("?r2")} .
                }}
                """
            elif answer_type_rep is not None:
                query_with_cvt = f"""
                SELECT DISTINCT ?r1, ?r2 where {{
                    ?s ?r1 ?cvt .
                    ?cvt ?r2 {grounded_item_rep} .
                    ?s ns:type.object.type {answer_type_rep} .
                    FILTER regex(?r1, "^http://rdf.freebase.com/ns/") .
                    FILTER regex(?r2, "^http://rdf.freebase.com/ns/") .
                    FILTER (?s != ?cvt && ?s != {grounded_item_rep} && ?s != {answer_type_rep}) .
                    FILTER (?cvt != {grounded_item_rep} && ?cvt != {answer_type_rep}) .
                    FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
                    {self.get_ignored_relations_filter("?r1")} .
                    {self.get_ignored_relations_filter("?r2")} .
                }}
                """
            else:
                raise NotImplementedError()
            
            rows = self.execute_query_with_odbc(query_with_cvt)
            for row in rows:
                r1 = row[0].replace('http://rdf.freebase.com/ns/', '')
                r2 = row[1].replace('http://rdf.freebase.com/ns/', '')
                results.append({
                    "s_expression": JOIN(
                        r1,
                        JOIN(r2, grounded_item['mid'])
                    )
                })

        elif FreebaseConstantForConstruction.get_constant_type(grounded_item['mid']) in [FREEBASE_CONSTANT_TYPE.TIME, FREEBASE_CONSTANT_TYPE.QUANTITY]:
            grounded_item_rep = grounded_item['mid']
            if answer_entity_rep is not None:
                query_wo_cvt = f"""
                SELECT DISTINCT ?x, ?v where {{
                    {answer_entity_rep} ?x ?v .
                    FILTER regex(?x, "^http://rdf.freebase.com/ns/") .
                    {self.get_ignored_relations_filter("?x")} .
                    FILTER (isNumeric(?v) || (
                        datatype(?v)
                        IN (<http://www.w3.org/2001/XMLSchema#date>, <http://www.w3.org/2001/XMLSchema#dateTime>, <http://www.w3.org/2001/XMLSchema#gYear>, <http://www.w3.org/2001/XMLSchema#gYearMonth>)
                    )) .
                }} 
                """
            elif answer_type_rep is not None:
                query_wo_cvt = f"""
                SELECT DISTINCT ?x, ?v where {{
                    ?s ?x ?v .
                    ?s ns:type.object.type {answer_type_rep} .
                    FILTER(?s != ?v && ?s != {answer_type_rep}) .
                    FILTER regex(?x, "^http://rdf.freebase.com/ns/") .
                    {self.get_ignored_relations_filter("?x")} .
                    FILTER (isNumeric(?v) || (
                        datatype(?v)
                        IN (<http://www.w3.org/2001/XMLSchema#date>, <http://www.w3.org/2001/XMLSchema#dateTime>, <http://www.w3.org/2001/XMLSchema#gYear>, <http://www.w3.org/2001/XMLSchema#gYearMonth>)
                    )) .
                }} 
                """
            else:
                raise NotImplementedError()
            rows = self.execute_query_with_odbc(query_wo_cvt)
            for row in rows:
                relation = row[0].replace('http://rdf.freebase.com/ns/', '')
                literal_str = row[1]
                
                for operator in COMPARISON_OPERATORS:
                    if compare_literal(literal_str, grounded_item_rep, operator):
                        for cmp_function in OPERATOR_FUNCTION[operator]:
                            results.append({
                                "s_expression": CMP(cmp_function, relation, grounded_item_rep)
                            })
            
            if answer_entity_rep is not None:
                query_with_cvt = f"""
                SELECT DISTINCT ?r1, ?r2, ?v where {{
                    {answer_entity_rep} ?r1 ?cvt .
                    ?cvt ?r2 ?v .
                    FILTER (?cvt!={answer_entity_rep} && ?cvt!=?v) .
                    FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}.
                    FILTER regex(?r1, "^http://rdf.freebase.com/ns/") .
                    FILTER regex(?r2, "^http://rdf.freebase.com/ns/") .
                    {self.get_ignored_relations_filter("?r1")} .
                    {self.get_ignored_relations_filter("?r2")} .
                    FILTER (isNumeric(?v) || (
                        datatype(?v)
                        IN (<http://www.w3.org/2001/XMLSchema#date>, <http://www.w3.org/2001/XMLSchema#dateTime>, <http://www.w3.org/2001/XMLSchema#gYear>, <http://www.w3.org/2001/XMLSchema#gYearMonth>)
                    )) .
                }}
                """
            elif answer_type_rep is not None:
                query_with_cvt = f"""
                SELECT DISTINCT ?r1, ?r2, ?v where {{
                    ?s ?r1 ?cvt .
                    ?cvt ?r2 ?v .
                    ?s ns:type.object.type {answer_type_rep} .
                    FILTER (?s != ?cvt && ?s != ?v && ?s != {answer_type_rep}) .
                    FILTER (?cvt != ?v && ?cvt != {answer_type_rep}) .
                    FILTER (?v != {answer_type_rep}) .
                    FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}.
                    FILTER regex(?r1, "^http://rdf.freebase.com/ns/") .
                    FILTER regex(?r2, "^http://rdf.freebase.com/ns/") .
                    {self.get_ignored_relations_filter("?r1")} .
                    {self.get_ignored_relations_filter("?r2")} .
                    FILTER (isNumeric(?v) || (
                        datatype(?v)
                        IN (<http://www.w3.org/2001/XMLSchema#date>, <http://www.w3.org/2001/XMLSchema#dateTime>, <http://www.w3.org/2001/XMLSchema#gYear>, <http://www.w3.org/2001/XMLSchema#gYearMonth>)
                    )) .
                }}
                """
            else:
                raise NotImplementedError()
            rows = self.execute_query_with_odbc(query_with_cvt)
            for row in rows:
                r1 = row[0].replace('http://rdf.freebase.com/ns/', '')
                r2 = row[1].replace('http://rdf.freebase.com/ns/', '')
                literal_str = row[2]
                for operator in COMPARISON_OPERATORS:
                    if compare_literal(literal_str, grounded_item_rep, operator):
                        for cmp_function in OPERATOR_FUNCTION[operator]:
                            results.append({
                                "s_expression": CMP(
                                    cmp_function, 
                                    JOIN(r1, r2), 
                                    grounded_item_rep
                                )
                            })
        elif grounded_item["type"].lower() == "class":
            return [{
                "s_expression": JOIN("type.object.type", grounded_item['mid'])
            }]
        else:
            raise Exception(f"type: {grounded_item['type']}, value: {grounded_item['mid']}")
        
        return results
    
    def query_one_hop_paths_reversed(self, grounded_item, answer_entity=None, answer_type=None):
        """
        @param answer_entity: {"type": , "mid":}
        @param grounded_item: {"type": , "mid":}
        @param answer_type: {"type": , "mid":}
        规定: 本函数中，离 answer_entity 或者 answer_type 最近的关系记作 r1, 依次类推
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
            if answer_entity["type"].lower() in ["entity", "class"]:
                answer_entity_rep = f"ns:{answer_entity['mid']}"
            elif answer_entity["type"].lower() == "literal":
                '''本函数中答案出现在 object 的位置，因此可以是 literal'''
                answer_entity_rep = answer_entity['mid']
            else:
                raise NotImplementedError()
            answer_type_rep = None
        elif answer_type is not None:
            if answer_type["type"].lower() == "class":
                answer_type_rep = f"ns:{answer_type['mid']}"
            else:
                raise NotImplementedError()
            answer_entity_rep = None
        
        if (grounded_item["type"].lower() == 'entity'):
            if grounded_item["type"].lower() == 'entity':
                grounded_item_rep = f"ns:{grounded_item['mid']}"
            else:
                grounded_item_rep = grounded_item['mid']
            # 不含 CVT 节点的一跳路径
            if answer_entity_rep is not None:
                query_wo_cvt = f"""
                SELECT DISTINCT ?x where {{
                    {grounded_item_rep} ?x {answer_entity_rep} .
                    FILTER regex(?x, "^http://rdf.freebase.com/ns/") .
                    {self.get_ignored_relations_filter("?x")} .
                }}
                """
            elif answer_type_rep is not None:
                query_wo_cvt = f"""
                SELECT DISTINCT ?x where {{
                    {grounded_item_rep} ?x ?s .
                    ?s ns:type.object.type {answer_type_rep} .
                    FILTER(?s!={grounded_item_rep} && ?s != {answer_type_rep})
                    FILTER regex(?x, "^http://rdf.freebase.com/ns/") .
                    {self.get_ignored_relations_filter("?x")} .
                }}
                """
            else:
                raise NotImplementedError()
            rows = self.execute_query_with_odbc(query_wo_cvt)
            for row in rows:
                relation = row[0].replace('http://rdf.freebase.com/ns/', '')
                results.append({
                    "s_expression": JOIN(R(relation), grounded_item['mid'])
                })
            
            if answer_entity_rep is not None:
                query_with_cvt = f"""
                SELECT DISTINCT ?r1, ?r2 where {{
                    {grounded_item_rep} ?r2 ?cvt .
                    ?cvt ?r1 {answer_entity_rep} .
                    FILTER regex(?r1, "^http://rdf.freebase.com/ns/") .
                    FILTER regex(?r2, "^http://rdf.freebase.com/ns/") .
                    FILTER (?cvt!={answer_entity_rep} && ?cvt!={grounded_item_rep}) . 
                    FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
                    {self.get_ignored_relations_filter("?r1")} .
                    {self.get_ignored_relations_filter("?r2")} .
                }}
                """
            elif answer_type_rep is not None:
                query_with_cvt = f"""
                SELECT DISTINCT ?r1, ?r2 where {{
                    {grounded_item_rep} ?r2 ?cvt .
                    ?cvt ?r1 ?s .
                    ?s ns:type.object.type {answer_type_rep} .
                    FILTER regex(?r1, "^http://rdf.freebase.com/ns/") .
                    FILTER regex(?r2, "^http://rdf.freebase.com/ns/") .
                    FILTER (?cvt!=?s && ?cvt!={grounded_item_rep} && ?cvt!={answer_type_rep}) . 
                    FILTER (?s != {grounded_item_rep} && ?s != {answer_type_rep})
                    FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
                    {self.get_ignored_relations_filter("?r1")} .
                    {self.get_ignored_relations_filter("?r2")} .
                }}
                """
            else:
                raise NotImplementedError()

            rows = self.execute_query_with_odbc(query_with_cvt)
            for row in rows:
                r1 = row[0].replace('http://rdf.freebase.com/ns/', '')
                r2 = row[1].replace('http://rdf.freebase.com/ns/', '')
                results.append({
                    "s_expression": JOIN(
                        R(r1),
                        JOIN(R(r2), grounded_item['mid'])
                    )
                })


        elif grounded_item["type"].lower() == 'class':
            '''对于 class 只能枚举关系 type.object.type'''
            pass
        elif FreebaseConstantForConstruction.get_constant_type(grounded_item['mid']) in [FREEBASE_CONSTANT_TYPE.TIME, FREEBASE_CONSTANT_TYPE.QUANTITY, FREEBASE_CONSTANT_TYPE.STRING]:
            '''RDF 数据格式中，Literal 不可能作为 subject'''
            pass
        else:
            raise Exception(f"type: {grounded_item['type']}, value: {grounded_item['mid']}")   
        
        return results

    def get_one_hop_arg_relation(self, answer_entity=None, answer_type=None):
        """
        @param answer_entity: {"type": , "mid":}
        @param answer_type: {"type": , "mid":} 先试一下
        得到 ARGMAX 和 ARGMIN 相关子表达式
        规定: 离答案最近的是 r1, 依次类推

        这个函数没有对应的 _reversed() 函数，因为 literal 不能出现在 subject 的位置
        """
        arg_results = list() # ARGMIN / ARGMAX 相关结果
        if (not answer_entity) and (not answer_type):
            return arg_results
        
        if answer_entity is not None:
            if answer_entity["type"].lower() == "entity":
                answer_entity_rep = f"ns:{answer_entity['mid']}"
            elif answer_entity["type"].lower() == "class":
                '''不支持 class 的 arg relation'''
                return arg_results
            elif answer_entity["type"].lower() == "literal":
                '''Literal 不能作为 subject'''
                return arg_results
            else:
                raise NotImplementedError()
            answer_type_rep = None
        elif answer_type is not None:
            if answer_type["type"].lower() == "class":
                answer_type_rep = f"ns:{answer_type['mid']}"
            else:
                raise NotImplementedError()
            answer_entity_rep = None
        
        # 假设对于同一个关系有多个取值，这个查询会对关系去重
        if answer_entity_rep is not None:
            query_wo_cvt = f"""
            SELECT DISTINCT ?x where {{
                {answer_entity_rep} ?x ?v .
                FILTER regex(?x, "^http://rdf.freebase.com/ns/") .
                {self.get_ignored_relations_filter("?x")} .
                FILTER (isNumeric(?v) || (
                    datatype(?v)
                    IN (<http://www.w3.org/2001/XMLSchema#date>, <http://www.w3.org/2001/XMLSchema#dateTime>, <http://www.w3.org/2001/XMLSchema#gYear>, <http://www.w3.org/2001/XMLSchema#gYearMonth>)
                ))
            }}
            """
        elif answer_type_rep is not None:
            query_wo_cvt = f"""
            SELECT DISTINCT ?x where {{
                ?s ?x ?v .
                ?s ns:type.object.type {answer_type_rep} .
                FILTER (?s != ?v && ?s != {answer_type_rep} && ?v != {answer_type_rep}) .
                FILTER regex(?x, "^http://rdf.freebase.com/ns/") .
                {self.get_ignored_relations_filter("?x")} .
                FILTER (isNumeric(?v) || (
                    datatype(?v)
                    IN (<http://www.w3.org/2001/XMLSchema#date>, <http://www.w3.org/2001/XMLSchema#dateTime>, <http://www.w3.org/2001/XMLSchema#gYear>, <http://www.w3.org/2001/XMLSchema#gYearMonth>)
                ))
            }}
            """
        else:
            raise NotImplementedError()
        rows = self.execute_query_with_odbc(query_wo_cvt)
        for row in rows:
            relation = row[0].replace('http://rdf.freebase.com/ns/', '')
            for function in ARG_FUNCTIONS:
                arg_results.append({
                    "function": function,
                    "relation": relation
                })
        
        if answer_entity_rep is not None:
            query_with_cvt = f"""
            SELECT DISTINCT ?r1, ?r2 where {{
                {answer_entity_rep} ?r1 ?cvt .
                ?cvt ?r2 ?v .
                FILTER regex(?r1, "^http://rdf.freebase.com/ns/") .
                FILTER regex(?r2, "^http://rdf.freebase.com/ns/") .
                {self.get_ignored_relations_filter("?r1")} .
                {self.get_ignored_relations_filter("?r2")} .
                FILTER (?cvt!={answer_entity_rep} && ?cvt!=?v) .
                FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}.
                FILTER (isNumeric(?v) || (
                    datatype(?v)
                    IN (<http://www.w3.org/2001/XMLSchema#date>, <http://www.w3.org/2001/XMLSchema#dateTime>, <http://www.w3.org/2001/XMLSchema#gYear>, <http://www.w3.org/2001/XMLSchema#gYearMonth>)
                ))
            }}
            """
        elif answer_type_rep is not None:
            query_with_cvt = f"""
            SELECT DISTINCT ?r1, ?r2 where {{
                ?s ?r1 ?cvt .
                ?cvt ?r2 ?v .
                ?s ns:type.object.type {answer_type_rep} .
                FILTER regex(?r1, "^http://rdf.freebase.com/ns/") .
                FILTER regex(?r2, "^http://rdf.freebase.com/ns/") .
                {self.get_ignored_relations_filter("?r1")} .
                {self.get_ignored_relations_filter("?r2")} .
                FILTER (?s != ?cvt && ?s != ?v && ?s != {answer_type_rep}) .
                FILTER (?cvt != ?v && ?cvt != {answer_type_rep}) .
                FILTER (?v != {answer_type_rep}) .
                FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}.
                FILTER (isNumeric(?v) || (
                    datatype(?v)
                    IN (<http://www.w3.org/2001/XMLSchema#date>, <http://www.w3.org/2001/XMLSchema#dateTime>, <http://www.w3.org/2001/XMLSchema#gYear>, <http://www.w3.org/2001/XMLSchema#gYearMonth>)
                ))
            }}
            """
        else:
            raise NotImplementedError()
        rows = self.execute_query_with_odbc(query_with_cvt)
        for row in rows:
            r1 = row[0].replace('http://rdf.freebase.com/ns/', '')
            r2 = row[1].replace('http://rdf.freebase.com/ns/', '')
            for function in ARG_FUNCTIONS:
                arg_results.append({
                    "function": function,
                    "relation": JOIN(r1, r2)
                })
        
        return arg_results

    def query_multivariate_atomic_queries(
        self, 
        grounded_item_0, 
        grounded_item_1, 
        pattern_relation_list,
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
            if answer_entity["type"].lower() in ["entity", "class"]:
                answer_entity_rep = f"ns:{answer_entity['mid']}"
            elif answer_entity["type"].lower() == "literal":
                answer_entity_rep = answer_entity['mid']
            else:
                raise NotImplementedError()
            answer_type_rep = None
        elif answer_type is not None:
            if answer_type["type"].lower() == "class":
                answer_type_rep = f"ns:{answer_type['mid']}"
            else:
                raise NotImplementedError()
            answer_entity_rep = None
        
        if grounded_item_0["type"].lower() in ["entity", "class"]:
            grounded_item_0_rep = f"ns:{grounded_item_0['mid']}"
        elif grounded_item_0["type"].lower() == "literal":
            grounded_item_0_rep = grounded_item_0['mid']
        else:
            raise NotImplementedError()

        if grounded_item_1["type"].lower() in ["entity", "class"]:
            grounded_item_1_rep = f"ns:{grounded_item_1['mid']}"
        elif grounded_item_1["type"].lower() == "literal":
            grounded_item_1_rep = grounded_item_1['mid']
        else:
            raise NotImplementedError()

        if answer_entity_rep is not None:
            for (relation_0, relation_1, relation_2) in itertools.permutations(pattern_relation_list, len(pattern_relation_list)):
                query = f"""
                ASK {{
                    ?cvt ns:{relation_0} {answer_entity_rep} .
                    ?cvt ns:{relation_1} {grounded_item_0_rep} .
                    ?cvt ns:{relation_2} {grounded_item_1_rep} .
                    FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
                }}
                """
                rows = self.execute_query_with_odbc(query)
                if len(rows) > 0 and rows[0][0] == 1: # pattern 能够被实例化
                    results.append({
                        "s_expression": JOIN(
                            R(relation_0), AND(JOIN(relation_1, grounded_item_0['mid']), JOIN(relation_2, grounded_item_1['mid']))
                        )
                    })
        elif answer_type_rep is not None:
            for (relation_0, relation_1, relation_2) in itertools.permutations(pattern_relation_list, len(pattern_relation_list)):
                query = f"""
                ASK {{
                    ?cvt ns:{relation_0} ?s .
                    ?s ns:type.object.type {answer_type_rep} .
                    ?cvt ns:{relation_1} {grounded_item_0_rep} .
                    ?cvt ns:{relation_2} {grounded_item_1_rep} .
                    FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
                    FILTER (?s != ?cvt && ?s != {answer_type_rep}) .
                }}
                """
                rows = self.execute_query_with_odbc(query)
                if len(rows) > 0 and rows[0][0] == 1: # pattern 能够被实例化
                    results.append({
                        "s_expression": JOIN(
                            R(relation_0), AND(JOIN(relation_1, grounded_item_0['mid']), JOIN(relation_2, grounded_item_1['mid']))
                        )
                    })
        
        return results

    def query_multivariate_relations(
        self, 
        grounded_item_0, 
        grounded_item_1, 
        pattern_relation_list,
    ):
        """
        与 query_multivariate_atomic_queries() 的差别: 本函数只有 grounded_item 一端是常量，另一端并不固定
        """
        results = list()
        if (not grounded_item_0) or (not grounded_item_1):
            return results
        
        if grounded_item_0["type"].lower() in ["entity", "class"]:
            grounded_item_0_rep = f"ns:{grounded_item_0['mid']}"
        elif grounded_item_0["type"].lower() == "literal":
            grounded_item_0_rep = grounded_item_0['mid']
        else:
            raise NotImplementedError()

        if grounded_item_1["type"].lower() in ["entity", "class"]:
            grounded_item_1_rep = f"ns:{grounded_item_1['mid']}"
        elif grounded_item_1["type"].lower() == "literal":
            grounded_item_1_rep = grounded_item_1['mid']
        else:
            raise NotImplementedError()
        if len(pattern_relation_list) != 3:
            raise Exception(f"pattern_relation_list: {len(pattern_relation_list)} {pattern_relation_list}")
        relation_0, relation_1, relation_2 = pattern_relation_list[0], pattern_relation_list[1], pattern_relation_list[2]

        query = f"""
        ASK {{
            ?cvt ns:{relation_0} ?x .
            ?cvt ns:{relation_1} {grounded_item_0_rep} .
            ?cvt ns:{relation_2} {grounded_item_1_rep} .
            FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
        }}
        """
        rows = self.execute_query_with_odbc(query)
        if len(rows) > 0 and rows[0][0] == 1:
            results.append({
                "s_expression": JOIN(
                    R(relation_0), AND(JOIN(relation_1, grounded_item_0['mid']), JOIN(relation_2, grounded_item_1['mid']))
                )
            })
        
        query = f"""
        ASK {{
            ?cvt ns:{relation_0} ?x .
            ?cvt ns:{relation_1} {grounded_item_1_rep} .
            ?cvt ns:{relation_2} {grounded_item_0_rep} .
            FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
        }}
        """
        rows = self.execute_query_with_odbc(query)
        if len(rows) > 0 and rows[0][0] == 1:
            results.append({
                "s_expression": JOIN(
                    R(relation_0), AND(JOIN(relation_1, grounded_item_1['mid']), JOIN(relation_2, grounded_item_0['mid']))
                )
            })
        
        query = f"""
        ASK {{
            ?cvt ns:{relation_0} {grounded_item_0_rep} .
            ?cvt ns:{relation_1} ?x .
            ?cvt ns:{relation_2} {grounded_item_1_rep} .
            FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
        }}
        """
        rows = self.execute_query_with_odbc(query)
        if len(rows) > 0 and rows[0][0] == 1:
            results.append({
                "s_expression": JOIN(
                    R(relation_1), AND(JOIN(relation_0, grounded_item_0['mid']), JOIN(relation_2, grounded_item_1['mid']))
                )
            })
        
        query = f"""
        ASK {{
            ?cvt ns:{relation_0} {grounded_item_1_rep} .
            ?cvt ns:{relation_1} ?x .
            ?cvt ns:{relation_2} {grounded_item_0_rep} .
            FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
        }}
        """
        rows = self.execute_query_with_odbc(query)
        if len(rows) > 0 and rows[0][0] == 1:
            results.append({
                "s_expression": JOIN(
                    R(relation_1), AND(JOIN(relation_0, grounded_item_1['mid']), JOIN(relation_2, grounded_item_0['mid']))
                )
            })
        
        query = f"""
        ASK {{
            ?cvt ns:{relation_0} {grounded_item_0_rep} .
            ?cvt ns:{relation_1} {grounded_item_1_rep} .
            ?cvt ns:{relation_2} ?x .
            FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
        }}
        """
        rows = self.execute_query_with_odbc(query)
        if len(rows) > 0 and rows[0][0] == 1:
            results.append({
                "s_expression": JOIN(
                    R(relation_2), AND(JOIN(relation_0, grounded_item_0['mid']), JOIN(relation_1, grounded_item_1['mid']))
                )
            })
        
        query = f"""
        ASK {{
            ?cvt ns:{relation_0} {grounded_item_1_rep} .
            ?cvt ns:{relation_1} {grounded_item_0_rep} .
            ?cvt ns:{relation_2} ?x .
            FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
        }}
        """
        rows = self.execute_query_with_odbc(query)
        if len(rows) > 0 and rows[0][0] == 1:
            results.append({
                "s_expression": JOIN(
                    R(relation_2), AND(JOIN(relation_0, grounded_item_1['mid']), JOIN(relation_1, grounded_item_0['mid']))
                )
            })

        return results


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
        rows = self.execute_query_with_odbc(query)
        name = rows[0][0] if len(rows) >= 1 else "" # 返回空串，处理起来一致

        if not name:
            query2 = f"""
            SELECT DISTINCT ?x WHERE {{
                {kb_item_rep} ns:common.topic.alias ?x .
                FILTER (langMatches( lang(?x), "EN" ) )
            }} LIMIT 1
            """
            rows = self.execute_query_with_odbc(query2)
            name = rows[0][0] if len(rows) >= 1 else ""
        
        return name
    
    def get_one_hop_relations(self, item=None):
        '''
        从 item 出发的一跳关系
        @param item: {"type": , "mid":}
        @param item_type: {"type": "class" , "mid":}
        感觉从 item_type 出发查询关系，很容易超时；但是最好还是实现一下代码，有没有实现还是差挺多的
        规定: 本函数中，离 item 最近的关系记作 r1, 依次类推
        @return: List of tuple
        '''
        results = list()
        if not item:
            return results

        if item is not None:
            if item["type"].lower() in ["entity", "class"]:
                item_rep = f"ns:{item['mid']}"
            elif item["type"].lower() == "literal":
                '''RDF 规范中，literal 不能位于 subject 位置'''
                return results
            else:
                raise NotImplementedError()
        
        if item_rep is not None:
            if item["type"].lower() == "entity":
                query_wo_cvt = f"""
                SELECT DISTINCT ?x where {{
                    {item_rep} ?x ?o .
                    ?o ns:type.object.type ns:common.topic . # Literal / Entity, Class 不管
                    FILTER ({item_rep} != ?o) .
                    FILTER regex(?x, "^http://rdf.freebase.com/ns/") .
                    {self.get_ignored_relations_filter("?x")} .
                }}
                """
                rows = self.execute_query_with_odbc(query_wo_cvt)
                for row in rows:
                    relation = row[0].replace('http://rdf.freebase.com/ns/', '')
                    results.append({
                        "s_expression": JOIN(R(relation), item['mid'])
                    })
                
                query_with_cvt = f"""
                SELECT DISTINCT ?r1, ?r2 where {{
                    {item_rep} ?r1 ?cvt .
                    ?cvt ?r2 ?o .
                    ?o ns:type.object.type ns:common.topic .
                    FILTER regex(?r1, "^http://rdf.freebase.com/ns/") .
                    FILTER regex(?r2, "^http://rdf.freebase.com/ns/") .
                    FILTER (?cvt!={item_rep} && ?cvt!=?o && {item_rep} != ?o) . 
                    FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
                    {self.get_ignored_relations_filter("?r1")} .
                    {self.get_ignored_relations_filter("?r2")} .
                }}
                """
                rows = self.execute_query_with_odbc(query_with_cvt)
                for row in rows:
                    r1 = row[0].replace('http://rdf.freebase.com/ns/', '')
                    r2 = row[1].replace('http://rdf.freebase.com/ns/', '')
                    results.append({
                        "s_expression": JOIN(R(r2), JOIN(R(r1), item['mid']))
                    })
            elif FreebaseConstantForConstruction.get_constant_type(item['mid']) in [FREEBASE_CONSTANT_TYPE.TIME, FREEBASE_CONSTANT_TYPE.QUANTITY, FREEBASE_CONSTANT_TYPE.STRING]:
                pass # Literal 不能出现在 subject 位置
            elif item["type"].lower() == "class":
                pass # 对于 class, 仅枚举 JOIN type.object.type {item}, 方向不符
            else:
                raise Exception(f"item: {item}")

        return results
    
    def get_one_hop_relations_reversed(self, item=None):
        '''
        从 item 出发的一跳关系
        @param item: {"type": , "mid":}
        @param item_type: {"type": "class" , "mid":}
        感觉从 item_type 出发查询关系，很容易超时；但是最好还是实现一下代码，有没有实现还是差挺多的
        规定: 本函数中，离 item 最近的关系记作 r1, 依次类推
        @return: List of tuple
        TODO: 注意在这个函数的返回结果中，对于关系并没有带上 R(), 从而方便外层调用
        '''
        results = list()
        if not item:
            return results

        if item is not None:
            if item["type"].lower() in ["entity", "class"]:
                item_rep = f"ns:{item['mid']}"
            elif item["type"].lower() == "literal":
                item_rep = item['mid']
            else:
                raise NotImplementedError()
        
        if item_rep is not None:
            if item["type"].lower() == "entity" or (FreebaseConstantForConstruction.get_constant_type(item["mid"]) is FREEBASE_CONSTANT_TYPE.STRING):
                query_wo_cvt = f"""
                SELECT DISTINCT ?x where {{
                    ?o ?x {item_rep} .
                    ?o ns:type.object.type ns:common.topic . # Literal / Entity, Class 不管
                    FILTER ({item_rep} != ?o) .
                    FILTER regex(?x, "^http://rdf.freebase.com/ns/") .
                    {self.get_ignored_relations_filter("?x")} .
                }}
                """
                rows = self.execute_query_with_odbc(query_wo_cvt)
                for row in rows:
                    relation = row[0].replace('http://rdf.freebase.com/ns/', '')
                    results.append({
                        "s_expression": JOIN(relation, item['mid'])
                    })
                
                query_with_cvt = f"""
                SELECT DISTINCT ?r1, ?r2 where {{
                    ?o ?r2 ?cvt .
                    ?cvt ?r1 {item_rep} .
                    ?o ns:type.object.type ns:common.topic .
                    FILTER regex(?r1, "^http://rdf.freebase.com/ns/") .
                    FILTER regex(?r2, "^http://rdf.freebase.com/ns/") .
                    FILTER (?cvt!={item_rep} && ?cvt!=?o && {item_rep} != ?o) . 
                    FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
                    {self.get_ignored_relations_filter("?r1")} .
                    {self.get_ignored_relations_filter("?r2")} .
                }}
                """
                rows = self.execute_query_with_odbc(query_with_cvt)
                for row in rows:
                    r1 = row[0].replace('http://rdf.freebase.com/ns/', '')
                    r2 = row[1].replace('http://rdf.freebase.com/ns/', '')
                    results.append({
                        "s_expression": JOIN(r2, JOIN(r1, item['mid']))
                    })
            
            elif FreebaseConstantForConstruction.get_constant_type(item['mid']) in [FREEBASE_CONSTANT_TYPE.TIME, FREEBASE_CONSTANT_TYPE.QUANTITY]:
                '''
                理论上对于 Literal 应该支持范围查询的
                实践中发现两端都不确定的范围查询，搜索空间太大了，肯定超时
                '''
                query_wo_cvt = f"""
                SELECT DISTINCT ?x where {{
                    ?o ?x {item_rep} .
                    ?o ns:type.object.type ns:common.topic . # Literal / Entity, Class 不管
                    FILTER ({item_rep} != ?o) .
                    FILTER regex(?x, "^http://rdf.freebase.com/ns/") .
                    {self.get_ignored_relations_filter("?x")} .
                }} 
                """
                rows = self.execute_query_with_odbc(query_wo_cvt)
                for row in rows:
                    relation = row[0].replace('http://rdf.freebase.com/ns/', '')
                    operator = '='
                    for cmp_function in OPERATOR_FUNCTION[operator]:
                        results.append({
                            "s_expression": CMP(cmp_function, relation, item['mid'])
                        })

                query_with_cvt = f"""
                SELECT DISTINCT ?r1, ?r2 where {{
                    ?o ?r2 ?cvt .
                    ?cvt ?r1 {item_rep} .
                    ?o ns:type.object.type ns:common.topic .
                    FILTER (?o != ?cvt && ?o != {item_rep} && ?cvt != {item_rep}) .
                    FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}.
                    FILTER regex(?r1, "^http://rdf.freebase.com/ns/") .
                    FILTER regex(?r2, "^http://rdf.freebase.com/ns/") .
                    {self.get_ignored_relations_filter("?r1")} .
                    {self.get_ignored_relations_filter("?r2")} .
                }} 
                """
                rows = self.execute_query_with_odbc(query_with_cvt)
                for row in rows:
                    r1 = row[0].replace('http://rdf.freebase.com/ns/', '')
                    r2 = row[1].replace('http://rdf.freebase.com/ns/', '')
                    operator = '='
                    for cmp_function in OPERATOR_FUNCTION[operator]:
                        results.append({
                            "s_expression": CMP(
                                cmp_function, 
                                JOIN(r2, r1), 
                                item['mid']
                            )
                        })

            elif item["type"].lower() == "class":
                # 对于 class, 仅枚举 JOIN type.object.type {item}
                return [{
                    "s_expression": JOIN("type.object.type", item['mid'])
                }]
            else:
                raise Exception(f"item: {item}")

        return results
    
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
        rows = self.execute_query_with_odbc(query)
        types = set()
        for row in rows:
            types.add(row[0].replace('http://rdf.freebase.com/ns/', ''))
        return list(types)

    def check_class(self, item, type):
        if (item is None) or (type is None):
            return False
        if item['type'] == 'entity':
            item_rep = f"ns:{item['mid']}"
        else: # 不应该出现 class 或者 literal, 其中 literal 没有类型吧
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
        rows = self.execute_query_with_odbc(query)
        return (len(rows) > 0 and rows[0][0] == 1)
    
    def enumerate_multivariate_relations(self):
        """
        多元候选关系
        - http://rdf.freebase.com/ns/ 开头
        - 能够指向一个 CVT 节点 FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
        - 因为 Freebase 存在逆关系，因此无需考虑方向了
        """
        query = f"""
        SELECT DISTINCT ?p WHERE {{
            ?s ?p ?cvt .
            FILTER regex(?p, "^http://rdf.freebase.com/ns/") .
            FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
        }}
        """
        rows = self.execute_query_with_odbc(query)
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
        rows = self.execute_query_with_odbc(query)
        results = set()
        for row in rows:
            results.add(row[0].replace('http://rdf.freebase.com/ns/', ''))
        return results
    
    def check_cvt_relation(self, relation):
        '''某个关系是否能连到 CVT 上'''
        query = f"""
        ASK {{
        {{?s ns:{relation} ?cvt .}} UNION {{?cvt ns:{relation} ?s .}}
        FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
        }}
        """
        rows = self.execute_query_with_odbc(query)
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
        rows = self.execute_query_with_odbc(query)
        return {
            "relation_1": relation_1,
            "relation_2": relation_2,
            "existence": len(rows) > 0 and rows[0][0] == 1
        }
    
    def check_three_relation_cvt_existence(self, relation_1, relation_2, relation_3):
        """
        三个关系在 Freebase 上是否能连到同一个 CVT 节点上
        """
        query = f"""
        ASK {{
            {{?cvt ns:{relation_1} ?o1 . ?cvt ns:{relation_2} ?o2 . ?cvt ns:{relation_3} ?o3 .}} 
            UNION {{?o1 ns:{relation_1} ?cvt . ?cvt ns:{relation_2} ?o2 . ?cvt ns:{relation_3} ?o3 .}}
            UNION {{?cvt ns:{relation_1} ?o1 . ?o2 ns:{relation_2} ?cvt . ?cvt ns:{relation_3} ?o3 .}}
            UNION {{?o1 ns:{relation_1} ?cvt . ?o2 ns:{relation_2} ?cvt . ?cvt ns:{relation_3} ?o3 .}}
            UNION {{?cvt ns:{relation_1} ?o1 . ?cvt ns:{relation_2} ?o2 . ?o3 ns:{relation_3} ?cvt .}} 
            UNION {{?o1 ns:{relation_1} ?cvt . ?cvt ns:{relation_2} ?o2 . ?o3 ns:{relation_3} ?cvt .}}
            UNION {{?cvt ns:{relation_1} ?o1 . ?o2 ns:{relation_2} ?cvt . ?o3 ns:{relation_3} ?cvt .}}
            UNION {{?o1 ns:{relation_1} ?cvt . ?o2 ns:{relation_2} ?cvt . ?o3 ns:{relation_3} ?cvt .}}
            FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
        }} 
        """
        rows = self.execute_query_with_odbc(query)
        return {
            "relation_1": relation_1,
            "relation_2": relation_2,
            "relation_3": relation_3,
            "existence": len(rows) > 0 and rows[0][0] == 1
        }
    
    def get_cvt_connected_relation(self, rel):
        """
        查找和 rel 通过一个 CVT 节点相连的其他关系
        """
        rel_conn_info = (
            lambda conn_trip, rel, var: f"""
            select distinct ?p WHERE {{
                {{
                    select distinct {var}
                    where {{
                        ?s ns:{rel} ?o.
                        FILTER NOT EXISTS {{ {var} ns:type.object.name ?name . }} . # 要求 var 是一个 CVT 节点
                    }}
                    LIMIT 10000000 # 数量级限制，超过这一限制的就只能忽略了
                }}
                {conn_trip} # 两个关系相连的方式，考虑方向
                FILTER (strstarts(str(?p),"http://rdf.freebase.com/ns/")) .
            }}
            """
        )
        ss_trip = "?s ?p ?t." # 第一个 s 表示CVT节点相对于 {rel} 的位置；第二个 s 表示CVT节点相对于查询得到的新关系的位置
        so_trip = "?t ?p ?s."
        os_trip = "?o ?p ?t."
        oo_trip = "?t ?p ?o."
        results = dict()
        results["S-S"] = list(self.get_execution_result_one_variable(
            rel_conn_info(ss_trip, rel, "?s")
        )) # ?s 是 cvt 节点，两个关系通过 ?s 连接起来
        results["S-O"] = list(self.get_execution_result_one_variable(
            rel_conn_info(so_trip, rel, "?s")
        )) 
        results["O-S"] = list(self.get_execution_result_one_variable(
            rel_conn_info(os_trip, rel, "?o")
        )) 
        results["O-O"] = list(self.get_execution_result_one_variable(
            rel_conn_info(oo_trip, rel, "?o")
        )) 
        results["rel"] = rel
        return results

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
        rows = self.execute_query_with_odbc(query)
        label = rows[0][0] if len(rows) >= 1 else None
        return {
            "entity": entity,
            "label": label
        }

    def query_entity_aliases(self, entity):
        query = f"""
        SELECT DISTINCT ?alias WHERE {{
            ns:{entity} ns:common.topic.alias ?alias .
            FILTER(LANG(?alias) = 'en')
        }}
        """
        rows = self.execute_query_with_odbc(query)
        aliases = set()
        for row in rows:
            aliases.add(row[0])
        return {
            "entity": entity,
            "aliases": list(aliases)
        } 
    
    # def query_constant_connected_cvt_pattern(self, constant_0, constant_1, constant_2):
    #     """查询的前提是假设指向实体的关系都有逆关系，指向 literal 的没有逆关系"""
    #     # 打个补丁，数据格式有点对不上
    #     if constant_0.startswith('m.') or constant_0.startswith('g.'):
    #         constant_0 = f'ns:{constant_0}'
    #     if constant_1.startswith('m.') or constant_1.startswith('g.'):
    #         constant_1 = f'ns:{constant_1}'
    #     if constant_2.startswith('m.') or constant_2.startswith('g.'):
    #         constant_2 = f'ns:{constant_2}'
    #     constant_0 = FreebaseConstantSerializer.get_freebase_constant(constant_0)
    #     constant_1 = FreebaseConstantSerializer.get_freebase_constant(constant_1)
    #     constant_2 = FreebaseConstantSerializer.get_freebase_constant(constant_2)

    #     '''
    #     无论是实体还是 literal, ?cvt ?p1 {constant_0} 都应该是能够查询到的
    #     ?cvt ?p1 {constant_0} 的写法，应该还可以避免 ?cvt 是个 literal
    #     '''
    #     query = f"""
    #     SELECT DISTINCT ?p1, ?p2, ?p3 WHERE {{
    #         ?cvt ?p1 {constant_0} .
    #         ?cvt ?p2 {constant_1} .
    #         ?cvt ?p3 {constant_2} .
    #         FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
    #         FILTER (strstarts(str(?p1),"http://rdf.freebase.com/ns/")) .
    #         FILTER (strstarts(str(?p2),"http://rdf.freebase.com/ns/")) .
    #         FILTER (strstarts(str(?p3),"http://rdf.freebase.com/ns/")) .
    #         FILTER (?p1 != ?p2 && ?p1 != ?p3 && ?p2 != ?p3 ) .
    #     }}
    #     """
    #     results = list()
    #     rows = self.execute_query_with_odbc(query)
    #     for row in rows:
    #         results.append((
    #             row[0].replace('http://rdf.freebase.com/ns/', ''),
    #             row[1].replace('http://rdf.freebase.com/ns/', ''),
    #             row[2].replace('http://rdf.freebase.com/ns/', '')
    #         ))
    #     return {
    #         "constant_list": (constant_0, constant_1, constant_2),
    #         "pattern_list": results
    #     }
 
 
class SparqlOdbcQuerierNoSexpr(ConcurrentExecutor):
    def __init__(self, odbc_config, sparql_wrapper_path, logger, timeout=6, sparql_cache_dir = None, direct_manage_2hop = False):
        super().__init__(logger)
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
        self.direct_manage_2hop = direct_manage_2hop    #一个技巧，探索a <-r1- ?cvt -r2->时，可以先探索第一跳的r1，清除blacklist并排序后，再去找r2，会快很多，但损失一些recall
        if self.sparql_cache_dir is not None:
            if not os.path.isfile(self.sparql_cache_dir):
                dump_json(dict(), self.sparql_cache_dir)
            self.cached_results = load_json(self.sparql_cache_dir)
            #每次保存一个backup文件，防止修改源文件造成缓存损坏
            dump_json(self.cached_results, sparql_cache_dir.split(".")[0]+"_backup.json")
            self.num_new_cache = 0
        else:
            self.cached_results = None
    
    def format_sparql(self, sparql):
        return sparql.replace("\n", " ").replace(" ", "")
    
    def write_current_cache_to_file(self):
        dump_json(self.cached_results, self.sparql_cache_dir)

    def update_cache_results(self, sparql, results, save_split = 1000):
        #更新cache；同时，如果新增了超过save_split个缓存，则将新增的储存到文件中
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
        connection = pyodbc.connect(
            self.odbc_config
        )
        connection.setdecoding(pyodbc.SQL_CHAR, encoding='utf8')
        connection.setdecoding(pyodbc.SQL_WCHAR, encoding='utf8')
        connection.setencoding(encoding='utf8')
        connection.timeout = self.timeout
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

    def execute_query(self, query, retry=4):
        for idx in range(retry): # 有时候出现 502 错误，难以定位原因，就重试几次吧
            if idx > 0:
                self.logger.info(f"Retrying execute_query(); idx:{idx}")
            try:
                query = f"{self.SPARQL_PREFIX} {query}"
                self.sparql_wrapper.setQuery(query)
                results = self.sparql_wrapper.query().convert()
                return results['results']['bindings']
            except Exception as err:
                self.logger.error(f"Query Execution Failed: {query}, error: {str(err)}")
        return []
    
    def get_execution_result_one_variable(self, query, service = "odbc"):
        #整理结果
        if service == "odbc":
            rows = self.execute_query_with_odbc(query)
            results = set()
            for row in rows:
                results.add(str(row[0]).replace('http://rdf.freebase.com/ns/', ''))
        elif service == "sparql_wrapper":
            rows = self.execute_query(query, retry=1)
            results = set()
            for r in rows:
                for key, inner_dict in r.items():
                    # 这里的 inner_dict 就是 {'type': 'uri', 'value': '...'}
                    if "value" in inner_dict:
                        val = inner_dict["value"]
                        # 处理字符串替换
                        clean_val = val.replace('http://rdf.freebase.com/ns/', '')
                        results.add(clean_val)
        return results
    
    def get_domain(self, item):
        query = f"""
        SELECT DISTINCT ?domain WHERE {{
            ns:{item} rdfs:domain ?domain
        }}
        """
        rows = self.execute_query_with_odbc(query)
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
        rows = self.execute_query_with_odbc(query)
        rtn = set()
        for row in rows:
            rtn.add(row[0].replace('http://rdf.freebase.com/ns/', ''))
        return rtn

    def get_execution_result_one_variable_sparql_wrapper(self, query, retry=4):
        """
        答案类型是 Literal 的特殊实现
        会把 Literal 的类型拼接起来
        注意，调用此函数时，查询目标应该只有一个
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
        rows = self.execute_query_with_odbc(query)
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
        rows = self.execute_query_with_odbc(query)
        name = rows[0][0] if len(rows) >= 1 else "" # 返回空串，处理起来一致

        if not name:
            query2 = f"""
            SELECT DISTINCT ?x WHERE {{
                {kb_item_rep} ns:common.topic.alias ?x .
                FILTER (langMatches( lang(?x), "EN" ) )
            }} LIMIT 1
            """
            rows = self.execute_query_with_odbc(query2)
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
        rows = self.execute_query_with_odbc(query)
        types = set()
        for row in rows:
            types.add(row[0].replace('http://rdf.freebase.com/ns/', ''))
        return list(types)

    def check_class(self, item, type):
        if (item is None) or (type is None):
            return False
        if item['type'] == 'entity':
            item_rep = f"ns:{item['mid']}"
        else: # 不应该出现 class 或者 literal, 其中 literal 没有类型吧
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
        rows = self.execute_query_with_odbc(query)
        return (len(rows) > 0 and rows[0][0] == 1)
    
    def enumerate_multivariate_relations(self):
        """
        多元候选关系
        - http://rdf.freebase.com/ns/ 开头
        - 能够指向一个 CVT 节点 FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
        - 因为 Freebase 存在逆关系，因此无需考虑方向了
        """
        query = f"""
        SELECT DISTINCT ?p WHERE {{
            ?s ?p ?cvt .
            FILTER regex(?p, "^http://rdf.freebase.com/ns/") .
            FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
        }}
        """
        rows = self.execute_query_with_odbc(query)
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
        rows = self.execute_query_with_odbc(query)
        results = set()
        for row in rows:
            results.add(row[0].replace('http://rdf.freebase.com/ns/', ''))
        return results
    
    def check_cvt_relation(self, relation):
        '''某个关系是否能连到 CVT 上'''
        query = f"""
        ASK {{
        {{?s ns:{relation} ?cvt .}} UNION {{?cvt ns:{relation} ?s .}}
        FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
        }}
        """
        rows = self.execute_query_with_odbc(query)
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
        rows = self.execute_query_with_odbc(query)
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
        rows = self.execute_query_with_odbc(query)
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
            for row in self.execute_query_with_odbc(query):
                entity_labels[row[0].replace('http://rdf.freebase.com/ns/', '')] = row[1]
        return entity_labels

    def query_entity_aliases(self, entity):
        query = f"""
        SELECT DISTINCT ?alias WHERE {{
            ns:{entity} ns:common.topic.alias ?alias .
            FILTER(LANG(?alias) = 'en')
        }}
        """
        rows = self.execute_query_with_odbc(query)
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
        给定一个LF，
        获取从start_point出发，经由prev_path后，最后新增一跳p可到达end_point，满足条件的p
        如果end_point是None，那么相当于end_point可以是任意的，此时p是从start_point出发，经由prev_path后的全部可能的下一个p
        实际上需要合并one_hop_path与one_hop_reversed，但不考虑type的处理
        注意：如果终点是任意的且prev_path不为空，那么每次查询的结果都会包含一个逆向关系，即终点被绑定为路径中的上一个变量。我们暂时不对其进行处理。
        get_end_points：如果end_point是None，设置get_end_points=True会返回变量的可能取值。
        '''
        def add_cvt_neq_filter(expand_point_rep, end_point_rep):
            #很抽象， FILTER (a1 != a2) 会导致查不出结果，影响phase2
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
            #优化：针对?IQ的度数很大的情况，思路是，如果计算LF此时的expand_point不大，那么直接穷举expand_point的情况
            #注意到，如果我们求?cvt 由于可能的?x ?r1 ?cvt ?r2 ?y太多，经常超时；我们尝试将其手动改写为两部分
            temp_q = f"SELECT DISTINCT COUNT(*) AS ?cnt WHERE {{ {sparql_gp} }}"
            expand_point_num = int(list(self.get_execution_result_one_variable(temp_q))[0])
        if LF is not None and expand_point_num <= 3:
            temp_q = f"SELECT DISTINCT {expand_point_rep} WHERE {{ {sparql_gp} }}"
            res_temp = self.get_execution_result_one_variable(temp_q)
            #我们要求,这些expand point的可能取值都要是实体
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
                    for row in self.execute_query_with_odbc(query_wo_cvt):
                        r = row[0].replace('http://rdf.freebase.com/ns/', '')
                        reversed = int(row[1]) == 1
                        if reversed:
                            results_wo_cvt.append("^"+r)
                        else:
                            results_wo_cvt.append(r)
                    #这里的出发点是，只保留那些用时长的空查询（防止查炸时候，大量快速返回的空结果毁坏缓存）
                    if time.time() - time_1 > 0.5 or len(results_wo_cvt) > 0:
                        self.update_cache_results(formated_sparql_wo_cvt, results_wo_cvt)
            else:
                for row in self.execute_query_with_odbc(query_wo_cvt):
                    r = row[0].replace('http://rdf.freebase.com/ns/', '')
                    reversed = int(row[1]) == 1
                    if reversed:
                        results_wo_cvt.append("^"+r)
                    else:
                        results_wo_cvt.append(r)
        else:
            assert(0)
        # else:
        #     for row in self.execute_query_with_odbc(query_wo_cvt):
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
                            for row in self.execute_query_with_odbc(query):
                                r1 = row[0].replace('http://rdf.freebase.com/ns/', '')
                                r2 = row[1].replace('http://rdf.freebase.com/ns/', '')
                                if idx == 0:
                                    query_result.append(r1 + "/" + r2)
                                else:
                                    query_result.append("^" + r1 + "/" + "^" + r2)
                            #这里的出发点是，只保留那些用时长的空查询（防止查炸时候，大量快速返回的空结果毁坏缓存）
                            if time.time() - time_1 > 0.2 or len(results_w_cvt) > 0:
                                self.update_cache_results(formated_sparql, query_result)
                    else:
                        query_result = []
                        for row in self.execute_query_with_odbc(query):
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
                        #这里的出发点是，只保留那些用时长的空查询（防止查炸时候，大量快速返回的空结果毁坏缓存）
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
                        #这里的出发点是，只保留那些用时长的空查询（防止查炸时候，大量快速返回的空结果毁坏缓存）
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
                        for row in self.execute_query_with_odbc(query_with_cvt_phase2):
                            r1 = row[0].replace('http://rdf.freebase.com/ns/', '')
                            r2 = row[1].replace('http://rdf.freebase.com/ns/', '')
                            #我们规定，CVT节点的两条边都是出度，因而是REVERSED的
                            results_w_cvt.append("^" + r1 + "/" + r2)
                        #这里的出发点是，只保留那些用时长的空查询（防止查炸时候，大量快速返回的空结果毁坏缓存）
                        if time.time() - time_1 > 0.2 or len(results_w_cvt) > 0:
                            self.update_cache_results(formated_sparql_w_cvt, results_w_cvt)
                else:
                    for row in self.execute_query_with_odbc(query_with_cvt_phase2):
                        r1 = row[0].replace('http://rdf.freebase.com/ns/', '')
                        r2 = row[1].replace('http://rdf.freebase.com/ns/', '')
                        results_w_cvt.append("^" + r1 + "/" + r2)
            else:
                assert(0)
            # else: 
            #     #if "?cvt" not in sparql_gp:
            #     for row in self.execute_query_with_odbc(query_with_cvt_phase2):
            #         r1 = row[0].replace('http://rdf.freebase.com/ns/', '')
            #         r2 = row[1].replace('http://rdf.freebase.com/ns/', '')
            #         end_point = row[2].replace('http://rdf.freebase.com/ns/', '')
            #         results.append({"relation":  "^" + r1 + "/" + r2, "end":end_point})
            #===============================================================================================================================
        results = results_wo_cvt + results_w_cvt
        return results     


    def get_next_hop_items_with_LF(self, LF:SimpleGraph, expand_point:Node, answer = None, semantic_sim_ranker = None, question = None):
        '''
        给定一个LF，
        获取从start_point出发，经由prev_path后，最后新增一跳p可到达end_point，满足条件的p
        如果end_point是None，那么相当于end_point可以是任意的，此时p是从start_point出发，经由prev_path后的全部可能的下一个p
        实际上需要合并one_hop_path与one_hop_reversed，但不考虑type的处理
        注意：如果终点是任意的且prev_path不为空，那么每次查询的结果都会包含一个逆向关系，即终点被绑定为路径中的上一个变量。我们暂时不对其进行处理。
        get_end_points：如果end_point是None，设置get_end_points=True会返回变量的可能取值。
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
        #优化：针对?IQ的度数很大的情况，思路是，如果计算LF此时的expand_point不大，那么直接穷举expand_point的情况
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
        for row in self.execute_query_with_odbc(query_wo_cvt):
            ent = row[0].replace('http://rdf.freebase.com/ns/', '')
            label = row[1]
            results.append({"mid": ent, "label":label})         
        #第二部分：CVT查询========================================================================================================================
        if extend_point_values_rep is not None:
        #注意到，如果我们求?cvt 由于可能的?x ?r1 ?cvt ?r2 ?y太多，经常超时；我们尝试将其手动改写为两部分
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
        for row in self.execute_query_with_odbc(query_with_cvt_phase2):
            ent = row[0].replace('http://rdf.freebase.com/ns/', '')
            label = row[1]
            results.append({"mid": ent, "label":label})         
        return results     


class SparqlOdbcQuerierNoSexprWikidata(ConcurrentExecutor):
    def __init__(self, odbc_config, sparql_wrapper_path, logger, timeout=10, sparql_cache_dir = None):
        super().__init__(logger)
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
            #每次保存一个backup文件，防止修改源文件造成缓存损坏
            dump_json(self.cached_results, sparql_cache_dir.split(".")[0]+"_backup.json")
            self.num_new_cache = 0
        else:
            self.cached_results = None
    
    def format_sparql(self, sparql):
        return sparql.replace("\n", " ").replace(" ", "")
    
    def write_current_cache_to_file(self):
        dump_json(self.cached_results, self.sparql_cache_dir)

    def update_cache_results(self, sparql, results, save_split = 1000):
        #更新cache；同时，如果新增了超过save_split个缓存，则将新增的储存到文件中
        self.cached_results[sparql] = results
        self.num_new_cache += 1
        if self.num_new_cache == save_split:
            self.write_current_cache_to_file()
            self.num_new_cache = 0
            self.cached_results = load_json(self.sparql_cache_dir)

    def connect(self):
        if self.connection is None:
            connection = pyodbc.connect(
                self.odbc_config
            )
            connection.setdecoding(pyodbc.SQL_CHAR, encoding='utf8')
            connection.setdecoding(pyodbc.SQL_WCHAR, encoding='utf8')
            connection.setencoding(encoding='utf8')
            connection.timeout = self.timeout
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
        for idx in range(retry): # 有时候出现 502 错误，难以定位原因，就重试几次吧
            if idx > 0:
                self.logger.info(f"execute_query(); idx:{idx}")
            try:
                complete_query = f"{self.SPARQL_PREFIX} {query}"
                self.sparql_wrapper.setQuery(complete_query)
                results = self.sparql_wrapper.query().convert()
                return results['results']['bindings'] # ASK 类型语句会报错，但是本身我们也处理不了 ASK
            except Exception as err:
                self.logger.error(f"Query Execution Failed: {query}, error: {str(err)}")
        return []
    
    
    def get_execution_result_one_variable(self, query, service = "odbc"):
        if service == "odbc":
            rows = self.execute_query_with_odbc(query)
            results = set()
            for row in rows:
                results.add(str(row[0]))
        elif service == "sparql_wrapper":
            rows = self.execute_query(query, retry=1)
            results = set()
            for r in rows:
                results.add(str(list(r.values())[0]['value']))
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
            query_result = self.execute_query_with_odbc(query)
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
        rows = self.execute_query_with_odbc(query)
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
        rows = self.execute_query_with_odbc(query)
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
        rows = self.execute_query_with_odbc(query)
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
        rows = self.execute_query_with_odbc(query)
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
        rows = self.execute_query_with_odbc(query1)
        freebase_mid = post_process_mid(rows[0][0]) if len(rows) >= 1 else "" # 返回空串，处理起来一致
        
        if not freebase_mid:
            '''Google KG id, 同样出现在 freebase 中，g.123'''
            query2 = f"""
            SELECT ?o WHERE {{
                wd:{wikidata_mid} wdt:P2671 ?o .
            }}
            """
            rows = self.execute_query_with_odbc(query2)
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
        rows = self.execute_query_with_odbc(query1)
        wikidata_mid = rows[0][0].replace('http://www.wikidata.org/entity/', '') if len(rows) >= 1 else None
        
        if not wikidata_mid:
            '''Google KG id, 同样出现在 freebase 中，g.123'''
            query2 = f"""
            SELECT ?o WHERE {{
                ?s wdt:P2671 {processed_freebase_mid}.
            }}
            """
            rows = self.execute_query_with_odbc(query2)
            wikidata_mid = rows[0][0].replace('http://www.wikidata.org/entity/', '') if len(rows) >= 1 else None # 返回空串，处理起来一致
        
        return {
            "freebase_mid": processed_freebase_mid,
            "wikidata_mid": wikidata_mid
        }
    
    # def query_constant_connected_cvt_pattern(self, constant_0, constant_1, constant_2):
    #     constant_0 = ConstantSerializer.get_wikidata_constant(constant_0)
    #     constant_1 = ConstantSerializer.get_wikidata_constant(constant_1)
    #     constant_2 = ConstantSerializer.get_wikidata_constant(constant_2)
    #     results = list()

    #     if (constant_0 is None) or (constant_1 is None) or (constant_2 is None):
    #         return {
    #             "constant_list": (constant_0, constant_1, constant_2),
    #             "pattern_list": results
    #         }
        
    #     for (subj, obj_ps, obj_pq) in itertools.permutations([constant_0, constant_1, constant_2], 3):
    #         query = f"""
    #         SELECT DISTINCT ?p, ?ps, ?pq {{
    #             {subj} ?p [
    #                 ?ps {obj_ps};
    #                 ?pq {obj_pq}
    #             ] .
    #             FILTER (strstarts(str(?p), "http://www.wikidata.org/prop/P")).
    #             FILTER (strstarts(str(?ps), "http://www.wikidata.org/prop/statement/P")).
    #             FILTER (strstarts(str(?pq), "http://www.wikidata.org/prop/qualifier/P")).
    #         }}
    #         """
            
    #         rows = self.execute_query_with_odbc(query)
    #         for row in rows:
    #             results.append((
    #                 row[0].replace('http://www.wikidata.org/prop/', 'p:'),
    #                 row[1].replace('http://www.wikidata.org/prop/statement/', 'ps:'),
    #                 row[2].replace('http://www.wikidata.org/prop/qualifier/', 'pq:')
    #             ))
        
    #     return {
    #         "constant_list": (constant_0, constant_1, constant_2),
    #         "pattern_list": results
    #     }

    def query_constant_connected_cvt_pattern(self, constant_0, constant_1, constant_2):
        constant_0_sparql_rep = ConstantSerializer.get_wikidata_constant(constant_0)
        constant_1_sparql_rep = ConstantSerializer.get_wikidata_constant(constant_1)
        constant_2_sparql_rep = ConstantSerializer.get_wikidata_constant(constant_2)
        results = list()

        if (constant_0 is None) or (constant_1 is None) or (constant_2 is None):
            return {
                "constant_list": (constant_0, constant_1, constant_2),
                "pattern_list": results
            }

        if ConstantSerializer.get_constant_type(constant_0) is WIKIDATA_CONSTANT_TYPE.ENTITY:
            # 按照 RDF 规范，Literal 不能出现在主语位置
            query = f"""
            SELECT DISTINCT ?p1, ?p2, ?p3 WHERE {{
                {constant_0_sparql_rep} ?p1 ?statement .
                ?statement ?p2 {constant_1_sparql_rep} .
                ?statement ?p3 {constant_2_sparql_rep} .
                FILTER (strstarts(str(?p1), "http://www.wikidata.org/prop/P")).
                FILTER (strstarts(str(?p2), "http://www.wikidata.org/prop/")).
                FILTER (strstarts(str(?p3), "http://www.wikidata.org/prop/")).
                FILTER (strstarts(str(?statement), "http://www.wikidata.org/entity/statement/")).
                FILTER (?p1 != ?p2 && ?p1 != ?p3 && ?p2 != ?p3 ) .
            }}
            """
            rows = self.execute_query_with_odbc(query)
            for row in rows:
                results.append((
                    constant_0, row[0].replace('http://www.wikidata.org/prop/', 'p:'),
                    row[1].replace('http://www.wikidata.org/prop/statement/', 'ps:').replace('http://www.wikidata.org/prop/qualifier/', 'pq:'), constant_1,
                    row[2].replace('http://www.wikidata.org/prop/statement/', 'ps:').replace('http://www.wikidata.org/prop/qualifier/', 'pq:'), constant_2
                ))
        
        if ConstantSerializer.get_constant_type(constant_1) is WIKIDATA_CONSTANT_TYPE.ENTITY:
            query = f"""
            SELECT DISTINCT ?p1, ?p2, ?p3 WHERE {{
                {constant_1_sparql_rep} ?p1 ?statement .
                ?statement ?p2 {constant_0_sparql_rep} .
                ?statement ?p3 {constant_2_sparql_rep} .
                FILTER (strstarts(str(?p1), "http://www.wikidata.org/prop/P")).
                FILTER (strstarts(str(?p2), "http://www.wikidata.org/prop/")).
                FILTER (strstarts(str(?p3), "http://www.wikidata.org/prop/")).
                FILTER (strstarts(str(?statement), "http://www.wikidata.org/entity/statement/")).
                FILTER (?p1 != ?p2 && ?p1 != ?p3 && ?p2 != ?p3 ) .
            }}
            """
            rows = self.execute_query_with_odbc(query)
            for row in rows:
                results.append((
                    constant_1, row[0].replace('http://www.wikidata.org/prop/', 'p:'),
                    row[1].replace('http://www.wikidata.org/prop/statement/', 'ps:').replace('http://www.wikidata.org/prop/qualifier/', 'pq:'), constant_0,
                    row[2].replace('http://www.wikidata.org/prop/statement/', 'ps:').replace('http://www.wikidata.org/prop/qualifier/', 'pq:'), constant_2
                ))
        
        if ConstantSerializer.get_constant_type(constant_2) is WIKIDATA_CONSTANT_TYPE.ENTITY:
            query = f"""
            SELECT DISTINCT ?p1, ?p2, ?p3 WHERE {{
                {constant_2_sparql_rep} ?p1 ?statement .
                ?statement ?p2 {constant_0_sparql_rep} .
                ?statement ?p3 {constant_1_sparql_rep} .
                FILTER (strstarts(str(?p1), "http://www.wikidata.org/prop/P")).
                FILTER (strstarts(str(?p2), "http://www.wikidata.org/prop/")).
                FILTER (strstarts(str(?p3), "http://www.wikidata.org/prop/")).
                FILTER (strstarts(str(?statement), "http://www.wikidata.org/entity/statement/")).
                FILTER (?p1 != ?p2 && ?p1 != ?p3 && ?p2 != ?p3 ) .
            }}
            """
            rows = self.execute_query_with_odbc(query)
            for row in rows:
                results.append((
                    constant_2, row[0].replace('http://www.wikidata.org/prop/', 'p:'),
                    row[1].replace('http://www.wikidata.org/prop/statement/', 'ps:').replace('http://www.wikidata.org/prop/qualifier/', 'pq:'), constant_0,
                    row[2].replace('http://www.wikidata.org/prop/statement/', 'ps:').replace('http://www.wikidata.org/prop/qualifier/', 'pq:'), constant_1
                ))

        
        return {
            "constant_list": (constant_0, constant_1, constant_2),
            "pattern_list": results
        }


    def instantiate_multivariate_graph_pattern(self, prop_p, prop_ps, prop_pq, limit=500):
        """
        图模式在 KB 上实例化得到的事实
        """
        query = f"""
        SELECT DISTINCT ?s, ?o_ps, ?o_pq {{
            ?s p:{prop_p} [
            ps:{prop_ps} ?o_ps;
            pq:{prop_pq} ?o_pq
            ] .
        }} LIMIT {limit}
        """
        rows = self.execute_query(query)
        results = set()
        for row in rows:
            s_binding = row['s']
            s_serialized = ConstantSerializer(s_binding['type'], s_binding['value'], s_binding.get('datatype', None), s_binding.get('xml:lang', None))
            ops_binding = row['o_ps']
            ops_serialized = ConstantSerializer(ops_binding['type'], ops_binding['value'], ops_binding.get('datatype', None), ops_binding.get('xml:lang', None))
            opq_binding = row['o_pq']
            opq_serialized = ConstantSerializer(opq_binding['type'], opq_binding['value'], opq_binding.get('datatype', None), opq_binding.get('xml:lang', None))
            results.add((
                s_serialized.__repr__(), ops_serialized.__repr__(), opq_serialized.__repr__()
            ))
            
        return {
            "prop_p": prop_p,
            "prop_ps": prop_ps,
            "prop_pq": prop_pq,
            "results": list(results)
        }

    def query_one_hop_paths(self, grounded_item, answer_entity=None, answer_type=None):
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

            rows = self.execute_query_with_odbc(query_wo_qualifier)
            for row in rows:
                p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
                ps_prop = row[1].replace('http://www.wikidata.org/prop/statement/', 'ps:')
                results.append({
                    "s_expression": JOIN(p_prop, JOIN(ps_prop, grounded_item_rep)) 
                })
            
            rows = self.execute_query_with_odbc(query_with_qualifier)
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
            
            rows = self.execute_query_with_odbc(query_wo_qualifier)
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

            rows = self.execute_query_with_odbc(query_with_qualifier)
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
        
        rows = self.execute_query_with_odbc(query_wo_qualifier)
        for row in rows:
            p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
            ps_prop = row[1].replace('http://www.wikidata.org/prop/statement/', 'ps:')
            results.append({
                "s_expression": JOIN(R(ps_prop), JOIN(R(p_prop), grounded_item_rep))
            })
        
        rows = self.execute_query_with_odbc(query_with_qualifier)
        for row in rows:
            p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
            pq_prop = row[1].replace('http://www.wikidata.org/prop/qualifier/', 'pq:')
            results.append({
                "s_expression": JOIN(R(pq_prop), JOIN(R(p_prop), grounded_item_rep))
            })
            
        return results
    
    def query_one_hop_paths_reversed(self, grounded_item, answer_entity=None, answer_type=None):
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

            rows = self.execute_query_with_odbc(query_wo_qualifier)
            for row in rows:
                p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
                ps_prop = row[1].replace('http://www.wikidata.org/prop/statement/', 'ps:')
                results.append({
                    "s_expression": JOIN(R(ps_prop), JOIN(R(p_prop), grounded_item_rep)) 
                })
            
            rows = self.execute_query_with_odbc(query_with_qualifier)
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
            rows = self.execute_query_with_odbc(query_wo_qualifier)
            for row in rows:
                p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
                ps_prop = row[1].replace('http://www.wikidata.org/prop/statement/', 'ps:')
                results.append({
                    "s_expression": JOIN(p_prop, JOIN(ps_prop, grounded_item_rep)) 
                })
            
            rows = self.execute_query_with_odbc(query_with_qualifier)
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
            rows = self.execute_query_with_odbc(query_wo_qualifier)
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
            
            rows = self.execute_query_with_odbc(query_with_qualifier)
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
        arg_results = list() # ARGMIN / ARGMAX 相关结果
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
        
        rows = self.execute_query_with_odbc(query_wo_qualifier)
        for row in rows:
            p_prop = row[0].replace('http://www.wikidata.org/prop/', 'p:')
            ps_prop = row[1].replace('http://www.wikidata.org/prop/statement/', 'ps:')
            for function in ARG_FUNCTIONS:
                arg_results.append({
                    "function": function,
                    "relation": JOIN(p_prop, ps_prop)
                })
        
        rows = self.execute_query_with_odbc(query_with_qualifier)
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
                rows = self.execute_query_with_odbc(query)
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
                rows = self.execute_query_with_odbc(query)
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
                rows = self.execute_query_with_odbc(query)
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
                rows = self.execute_query_with_odbc(query)
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
        rows = self.execute_query_with_odbc(query)
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
        rows = self.execute_query_with_odbc(query)
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
            rows = self.execute_query_with_odbc(query)
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
            rows = self.execute_query_with_odbc(query)
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
            rows = self.execute_query_with_odbc(query)
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
            rows = self.execute_query_with_odbc(query)
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
        rows = self.execute_query_with_odbc(query)
        return (len(rows) > 0 and rows[0][0] == 1)

    def check_all_fixed_statement_num(self, prop_p, prop_ps, prop_pq, s, o_ps, o_pq):
        """
        CVT / Statement 相关的三个常量都固定，获得此时 statement 节点的数量信息
        """
        from components.wikipedia_utils import ConstantSerializer
        s = ConstantSerializer.get_wikidata_constant(s)
        o_ps = ConstantSerializer.get_wikidata_constant(o_ps)
        o_pq = ConstantSerializer.get_wikidata_constant(o_pq)
        query = f"""
        SELECT COUNT(DISTINCT ?statement) {{
            {s} p:{prop_p} ?statement .
            ?statement ps:{prop_ps} {o_ps} .
            ?statement pq:{prop_pq} {o_pq} .
            FILTER (strstarts(str(?statement), "http://www.wikidata.org/entity/statement/")).
        }}
        """
        rows = self.execute_query_with_odbc(query)
        if len(rows) != 1:
            results = 0
        else:
            results = rows[0][0]
        return {
            "prop_p": prop_p,
            "prop_ps": prop_ps,
            "prop_pq": prop_pq,
            "s": s,
            "o_ps": o_ps,
            "o_pq": o_pq,
            "count": results
        }

    def check_s_ops_fixed_statement_num(self, prop_p, prop_ps, prop_pq, s, o_ps):
        """
        CVT / Statement 相关s, o_ps 固定，获得此时 statement 节点的数量信息
        """
        s = ConstantSerializer.get_wikidata_constant(s)
        o_ps = ConstantSerializer.get_wikidata_constant(o_ps)
        query = f"""
        SELECT COUNT(DISTINCT ?statement) {{
            {s} p:{prop_p} ?statement .
            ?statement ps:{prop_ps} {o_ps} .
            FILTER (strstarts(str(?statement), "http://www.wikidata.org/entity/statement/")).
        }}
        """
        rows = self.execute_query_with_odbc(query)
        if len(rows) != 1:
            results = 0
        else:
            results = rows[0][0]
        return {
            "prop_p": prop_p,
            "prop_ps": prop_ps,
            "prop_pq": prop_pq,
            "s": s,
            "o_ps": o_ps,
            "count": results
        }
    
    def check_s_opq_fixed_statement_num(self, prop_p, prop_ps, prop_pq, s, o_pq):
        """
        CVT / Statement 相关s, o_ps 固定，获得此时 statement 节点的数量信息
        """
        s = ConstantSerializer.get_wikidata_constant(s)
        o_pq = ConstantSerializer.get_wikidata_constant(o_pq)
        query = f"""
        SELECT COUNT(DISTINCT ?statement) {{
            {s} p:{prop_p} ?statement .
            ?statement pq:{prop_pq} {o_pq} .
            FILTER (strstarts(str(?statement), "http://www.wikidata.org/entity/statement/")).
        }}
        """
        rows = self.execute_query_with_odbc(query)
        if len(rows) != 1:
            results = 0
        else:
            results = rows[0][0]
        return {
            "prop_p": prop_p,
            "prop_ps": prop_ps,
            "prop_pq": prop_pq,
            "s": s,
            "o_pq": o_pq,
            "count": results
        }

    def get_reverse_property(self):
        query = """
        SELECT DISTINCT ?p1 ?p2 where {
            ?p1 wdt:P1696 ?p2 .
            FILTER (strstarts(str(?p1), "http://www.wikidata.org/entity/P")) .
            FILTER (strstarts(str(?p2), "http://www.wikidata.org/entity/P")) .
        } 
        """
        results = set()
        rows = self.execute_query_with_odbc(query)
        for row in rows:
            results.add((
                row[0].replace("http://www.wikidata.org/entity/", ""),
                row[1].replace("http://www.wikidata.org/entity/", "")
            ))
        return results

    def expand_next_hop_path_with_LF(self, LF:SimpleGraph, expand_point:Node, end_point = None, answer = None):
        '''
        给定一个LF，
        获取从start_point出发，经由prev_path后，最后新增一跳p可到达end_point，满足条件的p
        如果end_point是None，那么相当于end_point可以是任意的，此时p是从start_point出发，经由prev_path后的全部可能的下一个p
        实际上需要合并one_hop_path与one_hop_reversed，但不考虑type的处理
        get_end_points：如果end_point是None，设置get_end_points=True会返回变量的可能取值。
        '''
        '''
        本质上，wd的CVT与否都是两跳：一跳被解释为p+ps，包含cvt节点的实际上也是p+ps，只是多了PQ
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
            #优化：针对?IQ的度数很大的情况，思路是，如果计算LF此时的expand_point不大，那么直接穷举expand_point的情况
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
        #rn_result = self.execute_query_with_odbc(query_rn)
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


    def get_next_hop_items_with_LF(self, LF:SimpleGraph, expand_point:Node,  answer = None, semantic_sim_ranker = None, question = None):
        '''
        给定一个LF，
        获取从start_point出发，经由prev_path后，最后新增一跳p可到达end_point，满足条件的p
        如果end_point是None，那么相当于end_point可以是任意的，此时p是从start_point出发，经由prev_path后的全部可能的下一个p
        实际上需要合并one_hop_path与one_hop_reversed，但不考虑type的处理
        注意：如果终点是任意的且prev_path不为空，那么每次查询的结果都会包含一个逆向关系，即终点被绑定为路径中的上一个变量。我们暂时不对其进行处理。
        get_end_points：如果end_point是None，设置get_end_points=True会返回变量的可能取值。
        '''
        raise Exception("need modify")
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
        #优化：针对?IQ的度数很大的情况，思路是，如果计算LF此时的expand_point不大，那么直接穷举expand_point的情况
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
        for row in self.execute_query_with_odbc(query_wo_cvt):
            ent = row[0].replace('http://rdf.freebase.com/ns/', '')
            label = row[1]
            results.append({"mid": ent, "label":label})         
        #第二部分：CVT查询========================================================================================================================
        if extend_point_values_rep is not None:
        #注意到，如果我们求?cvt 由于可能的?x ?r1 ?cvt ?r2 ?y太多，经常超时；我们尝试将其手动改写为两部分
            query_with_cvt_phase1 = f"""
            SELECT DISTINCT ?r1 where {{
                VALUES {expand_point_rep} {{ {extend_point_values_rep} }}.
                ?cvt ?r1 {expand_point_rep} .
                FILTER regex(?r1, "^http://rdf.freebase.com/ns/") .
                FILTER NOT EXISTS {{ ?cvt ns:type.object.name ?name . }}. 
                {self.get_ignored_relations_filter("?r1")} .
            }}"""
            #这里，需要使用语义相似度对r1进行一次排序（避免太多超时）
            r1_results = list(self.get_execution_result_one_variable(query_with_cvt_phase1))
            if semantic_sim_ranker is not None:
                if len(r1_results) > CVT_R1_NUM:
                    top_k_results = semantic_sim_ranker.get_semantic_sim_topk(question, r1_results, CVT_R1_NUM)
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
                for blackr in CVT_R1_BLACKLIST:
                    if blackr in r1:
                        throw = True
                        break
                if not throw:
                    temp.append(r1)
            r1_results = temp
            if semantic_sim_ranker is not None:
                if len(r1_results) > CVT_R1_NUM:
                    top_k_results = semantic_sim_ranker.get_semantic_sim_topk(question, r1_results, CVT_R1_NUM)
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
        for row in self.execute_query_with_odbc(query_with_cvt_phase2):
            ent = row[0].replace('http://rdf.freebase.com/ns/', '')
            label = row[1]
            results.append({"mid": ent, "label":label})         
        return results     


