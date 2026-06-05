#!/usr/bin/env python3
"""
单元测试: executor 双后端 (ODBC/SPARQLWrapper) 格式一致性验证
/ Unit test: verify format consistency between ODBC and SPARQLWrapper backends.

测试范围 / Scope:
  1. _bindings_to_rows 类型还原 / type coercion
  2. _execute_query 单列查询  / single-column SELECT
  3. _execute_query 多列查询  / multi-column SELECT
  4. _execute_query ASK 查询  / ASK queries
  5. _execute_query COUNT 查询 / COUNT queries
  6. SPARQLWrapper ↔ ODBC 格式对齐 / format alignment

用法 / Usage:
  python tests/test_executor_backends.py                          # 仅 SPARQLWrapper 测试
  python tests/test_executor_backends.py --odbc                   # 包含 ODBC 对比测试
  python tests/test_executor_backends.py --endpoint <SPARQL_URL>  # 指定 SPARQL endpoint
"""

from __future__ import annotations

import sys
import os
import unittest
import logging
from typing import List, Tuple

# Ensure project root in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.sparql.executor import (
    HAS_PYODBC,
    SparqlOdbcQuerierNoSexpr,
    SparqlOdbcQuerierNoSexprWikidata,
)

# ============================================================
# Test SPARQL Endpoint (可外网访问的测试端点 / publicly accessible test endpoint)
# Wikidata 查询端点，无需认证即可使用 / Wikidata query endpoint, no auth needed
# ============================================================
DEFAULT_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"


# ============================================================
# 1. _bindings_to_rows 类型还原测试
# ============================================================

class TestBindingsToRows(unittest.TestCase):
    """测试 SPARQLWrapper dict → ODBC tuple 转换 + 类型还原."""

    @classmethod
    def setUpClass(cls):
        """Create a dummy executor just to access _bindings_to_rows."""
        cls.logger = logging.getLogger("test_bindings")
        cls.logger.setLevel(logging.CRITICAL)
        cls.executor = SparqlOdbcQuerierNoSexprWikidata(
            sparql_wrapper_path=DEFAULT_SPARQL_ENDPOINT,
            logger=cls.logger,
            service="sparql_wrapper",
        )

    def test_empty_bindings(self):
        """空结果返回空列表 / Empty bindings → empty list."""
        result = self.executor._bindings_to_rows([])
        self.assertEqual(result, [])

    def test_single_uri_column(self):
        """单列 URI — 对应 ODBC row[0], 值为完整 URI 字符串."""
        bindings = [
            {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q42"}},
            {"item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q513"}},
        ]
        rows = self.executor._bindings_to_rows(bindings)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], "http://www.wikidata.org/entity/Q42")
        self.assertEqual(rows[1][0], "http://www.wikidata.org/entity/Q513")
        # 确认是 tuple 格式，支持 row[0] 访问 / Verify tuple format
        self.assertIsInstance(rows[0], tuple)

    def test_multi_column_uri_and_literal(self):
        """多列 URI + Label — 对应 ODBC row[0], row[1]."""
        bindings = [
            {
                "entity": {"type": "uri", "value": "http://www.wikidata.org/entity/Q42"},
                "label": {"type": "literal", "value": "Douglas Adams"},
            },
            {
                "entity": {"type": "uri", "value": "http://www.wikidata.org/entity/Q80"},
                "label": {"type": "literal", "value": "Tim Berners-Lee"},
            },
        ]
        rows = self.executor._bindings_to_rows(bindings)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], "http://www.wikidata.org/entity/Q42")  # URI column
        self.assertEqual(rows[0][1], "Douglas Adams")                        # Literal column
        self.assertEqual(rows[1][1], "Tim Berners-Lee")

    def test_integer_type_coercion(self):
        """COUNT 返回的 integer → Python int (模拟 ODBC 行为)."""
        bindings = [
            {"cnt": {"type": "typed-literal", "datatype": "http://www.w3.org/2001/XMLSchema#integer", "value": "42"}},
        ]
        rows = self.executor._bindings_to_rows(bindings)
        self.assertEqual(rows[0][0], 42)
        self.assertIsInstance(rows[0][0], int)

    def test_boolean_type_coercion(self):
        """BIND True/False → Python int 1/0 (模拟 ODBC Virtuoso 行为)."""
        bindings = [
            {"reversed": {"type": "typed-literal", "datatype": "http://www.w3.org/2001/XMLSchema#boolean", "value": "true"}},
            {"reversed": {"type": "typed-literal", "datatype": "http://www.w3.org/2001/XMLSchema#boolean", "value": "false"}},
        ]
        rows = self.executor._bindings_to_rows(bindings)
        self.assertEqual(rows[0][0], 1)
        self.assertIsInstance(rows[0][0], int)
        self.assertEqual(rows[1][0], 0)

    def test_float_type_coercion(self):
        """Float/double → Python float."""
        bindings = [
            {"score": {"type": "typed-literal", "datatype": "http://www.w3.org/2001/XMLSchema#double", "value": "3.14159"}},
            {"score": {"type": "typed-literal", "datatype": "http://www.w3.org/2001/XMLSchema#float", "value": "2.718"}},
        ]
        rows = self.executor._bindings_to_rows(bindings)
        self.assertAlmostEqual(rows[0][0], 3.14159)
        self.assertIsInstance(rows[0][0], float)
        self.assertAlmostEqual(rows[1][0], 2.718)

    def test_decimal_type_coercion(self):
        """Decimal → float."""
        bindings = [
            {"price": {"type": "typed-literal", "datatype": "http://www.w3.org/2001/XMLSchema#decimal", "value": "99.95"}},
        ]
        rows = self.executor._bindings_to_rows(bindings)
        self.assertEqual(rows[0][0], 99.95)
        self.assertIsInstance(rows[0][0], float)

    def test_mixed_type_columns(self):
        """混合类型多列: BIND boolean + URI string + COUNT integer."""
        bindings = [
            {
                "reversed": {"type": "typed-literal", "datatype": "http://www.w3.org/2001/XMLSchema#boolean", "value": "true"},
                "property": {"type": "uri", "value": "http://rdf.freebase.com/ns/people.person.parents"},
                "cnt": {"type": "typed-literal", "datatype": "http://www.w3.org/2001/XMLSchema#integer", "value": "5"},
            },
        ]
        rows = self.executor._bindings_to_rows(bindings)
        row = rows[0]
        self.assertEqual(row[0], 1)                                            # boolean → int
        self.assertEqual(row[1], "http://rdf.freebase.com/ns/people.person.parents")  # uri → str
        self.assertEqual(row[2], 5)                                            # integer → int
        # 验证调用方常用的 .replace() 操作仍然有效 / Verify common .replace() still works
        self.assertEqual(row[1].replace("http://rdf.freebase.com/ns/", ""), "people.person.parents")

    def test_column_order_preserved(self):
        """列顺序与 SELECT 变量顺序一致 (Python 3.7+ dict 保序)."""
        bindings = [
            {
                "third": {"type": "literal", "value": "c"},
                "first": {"type": "literal", "value": "a"},
                "second": {"type": "literal", "value": "b"},
            },
        ]
        rows = self.executor._bindings_to_rows(bindings)
        # 列顺序应反映 bindings dict 的 key 顺序
        self.assertEqual(rows[0], ("c", "a", "b"))


# ============================================================
# 2. _execute_query 功能测试 (SPARQLWrapper, 真实 Wikidata 查询)
# ============================================================

class TestExecuteQuerySPARQLWrapper(unittest.TestCase):
    """
    通过 Wikidata 真实 SPARQL endpoint 测试 _execute_query.
    所有测试使用简单查询，返回结果在 1 行以内，不会对 Wikidata 造成压力.
    """

    @classmethod
    def setUpClass(cls):
        cls.logger = logging.getLogger("test_execute")
        cls.logger.setLevel(logging.CRITICAL)
        cls.executor = SparqlOdbcQuerierNoSexprWikidata(
            sparql_wrapper_path=DEFAULT_SPARQL_ENDPOINT,
            logger=cls.logger,
            service="sparql_wrapper",
            timeout=15,
        )

    def test_single_uri_result(self):
        """SELECT ?item WHERE { wd:Q42 rdfs:label ?item . FILTER(LANG(?item)='en') }
        返回单列 URI 结果，应能通过 row[0] 访问."""
        query = "SELECT ?item WHERE { wd:Q42 rdfs:label ?item . FILTER(LANG(?item)='en') }"
        rows = self.executor._execute_query(query, retry=2)
        self.assertGreater(len(rows), 0, "Should return at least 1 result from Wikidata")
        row = rows[0]
        # row[0] 应为字符串 / should be string
        self.assertIsInstance(row[0], str)
        self.assertIn("Douglas Adams", row[0], f"Expected 'Douglas Adams', got '{row[0]}'")

    def test_integer_count_result(self):
        """SELECT (COUNT(?item) AS ?cnt) — 返回 int 类型."""
        query = """
        SELECT (COUNT(?item) AS ?cnt) WHERE {
            wd:Q42 ?prop ?item .
        }
        """
        rows = self.executor._execute_query(query, retry=2)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertIsInstance(row[0], int, f"COUNT result should be int, got {type(row[0])}")
        self.assertGreater(row[0], 100, f"Q42 should have many statements, got {row[0]}")

    def test_ask_query_true(self):
        """ASK { wd:Q42 wdt:P31 wd:Q5 } — Q42 是 human, 应返回 [(1,)]."""
        query = "ASK { wd:Q42 wdt:P31 wd:Q5 }"
        rows = self.executor._execute_query(query, retry=2)
        self.assertEqual(rows, [(1,)], f"ASK should return [(1,)], got {rows}")

    def test_ask_query_false(self):
        """ASK { wd:Q42 wdt:P31 wd:Q123456789 } — 不存在的 class, 应返回 [(0,)]."""
        query = "ASK { wd:Q42 wdt:P31 wd:Q123456789 }"
        rows = self.executor._execute_query(query, retry=2)
        # Q123456789 极大概率不存在，ASK 返回 false
        # ASK false → [(0,)]
        self.assertEqual(rows[0][0], 0, f"ASK false should have row[0][0] == 0, got {rows}")

    def test_multi_column_bind_reversed(self):
        """模拟 expand_next_hop 的 BIND False/True 查询模式.
        / Simulate the BIND False/True pattern used in expand_next_hop."""
        query = """
        SELECT ?x ?reversed WHERE {
            { wd:Q42 ?x wd:Q513 . BIND(False AS ?reversed) }
            UNION
            { wd:Q513 ?x wd:Q42 . BIND(True AS ?reversed) }
        }
        LIMIT 5
        """
        rows = self.executor._execute_query(query, retry=2)
        self.assertGreater(len(rows), 0, "Should find at least one relation between Q42 and Q513")
        for row in rows:
            # row[0] = ?x (URI), row[1] = ?reversed (int)
            self.assertIsInstance(row[0], str, f"?x should be str, got {type(row[0])}")
            self.assertIn(row[1], (0, 1), f"?reversed should be 0 or 1 (int), got {row[1]}")
            self.assertIsInstance(row[1], int, f"?reversed should be int, got {type(row[1])}")

    def test_empty_result(self):
        """无结果的 SELECT 返回空列表."""
        query = "SELECT ?item WHERE { wd:Q42 wdt:P999999 ?item }"  # 不存在的属性
        rows = self.executor._execute_query(query, retry=2)
        self.assertEqual(rows, [])

    def test_uft8_label(self):
        """UTF-8 编码的 label 能正确返回."""
        query = """
        SELECT ?label WHERE {
            wd:Q183 rdfs:label ?label .
            FILTER(LANG(?label)='de')
        }
        """
        rows = self.executor._execute_query(query, retry=2)
        self.assertGreater(len(rows), 0)
        # 德国 → Deutschland (German)
        labels = [row[0] for row in rows]
        self.assertIn("Deutschland", labels, f"Expected 'Deutschland' in {labels}")

    def test_get_execution_result_one_variable(self):
        """get_execution_result_one_variable 应返回 set."""
        query = "SELECT ?item WHERE { wd:Q42 rdfs:label ?item . FILTER(LANG(?item)='en') }"
        results = self.executor.get_execution_result_one_variable(query)
        self.assertIsInstance(results, set)
        self.assertGreater(len(results), 0)
        self.assertIn("Douglas Adams", results)


# ============================================================
# 3. ODBC / SPARQLWrapper 对比测试
# ============================================================

class TestBackendConsistency:
    """
    ODBC 与 SPARQLWrapper 结果格式对比.
    需要 ODBC 环境配置: --odbc 参数启用.
    / Compare ODBC vs SPARQLWrapper output format.
      Requires ODBC; use --odbc flag.

    用法 / Usage:
      python -m pytest tests/test_executor_backends.py -k TestBackendConsistency --odbc
    """

    @staticmethod
    def build_fb_odbc_executor(odbc_config, sparql_wrapper_path, logger):
        """Build Freebase executor with ODBC backend."""
        from src.core.common import ODBC_CONFIG_DKILAB, SPARQL_wrapper_path_dkilab
        return SparqlOdbcQuerierNoSexpr(
            sparql_wrapper_path=sparql_wrapper_path or SPARQL_wrapper_path_dkilab,
            logger=logger,
            service="odbc",
            odbc_config=odbc_config or ODBC_CONFIG_DKILAB,
        )

    @staticmethod
    def build_fb_sparqlwrapper_executor(sparql_wrapper_path, logger):
        """Build Freebase executor with SPARQLWrapper backend."""
        from src.core.common import SPARQL_wrapper_path_dkilab
        return SparqlOdbcQuerierNoSexpr(
            sparql_wrapper_path=sparql_wrapper_path or SPARQL_wrapper_path_dkilab,
            logger=logger,
            service="sparql_wrapper",
        )

    @staticmethod
    def run_consistency_test(sparql_query: str,
                             odbc_executor: SparqlOdbcQuerierNoSexpr,
                             sw_executor: SparqlOdbcQuerierNoSexpr,
                             retry: int = 1) -> Tuple[bool, str, List, List]:
        """
        对比 ODBC 和 SPARQLWrapper 对同一查询的返回结果.
        / Compare ODBC and SPARQLWrapper results for the same query.

        Returns:
            (consistent: bool, message: str, odbc_rows: list, sw_rows: list)
        """
        odbc_rows = odbc_executor._execute_query(sparql_query, retry=retry)
        sw_rows = sw_executor._execute_query(sparql_query, retry=retry)

        if len(odbc_rows) != len(sw_rows):
            return False, f"Row count mismatch: ODBC={len(odbc_rows)}, SW={len(sw_rows)}", odbc_rows, sw_rows

        for i, (orow, swrow) in enumerate(zip(odbc_rows, sw_rows)):
            if len(orow) != len(swrow):
                return False, f"Column count mismatch at row {i}: ODBC={len(orow)}, SW={len(swrow)}", odbc_rows, sw_rows
            for j, (oval, sval) in enumerate(zip(orow, swrow)):
                if type(oval) != type(sval):
                    return False, (
                        f"Type mismatch at row {i} col {j}: "
                        f"ODBC={type(oval).__name__}({oval}), SW={type(sval).__name__}({sval})"
                    ), odbc_rows, sw_rows
                # Fuzzy float comparison
                if isinstance(oval, float):
                    if abs(oval - sval) > 1e-9:
                        return False, f"Float mismatch at row {i} col {j}: {oval} vs {sval}", odbc_rows, sw_rows
                elif isinstance(oval, int) and abs(oval) < 2**31:
                    if oval != sval:
                        return False, f"Value mismatch at row {i} col {j}: {oval} vs {sval}", odbc_rows, sw_rows
                elif isinstance(oval, str):
                    # URIs may have minor normalization differences; strip ns: prefix for comparison
                    ov = oval.replace('http://rdf.freebase.com/ns/', '')
                    sv = sval.replace('http://rdf.freebase.com/ns/', '')
                    if ov != sv:
                        return False, f"String mismatch at row {i} col {j}: '{ov}' vs '{sv}'", odbc_rows, sw_rows

        return True, "OK", odbc_rows, sw_rows


# ============================================================
# Runner
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="executor 双后端格式一致性测试 / Backend format consistency tests"
    )
    parser.add_argument("--odbc", action="store_true",
                        help="同时运行 ODBC 对比测试 / Also run ODBC comparison tests")
    parser.add_argument("--endpoint", default=DEFAULT_SPARQL_ENDPOINT,
                        help=f"SPARQL 端点 URL / SPARQL endpoint URL (default: {DEFAULT_SPARQL_ENDPOINT})")
    parser.add_argument("--odbc-config", default=None,
                        help="ODBC 连接字符串 / ODBC connection string")
    parser.add_argument("--fb-endpoint", default=None,
                        help="Freebase SPARQL endpoint (需 VPN/内网)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="详细输出 / Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Override endpoint
    if args.endpoint != DEFAULT_SPARQL_ENDPOINT:
        DEFAULT_SPARQL_ENDPOINT = args.endpoint

    print("=" * 60)
    print("executor 双后端格式一致性测试")
    print(f"SPARQL Endpoint: {DEFAULT_SPARQL_ENDPOINT}")
    print(f"ODBC Tests: {'enabled' if args.odbc else 'disabled (use --odbc)'}")
    print(f"HAS_PYODBC: {HAS_PYODBC}")
    print("=" * 60 + "\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Always run _bindings_to_rows tests (no backend needed)
    suite.addTests(loader.loadTestsFromTestCase(TestBindingsToRows))

    # Always run SPARQLWrapper tests
    suite.addTests(loader.loadTestsFromTestCase(TestExecuteQuerySPARQLWrapper))

    # Run
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # --- ODBC comparison tests (manual, require ODBC env) ---
    if args.odbc:
        if not HAS_PYODBC:
            print("\n⚠️  ODBC 测试跳过: pyodbc 未安装 / ODBC skipped: pyodbc not installed")
            sys.exit(0 if result.wasSuccessful() else 1)

        print("\n" + "=" * 60)
        print("ODBC / SPARQLWrapper 格式对比测试")
        print("Format comparison tests")
        print("=" * 60)

        logger = logging.getLogger("odbc_test")
        logger.setLevel(logging.WARNING)

        # Build executors
        odbc_ex = TestBackendConsistency.build_fb_odbc_executor(
            args.odbc_config, args.fb_endpoint, logger)
        sw_ex = TestBackendConsistency.build_fb_sparqlwrapper_executor(
            args.fb_endpoint, logger)

        test_queries = [
            # (description, query)
            ("Single URI SELECT",
             "SELECT ?domain WHERE { ns:m.02mjmr rdfs:domain ?domain } LIMIT 1"),
            ("COUNT query",
             "SELECT (COUNT(?p) AS ?cnt) WHERE { ns:m.02mjmr ?p ?o }"),
            ("ASK true",
             "ASK { ns:m.02mjmr ns:type.object.type ns:m.0n4gvhd }"),
            ("BIND False/True",
             "SELECT ?x ?reversed WHERE { ns:m.02mjmr ?x ns:en.barack_obama . BIND(False AS ?reversed) } LIMIT 3"),
        ]

        for desc, query in test_queries:
            consistent, msg, odbc_r, sw_r = TestBackendConsistency.run_consistency_test(
                query, odbc_ex, sw_ex)
            status = "✅" if consistent else "❌"
            print(f"  {status} {desc}")
            if not consistent:
                print(f"     {msg}")
                print(f"     ODBC rows: {odbc_r[:3]}")
                print(f"     SW   rows: {sw_r[:3]}")

    sys.exit(0 if result.wasSuccessful() else 1)
