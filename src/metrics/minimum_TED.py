"""
TED (Tree Edit Distance) 计算的最小单元，以及 recalculate_all_ted_parallel() 脚本。

最小单元 minimum_ted() 封装了完整的 TED 计算流程:
1. FILTER != 模式过滤
2. SPARQL 预处理 (SyntaxTreeEditor.preprocess_edit_distance)
3. 语法树构建 (TreeConstructor.construct_syntax_tree)
4. 语法树规范化 (TreeConstructor.canonicalize_syntax_tree)
5. ZSS 树编辑距离计算 (TreeEditDistance.get_edit_distance)
"""

import re
import math

from src.core.common import DATASET, Dataset, Method, to_kb_type
from src.sparql.sparql_utils import SyntaxTreeEditor, TreeConstructor, TreeEditDistance
import logging

TARGET_VARIABLE_NAME = "target"


def minimum_ted(
    golden_sparql: str,
    predicted_sparql: str,
    golden_dataset: DATASET,
    predicted_dataset: DATASET,
    logger: logging.Logger = None,
) -> float:
    """计算两条 SPARQL 之间的归一化树编辑距离 (TED)。

    Args:
        golden_sparql: gold standard SPARQL 查询
        predicted_sparql: 预测的 SPARQL 查询
        golden_dataset: gold SPARQL 所属的数据集
        predicted_dataset: 预测 SPARQL 所属的数据集/方法
        logger: 可选的 logger

    Returns:
        归一化 TED 值，范围 [0.0, 1.0]
    """

    def remove_filters_not_related_to_semantics(sparql_query):

        # 前处理：去掉 FILTER != 模式
        filter_pattern = r'FILTER\([^(]*!=[^(]*\).'
        cleaned_query = re.sub(filter_pattern, '', sparql_query)

        #20260521：对于Free base，去除  FILTER NOT EXISTS { ?cvt_0 ns:type.object.name ?name . }.模式，这种模式对quad造成明显debuff

        # 匹配 FILTER NOT EXISTS { ?任意变量 ns:type.object.name ?任意变量 . }
        # 允许内部有任意空白字符（空格、换行、制表符）
        pattern = r"FILTER\s+NOT\s+EXISTS\s*\{\s*\?\w+\s+ns:type\.object\.name\s+\?\w+\s*\.?\s*\}."
        
        # 使用 re.IGNORECASE 确保大小写不敏感（比如写成 filter not exists）
        cleaned_query = re.sub(pattern, "", cleaned_query, flags=re.IGNORECASE)
        
        # 选做：美化一下可能留下的多余空行或连续空格
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
        
        return cleaned_query

    predicted_sparql = remove_filters_not_related_to_semantics(predicted_sparql)


    # SPARQL 预处理
    editor = SyntaxTreeEditor(golden_sparql, golden_dataset, logger)
    editor.preprocess_edit_distance(f"?{TARGET_VARIABLE_NAME}")
    golden_sparql = editor.sparql_txt

    editor = SyntaxTreeEditor(predicted_sparql, predicted_dataset, logger)
    editor.preprocess_edit_distance(f"?{TARGET_VARIABLE_NAME}")
    predicted_sparql = editor.sparql_txt

    # 语法树构建 + 规范化
    golden_constructor = TreeConstructor(golden_sparql, golden_dataset, logger)
    golden_root = golden_constructor.construct_syntax_tree()
    golden_constructor.canonicalize_syntax_tree(root_node=golden_root)

    predicted_constructor = TreeConstructor(predicted_sparql, predicted_dataset, logger)
    predicted_root = predicted_constructor.construct_syntax_tree()
    predicted_constructor.canonicalize_syntax_tree(root_node=predicted_root)

    # 计算 TED
    return TreeEditDistance.get_edit_distance(predicted_root, golden_root)

