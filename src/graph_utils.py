import networkx as nx
import re
import copy
from enum import Enum
from networkx.readwrite import json_graph

from .common import (
    FreebaseConstantForConstruction, FREEBASE_CONSTANT_TYPE,
    WikidataConstantForConstruction, WIKIDATA_CONSTANT_TYPE
)
from .utils import (load_json)

class SYMBOL_TYPE(Enum):
    ENTITY = 1
    CLASS = 2 
    LITERAL = 3
    RELATION = 4

# 适配其他版本的 S-expression, 例如 GrailQA
OPERATOR_MAPPING = {
    "EQ": "JOIN",
    "ge": "GE",
    'gt': 'GT',
    "le": "LE",
    "lt": "LT"
}

def lisp_to_nested_expression(lisp_string: str) -> list:
    """
    Takes a logical form as a lisp string and returns a nested list representation of the lisp.
    For example, "(count (division first))" would get mapped to ['count', ['division', 'first']].
    """
    stack: list = []
    current_expression: list = []
    tokens = lisp_string.split()
    for token in tokens:
        while token[0] == '(':
            nested_expression: list = []
            current_expression.append(nested_expression)
            stack.append(current_expression)
            current_expression = nested_expression
            token = token[1:]
        current_expression.append(token.replace(')', ''))
        while token[-1] == ')':
            current_expression = stack.pop()
            token = token[:-1]
    return current_expression[0]

def _linearize_lisp_expression(expression: list, sub_formula_id):
    sub_formulas = []
    for i, e in enumerate(expression):
        parent_flag = None
        if (expression[0] in ['LT', 'LE', 'GT', 'GE', 'EQ', 'lt', 'le', 'gt', 'ge', 'JOIN']) and i == 1:
            parent_flag = expression[0]
        elif (expression[0] in ['ARGMIN', 'ARGMAX']) and i == 2:
            parent_flag = expression[0]
        if e[0] == 'JOIN' and parent_flag:
            e[0] = f"{parent_flag}_JOIN"
        
        if isinstance(e, list) and e[0] != 'R':
            sub_formulas.extend(_linearize_lisp_expression(e, sub_formula_id))
            expression[i] = '#' + str(sub_formula_id[0] - 1)

    sub_formulas.append(expression)
    sub_formula_id[0] += 1
    return sub_formulas

def node_match(node1, node2):
    if ('value' in node1) or ('value' in node2):
        if ('value' not in node1) or ('value' not in node2):
            return False
        return node1['value'] == node2['value']

    if ('type' in node1) or ('type' in node2):
        if ('type' not in node1) or ('type' not in node2):
            return False
        return node1['type'] == node2['type']
    
    return True

def edge_match(edge1, edge2):
    if ('value' in edge1) or ('value' in edge2):
        if ('value' not in edge1) or ('value' not in edge2):
            return False
        return edge1['value'] == edge2['value']

    if ('operator' in edge1) or ('operator' in edge2):
        if ('operator' not in edge1) or ('operator' not in edge2):
            return False
        return edge1['operator'] == edge2['operator']
    
    return True

def node_subst_cost(node1, node2):
    """
    节点的替换代价为 1
    """
    if ('value' in node1) or ('value' in node2):
        if ('value' not in node1) or ('value' not in node2):
            return 1
        return 0 if (node1['value'] == node2['value']) else 1

    if ('type' in node1) or ('type' in node2):
        if ('type' not in node1) or ('type' not in node2):
            return 1
        return 0 if (node1['type'] == node2['type']) else 1
    
    return 0

def edge_subst_cost(edge1, edge2):
    if ('value' in edge1) or ('value' in edge2):
        if ('value' not in edge1) or ('value' not in edge2):
            return 1
        return 0 if (edge1['value'] == edge2['value']) else 1

    if ('operator' in edge1) or ('operator' in edge2):
        if ('operator' not in edge1) or ('operator' not in edge2):
            return 1
        return 0 if (edge1['operator'] == edge2['operator']) else 1
    
    return 0

class GraphEquivalenceUtil(object):
    def __init__(
        self,
        logger,
        reverse_property_path="ontology/reverse_properties",
        domain_range_label_path="data/input/common/fb_relations_domain_range_label.json"
    ):
        self.graph_constructor = GraphConstructor.instance(
            logger, reverse_property_path, domain_range_label_path
        )
        self.logger = logger

    @classmethod
    def instance(cls, *args, **kwargs):
        if not hasattr(GraphEquivalenceUtil, "_instance"):
            GraphEquivalenceUtil._instance = GraphEquivalenceUtil(*args, **kwargs)
        return GraphEquivalenceUtil._instance

    def calc_edit_distance(self, simulated_query, golden_query):
        try:
            golden_query = self.dataset_sexp_pre_process(golden_query) # 处理数据集中的 S 表达式的一些格式不同
            simulated_query_graph = self.graph_constructor.logical_form_to_graph(simulated_query)
            golden_query_graph = self.graph_constructor.logical_form_to_graph(golden_query)
            # simulated_graph_serialized = json_graph.node_link_data(simulated_query_graph)
            # golden_graph_serialized = json_graph.node_link_data(golden_query_graph)
            normalized_ged = self.get_normalized_edit_distance(
                simulated_query_graph, golden_query_graph
            )
            return normalized_ged
        except Exception as e:
            self.logger.error(f"exception: {e}; simulated_query: {simulated_query}; golden_query: {golden_query}")
            return 1.0 # 最大值
    
    def calc_edit_distance_comparing_methods(self, simulated_query, golden_query):
        """
        一些对比方法构造的 simulated query 也是旧格式的
        """
        try:
            simulated_query = self.dataset_sexp_pre_process(simulated_query)
            golden_query = self.dataset_sexp_pre_process(golden_query) # 处理数据集中的 S 表达式的一些格式不同
            simulated_query_graph = self.graph_constructor.logical_form_to_graph(simulated_query)
            golden_query_graph = self.graph_constructor.logical_form_to_graph(golden_query)
            normalized_ged = self.get_normalized_edit_distance(
                simulated_query_graph, golden_query_graph
            )
            return normalized_ged
        except Exception as e:
            self.logger.error(f"exception: {e}; simulated_query: {simulated_query}; golden_query: {golden_query}")
            return 1.0 # 最大值

    def get_normalized_edit_distance(self, simulated_query_graph, golden_query_graph):
        '''
        默认情况下，graph_edit_distance 中
        - node_del_cost = 1
        - node_ins_cost = 1
        - edge_del_cost = 1
        - edge_ins_cost = 1
        '''
        empty_graph = nx.DiGraph()
        # |G_s|
        simulated_size = nx.graph_edit_distance(
            empty_graph, simulated_query_graph,
            node_subst_cost=node_subst_cost,
            edge_subst_cost=edge_subst_cost,
            timeout=5
        )
        # |G_g|
        golden_size = nx.graph_edit_distance(
            empty_graph, golden_query_graph,
            node_subst_cost=node_subst_cost,
            edge_subst_cost=edge_subst_cost,
            timeout=5
        )
        edit_distance = nx.graph_edit_distance(
            simulated_query_graph, golden_query_graph,
            node_subst_cost=node_subst_cost,
            edge_subst_cost=edge_subst_cost,
            timeout=5
        )
        if (not simulated_size) or (not golden_size):
            # 出错了，返回最大值 1.0
            return 1.0
        # (edit_distance) / max(|G_s|, |G_g|)
        return edit_distance / (max(simulated_size, golden_size))

    def dataset_sexp_pre_process(self, sexp):
        '''
        GrailQA 给出的 S-expression 中, AND 和 ARGMAX / ARGMIN 后面可能跟着一个 class, 这里做个处理
        '''
        new_tokens = list()
        sexp = sexp.replace('(', ' ( ')
        sexp = sexp.replace(')', ' ) ')
        tokens = sexp.split()
        tokens = [x for x in tokens if len(x)]
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            tok = tok.strip()
            tok = self.process_literal_in_dataset(tok)
            try:
                if tok in ['AND', 'ARGMIN', 'ARGMAX']:
                    if re.fullmatch("[a-zA-Z_]+\.[a-zA-Z_]+", tokens[i+1]):
                        new_tokens.append(tok)
                        new_tokens.extend(['(', 'JOIN', 'type.object.type', tokens[i+1], ')'])
                        i += 2
                        continue
            except:
                pass
            new_tokens.append(tok)
            i += 1
        
        new_sexp = ""
        for (idx, tok) in enumerate(new_tokens):
            if idx == 0 or (new_tokens[idx-1] == '(') or (tok == ')'):
                new_sexp = f"{new_sexp}{tok}"
            else:
                new_sexp = f"{new_sexp} {tok}"
        return new_sexp
    
    def simulated_sexp_pre_process(self, sexp):
        '''
        仅将其中的 EQ 替换成 JOIN
        '''
        new_tokens = list()
        sexp = sexp.replace('(', ' ( ')
        sexp = sexp.replace(')', ' ) ')
        tokens = sexp.split()
        tokens = [x for x in tokens if len(x)]
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            tok = tok.strip()
            if tok == 'EQ':
                new_tokens.append("JOIN")
            else:
                new_tokens.append(tok)
            i += 1
        
        new_sexp = ""
        for (idx, tok) in enumerate(new_tokens):
            if idx == 0 or (new_tokens[idx-1] == '(') or (tok == ')'):
                new_sexp = f"{new_sexp}{tok}"
            else:
                new_sexp = f"{new_sexp} {tok}"
        return new_sexp
    
    def process_literal_in_dataset(self, literal):
        '''数据集中的 literal 可能存在格式差别，对此我们做个替换（旧的格式改成新的）'''
        if "^^http://www.w3.org/2001/XMLSchema" in literal:
            return f'"{literal.split("^^")[0]}"^^<{literal.split("^^")[1]}>'
        elif literal.endswith("@en"):
            return literal
        elif literal.startswith('"') and literal.endswith('"'):
            return f"{literal}@en" 
        else:
            try:
                value = float(literal)
                return f'"{value}"^^<http://www.w3.org/2001/XMLSchema#float>'
            except Exception:
                return literal


class GraphConstructor(object):
    def __init__(
        self,
        logger,
        reverse_property_path="ontology/reverse_properties",
        domain_range_label_path="data/input/common/fb_relations_domain_range_label.json"
    ):
        self.logger = logger
        self.reverse_properties = {}

        with open(reverse_property_path, 'r') as f:
            for line in f:
                self.reverse_properties[line.split('\t')[0]] = line.split('\t')[1].replace('\n', '')
        
        self.relation_domain_range = load_json(domain_range_label_path)

    @classmethod
    def instance(cls, *args, **kwargs):
        if not hasattr(GraphConstructor, "_instance"):
            GraphConstructor._instance = GraphConstructor(*args, **kwargs)
        return GraphConstructor._instance
    
    def get_symbol_type(self, symbol):
        symbol_type = FreebaseConstantForConstruction.get_constant_type(symbol)
        if symbol_type is FREEBASE_CONSTANT_TYPE.ENTITY:
            return SYMBOL_TYPE.ENTITY
        elif symbol_type is FREEBASE_CONSTANT_TYPE.CLASS:
            return SYMBOL_TYPE.CLASS
        elif symbol_type in [FREEBASE_CONSTANT_TYPE.QUANTITY, FREEBASE_CONSTANT_TYPE.TIME, FREEBASE_CONSTANT_TYPE.STRING]:
            return SYMBOL_TYPE.LITERAL
        elif re.fullmatch("[a-zA-Z_]+\.[a-zA-Z_]+\.[a-zA-z_]+", symbol) or re.fullmatch("[a-zA-Z_]+\.[a-zA-Z_]+\.[a-zA-z_]+\.[a-zA-z_]+", symbol) or re.fullmatch("[a-zA-Z_]+\.[a-zA-Z_]+\.[a-zA-z_]+\.[a-zA-z_]+\.[a-zA-z_]+", symbol):
            return SYMBOL_TYPE.RELATION
        else:
            return None    

    def _add_type(self, node, value):
        if 'type' in node:
            node['type'].add(value)
        else:
            node['type'] = set({value})                                                                                                                        
    
    def logical_form_to_graph(self, lisp_program:list) -> nx.DiGraph:
        '''
        边的 attribute
        - value: Freebase 关系
        - operator: 函数名称

        节点的 attribute
        - value: 知识库常量 或 "dummy"
        - type: set()

        每个子图中，编号最小（0）的节点始终表示目标变量
        中间变量的 attribute (应该只有 type) 也需要维护；建图的过程中可能无法确定谁才是最终的目标变量
        '''
        expression = lisp_to_nested_expression(lisp_program)
        sub_programs = _linearize_lisp_expression(expression, [0])
        identical_index_r = {}
        idx2graph = {}
        target_idx = len(sub_programs) - 1 # 最终的 target variable 会位于的位置

        def get_root(idx):
            while idx in identical_index_r:
                idx = identical_index_r[idx]
            return idx
        
        for i, subp in enumerate(sub_programs):
            i = str(i)
            G = nx.DiGraph()
            G.add_node(0) # 目标变量节点
            if subp[0] == 'JOIN':
                '''
                subp[1] 我认为只有两种选择:
                - relation
                - R relation
                这两者只有 SPARQL 里面三元组方向的区别

                无论 subp[1] 是什么, subp[2] 有如下选择
                - item: entity / class / literal (对于旧版的 S-expression, TIME 和 QUANTITY 也可能出现在这个位置)
                - #n: 表示一个嵌套的子结构
                - relation 或者 R relation
                '''                
                if subp[2].startswith('#'): # 是一个子成分，需要完成图的合并
                    '''
                    将 subp[2] 所表示的子图的目标变量变为中间变量；新增一个目标变量，通过关系 subp[1] 连接目标和中间变量
                    '''
                    root2 = get_root(int(subp[2][1:]))
                    sub_graph:nx.DiGraph = idx2graph[root2]
                    sub_graph_relabel_mapping = {}
                    for n in sub_graph.nodes():
                        sub_graph_relabel_mapping[n] = n + 1
                    G:nx.DiGraph = nx.relabel_nodes(sub_graph, sub_graph_relabel_mapping, copy=True) # copy = True 返回一个拷贝
                    G.add_node(0) # 新的变量节点
                    if isinstance(subp[1], list): # R relation
                        relation = subp[1][1]
                        G.add_edge(1, 0, value=relation)
                        relation_r = self.reverse_properties.get(relation, None)
                        if relation_r:
                            G.add_edge(0, 1, value=relation_r)
                        if relation in self.relation_domain_range:
                            if "domain" in self.relation_domain_range[relation]:
                                domain = self.relation_domain_range[relation]["domain"]
                                self._add_type(G.nodes[1], domain)
                            if "range" in self.relation_domain_range[relation]:
                                range = self.relation_domain_range[relation]["range"]
                                self._add_type(G.nodes[0], range)

                    elif isinstance(subp[1], str): # relation
                        relation = subp[1]
                        G.add_edge(0, 1, value=relation)
                        relation_r = self.reverse_properties.get(relation, None)
                        if relation_r:
                            G.add_edge(1, 0, value=relation_r)
                        if relation in self.relation_domain_range:
                            if "domain" in self.relation_domain_range[relation]:
                                domain = self.relation_domain_range[relation]["domain"]
                                self._add_type(G.nodes[0], domain)
                            if "range" in self.relation_domain_range[relation]:
                                range = self.relation_domain_range[relation]["range"]
                                self._add_type(G.nodes[1], range)
                
                elif self.get_symbol_type(subp[2]) in [SYMBOL_TYPE.ENTITY, SYMBOL_TYPE.LITERAL]:
                    G.add_node(len(G.nodes()), value=subp[2]) # 指向常量的节点
                    if isinstance(subp[1], list): # R relation
                        relation = subp[1][1]
                        G.add_edge(len(G.nodes()) - 1, 0, value=relation)
                        relation_r = self.reverse_properties.get(relation, None)
                        if relation_r:
                            G.add_edge(0, len(G.nodes()) - 1, value=relation_r)
                        if relation in self.relation_domain_range:
                            '''常量节点 不考虑 'type' 属性了'''
                            if "range" in self.relation_domain_range[relation]:
                                range = self.relation_domain_range[relation]["range"]
                                self._add_type(G.nodes[0], range)
                    elif isinstance(subp[1], str): # relation
                        relation = subp[1]
                        G.add_edge(0, len(G.nodes()) - 1, value=relation)
                        relation_r = self.reverse_properties.get(relation, None)
                        if relation_r:
                            G.add_edge(len(G.nodes()) - 1, 0, value=relation_r)
                        if relation in self.relation_domain_range:
                            '''常量节点 不考虑 'type' 属性了'''
                            if "domain" in self.relation_domain_range[relation]:
                                domain = self.relation_domain_range[relation]["domain"]
                                self._add_type(G.nodes[0], domain)
                    else:
                        raise Exception(f"subp: {subp}")
                elif self.get_symbol_type(subp[2]) is SYMBOL_TYPE.CLASS:
                    if subp[1] not in [
                        'type.object.type', 
                        ['R', 'type.type.instance'],
                        'kg.object_profile.prominent_type',
                        'common.topic.notable_types'
                    ]:
                        raise Exception(f"subp: {subp}")
                    # 仅对于目标变量节点，添加类型约束
                    self._add_type(G.nodes[0], subp[2])
                else:
                    raise Exception(f"subp: {subp}")
                
                # graph_serialized = json_graph.node_link_data(G)
            
            elif subp[0] in ['EQ', 'LT', 'LE', 'GT', 'GE', 'lt', 'le', 'gt', 'ge']:
                '''
                subp[1]:
                    - 嵌套结构, #n
                    - 关系 / 逆关系
                subp[2]:
                    - 嵌套结构, #n
                    - time / number
                '''
                operator = subp[0]
                if operator in OPERATOR_MAPPING:
                    operator = OPERATOR_MAPPING[operator]
                if subp[1].startswith('#'):
                    # subp[1] 是一个多跳关系；从 subp[1] 子图的最后一个点出发，加一个函数边
                    var1 = int(subp[1][1:])
                    rooti = get_root(int(i))
                    root1 = get_root(var1)
                    if rooti > root1:
                        identical_index_r[rooti] = root1
                    else:
                        identical_index_r[root1] = rooti
                    
                    if subp[2].startswith('#'): # 嵌套
                        root2 = get_root(int(subp[2][1:]))
                        graph_1 = idx2graph[root1]
                        graph_2 = idx2graph[root2]
                        G = self.combine_compare_subgraphs(operator, graph_1, graph_2)
                    elif self.get_symbol_type(subp[2]) is SYMBOL_TYPE.LITERAL: # TIME / NUMBER
                        sub_graph:nx.DiGraph = idx2graph[root1]
                        # sub_graph_serialized = json_graph.node_link_data(sub_graph)
                        G = sub_graph.copy()
                        if operator == "JOIN": # 直接将中间变量替换为 grounded item
                            G.nodes[len(G.nodes()) - 1]['value'] = subp[2]
                        else:
                            G.add_node(len(G.nodes()), value=subp[2])
                            G.add_edge(len(G.nodes()) - 2, len(G.nodes()) - 1, operator=operator)
                    else:
                        raise Exception(f"subp: {subp}")
                elif isinstance(subp[1], list): # R relation
                    if subp[2].startswith('#'): # 嵌套
                        '''新增从 target variable 指向 subp[2] 头结点的一条边'''
                        root2 = get_root(int(subp[2][1:]))
                        subgraph_2 = idx2graph[root2]
                        # subgraph_2_serialized = json_graph.node_link_data(subgraph_2)
                        graph_2_mapping = {}
                        if operator == "JOIN": # 没有函数边
                            for n in subgraph_2.nodes():
                                graph_2_mapping[n] = n + 1
                            subgraph_2 = nx.relabel_nodes(subgraph_2, graph_2_mapping, copy=True)
                            G = nx.compose(G, subgraph_2) # G 本身有个目标变量节点
                            relation = subp[1][1]
                            G.add_edge(1, 0, value=relation)
                            relation_r = self.reverse_properties.get(relation, None)
                            if relation_r:
                                G.add_edge(0, 1, value=relation_r)
                            if relation in self.relation_domain_range:
                                if "range" in self.relation_domain_range[relation]:
                                    range = self.relation_domain_range[relation]["range"]
                                    self._add_type(G.nodes[0], range)
                                if "domain" in self.relation_domain_range[relation]:
                                    domain = self.relation_domain_range[relation]["domain"]
                                    self._add_type(G.nodes[1], domain)
                        else: # 多一条函数边
                            for n in subgraph_2.nodes():
                                graph_2_mapping[n] = n + 2
                            subgraph_2 = nx.relabel_nodes(subgraph_2, graph_2_mapping, copy=True)
                            G = nx.compose(G, subgraph_2) # G 本身有个目标变量节点
                            G.add_node(1) # 中间变量
                            relation = subp[1][1]
                            G.add_edge(1, 0, value=relation)
                            relation_r = self.reverse_properties.get(relation, None)
                            if relation_r:
                                G.add_edge(0, 1, value=relation_r)
                            if relation in self.relation_domain_range:
                                if "range" in self.relation_domain_range[relation]:
                                    range = self.relation_domain_range[relation]["range"]
                                    self._add_type(G.nodes[0], range)
                                if "domain" in self.relation_domain_range[relation]:
                                    domain = self.relation_domain_range[relation]["domain"]
                                    self._add_type(G.nodes[1], domain)
                            G.add_edge(1, 2, operator=operator)
                        
                    elif self.get_symbol_type(subp[2]) is SYMBOL_TYPE.LITERAL: # TIME / NUMBER
                        # G_serialized = json_graph.node_link_data(G)
                        if operator == "JOIN": # 没有函数边
                            G.add_node(len(G.nodes()), value=subp[2]) # grounded item
                            G.add_edge(len(G.nodes()) - 1, 0, value=relation)
                            relation_r = self.reverse_properties.get(relation, None)
                            if relation_r:
                                G.add_edge(0, len(G.nodes()) - 1, value=relation_r)
                            if relation in self.relation_domain_range:
                                '''len(G.nodes()) - 1 已经有 value 了，就不添加 type 信息'''
                                if "range" in self.relation_domain_range[relation]:
                                    range = self.relation_domain_range[relation]["range"]
                                    self._add_type(G.nodes[0], range)
                        else: # 多了一条函数边
                            G.add_node(len(G.nodes())) # 中间变量
                            G.add_node(len(G.nodes()), value=subp[2]) # grounded item
                            G.add_edge(len(G.nodes()) - 2, len(G.nodes()) - 1, operator=operator)
                            relation = subp[1][1]
                            G.add_edge(len(G.nodes()) - 2, 0, value=relation)
                            relation_r = self.reverse_properties.get(relation, None)
                            if relation_r:
                                G.add_edge(0, len(G.nodes()) - 2, value=relation_r)
                            if relation in self.relation_domain_range:
                                if "range" in self.relation_domain_range[relation]:
                                    range = self.relation_domain_range[relation]["range"]
                                    self._add_type(G.nodes[0], range)
                                if "domain" in self.relation_domain_range[relation]:
                                    domain = self.relation_domain_range[relation]["domain"]
                                    self._add_type(G.nodes[len(G.nodes()) - 2], domain)
                    else:
                        raise Exception(f"subp: {subp}")
                elif isinstance(subp[1], str): # relation
                    if subp[2].startswith('#'): # 嵌套
                        '''新增从 target variable 指向 subp[2] 头结点的一条边'''
                        root2 = get_root(int(subp[2][1:]))
                        subgraph_2 = idx2graph[root2]
                        # subgraph_2_serialized = json_graph.node_link_data(subgraph_2)
                        graph_2_mapping = {}
                        if operator == "JOIN": # 没有函数边
                            for n in subgraph_2.nodes():
                                graph_2_mapping[n] = n + 1
                            subgraph_2 = nx.relabel_nodes(subgraph_2, graph_2_mapping, copy=True)
                            G = nx.compose(G, subgraph_2)
                            relation = subp[1]
                            G.add_edge(0, 1, value=relation)
                            relation_r = self.reverse_properties.get(relation, None)
                            if relation_r:
                                G.add_edge(1, 0, value=relation_r)
                            if relation in self.relation_domain_range:
                                if "domain" in self.relation_domain_range[relation]:
                                    domain = self.relation_domain_range[relation]["domain"]
                                    self._add_type(G.nodes[0], domain)
                                if "range" in self.relation_domain_range[relation]:
                                    range = self.relation_domain_range[relation]["range"]
                                    self._add_type(G.nodes[1], range)
                        else:
                            for n in subgraph_2.nodes():
                                graph_2_mapping[n] = n + 2
                            subgraph_2 = nx.relabel_nodes(subgraph_2, graph_2_mapping, copy=True)
                            G = nx.compose(G, subgraph_2)
                            G.add_node(1) # 中间变量
                            relation = subp[1]
                            G.add_edge(0, 1, value=relation)
                            relation_r = self.reverse_properties.get(relation, None)
                            if relation_r:
                                G.add_edge(1, 0, value=relation_r)
                            if relation in self.relation_domain_range:
                                if "domain" in self.relation_domain_range[relation]:
                                    domain = self.relation_domain_range[relation]["domain"]
                                    self._add_type(G.nodes[0], domain)
                                if "range" in self.relation_domain_range[relation]:
                                    range = self.relation_domain_range[relation]["range"]
                                    self._add_type(G.nodes[1], range)
                            G.add_edge(1, 2, operator=operator)
                    elif self.get_symbol_type(subp[2]) is SYMBOL_TYPE.LITERAL: # TIME / NUMBER
                        if operator == "JOIN": # 没有函数边
                            G.add_node(len(G.nodes()), value=subp[2]) # grounded item
                            relation = subp[1]
                            G.add_edge(0, len(G.nodes()) - 1, value=relation)
                            relation_r = self.reverse_properties.get(relation, None)
                            if relation_r:
                                G.add_edge(len(G.nodes()) - 1, 0, value=relation_r)
                            if relation in self.relation_domain_range:
                                '''len(G.nodes()) - 1 已经有 value 属性了，就不管其 domain'''
                                if "domain" in self.relation_domain_range[relation]:
                                    domain = self.relation_domain_range[relation]["domain"]
                                    self._add_type(G.nodes[0], domain)
                        else:
                            G.add_node(len(G.nodes())) # 中间变量
                            G.add_node(len(G.nodes()), value=subp[2]) # grounded item
                            G.add_edge(len(G.nodes()) - 2, len(G.nodes()) - 1, operator=operator)
                            relation = subp[1]
                            G.add_edge(0, len(G.nodes()) - 2, value=relation)
                            relation_r = self.reverse_properties.get(relation, None)
                            if relation_r:
                                G.add_edge(len(G.nodes()) - 2, 0, value=relation_r)
                            if relation in self.relation_domain_range:
                                if "domain" in self.relation_domain_range[relation]:
                                    domain = self.relation_domain_range[relation]["domain"]
                                    self._add_type(G.nodes[0], domain)
                                if "range" in self.relation_domain_range[relation]:
                                    range = self.relation_domain_range[relation]["range"]
                                    self._add_type(G.nodes[len(G.nodes()) - 2], range)
                    else:
                        raise Exception(f"subp: {subp}")
                else:
                    raise Exception(f"subp: {subp}")
            
            elif subp[0] in ["ARGMIN", "ARGMAX"]:
                '''
                subp[1]: #n
                subp[2]:
                    - relation
                    - R relation
                    - #n, 多跳关系构成的子成分
                
                subp[1] 的目标变量变为中间变量，通过 subp[2] 中的关系（可能是多跳的）连向真正的目标变量
                '''
                if subp[1][0] == '#':
                    var1 = int(subp[1][1:])
                    rooti = get_root(int(i))
                    root1 = get_root(var1)
                    if rooti > root1:
                        identical_index_r[rooti] = root1
                    else:
                        identical_index_r[root1] = rooti
                        root1 = rooti
                    
                    sub_graph_1:nx.DiGraph = idx2graph[root1]
                    if subp[2][0] == '#':
                        var2 = int(subp[2][1:])
                        root2 = get_root(var2)
                        sub_graph_2:nx.DiGraph = copy.deepcopy(idx2graph[root2])
                        if root1 > root2:
                            identical_index_r[root1] = root2
                        else:
                            identical_index_r[root2] = root1
                            root2 = root1
                        # sub_graph_1_serialized = json_graph.node_link_data(sub_graph_1)
                        # sub_graph_2_serialized = json_graph.node_link_data(sub_graph_2)
                        sub_graph_2_len = len(sub_graph_2.nodes())
                        sub_graph_1_target_variable = copy.deepcopy(sub_graph_1.nodes[0])
                        sub_graph_2_target_variable = copy.deepcopy(sub_graph_2.nodes[0])
                        sub_graph_relabel_mapping = {}
                        # type_union 一定要在 relabel 之前获取!
                        type_union = sub_graph_1.nodes[0].get('type', set()) | sub_graph_2.nodes[0].get('type', set())
                        for n in sub_graph_1.nodes():
                            if n != 0:
                                sub_graph_relabel_mapping[n] = n + sub_graph_2_len - 1
                        nx.relabel_nodes(sub_graph_1, sub_graph_relabel_mapping, copy=False) # copy=False 原地修改
                        # 需要对目标节点进行合并
                        G = nx.compose(sub_graph_1, sub_graph_2)
                        if len(type_union) > 0:
                            if 'type' in G.nodes[0]:
                                G.nodes[0]['type'].update(type_union)
                            else:
                                G.nodes[0]['type'] = type_union
                        G.add_edge(sub_graph_2_len - 1, len(G.nodes()), operator=subp[0])
                        # G_serialized = json_graph.node_link_data(G)

                    elif isinstance(subp[2], list): # R relation
                        sub_graph_1_relabel_mapping = {}
                        for n in sub_graph_1.nodes():
                            sub_graph_1_relabel_mapping[n] = n + 2
                        G = nx.relabel_nodes(sub_graph_1, sub_graph_1_relabel_mapping, copy=True)
                        # G.add_node(len(G.nodes()))
                        relation = subp[2][1]
                        G.add_edge(1, 2, value=relation)
                        relation_r = self.reverse_properties.get(relation, None)
                        if relation_r:
                            G.add_edge(2, 1, value=relation_r)
                        if relation in self.relation_domain_range:
                            if "range" in self.relation_domain_range[relation]:
                                range = self.relation_domain_range[relation]["range"]
                                self._add_type(G.nodes[2], range)
                            if "domain" in self.relation_domain_range[relation]:
                                domain = self.relation_domain_range[relation]["domain"]
                                self._add_type(G.nodes[1], domain)
                        G.add_edge(1, 0, operator=subp[0])
                        # G_serialized = json_graph.node_link_data(G)
                    elif isinstance(subp[2], str): # relation
                        sub_graph_1_relabel_mapping = {}
                        for n in sub_graph_1.nodes():
                            sub_graph_1_relabel_mapping[n] = n + 2
                        G = nx.relabel_nodes(sub_graph_1, sub_graph_1_relabel_mapping, copy=True)
                        # G.add_node(len(G.nodes()))
                        relation = subp[2]
                        G.add_edge(2, 1, value=relation)
                        relation_r = self.reverse_properties.get(relation, None)
                        if relation_r:
                            G.add_edge(1, 2, value=relation_r)
                        if relation in self.relation_domain_range:
                            if "domain" in self.relation_domain_range[relation]:
                                domain = self.relation_domain_range[relation]["domain"]
                                self._add_type(G.nodes[2], domain)
                            if "range" in self.relation_domain_range[relation]:
                                range = self.relation_domain_range[relation]["range"]
                                self._add_type(G.nodes[1], range)
                        G.add_edge(1, 0, operator=subp[0])
                        # G_serialized = json_graph.node_link_data(G)
                else:
                    raise Exception(f"subp: {subp}")

            elif subp[0] == 'AND':
                var1 = int(subp[1][1:])
                rooti = get_root(int(i))
                root1 = get_root(var1)
                if rooti > root1:
                    identical_index_r[rooti] = root1
                else:
                    identical_index_r[root1] = rooti
                    root1 = rooti
                var2 = int(subp[2][1:])
                root2 = get_root(var2)
                if root1 > root2:
                    identical_index_r[root1] = root2
                else:
                    identical_index_r[root2] = root1
                
                sub_graph_1:nx.DiGraph = copy.deepcopy(idx2graph[root1])
                sub_graph_2:nx.DiGraph = copy.deepcopy(idx2graph[root2])
                # sub_graph_1_serialized = json_graph.node_link_data(sub_graph_1)
                # sub_graph_2_serialized = json_graph.node_link_data(sub_graph_2)
                sub_graph_relabel_mapping = {}
                # type_union 一定要在 relabel 之前获取!
                type_union = sub_graph_1.nodes[0].get('type', set()) | sub_graph_2.nodes[0].get('type', set())
                for n in sub_graph_2.nodes():
                    if n != 0:
                        sub_graph_relabel_mapping[n] = n + len(sub_graph_1.nodes()) - 1
                nx.relabel_nodes(sub_graph_2, sub_graph_relabel_mapping, copy=False)
                G = nx.compose(sub_graph_1, sub_graph_2)
                if len(type_union) > 0:
                    if 'type' in G.nodes[0]:
                        G.nodes[0]['type'].update(type_union)
                    else:
                        G.nodes[0]['type'] = type_union
                # G_serialized = json_graph.node_link_data(G)
            elif subp[0] == 'COUNT':
                idx = get_root(int(subp[1][1:]))
                G:nx.DiGraph = copy.deepcopy(idx2graph[idx])
                sub_graph_relabel_mapping = {}
                for n in G.nodes():
                    sub_graph_relabel_mapping[n] = n + 1
                nx.relabel_nodes(G, sub_graph_relabel_mapping, copy=False)
                G.add_node(0)
                G.add_edge(1, 0, value=subp[0])
            
            elif subp[0] in ['LT_JOIN', 'LE_JOIN', "GT_JOIN", "GE_JOIN", "EQ_JOIN", "ARGMIN_JOIN", "ARGMAX_JOIN", 'lt_JOIN', 'le_JOIN', 'gt_JOIN', 'ge_JOIN', 'JOIN_JOIN']:
                '''
                目标变量 --关系边--> 中间变量1 --关系边--> 中间变量2
                主要考虑关系的方向; 注意要给目标变量加个类型约束
                '''
                G.add_node(len(G.nodes())) # 中间变量1
                G.add_node(len(G.nodes())) # 中间变量2
                if isinstance(subp[1], str):
                    relation = subp[1]
                    G.add_edge(0, 1, value=relation)
                    relation_r = self.reverse_properties.get(relation, None)
                    if relation_r:
                        G.add_edge(1, 0, value=relation_r)
                    if relation in self.relation_domain_range:
                        if "domain" in self.relation_domain_range[relation]:
                            domain = self.relation_domain_range[relation]["domain"]
                            self._add_type(G.nodes[0], domain)
                        if "range" in self.relation_domain_range[relation]:
                            range = self.relation_domain_range[relation]["range"]
                            self._add_type(G.nodes[1], range)
                elif isinstance(subp[1], list):
                    relation = subp[1][1]
                    G.add_edge(1, 0, value=relation)
                    relation_r = self.reverse_properties.get(relation, None)
                    if relation_r:
                        G.add_edge(0, 1, value=relation_r)
                    if relation in self.relation_domain_range:
                        if "range" in self.relation_domain_range[relation]:
                            range = self.relation_domain_range[relation]["range"]
                            self._add_type(G.nodes[0], range)
                        if "domain" in self.relation_domain_range[relation]:
                            domain = self.relation_domain_range[relation]["domain"]
                            self._add_type(G.nodes[1], domain)
                else:
                    raise Exception(f"subp: {subp}")
                
                if isinstance(subp[2], str):
                    relation = subp[2]
                    G.add_edge(1, 2, value=relation)
                    relation_r = self.reverse_properties.get(relation, None)
                    if relation_r:
                        G.add_edge(2, 1, value=relation_r)
                    if "domain" in self.relation_domain_range[relation]:
                        domain = self.relation_domain_range[relation]["domain"]
                        self._add_type(G.nodes[1], domain)
                    if "range" in self.relation_domain_range[relation]:
                        range = self.relation_domain_range[relation]["range"]
                        self._add_type(G.nodes[2], range)
                elif isinstance(subp[2], list):
                    relation = subp[2][1]
                    G.add_edge(2, 1, value=relation)
                    relation_r = self.reverse_properties.get(relation, None)
                    if relation_r:
                        G.add_edge(1, 2, value=relation_r)
                    if "domain" in self.relation_domain_range[relation]:
                        domain = self.relation_domain_range[relation]["domain"]
                        self._add_type(G.nodes[2], domain)
                    if "range" in self.relation_domain_range[relation]:
                        range = self.relation_domain_range[relation]["range"]
                        self._add_type(G.nodes[1], range)
                else:
                    raise Exception(f"subp: {subp}")
            

            else: # 我们不考虑 TC, 遇到了就抛出异常，说我们处理不了吧
                raise Exception(f"subp[0]: {subp[0]}; subp: {subp}")
            
            # G_serialized = json_graph.node_link_data(G)
            idx2graph[get_root(int(i))] = copy.deepcopy(G)
            
        
        return copy.deepcopy(idx2graph[get_root(target_idx)])

    def combine_compare_subgraphs(self, operator, graph_1:nx.DiGraph(), graph_2:nx.DiGraph):
        '''
        EQ, LT, LE, GT, GE 这些比较级操作符后面跟着两个子图的情况
        #2 的 target variable 作为 #1 的 grounded_item, 保留 #1 的 target variable 不动
        实现上先对 # 2 进行 relabel, 然后合并 #1 和 #2, 最后 #1 的末尾节点添加一个指向 #2 target_variable 的函数边

        特殊情况为 operator == "JOIN"
        - 此时把 #1 的末尾节点替换成 #2 的 target variable, 保留边的原数据即可 --> 不知道 nx.compose 能行不
        '''
        graph_1_size = len(graph_1.nodes())
        graph_2_mapping = {}
        # graph_1_serialized = json_graph.node_link_data(graph_1)
        # graph_2_serialized = json_graph.node_link_data(graph_2)
        if operator == "JOIN":
            for n in graph_2.nodes():
                graph_2_mapping[n] = n + graph_1_size - 1
            new_node_value, new_node_type = self.combine_node(graph_1.nodes[len(graph_1.nodes()) - 1], graph_2.nodes[0])
            graph_2 = nx.relabel_nodes(graph_2, graph_2_mapping, copy=True)
            combined_graph:nx.DiGraph = nx.compose(graph_1, graph_2)
            combined_graph.nodes[graph_1.nodes[len(graph_1.nodes()) - 1]]["value"] = new_node_value
            combined_graph.nodes[graph_1.nodes[len(graph_1.nodes()) - 1]]["type"] = new_node_type
        else:
            for n in graph_2.nodes():
                graph_2_mapping[n] = n + graph_1_size
            graph_2 = nx.relabel_nodes(graph_2, graph_2_mapping, copy=True)
            combined_graph:nx.DiGraph = nx.compose(graph_1, graph_2)
            combined_graph.add_edge(graph_1_size - 1, graph_1_size, operator=operator)
        
        # combined_graph_serialized = json_graph.node_link_data(combined_graph)
        return combined_graph

    def combine_node(self, node_1, node_2):
        """
        合并两个目标节点
        - value
        - type: 合并即可
        """
        value = None
        if ('value' in node_1) and ('value' in node_2):
            if node_1['value'] != node_2['value']:
                raise Exception(f"node_1: {node_1}; node_2: {node_2}")
            else:
                value = node_1['value']
        elif 'value' in node_1:
            value = node_1['value']
        elif 'value' in node_2:
            value = node_2['value']
        else:
            pass

        type_set = set()
        if 'type' in node_1:
            type_set.update(node_1['type'])
        if 'type' in node_2:
            type_set.update(node_2['type'])

        return value, type_set


class GraphEquivalenceUtilWikidata(object):
    def __init__(
        self,
        logger,
        reverse_property_path="data/input/common/reverse_properties_wikidata",
    ):
        self.graph_constructor = GraphConstructorWikidata.instance(
            logger, reverse_property_path
        )
        self.logger = logger

    @classmethod
    def instance(cls, *args, **kwargs):
        if not hasattr(GraphEquivalenceUtilWikidata, "_instance"):
            GraphEquivalenceUtilWikidata._instance = GraphEquivalenceUtilWikidata(*args, **kwargs)
        return GraphEquivalenceUtilWikidata._instance

    def calc_edit_distance(self, simulated_query, golden_query):
        # try:
        #     # Wikidata 这边不需要做查询格式的修改；数据集中的 S-expression 是我们自己 parsing 的，格式与 Simulated Query 一致
        #     simulated_query_graph = self.graph_constructor.logical_form_to_graph(simulated_query)
        #     golden_query_graph = self.graph_constructor.logical_form_to_graph(golden_query)
        #     normalized_ged = self.get_normalized_edit_distance(
        #         simulated_query_graph, golden_query_graph
        #     )
        #     return normalized_ged
        # except Exception as e:
        #     self.logger.error(f"exception: {e}; simulated_query: {simulated_query}; golden_query: {golden_query}")
        #     return 1.0 # 最大值
        try:
            simulated_query_graph = self.graph_constructor.logical_form_to_graph(simulated_query)
            golden_query_graph = self.graph_constructor.logical_form_to_graph(golden_query)
            normalized_ged = self.get_normalized_edit_distance(
                simulated_query_graph, golden_query_graph
            )
            return normalized_ged
        except Exception as e:
            self.logger.error(f"exception: {e}; simulated_query: {simulated_query}; golden_query: {golden_query}")
            return 1.0 # 最大值
    
    def get_normalized_edit_distance(self, simulated_query_graph, golden_query_graph):
        '''
        默认情况下，graph_edit_distance 中
        - node_del_cost = 1
        - node_ins_cost = 1
        - edge_del_cost = 1
        - edge_ins_cost = 1
        '''
        empty_graph = nx.DiGraph()
        # |G_s|
        simulated_size = nx.graph_edit_distance(
            empty_graph, simulated_query_graph,
            node_subst_cost=node_subst_cost,
            edge_subst_cost=edge_subst_cost,
            timeout=5
        )
        # |G_g|
        golden_size = nx.graph_edit_distance(
            empty_graph, golden_query_graph,
            node_subst_cost=node_subst_cost,
            edge_subst_cost=edge_subst_cost,
            timeout=5
        )
        edit_distance = nx.graph_edit_distance(
            simulated_query_graph, golden_query_graph,
            node_subst_cost=node_subst_cost,
            edge_subst_cost=edge_subst_cost,
            timeout=5
        )
        if (not simulated_size) or (not golden_size):
            # 出错了，返回最大值 1.0
            return 1.0
        # (edit_distance) / max(|G_s|, |G_g|)
        return edit_distance / (max(simulated_size, golden_size))

    def process_literal_in_dataset(self, literal):
        '''数据集中的 literal 可能存在格式差别，对此我们做个替换（旧的格式改成新的）'''
        if "^^http://www.w3.org/2001/XMLSchema" in literal:
            return f'"{literal.split("^^")[0]}"^^<{literal.split("^^")[1]}>'
        elif literal.endswith("@en"):
            return literal
        elif literal.startswith('"') and literal.endswith('"'):
            return f"{literal}@en" 
        else:
            try:
                value = float(literal)
                return f'"{value}"^^<http://www.w3.org/2001/XMLSchema#float>'
            except Exception:
                return literal


class GraphConstructorWikidata(object):
    def __init__(
        self,
        logger,
        reverse_property_path="data/input/common/reverse_properties_wikidata"
    ):
        self.logger = logger
        
        reverse_properties = {}
        with open(reverse_property_path, 'r') as f:
            for line in f:
                reverse_properties[line.split('\t')[0]] = line.split('\t')[1].replace('\n', '')
        self.reverse_properties = {}
        # 仅保留 A 的逆关系是 B, 同时 B 的逆关系也是 A 的情况
        for (key, value) in reverse_properties.items():
            if (value in reverse_properties) and (reverse_properties[value] == key):
                self.reverse_properties[key] = value


    @classmethod
    def instance(cls, *args, **kwargs):
        if not hasattr(GraphConstructorWikidata, "_instance"):
            GraphConstructorWikidata._instance = GraphConstructorWikidata(*args, **kwargs)
        return GraphConstructorWikidata._instance
    
    def get_symbol_type(self, symbol):
        symbol_type = WikidataConstantForConstruction.get_constant_type(symbol)
        if symbol_type is WIKIDATA_CONSTANT_TYPE.ENTITY:
            return SYMBOL_TYPE.ENTITY
        elif symbol_type is WIKIDATA_CONSTANT_TYPE.CLASS:
            return SYMBOL_TYPE.CLASS
        elif symbol_type in [WIKIDATA_CONSTANT_TYPE.QUANTITY, WIKIDATA_CONSTANT_TYPE.TIME, WIKIDATA_CONSTANT_TYPE.STRING]:
            return SYMBOL_TYPE.LITERAL
        elif re.fullmatch(r"p:P\d+", symbol) or re.fullmatch(r"ps:P\d+", symbol) or re.fullmatch(r"pq:P\d+", symbol) or re.fullmatch(r"wdt:P\d+", symbol):
            return SYMBOL_TYPE.RELATION
        elif symbol == "wdt:P31/wdt:P279*": # 特判
            return SYMBOL_TYPE.RELATION 
        else:
            return None    

    def _add_type(self, node, value):
        if 'type' in node:
            node['type'].add(value)
        else:
            node['type'] = set({value})  

    def add_reverse_relation(self, relation, start_node, end_node, G:nx.DiGraph): 
        prefix, rel_v = relation.split(':')
        relation_r = self.reverse_properties.get(rel_v, None)
        if relation_r:
            G.add_edge(start_node, end_node, value=f"{prefix}:{relation_r}")                                                                                                                     
    
    def logical_form_to_graph(self, lisp_program:list) -> nx.DiGraph:
        '''
        边的 attribute
        - value: Freebase 关系
        - operator: 函数名称

        节点的 attribute
        - value: 知识库常量 或 "dummy"
        - type: set()

        每个子图中，编号最小（0）的节点始终表示目标变量
        其实只关注目标变量的attribute, 中间变量的 attribute 可以忽略
        '''
        expression = lisp_to_nested_expression(lisp_program)
        sub_programs = _linearize_lisp_expression(expression, [0])
        identical_index_r = {}
        idx2graph = {}
        target_idx = len(sub_programs) - 1 # 最终的 target variable 会位于的位置

        def get_root(idx):
            while idx in identical_index_r:
                idx = identical_index_r[idx]
            return idx
        
        for i, subp in enumerate(sub_programs):
            i = str(i)
            G = nx.DiGraph()
            G.add_node(0) # 目标变量节点
            if subp[0] == 'JOIN':
                '''
                subp[1] 我认为只有两种选择:
                - relation
                - R relation
                这两者只有 SPARQL 里面三元组方向的区别

                无论 subp[1] 是什么, subp[2] 有如下选择
                - item: entity / class / literal (对于旧版的 S-expression, TIME 和 QUANTITY 也可能出现在这个位置)
                - #n: 表示一个嵌套的子结构
                - relation 或者 R relation
                '''                
                if subp[2].startswith('#'): # 是一个子成分，需要完成图的合并
                    '''
                    将 subp[2] 所表示的子图的目标变量变为中间变量；新增一个目标变量，通过关系 subp[1] 连接目标和中间变量
                    '''
                    root2 = get_root(int(subp[2][1:]))
                    sub_graph:nx.DiGraph = idx2graph[root2]
                    sub_graph_relabel_mapping = {}
                    for n in sub_graph.nodes():
                        sub_graph_relabel_mapping[n] = n + 1
                    G:nx.DiGraph = nx.relabel_nodes(sub_graph, sub_graph_relabel_mapping, copy=True) # copy = True 返回一个拷贝
                    G.add_node(0) # 新的变量节点
                    if isinstance(subp[1], list): # R relation
                        relation = subp[1][1]
                        G.add_edge(1, 0, value=relation)
                        self.add_reverse_relation(relation, 0, 1, G)

                    elif isinstance(subp[1], str): # relation
                        relation = subp[1]
                        G.add_edge(0, 1, value=relation)
                        self.add_reverse_relation(relation, 1, 0, G)

                elif self.get_symbol_type(subp[2]) in [SYMBOL_TYPE.ENTITY, SYMBOL_TYPE.LITERAL]:
                    G.add_node(len(G.nodes()), value=subp[2]) # 指向常量的节点
                    if isinstance(subp[1], list): # R relation
                        relation = subp[1][1]
                        G.add_edge(len(G.nodes()) - 1, 0, value=relation)
                        self.add_reverse_relation(relation, 0, len(G.nodes()) - 1, G)
                        
                    elif isinstance(subp[1], str): # relation
                        relation = subp[1]
                        G.add_edge(0, len(G.nodes()) - 1, value=relation)
                        self.add_reverse_relation(relation, len(G.nodes()) - 1, 0, G)
                    else:
                        raise Exception(f"subp: {subp}")
                elif self.get_symbol_type(subp[2]) is SYMBOL_TYPE.CLASS:
                    if subp[1] not in [
                        'wdt:P31', 
                        'wdt:P31/wdt:P279*',
                        'wdt:P279'
                    ]:
                        raise Exception(f"subp: {subp}")
                    # 仅对于目标变量节点，添加类型约束
                    self._add_type(G.nodes[0], subp[2])
                else:
                    raise Exception(f"subp: {subp}")
                
                # graph_serialized = json_graph.node_link_data(G)
            
            elif subp[0] in ['EQ', 'LT', 'LE', 'GT', 'GE', 'lt', 'le', 'gt', 'ge']:
                '''
                subp[1]:
                    - 嵌套结构, #n
                    - 关系 / 逆关系
                subp[2]:
                    - 嵌套结构, #n
                    - time / number
                '''
                operator = subp[0]
                if operator in OPERATOR_MAPPING:
                    operator = OPERATOR_MAPPING[operator]
                if subp[1].startswith('#'):
                    # subp[1] 是一个多跳关系；从 subp[1] 子图的最后一个点出发，加一个函数边
                    var1 = int(subp[1][1:])
                    rooti = get_root(int(i))
                    root1 = get_root(var1)
                    if rooti > root1:
                        identical_index_r[rooti] = root1
                    else:
                        identical_index_r[root1] = rooti
                    
                    if subp[2].startswith('#'): # 嵌套
                        root2 = get_root(int(subp[2][1:]))
                        graph_1 = idx2graph[root1]
                        graph_2 = idx2graph[root2]
                        G = self.combine_compare_subgraphs(operator, graph_1, graph_2)
                    elif self.get_symbol_type(subp[2]) is SYMBOL_TYPE.LITERAL: # TIME / NUMBER
                        sub_graph:nx.DiGraph = idx2graph[root1]
                        G = sub_graph.copy()
                        G.add_node(len(G.nodes()), value=subp[2])
                        G.add_edge(len(G.nodes()) - 2, len(G.nodes()) - 1, operator=operator)
                    else:
                        raise Exception(f"subp: {subp}")
                elif isinstance(subp[1], list): # R relation
                    if subp[2].startswith('#'): # 嵌套
                        '''新增从 target variable 指向 subp[2] 头结点的一条边'''
                        root2 = get_root(int(subp[2][1:]))
                        subgraph_2 = idx2graph[root2]
                        graph_2_mapping = {}
                        for n in subgraph_2.nodes():
                            graph_2_mapping[n] = n + 2
                        subgraph_2 = nx.relabel_nodes(subgraph_2, graph_2_mapping, copy=True)
                        G = nx.compose(G, subgraph_2)
                        G.add_node(1) # 中间变量
                        relation = subp[1][1]
                        G.add_edge(1, 0, value=relation)
                        self.add_reverse_relation(relation, 0, 1, G)
                        G.add_edge(1, 2, operator=operator)
                        
                    elif self.get_symbol_type(subp[2]) is SYMBOL_TYPE.LITERAL: # TIME / NUMBER
                        G.add_node(len(G.nodes())) # 中间变量
                        G.add_node(len(G.nodes()), value=subp[2]) # grounded item
                        G.add_edge(len(G.nodes()) - 2, len(G.nodes()) - 1, operator=operator)
                        relation = subp[1][1]
                        G.add_edge(len(G.nodes()) - 2, 0, value=relation)
                        self.add_reverse_relation(relation, 0, len(G.nodes()) - 2, G)
                    else:
                        raise Exception(f"subp: {subp}")
                elif isinstance(subp[1], str): # relation
                    if subp[2].startswith('#'): # 嵌套
                        '''新增从 target variable 指向 subp[2] 头结点的一条边'''
                        root2 = get_root(int(subp[2][1:]))
                        subgraph_2 = idx2graph[root2]
                        graph_2_mapping = {}
                        for n in subgraph_2.nodes():
                            graph_2_mapping[n] = n + 2
                        subgraph_2 = nx.relabel_nodes(subgraph_2, graph_2_mapping, copy=True)
                        G = nx.compose(G, subgraph_2)
                        G.add_node(1) # 中间变量
                        relation = subp[1]
                        G.add_edge(0, 1, value=relation)
                        self.add_reverse_relation(relation, 1, 0, G)
                        G.add_edge(1, 2, operator=operator)
                    elif self.get_symbol_type(subp[2]) is SYMBOL_TYPE.LITERAL: # TIME / NUMBER
                        G.add_node(len(G.nodes())) # 中间变量
                        G.add_node(len(G.nodes()), value=subp[2]) # grounded item
                        G.add_edge(len(G.nodes()) - 2, len(G.nodes()) - 1, operator=operator)
                        relation = subp[1]
                        G.add_edge(0, len(G.nodes()) - 2, value=relation)
                        self.add_reverse_relation(relation, len(G.nodes()) - 2, 0, G)
                    else:
                        raise Exception(f"subp: {subp}")
                else:
                    raise Exception(f"subp: {subp}")
            
            elif subp[0] in ["ARGMIN", "ARGMAX"]:
                '''
                subp[1]: #n
                subp[2]:
                    - relation
                    - R relation
                    - #n, 多跳关系构成的子成分
                
                subp[1] 的目标变量变为中间变量，通过 subp[2] 中的关系（可能是多跳的）连向真正的目标变量
                '''
                if subp[1][0] == '#':
                    var1 = int(subp[1][1:])
                    rooti = get_root(int(i))
                    root1 = get_root(var1)
                    if rooti > root1:
                        identical_index_r[rooti] = root1
                    else:
                        identical_index_r[root1] = rooti
                        root1 = rooti
                    
                    sub_graph_1:nx.DiGraph = idx2graph[root1]
                    if subp[2][0] == '#':
                        var2 = int(subp[2][1:])
                        root2 = get_root(var2)
                        sub_graph_2:nx.DiGraph = copy.deepcopy(idx2graph[root2])
                        if root1 > root2:
                            identical_index_r[root1] = root2
                        else:
                            identical_index_r[root2] = root1
                            root2 = root1
                        # sub_graph_1_serialized = json_graph.node_link_data(sub_graph_1)
                        # sub_graph_2_serialized = json_graph.node_link_data(sub_graph_2)
                        sub_graph_2_len = len(sub_graph_2.nodes())
                        sub_graph_1_target_variable = copy.deepcopy(sub_graph_1.nodes[0])
                        sub_graph_2_target_variable = copy.deepcopy(sub_graph_2.nodes[0])
                        sub_graph_relabel_mapping = {}
                        # type_union 一定要在 relabel 之前获取!
                        type_union = sub_graph_1.nodes[0].get('type', set()) | sub_graph_2.nodes[0].get('type', set())
                        for n in sub_graph_1.nodes():
                            if n != 0:
                                sub_graph_relabel_mapping[n] = n + sub_graph_2_len - 1
                        nx.relabel_nodes(sub_graph_1, sub_graph_relabel_mapping, copy=False) # copy=False 原地修改
                        # 需要对目标节点进行合并
                        G = nx.compose(sub_graph_1, sub_graph_2)
                        if len(type_union) > 0:
                            if 'type' in G.nodes[0]:
                                G.nodes[0]['type'].update(type_union)
                            else:
                                G.nodes[0]['type'] = type_union
                        G.add_edge(sub_graph_2_len - 1, len(G.nodes()), operator=subp[0])
                        # G_serialized = json_graph.node_link_data(G)

                    elif isinstance(subp[2], list): # R relation
                        sub_graph_1_relabel_mapping = {}
                        for n in sub_graph_1.nodes():
                            sub_graph_1_relabel_mapping[n] = n + 2
                        G = nx.relabel_nodes(sub_graph_1, sub_graph_1_relabel_mapping, copy=True)
                        # G.add_node(len(G.nodes()))
                        relation = subp[2][1]
                        G.add_edge(1, 2, value=relation)
                        self.add_reverse_relation(relation, 2, 1, G)
                        G.add_edge(1, 0, operator=subp[0])
                        # G_serialized = json_graph.node_link_data(G)
                    elif isinstance(subp[2], str): # relation
                        sub_graph_1_relabel_mapping = {}
                        for n in sub_graph_1.nodes():
                            sub_graph_1_relabel_mapping[n] = n + 2
                        G = nx.relabel_nodes(sub_graph_1, sub_graph_1_relabel_mapping, copy=True)
                        # G.add_node(len(G.nodes()))
                        relation = subp[2]
                        G.add_edge(2, 1, value=relation)
                        self.add_reverse_relation(relation, 1, 2, G)
                        G.add_edge(1, 0, operator=subp[0])
                        # G_serialized = json_graph.node_link_data(G)
                else:
                    raise Exception(f"subp: {subp}")

            elif subp[0] == 'AND':
                var1 = int(subp[1][1:])
                rooti = get_root(int(i))
                root1 = get_root(var1)
                if rooti > root1:
                    identical_index_r[rooti] = root1
                else:
                    identical_index_r[root1] = rooti
                    root1 = rooti
                var2 = int(subp[2][1:])
                root2 = get_root(var2)
                if root1 > root2:
                    identical_index_r[root1] = root2
                else:
                    identical_index_r[root2] = root1
                
                sub_graph_1:nx.DiGraph = copy.deepcopy(idx2graph[root1])
                sub_graph_2:nx.DiGraph = copy.deepcopy(idx2graph[root2])
                # sub_graph_1_serialized = json_graph.node_link_data(sub_graph_1)
                # sub_graph_2_serialized = json_graph.node_link_data(sub_graph_2)
                sub_graph_relabel_mapping = {}
                # type_union 一定要在 relabel 之前获取!
                type_union = sub_graph_1.nodes[0].get('type', set()) | sub_graph_2.nodes[0].get('type', set())
                for n in sub_graph_2.nodes():
                    if n != 0:
                        sub_graph_relabel_mapping[n] = n + len(sub_graph_1.nodes()) - 1
                nx.relabel_nodes(sub_graph_2, sub_graph_relabel_mapping, copy=False)
                
                G = nx.compose(sub_graph_1, sub_graph_2)
                if len(type_union) > 0:
                    if 'type' in G.nodes[0]:
                        G.nodes[0]['type'].update(type_union)
                    else:
                        G.nodes[0]['type'] = type_union
                # G_serialized = json_graph.node_link_data(G)
            elif subp[0] == 'COUNT':
                idx = get_root(int(subp[1][1:]))
                G:nx.DiGraph = copy.deepcopy(idx2graph[idx])
                sub_graph_relabel_mapping = {}
                for n in G.nodes():
                    sub_graph_relabel_mapping[n] = n + 1
                nx.relabel_nodes(G, sub_graph_relabel_mapping, copy=False)
                G.add_node(0)
                G.add_edge(1, 0, value=subp[0])
            
            elif subp[0] in ['LT_JOIN', 'LE_JOIN', "GT_JOIN", "GE_JOIN", "EQ_JOIN", "ARGMIN_JOIN", "ARGMAX_JOIN", 'lt_JOIN', 'le_JOIN', 'gt_JOIN', 'ge_JOIN', 'JOIN_JOIN']:
                '''
                目标变量 --关系边--> 中间变量1 --关系边--> 中间变量2
                主要考虑关系的方向; 注意要给目标变量加个类型约束
                '''
                G.add_node(len(G.nodes())) # 中间变量1
                G.add_node(len(G.nodes())) # 中间变量2
                if isinstance(subp[1], str):
                    relation = subp[1]
                    G.add_edge(0, 1, value=relation)
                    prefix, rel_v = relation.split(':')
                    relation_r = self.reverse_properties.get(rel_v, None)
                    if relation_r:
                        G.add_edge(1, 0, value=f"{prefix}:{relation_r}")
                elif isinstance(subp[1], list):
                    relation = subp[1][1]
                    G.add_edge(1, 0, value=relation)
                    prefix, rel_v = relation.split(':')
                    relation_r = self.reverse_properties.get(rel_v, None)
                    if relation_r:
                        G.add_edge(0, 1, value=f"{prefix}:{relation_r}")
                else:
                    raise Exception(f"subp: {subp}")
                
                if isinstance(subp[2], str):
                    relation = subp[2]
                    G.add_edge(1, 2, value=relation)
                    self.add_reverse_relation(relation, 2, 1, G)
                elif isinstance(subp[2], list):
                    relation = subp[2][1]
                    G.add_edge(2, 1, value=relation)
                    self.add_reverse_relation(relation, 1, 2, G)
                else:
                    raise Exception(f"subp: {subp}")
            

            else: # 我们不考虑 TC, 遇到了就抛出异常，说我们处理不了吧
                raise Exception(f"subp[0]: {subp[0]}; subp: {subp}")
            
            idx2graph[get_root(int(i))] = copy.deepcopy(G)
            # G_serialized = json_graph.node_link_data(G)
        
        return copy.deepcopy(idx2graph[get_root(target_idx)])

    def combine_compare_subgraphs(self, operator, graph_1:nx.DiGraph(), graph_2:nx.DiGraph):
        '''
        EQ, LT, LE, GT, GE 这些比较级操作符后面跟着两个子图的情况
        #2 的 target variable 作为 #1 的 grounded_item, 保留 #1 的 target variable 不动
        实现上先对 # 2 进行 relabel, 然后合并 #1 和 #2, 最后 #1 的末尾节点添加一个指向 #2 target_variable 的函数边
        '''
        graph_1_size = len(graph_1.nodes())
        graph_2_mapping = {}
        for n in graph_2.nodes():
            graph_2_mapping[n] = n + graph_1_size
        graph_2 = nx.relabel_nodes(graph_2, graph_2_mapping, copy=True)
        combined_graph:nx.DiGraph = nx.compose(graph_1, graph_2)
        combined_graph.add_edge(graph_1_size - 1, graph_1_size, operator=operator)
        return combined_graph

    def combine_target_variable(self, node_1, node_2):
        """
        合并两个目标节点
        - value
        - type: 合并即可
        """
        value = None
        if ('value' in node_1) and ('value' in node_2):
            if node_1['value'] != node_2['value']:
                raise Exception(f"node_1: {node_1}; node_2: {node_2}")
            else:
                value = node_1['value']
        elif 'value' in node_1:
            value = node_1['value']
        elif 'value' in node_2:
            value = node_2['value']
        else:
            pass

        type_set = set()
        if 'type' in node_1:
            type_set.update(node_1['type'])
        if 'type' in node_2:
            type_set.update(node_2['type'])

        return value, type_set
