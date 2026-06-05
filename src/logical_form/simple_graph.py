from enum import Enum
from src.logical_form.s_expression_utils import (
    JOIN, CMP, R, AND, ARG, COUNT
)
from src.core.utils import (
    convert_number,
    compare_literal
)
from src.core.common import FUNCTION_OPERATOR, KB_TYPE
import copy
import itertools
import re

class NodeType(Enum):
    ENTITY = 1,
    CLASS = 2,
    LITERAL = 3,
    VARIABLE = 4

class EdgeType(Enum):
    NON_CVT = 1,
    CVT = 2,
    VARIABLE = 3
    

class Node:
    def __init__(self, value, type: NodeType) -> None:
        self.value = value
        self.type = type
    
    def __repr__(self) -> str:
        return str(self.value)# +":"+ str(self.type)
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __hash__(self):
        return hash((self.type, self.value))

    def __eq__(self, other):
        return self.type == other.type and self.value == other.value
        


class SimpleGraph:
    '''
    把搜索的对象首先实现为简单的有向图，然后可以方便地转为SPARQL或Sexpr
    只需要包含：点与边，边标明对应的property
    对于每个点，记录的是它的出边
    #0919：能否在其上建模ARG，COUNT与数值比较，这样一个simplegrah其实就是query的整体表示了
    '''
    def __init__(self, center_node:Node, ground_kb=KB_TYPE.FREEBASE) -> None:
        '''
        由于我们的搜索方式，简单图的初始化总是只包含一个变量，这个变量也就成为图的中心点
        邻接表格式：{entity1: [{"edge": edge, "to": entity2}]}
        对于图中的变量，我们记录从中心点出发，到其的路径，称为关键路径key path
        '''
        #图所在的KB，影响grounding时的前缀
        self.kb = ground_kb
        #图的中心，对应查询的目标变量
        self.center_node = center_node
        self.nodes = [center_node]
        #改成用edge列表来存储（似乎并不需要邻接表）
        self.edges = []
        #邻接表同时储存出度和入度，便于遍历
        self.key_paths = {} 
        self.key_paths[center_node] = []
        self.count_query = False
        #记录最高级
        #形式是：{"type":"ARGMIN"或"ARGMAX", "node"，作用的变量节点， "property"：该节点上延伸出的目标数值属性}
        self.arg_function = None
        #记录类型约束
        #形式是：{"node"：被约束的变量节点，"type"：类型的mid }
        self.type_constraints = []
        #记录数值与时间的比较
        #形式是：{"node":被约束的变量节点， "property":数值属性， "operator"：比较符， "value"，对应的值}
        self.comparisons = []
    
    # def __init__(self, sparql:str, kb:KB_TYPE):
    #     #从SPARQL转成图
    #     """SELECT DISTINCT ?OUTQ where { \n?OUTQ ns:music.release.engineers ns:m.05c7ffh. \n?OUTQ ns:type.object.type ns:music.release. 
    #     \n?OUTQ ns:music.release.release_date ?arg_value .\n\n}ORDER BY DESC (?arg_value) LIMIT 1 \n"""
    #     sparql = sparql.replace("\n", ".")
    #     sparql_gp_begin = 0
    #     i = 0
    #     while(sparql[i] != "{"):
    #         i += 1
    #     sparql_gp_begin = i
    #     sparql_gp_end = sparql_gp_begin+1
    #     while(sparql[i] != "}"):
    #         i += 1
    #     sparql_gp_end = i
    #     sparql_gp = sparql[sparql_gp_begin: sparql_gp_end].strip("{}")
    #     self.center_node = Node(re.findall("\?w+", sparql[:sparql_gp_begin])[0], NodeType.VARIABLE)
    #     self.count_query = "COUNT" in sparql[:sparql_gp_begin]
    #     triple_or_filters = sparql_gp.split(".")
    #     for t in triple_or_filters:
    #         if "?arg_value" in t:
            
    #         elif "FILTERS" in t:

    def hash(self):
        return hash(self.__str__())

    def get_new_cvt_node(self):
        '''
        生成一个新的cvt_node并返回
        '''
        cvt_num = len(self.get_cvt_nodes())
        cvt_node = Node("?cvt_"+str(cvt_num), type=NodeType.VARIABLE)
        return cvt_node

    def __str__(self) -> str:
        if len(self.edges) == 0 and len(self.type_constraints) == 0 and len(self.comparisons) == 0 and not self.arg_function:
            return str(self.center_node)
        else:
            string = ""
            if self.count_query:
                string += "COUNT, "
            if self.arg_function:
                string += f"{self.arg_function['type']}({self.arg_function['node'].value}.{self.arg_function['property']}) "
            for edge in self.edges:
                string += f"( {str(edge['from'])} -- {edge['edge']}--> {str(edge['to'])} ) "
            for constraint in self.comparisons:
                string += f"( {str(constraint['node'])} {constraint['property']} {constraint['operator']} {constraint['value']} ) "
            for type_constraint in self.type_constraints:
                string += f"type({str(type_constraint['node'])})={type_constraint['type']}"
            return string

    def add_node(self, node:Node):
        #要求是，每个变量只能出现一次
        if node.type == NodeType.VARIABLE:
            if node in self.nodes:
                assert(0)
        if node not in self.nodes:
            self.nodes.append(node)

    def get_deg(self, node:Node):
        assert(node in self.nodes)
        deg = 0
        for e in self.edges:
            if e['from'] == node or e['to'] == node:
                deg+= 1
        return deg
       # return len(self.in_dict[node]) + len(self.out_dict[node])

    def attach_node(self, attach_point:Node, edge, node:Node, direction="out"):
        """
        将一个新的节点连接到graph上，链接点在attach point上，边的label为edge
        方向：out为 attach point --edge--> node， in 为attach_point <--edge-- node
        """
        assert direction in ['in', 'out']
        #我们约定，变量只能和变量链接
        assert(attach_point.type == NodeType.VARIABLE)
        self.add_node(node)
        if direction == "out":
            self.edges.append({"edge":edge, "from": attach_point, "to":node})
            self.key_paths[node] = self.key_paths[attach_point]+[edge]
        else:
            self.edges.append({"edge":edge, "from": node, "to":attach_point})
            self.key_paths[node] = self.key_paths[attach_point]+["^"+edge]    #需要取反
        # self.out_dict[attach_point]=[(edge, node)]
        # self.in_dict[node]=[(edge, attach_point)]
        #维护并更新关键路径

    def to_Sexpr(self):
        """
        自身转化为S_expression
        """

        def property_path_to_sparql(property_path:str):
            #将形如p1/^p2形式的关系路径str转换为SEPXR
            total_expr = None
            properties = property_path.split("/")
            for p in reversed(properties):
                if p.startswith("^"):
                    p_expr = R(p.strip("^"))
                else:
                    p_expr = p
                if total_expr is None:
                    total_expr = p_expr
                else:
                    total_expr = JOIN(p_expr, total_expr)
            return total_expr
        
        def manage_xsd_type(token):
            token = token.replace("<","").replace(">","").replace('"', "")
            token = token.replace("-08:00", "")
            return token
        
        operator_symbol_dict = {">=":"ge", ">":"gt", "<=":"le", "<":"lt"}
        vis = {}    #记录已经访问过的边
        def recursively_build_sexpr_for_variable_node(cur_node):
            child_Sexprs = []
            edges = [e for e in self.edges if e['from'] == cur_node or e['to'] == cur_node]
            for tc in self.type_constraints:
                if tc['node'] == cur_node:
                    child_Sexprs.append(tc['type'])
            for compare in self.comparisons:
                if compare['node'] == cur_node:
                    #处理XSD，删除<>
                    value = manage_xsd_type(compare['value'])
                    child_Sexprs.append(CMP(operator_symbol_dict[compare['operator']], \
                                            property_path_to_sparql(compare['property']), value))
            for e in edges:
                if hash(frozenset(e.items())) in vis:
                    continue
                else:
                    vis[hash(frozenset(e.items()))] = True
                    if e['from'] == cur_node:
                        if e['to'].type != NodeType.VARIABLE:
                            #处理XSD，删除<>
                            if "XMLSchema#" in str(e['to']):
                                child_Sexprs.append(JOIN(e['edge'], manage_xsd_type(str(e['to']))))
                            else:
                                child_Sexprs.append(JOIN(e['edge'], str(e['to'])))
                        else:
                            child_Sexprs.append(JOIN(e['edge'], recursively_build_sexpr_for_variable_node(e['to'])))
                    if e['to'] == cur_node:
                        if e['from'].type != NodeType.VARIABLE:
                            #处理XSD，删除<>
                            if "XMLSchema#" in str(e['from']):
                                child_Sexprs.append(JOIN(R(e['edge'], manage_xsd_type(str(e['from'])))))
                            else:
                                child_Sexprs.append(JOIN(R(e['edge']), str(e['from'])))
                        else:
                            child_Sexprs.append(JOIN(R(e['edge']), recursively_build_sexpr_for_variable_node(e['from'])))
            assert(len(child_Sexprs) > 0)
            cur_Sexpr = child_Sexprs[0]
            if len(child_Sexprs) > 1:
                for expr in child_Sexprs[1:]:
                    cur_Sexpr = AND(cur_Sexpr, expr)
            #最后放ARG
            if self.arg_function is not None and self.arg_function['node'] == cur_node:
                cur_Sexpr = ARG(self.arg_function['type'], cur_Sexpr, property_path_to_sparql(self.arg_function['property']))
            return cur_Sexpr
        graph_sexpr = recursively_build_sexpr_for_variable_node(self.center_node)
        if self.count_query:
            graph_sexpr = COUNT(graph_sexpr)
        return graph_sexpr

    def to_sparql_query_fast_PR_with_entity_answers(self, answers:list):
        #当答案是一系列实体时，通过一些技巧，重写SPARQL，使其执行速度更快
        #返回两个不同的SPARQL，分别判断P和R
        answers_rep = ""
        if self.kb == KB_TYPE.FREEBASE:
            answers_rep = " ".join(["ns:"+ans['mid'] for ans in answers])
        elif self.kb == KB_TYPE.WIKIDATA:
            answers_rep = " ".join([ans['mid'] for ans in answers])
        else:
            raise Exception("not implemented")
        r_sparql = "SELECT DISTINCT "+ self.center_node.value + " where { \n"
        r_sparql += f'VALUES {self.center_node.value} {{ {answers_rep} }}.'
        r_sparql += self.to_sparql_gp() + "\n}"
        if self.arg_function is not None:
            if self.arg_function['type'] == "ARGMAX":
                r_sparql += "ORDER BY DESC (?arg_value) LIMIT 1 \n"
            else:
                r_sparql += "ORDER BY ?arg_value LIMIT 1 \n"
        p_sparql = "SELECT COUNT( DISTINCT " + self.center_node.value + ") AS ?cnt where { \n"
        p_sparql += self.to_sparql_gp() + "\n}"
        if self.arg_function is not None:
            if self.arg_function['type'] == "ARGMAX":
                p_sparql += "ORDER BY DESC (?arg_value) LIMIT 1 \n"
            else:
                p_sparql += "ORDER BY ?arg_value LIMIT 1 \n"
        return p_sparql, r_sparql

    def to_sparql_query(self):
        """
        将graph转换为查询目标为center_node的sparql
        """
        if not self.count_query:
            sparql = "SELECT DISTINCT "+ self.center_node.value + " where { \n"
        else:
            sparql = "SELECT DISTINCT  COUNT(" + self.center_node.value + ") AS ?cnt where { \n"
        sparql += self.to_sparql_gp() + "\n}"
        #todo：添加数值比较与最高级
        if self.arg_function is not None:
            if self.arg_function['type'] == "ARGMAX":
                sparql += "ORDER BY DESC (?arg_value) LIMIT 1 \n"
            else:
                sparql += "ORDER BY ?arg_value LIMIT 1 \n"
        return sparql


    def to_sparql_query_easy_to_read(self):
        """
        将graph转换为查询目标为center_node的sparql，为标注实验设计，删除了一些不好阅读的FILTER，并合并p/ps
        """
        if not self.count_query:
            sparql = "SELECT DISTINCT "+ self.center_node.value + " where { \n"
        else:
            sparql = "SELECT DISTINCT  COUNT(" + self.center_node.value + ") AS ?cnt where { \n"
        sparql += self.to_sparql_gp_easy_to_read() + "\n}"
        #todo：添加数值比较与最高级
        if self.arg_function is not None:
            if self.arg_function['type'] == "ARGMAX":
                sparql += "ORDER BY DESC (?arg_value) LIMIT 1 \n"
            else:
                sparql += "ORDER BY ?arg_value LIMIT 1 \n"
        return sparql

    def get_format_property_for_argorcmp(self, p):
        if self.kb == KB_TYPE.FREEBASE:
            #需要拆开，并添加ns:
            if "/" in p:
                r1, r2 = p.split("/")[0], p.split("/")[1]
                if r1.startswith("^"):
                    r1 = "^ns:"+ r1.strip("^")
                else:
                    r1 = "ns:" + r1
                if r2.startswith("^"):
                    r2 = "^ns:"+ r2.strip("^")
                else:
                    r2 = "ns:" + r2                   
                format_property = f"{r1}/{r2}"
            else:
                if p.startswith("^"):
                    format_property = "^ns:" + p.strip("^")
                else:
                    format_property = "ns:" + p
        elif self.kb == KB_TYPE.WIKIDATA:
            #已经保留了前缀，不需要操作
            format_property = p
        else:
            raise Exception("not implemented")
        return format_property
    

    def get_sparql_gp_with_answer(self, answer):
        #获取将centernode替换为answer的id的sparql
        #将三元组中，包含答案变量的换成相应的mid；删除约束答案变量的类型、filter与arg。
        sparql = ""
        for edge_item in self.edges:
            s = edge_item['from'].value
            o = edge_item['to'].value
            if self.kb == KB_TYPE.FREEBASE:
                if edge_item['from'].type == NodeType.ENTITY:
                    s = "ns:" + s
                elif edge_item['from'] == self.center_node:
                    s = "ns:" + answer['mid']
                if edge_item['to'].type == NodeType.ENTITY:
                    o = "ns:" + o 
                elif edge_item['to'] == self.center_node:
                    o = "ns:" + answer['mid']
                sparql += s + " " + "ns:" + edge_item['edge'] + " " + o + ". \n"
            elif self.kb == KB_TYPE.WIKIDATA:
                if edge_item['from'] == self.center_node:
                    s = answer['mid']
                if edge_item['to'] == self.center_node:
                    o = answer['mid']
                sparql += s + " " + edge_item['edge'] + " " + o + ". \n"
            else:
                raise Exception("not implemented")                
        for type_constraint in self.type_constraints:
            if type_constraint['node'] == self.center_node:
                continue
            if self.kb == KB_TYPE.FREEBASE:
                s = type_constraint['node'].value
                o = "ns:" + type_constraint['type']
                sparql += s + " ns:type.object.type " + o + ". \n"
            elif self.kb == KB_TYPE.WIKIDATA:
                s = type_constraint['node'].value
                o = type_constraint['type']
                sparql += s + " wdt:P31/wdt:P279* " + o + ". \n"
            else:
                raise Exception("not implemented")    
        for idx, cmp in enumerate(self.comparisons):
            #需要引出一条value三元组
            if cmp['node'] == self.center_node:
                continue
            formated_property = self.get_format_property_for_argorcmp(cmp['property'])
            sparql += f"{cmp['node'].value} {formated_property} ?value_{str(idx)} .\n"
            sparql += f"FILTER (?value_{str(idx)} {cmp['operator']} {cmp['value']}) .\n"
        if self.arg_function is not None:
            if self.arg_function['node'] == self.center_node:
                pass
            formated_property = self.get_format_property_for_argorcmp(self.arg_function['property'])
            sparql += f"{self.arg_function['node'].value} {formated_property} ?arg_value .\n"
        #添加一系列filters
        # ents_str = ", ".join(["ns:" + item.value for item in self.nodes if item.type == NodeType.ENTITY])
        #         elif len(ents_str) > 0:
        #             #sparql += f"FILTER ( {node.value} NOT IN ({ents_str}) )\n"
        if self.kb == KB_TYPE.FREEBASE:
            for node in self.nodes:
                if node.type == NodeType.VARIABLE:
                    if "?cvt" in node.value:
                        sparql += f"FILTER NOT EXISTS {{ {node.value} ns:type.object.name ?name . }}. \n"
        #注意，要删除答案变量
        non_cvt_vars = set([node for node in self.nodes if node.type == NodeType.VARIABLE])
        non_cvt_vars.remove(self.center_node)
        combinations = itertools.combinations(non_cvt_vars, 2)
        neqs = []
        for combination in combinations:
            neqs.append(f" {combination[0].value} != {combination[1].value} ")
        if len(neqs) > 0:
            sparql += "FILTER ( "+ "&&".join(neqs) +" ). \n"
        #各nodes之间的关系不能相同
        return sparql


    def to_sparql_gp_easy_to_read(self):
        sparql = ""
        if self.kb == KB_TYPE.WIKIDATA:
            #需要合并p和ps
            new_edges = []
            wdt_edges = []
            deleted_cvt_nodes = []
            #想法：遍历edges，寻找?cvt节点，如果?cvt节点仅有两端，且分别是p/ps，那么可以合并为wdt:
            for cvt in self.get_cvt_nodes():
                associated_edges = [item for item in self.edges if item['from'] == cvt or item['to'] == cvt]
                if len(associated_edges) == 2:
                    p_edge = [item for item in associated_edges if item['edge'].startswith("p:")]
                    p_edge = p_edge[0] if len(p_edge[0]) > 0 else None
                    ps_edge = [item for item in associated_edges if item['edge'].startswith("ps:")]
                    ps_edge = ps_edge[0] if len(ps_edge[0]) > 0 else None
                    if p_edge is not None and ps_edge is not None:
                        from_node = p_edge['from']
                        to_node = ps_edge['to']
                        relation = p_edge['edge'].split(":")[1]
                        wdt_edges.append({"from":from_node, "to": to_node, "edge":"wdt:"+relation})
                        deleted_cvt_nodes.append(cvt)
            new_edges += wdt_edges
            for edge in self.edges:
                if edge['from'] in deleted_cvt_nodes or edge['to'] in deleted_cvt_nodes:
                    continue
                else:
                    new_edges.append(edge)
            self.edges = new_edges
        for edge_item in self.edges:
            s = edge_item['from'].value
            o = edge_item['to'].value
            if self.kb == KB_TYPE.FREEBASE:
                if edge_item['from'].type == NodeType.ENTITY:
                    s = "ns:" + s
                if edge_item['to'].type == NodeType.ENTITY:
                    o = "ns:" + o 
                sparql += s + " " + "ns:" + edge_item['edge'] + " " + o + ". \n"
            elif self.kb == KB_TYPE.WIKIDATA:
                sparql += s + " " + edge_item['edge'] + " " + o + ". \n"
            else:
                raise Exception("not implemented")                
        for type_constraint in self.type_constraints:
            if self.kb == KB_TYPE.FREEBASE:
                s = type_constraint['node'].value
                o = "ns:" + type_constraint['type']
                sparql += s + " ns:type.object.type " + o + ". \n"
            elif self.kb == KB_TYPE.WIKIDATA:
                s = type_constraint['node'].value
                o = "ns:" + type_constraint['type']
                sparql += s + " wdt:P31/wdt:P279* " + o + ". \n"
            else:
                raise Exception("not implemented")    
        for idx, cmp in enumerate(self.comparisons):
            #需要引出一条value三元组
            formated_property = self.get_format_property_for_argorcmp(cmp['property'])
            sparql += f"{cmp['node'].value} {formated_property} ?value_{str(idx)} .\n"
            if cmp.get('convert_to_year'):
                sparql += f"FILTER (YEAR(?value_{str(idx)}) {cmp['operator']} {cmp['value']}) .\n"
            else:
                sparql += f"FILTER (?value_{str(idx)} {cmp['operator']} {cmp['value']}) .\n"
        if self.arg_function is not None:
            formated_property = self.get_format_property_for_argorcmp(self.arg_function['property'])
            sparql += f"{self.arg_function['node'].value} {formated_property} ?arg_value .\n"
        if self.kb == KB_TYPE.FREEBASE:
            for node in self.nodes:
                if node.type == NodeType.VARIABLE:
                    if "?cvt" in node.value:
                        sparql += f"FILTER NOT EXISTS {{ {node.value} ns:type.object.name ?name . }}. \n"
        # non_cvt_vars = set([node for node in self.nodes if node.type == NodeType.VARIABLE])
        # combinations = itertools.combinations(non_cvt_vars, 2)
        # neqs = []
        # for combination in combinations:
        #     neqs.append(f" {combination[0].value} != {combination[1].value} ")
        # if len(neqs) > 0:
        #     sparql += "FILTER ( "+ "&&".join(neqs) +" ). \n"
        #各nodes之间的关系不能相同
        return sparql
    
    def to_sparql_gp(self):
        sparql = ""
        for edge_item in self.edges:
            s = edge_item['from'].value
            o = edge_item['to'].value
            if self.kb == KB_TYPE.FREEBASE:
                if edge_item['from'].type == NodeType.ENTITY:
                    s = "ns:" + s
                if edge_item['to'].type == NodeType.ENTITY:
                    o = "ns:" + o 
                sparql += s + " " + "ns:" + edge_item['edge'] + " " + o + ". \n"
            elif self.kb == KB_TYPE.WIKIDATA:
                sparql += s + " " + edge_item['edge'] + " " + o + ". \n"
            else:
                raise Exception("not implemented")                
        for type_constraint in self.type_constraints:
            if self.kb == KB_TYPE.FREEBASE:
                s = type_constraint['node'].value
                o = "ns:" + type_constraint['type']
                sparql += s + " ns:type.object.type " + o + ". \n"
            elif self.kb == KB_TYPE.WIKIDATA:
                s = type_constraint['node'].value
                o = "ns:" + type_constraint['type']
                sparql += s + " wdt:P31/wdt:P279* " + o + ". \n"
            else:
                raise Exception("not implemented")    
        for idx, cmp in enumerate(self.comparisons):
            #需要引出一条value三元组
            formated_property = self.get_format_property_for_argorcmp(cmp['property'])
            sparql += f"{cmp['node'].value} {formated_property} ?value_{str(idx)} .\n"
            if cmp.get('convert_to_year'):
                sparql += f"FILTER (YEAR(?value_{str(idx)}) {cmp['operator']} {cmp['value']}) .\n"
            else:
                sparql += f"FILTER (?value_{str(idx)} {cmp['operator']} {cmp['value']}) .\n"
        if self.arg_function is not None:
            formated_property = self.get_format_property_for_argorcmp(self.arg_function['property'])
            sparql += f"{self.arg_function['node'].value} {formated_property} ?arg_value .\n"
        #添加一系列filters
        # ents_str = ", ".join(["ns:" + item.value for item in self.nodes if item.type == NodeType.ENTITY])
        #         elif len(ents_str) > 0:
        #             #sparql += f"FILTER ( {node.value} NOT IN ({ents_str}) )\n"
        #加上一个Filter:
        if self.kb == KB_TYPE.FREEBASE:
            for node in self.nodes:
                if node.type == NodeType.VARIABLE:
                    if "?cvt" in node.value:
                        sparql += f"FILTER NOT EXISTS {{ {node.value} ns:type.object.name ?name . }}. \n"
        non_cvt_vars = set([node for node in self.nodes if node.type == NodeType.VARIABLE])
        combinations = itertools.combinations(non_cvt_vars, 2)
        neqs = []
        for combination in combinations:
            neqs.append(f" {combination[0].value} != {combination[1].value} ")
        if len(neqs) > 0:
            sparql += "FILTER ( "+ "&&".join(neqs) +" ). \n"
        #各nodes之间的关系不能相同
        return sparql


    def add_type_constaint(self, node:Node, type):
        #为某个节点添加类型约束。这里，将类型(rdf:type)与其他三元组进行不同的处理
        assert(node in self.nodes)
        self.type_constraints.append({"node":node, "type":type})

    def add_comparison(self, node:Node, property, operator, value):
        assert(operator in FUNCTION_OPERATOR.values())
        self.comparisons.append({"node":node, "property":property, "operator": operator, "value":value})
        return 0
    
    def get_key_path(self, variable:Node):
        assert(variable.type == NodeType.VARIABLE)
        return self.key_paths[variable]
    
    def get_cvt_nodes(self):
        cvt_nodes = [node for node in self.nodes if "?cvt" in node.value]
        return cvt_nodes

    def get_cvt_structures(self):
        """
        返回图中CVT的相关结构
        返回格式：以cvt为中心的edge:
        例如：[{?cvt_1:[(?v1, p1, ?cvt), (?cvt, p2, ?v2)]}]
        """
        res = {}
        cvt_nodes = self.get_cvt_nodes()
        for node in cvt_nodes:
            edges = [item for item in self.edges if item['from'] == node or item['to'] == node]
            res[node] = edges
        return res

    def get_cvt_star(self, cvt_node):
        star = []
        for e in self.edges:
            if e['from']  == cvt_node or e['to'] == cvt_node:
                star.append(e)
        return star
    
    def no_constant(self):
        const_node_num = len([node for node in self.nodes if node.type != NodeType.VARIABLE])
        return const_node_num == 0

def get_attached_graph(graph_original:SimpleGraph, graph_appended_original:SimpleGraph):
    """
    将graph_appended链接到graph上，形成一个新的图
    返回deepcopy，不改变原有的对象
    要求：两个图之间刚好有一个公共点
    """
    #首先，将graph_tobe_attach中的CVT进行重命名
    graph = copy.deepcopy(graph_original)
    graph_appended = copy.deepcopy(graph_appended_original)
    cvt_replacement = {}
    for index, cvt_node in enumerate(graph_appended.get_cvt_nodes()):
        index = int(cvt_node.value.split("_")[-1])
        new_index = index + len(graph.get_cvt_nodes())
        cvt_replacement[cvt_node] =Node("?cvt_"+str(new_index), NodeType.VARIABLE)
    if len(graph_appended.get_cvt_nodes()) > 0:
        rename_node(graph_appended, cvt_replacement)
    merge_point = [item for item in set(graph.nodes).intersection(set(graph_appended.nodes)) if item.type == NodeType.VARIABLE]
    duplicate_points = [item for item in set(graph.nodes).intersection(set(graph_appended.nodes)) if item.type != NodeType.VARIABLE]
    if len(merge_point) != 1:
        print("在将", graph_appended, "链接到", graph, "时发生错误")
        assert(0)
    merge_point = list(merge_point)[0]
    #更新邻接表和边
    for node in graph_appended.nodes:
        if node != merge_point and node not in duplicate_points:
            graph.add_node(node)
    graph.edges += graph_appended.edges
    #更新关键路径
    for node in graph_appended.nodes:
        if node.type == NodeType.VARIABLE:
            graph.key_paths[node] = graph.key_paths[merge_point] + graph_appended.key_paths[node]
    #更新solution modifiers
    graph.comparisons += graph_appended.comparisons
    #旧图与新图，应当至多只有一个arg_function
    assert(graph.arg_function is None or graph_appended.arg_function is None)
    if graph_appended.arg_function is not None:
        graph.arg_function = graph_appended.arg_function
    return graph


def rename_node(graph:SimpleGraph, replacement:dict):
    """
    输入：一个图graph，一系列新旧节点的对应关系，形如{old_node1: new_node1}
    将g的old_node替换为new_node，返回修改后的图。
    注意：node_new不能与graph中某个节点同名。
    #采取直接修改nodes中的Node中的value与type实现
    """
    for old_node, new_node in replacement.items():
        graph.nodes[graph.nodes.index(old_node)] = new_node 
    new_key_paths = {}
    temp = []
    for k, v in graph.key_paths.items():
        if k in replacement:
            temp.append((replacement[k], v))
        else:
            temp.append((k, v))
    for item in temp:
        new_key_paths[item[0]] = item[1]
    graph.key_paths = new_key_paths
    for idx, e in enumerate(graph.edges):
        for old_node, new_node in replacement.items():
            if e['from'] == old_node:
                e['from'] = new_node
                graph.edges[idx] = e
            elif e['to'] == old_node:
                e['to'] = new_node
                graph.edges[idx] = e
    return graph

