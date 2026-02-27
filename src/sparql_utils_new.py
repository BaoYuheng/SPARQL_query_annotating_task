from src.awudima import parser
from src.awudima import (
    unary_operators, unary_expression_list, binary_operators,
    ternary_operators, quaternary_operators, aggregate_functions, #tc_operators,
    BGP, GGP, UnionGP, OptionalGP, Bind, Filter, GraphGP, ServiceGP, MinusGP, ValuesClause,
    TriplePattern, RDFTerm, PropertyPath, Operator, Expression,
    SelectQuery, Query, AskQuery, PathTerm
)
from src.common import DATASET
from src.utils import load_json
import re
import copy
from anytree import Node, RenderTree
from collections import defaultdict
from queue import Queue
from functools import cmp_to_key
from zss import simple_distance

#修改为whzhou的绝对地址
#20260122 eswc rebuttal
REVERSE_RELATION_PATH = "/home5/whzhou/STAR-QC/data/input/common/freebase_reverse_relation_map.json"
RELATION_DOMAIN_RANGE_PATH = "/home5/whzhou/STAR-QC/data/input/common/fb_relations_domain_range_label.json"
WIKIDATA_REVERSE_RELATION_PATH = "/home5/whzhou/STAR-QC/data/input/common/reverse_properties_wikidata"

def is_operator(oper_str:str):
    if not isinstance(oper_str, str):
        return False
    if oper_str.upper() in unary_operators or oper_str in unary_operators:
        return True
    if oper_str.upper() in unary_expression_list:
        return True
    if oper_str.upper() in binary_operators or oper_str in binary_operators:
        return True
    if oper_str.upper() in aggregate_functions:
        return True
    if oper_str.upper() in ternary_operators:
        return True
    if oper_str.upper() in quaternary_operators:
        return True
    #20260122 eswc rebuttal修改
    # if oper_str.upper() in tc_operators:
    #     return True
    if oper_str in ['<', '>', '<=', '>=', '=']:
        return True
    #需要增加xsd:type
    if "xsd:" in oper_str:
        return True
    return False

def load_relation_dicts():
    '''
    收集到的数据中，存在 A 的逆关系是 B, 但是 B 的逆关系不是 A 的情况
    这应该是数据有一些缺漏？我们在读取数据的时候，如果 A 的逆关系是 B, 则自动补充 B 的逆关系是 A
    '''
    print("Calling load_relation_dicts()")
    reverse_relations, relation_domain_range = dict(), dict()
    for (key, value) in load_json(REVERSE_RELATION_PATH).items():
        if len(value) == 1: # 验证过了，所有关系至多有一个逆关系
            reverse_relations[key] = value[0]
            reverse_relations[value[0]] = key

    with open(RELATION_DOMAIN_RANGE_PATH, 'r') as f:
        for k, v in load_json(RELATION_DOMAIN_RANGE_PATH).items():
            relation_domain_range[k] = v
    
    return reverse_relations, relation_domain_range

def load_reverse_relation_wikidata():
    print("Calling load_reverse_relation_wikidata()")
    wikidata_reverse_relations = dict()
    with open(WIKIDATA_REVERSE_RELATION_PATH, 'r') as f:
        for line in f:
            '''
            Wikidata 的记录中，确实存在 A 的逆关系是 B, 但是 B 的逆关系不是 A 的情况；
            但是这和逆关系的定义相违背, 我们认为可能是数据质量问题，因此在这里，都假设逆关系是对称的
            '''
            rel, rel_rev = line.split('\t')[0], line.split('\t')[1].replace('\n', '')
            wikidata_reverse_relations[rel] = rel_rev
            wikidata_reverse_relations[rel_rev] = rel
    return wikidata_reverse_relations
        

def merge_dicts(dict1, dict2):
    merged_dict = dict1
    if dict2:
        for key, value in dict2.items():
            if key in merged_dict:
                merged_dict[key] += value
            else:
                merged_dict[key] = value
    return merged_dict

def get_relation_label(relation_iri):
    return relation_iri.split("/")[-1]

def get_relation_iri_prefix(relation_iri):
    return "/".join(relation_iri.split("/")[:-1])

def cmp_triple_patterns(tp1:TriplePattern, tp2:TriplePattern):
    s1, p1, o1 = tp1.subject, tp1.property, tp1.object
    s2, p2, o2 = tp2.subject, tp2.property, tp2.object
    #首先，两个三元组中，常量较多的那个小
    const_tp1 = [arg for arg in [s1, p1, o1] if arg.is_constant]
    const_tp2 = [arg for arg in [s2, p2, o2] if arg.is_constant]
    if len(const_tp1) < len(const_tp2):
        return 1
    elif len(const_tp1) > len(const_tp2):
        return -1
    #如果常量数量相同，那么常量位置的和比较小的那个小（比如A和B都只有一个常量，A常量在主语，B常量在谓语，那么A比B小）
    const_idx_sum_tp1 = sum([100*i for i, arg in enumerate([s1, p1, o1]) if arg.is_constant])
    const_idx_sum_tp2 = sum([100*i for i, arg in enumerate([s2, p2, o2]) if arg.is_constant])
    if const_idx_sum_tp1 < const_idx_sum_tp2:
        return -1
    elif const_idx_sum_tp1 > const_idx_sum_tp1:
        return 1
    #如果常量位置数量都相同，那么比较常量拼接的字典序
    const_concat_tp1 = ",".join([str(c) for c in const_tp1])
    const_concat_tp2 = ",".join([str(c) for c in const_tp2])
    if const_concat_tp1 < const_concat_tp2:
        return -1
    elif const_concat_tp1 > const_concat_tp2:
        return 1
    #最后的办法：直接比较三元组string
    if str(tp1) > str(tp2):
        return 1
    elif str(tp1) < str(tp2):
        return -1
    else:
        return 0


class GGPTreeNode:
    def __init__(self, node, parent):
        self.node = node
        self.type = type(node)
        self.parent = parent # 只会有一个 parent
        self.children = [] # 任意个 children
        self.var_set = self.get_var_set()
        self.varsetrelation = []

    def add_child(self,child):
        self.children.append(child)

    def change_parent(self,new_parent):
        self.parent = new_parent

    def to_str(self,level = 0):
        # 重排序等依赖这个方法，不要修改!
        tree_list = []
        indent = "-" * level
        if type(self.node) in [
           RDFTerm, ValuesClause, Filter, Operator
        ]:
            tree_list.append(f"{indent}{self.type} {self.node.to_str()}\n")
        elif isinstance(self.node, str):
            tree_list.append(f"{indent}{self.type} {self.node}\n")
        else:
            tree_list.append(f"{indent}{self.type}\n")
        for child in self.children:
            tree_list.extend(child.to_str(level + 1))
        return tree_list
    
    def get_label(self):
        if len(self.children) > 0:
            if type(self.node) == str:
                return self.node
            else:
                return str(self.type)
        else: # 叶子节点
            if type(self.node) == str:
                return self.node
            else: # 语法树中的 item
                return self.node.to_str()

    def Filter_var(self):
        for child in self.children:
            child.Filter_var()
            self.varsetrelation.extend(child.varsetrelation)
        Exist_list = []
        out_list = []
        for child in self.children:
            if child.type == 'NotExistFilter' or child.type =='ExistFilter':
                Exist_list.append(child)
            else:
                out_list.append(child)

        out_list_varset = set()
        for child in out_list:
            out_list_varset = out_list_varset.union(child.var_set)
        for ef in Exist_list:
            varset = ef.var_set
            self.varsetrelation.append((str(ef.type),str(bool(varset-(varset.intersection(out_list_varset))))))


    # def tree_normoliazation(self):
    #     for child in self.children:
    #         child.tree_normoliazation()
    #         self.varsetrelation.extend(child.varsetrelation)
    #     child_normol = []
    #     not_opt_minus_bind = []
    #     for child in self.children:
    #         if child.type == OptionalGP or child.type == MinusGP or child.type == Bind:

    #             # if not_opt_minus_bind:
    #             #     child_normol.extend(sequence_normoliazation(not_opt_minus_bind))
    #             #                         # print(self.varsetrelation)
    #             #     child_normol.append(child)
    #             #     not_opt_minus_bind = []
    #             # else:
    #             #     child_normol.append(child)
    #             #     not_opt_minus_bind = []
    #             child_normol.extend(sequence_normoliazation(not_opt_minus_bind))
                
    #             varset = child.var_set
    #             before_varset = set()
    #             after_varset = set()
    #             for c in child_normol:
    #                 before_varset = before_varset.union(c.var_set)
    #             for c in self.children[len(child_normol)+1:]:
    #                 after_varset = after_varset.union(c.var_set)

    #             self.varsetrelation.append((str(child.type),str(bool(varset.intersection(before_varset))),str(bool(varset.intersection(after_varset)))))
            
    #             child_normol.append(child)
    #             not_opt_minus_bind = []

    #         else:
    #             not_opt_minus_bind.append(child)

    #     if not_opt_minus_bind:
    #         child_normol.extend(sequence_normoliazation(not_opt_minus_bind))
    #         not_opt_minus_bind = []
    #     self.children = child_normol
    
    def get_depth(self):
        tree_depth = 1
        max_child_depth = 0
        for child in self.children:
            child_depth = child.get_depth()
            max_child_depth = max(max_child_depth,child_depth)
        tree_depth += max_child_depth
        return tree_depth

    def get_width(self):
        tree_width = 1
        max_child_width = 0
        for child in self.children:
            child_width = child.get_width()
            max_child_width = max(max_child_width,child_width)
        tree_width = max(len(self.children),max_child_width)
        return tree_width

    def get_dfs_path(self,now_path,total_path):
        
        now_path.append(str(self.type))
        if not self.children:
            total_path.append(now_path[1:])

        temp = now_path

        for child in self.children:  
            total_path = child.get_dfs_path(now_path,total_path)
            now_path.pop()
            
        return total_path

    def get_bgps(self):
        bgps = []
        for child in self.children:
            child_bgps = child.get_bgps()
            bgps.extend(child_bgps)
        if isinstance(self.node,BGP):
            bgps.append(str(self.node))
        return bgps

    def get_ngram(self,n):
        ngram_list = []

        for child in self.children:
            child_ngram = child.get_ngram(n)
            ngram_list.extend(child_ngram)
        
        if len(self.children)>n:
            for i in range(0,len(self.children)-n+1):
                ngram_list.append([str(c.type) for c in self.children[i:i+n]])
        
        return ngram_list

    def get_var_set(self):
        if isinstance(self.node,Bind):
            return set(self.node.as_var)
        else:
            var_set = set()
            pattern = r'\?var\d+'
            for var in re.findall(pattern, str(self.node)):
                var_set.add(var)
            return var_set

    def get_keyword_frequency(self):
        keyword_frequecy = defaultdict(int)
        for child in self.children:
            child_frequecy = child.get_keyword_frequency()
            keyword_frequecy = merge_dicts(keyword_frequecy,child_frequecy)
        keyword_frequecy[str(self.type)] += 1
        return keyword_frequecy

    def __str__(self):
        return '\n'.join(self.to_str())
    
    def visualize(self):
        def create_node(cur_ggp_node:GGPTreeNode, parent= None):
            if len(cur_ggp_node.children) > 0:
                if cur_ggp_node.type == Expression:
                    cur_node = Node(str(cur_ggp_node.node.oper), parent=parent)
                elif cur_ggp_node.type == str:
                    cur_node = Node(cur_ggp_node.node, parent=parent)
                else:
                    cur_node = Node(str(cur_ggp_node.type)[2:-2].split(".")[-1], parent=parent)
            else:
                cur_node = Node(str(cur_ggp_node.node), parent=parent)
            for sub in cur_ggp_node.children:
                create_node(sub, cur_node)
            return cur_node
        root_node = create_node(self, None)
        for pre, fill, node in RenderTree(root_node):
            print("%s%s" % (pre, node.name))


class TreeConstructor:
    # 类公共变量
    reverse_relations, relation_domain_range = load_relation_dicts()
    wikidata_reverse_relations = load_reverse_relation_wikidata()

    def __init__(self, sparql_txt, dataset_type:DATASET, logger):
        self.sparql_query:Query = parser.parse_sparql(sparql_txt)
        self.prefixes = self.sparql_query.prefixes
        self.dataset_type = dataset_type
        self.logger = logger
        self.existential_variables = [] #需要用来为中间变量命名
    
    def construct_syntax_tree(self):
        #tree_root = GGPTreeNode(self.sparql_query, None)
        # tree_root = GGPTreeNode("ROOT",None)
        # self._construct_syntax_tree(tree_root, self.sparql_query)
        # self.tree_root = tree_root
        # return self.tree_root

        tree_root = GGPTreeNode("ROOT", None)
        self._construct_syntax_tree(tree_root, self.sparql_query)
        self.tree_root = tree_root
        return self.tree_root
    
    def _construct_syntax_tree(self, parent_node:GGPTreeNode, current):
        
        def decompose_property_path(property_path):
            #将property_path展开成一个子property_path的序列。目前只考虑展开/
            res = [property_path]
            while 1:
                changed = False
                for idx in range (0, len(res)):
                    if isinstance(res[idx], PropertyPath) and res[idx].oper == "/": #展开
                        changed = True
                        res = res[0:idx] + [res[idx].left_path, res[idx].right_path] + res[idx+1:]
                        break
                if not changed:
                    break
            return res

        if isinstance(current, Bind) or isinstance(current, ServiceGP) or isinstance(current, MinusGP) or isinstance(current, OptionalGP):
            # 数据集中不应该存在上述 GP
            raise Exception(f"current: {current}")
        
        # Terminal node
        elif isinstance(current, RDFTerm):
            current.expand_syntax_forms(self.prefixes) # 前缀替换
            current_node = GGPTreeNode(current, parent_node)
            current_node.is_variable = not current.is_constant   #bao 用这个来判断是不是变量
            parent_node.add_child(current_node)
            return
        
        elif isinstance(current, PropertyPath):
            #property path 在 triple_pattern处已经处理。这里不应再出现property_path
            raise NotImplementedError(f"PropertyPath: {current}")

        elif isinstance(current, PathTerm):
            if isinstance(current, PathTerm):
                #把pathTerm的value与is_constant等内容填了，方便排序
                current.is_constant = True
                current.value = str(current)
            current.expand_syntax_forms(self.prefixes) # 前缀替换
            current_node = GGPTreeNode(current, parent_node)
            parent_node.add_child(current_node)

        elif isinstance(current, ValuesClause):
            # if len(current.variables) == 1 and len(current.values) == 1:
            #     # 直接完成 variable 到 value 的替换
            #     variable_rdf_term = current.variables[0]
            #     if len(current.values[0]) != 1:
            #         raise NotImplementedError(f"ValuesClause: {current}")
            #     value_rdf_term:RDFTerm = current.values[0][0]
            #     self.variable_mapping[variable_rdf_term] = value_rdf_term
            # else:
            #     raise NotImplementedError(f"ValuesClause: {current}")
            # return

            # valuesClause 在预处理阶段已经处理过了，此处不应该出现
            raise Exception(f"ValuesClause: {current}")

        # Filter 底下的 Expression, 视作一个复杂元素；其他情况下的 Expression, 视作 terminal node
        # elif isinstance(current, Expression):
        #     current_node = GGPTreeNode(current, parent_node)
        #     parent_node.add_child(current_node)
        #     return

        elif isinstance(current, Expression):   #添加表达Expression的节点
            current_node = GGPTreeNode(current, parent_node)
            parent_node.add_child(current_node)
            # 暂时只处理二元 Expression
            for sub_expr in [current.left_expr, current.right_expr, current.ternary_expr, current.quaternary_expr, current.oper]:
            # for sub_expr in [current.left_expr, current.right_expr, current.ternary_expr, current.quaternary_expr]:
                if sub_expr is not None:
                    self._construct_syntax_tree(current_node, sub_expr)

        elif isinstance(current, TriplePattern):
            if isinstance(current.property, RDFTerm) or isinstance(current.property, PathTerm):
                #property是单个RDFTerm，或是PathTerm
                current_node = GGPTreeNode(current, parent_node)
                parent_node.add_child(current_node)
                self._construct_syntax_tree(current_node, current.subject)
                self._construct_syntax_tree(current_node, current.property)
                self._construct_syntax_tree(current_node, current.object)
            else:
                #Property是一关系路径
                decomposed_path = decompose_property_path(current.property)
                #创建一系列新的三元组
                begin = current.subject
                for idx, path in enumerate(decomposed_path):
                    if idx == len(decomposed_path)-1:
                        end = current.object
                    else: # 关系路径中间，新增一些中间变量
                        end = RDFTerm("?exist_"+str(len(self.existential_variables)), is_const=False)
                        self.existential_variables.append(end)
                    triple = TriplePattern(begin, path, end)
                    triple_node = GGPTreeNode(triple, parent_node)
                    parent_node.add_child(triple_node)
                    self._construct_syntax_tree(triple_node, triple.subject)
                    self._construct_syntax_tree(triple_node, triple.property)
                    self._construct_syntax_tree(triple_node, triple.object)
                    begin = end
        
        elif isinstance(current, BGP):
            for _tri in current.triples:
                self._construct_syntax_tree(parent_node, _tri)
            for _filter in current.filters:
                self._construct_syntax_tree(parent_node, _filter)
        
        elif isinstance(current, list): # 用花括号括起来的 GGP
            for _ggp in current:
                self._construct_syntax_tree(parent_node, _ggp)
        
        elif isinstance(current, Filter):
            # 处理 Filter, 然后处理 Filter 下的 Expression 节点
            current_node = GGPTreeNode(current, parent_node)
            parent_node.add_child(current_node)
            #expression:Expression = current.expression
            #expression_node = GGPTreeNode(current.expression, parent_node)
            # current_node.add_child(expression_node)
            # 暂时只处理二元 Expression
            # self._construct_syntax_tree(expression_node, expression.left_expr)
            # self._construct_syntax_tree(expression_node, expression.oper)
            # self._construct_syntax_tree(expression_node, expression.right_expr)
            self._construct_syntax_tree(current_node, current.expression)

        # elif isinstance(ggp, SelectQuery) or isinstance(ggp, AskQuery):
        #     ggp_node = GGPTreeNode(ggp, parent_node)
        #     parent_node.add_child(ggp_node)
        #     _construct_syntax_tree(ggp_node,ggp.ggp)
        elif isinstance(current, AskQuery):
            # ggp_node = GGPTreeNode(current, parent_node)
            # parent_node.add_child(ggp_node)
            # self._construct_syntax_tree(ggp_node,current.ggp)
            raise NotImplementedError(f"Cannot handle AskQuery yet: {current}")
        elif isinstance(current, SelectQuery):
            #修改：一个Query node, 下跟solution modifiers和ggp
            query_node = GGPTreeNode(current, parent_node)
            parent_node.add_child(query_node)
            '''暂时只处理单个投影变量的情况'''
            if len(current.projections) != 1:
                raise NotImplementedError(f"ggp.projections: {current.projections}")
            #projection_node = GGPTreeNode(current.projections[0], parent_node)
            #parent_node.add_child(projection_node)

            # projection 作为一个原子节点，不会往下细分了
            projection_node = GGPTreeNode(current.projections[0], query_node)
            query_node.add_child(projection_node)
            if current.distinct:
                # distinct_node = GGPTreeNode('distinct', parent_node)
                # parent_node.add_child(distinct_node)
                distinct_node = GGPTreeNode('distinct', query_node)
                query_node.add_child(distinct_node)
            if current.order_by:
                for _order_by_item in current.order_by: 
                    # _order_by_item: 简单的 Expression, 这边直接处理
                    # expression_node = GGPTreeNode("order by", parent_node)
                    # parent_node.add_child(expression_node)
                    # expression_node = GGPTreeNode("order by", query_node)
                    # query_node.add_child(expression_node)
                    # self._construct_order_by_left_expr(expression_node, _order_by_item.left_expr)
                    # self._construct_syntax_tree(expression_node, _order_by_item.oper)
                    # self._construct_syntax_tree(expression_node, _order_by_item.right_expr)

                    #改为通用的处理Expression的方法
                    order_by_node = GGPTreeNode("order by", query_node)
                    query_node.add_child(order_by_node)
                    self._construct_syntax_tree(order_by_node, _order_by_item)
            if current.limit != -1:
                # node = GGPTreeNode(f"LIMIT {current.limit}", parent_node)
                # parent_node.add_child(node)
                node = GGPTreeNode(f"LIMIT {current.limit}", query_node)
                query_node.add_child(node)
            if current.offset != -1:
                # node = GGPTreeNode(f"OFFSET {current.offset}", parent_node)
                # parent_node.add_child(node)
                node = GGPTreeNode(f"OFFSET {current.offset}", query_node)
                query_node.add_child(node)
            if current.group_by or current.having:
                raise NotImplementedError(f"group by or having: {current}")
            #这里有问题，应当使用query patterns作为parent的子节点
            # query_pattern_node = GGPTreeNode(current.ggp, parent_node)
            # parent_node.add_child(query_pattern_node)
            self._construct_syntax_tree(query_node, current.ggp)

        
        elif isinstance(current, GGP) or isinstance(current, UnionGP):
            ggp_node = GGPTreeNode(current,parent_node)
            parent_node.add_child(ggp_node)
            self._construct_syntax_tree(ggp_node,current.ggps)
        
        elif is_operator(current): # 较为特殊的 Terminal Node
            current_node = GGPTreeNode(current, parent_node)
            parent_node.add_child(current_node)
            return
        
        elif current is None: # 有一些 Expression 的部分项为 None, 放在这处理
            return

        else:
            return
            raise NotImplementedError(f"current: {current} {type(current)}")
    
    def _construct_order_by_left_expr(self, parent_node:GGPTreeNode, current):
        if isinstance(current, Expression):
            expression_node = GGPTreeNode(current, parent_node)
            parent_node.add_child(expression_node)

            if len(current.left_expr) != 1:
                raise NotImplementedError(f"order by current.left_expr: {current.left_expr}")
            left_expr_node = GGPTreeNode(current.left_expr[0], expression_node)
            expression_node.add_child(left_expr_node)
            self._construct_syntax_tree(expression_node, current.oper)
            # 此处的 right_expr 为 None
        else:
            raise NotImplementedError(f"current: {current} {type(current)}")

    def get_node_of_specific_type_in_subtree(self, root_node:GGPTreeNode, node_type):
        res = []
        q = Queue()
        q.put(root_node)
        while not q.empty():
            top = q.get()
            if top.type == node_type:
                res.append(top)
            for child in top.children:
                q.put(child)
        return res

    def _convert_to_unf(self, root_node:GGPTreeNode):
        '''
        将SPARQL语法树转换为Union Normal Form。考虑分配律：A OP (B UNION C) = (A OP B) UNION (A OP C)
        '''
        #首先，bfs收集所有的union 
        union_nodes = self.get_node_of_specific_type_in_subtree(root_node, UnionGP)
        #之后，对每个union节点依次往上提
        for union in union_nodes:
            #判断其是否是最顶层
            pass
        return 0
    
    def _add_reverse_relations(self, root_node:GGPTreeNode):
        #为有逆关系的三元组添加逆的三元组：对于每个(s, p, o)所在的BGP，都添加(o, p_r, s)
        #这里默认不考虑关系路径
        triples = self.get_node_of_specific_type_in_subtree(root_node, TriplePattern)
        for triple in triples:
            parent = triple.parent
            if self.dataset_type in [DATASET.CWQ, DATASET.GRAIL, DATASET.WEBQ, DATASET.SIMULATED_FREEBASE, DATASET.QGG, DATASET.QUERYAGENT, DATASET.BINDER, DATASET.LSQ]:
                if get_relation_label(triple.node.property.value) in TreeConstructor.reverse_relations:
                    reversed_p_label = TreeConstructor.reverse_relations[get_relation_label(triple.node.property.value)]
                    reversed_p = RDFTerm(value = get_relation_iri_prefix(triple.node.property.value) + "/" + reversed_p_label,
                                        is_iri = True,
                                        is_const = True
                    )
                #增加一个新的三元组
                    new_triple = TriplePattern(triple.node.object, reversed_p, triple.node.subject)
                    self._construct_syntax_tree(parent, new_triple)
            elif self.dataset_type in [DATASET.LC2, DATASET.QALD, DATASET.SIMULATED_WIKIDATA]:
                if get_relation_label(triple.node.property.value) in TreeConstructor.wikidata_reverse_relations:
                    reversed_p_label = TreeConstructor.wikidata_reverse_relations[get_relation_label(triple.node.property.value)]
                    reversed_p = RDFTerm(value = get_relation_iri_prefix(triple.node.property.value) + "/" + reversed_p_label,
                                        is_iri = True,
                                        is_const = True
                    )
                    #增加一个新的三元组
                    new_triple = TriplePattern(triple.node.object, reversed_p, triple.node.subject)
                    self._construct_syntax_tree(parent, new_triple)
            else:
                raise NotImplementedError(f"dataset: {self.dataset_type}")

    def _add_domain_range(self, root_node:GGPTreeNode):
        #为三元组的主语与宾语添加类型限制
        # 调用此函数时，仅指向单个常量的变量，应该已经被替换为常量了
        triples = self.get_node_of_specific_type_in_subtree(root_node, TriplePattern)
        for triple in triples:
            parent = triple.parent
            rdf_type = RDFTerm(get_relation_iri_prefix(triple.node.property.value) + "/" + "type.object.type",  is_const=True, is_iri=True)
            if get_relation_label(triple.node.property.value) in TreeConstructor.relation_domain_range:
                # 已经是一个类型约束的三元组，则不需要进一步处理 domain 和 range
                if get_relation_label(triple.node.property.value) in ['type.object.type', 'type.type.instance']:
                    continue 
                if 'domain' in TreeConstructor.relation_domain_range[get_relation_label(triple.node.property.value)]:
                    if not triple.node.subject.is_constant: # 常量无需添加类型约束
                        domain_label = TreeConstructor.relation_domain_range[get_relation_label(triple.node.property.value)]['domain']
                        domain = RDFTerm(get_relation_iri_prefix(triple.node.property.value) + "/" + domain_label,
                                        is_const = True, 
                                        is_iri = True)

                        subject_type_triple = TriplePattern(triple.node.subject, rdf_type, domain)
                        self._construct_syntax_tree(parent, subject_type_triple)
                if 'range' in TreeConstructor.relation_domain_range[get_relation_label(triple.node.property.value)]:
                    if not triple.node.object.is_constant:
                        range_label = TreeConstructor.relation_domain_range[get_relation_label(triple.node.property.value)]['range']
                        range =  RDFTerm(get_relation_iri_prefix(triple.node.property.value) + "/" + range_label,
                                        is_const = True, 
                                        is_iri = True)
                        object_type_triple = TriplePattern(triple.node.object, rdf_type, range)
                        self._construct_syntax_tree(parent, object_type_triple)
    
    def _remove_duplicated_children(self, current_node:GGPTreeNode):
        for child in current_node.children:
            # self._reorder_children(child)
            self._remove_duplicated_children(child)
        if current_node.type == GGP or current_node.type == UnionGP or current_node.type == BGP:
            if len(current_node.children) > 1:
                #对孩子去重
                children_strs = []
                deduplicated_children = []
                for child in current_node.children:
                    if str(child) not in children_strs:
                        children_strs.append(str(child))
                        deduplicated_children.append(child)
                current_node.children = deduplicated_children
                current_node.children.sort(key = lambda i: str(i))

    def _reorder_children(self, current_node:GGPTreeNode):
        def cmp_normalized_nodes(node1, node2):
            masked_1 = mask_var_index(node1)
            masked_2 = mask_var_index(node2)
            if masked_1 > masked_2:
                return 1
            elif masked_1 < masked_2:
                return -1 
            else:
                if str(node1) > str(node2):
                    return 1
                elif str(node1) < str(node2):
                    return -1
                else:
                    assert(0)


        def mask_var_index(node:GGPTreeNode):
            res_str = str(node)
            #考虑到由于变量命名不同导致的排序不同，对变量全部mask为?X
            vars = list(set(re.findall("\?\w+\d+", str(res_str))))
            vars.sort(key = lambda i:len(str(i)), reverse = True) #考虑?x10 与?x1, 先mask长的
            for v in vars:
                res_str = res_str.replace(v, "?X")
            return res_str
        
        #目前只考虑GGP与UnionGP内元素的重排序。同时，对于这些元素，如果它们存在相同的两个孩子，那么只保留一个
        for child in current_node.children:
            self._reorder_children(child)
        if current_node.type == GGP or current_node.type == UnionGP or current_node.type == BGP:
            if len(current_node.children) > 1:
                current_node.children.sort(key = cmp_to_key(cmp_normalized_nodes))
        return
    
    def _rename_vars(self, root_node:GGPTreeNode):
        '''
        出发点：在KBQA中，不应该出现主谓宾全为变量的三元组。
        同时，所有的变量都应当在三元组模式中被引入。
        那么，我们直接取所有的三元组模式，并按照它们的常量为其排序，按照三元组的次序为变量编号。
        唯一有可能出现错误的地方是，如果有两个三元组，它们的常量位置与值完全相同。
        为了解决这一点，我们尝试在三元组排序的同时进行变量的重命名。
        '''

        def rename_rdfterms_in_ggp_tree(root_node, rename_dict):
            #根据变量重命名表，对GGPTree中的节点重命名
            def rename_expr(expr_node, rename_dict):
                #重命名某个Expression节点
                if isinstance(expr_node, RDFTerm):
                    if expr_node.value in rename_dict:
                        expr_node.value = rename_dict[expr_node.value]
                    if expr_node.value.startswith("$"):
                        expr_node.value = "?" + expr_node.value[1:]
                # else: # 这里是不是对 Expression 重命名？
                elif isinstance(expr_node, Expression):
                    for arg in [expr_node.left_expr, expr_node.right_expr, expr_node.ternary_expr, expr_node.quaternary_expr]:
                        if arg is not None:
                            rename_expr(arg, rename_dict)
            #获取所有叶子节点
            leaves = []
            q = Queue()
            q.put(root_node)
            while not q.empty():
                top = q.get()
                if len(top.children) == 0:
                    leaves.append(top)
                for child in top.children:
                    q.put(child)
            #对两种情况重命名：
            #print([l.node.value for l in leaves if hasattr(l.node, "value")])
            for leaf in leaves:
                if leaf.type == RDFTerm:
                    if leaf.node.value in rename_dict:
                        leaf.node.value = rename_dict[leaf.node.value]
                    if leaf.node.value.startswith("$"):
                        leaf.node.value = "?" + leaf.node.value[1:]
                elif leaf.type == Expression:
                    rename_expr(leaf.node, rename_dict)
            #print([l.node.value for l in leaves if hasattr(l.node, "value")])

        def rename_vars_in_triple_referring_to_dict(tp:TriplePattern, rename_dict):
            if tp.subject.value in rename_dict:
                tp.subject.value = rename_dict[tp.subject.value]
            if tp.property.value in rename_dict:
                tp.property.value = rename_dict[tp.property.value]
            if tp.object.value in rename_dict:
                tp.object.value = rename_dict[tp.object.value]
            return tp
        
        #获取所有三元组
        triples = self.get_node_of_specific_type_in_subtree(root_node, TriplePattern)
        triples = [
            TriplePattern(ggp.children[0].node, ggp.children[1].node, ggp.children[2].node)
            for ggp in triples
        ]
        #开始对三元组排序的同时确定变量的排序。
        rename_dict = {}
        #sorted_triples = []
        while len(triples) > 0:
            #每次取出当前序最小的一个三元组
            min_triple = min(triples, key = cmp_to_key(cmp_triple_patterns))
            triples.remove(min_triple)
            #重命名它的变量
            vars_in_min = [t for t in [min_triple.subject, min_triple.property, min_triple.object] if not t.is_constant]
            vars_in_min.sort(key = lambda i:i.value)
            for v in vars_in_min:
                if not v.value.startswith("$"):
                    #还没有被重命名的变量
                    new_name = "$var" + str(len(rename_dict))
                    rename_dict[v.value] = new_name
                    #对还未处理的三元组中的变量进行相应的替换
            temp = []
            for tp in triples:
                temp.append(rename_vars_in_triple_referring_to_dict(tp, rename_dict))
            triples = temp
        #现在，我们只需要对树中的每个rdfterm变量，按照该rename_dict进行重命名。
        rename_rdfterms_in_ggp_tree(root_node, rename_dict)
        return 0 
        
    def canonicalize_syntax_tree(self, root_node:GGPTreeNode):
        #转变为Union Normal Form
        #self._convert_to_unf()
        #重排后的变量重命名
        #root_node.visualize()
        # self._rename_vars(root_node)
        #根据字典序重排有交换律的各节点孩子
        #添加关系的逆、domain、range

        # self._add_reverse_relations(root_node)
        # self._add_domain_range(root_node)

        if self.dataset_type in [
            DATASET.CWQ, DATASET.GRAIL, DATASET.WEBQ, DATASET.SIMULATED_FREEBASE, DATASET.QGG, DATASET.QUERYAGENT, DATASET.BINDER, DATASET.LSQ
        ]: # 仅对于 Freebase 上的数据集，考虑类型约束
            self._add_domain_range(root_node)
        
        self._add_reverse_relations(root_node)

        #去重并重排孩子
        #对孩子去重
        self._remove_duplicated_children(root_node)
        #重排后的变量重命名
        self._rename_vars(root_node)
        #重排孩子
        self._reorder_children(root_node)
        #root_node.visualize()
        return 0

class TreeEditDistance:

    @staticmethod
    def get_label(node:GGPTreeNode):
        # return node.to_str()
        return node.get_label()
    
    @staticmethod
    def get_children(node):
        if isinstance(node, GGPTreeNode):
            return node.children
        else:
            return []
    
    @staticmethod
    def label_dist(a, b):
        if a == b:
            return 0
        else:
            return 1
    
    @staticmethod
    def get_edit_distance(tree_root_0:GGPTreeNode, tree_root_1:GGPTreeNode):
        dummy_root = GGPTreeNode("ROOT", None)
        distance = simple_distance(
            tree_root_0, 
            tree_root_1,
            get_label=TreeEditDistance.get_label,
            get_children=TreeEditDistance.get_children,
            label_dist=TreeEditDistance.label_dist
        )
        construction_cost_0 = simple_distance(
            dummy_root, 
            tree_root_0, 
            get_label=TreeEditDistance.get_label,
            get_children=TreeEditDistance.get_children,
            label_dist=TreeEditDistance.label_dist
        )
        construction_cost_1 = simple_distance(
            dummy_root, 
            tree_root_1, 
            get_label=TreeEditDistance.get_label,
            get_children=TreeEditDistance.get_children,
            label_dist=TreeEditDistance.label_dist
        )

        normed_distance = distance / (max(construction_cost_0, construction_cost_1))
        normed_distance = min(normed_distance, 1.0) # 限定在 [0.0, 1.0]
        
        return normed_distance

class SyntaxTreeEditor:
    """
    用于 Test Suite Accuracy 的计算
    - 构建语法树，做一些预处理
    - 提供节点替换的 API, 用于生成新的测试用例
    """
    def __init__(self, sparql_txt, dataset:DATASET, logger):
        self.dataset = dataset
        sparql_txt = sparql_txt.replace(' OR ', ' || ')
        self.sparql_query:SelectQuery = parser.parse_sparql(sparql_txt) # 我们只考虑 SelectQuery
        # 先按照前缀展开？但是一些后处理的规则就得发生变化了
        self.sparql_query.expand_syntax_forms()
        if not isinstance(self.sparql_query, SelectQuery):
            raise NotImplementedError(f"query: {self.sparql_query}; type: {type(self.sparql_query)}")
        self.leafs = list()
        self.items = set()
        self.logger = logger
        self._traverse_leaf_nodes(self.sparql_query)
    
    @property
    def sparql_txt(self):
        return self.sparql_query.to_str()

    def _traverse_leaf_nodes(self, current):
        if isinstance(current, ServiceGP) or isinstance(current, MinusGP) or isinstance(current, OptionalGP):
            # 数据集中不应该存在上述 GP
            raise NotImplementedError(f"current: {current}")

        elif isinstance(current, Bind):
            self._traverse_leaf_nodes(current.expression)
            # self.leafs.append(current.as_var)
            # self._traverse_leaf_nodes(current.as_var)

        # Terminal node
        elif isinstance(current, RDFTerm):
            current.expand_syntax_forms(self.sparql_query.prefixes) # 按照前缀展开
            self.leafs.append(current)

            if "?" not in current.to_str() and "P" not in current.to_str():
                self.items.add(current)

            return
        
        elif isinstance(current, PropertyPath):
            # 将整个 PropertyPath 视作叶子节点；虽然我们会对 PropertyPath 做拆分，但是替换时不会对其做替换
            self.leafs.append(current)
        
        elif isinstance(current, PathTerm):
            # 同样，目前处理为叶子节点
            self.leafs.append(current)
        
        elif isinstance(current, ValuesClause): 
            current.expand_syntax_forms(self.sparql_query.prefixes)
            self.leafs.append(current)
            return
        
        elif isinstance(current, Expression):
            for sub_expr in [current.left_expr, current.right_expr, current.ternary_expr, current.quaternary_expr]:
                if sub_expr is not None:
                    self._traverse_leaf_nodes(sub_expr)
        
        elif isinstance(current, TriplePattern):
            self._traverse_leaf_nodes(current.subject)
            self._traverse_leaf_nodes(current.property)
            self._traverse_leaf_nodes(current.object)
        
        elif isinstance(current, BGP):
            for _tri in current.triples:
                self._traverse_leaf_nodes(_tri)
            for _filter in current.filters:
                self._traverse_leaf_nodes(_filter)

        elif isinstance(current, list): # 用花括号括起来的 GGP
            for _ggp in current:
                self._traverse_leaf_nodes(_ggp)
        
        elif isinstance(current, Filter):
            self._traverse_leaf_nodes(current.expression)
        
        elif isinstance(current, SelectQuery):
            # projection 的替换，是另外实现的
            # distinct, limit, offset 我们不做修改
            # order by 里面只出现非目标变量，我们不会对其做修改
            self._traverse_leaf_nodes(current.ggp)
        
        elif isinstance(current, AskQuery):
            raise NotImplementedError(f"current: {current} {type(current)}")
        
        elif isinstance(current, GGP) or isinstance(current, UnionGP):
            self._traverse_leaf_nodes(current.ggps)
        
        elif is_operator(current): # 不会对 operator 做替换
            # self.leafs.append(current)
            return
        
        elif current is None: # 有一些 Expression 的部分项为 None, 放在这处理
            return

        else:
            raise NotImplementedError(f"current: {current} {type(current)}")
    
    def update_leaf_value(self, old_str, new_value):
        """
        遍历 self.leafs, 找到值和 old_value 一致的叶子节点，并将其更新为 new_value
        更新成功返回 True, 更新失败（未找到这样的节点）返回 False
        @param old_str: 语法树中 item 调用 to_str() 得到的字符串
        @param new_value: 语法树中 item 的 value 属性
        这里假设对于 Literal, 更新后值的类型等都是不变的
        """
        flag = False
        # print(old_str,new_value)
        for leaf in self.leafs:
            # print(leaf,old_str)
            if isinstance(leaf, RDFTerm):
                if leaf.to_str() == old_str:
                    # 对于数值和时间，类型等信息不会变，只有 value 变化 (但 value 只有前缀部分)
                    # 对于其他类型(entity, class, 文本型字面量)，同样只有 value 变化
                    leaf.value = new_value
                    leaf.expand_syntax_forms(self.sparql_query.prefixes)
                    flag = True
            else: # ValuesClause 在预处理阶段已经删除; 但是 PropertyPath 可能作为叶子节点
                pass

            if ":" in leaf.to_str():
                
                leaf_str = leaf.to_str().strip('<').strip('>').split(":")
                if leaf_str[0] in old_str and leaf_str[1] in old_str:
                    leaf.value = new_value
                    leaf.expand_syntax_forms(self.sparql_query.prefixes)
                    flag = True
        return flag
    
    def is_consist_node(self,item):
        flag = False
        for leaf in self.leafs:
            if isinstance(leaf, RDFTerm):
                if leaf.to_str() == item:
                    flag = True
            else:
                pass
            
            if ":" in leaf.to_str():
                leaf_str = leaf.to_str().strip('<').strip('>').split(":")
                if leaf_str[0] in item and leaf_str[1] in item:
                    flag = True
        return flag

    def replace_leaf_with_variable(self, old_value, variable_name):
        """
        遍历 self.leafs, 找到值和 old_value 一致的叶子节点，并将其更新为 new_value
        更新成功返回 True, 更新失败（未找到这样的节点）返回 False
        """
        flag = False
        for leaf in self.leafs:
            if isinstance(leaf, RDFTerm):
                if leaf.to_str() == old_value: # 简单粗暴一些的判断
                    leaf.is_bnode = False
                    leaf.is_constant = False
                    leaf.is_iri = False
                    leaf.is_nil = False
                    leaf.is_typed_literal = False
                    leaf.lang_tag = None
                    leaf.prefix = None
                    leaf.value = variable_name
                    leaf.xsd_datatype = None
                    leaf.expand_syntax_forms(self.sparql_query.prefixes)
                    flag = True
            # ValuesClause 在预处理阶段已经删除 但是 PropertyPath 可能作为叶子节点
            else:
                pass
        if flag:
            self.sparql_query.projections.append(variable_name)
            self.update()
        return flag
    
    def remove_projection_variable(self, variable_name):
        updated_projections = []
        for _proj in self.sparql_query.projections:
            if isinstance(_proj, RDFTerm):
                if _proj.value != variable_name:
                    updated_projections.append(_proj)
            elif isinstance(_proj, Expression):
                if _proj.right_expr.value != variable_name:
                    updated_projections.append(_proj)
            else:
                raise NotImplementedError(f"_proj: {_proj} {type(_proj)}")
        modified = (updated_projections != self.sparql_query.projections)
        self.sparql_query.projections = updated_projections
        self.update()
        return modified
    
    def _rename_target_variable(self, target_variable_name):
        """
        1. 检查是否只有一个 target variable, 如果存在多个，则抛出异常
        2. 检查 target variable 的名字是否一致，如果不一致，则更新 projections 和叶子节点
        对于 SPARQL 查询的目标变量重命名，避免替换之后，出现变量重名的问题
        """
        if len(self.sparql_query.projections) != 1: # 多意图 SPARQL, 这里会报错
            raise NotImplementedError(f"多个投影变量，sparql_txt: {self.sparql_txt}; projections: {self.sparql_query.projections}")

        old_variable_name = None
        if isinstance(self.sparql_query.projections[0], RDFTerm):
            if self.sparql_query.projections[0].value == target_variable_name:
                return False
        elif isinstance(self.sparql_query.projections[0], Expression):
            if self.sparql_query.projections[0].right_expr.value == target_variable_name:
                return False
            # 直接在这边特殊处理 --> 形如 SELECT (COUNT(?x0) AS ?value)，我们对 ?value 重命名即可，?value 在 SPARQL 查询中的其他地方应该不会出现
            self.sparql_query.projections[0].right_expr.value = target_variable_name
            return True
        else:
            raise NotImplementedError(f"self.sparql_query.projections[0]: {self.sparql_query.projections[0]}")
        
        old_variable_name = self.sparql_query.projections[0].value
        for leaf in self.leafs:
            if isinstance(leaf, RDFTerm):
                if leaf.value == old_variable_name:
                    leaf.value = target_variable_name
            elif isinstance(leaf, ValuesClause):
                # Values 只在 GrailQA 数据集中出现，同时 Values 中出现的变量不可能是 target variable
                pass 
            elif isinstance(leaf, PropertyPath):
                pass
            elif isinstance(leaf, PathTerm):
                pass
            else:
                raise NotImplementedError(f"leaf: {leaf}")
        
        self.sparql_query.projections[0].value = target_variable_name
        self.update()
        return True

    def _remove_outer_select_grailqa(self):
        new_sparql_root = None
        if not isinstance(self.sparql_query, SelectQuery):
            raise NotImplementedError(f"root: {self.sparql_query} {type(self.sparql_query)}")
        for _ggp in self.sparql_query.ggp.ggps:
            if isinstance(_ggp, SelectQuery): # 找到内层的首个 SelectQuery, 其实就是把最外层的 SelectQuery 扔掉
                new_sparql_root = _ggp
        # 检查外层 Select 是否具有 COUNT, 如果有，需要传递给内层 Select
        if isinstance(self.sparql_query.projections[0], Expression):
            left_expr = self.sparql_query.projections[0].left_expr
            if isinstance(left_expr, Expression):
                if left_expr.oper.upper() == 'COUNT': # 有 COUNT flag
                    old_value = new_sparql_root.projections[0]
                    new_sparql_root.projections[0] = Expression(
                        left_expr=Expression(left_expr=True, oper='COUNT', right_expr=copy.deepcopy(old_value)),
                        oper='AS',
                        right_expr=copy.deepcopy(self.sparql_query.projections[0].right_expr)
                    )
                    new_sparql_root.distinct = False # distinct 已经出现在 projections[0] 里面了
        self.sparql_query = new_sparql_root
        self.update()
    
    def _remove_extra_filter_lines(self):
        lines = self.sparql_txt.split('\n')
        if self.dataset in [DATASET.CWQ, DATASET.WEBQ]:
            try:
                where_clause_idx = None
                for (idx, line) in enumerate(lines):
                    if line.strip().endswith('WHERE {'):
                        where_clause_idx = idx
                target_idx_list = [where_clause_idx + 1, where_clause_idx + 2]
                filtered_lines = [
                    ln for (idx, ln) in enumerate(lines)
                    if (not ln.startswith('FILTER')) or (idx not in target_idx_list)
                ]
            except:
                filtered_lines = lines
            self.sparql_query:SelectQuery = parser.parse_sparql("\n".join(filtered_lines))
            self.update()
        elif self.dataset == DATASET.GRAIL:
            filtered_lines = [
                ln for ln in lines
                if not (ln.startswith('FILTER') and (' != ' in ln))
            ]
            self.sparql_query:SelectQuery = parser.parse_sparql("\n".join(filtered_lines)) 
            self.update()
        elif self.dataset in [DATASET.SIMULATED_FREEBASE, DATASET.SIMULATED_WIKIDATA]:
            filtered_lines = [
                ln for ln in lines
                if not ln.startswith('FILTER ((! isLiteral')
            ]
            self.sparql_query:SelectQuery = parser.parse_sparql("\n".join(filtered_lines))
            self.update()
        elif self.dataset == DATASET.QGG:
            filtered_lines = [
                ln for ln in lines
                if not (
                    (ln.startswith('FILTER') and (' != ' in ln)) or # FILTER (?e%s!=%s)
                    (ln.startswith('FILTER ((! isLiteral'))
                )
            ]
            self.sparql_query:SelectQuery = parser.parse_sparql("\n".join(filtered_lines)) 
            self.update()
        elif self.dataset == DATASET.QUERYAGENT:
            filtered_lines = [
                ln for ln in lines
                if not (
                    (ln.startswith('FILTER') and (' != ' in ln)) or # FILTER (?e%s!=%s)
                    (ln.startswith('FILTER ((! isLiteral'))
                )
            ]
            self.sparql_query:SelectQuery = parser.parse_sparql("\n".join(filtered_lines)) 
            self.update()
        elif self.dataset == DATASET.BINDER:
            filtered_lines = [
                ln for ln in lines
                if not (
                    (ln.startswith('FILTER') and (' != ' in ln)) or # FILTER (?e%s!=%s)
                    (ln.startswith('FILTER ((! isLiteral'))
                )
            ]
            self.sparql_query:SelectQuery = parser.parse_sparql("\n".join(filtered_lines)) 
            self.update()
    
    def _add_prefix(self):
        if self.dataset in [DATASET.QUERYAGENT]:
            sparql_txt = "PREFIX : <http://rdf.freebase.com/ns/>\n" + self.sparql_txt
            self.sparql_query:SelectQuery = parser.parse_sparql(sparql_txt) 
            self.update()
    
    def _break_wdt_wikidata2(self):
        """
        Wikidata 上 wdt 关系的处理
        目前暂时不使用这个函数，其实现思路是 将 ?x wdt:p ?y 拆成 ?x p:p ?z ?z ps:p ?y. 会添加存在性变量与增加三元组
        """
        def get_bgps_on_syntax_tree(current_node, res:list):
            #返回语法树上所有的TriplePattern，以及它们的父亲
            if "Query" in str(type(current_node)):
                get_bgps_on_syntax_tree(current_node.ggp)
            elif "GP" in str(type(current_node)):
                if "BGP" in str(type(current_node)):
                    res.append(current_node)
                elif isinstance(current_node.ggps, list):
                    for ggp in current_node.ggps:
                        get_bgps_on_syntax_tree(ggp, res)
                else:
                    get_bgps_on_syntax_tree(current_node.ggps, res)                    
            elif type(current_node) == Filter:
                if "EXISTS" in str(current_node):
                    get_bgps_on_syntax_tree(current_node.expression)
            elif type(current_node.type) == Expression:
                if "EXISTS" in str(current_node):
                    for arg in [current_node.left_expr, current_node.right_expr, current_node.ternary_expr, current_node.quaternary_expr]:
                        if arg is not None:
                            get_bgps_on_syntax_tree(arg, res)
            else:
                raise NotImplementedError(f"current_node: {current_node} {type(current_node)}")

        #首先，获取所有变量
        vars = list(set(re.findall("\?\w+", self.sparql_txt)))
        existence_vars_cnt = 0
        bgps = []
        get_bgps_on_syntax_tree(self.sparql_query.ggp, bgps)
        for bgp in bgps:
            new_triples = []
            for tp in bgp.triples:
                if tp.property.value.startswith("wdt:"):
                    pid = tp.property.value.split(":")[-1]
                    new_existent_var = RDFTerm(value = "?exist_var"+str(existence_vars_cnt), is_const=False)
                    p = RDFTerm(value = "p:" + pid, is_const = True, is_iri = True)
                    ps = RDFTerm(value = "ps:" + pid, is_const = True, is_iri = True)
                    existence_vars_cnt += 1
                    assert(new_existent_var.value not in vars)
                    new_triples.append(TriplePattern(tp.subject, p, new_existent_var))
                    new_triples.append(TriplePattern(new_existent_var, ps, tp.object))
                else:
                    new_triples.append(tp)
            bgp.triples = new_triples

    def _break_wdt_wikidata(self):
        """
        Wikidata 上 wdt: 关系的处理
        这个版本将 ?x wdt:p ?y 拆成 ?x p:p/ps:p ?y，即修改为关系路径
        目前使用该函数，会在 construct_syntax_tree 的 TriplePatterns 处理关系路径
        """

        def break_wdt_into_path(property:RDFTerm):
            #将一个RDFTerm类型的wdt property切分为p/ps
            # if self.dataset is DATASET.LC2:
            #     pid = property.value.split(":")[-1]
            #     p = RDFTerm(value = "p:" + pid, is_const = True, is_iri = True)
            #     ps = RDFTerm(value = "ps:" + pid, is_const = True, is_iri = True)
            # elif self.dataset in [DATASET.QALD, DATASET.SIMULATED_WIKIDATA]:
            #     pid = property.value.split("/")[-1]
            #     p = RDFTerm(value = "http://www.wikidata.org/prop/" + pid, is_const = True, is_iri = True)
            #     ps = RDFTerm(value = "http://www.wikidata.org/prop/statement/" + pid, is_const = True, is_iri = True)
            
            pid = property.value.split("/")[-1]
            p = RDFTerm(value = "http://www.wikidata.org/prop/" + pid, is_const = True, is_iri = True)
            ps = RDFTerm(value = "http://www.wikidata.org/prop/statement/" + pid, is_const = True, is_iri = True)
            path = PropertyPath(p, "/", ps)
            return path
        
        def break_wdt_in_property_path(property_path: PropertyPath):
            #将一个PropertyPath中的各wdt:切分为p/ps
            # for sub_path in [property_path.left_path, property_path.right_path]:
            #     if isinstance(sub_path, RDFTerm):   #单个 RDFTerm
            #         if self.dataset == DATASET.LC2:
            #             if sub_path.value.startswith("wdt:"):
            #                 sub_path = break_wdt_into_path(sub_path)
            #         elif self.dataset == DATASET.QALD:
            #             if sub_path.value.startswith("http://www.wikidata.org/prop/direct"):
            #                 sub_path = break_wdt_into_path(sub_path)
            #     elif isinstance(sub_path, PathTerm): #单元操作符，如 P233+
            #         #看做一个整体，不拆
            #         pass
            #     else:
            #         break_wdt_in_property_path(sub_path)

            # 之前的实现没有把修改后的内容赋值回去
            if isinstance(property_path.left_path, RDFTerm):   #单个 RDFTerm
                # if self.dataset is DATASET.LC2:
                #     if property_path.left_path.value.startswith("wdt:"):
                #         property_path.left_path = break_wdt_into_path(property_path.left_path)
                # elif self.dataset in [DATASET.QALD, DATASET.SIMULATED_WIKIDATA]:
                #     if property_path.left_path.value.startswith("http://www.wikidata.org/prop/direct"):
                #         property_path.left_path = break_wdt_into_path(property_path.left_path)
                
                if property_path.left_path.value.startswith("http://www.wikidata.org/prop/direct"):
                    property_path.left_path = break_wdt_into_path(property_path.left_path)
                return 
            elif isinstance(property_path.left_path, PathTerm): #单元操作符，如 P233+
                #看做一个整体，不拆
                return
            else:
                break_wdt_in_property_path(property_path.left_path)
            
            if isinstance(property_path.right_path, RDFTerm):   #单个 RDFTerm
                # if self.dataset is DATASET.LC2:
                #     if property_path.right_path.value.startswith("wdt:"):
                #         property_path.right_path = break_wdt_into_path(property_path.right_path)
                # elif self.dataset in [DATASET.QALD, DATASET.SIMULATED_WIKIDATA]:
                #     if property_path.right_path.value.startswith("http://www.wikidata.org/prop/direct"):
                #         property_path.right_path = break_wdt_into_path(property_path.right_path)
                if property_path.right_path.value.startswith("http://www.wikidata.org/prop/direct"):
                    property_path.right_path = break_wdt_into_path(property_path.right_path)
                return 
            elif isinstance(property_path.right_path, PathTerm): #单元操作符，如 P233+
                #看做一个整体，不拆
                return
            else:
                break_wdt_in_property_path(property_path.right_path)
        
        def get_bgps_on_syntax_tree(current_node, res:list):
            #返回语法树上所有的TriplePattern，以及它们的父亲
            if "Query" in str(type(current_node)):
                get_bgps_on_syntax_tree(current_node.ggp)
            elif "GP" in str(type(current_node)):
                if "BGP" in str(type(current_node)):
                        res.append(current_node)
                elif isinstance(current_node.ggps, list):
                    for ggp in current_node.ggps:
                        get_bgps_on_syntax_tree(ggp, res)
                else:
                    get_bgps_on_syntax_tree(current_node.ggps, res)                    
            elif type(current_node) == Filter:
                if "EXISTS" in str(current_node):
                    get_bgps_on_syntax_tree(current_node.expression, res)
            elif type(current_node) == Expression:
                if "EXISTS" in str(current_node):
                    for arg in [current_node.left_expr, current_node.right_expr, current_node.ternary_expr, current_node.quaternary_expr]:
                        if arg is not None:
                            get_bgps_on_syntax_tree(arg, res)
            else:
                raise Exception(NotImplementedError(str(current_node.type), str(current_node)))

        bgps = []
        get_bgps_on_syntax_tree(self.sparql_query.ggp, bgps)
        for bgp in bgps:
            for tp in bgp.triples:
                if isinstance(tp.property, RDFTerm):
                    #如果是RDFTerm
                    # if self.dataset is DATASET.LC2:
                    #     if tp.property.value.startswith("wdt:"):
                    #         path = break_wdt_into_path(tp.property)
                    #         tp.property = path
                    # elif self.dataset in [DATASET.QALD, DATASET.SIMULATED_WIKIDATA]:
                    #     if tp.property.value.startswith("http://www.wikidata.org/prop/direct"):
                    #         path = break_wdt_into_path(tp.property)
                    #         tp.property = path   
                    if tp.property.value.startswith("http://www.wikidata.org/prop/direct"):
                        path = break_wdt_into_path(tp.property)
                        tp.property = path            
                elif isinstance(tp.property, PropertyPath):
                    break_wdt_in_property_path(tp.property)
                else:
                    assert(isinstance(tp.property, PathTerm))
        self.update()
    
    def _rewrite_minmax_grailqa(self):
        """
        观察: 
        - GrailQA 中，ARGMIN和ARGMAX只出现在最外层
        - 特征：查询ggp中有一个子查询，子查询是select(min(?x) as ..) 或select(max(?x) as ...)
        处理:
        - 将该子查询删掉，同时添加ORDER BY Limit1即可
        0527 补充: 好像少了一个分隔三元组的 '.'
        """
        argminmax_subq = None
        for ggp in self.sparql_query.ggp.ggps:
            if hasattr(ggp, "ggps"):
                if isinstance(ggp.ggps[0], SelectQuery) and len(ggp.ggps[0].projections) == 1:
                    #考察其selection是否是min(?x) as ?y 或 max(?x) as ?y
                    projection = ggp.ggps[0].projections[0]
                    if projection.oper == "AS" and (projection.left_expr.oper == "MIN" or projection.left_expr.oper == "MAX"):
                        sort_var = projection.right_expr
                        order = projection.left_expr.oper
                        argminmax_subq = ggp
                        break
        #删除这个多余的subq对应节点
        if argminmax_subq is not None:
            self.sparql_query.ggp.ggps.remove(argminmax_subq)
            #添加ORDER BY和Limit1
            if order == "MAX":
                order_by = [Expression(left_expr=copy.deepcopy(sort_var), oper='DESC')]
            else:
                order_by= [copy.deepcopy(sort_var)]
            self.sparql_query.order_by = order_by
            self.sparql_query.limit = 1
            self.update()

    def _rewrite_minmax_kb_binder(self):
        """
        观察: 
        - GrailQA 中，ARGMIN和ARGMAX只出现在最外层
        - 特征：查询ggp中有一个子查询，子查询是select(min(?x) as ..) 或select(max(?x) as ...)
        处理:
        - 将该子查询删掉，同时添加ORDER BY Limit1即可
        0527 补充: 好像少了一个分隔三元组的 '.'
        """
        argminmax_subq = None
        for ggp in self.sparql_query.ggp.ggps:
            if hasattr(ggp, "ggps"):
                if isinstance(ggp.ggps[0], SelectQuery) and len(ggp.ggps[0].projections) == 1:
                    # 出现子查询的唯一情况，就是 ARGMIN, ARGMAX
                    order_by = ggp.ggps[0].order_by
                    argminmax_subq = ggp
                    break

        #删除这个多余的subq对应节点
        if argminmax_subq is not None:
            self.sparql_query.ggp.ggps.remove(argminmax_subq)
            self.sparql_query.order_by = copy.deepcopy(order_by)
            self.sparql_query.limit = 1
            self.update()

    def manage_limit_lc2(self):
        #将LC2中，所有limit5 改为 limit 1
        if self.sparql_query.limit > 0:
            #在树上更换
            self.sparql_query.limit = 1
            self.update()
    
    def _assign_value_clause(self):
        """
        GrailQA 中常通过一个 ValuesClause, 把一个 mentioned entity 放进去
        不管是做叶子节点的替换，还是计算编辑距离，这一写法都带来了困难
        处理方式:
        - 如果语法树中存在 ValuesClause, 且为变量和 item 之间的一一映射
            - 记录这样的映射
            - 语法树中删除这样的 ValuesClause
            - 遍历叶子节点，将变量替换为 item
            - update()
        """
        values_clause = list()
        variable_mapping = dict()
        for ggp in self.sparql_query.ggp.ggps:
            if isinstance(ggp, ValuesClause):
                if len(ggp.variables) == 1 and len(ggp.values) == 1:
                    variable_rdf_term = ggp.variables[0]
                    value_rdf_term:RDFTerm = ggp.values[0][0]
                    values_clause.append(ggp)
                    variable_mapping[variable_rdf_term] = value_rdf_term
        
        for _v_clause in values_clause:
            self.sparql_query.ggp.ggps.remove(_v_clause)
        
        for leaf in self.leafs:
            if isinstance(leaf, RDFTerm):
                if leaf in variable_mapping:
                    item:RDFTerm = variable_mapping[leaf]
                    leaf.is_constant = True
                    leaf.is_bnode = item.is_bnode
                    leaf.is_iri = item.is_iri
                    leaf.is_nil = item.is_nil
                    leaf.is_typed_literal = item.is_typed_literal
                    leaf.lang_tag = item.lang_tag
                    leaf.prefix = item.prefix
                    leaf.value = item.value
                    leaf.xsd_datatype = item.xsd_datatype
                    leaf.expand_syntax_forms(self.sparql_query.prefixes)
        
        self.update()
    
    def _update_equal_filter(self):
        """
        FILTER ?v = RDFTerm, 可以将 SPARQL 中的 ?v 统一替换成 RDFTerm
        """
        filter_clause = list()
        variable_mapping = dict()
        for ggp in self.sparql_query.ggp.ggps:
            if isinstance(ggp, Filter):
                _exp:Expression = ggp.expression
                if isinstance(_exp.left_expr, RDFTerm) and (not _exp.left_expr.is_constant): # variable
                    if _exp.oper == '=': # '='
                        if isinstance(_exp.right_expr, RDFTerm) and _exp.right_expr.is_constant: # constant
                            filter_clause.append(ggp)
                            variable_mapping[_exp.left_expr] = _exp.right_expr
        
        for _f_clause in filter_clause:
            self.sparql_query.ggp.ggps.remove(_f_clause)
        
        for leaf in self.leafs:
            if isinstance(leaf, RDFTerm):
                if leaf in variable_mapping:
                    item:RDFTerm = variable_mapping[leaf]
                    leaf.is_constant = True
                    leaf.is_bnode = item.is_bnode
                    leaf.is_iri = item.is_iri
                    leaf.is_nil = item.is_nil
                    leaf.is_typed_literal = item.is_typed_literal
                    leaf.lang_tag = item.lang_tag
                    leaf.prefix = item.prefix
                    leaf.value = item.value
                    leaf.xsd_datatype = item.xsd_datatype
                    leaf.expand_syntax_forms(self.sparql_query.prefixes)
        
        self.update()


    def _update_time_literal_format(self):
        if self.dataset in [DATASET.CWQ, DATASET.WEBQ, DATASET.QUERYAGENT]:
            new_sparql_txt = self.sparql_txt.replace('xsd:dateTime', '<http://www.w3.org/2001/XMLSchema#dateTime>')
            self.sparql_query:SelectQuery = parser.parse_sparql(new_sparql_txt) # 我们只考虑 SelectQuery
            self.update()
        elif self.dataset in [DATASET.QGG]:
            new_sparql_txt = self.sparql_txt.replace('xsd:date', '<http://www.w3.org/2001/XMLSchema#date>')
            self.sparql_query:SelectQuery = parser.parse_sparql(new_sparql_txt) # 我们只考虑 SelectQuery
            self.update()
        elif self.dataset in [DATASET.BINDER]:
            new_sparql_txt = self.sparql_txt.replace('^^xsd:dateTime', '^^<http://www.w3.org/2001/XMLSchema#dateTime>')
            self.sparql_query:SelectQuery = parser.parse_sparql(new_sparql_txt) # 我们只考虑 SelectQuery
            self.update()
        else:
            raise NotImplementedError(f"dataset: {self.dataset}")
    
    def _add_distinct_flag(self):
        if isinstance(self.sparql_query.projections[0], RDFTerm):
            self.sparql_query.distinct = True
            self.update()
    
    def preprocess_test_suite(self, target_variable_name):
        if self.dataset in [DATASET.QUERYAGENT]:
            self._add_prefix()
        if self.dataset is DATASET.GRAIL:
            self._remove_outer_select_grailqa()
            self._assign_value_clause()
        if self.dataset in [DATASET.CWQ, DATASET.WEBQ, DATASET.QGG, DATASET.QUERYAGENT, DATASET.BINDER]:
            self._update_time_literal_format()
        self._rename_target_variable(target_variable_name)
    
    def preprocess_edit_distance(self, target_variable_name):
        if self.dataset in [DATASET.QUERYAGENT]:
            self._add_prefix()
        if self.dataset is DATASET.BINDER:
            self._rewrite_minmax_kb_binder()
        self._remove_extra_filter_lines()
        self._update_equal_filter()
        if self.dataset is DATASET.GRAIL:
            self._remove_outer_select_grailqa()
            self._rewrite_minmax_grailqa()
            self._assign_value_clause()
        if self.dataset in [DATASET.CWQ, DATASET.WEBQ, DATASET.QGG, DATASET.QUERYAGENT, DATASET.BINDER]:
            self._update_time_literal_format()
        if self.dataset in [DATASET.LC2, DATASET.QALD, DATASET.SIMULATED_WIKIDATA]:
            self._break_wdt_wikidata()
        if self.dataset in [DATASET.LC2, DATASET.QALD, DATASET.QGG]:
            self._add_distinct_flag()
        self._rename_target_variable(target_variable_name)
    
    def update(self):
        pass
        self.sparql_query.expand_syntax_forms()
        self.items = set()
        self.leafs = list()
        self._traverse_leaf_nodes(self.sparql_query)