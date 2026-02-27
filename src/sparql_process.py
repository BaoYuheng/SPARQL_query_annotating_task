from src.awudima import parser
from src.awudima import (
    BGP, GGP, UnionGP, OptionalGP, Bind, Filter, GraphGP, ServiceGP, MinusGP, ValuesClause,
    TriplePattern, RDFTerm, PropertyPath, Operator, Expression,
    SelectQuery, Query, AskQuery,
    unary_operators, unary_expression_list, binary_operators,
    ternary_operators, quaternary_operators, aggregate_functions
) 
from src.utils import (
    load_json, dump_json, setup_custom_logger
)
from common import DATASET
from tqdm import tqdm
from functools import cmp_to_key
from collections import defaultdict
from queue import Queue
import re
from zss import simple_distance
from anytree import Node, RenderTree


logger = setup_custom_logger("data/test/sparql.txt")

#处理关系的逆、关系的domain和range
REVERSE_RELATION_PATH = "data/input/common/freebase_reverse_relation_map.json" 
reverse_relations = {}
RELATION_DOMAIN_RANGE_PATH = "data/input/common/fb_relations_domain_range_label.json"
relation_domain_range = {}
WIKIDATA_REVERSE_RELATION_PATH = "data/input/common/reverse_properties_wikidata"
wikidata_reverse_relations = {} 

def get_relation_label(relation_iri):
    return relation_iri.split("/")[-1]

def get_relation_iri_prefix(relation_iri):
    return "/".join(relation_iri.split("/")[:-1])

def load_relation_dicts():
    '''
    收集到的数据中，存在 A 的逆关系是 B, 但是 B 的逆关系不是 A 的情况
    这应该是数据有一些缺漏？我们在读取数据的时候，如果 A 的逆关系是 B, 则自动补充 B 的逆关系是 A
    '''
    for (key, value) in load_json(REVERSE_RELATION_PATH).items():
        if len(value) == 1: # 验证过了，所有关系至多有一个逆关系
            reverse_relations[key] = value[0]
            reverse_relations[value[0]] = key

    with open(RELATION_DOMAIN_RANGE_PATH, 'r') as f:
        for k, v in load_json(RELATION_DOMAIN_RANGE_PATH).items():
            relation_domain_range[k] = v

def load_reverse_relation_wikidata():
    with open(WIKIDATA_REVERSE_RELATION_PATH, 'r') as f:
        for line in f:
            '''
            Wikidata 的记录中，确实存在 A 的逆关系是 B, 但是 B 的逆关系不是 A 的情况；
            但是这和逆关系的定义相违背, 我们认为可能是数据质量问题，因此在这里，都假设逆关系是对称的
            '''
            rel, rel_rev = line.split('\t')[0], line.split('\t')[1].replace('\n', '')
            wikidata_reverse_relations[rel] = rel_rev
            wikidata_reverse_relations[rel_rev] = rel
        
def pre_process_sparql(sparql_query, dataset):
    lines = sparql_query.split('\n')
    if dataset in [DATASET.CWQ, DATASET.WEBQ]:
        try:
            where_clause_idx = lines.index('WHERE {')
            target_idx_list = [where_clause_idx + 1, where_clause_idx + 2]
            filtered_lines = [
                ln for (idx, ln) in enumerate(lines)
                if (not ln.startswith('FILTER')) or (idx not in target_idx_list)
            ]
        except:
            filtered_lines = lines
    elif dataset == DATASET.GRAIL:
        filtered_lines = [
            ln for (idx, ln) in enumerate(lines) 
            if (not ln.startswith('FILTER')) or (idx != len(lines) - 3)
        ]
    else:
        raise NotImplementedError(f"dataset: {dataset}")
    return "\n".join(filtered_lines)

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
    def __init__(self, sparql_txt, dataset_type:DATASET =DATASET.CWQ ):
        self.sparql_query:Query = parser.parse_sparql(sparql_txt)
        self.prefixes = self.sparql_query.prefixes
        self.variable_mapping = dict()
        self.dataset_type = dataset_type
        self.leaves_item = set()
        #如果第一次初始化，载入关系相关dict
        if len(relation_domain_range) == 0 or len(reverse_relations) == 0:
            load_relation_dicts()
        if len(wikidata_reverse_relations) == 0:
            load_reverse_relation_wikidata()
    
    def construct_syntax_tree(self):
        #tree_root = GGPTreeNode(self.sparql_query, None)
        # tree_root = GGPTreeNode("ROOT",None)
        # self._construct_syntax_tree(tree_root, self.sparql_query)
        # self.tree_root = tree_root
        # return self.tree_root

        tree_root = GGPTreeNode("ROOT", None)
        count_flag = False
        if self.dataset_type is DATASET.GRAIL: # GrailQA 最外层的查询，只是给目标变量换个名字，可忽略
            new_sparql_root = None
            if not isinstance(self.sparql_query, SelectQuery):
                raise NotImplementedError(f"root: {self.sparql_query} {type(self.sparql_query)}")
            for _ggp in self.sparql_query.ggp.ggps:
                if isinstance(_ggp, SelectQuery): # 找到内层的首个 SelectQuery, 其实就是把最外层的 SelectQuery 扔掉
                    new_sparql_root = _ggp
            try: # 不够严谨，可能出现误判
                if isinstance(self.sparql_query.projections[0], Expression):

                    left_expr = self.sparql_query.projections[0].left_expr
                    if left_expr.oper.upper() == 'COUNT':
                        count_flag = True
            except Exception as e:
                # print(f"error when searching for count flag: {e} {self.sparql_query}")
                pass

            if new_sparql_root is None:
                raise NotImplementedError(f"new_sparql_root not found: {self.sparql_query}")
            self.sparql_query = new_sparql_root

        self._construct_syntax_tree(tree_root, self.sparql_query, count_flag)
        self.tree_root = tree_root
        return self.tree_root
    
    def _construct_syntax_tree(self, parent_node:GGPTreeNode, current, count_flag=False):
        if isinstance(current, ServiceGP) or isinstance(current, MinusGP) or isinstance(current, OptionalGP):
            # 数据集中不应该存在上述 GP
            raise Exception(f"current: {current}")
        elif isinstance(current, Bind):
            current_node = GGPTreeNode(current, parent_node)
            parent_node.add_child(current_node)
            self._construct_syntax_tree(current_node, current.expression)
            return
        # Terminal node
        elif isinstance(current, RDFTerm):
            if current in self.variable_mapping:
                current = self.variable_mapping[current]
            # current.expand_syntax_forms(self.prefixes) # 前缀替换
            current_node = GGPTreeNode(current, parent_node)
            parent_node.add_child(current_node)
            self.leaves_item.add(current.to_str())
            return
        
        elif isinstance(current, PropertyPath):
            # current.expand_syntax_forms(self.prefixes)
            current_node = GGPTreeNode(current, parent_node)
            parent_node.add_child(current_node)
            # self.leaves_item.add(current.to_str())
            return
            # raise NotImplementedError(f"PropertyPath: {current}")
        elif isinstance(current, ValuesClause):
            if len(current.variables) == 1 and len(current.values) == 1:
                # 直接完成 variable 到 value 的替换
                variable_rdf_term = current.variables[0]
                if len(current.values[0]) != 1:
                    raise NotImplementedError(f"ValuesClause: {current}")
                value_rdf_term:RDFTerm = current.values[0][0]
                self.variable_mapping[variable_rdf_term] = value_rdf_term
            else:
                current_node = GGPTreeNode(current, parent_node)
                parent_node.add_child(current_node)
                for value in current.values:
                    self._construct_syntax_tree(current_node, value)
            return

        # Filter 底下的 Expression, 视作一个复杂元素；其他情况下的 Expression, 视作 terminal node
        # elif isinstance(current, Expression):
        #     current_node = GGPTreeNode(current, parent_node)
        #     parent_node.add_child(current_node)
        #     return

        elif isinstance(current, Expression):   #添加表达Expression的节点
            current_node = GGPTreeNode(current, parent_node)
            parent_node.add_child(current_node)
            # 暂时只处理二元 Expression
            for sub_expr in [current.left_expr, current.right_expr, current.ternary_expr, current.quaternary_expr]:
                if sub_expr is not None:
                    self._construct_syntax_tree(current_node, sub_expr)

        elif isinstance(current, TriplePattern):
            current_node = GGPTreeNode(current, parent_node)
            parent_node.add_child(current_node)
            self._construct_syntax_tree(current_node, current.subject)
            self._construct_syntax_tree(current_node, current.property)
            self._construct_syntax_tree(current_node, current.object)
        
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
            ggp_node = GGPTreeNode(current, parent_node)
            parent_node.add_child(ggp_node)
            self._construct_syntax_tree(ggp_node,current.ggp)
        elif isinstance(current, SelectQuery):
            #修改：一个Query node, 下跟solution modifiers和ggp
            query_node = GGPTreeNode(current, parent_node)
            parent_node.add_child(query_node)
            '''暂时只处理单个投影变量的情况'''
            if len(current.projections) != 1:
                raise NotImplementedError(f"ggp.projections: {current.projections}")
            #projection_node = GGPTreeNode(current.projections[0], parent_node)
            #parent_node.add_child(projection_node)
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
            if count_flag:
                node = GGPTreeNode("COUNT", query_node)
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
            print(f"current: {current}")
            return

        else:
            print(current)
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
            if self.dataset_type in [DATASET.CWQ, DATASET.GRAIL, DATASET.WEBQ, DATASET.SIMULATED_FREEBASE]:
                if get_relation_label(triple.node.property.value) in reverse_relations:
                    reversed_p_label = reverse_relations[get_relation_label(triple.node.property.value)]
                    reversed_p = RDFTerm(value = get_relation_iri_prefix(triple.node.property.value) + "/" + reversed_p_label,
                                        is_iri = True,
                                        is_const = True
                    )
                    #增加一个新的三元组
                    new_triple = TriplePattern(triple.node.object, reversed_p, triple.node.subject)
                    self._construct_syntax_tree(parent, new_triple)
            elif self.dataset_type in [DATASET.LC2, DATASET.QALD, DATASET.SIMULATED_WIKIDATA]:
                if get_relation_label(triple.node.property.value) in wikidata_reverse_relations:
                    reversed_p_label = wikidata_reverse_relations[get_relation_label(triple.node.property.value)]
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
            if get_relation_label(triple.node.property.value) in relation_domain_range:
                # 已经是一个类型约束的三元组，则不需要进一步处理 domain 和 range
                if get_relation_label(triple.node.property.value) in ['type.object.type', 'type.type.instance']:
                    continue
                if 'domain' in relation_domain_range[get_relation_label(triple.node.property.value)]:
                    if not self._node_is_constant(triple.node.subject): # 常量无需添加类型约束
                        domain_label = relation_domain_range[get_relation_label(triple.node.property.value)]['domain']
                        domain = RDFTerm(get_relation_iri_prefix(triple.node.property.value) + "/" + domain_label,
                                        is_const = True, 
                                        is_iri = True)

                        subject_type_triple = TriplePattern(triple.node.subject, rdf_type, domain)
                        self._construct_syntax_tree(parent, subject_type_triple)
                if 'range' in relation_domain_range[get_relation_label(triple.node.property.value)]:
                    if not self._node_is_constant(triple.node.object):
                        range_label = relation_domain_range[get_relation_label(triple.node.property.value)]['range']
                        range =  RDFTerm(get_relation_iri_prefix(triple.node.property.value) + "/" + range_label,
                                        is_const = True, 
                                        is_iri = True)
                        object_type_triple = TriplePattern(triple.node.object, rdf_type, range)
                        self._construct_syntax_tree(parent, object_type_triple)
    
    def _node_is_constant(self, node):
        # 特殊情况，通过 ValuesClause, 单个常量被赋值给一个变量
        return node.is_constant or (node in self.variable_mapping)

    def _reorder_children(self, current_node:GGPTreeNode):
        #目前只考虑GGP与UnionGP内元素的重排序。同时，对于这些元素，如果它们存在相同的两个孩子，那么只保留一个
        for child in current_node.children:
            self._reorder_children(child)
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
                else:
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
        triples = [ggp.node for ggp in triples]
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
        self._rename_vars(root_node)
        #根据字典序重排有交换律的各节点孩子
        #添加关系的逆、domain、range

        # self._add_reverse_relations(root_node)
        # self._add_domain_range(root_node)

        # 感觉两者的顺序应该调换，先添加 domain range 相关的三元组，然后还要根据添加后的三元组，补充相应的逆关系
        self._add_domain_range(root_node)
        self._add_reverse_relations(root_node)

        #去重并重排孩子
        self._reorder_children(root_node)
        #root_node.visualize()
        return 0
    

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


def merge_dicts(dict1, dict2):
    merged_dict = dict1
    if dict2:
        for key, value in dict2.items():
            if key in merged_dict:
                merged_dict[key] += value
            else:
                merged_dict[key] = value
    return merged_dict

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
    if oper_str in ['<', '>', '<=', '>=', '=']:
        return True
    #需要增加xsd:type
    if "xsd:" in oper_str:
        return True
    return False

class TreeEditDistance:
    @staticmethod
    def get_label(node):
        return node.to_str()
    
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
    def get_edit_distance(sparql_txt_0, sparql_txt_1):
        try:
            sparql_query_0 = parser.parse_sparql(sparql_txt_0)
            tree_root_0 = GGPTreeNode(sparql_query_0, None)
            _construct_syntax_tree(tree_root_0, sparql_query_0.ggp)
            print(tree_root_0)

            sparql_query_1 = parser.parse_sparql(sparql_txt_1)
            tree_root_1 = GGPTreeNode(sparql_query_1, None)
            _construct_syntax_tree(tree_root_1, sparql_query_1.ggp)
            print(tree_root_1)
        except Exception as e:
            logger.error(f"sparql_txt_0: {sparql_txt_0}\n sparql_txt_1:{sparql_txt_1}\n error:{e}")

        distance = simple_distance(
            tree_root_0, 
            tree_root_1,
            get_label=TreeEditDistance.get_label,
            get_children=TreeEditDistance.get_children,
            label_dist=TreeEditDistance.label_dist
        )
        print(distance)

def tree_constructor_test():
    test_cases = [
        # (
        #     "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX : <http://rdf.freebase.com/ns/> \nSELECT (?x0 AS ?value) WHERE {\nSELECT DISTINCT ?x0  WHERE { \n?x0 :type.object.type :measurement_unit.absorbed_dose_unit . \nVALUES ?x1 { \"1.0\"^^<http://www.w3.org/2001/XMLSchema#float> } \n?x0 :measurement_unit.absorbed_dose_unit.dose_in_grays ?x1 . \nFILTER ( ?x0 != ?x1  )\n}\n}",
        #     DATASET.GRAIL
        # ), # 0. ValueClause
        # (
        #     "#MANUAL SPARQL\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT  ?x\nWHERE {\n  ns:m.0fjp3 ns:sports.sports_championship.events ?x . # World Series\n  {\n    { ?x ns:sports.sports_championship_event.runner_up ns:m.01ync . } # Colorado Rockies\n    UNION\n    { ?x ns:sports.sports_championship_event.champion ns:m.01ync . } # Colorado Rockies\n  }\n  ?x ns:time.event.start_date ?d .\n}",
        #     DATASET.WEBQ
        # ), # 1. UnionGP + GGP
        # (
        #     "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX : <http://rdf.freebase.com/ns/> \nSELECT (?x0 AS ?value) WHERE {\nSELECT DISTINCT ?x0  WHERE { \n?x0 :type.object.type :boats.ship_builder . \nVALUES ?x1 { :m.0444f9 } \n?x0 :boats.ship_builder.ships_built ?x1 . \nFILTER ( ?x0 != ?x1  )\n}\n}",
        #     DATASET.GRAIL
        # ), # 2. 子查询
        # (
        #     "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX : <http://rdf.freebase.com/ns/> \nSELECT (?x0 AS ?value) WHERE {\nSELECT DISTINCT ?x0  WHERE { \n?x0 :type.object.type :meteorology.beaufort_wind_force . \nFILTER (?x1 < \"7.0\"^^<http://www.w3.org/2001/XMLSchema#float>)\n?x0 :meteorology.beaufort_wind_force.minimum_wind_speed_km_h ?x1 . \nFILTER ( ?x0 != ?x1  )\n}\n}",
        #     DATASET.GRAIL
        # ), # 3. Filter
        # (
        #     "PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?x\nWHERE {\nFILTER (?x != ns:m.0jnq8)\nFILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))\nns:m.0jnq8 ns:sports.sports_team.championships ?x .\n?x ns:time.event.end_date ?sk0 .\n}\nORDER BY DESC(xsd:datetime(?sk0))\nLIMIT 1\n",
        #     DATASET.WEBQ
        # ), # 4. Solution Modifier, Order by
        # (
        #     "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX : <http://rdf.freebase.com/ns/> \nSELECT (?x0 AS ?value) WHERE {\nSELECT DISTINCT ?x0  WHERE { \n?x0 :type.object.type :cvg.computer_game_rating . \nVALUES ?x1 { :m.03h08yf } \n?x0 :cvg.computer_game_rating.rating_system ?x1 . \nFILTER ( ?x0 != ?x1  )\n}\n}",
        #     DATASET.GRAIL
        # ), # 5. 前缀替换
        # (
        #     "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX : <http://rdf.freebase.com/ns/> \nSELECT (?x0 AS ?value) WHERE {\nSELECT DISTINCT ?x0  WHERE { \n?x0 :type.object.type :measurement_unit.fuel_economy_unit . \nFILTER (?x1 <= \"0.01\"^^<http://www.w3.org/2001/XMLSchema#float>)\n?x0 :measurement_unit.fuel_economy_unit.economy_in_litres_per_kilometre ?x1 . \nFILTER ( ?x0 != ?x1  )\n}\n}",
        #     DATASET.GRAIL
        # ), # 6. 二元 Filter Expression
        (
            "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX : <http://rdf.freebase.com/ns/> \nSELECT (COUNT(?x0) AS ?value) WHERE {\nSELECT DISTINCT ?x0  WHERE { \n?x0 :type.object.type :exhibitions.exhibition_subject . \n?x1 :type.object.type :exhibitions.exhibition . \nVALUES ?x2 { :m.05x9lqc } \n?x0 :exhibitions.exhibition_subject.exhibitions_created_about_this_subject ?x1 . \n?x2 :exhibitions.exhibition_curator.exhibitions_curated ?x1 . \nFILTER ( ?x0 != ?x1 && ?x0 != ?x2 && ?x1 != ?x2  )\n}\n}",
            DATASET.GRAIL
        ), # 7. COUNT
    ]
    cwq_test_path = "data/input/cwq/cwq_dev_linking.json"
    grailqa_dev_path = "data/input/GrailQA_v1.0/grailqa_v1.0_dev_linking.json"
    webqsp_train_path = "data/input/WebQSP/webqsp_train_linking.json"
    cwq_sparqls = [item['golden_sparql_query'] for item in load_json(cwq_test_path)]
    webqsp_sparqls = [item['golden_sparql_query'] for item in load_json(webqsp_train_path)]
    grailqa_sparqls = [item['golden_sparql_query'] for item in load_json(grailqa_dev_path)]
    # test_cases = [(item, DATASET.GRAIL) for item in grailqa_sparqls]
    # test_cases = [(item, DATASET.WEBQ) for item in webqsp_sparqls]
    # test_cases = [(item, DATASET.CWQ) for item in cwq_sparqls]
    #test_cases = [test_cases[idx] for idx in [3]]
    for (idx, _case) in tqdm(enumerate(test_cases)):
        try:
            #print("-------------------------------------------------------------------------------")
            #print(idx)
            tree_constructor=TreeConstructor(pre_process_sparql(_case[0], _case[1]), _case[1])
            # syntax_tree = tree_constructor.construct_syntax_tree()
            
            # print(TreeConstructor(
            #     pre_process_sparql(_case[0], _case[1])
            # ).construct_syntax_tree())
            tree_constructor.canonicalize_syntax_tree(root_node=syntax_tree)
            # syntax_tree.visualize()
            # print(syntax_tree.serialize(tree_constructor.variable_mapping))
            syntax_tree_editor = SyntaxTreeEditor(pre_process_sparql(_case[0], _case[1]))
            print(syntax_tree_editor.get_sparql_text())
            syntax_tree_editor.update_leaf_value(
                'http://rdf.freebase.com/ns/m.05x9lqc', 'http://rdf.freebase.com/ns/m.just'
            )
            print(syntax_tree_editor.get_sparql_text())
        except Exception as e:
            print(e)
            print(_case[0])
            import traceback
            traceback.print_exc()
            pass

class SyntaxTreeEditor:
    def __init__(self, sparql_txt):
        self.sparql_query:Query = parser.parse_sparql(sparql_txt)
        self.leafs = list()
        self._traverse_leaf_nodes(self.sparql_query)
    
    def update_leaf_value(self, old_value, new_value):
        """
        遍历 self.leafs, 找到值和 old_value 一致的叶子节点，并将其更新为 new_value
        更新成功返回 True, 更新失败（未找到这样的节点）返回 False
        """
        flag = False
        for leaf in self.leafs:
            if isinstance(leaf, RDFTerm):
                if leaf.value == old_value:
                    leaf.value = new_value
                    leaf.expand_syntax_forms(self.sparql_query.prefixes)
                    flag = True
            elif isinstance(leaf, ValuesClause):
                for (idx, v_list) in enumerate(leaf.values):
                    for (nested_idx, v) in enumerate(v_list):
                        if v.value == old_value:
                            leaf.values[idx][nested_idx].value = new_value
                            flag = True
                leaf.expand_syntax_forms(self.sparql_query.prefixes)
            else:
                raise NotImplementedError(f"leaf: {leaf}")
        return flag
    
    def replace_leaf_with_variable(self, old_value, variable_name):
        """
        遍历 self.leafs, 找到值和 old_value 一致的叶子节点，并将其更新为 new_value
        更新成功返回 True, 更新失败（未找到这样的节点）返回 False
        """
        flag = False
        for leaf in self.leafs:
            if isinstance(leaf, RDFTerm):
                if leaf.value == old_value:
                    leaf.value = variable_name
                    leaf.is_constant = False
                    leaf.expand_syntax_forms(self.sparql_query.prefixes)
                    flag = True
            elif isinstance(leaf, ValuesClause):
                pass # 如果要做替换，ValuesClause 还是应该提前处理掉
                # for (idx, v_list) in enumerate(leaf.values):
                #     for (nested_idx, v) in enumerate(v_list):
                #         if v.value == old_value:
                #             leaf.values[idx][nested_idx].value = variable_name
                #             flag = True
                # leaf.expand_syntax_forms(self.sparql_query.prefixes)
            else:
                raise NotImplementedError(f"leaf: {leaf}")
        return flag
    
    def get_sparql_text(self):
        return self.sparql_query.to_str()
    
    def _traverse_leaf_nodes(self, current):
        if isinstance(current, Bind) or isinstance(current, ServiceGP) or isinstance(current, MinusGP) or isinstance(current, OptionalGP):
            # 数据集中不应该存在上述 GP
            raise Exception(f"current: {current}")
        
        # Terminal node
        elif isinstance(current, RDFTerm):
            current.expand_syntax_forms(self.sparql_query.prefixes)
            self.leafs.append(current)
            return
        
        elif isinstance(current, PropertyPath):
            raise NotImplementedError(f"PropertyPath: {current}")
        
        elif isinstance(current, ValuesClause): # ValuesClause 视作特殊的 leaf node
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
        
        elif isinstance(current, AskQuery) or isinstance(current, SelectQuery):
            # projection 里面只有变量，应该不做替换，不关注
            # distinct 不关注
            # order by 和 limit 理论上是有关注的必要，但是对于目前的数据集可以先不考虑吧
            self._traverse_leaf_nodes(current.ggp)
        
        elif isinstance(current, GGP) or isinstance(current, UnionGP):
            self._traverse_leaf_nodes(current.ggps)
        
        elif is_operator(current): # 暂时先不考虑替换 operator
            # self.leafs.append(current)
            return
        
        elif current is None: # 有一些 Expression 的部分项为 None, 放在这处理
            print(f"current: {current}")
            return

        else:
            print(current)
            raise NotImplementedError(f"current: {current} {type(current)}")
        


    

"""
数据集分析的相关代码
"""
def get_sparql_prefix():
    """
    结论: 对于 CWQ / WebQSP, 如果 'WHERE {' 后面的两行是以 FILTER 开头的，则去除
    """
    filter_set = set()
    prefix_set = set()
    """CWQ"""
    data_file_list = [
        f'data/input/cwq/cwq_{split}_linking.json' for split in ['train', 'dev', 'test']
    ]
    # """WebQSP"""
    # data_file_list = [
    #     f'data/input/WebQSP/webqsp_{split}_linking.json' for split in ['train', 'test']
    # ]
    for data_file in data_file_list:
        data = load_json(data_file)
        for item in tqdm(data):
            sparql = item["golden_sparql_query"]
            lines = sparql.split('\n')
            try:
                where_clause = lines.index('WHERE {')
                filter_lines = [
                    ln for (idx, ln) in enumerate(lines) 
                    if ln.startswith('FILTER') and (idx not in [where_clause+1, where_clause+2])
                ]
                filter_set.update(filter_lines)

                prefix_set.update([
                    ln for (idx, ln) in enumerate(lines) 
                    if ln.startswith('FILTER') and (idx in [where_clause+1, where_clause+2])
                ])
            except:
                '''这些 MANUAL SPARQL 中，没有添加额外的 FILTER'''
                pass
    """
    基本上，数据集中统一添加的 FILTER, 只会出现在 WHERE { 后面的两行
    特殊情况: FILTER (?x != ns:m.09b6zr)?x ns:film.actor.film ?c . 
    - 有一些 MANUAL SPARQL, 上述 FILTER 出现在中间，而且 m.09b6zr 没有在 SPARQL 其他部分出现，那么这种是不能删除的（表达了一定语义）
    对应的问题是 Who did George W. Bush run against his second term that held his governmental position the earliest?
    - m.09b6zr 是 George W. Bush，run against, 需要从竞选者中排除自己，算是语义上有表达
    """
    logger.info(f"filter_set: {len(filter_set)}")
    for filter_item in filter_set:
        if "xsd:" not in filter_item:
            logger.info(filter_item)
    logger.info(f"prefix set: {len(prefix_set)}")
    for filter_item in prefix_set:
        logger.info(filter_item)

def get_sparql_prefix_grailqa():
    """
    结论: 对于 GrailQA, 如果倒数第三行是 FILTER 开头的，则去除
    """
    filter_set = set()
    prefix_set = set()
    data_file_list = [
        f'data/input/GrailQA_v1.0/grailqa_v1.0_{split}.json' for split in ['train', 'dev']
    ]
    for data_file in data_file_list:
        data = load_json(data_file)
        for item in tqdm(data):
            sparql = item["sparql_query"]
            lines = sparql.split('\n')
            try:
                filter_lines = [
                    ln for (idx, ln) in enumerate(lines) 
                    if ln.startswith('FILTER') and (idx != len(lines) - 3)
                ]
                filter_set.update(filter_lines)

                prefix_set.update([
                    ln for (idx, ln) in enumerate(lines) 
                    if ln.startswith('FILTER') and (idx == len(lines) - 3)
                ])
            except:
                logger.info(sparql)
    """
    GrailQA 的 SPARQL 比较规整，统一对变量添加 != 约束
    """
    logger.info(f"filter_set: {len(filter_set)}")
    # for filter_item in filter_set:
    #     logger.info(filter_item)
    logger.info(f"prefix set: {len(prefix_set)}")
    for filter_item in prefix_set:
        logger.info(filter_item)

def check_special_item_in_sparql_grailqa():
    '''GrailQA'''
    src_data = load_json("data/input/GrailQA_v1.0/grailqa_v1.0_train_linking.json")
    sparql_list = [
        pre_process_sparql(item["golden_sparql_query"], DATASET.GRAIL) for item in src_data
    ]
    for sparql in tqdm(sparql_list):
        try:
            special_item = _detect_special_items(sparql)
            if len(special_item) > 0:
                logger.info(f"sparql: {sparql}")
                logger.info(f"special_item: {special_item}")
                logger.info("")
        except Exception as e:
            logger.error(f"exception: {e};")
            logger.error(f"sparql: {sparql}")
            logger.error("")

def check_special_item_in_sparql():
    '''CWQ'''
    # src_data = load_json("data/input/cwq/cwq_train_linking.json")
    # sparql_list = [
    #     process_sparql(item["golden_sparql_query"], DATASET.CWQ) for item in src_data
    # ]
    '''WebQSP'''
    src_data = load_json("data/input/WebQSP/webqsp_train_linking.json")
    sparql_list = [
        pre_process_sparql(item["golden_sparql_query"], DATASET.WEBQ) for item in src_data
    ]
    for sparql in tqdm(sparql_list):
        try:
            special_item = _detect_special_items(sparql)
            if len(special_item) > 0:
                logger.info(f"sparql: {sparql}")
                logger.info(f"special_item: {special_item}")
                logger.info("")
        except Exception as e:
            logger.error(f"exception: {e};")
            logger.error(f"sparql: {sparql}")
            logger.error("")

def _detect_special_items(sparql_txt):
    query = parser.parse_sparql(sparql_txt)
    return _traverse_query(query)

def _traverse_query(sparql_query):
    special_items_set = set()
    def _traverse(ggp_list):
        for ggp in ggp_list:
            if isinstance(ggp, BGP) or isinstance(ggp, Filter) or isinstance(ggp, ValuesClause):
                continue
            elif isinstance(ggp, Bind) or isinstance(ggp, ServiceGP) or isinstance(ggp, MinusGP) or isinstance(ggp, OptionalGP):
                special_items_set.add(ggp)
            elif isinstance(ggp, UnionGP):
                _traverse(ggp.ggps)
            elif isinstance(ggp, GGP):
                _traverse(ggp.ggps)
            elif isinstance(ggp, SelectQuery): # 子查询
                _traverse(ggp.ggp.ggps)

    _traverse(sparql_query.ggp.ggps)
    return special_items_set

def sparql_syntax_tree():
    '''WebQSP'''
    src_data = load_json("data/input/WebQSP/webqsp_train_linking.json")
    sparql_list = [
        pre_process_sparql(item["golden_sparql_query"], DATASET.WEBQ) for item in src_data
    ]
    sparql_list = [
        sparql_list[idx] for idx in [2,5]
    ]
    for sparql in tqdm(sparql_list):
        try:
            syntax_tree = construct_syntax_tree(sparql)
            print(sparql)
            for line in syntax_tree.to_str():
                print(line)
            print()
        except:
            print(sparql)


if __name__=='__main__':
    # check_special_item_in_sparql_grailqa()
    # check_special_item_in_sparql()
    # get_sparql_prefix()
    # get_sparql_prefix_grailqa()

    # sparql_syntax_tree()

    # TreeEditDistance.get_edit_distance(
    #     pre_process_sparql("PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?x\nWHERE {\nFILTER (?x != ns:m.0jmfb)\nFILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))\nns:m.0jmfb ns:sports.sports_team.championships ?x .\n?x ns:sports.sports_championship_event.result \"4 - 0\"@en . \n}\n", DATASET.CWQ),
    #     pre_process_sparql("PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX : <http://rdf.freebase.com/ns/> \nSELECT (?x0 AS ?value) WHERE {\nSELECT DISTINCT ?x0  WHERE { \n?x0 :type.object.type :measurement_unit.measurement_system . \nVALUES ?x1 { :g.122rvtfy } \n?x0 :measurement_unit.measurement_system.current_density_units ?x1 . \nFILTER ( ?x0 != ?x1  )\n}\n}", DATASET.GRAIL)
    # )

    # tree_constructor_test()
    c = TreeConstructor('select ?x where { ?x p:P161/ps:P161 wd:Q262502. ?x p:P580/ps:P580 ?time. Filter(year(?time)=2016 && ?x!="sdaf").}')
    c.construct_syntax_tree()
    print(c.leaves_item)