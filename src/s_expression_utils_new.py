import re
from typing import List, Tuple, Dict
from src.common import (
    WikidataConstantForConstruction, WIKIDATA_CONSTANT_TYPE,
    FreebaseConstantForConstruction, FREEBASE_CONSTANT_TYPE,
    WIKIDATA_PREFIX_LIST
)


def JOIN(relation, exp):
    return f"(JOIN {relation} {exp})"

def AND(exp1, exp2):
    return f"(AND {exp1} {exp2})"

def ARG(op, exp, relation):
    '''[EXP] 作为 {exp} 参数位置的占位符'''
    assert op in ["ARGMAX", "ARGMIN"]
    return f"({op} {exp} {relation})"

def CMP(op, relation, exp):
    assert op in ["LT", "LE", "GT", "GE", "EQ"], print(op)
    # assert op in ["lt", "le", "gt", "ge"]
    return f"({op} {relation} {exp})"

def COUNT(exp):
    return f"(COUNT {exp})"

def R(relation):
    return f"(R {relation})"

def sexp_to_sparql(lisp_program: str):
    '''
    变量说明:
    - ?x{i}: entity 集合
    - ?y{i}: 中间变量
    - ?lit{i}: string 集合
    - ?v{i}: time / number 集合
    - ?arg{i}: ARGMIN / ARGMAX 的排序目标
    '''
    clauses = []
    # 仅支持出现一次 ARGMAX 的场景
    order_clauses = None # [变量名，方向, (LIMIT) n] --> [?arg0, ASC / DESC, 1]
    entities = set()  # collect entites for filtering
    classes = set()
    identical_variables_r = {}  # key should be larger than value
    expression = lisp_to_nested_expression(lisp_program)
    count = False

    sub_programs = _linearize_lisp_expression(expression, [0])
    question_var = len(sub_programs) - 1
    
    def get_root(var: int):
        while var in identical_variables_r:
            var = identical_variables_r[var]

        return var

    for i, subp in enumerate(sub_programs):
        '''
        标记说明:
        - x{i} 表示一般变量
        - y{i} 多跳关系中的中间变量
        - z{i} 多跳关系中的目标变量
        - v{i} 表示一个指示 TIME / QUANTITY 的变量
        - arg{i} 是 ORDER BY 操作符的对象
        - lit{i} 表示一个指示 STRING 的变量
        '''
        i = str(i)
        if subp[0] == 'JOIN':
            '''
            subp[1] 我认为只有两种选择:
            - relation
            - R relation
            这两者只有 SPARQL 里面三元组方向的区别

            无论 subp[1] 是什么, subp[2] 有如下选择
            - item: entity / class / literal
            - #n: 表示一个嵌套的子结构, 有可能代表关系的组合
            - relation --> 合起来表示一个多跳的关系
            - R relation
            '''
            if isinstance(subp[1], list):  # R relation
                if subp[2][:2] in ["m.", "g."]:  # entity
                    clauses.append("ns:" + subp[2] + " ns:" + subp[1][1] + " ?x" + i + " .")
                    entities.add(subp[2])
                elif re.fullmatch("[a-zA-Z_0-9]+\.[a-zA-Z_0-9]+", subp[2]): # class
                    clauses.append("ns:" + subp[2] + " ns:" + subp[1][1] + " ?x" + i + " .")
                    classes.add(subp[2])
                elif subp[2][0] == '#':  # 嵌套子结构
                    clauses.append(f"?x{get_root(int(subp[2][1:]))} ns:{subp[1][1]} ?x{i} .")
                elif FreebaseConstantForConstruction.get_constant_type(subp[2]) in [FREEBASE_CONSTANT_TYPE.TIME, FREEBASE_CONSTANT_TYPE.QUANTITY, FREEBASE_CONSTANT_TYPE.STRING]: # literal
                    if FreebaseConstantForConstruction.get_constant_type(subp[2])  is FREEBASE_CONSTANT_TYPE.STRING:
                        subp[2] = subp[2].replace("@en", "")
                        clauses.append(f"?lit{i} ns:{subp[1][1]} ?x{i} . FILTER (isLiteral( ?lit{i} )) . FILTER (str( ?lit{i} ) = {subp[2]} ) .")
                    # 其他类型的 Literal, 传入 S-expression 时就已经处理好了
                    else:
                        clauses.append(subp[2] + " ns:" + subp[1][1] + " ?x" + i + " .")
                else: 
                    raise Exception(f"subp: {subp}")

            elif isinstance(subp[1], str): # relation
                if subp[2][:2] in ["m.", "g."]:  # entity
                    clauses.append("?x" + i + " ns:" + subp[1] + " ns:" + subp[2] + " .")
                    entities.add(subp[2])
                elif re.fullmatch("[a-zA-Z_0-9]+\.[a-zA-Z_0-9]+", subp[2]): # class
                    clauses.append("?x" + i + " ns:" + subp[1] + " ns:" + subp[2] + " .")
                    classes.add(subp[2])
                elif subp[2][0] == '#':  # variable
                    clauses.append(f"?x{i} ns:{subp[1]} ?x{get_root(int(subp[2][1:]))} .")
                elif FreebaseConstantForConstruction.get_constant_type(subp[2]) in [FREEBASE_CONSTANT_TYPE.TIME, FREEBASE_CONSTANT_TYPE.QUANTITY, FREEBASE_CONSTANT_TYPE.STRING]:  # literal
                    if FreebaseConstantForConstruction.get_constant_type(subp[2]) is FREEBASE_CONSTANT_TYPE.STRING:
                        subp[2] = subp[2].replace("@en", "")
                        clauses.append(f"?x{i} ns:{subp[1]} ?lit{i} . FILTER (isLiteral( ?lit{i} )) . FILTER (str( ?lit{i} ) = {subp[2]} )")
                    # 其他类型的 Literal, 传入 S-expression 时就已经处理好了
                    else:
                        clauses.append("?x" + i + " ns:" + subp[1] + " " + subp[2] + " .")
                else: 
                    raise Exception(f"subp: {subp}")
            else:
                raise Exception(f"subp[1]: {subp[1]}; sub_programs: {sub_programs}")
    
        elif subp[0] == 'AND': 
            '''
            subp[1]: 嵌套子成分
            subp[2]: 嵌套子成分
            '''
            var1 = int(subp[1][1:])
            rooti = get_root(int(i))
            root1 = get_root(var1)
            if rooti > root1:
                identical_variables_r[rooti] = root1
            else:
                identical_variables_r[root1] = rooti
                root1 = rooti
            var2 = int(subp[2][1:])
            root2 = get_root(var2)
            if root1 > root2:
                identical_variables_r[root1] = root2
            else:
                identical_variables_r[root2] = root1
        elif subp[0] in ['LE', 'LT', 'GE', 'GT', 'EQ']:  
            '''
            subp[1]:
                - 嵌套结构, #n
                - 关系 / 逆关系
            subp[2]:
                - 嵌套结构, #n
                - time / number
            '''
            if subp[0] == 'LE':
                op = "<="
            elif subp[0] == 'LT':
                op = "<"
            elif subp[0] == 'GE':
                op = ">="
            elif subp[0] == 'GT':
                op = ">"
            elif subp[0] == 'EQ':
                op = "="
            else:
                raise Exception(f"op: {op}; sub_programs: {sub_programs}")
            if subp[1].startswith('#'): # 嵌套
                var1 = int(subp[1][1:])
                rooti = get_root(int(i))
                root1 = get_root(var1)
                if rooti > root1:
                    identical_variables_r[rooti] = root1
                else:
                    identical_variables_r[root1] = rooti

                if subp[2].startswith('#'): # 嵌套
                    root2 = get_root(int(subp[2][1:]))
                    # 嵌套的变量应该是以 x 开头的，因为只是单纯去求解一个值
                    clauses.append(f"FILTER ( ?v{root1} {op} ?x{root2} ) .")
                else: # literal, 并且只能是 time / number
                    if FreebaseConstantForConstruction.get_constant_type(subp[2]) not in [FREEBASE_CONSTANT_TYPE.QUANTITY, FREEBASE_CONSTANT_TYPE.TIME]:
                        raise Exception(f"subp[2]: {subp[2]}, sub_programs: {sub_programs}")
                    clauses.append(f"FILTER ( ?v{root1} {op} {subp[2]} ) .")
            elif isinstance(subp[1], list): # R relation
                if subp[2].startswith('#'): # 嵌套
                    root2 = get_root(int(subp[2][1:]))
                    clauses.append(f"?v{i} ns:{subp[1][1]} ?x{i} . FILTER ( ?v{i} {op} ?x{root2} ) .")
                else: # literal, 并且只能是 time / number
                    if FreebaseConstantForConstruction.get_constant_type(subp[2]) not in [FREEBASE_CONSTANT_TYPE.QUANTITY, FREEBASE_CONSTANT_TYPE.TIME]:
                        raise Exception(f"subp[2]: {subp[2]}, sub_programs: {sub_programs}")
                    clauses.append(f"?v{i} ns:{subp[1][1]} ?x{i} . FILTER ( ?v{i} {op} {subp[2]} ) .")
            elif isinstance(subp[1], str): # relation
                if subp[2].startswith('#'): # 嵌套
                    root2 = get_root(int(subp[2][1:]))
                    clauses.append(f"?x{i} ns:{subp[1]} ?v{i}. FILTER ( ?v{i} {op} ?x{root2} ) .")
                else: # literal, 并且只能是 time / number
                    if FreebaseConstantForConstruction.get_constant_type(subp[2]) not in [FREEBASE_CONSTANT_TYPE.QUANTITY, FREEBASE_CONSTANT_TYPE.TIME]:
                        raise Exception(f"subp[2]: {subp[2]}, sub_programs: {sub_programs}")
                    clauses.append(f"?x{i} ns:{subp[1]} ?v{i} . FILTER ( ?v{i} {op} {subp[2]} ) .")
            else:
                raise Exception(f"subp: {subp}; sub_programs: {sub_programs}")

        elif subp[0] in ["ARGMIN", "ARGMAX"]:
            # TODO: 暂时只考虑 Sexp 中只出现一次 ARGMAX 的情况
            '''
            subp[1]: #n
            subp[2]:
                - relation
                - R relation
            '''
            if subp[1][0] == '#':
                var1 = int(subp[1][1:])
                rooti = get_root(int(i))
                root1 = get_root(var1)
                if rooti > root1:
                    identical_variables_r[rooti] = root1
                else:
                    identical_variables_r[root1] = rooti
                    root1 = rooti
                
                if subp[2][0] == '#': # 合并变量即可
                    var2 = int(subp[2][1:])
                    root2 = get_root(var2)
                    if root1 > root2:
                        identical_variables_r[root1] = root2
                    else:
                        identical_variables_r[root2] = root1
                        root2 = root1
                    
                    # 标记一下变量，后面加个 order by
                    if subp[0] == 'ARGMIN':
                        order_clauses = [f"?arg{root2}", "ASC", 1] # 目前默认都是 LIMIT 1
                    elif subp[0] == 'ARGMAX':
                        order_clauses = [f"?arg{root2}", "DESC", 1]   
    
                elif isinstance(subp[2], list): # R relation
                    clauses.append(f"?arg{root1} ns:{subp[2][1]} ?x{root1} .")
                    if subp[0] == 'ARGMIN':
                        order_clauses = [f"?arg{root1}", "ASC", 1] # 目前默认都是 LIMIT 1
                    elif subp[0] == 'ARGMAX':
                        order_clauses = [f"?arg{root1}", "DESC", 1]
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?x{root1} ns:{subp[2]} ?arg{root1} .")
                    if subp[0] == 'ARGMIN':
                        order_clauses = [f"?arg{root1}", "ASC", 1] # 目前默认都是 LIMIT 1
                    elif subp[0] == 'ARGMAX':
                        order_clauses = [f"?arg{root1}", "DESC", 1]
            else:  
                raise Exception(f"subp: {subp}; sub_programs: {sub_programs}")


        elif subp[0] == 'COUNT':  # this is easy, since it can only be applied to the quesiton node
            var = int(subp[1][1:])
            root_var = get_root(var)
            identical_variables_r[int(i)] = root_var  # COUNT can only be the outtermost
            count = True
        
        elif subp[0] in ['ARGMIN_JOIN', 'ARGMAX_JOIN']:
            '''
            subp[1]:
            - relation
            - R relation

            subp[2]
            - relation
            - R relation
            '''
            if isinstance(subp[1], list): # R relation
                if isinstance(subp[2], list): # R relation
                    clauses.append(f"?arg{i} ns:{subp[2][1]} ?y{i} . ?y{i} ns:{subp[1][1]} ?x{i} .")
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?y{i} ns:{subp[2]} ?arg{i} . ?y{i} ns:{subp[1][1]} ?x{i} .")
                else:
                    raise Exception(f"subp: {subp}")
            elif isinstance(subp[1], str): # relation
                if isinstance(subp[2], list): # R relation
                    clauses.append(f"?arg{i} ns:{subp[2][1]} ?y{i} . ?x{i} ns:{subp[1]} ?y{i} .")
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?y{i} ns:{subp[2]} ?arg{i} . ?x{i} ns:{subp[1]} ?y{i} .")
                else:
                    raise Exception(f"subp: {subp}")
            else:
                raise Exception(f"subp: {subp}")


        elif subp[0] in ['LT_JOIN', 'LE_JOIN', "GT_JOIN", "GE_JOIN", "EQ_JOIN"]:
            '''
            subp[1]:
            - relation
            - R relation

            subp[2]
            - relation
            - R relation
            '''
            if isinstance(subp[1], list): # R relation
                if isinstance(subp[2], list): # R relation
                    clauses.append(f"?v{i} ns:{subp[2][1]} ?y{i} . ?y{i} ns:{subp[1][1]} ?x{i} .")
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?y{i} ns:{subp[2]} ?v{i} . ?y{i} ns:{subp[1][1]} ?x{i} .")
                else:
                    raise Exception(f"subp: {subp}")
            elif isinstance(subp[1], str):
                if isinstance(subp[2], list): # R relation
                    clauses.append(f"?v{i} ns:{subp[2][1]} ?y{i} . ?x{i} ns:{subp[1]} ?y{i} .")
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?y{i} ns:{subp[2]} ?v{i} . ?x{i} ns:{subp[1]} ?y{i} .")
                else:
                    raise Exception(f"subp: {subp}")
            else:
                raise Exception(f"subp: {subp}")
        
        else:
            raise Exception(f"subp: {subp}")
    
    #  Merge identical variables
    for i in range(len(clauses)):
        for k in identical_variables_r:
            clauses[i] = clauses[i].replace(f'?x{k} ', f'?x{get_root(k)} ')
            clauses[i] = clauses[i].replace(f'?y{k} ', f'?y{get_root(k)} ')
            clauses[i] = clauses[i].replace(f'?v{k} ', f'?v{get_root(k)} ')
            clauses[i] = clauses[i].replace(f'?lit{k} ', f'?lit{get_root(k)} ')
            clauses[i] = clauses[i].replace(f'?arg{k} ', f'?arg{get_root(k)} ')
    
    if order_clauses is not None:
        for k in identical_variables_r:
            order_clauses[0] = order_clauses[0].replace(f'?arg{k}', f'?arg{get_root(k)}')

    question_var = get_root(question_var)

    for i in range(len(clauses)):
        clauses[i] = clauses[i].replace(f'?x{question_var} ', f'?x ')
    
    # if order_clauses is not None:
    #     arg_clauses = clauses[:]
    
    # TODO: 值得商榷，我感觉没什么必要
    # for entity in entities:
    #     clauses.append(f'FILTER (?x != ns:{entity})')
    clauses.insert(0,
                   f"FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))")
    clauses.insert(0, "WHERE {")
    if count:
        clauses.insert(0, f"SELECT COUNT DISTINCT ?x")
        
    # elif order_clauses is not None: # ARGMIN / ARGMAX, 如果存在多个取值相同且都为 top 的元素，SPARQL 需要做特殊处理
    #     clauses.insert(0, "{SELECT " + order_clauses[0]) # 能处理多个实体有相同值的情况
    #     clauses = arg_clauses + clauses
    #     clauses.insert(0, "WHERE {")
    #     clauses.insert(0, f"SELECT DISTINCT ?x")
    else:
        clauses.insert(0, f"SELECT DISTINCT ?x")
    clauses.insert(0, "PREFIX ns: <http://rdf.freebase.com/ns/>")

    clauses.append('}')
    if order_clauses is not None:
        clauses.append(f"ORDER BY {order_clauses[1]}({order_clauses[0]}) LIMIT {order_clauses[2]}")
        # clauses.append('}')
        # clauses.append('}')
        
    return '\n'.join(clauses)

def sexp_to_sparql_for_edit_distance(lisp_program: str):
    '''
    变量说明:
    - ?x{i}: entity 集合
    - ?y{i}: 中间变量
    - ?lit{i}: string 集合
    - ?v{i}: time / number 集合
    - ?arg{i}: ARGMIN / ARGMAX 的排序目标

    编辑距离计算版本的 sexp_to_sparql(): 有修改的地方，都标注出来了 (EDIT:)
    '''
    clauses = []
    # 仅支持出现一次 ARGMAX 的场景
    order_clauses = None # [变量名，方向, (LIMIT) n] --> [?arg0, ASC / DESC, 1]
    entities = set()  # collect entites for filtering
    classes = set()
    identical_variables_r = {}  # key should be larger than value
    expression = lisp_to_nested_expression(lisp_program)
    count = False

    sub_programs = _linearize_lisp_expression(expression, [0])
    question_var = len(sub_programs) - 1
    
    def get_root(var: int):
        while var in identical_variables_r:
            var = identical_variables_r[var]

        return var

    for i, subp in enumerate(sub_programs):
        '''
        标记说明:
        - x{i} 表示一般变量
        - y{i} 多跳关系中的中间变量
        - z{i} 多跳关系中的目标变量
        - v{i} 表示一个指示 TIME / QUANTITY 的变量
        - arg{i} 是 ORDER BY 操作符的对象
        - lit{i} 表示一个指示 STRING 的变量
        '''
        i = str(i)
        if subp[0] == 'JOIN':
            '''
            subp[1] 我认为只有两种选择:
            - relation
            - R relation
            这两者只有 SPARQL 里面三元组方向的区别

            无论 subp[1] 是什么, subp[2] 有如下选择
            - item: entity / class / literal
            - #n: 表示一个嵌套的子结构, 有可能代表关系的组合
            - relation --> 合起来表示一个多跳的关系
            - R relation
            '''
            if isinstance(subp[1], list):  # R relation
                if subp[2][:2] in ["m.", "g."]:  # entity
                    clauses.append("ns:" + subp[2] + " ns:" + subp[1][1] + " ?x" + i + " .")
                    entities.add(subp[2])
                elif re.fullmatch("[a-zA-Z_0-9]+\.[a-zA-Z_0-9]+", subp[2]): # class
                    clauses.append("ns:" + subp[2] + " ns:" + subp[1][1] + " ?x" + i + " .")
                    classes.add(subp[2])
                elif subp[2][0] == '#':  # 嵌套子结构
                    clauses.append(f"?x{get_root(int(subp[2][1:]))} ns:{subp[1][1]} ?x{i} .")
                elif FreebaseConstantForConstruction.get_constant_type(subp[2]) in [FREEBASE_CONSTANT_TYPE.TIME, FREEBASE_CONSTANT_TYPE.QUANTITY, FREEBASE_CONSTANT_TYPE.STRING]: # literal
                    if FreebaseConstantForConstruction.get_constant_type(subp[2])  is FREEBASE_CONSTANT_TYPE.STRING:
                        subp[2] = subp[2].replace("@en", "")
                        clauses.append(f"?lit{i} ns:{subp[1][1]} ?x{i} . FILTER (isLiteral( ?lit{i} )) . FILTER (str( ?lit{i} ) = {subp[2]} ) .")
                    # 其他类型的 Literal, 传入 S-expression 时就已经处理好了
                    else:
                        clauses.append(subp[2] + " ns:" + subp[1][1] + " ?x" + i + " .")
                else: 
                    raise Exception(f"subp: {subp}")

            elif isinstance(subp[1], str): # relation
                if subp[2][:2] in ["m.", "g."]:  # entity
                    clauses.append("?x" + i + " ns:" + subp[1] + " ns:" + subp[2] + " .")
                    entities.add(subp[2])
                elif re.fullmatch("[a-zA-Z_0-9]+\.[a-zA-Z_0-9]+", subp[2]): # class
                    clauses.append("?x" + i + " ns:" + subp[1] + " ns:" + subp[2] + " .")
                    classes.add(subp[2])
                elif subp[2][0] == '#':  # variable
                    clauses.append(f"?x{i} ns:{subp[1]} ?x{get_root(int(subp[2][1:]))} .")
                elif FreebaseConstantForConstruction.get_constant_type(subp[2]) in [FREEBASE_CONSTANT_TYPE.TIME, FREEBASE_CONSTANT_TYPE.QUANTITY, FREEBASE_CONSTANT_TYPE.STRING]:  # literal
                    if FreebaseConstantForConstruction.get_constant_type(subp[2]) is FREEBASE_CONSTANT_TYPE.STRING:
                        subp[2] = subp[2].replace("@en", "")
                        clauses.append(f"?x{i} ns:{subp[1]} ?lit{i} . FILTER (isLiteral( ?lit{i} )) . FILTER (str( ?lit{i} ) = {subp[2]} )")
                    # 其他类型的 Literal, 传入 S-expression 时就已经处理好了
                    else:
                        clauses.append("?x" + i + " ns:" + subp[1] + " " + subp[2] + " .")
                else: 
                    raise Exception(f"subp: {subp}")
            else:
                raise Exception(f"subp[1]: {subp[1]}; sub_programs: {sub_programs}")
    
        elif subp[0] == 'AND': 
            '''
            subp[1]: 嵌套子成分
            subp[2]: 嵌套子成分
            '''
            var1 = int(subp[1][1:])
            rooti = get_root(int(i))
            root1 = get_root(var1)
            if rooti > root1:
                identical_variables_r[rooti] = root1
            else:
                identical_variables_r[root1] = rooti
                root1 = rooti
            var2 = int(subp[2][1:])
            root2 = get_root(var2)
            if root1 > root2:
                identical_variables_r[root1] = root2
            else:
                identical_variables_r[root2] = root1
        elif subp[0] in ['LE', 'LT', 'GE', 'GT', 'EQ']:  
            '''
            subp[1]:
                - 嵌套结构, #n
                - 关系 / 逆关系
            subp[2]:
                - 嵌套结构, #n
                - time / number
            '''
            if subp[0] == 'LE':
                op = "<="
            elif subp[0] == 'LT':
                op = "<"
            elif subp[0] == 'GE':
                op = ">="
            elif subp[0] == 'GT':
                op = ">"
            elif subp[0] == 'EQ':
                op = "="
            else:
                raise Exception(f"op: {op}; sub_programs: {sub_programs}")
            if subp[1].startswith('#'): # 嵌套
                var1 = int(subp[1][1:])
                rooti = get_root(int(i))
                root1 = get_root(var1)
                if rooti > root1:
                    identical_variables_r[rooti] = root1
                else:
                    identical_variables_r[root1] = rooti

                if subp[2].startswith('#'): # 嵌套
                    root2 = get_root(int(subp[2][1:]))
                    # 嵌套的变量应该是以 x 开头的，因为只是单纯去求解一个值
                    clauses.append(f"FILTER ( ?v{root1} {op} ?x{root2} ) .")
                else: # literal, 并且只能是 time / number
                    if FreebaseConstantForConstruction.get_constant_type(subp[2]) not in [FREEBASE_CONSTANT_TYPE.QUANTITY, FREEBASE_CONSTANT_TYPE.TIME]:
                        raise Exception(f"subp[2]: {subp[2]}, sub_programs: {sub_programs}")
                    clauses.append(f"FILTER ( ?v{root1} {op} {subp[2]} ) .")
            elif isinstance(subp[1], list): # R relation
                if subp[2].startswith('#'): # 嵌套
                    root2 = get_root(int(subp[2][1:]))
                    clauses.append(f"?v{i} ns:{subp[1][1]} ?x{i} . FILTER ( ?v{i} {op} ?x{root2} ) .")
                else: # literal, 并且只能是 time / number
                    if FreebaseConstantForConstruction.get_constant_type(subp[2]) not in [FREEBASE_CONSTANT_TYPE.QUANTITY, FREEBASE_CONSTANT_TYPE.TIME]:
                        raise Exception(f"subp[2]: {subp[2]}, sub_programs: {sub_programs}")
                    clauses.append(f"?v{i} ns:{subp[1][1]} ?x{i} . FILTER ( ?v{i} {op} {subp[2]} ) .")
            elif isinstance(subp[1], str): # relation
                if subp[2].startswith('#'): # 嵌套
                    root2 = get_root(int(subp[2][1:]))
                    clauses.append(f"?x{i} ns:{subp[1]} ?v{i}. FILTER ( ?v{i} {op} ?x{root2} ) .")
                else: # literal, 并且只能是 time / number
                    if FreebaseConstantForConstruction.get_constant_type(subp[2]) not in [FREEBASE_CONSTANT_TYPE.QUANTITY, FREEBASE_CONSTANT_TYPE.TIME]:
                        raise Exception(f"subp[2]: {subp[2]}, sub_programs: {sub_programs}")
                    clauses.append(f"?x{i} ns:{subp[1]} ?v{i} . FILTER ( ?v{i} {op} {subp[2]} ) .")
            else:
                raise Exception(f"subp: {subp}; sub_programs: {sub_programs}")

        elif subp[0] in ["ARGMIN", "ARGMAX"]:
            # TODO: 暂时只考虑 Sexp 中只出现一次 ARGMAX 的情况
            '''
            subp[1]: #n
            subp[2]:
                - relation
                - R relation
            '''
            if subp[1][0] == '#':
                var1 = int(subp[1][1:])
                rooti = get_root(int(i))
                root1 = get_root(var1)
                if rooti > root1:
                    identical_variables_r[rooti] = root1
                else:
                    identical_variables_r[root1] = rooti
                    root1 = rooti
                
                if subp[2][0] == '#': # 合并变量即可
                    var2 = int(subp[2][1:])
                    root2 = get_root(var2)
                    if root1 > root2:
                        identical_variables_r[root1] = root2
                    else:
                        identical_variables_r[root2] = root1
                        root2 = root1
                    
                    # 标记一下变量，后面加个 order by
                    if subp[0] == 'ARGMIN':
                        order_clauses = [f"?arg{root2}", "ASC", 1] # 目前默认都是 LIMIT 1
                    elif subp[0] == 'ARGMAX':
                        order_clauses = [f"?arg{root2}", "DESC", 1]   
    
                elif isinstance(subp[2], list): # R relation
                    clauses.append(f"?arg{root1} ns:{subp[2][1]} ?x{root1} .")
                    if subp[0] == 'ARGMIN':
                        order_clauses = [f"?arg{root1}", "ASC", 1] # 目前默认都是 LIMIT 1
                    elif subp[0] == 'ARGMAX':
                        order_clauses = [f"?arg{root1}", "DESC", 1]
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?x{root1} ns:{subp[2]} ?arg{root1} .")
                    if subp[0] == 'ARGMIN':
                        order_clauses = [f"?arg{root1}", "ASC", 1] # 目前默认都是 LIMIT 1
                    elif subp[0] == 'ARGMAX':
                        order_clauses = [f"?arg{root1}", "DESC", 1]
            else:  
                raise Exception(f"subp: {subp}; sub_programs: {sub_programs}")


        elif subp[0] == 'COUNT':  # this is easy, since it can only be applied to the quesiton node
            var = int(subp[1][1:])
            root_var = get_root(var)
            identical_variables_r[int(i)] = root_var  # COUNT can only be the outtermost
            count = True
        
        elif subp[0] in ['ARGMIN_JOIN', 'ARGMAX_JOIN']:
            '''
            subp[1]:
            - relation
            - R relation

            subp[2]
            - relation
            - R relation
            '''
            if isinstance(subp[1], list): # R relation
                if isinstance(subp[2], list): # R relation
                    clauses.append(f"?arg{i} ns:{subp[2][1]} ?y{i} . ?y{i} ns:{subp[1][1]} ?x{i} .")
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?y{i} ns:{subp[2]} ?arg{i} . ?y{i} ns:{subp[1][1]} ?x{i} .")
                else:
                    raise Exception(f"subp: {subp}")
            elif isinstance(subp[1], str): # relation
                if isinstance(subp[2], list): # R relation
                    clauses.append(f"?arg{i} ns:{subp[2][1]} ?y{i} . ?x{i} ns:{subp[1]} ?y{i} .")
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?y{i} ns:{subp[2]} ?arg{i} . ?x{i} ns:{subp[1]} ?y{i} .")
                else:
                    raise Exception(f"subp: {subp}")
            else:
                raise Exception(f"subp: {subp}")


        elif subp[0] in ['LT_JOIN', 'LE_JOIN', "GT_JOIN", "GE_JOIN", "EQ_JOIN"]:
            '''
            subp[1]:
            - relation
            - R relation

            subp[2]
            - relation
            - R relation
            '''
            if isinstance(subp[1], list): # R relation
                if isinstance(subp[2], list): # R relation
                    clauses.append(f"?v{i} ns:{subp[2][1]} ?y{i} . ?y{i} ns:{subp[1][1]} ?x{i} .")
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?y{i} ns:{subp[2]} ?v{i} . ?y{i} ns:{subp[1][1]} ?x{i} .")
                else:
                    raise Exception(f"subp: {subp}")
            elif isinstance(subp[1], str):
                if isinstance(subp[2], list): # R relation
                    clauses.append(f"?v{i} ns:{subp[2][1]} ?y{i} . ?x{i} ns:{subp[1]} ?y{i} .")
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?y{i} ns:{subp[2]} ?v{i} . ?x{i} ns:{subp[1]} ?y{i} .")
                else:
                    raise Exception(f"subp: {subp}")
            else:
                raise Exception(f"subp: {subp}")
        
        else:
            raise Exception(f"subp: {subp}")
    
    #  Merge identical variables
    for i in range(len(clauses)):
        for k in identical_variables_r:
            clauses[i] = clauses[i].replace(f'?x{k} ', f'?x{get_root(k)} ')
            clauses[i] = clauses[i].replace(f'?y{k} ', f'?y{get_root(k)} ')
            clauses[i] = clauses[i].replace(f'?v{k} ', f'?v{get_root(k)} ')
            clauses[i] = clauses[i].replace(f'?lit{k} ', f'?lit{get_root(k)} ')
            clauses[i] = clauses[i].replace(f'?arg{k} ', f'?arg{get_root(k)} ')
    
    if order_clauses is not None:
        for k in identical_variables_r:
            order_clauses[0] = order_clauses[0].replace(f'?arg{k}', f'?arg{get_root(k)}')

    question_var = get_root(question_var)

    for i in range(len(clauses)):
        clauses[i] = clauses[i].replace(f'?x{question_var} ', f'?x ')
    
    # if order_clauses is not None:
    #     arg_clauses = clauses[:]
    
    # TODO: 值得商榷，我感觉没什么必要
    # for entity in entities:
    #     clauses.append(f'FILTER (?x != ns:{entity})')
    clauses.insert(0,
                   f"FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))")
    clauses.insert(0, "WHERE {")
    if count:
        # clauses.insert(0, f"SELECT COUNT DISTINCT ?x")
        # EDIT: 0526 修改，仅仅是因为 SPARQL parser 的格式要求，对于 SPARQL 的执行结果等应该没有影响
        clauses.insert(0, f"SELECT (COUNT (DISTINCT ?x) as ?cnt)")
    # elif order_clauses is not None: # ARGMIN / ARGMAX, 如果存在多个取值相同且都为 top 的元素，SPARQL 需要做特殊处理
    #     clauses.insert(0, "{SELECT " + order_clauses[0]) # 能处理多个实体有相同值的情况
    #     clauses = arg_clauses + clauses
    #     clauses.insert(0, "WHERE {")
    #     clauses.insert(0, f"SELECT DISTINCT ?x")
    else:
        clauses.insert(0, f"SELECT DISTINCT ?x")
    clauses.insert(0, "PREFIX ns: <http://rdf.freebase.com/ns/>")

    clauses.append('}')
    # EDIT:
    if order_clauses is not None:
        if order_clauses[1] == "DESC":
            clauses.append(f"ORDER BY {order_clauses[1]}({order_clauses[0]}) LIMIT {order_clauses[2]}")
        elif order_clauses[1] == "ASC":
            clauses.append(f"ORDER BY {order_clauses[0]} LIMIT {order_clauses[2]}")
        else:
            raise Exception(f"order by operator: {order_clauses[1]}")
        # clauses.append('}')
        # clauses.append('}')
        
    return '\n'.join(clauses)

def sexp_to_sparql_for_test_suite(lisp_program: str):
    '''
    计算 Test Suite Accuracy 时，同样需要使用 parser 对 SPARQL 做解析，因此 SPARQL 的格式上有一些修改
    修改的地方同样用 EDIT: 标识
    变量说明:
    - ?x{i}: entity 集合
    - ?y{i}: 中间变量
    - ?lit{i}: string 集合
    - ?v{i}: time / number 集合
    - ?arg{i}: ARGMIN / ARGMAX 的排序目标
    '''
    clauses = []
    # 仅支持出现一次 ARGMAX 的场景
    order_clauses = None # [变量名，方向, (LIMIT) n] --> [?arg0, ASC / DESC, 1]
    entities = set()  # collect entites for filtering
    classes = set()
    identical_variables_r = {}  # key should be larger than value
    expression = lisp_to_nested_expression(lisp_program)
    count = False

    sub_programs = _linearize_lisp_expression(expression, [0])
    question_var = len(sub_programs) - 1
    
    def get_root(var: int):
        while var in identical_variables_r:
            var = identical_variables_r[var]

        return var

    for i, subp in enumerate(sub_programs):
        '''
        标记说明:
        - x{i} 表示一般变量
        - y{i} 多跳关系中的中间变量
        - z{i} 多跳关系中的目标变量
        - v{i} 表示一个指示 TIME / QUANTITY 的变量
        - arg{i} 是 ORDER BY 操作符的对象
        - lit{i} 表示一个指示 STRING 的变量
        '''
        i = str(i)
        if subp[0] == 'JOIN':
            '''
            subp[1] 我认为只有两种选择:
            - relation
            - R relation
            这两者只有 SPARQL 里面三元组方向的区别

            无论 subp[1] 是什么, subp[2] 有如下选择
            - item: entity / class / literal
            - #n: 表示一个嵌套的子结构, 有可能代表关系的组合
            - relation --> 合起来表示一个多跳的关系
            - R relation
            '''
            if isinstance(subp[1], list):  # R relation
                if subp[2][:2] in ["m.", "g."]:  # entity
                    clauses.append("ns:" + subp[2] + " ns:" + subp[1][1] + " ?x" + i + " .")
                    entities.add(subp[2])
                elif re.fullmatch("[a-zA-Z_0-9]+\.[a-zA-Z_0-9]+", subp[2]): # class
                    clauses.append("ns:" + subp[2] + " ns:" + subp[1][1] + " ?x" + i + " .")
                    classes.add(subp[2])
                elif subp[2][0] == '#':  # 嵌套子结构
                    clauses.append(f"?x{get_root(int(subp[2][1:]))} ns:{subp[1][1]} ?x{i} .")
                elif FreebaseConstantForConstruction.get_constant_type(subp[2]) in [FREEBASE_CONSTANT_TYPE.TIME, FREEBASE_CONSTANT_TYPE.QUANTITY, FREEBASE_CONSTANT_TYPE.STRING]: # literal
                    if FreebaseConstantForConstruction.get_constant_type(subp[2])  is FREEBASE_CONSTANT_TYPE.STRING:
                        subp[2] = subp[2].replace("@en", "")
                        clauses.append(f"?lit{i} ns:{subp[1][1]} ?x{i} . FILTER (isLiteral( ?lit{i} )) . FILTER (str( ?lit{i} ) = {subp[2]} ) .")
                    # 其他类型的 Literal, 传入 S-expression 时就已经处理好了
                    else:
                        clauses.append(subp[2] + " ns:" + subp[1][1] + " ?x" + i + " .")
                else: 
                    raise Exception(f"subp: {subp}")

            elif isinstance(subp[1], str): # relation
                if subp[2][:2] in ["m.", "g."]:  # entity
                    clauses.append("?x" + i + " ns:" + subp[1] + " ns:" + subp[2] + " .")
                    entities.add(subp[2])
                elif re.fullmatch("[a-zA-Z_0-9]+\.[a-zA-Z_0-9]+", subp[2]): # class
                    clauses.append("?x" + i + " ns:" + subp[1] + " ns:" + subp[2] + " .")
                    classes.add(subp[2])
                elif subp[2][0] == '#':  # variable
                    clauses.append(f"?x{i} ns:{subp[1]} ?x{get_root(int(subp[2][1:]))} .")
                elif FreebaseConstantForConstruction.get_constant_type(subp[2]) in [FREEBASE_CONSTANT_TYPE.TIME, FREEBASE_CONSTANT_TYPE.QUANTITY, FREEBASE_CONSTANT_TYPE.STRING]:  # literal
                    if FreebaseConstantForConstruction.get_constant_type(subp[2]) is FREEBASE_CONSTANT_TYPE.STRING:
                        subp[2] = subp[2].replace("@en", "")
                        clauses.append(f"?x{i} ns:{subp[1]} ?lit{i} . FILTER (isLiteral( ?lit{i} )) . FILTER (str( ?lit{i} ) = {subp[2]} )")
                    # 其他类型的 Literal, 传入 S-expression 时就已经处理好了
                    else:
                        clauses.append("?x" + i + " ns:" + subp[1] + " " + subp[2] + " .")
                else: 
                    raise Exception(f"subp: {subp}")
            else:
                raise Exception(f"subp[1]: {subp[1]}; sub_programs: {sub_programs}")
    
        elif subp[0] == 'AND': 
            '''
            subp[1]: 嵌套子成分
            subp[2]: 嵌套子成分
            '''
            var1 = int(subp[1][1:])
            rooti = get_root(int(i))
            root1 = get_root(var1)
            if rooti > root1:
                identical_variables_r[rooti] = root1
            else:
                identical_variables_r[root1] = rooti
                root1 = rooti
            var2 = int(subp[2][1:])
            root2 = get_root(var2)
            if root1 > root2:
                identical_variables_r[root1] = root2
            else:
                identical_variables_r[root2] = root1
        elif subp[0] in ['LE', 'LT', 'GE', 'GT', 'EQ']:  
            '''
            subp[1]:
                - 嵌套结构, #n
                - 关系 / 逆关系
            subp[2]:
                - 嵌套结构, #n
                - time / number
            '''
            if subp[0] == 'LE':
                op = "<="
            elif subp[0] == 'LT':
                op = "<"
            elif subp[0] == 'GE':
                op = ">="
            elif subp[0] == 'GT':
                op = ">"
            elif subp[0] == 'EQ':
                op = "="
            else:
                raise Exception(f"op: {op}; sub_programs: {sub_programs}")
            if subp[1].startswith('#'): # 嵌套
                var1 = int(subp[1][1:])
                rooti = get_root(int(i))
                root1 = get_root(var1)
                if rooti > root1:
                    identical_variables_r[rooti] = root1
                else:
                    identical_variables_r[root1] = rooti

                if subp[2].startswith('#'): # 嵌套
                    root2 = get_root(int(subp[2][1:]))
                    # 嵌套的变量应该是以 x 开头的，因为只是单纯去求解一个值
                    clauses.append(f"FILTER ( ?v{root1} {op} ?x{root2} ) .")
                else: # literal, 并且只能是 time / number
                    if FreebaseConstantForConstruction.get_constant_type(subp[2]) not in [FREEBASE_CONSTANT_TYPE.QUANTITY, FREEBASE_CONSTANT_TYPE.TIME]:
                        raise Exception(f"subp[2]: {subp[2]}, sub_programs: {sub_programs}")
                    clauses.append(f"FILTER ( ?v{root1} {op} {subp[2]} ) .")
            elif isinstance(subp[1], list): # R relation
                if subp[2].startswith('#'): # 嵌套
                    root2 = get_root(int(subp[2][1:]))
                    clauses.append(f"?v{i} ns:{subp[1][1]} ?x{i} . FILTER ( ?v{i} {op} ?x{root2} ) .")
                else: # literal, 并且只能是 time / number
                    if FreebaseConstantForConstruction.get_constant_type(subp[2]) not in [FREEBASE_CONSTANT_TYPE.QUANTITY, FREEBASE_CONSTANT_TYPE.TIME]:
                        raise Exception(f"subp[2]: {subp[2]}, sub_programs: {sub_programs}")
                    clauses.append(f"?v{i} ns:{subp[1][1]} ?x{i} . FILTER ( ?v{i} {op} {subp[2]} ) .")
            elif isinstance(subp[1], str): # relation
                if subp[2].startswith('#'): # 嵌套
                    root2 = get_root(int(subp[2][1:]))
                    clauses.append(f"?x{i} ns:{subp[1]} ?v{i}. FILTER ( ?v{i} {op} ?x{root2} ) .")
                else: # literal, 并且只能是 time / number
                    if FreebaseConstantForConstruction.get_constant_type(subp[2]) not in [FREEBASE_CONSTANT_TYPE.QUANTITY, FREEBASE_CONSTANT_TYPE.TIME]:
                        raise Exception(f"subp[2]: {subp[2]}, sub_programs: {sub_programs}")
                    clauses.append(f"?x{i} ns:{subp[1]} ?v{i} . FILTER ( ?v{i} {op} {subp[2]} ) .")
            else:
                raise Exception(f"subp: {subp}; sub_programs: {sub_programs}")

        elif subp[0] in ["ARGMIN", "ARGMAX"]:
            # TODO: 暂时只考虑 Sexp 中只出现一次 ARGMAX 的情况
            '''
            subp[1]: #n
            subp[2]:
                - relation
                - R relation
            '''
            if subp[1][0] == '#':
                var1 = int(subp[1][1:])
                rooti = get_root(int(i))
                root1 = get_root(var1)
                if rooti > root1:
                    identical_variables_r[rooti] = root1
                else:
                    identical_variables_r[root1] = rooti
                    root1 = rooti
                
                if subp[2][0] == '#': # 合并变量即可
                    var2 = int(subp[2][1:])
                    root2 = get_root(var2)
                    if root1 > root2:
                        identical_variables_r[root1] = root2
                    else:
                        identical_variables_r[root2] = root1
                        root2 = root1
                    
                    # 标记一下变量，后面加个 order by
                    if subp[0] == 'ARGMIN':
                        order_clauses = [f"?arg{root2}", "ASC", 1] # 目前默认都是 LIMIT 1
                    elif subp[0] == 'ARGMAX':
                        order_clauses = [f"?arg{root2}", "DESC", 1]   
    
                elif isinstance(subp[2], list): # R relation
                    clauses.append(f"?arg{root1} ns:{subp[2][1]} ?x{root1} .")
                    if subp[0] == 'ARGMIN':
                        order_clauses = [f"?arg{root1}", "ASC", 1] # 目前默认都是 LIMIT 1
                    elif subp[0] == 'ARGMAX':
                        order_clauses = [f"?arg{root1}", "DESC", 1]
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?x{root1} ns:{subp[2]} ?arg{root1} .")
                    if subp[0] == 'ARGMIN':
                        order_clauses = [f"?arg{root1}", "ASC", 1] # 目前默认都是 LIMIT 1
                    elif subp[0] == 'ARGMAX':
                        order_clauses = [f"?arg{root1}", "DESC", 1]
            else:  
                raise Exception(f"subp: {subp}; sub_programs: {sub_programs}")


        elif subp[0] == 'COUNT':  # this is easy, since it can only be applied to the quesiton node
            var = int(subp[1][1:])
            root_var = get_root(var)
            identical_variables_r[int(i)] = root_var  # COUNT can only be the outtermost
            count = True
        
        elif subp[0] in ['ARGMIN_JOIN', 'ARGMAX_JOIN']:
            '''
            subp[1]:
            - relation
            - R relation

            subp[2]
            - relation
            - R relation
            '''
            if isinstance(subp[1], list): # R relation
                if isinstance(subp[2], list): # R relation
                    clauses.append(f"?arg{i} ns:{subp[2][1]} ?y{i} . ?y{i} ns:{subp[1][1]} ?x{i} .")
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?y{i} ns:{subp[2]} ?arg{i} . ?y{i} ns:{subp[1][1]} ?x{i} .")
                else:
                    raise Exception(f"subp: {subp}")
            elif isinstance(subp[1], str): # relation
                if isinstance(subp[2], list): # R relation
                    clauses.append(f"?arg{i} ns:{subp[2][1]} ?y{i} . ?x{i} ns:{subp[1]} ?y{i} .")
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?y{i} ns:{subp[2]} ?arg{i} . ?x{i} ns:{subp[1]} ?y{i} .")
                else:
                    raise Exception(f"subp: {subp}")
            else:
                raise Exception(f"subp: {subp}")


        elif subp[0] in ['LT_JOIN', 'LE_JOIN', "GT_JOIN", "GE_JOIN", "EQ_JOIN"]:
            '''
            subp[1]:
            - relation
            - R relation

            subp[2]
            - relation
            - R relation
            '''
            if isinstance(subp[1], list): # R relation
                if isinstance(subp[2], list): # R relation
                    clauses.append(f"?v{i} ns:{subp[2][1]} ?y{i} . ?y{i} ns:{subp[1][1]} ?x{i} .")
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?y{i} ns:{subp[2]} ?v{i} . ?y{i} ns:{subp[1][1]} ?x{i} .")
                else:
                    raise Exception(f"subp: {subp}")
            elif isinstance(subp[1], str):
                if isinstance(subp[2], list): # R relation
                    clauses.append(f"?v{i} ns:{subp[2][1]} ?y{i} . ?x{i} ns:{subp[1]} ?y{i} .")
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?y{i} ns:{subp[2]} ?v{i} . ?x{i} ns:{subp[1]} ?y{i} .")
                else:
                    raise Exception(f"subp: {subp}")
            else:
                raise Exception(f"subp: {subp}")
        
        else:
            raise Exception(f"subp: {subp}")
    
    #  Merge identical variables
    for i in range(len(clauses)):
        for k in identical_variables_r:
            clauses[i] = clauses[i].replace(f'?x{k} ', f'?x{get_root(k)} ')
            clauses[i] = clauses[i].replace(f'?y{k} ', f'?y{get_root(k)} ')
            clauses[i] = clauses[i].replace(f'?v{k} ', f'?v{get_root(k)} ')
            clauses[i] = clauses[i].replace(f'?lit{k} ', f'?lit{get_root(k)} ')
            clauses[i] = clauses[i].replace(f'?arg{k} ', f'?arg{get_root(k)} ')
    
    if order_clauses is not None:
        for k in identical_variables_r:
            order_clauses[0] = order_clauses[0].replace(f'?arg{k}', f'?arg{get_root(k)}')

    question_var = get_root(question_var)

    for i in range(len(clauses)):
        clauses[i] = clauses[i].replace(f'?x{question_var} ', f'?x ')
    
    # if order_clauses is not None:
    #     arg_clauses = clauses[:]
    
    # TODO: 值得商榷，我感觉没什么必要
    # for entity in entities:
    #     clauses.append(f'FILTER (?x != ns:{entity})')
    clauses.insert(0,
                   f"FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))")
    clauses.insert(0, "WHERE {")
    if count:
        # clauses.insert(0, f"SELECT COUNT DISTINCT ?x")
        # EDIT: 0526 修改，仅仅是因为 SPARQL parser 的格式要求，对于 SPARQL 的执行结果等应该没有影响
        clauses.insert(0, f"SELECT (COUNT (DISTINCT ?x) as ?cnt)")
    # elif order_clauses is not None: # ARGMIN / ARGMAX, 如果存在多个取值相同且都为 top 的元素，SPARQL 需要做特殊处理
    #     clauses.insert(0, "{SELECT " + order_clauses[0]) # 能处理多个实体有相同值的情况
    #     clauses = arg_clauses + clauses
    #     clauses.insert(0, "WHERE {")
    #     clauses.insert(0, f"SELECT DISTINCT ?x")
    else:
        clauses.insert(0, f"SELECT DISTINCT ?x")
    clauses.insert(0, "PREFIX ns: <http://rdf.freebase.com/ns/>")

    clauses.append('}')
    if order_clauses is not None:
        clauses.append(f"ORDER BY {order_clauses[1]}({order_clauses[0]}) LIMIT {order_clauses[2]}")
        # clauses.append('}')
        # clauses.append('}')
        
    return '\n'.join(clauses)


def _linearize_lisp_expression(expression: list, sub_formula_id):
    sub_formulas = []
    for i, e in enumerate(expression):
        parent_flag = None
        if (expression[0] in ['LT', 'LE', 'GT', 'GE', 'EQ']) and i == 1:
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


def lisp_to_nested_expression(lisp_string: str) -> List:
    """
    Takes a logical form as a lisp string and returns a nested list representation of the lisp.
    For example, "(count (division first))" would get mapped to ['count', ['division', 'first']].
    """
    stack: List = []
    current_expression: List = []
    tokens = lisp_string.split()
    for token in tokens:
        while token[0] == '(':
            nested_expression: List = []
            current_expression.append(nested_expression)
            stack.append(current_expression)
            current_expression = nested_expression
            token = token[1:]
        current_expression.append(token.replace(')', ''))
        while token[-1] == ')':
            current_expression = stack.pop()
            token = token[:-1]
    return current_expression[0]


def reverse_process_literal(literal_item):
    """
    做的事情和 process_grounded_item_literal_grailqa 相反
    一些 KBQA 方法，只接受这种格式的 Literal 作为输入
    """
    split_res = literal_item.split("^^")
    value = split_res[0]
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1] # 去掉头尾的引号
    if value.endswith("-08:00"):
        value = value[:-6]
    
    suffix = split_res[1] if len(split_res) >= 2 else ""
    if suffix.startswith('<') and suffix.endswith('>'):
        suffix = suffix[1:-1]
    
    if len(suffix):
        return "^^".join([value, suffix])
    return value

def extract_value_from_literal(literal_item):
    """
    "\"89\"^^<http://www.w3.org/2001/XMLSchema#integer> --> 89
    \"2005-02-23-08:00\"^^<http://www.w3.org/2001/XMLSchema#date> --> 2005-02-23
    \"ar\"@en --> ar
    """
    if "http://www.w3.org/2001/XMLSchema" in literal_item:
        split_res = literal_item.split("^^")
        value = split_res[0]
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1] # 去掉头尾的引号
        if value.endswith("-08:00"):
            value = value[:-6]
    elif literal_item.endswith("@en"):
        value = literal_item[:-3]
        value = value[1:-1] # 去掉头尾的引号
    elif literal_item.startswith('"') and literal_item.endswith('"'):
        value = literal_item[1:-1]
    else:
        value = literal_item
    return value    

def post_process_sexp_grailqa(sexp):
    sexp = sexp.replace('(', ' ( ').replace(')', ' ) ')
    tokens = sexp.split()
    comparator_mappings = {
        "LE": 'le',
        "GE": 'ge',
        "LT": 'lt',
        "GT": 'gt',
        "EQ": 'JOIN'
    }
    '''GrailQA 没有 "abc"@en 形式的 Literal'''
    tokens = [x.strip() for x in tokens if len(x)]
    new_tokens = list()
    idx = 0
    while (idx < len(tokens)):
        tok = tokens[idx]
        # Literal 的处理
        if ('http://www.w3.org/2001/XMLSchema') in tok or (tok.startswith('"') and tok.endswith('"')):
           tok = reverse_process_literal(tok)
           idx += 1
           new_tokens.append(tok)
        elif tok in comparator_mappings:
            tok = comparator_mappings[tok]
            new_tokens.append(tok)
            idx += 1
        # # JOIN type.object.type class --> class; 容易导致 Bug, 先不管了
        # elif (idx + 1 < len(tokens)) and (tokens[idx + 1] == 'type.object.type') and (tok == 'JOIN'):
        #     idx += 2
        else:
            new_tokens.append(tok)
            idx += 1
    
    processed_sexp = " ".join(new_tokens).replace("( ", "(").replace(" )", ")")
    return processed_sexp

    # # 去掉 class 周围的括号; 很大概率导致 Bug
    # final_tokens = []
    # for tok in processed_sexp.split():
    #     if re.match(r"\(+[a-zA-Z_]+\.[a-zA-Z_]+\)+", tok):
    #         tok = tok.replace('(', '', 1) # replace first occurence
    #         assert tok[-1] == ')'
    #         tok = tok[:-1]
    #         final_tokens.append(tok) # 去掉一对括号
    #     else:
    #         final_tokens.append(tok)
    # return " ".join(final_tokens)


def post_process_sexp(sexp):
    sexp = sexp.replace('(', ' ( ').replace(')', ' ) ')
    tokens = sexp.split()
    comparator_mappings = {
        "LE": 'le',
        "GE": 'ge',
        "LT": 'lt',
        "GT": 'gt',
        "EQ": 'JOIN'
    }
    '''TODO: CWQ 有 "abc"@en 形式的 Literal'''
    tokens = [x.strip() for x in tokens if len(x)]
    new_tokens = list()
    idx = 0
    while (idx < len(tokens)):
        tok = tokens[idx]
        # Literal 的处理
        if ('http://www.w3.org/2001/XMLSchema') in tok or (tok.startswith('"') and tok.endswith('"')):
           tok = reverse_process_literal(tok)
           idx += 1
           new_tokens.append(tok)
        elif tok in comparator_mappings:
            tok = comparator_mappings[tok]
            new_tokens.append(tok)
            idx += 1
        else:
            new_tokens.append(tok)
            idx += 1
    
    processed_sexp = " ".join(new_tokens).replace("( ", "(").replace(" )", ")")
    return processed_sexp

def sexp_to_sparql_wikidata(lisp_program: str):
    '''
    变量说明:
    - ?x{i}: entity 集合
    - ?y{i}: 中间变量
    - ?lit{i}: string 集合
    - ?v{i}: time / number 集合
    - ?arg{i}: ARGMIN / ARGMAX 的排序目标
    '''
    clauses = []
    # 仅支持出现一次 ARGMAX 的场景
    order_clauses = None # [变量名, 方向, (LIMIT) n] --> [?arg0, ASC / DESC, 1]
    entities = set()  # collect entites for filtering
    classes = set()
    identical_variables_r = {}  # key should be larger than value
    expression = lisp_to_nested_expression(lisp_program)
    count = False

    sub_programs = _linearize_lisp_expression(expression, [0])
    question_var = len(sub_programs) - 1
    
    def get_root(var: int):
        while var in identical_variables_r:
            var = identical_variables_r[var]

        return var

    for i, subp in enumerate(sub_programs):
        i = str(i)
        if subp[0] == 'JOIN':
            '''
            subp[1] 我认为只有两种选择:
            - relation
            - R relation
            这两者只有 SPARQL 里面三元组方向的区别

            无论 subp[1] 是什么, subp[2] 有如下选择
            - item: entity / class / literal
            - #n: 表示一个嵌套的子结构
            - relation --> 合起来表示一个多跳的关系
            - R relation
            '''
            if isinstance(subp[1], list):  # R relation
                if WikidataConstantForConstruction.get_constant_type(subp[2]) is WIKIDATA_CONSTANT_TYPE.ENTITY: # entity
                    clauses.append(subp[2] + " " + subp[1][1] + " ?x" + i + " .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':  # 嵌套子结构
                    clauses.append(f"?x{get_root(int(subp[2][1:]))} {subp[1][1]} ?x{i} .")
                elif WikidataConstantForConstruction.get_constant_type(subp[2]) in [WIKIDATA_CONSTANT_TYPE.TIME, WIKIDATA_CONSTANT_TYPE.QUANTITY, WIKIDATA_CONSTANT_TYPE.STRING]: # literal
                    if WikidataConstantForConstruction.get_constant_type(subp[2]) == WIKIDATA_CONSTANT_TYPE.STRING:
                        subp[2] = subp[2].replace("@en", "")
                        clauses.append(f"?lit{i} {subp[1][1]} ?x{i} . FILTER (isLiteral( ?lit{i} )) . FILTER (str( ?lit{i} ) = {subp[2]} ) .")
                    # 其他类型的 Literal, 传入 S-expression 时就已经处理好了
                    else:
                        clauses.append(subp[2] + " " + subp[1][1] + " ?x" + i + " .")
                else: 
                    raise Exception(f"subp: {subp}")

            elif isinstance(subp[1], str): # relation
                if WikidataConstantForConstruction.get_constant_type(subp[2]) is WIKIDATA_CONSTANT_TYPE.ENTITY:  # entity
                    clauses.append("?x" + i + " " + subp[1] + " " + subp[2] + " .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':  # variable
                    clauses.append(f"?x{i} {subp[1]} ?x{get_root(int(subp[2][1:]))} .")
                elif WikidataConstantForConstruction.get_constant_type(subp[2]) in [WIKIDATA_CONSTANT_TYPE.TIME, WIKIDATA_CONSTANT_TYPE.QUANTITY, WIKIDATA_CONSTANT_TYPE.STRING]:  # literal
                    if WikidataConstantForConstruction.get_constant_type(subp[2]) is WIKIDATA_CONSTANT_TYPE.STRING:
                        subp[2] = subp[2].replace("@en", "")
                        clauses.append(f"?x{i} {subp[1]} ?lit{i} . FILTER (isLiteral( ?lit{i} )) . FILTER (str( ?lit{i} ) = {subp[2]} )")
                    # 其他类型的 Literal, 传入 S-expression 时就已经处理好了
                    else:
                        clauses.append("?x" + i + " " + subp[1] + " " + subp[2] + " .")
                else: 
                    raise Exception(f"subp: {subp}")
            else:
                raise Exception(f"subp[1]: {subp[1]}; sub_programs: {sub_programs}")
    
        elif subp[0] == 'AND': 
            var1 = int(subp[1][1:])
            rooti = get_root(int(i))
            root1 = get_root(var1)
            if rooti > root1:
                identical_variables_r[rooti] = root1
            else:
                identical_variables_r[root1] = rooti
                root1 = rooti
            var2 = int(subp[2][1:])
            root2 = get_root(var2)
            if root1 > root2:
                identical_variables_r[root1] = root2
            else:
                identical_variables_r[root2] = root1
        elif subp[0] in ['LE', 'LT', 'GE', 'GT', 'EQ']:  
            '''
            subp[1]:
                - 嵌套结构, #n
                - 关系 / 逆关系
            subp[2]:
                - 嵌套结构, #n
                - time / number
            '''
            if subp[0] == 'LE':
                op = "<="
            elif subp[0] == 'LT':
                op = "<"
            elif subp[0] == 'GE':
                op = ">="
            elif subp[0] == 'GT':
                op = ">"
            elif subp[0] == 'EQ':
                op = "="
            else:
                raise Exception(f"op: {op}; sub_programs: {sub_programs}")
            if subp[1].startswith('#'): # 嵌套
                var1 = int(subp[1][1:])
                rooti = get_root(int(i))
                root1 = get_root(var1)
                if rooti > root1:
                    identical_variables_r[rooti] = root1
                else:
                    identical_variables_r[root1] = rooti

                if subp[2].startswith('#'): # 嵌套
                    root2 = get_root(int(subp[2][1:]))
                    # 嵌套的变量应该是以 x 开头的，因为只是单纯去求解一个值
                    clauses.append(f"FILTER ( ?v{root1} {op} ?x{root2} ) .")
                else: # literal, 并且只能是 time / number
                    if WikidataConstantForConstruction.get_constant_type(subp[2]) not in [WIKIDATA_CONSTANT_TYPE.QUANTITY, WIKIDATA_CONSTANT_TYPE.TIME]:
                        raise Exception(f"subp[2]: {subp[2]}, sub_programs: {sub_programs}")
                    clauses.append(f"FILTER ( ?v{root1} {op} {subp[2]} ) .")
            elif isinstance(subp[1], list): # R relation
                if subp[2].startswith('#'): # 嵌套
                    root2 = get_root(int(subp[2][1:]))
                    clauses.append(f"?v{i} {subp[1][1]} ?x{i} . FILTER ( ?v{i} {op} ?v{root2} ) .")
                else: # literal, 并且只能是 time / number
                    if WikidataConstantForConstruction.get_constant_type(subp[2]) not in [WIKIDATA_CONSTANT_TYPE.QUANTITY, WIKIDATA_CONSTANT_TYPE.TIME]:
                        raise Exception(f"subp[2]: {subp[2]}, sub_programs: {sub_programs}")
                    clauses.append(f"?v{i} {subp[1][1]} ?x{i} . FILTER ( ?v{i} {op} {subp[2]} ) .")
            elif isinstance(subp[1], str): # relation
                if subp[2].startswith('#'): # 嵌套
                    root2 = get_root(int(subp[2][1:]))
                    clauses.append(f"?x{i} {subp[1]} ?v{i}. FILTER ( ?v{i} {op} ?x{root2} ) .")
                else: # literal, 并且只能是 time / number
                    if WikidataConstantForConstruction.get_constant_type(subp[2]) not in [WIKIDATA_CONSTANT_TYPE.QUANTITY, WIKIDATA_CONSTANT_TYPE.TIME]:
                        raise Exception(f"subp[2]: {subp[2]}, sub_programs: {sub_programs}")
                    clauses.append(f"?x{i} {subp[1]} ?v{i} . FILTER ( ?v{i} {op} {subp[2]} ) .")
            else:
                raise Exception(f"subp: {subp}; sub_programs: {sub_programs}")

        elif subp[0] in ["ARGMIN", "ARGMAX"]:
            # TODO: 暂时只考虑 Sexp 中只出现一次 ARGMAX 的情况
            '''
            subp[1]: #n
            subp[2]:
                - relation
                - R relation
                - #n
            '''
            if subp[1][0] == '#':
                var1 = int(subp[1][1:])
                rooti = get_root(int(i))
                root1 = get_root(var1)
                if rooti > root1:
                    identical_variables_r[rooti] = root1
                else:
                    identical_variables_r[root1] = rooti
                    root1 = rooti
                
                if subp[2][0] == '#': # 合并变量即可
                    var2 = int(subp[2][1:])
                    root2 = get_root(var2)
                    if root1 > root2:
                        identical_variables_r[root1] = root2
                    else:
                        identical_variables_r[root2] = root1
                        root2 = root1
                    
                    # 标记一下变量，后面加个 order by
                    if subp[0] == 'ARGMIN':
                        order_clauses = [f"?arg{root2}", "ASC", 1] # 目前默认都是 LIMIT 1
                    elif subp[0] == 'ARGMAX':
                        order_clauses = [f"?arg{root2}", "DESC", 1]   
    
                elif isinstance(subp[2], list): # R relation
                    clauses.append(f"?arg{root1} {subp[2][1]} ?x{root1} .")
                    if subp[0] == 'ARGMIN':
                        order_clauses = [f"?arg{root1}", "ASC", 1] # 目前默认都是 LIMIT 1
                    elif subp[0] == 'ARGMAX':
                        order_clauses = [f"?arg{root1}", "DESC", 1]
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?x{root1} {subp[2]} ?arg{root1} .")
                    if subp[0] == 'ARGMIN':
                        order_clauses = [f"?arg{root1}", "ASC", 1] # 目前默认都是 LIMIT 1
                    elif subp[0] == 'ARGMAX':
                        order_clauses = [f"?arg{root1}", "DESC", 1]
            else:  
                raise Exception(f"subp: {subp}; sub_programs: {sub_programs}")


        elif subp[0] == 'COUNT':  # this is easy, since it can only be applied to the quesiton node
            var = int(subp[1][1:])
            root_var = get_root(var)
            identical_variables_r[int(i)] = root_var  # COUNT can only be the outtermost
            count = True
        
        elif subp[0] in ['ARGMIN_JOIN', 'ARGMAX_JOIN']:
            '''
            subp[1]:
            - relation
            - R relation

            subp[2]
            - relation
            - R relation
            '''
            if isinstance(subp[1], list): # R relation
                if isinstance(subp[2], list): # R relation
                    clauses.append(f"?arg{i} {subp[2][1]} ?y{i} . ?y{i} {subp[1][1]} ?x{i} .")
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?y{i} {subp[2]} ?arg{i} . ?y{i} {subp[1][1]} ?x{i} .")
                else:
                    raise Exception(f"subp: {subp}")
            elif isinstance(subp[1], str): # relation
                if isinstance(subp[2], list): # R relation
                    clauses.append(f"?arg{i} {subp[2][1]} ?y{i} . ?x{i} {subp[1]} ?y{i} .")
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?y{i} {subp[2]} ?arg{i} . ?x{i} {subp[1]} ?y{i} .")
                else:
                    raise Exception(f"subp: {subp}")
            else:
                raise Exception(f"subp: {subp}")


        elif subp[0] in ['LT_JOIN', 'LE_JOIN', "GT_JOIN", "GE_JOIN", "EQ_JOIN"]:
            '''
            subp[1]:
            - relation
            - R relation

            subp[2]
            - relation
            - R relation
            '''
            if isinstance(subp[1], list): # R relation
                if isinstance(subp[2], list): # R relation
                    clauses.append(f"?v{i} {subp[2][1]} ?y{i} . ?y{i} {subp[1][1]} ?x{i} .")
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?y{i} {subp[2]} ?v{i} . ?y{i} {subp[1][1]} ?x{i} .")
                else:
                    raise Exception(f"subp: {subp}")
            elif isinstance(subp[1], str):
                if isinstance(subp[2], list): # R relation
                    clauses.append(f"?v{i} {subp[2][1]} ?y{i} . ?x{i} {subp[1]} ?y{i} .")
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?y{i} {subp[2]} ?v{i} . ?x{i} {subp[1]} ?y{i} .")
                else:
                    raise Exception(f"subp: {subp}")
            else:
                raise Exception(f"subp: {subp}")
        
        else:
            raise Exception(f"subp: {subp}")
    
    #  Merge identical variables
    for i in range(len(clauses)):
        for k in identical_variables_r:
            clauses[i] = clauses[i].replace(f'?x{k} ', f'?x{get_root(k)} ')
            clauses[i] = clauses[i].replace(f'?y{k} ', f'?y{get_root(k)} ')
            clauses[i] = clauses[i].replace(f'?v{k} ', f'?v{get_root(k)} ')
            clauses[i] = clauses[i].replace(f'?lit{k} ', f'?lit{get_root(k)} ')
            clauses[i] = clauses[i].replace(f'?arg{k} ', f'?arg{get_root(k)} ')
    
    # if order_clauses and (order_clauses[0].startswith('?v')):
    #     variable_index = int(order_clauses[0][2:])
    #     order_clauses[0] = f"?v{get_root(variable_index)}"
    if order_clauses is not None:
        for k in identical_variables_r:
            order_clauses[0] = order_clauses[0].replace(f'?arg{k}', f'?arg{get_root(k)}')

    question_var = get_root(question_var)

    for i in range(len(clauses)):
        clauses[i] = clauses[i].replace(f'?x{question_var} ', f'?x ')
    
    # if order_clauses is not None:
    #     arg_clauses = clauses[:]
    
    # TODO: 值得商榷，我感觉没什么必要
    # for entity in entities:
    #     clauses.append(f'FILTER (?x != ns:{entity})')
    clauses.insert(0,
                   f"FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))")
    clauses.insert(0, "WHERE {")
    if count:
        clauses.insert(0, f"SELECT COUNT DISTINCT ?x")
    # elif order_clauses is not None: # ARGMIN / ARGMAX, 如果存在多个取值相同且都为 top 的元素，SPARQL 需要做特殊处理
    #     clauses.insert(0, "{SELECT " + order_clauses[0])
    #     clauses = arg_clauses + clauses
    #     clauses.insert(0, "WHERE {")
    #     clauses.insert(0, f"SELECT DISTINCT ?x")
    else:
        clauses.insert(0, f"SELECT DISTINCT ?x")

    clauses.append('}')
    if order_clauses is not None:
        clauses.append(f"ORDER BY {order_clauses[1]}({order_clauses[0]}) LIMIT {order_clauses[2]}")
        # # 子查询的 WHERE 对应的 }
        # clauses.append('}')
        # # 子查询对应的 }
        # clauses.append('}')

    return '\n'.join(clauses)

def sexp_to_sparql_wikidata_for_edit_distance(lisp_program: str):
    '''
    变量说明:
    - ?x{i}: entity 集合
    - ?y{i}: 中间变量
    - ?lit{i}: string 集合
    - ?v{i}: time / number 集合
    - ?arg{i}: ARGMIN / ARGMAX 的排序目标

    用于编辑距离计算的版本，修改之处用 EDIT: 标识
    '''
    clauses = []
    # 仅支持出现一次 ARGMAX 的场景
    order_clauses = None # [变量名, 方向, (LIMIT) n] --> [?arg0, ASC / DESC, 1]
    entities = set()  # collect entites for filtering
    classes = set()
    identical_variables_r = {}  # key should be larger than value
    expression = lisp_to_nested_expression(lisp_program)
    count = False

    sub_programs = _linearize_lisp_expression(expression, [0])
    question_var = len(sub_programs) - 1
    
    def get_root(var: int):
        while var in identical_variables_r:
            var = identical_variables_r[var]

        return var

    for i, subp in enumerate(sub_programs):
        i = str(i)
        if subp[0] == 'JOIN':
            '''
            subp[1] 我认为只有两种选择:
            - relation
            - R relation
            这两者只有 SPARQL 里面三元组方向的区别

            无论 subp[1] 是什么, subp[2] 有如下选择
            - item: entity / class / literal
            - #n: 表示一个嵌套的子结构
            - relation --> 合起来表示一个多跳的关系
            - R relation
            '''
            if isinstance(subp[1], list):  # R relation
                if WikidataConstantForConstruction.get_constant_type(subp[2]) is WIKIDATA_CONSTANT_TYPE.ENTITY: # entity
                    clauses.append(subp[2] + " " + subp[1][1] + " ?x" + i + " .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':  # 嵌套子结构
                    clauses.append(f"?x{get_root(int(subp[2][1:]))} {subp[1][1]} ?x{i} .")
                elif WikidataConstantForConstruction.get_constant_type(subp[2]) in [WIKIDATA_CONSTANT_TYPE.TIME, WIKIDATA_CONSTANT_TYPE.QUANTITY, WIKIDATA_CONSTANT_TYPE.STRING]: # literal
                    if WikidataConstantForConstruction.get_constant_type(subp[2]) == WIKIDATA_CONSTANT_TYPE.STRING:
                        subp[2] = subp[2].replace("@en", "")
                        clauses.append(f"?lit{i} {subp[1][1]} ?x{i} . FILTER (isLiteral( ?lit{i} )) . FILTER (str( ?lit{i} ) = {subp[2]} ) .")
                    # 其他类型的 Literal, 传入 S-expression 时就已经处理好了
                    else:
                        clauses.append(subp[2] + " " + subp[1][1] + " ?x" + i + " .")
                else: 
                    raise Exception(f"subp: {subp}")

            elif isinstance(subp[1], str): # relation
                if WikidataConstantForConstruction.get_constant_type(subp[2]) is WIKIDATA_CONSTANT_TYPE.ENTITY:  # entity
                    clauses.append("?x" + i + " " + subp[1] + " " + subp[2] + " .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':  # variable
                    clauses.append(f"?x{i} {subp[1]} ?x{get_root(int(subp[2][1:]))} .")
                elif WikidataConstantForConstruction.get_constant_type(subp[2]) in [WIKIDATA_CONSTANT_TYPE.TIME, WIKIDATA_CONSTANT_TYPE.QUANTITY, WIKIDATA_CONSTANT_TYPE.STRING]:  # literal
                    if WikidataConstantForConstruction.get_constant_type(subp[2]) is WIKIDATA_CONSTANT_TYPE.STRING:
                        subp[2] = subp[2].replace("@en", "")
                        clauses.append(f"?x{i} {subp[1]} ?lit{i} . FILTER (isLiteral( ?lit{i} )) . FILTER (str( ?lit{i} ) = {subp[2]} )")
                    # 其他类型的 Literal, 传入 S-expression 时就已经处理好了
                    else:
                        clauses.append("?x" + i + " " + subp[1] + " " + subp[2] + " .")
                else: 
                    raise Exception(f"subp: {subp}")
            else:
                raise Exception(f"subp[1]: {subp[1]}; sub_programs: {sub_programs}")
    
        elif subp[0] == 'AND': 
            var1 = int(subp[1][1:])
            rooti = get_root(int(i))
            root1 = get_root(var1)
            if rooti > root1:
                identical_variables_r[rooti] = root1
            else:
                identical_variables_r[root1] = rooti
                root1 = rooti
            var2 = int(subp[2][1:])
            root2 = get_root(var2)
            if root1 > root2:
                identical_variables_r[root1] = root2
            else:
                identical_variables_r[root2] = root1
        elif subp[0] in ['LE', 'LT', 'GE', 'GT', 'EQ']:  
            '''
            subp[1]:
                - 嵌套结构, #n
                - 关系 / 逆关系
            subp[2]:
                - 嵌套结构, #n
                - time / number
            '''
            if subp[0] == 'LE':
                op = "<="
            elif subp[0] == 'LT':
                op = "<"
            elif subp[0] == 'GE':
                op = ">="
            elif subp[0] == 'GT':
                op = ">"
            elif subp[0] == 'EQ':
                op = "="
            else:
                raise Exception(f"op: {op}; sub_programs: {sub_programs}")
            if subp[1].startswith('#'): # 嵌套
                var1 = int(subp[1][1:])
                rooti = get_root(int(i))
                root1 = get_root(var1)
                if rooti > root1:
                    identical_variables_r[rooti] = root1
                else:
                    identical_variables_r[root1] = rooti

                if subp[2].startswith('#'): # 嵌套
                    root2 = get_root(int(subp[2][1:]))
                    # 嵌套的变量应该是以 x 开头的，因为只是单纯去求解一个值
                    clauses.append(f"FILTER ( ?v{root1} {op} ?x{root2} ) .")
                else: # literal, 并且只能是 time / number
                    if WikidataConstantForConstruction.get_constant_type(subp[2]) not in [WIKIDATA_CONSTANT_TYPE.QUANTITY, WIKIDATA_CONSTANT_TYPE.TIME]:
                        raise Exception(f"subp[2]: {subp[2]}, sub_programs: {sub_programs}")
                    clauses.append(f"FILTER ( ?v{root1} {op} {subp[2]} ) .")
            elif isinstance(subp[1], list): # R relation
                if subp[2].startswith('#'): # 嵌套
                    root2 = get_root(int(subp[2][1:]))
                    clauses.append(f"?v{i} {subp[1][1]} ?x{i} . FILTER ( ?v{i} {op} ?v{root2} ) .")
                else: # literal, 并且只能是 time / number
                    if WikidataConstantForConstruction.get_constant_type(subp[2]) not in [WIKIDATA_CONSTANT_TYPE.QUANTITY, WIKIDATA_CONSTANT_TYPE.TIME]:
                        raise Exception(f"subp[2]: {subp[2]}, sub_programs: {sub_programs}")
                    clauses.append(f"?v{i} {subp[1][1]} ?x{i} . FILTER ( ?v{i} {op} {subp[2]} ) .")
            elif isinstance(subp[1], str): # relation
                if subp[2].startswith('#'): # 嵌套
                    root2 = get_root(int(subp[2][1:]))
                    clauses.append(f"?x{i} {subp[1]} ?v{i}. FILTER ( ?v{i} {op} ?x{root2} ) .")
                else: # literal, 并且只能是 time / number
                    if WikidataConstantForConstruction.get_constant_type(subp[2]) not in [WIKIDATA_CONSTANT_TYPE.QUANTITY, WIKIDATA_CONSTANT_TYPE.TIME]:
                        raise Exception(f"subp[2]: {subp[2]}, sub_programs: {sub_programs}")
                    clauses.append(f"?x{i} {subp[1]} ?v{i} . FILTER ( ?v{i} {op} {subp[2]} ) .")
            else:
                raise Exception(f"subp: {subp}; sub_programs: {sub_programs}")

        elif subp[0] in ["ARGMIN", "ARGMAX"]:
            # TODO: 暂时只考虑 Sexp 中只出现一次 ARGMAX 的情况
            '''
            subp[1]: #n
            subp[2]:
                - relation
                - R relation
                - #n
            '''
            if subp[1][0] == '#':
                var1 = int(subp[1][1:])
                rooti = get_root(int(i))
                root1 = get_root(var1)
                if rooti > root1:
                    identical_variables_r[rooti] = root1
                else:
                    identical_variables_r[root1] = rooti
                    root1 = rooti
                
                if subp[2][0] == '#': # 合并变量即可
                    var2 = int(subp[2][1:])
                    root2 = get_root(var2)
                    if root1 > root2:
                        identical_variables_r[root1] = root2
                    else:
                        identical_variables_r[root2] = root1
                        root2 = root1
                    
                    # 标记一下变量，后面加个 order by
                    if subp[0] == 'ARGMIN':
                        order_clauses = [f"?arg{root2}", "ASC", 1] # 目前默认都是 LIMIT 1
                    elif subp[0] == 'ARGMAX':
                        order_clauses = [f"?arg{root2}", "DESC", 1]   
    
                elif isinstance(subp[2], list): # R relation
                    clauses.append(f"?arg{root1} {subp[2][1]} ?x{root1} .")
                    if subp[0] == 'ARGMIN':
                        order_clauses = [f"?arg{root1}", "ASC", 1] # 目前默认都是 LIMIT 1
                    elif subp[0] == 'ARGMAX':
                        order_clauses = [f"?arg{root1}", "DESC", 1]
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?x{root1} {subp[2]} ?arg{root1} .")
                    if subp[0] == 'ARGMIN':
                        order_clauses = [f"?arg{root1}", "ASC", 1] # 目前默认都是 LIMIT 1
                    elif subp[0] == 'ARGMAX':
                        order_clauses = [f"?arg{root1}", "DESC", 1]
            else:  
                raise Exception(f"subp: {subp}; sub_programs: {sub_programs}")


        elif subp[0] == 'COUNT':  # this is easy, since it can only be applied to the quesiton node
            var = int(subp[1][1:])
            root_var = get_root(var)
            identical_variables_r[int(i)] = root_var  # COUNT can only be the outtermost
            count = True
        
        elif subp[0] in ['ARGMIN_JOIN', 'ARGMAX_JOIN']:
            '''
            subp[1]:
            - relation
            - R relation

            subp[2]
            - relation
            - R relation
            '''
            if isinstance(subp[1], list): # R relation
                if isinstance(subp[2], list): # R relation
                    clauses.append(f"?arg{i} {subp[2][1]} ?y{i} . ?y{i} {subp[1][1]} ?x{i} .")
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?y{i} {subp[2]} ?arg{i} . ?y{i} {subp[1][1]} ?x{i} .")
                else:
                    raise Exception(f"subp: {subp}")
            elif isinstance(subp[1], str): # relation
                if isinstance(subp[2], list): # R relation
                    clauses.append(f"?arg{i} {subp[2][1]} ?y{i} . ?x{i} {subp[1]} ?y{i} .")
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?y{i} {subp[2]} ?arg{i} . ?x{i} {subp[1]} ?y{i} .")
                else:
                    raise Exception(f"subp: {subp}")
            else:
                raise Exception(f"subp: {subp}")


        elif subp[0] in ['LT_JOIN', 'LE_JOIN', "GT_JOIN", "GE_JOIN", "EQ_JOIN"]:
            '''
            subp[1]:
            - relation
            - R relation

            subp[2]
            - relation
            - R relation
            '''
            if isinstance(subp[1], list): # R relation
                if isinstance(subp[2], list): # R relation
                    clauses.append(f"?v{i} {subp[2][1]} ?y{i} . ?y{i} {subp[1][1]} ?x{i} .")
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?y{i} {subp[2]} ?v{i} . ?y{i} {subp[1][1]} ?x{i} .")
                else:
                    raise Exception(f"subp: {subp}")
            elif isinstance(subp[1], str):
                if isinstance(subp[2], list): # R relation
                    clauses.append(f"?v{i} {subp[2][1]} ?y{i} . ?x{i} {subp[1]} ?y{i} .")
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?y{i} {subp[2]} ?v{i} . ?x{i} {subp[1]} ?y{i} .")
                else:
                    raise Exception(f"subp: {subp}")
            else:
                raise Exception(f"subp: {subp}")
        
        else:
            raise Exception(f"subp: {subp}")
    
    #  Merge identical variables
    for i in range(len(clauses)):
        for k in identical_variables_r:
            clauses[i] = clauses[i].replace(f'?x{k} ', f'?x{get_root(k)} ')
            clauses[i] = clauses[i].replace(f'?y{k} ', f'?y{get_root(k)} ')
            clauses[i] = clauses[i].replace(f'?v{k} ', f'?v{get_root(k)} ')
            clauses[i] = clauses[i].replace(f'?lit{k} ', f'?lit{get_root(k)} ')
            clauses[i] = clauses[i].replace(f'?arg{k} ', f'?arg{get_root(k)} ')
    
    # if order_clauses and (order_clauses[0].startswith('?v')):
    #     variable_index = int(order_clauses[0][2:])
    #     order_clauses[0] = f"?v{get_root(variable_index)}"
    if order_clauses is not None:
        for k in identical_variables_r:
            order_clauses[0] = order_clauses[0].replace(f'?arg{k}', f'?arg{get_root(k)}')

    question_var = get_root(question_var)

    for i in range(len(clauses)):
        clauses[i] = clauses[i].replace(f'?x{question_var} ', f'?x ')
    
    # if order_clauses is not None:
    #     arg_clauses = clauses[:]
    
    # TODO: 值得商榷，我感觉没什么必要
    # for entity in entities:
    #     clauses.append(f'FILTER (?x != ns:{entity})')
    clauses.insert(0,
                   f"FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))")
    clauses.insert(0, "WHERE {")
    if count: # EDIT:
        clauses.insert(0, f"SELECT (COUNT (DISTINCT ?x) as ?cnt)")
    # elif order_clauses is not None: # ARGMIN / ARGMAX, 如果存在多个取值相同且都为 top 的元素，SPARQL 需要做特殊处理
    #     clauses.insert(0, "{SELECT " + order_clauses[0])
    #     clauses = arg_clauses + clauses
    #     clauses.insert(0, "WHERE {")
    #     clauses.insert(0, f"SELECT DISTINCT ?x")
    else:
        clauses.insert(0, f"SELECT DISTINCT ?x")

    clauses.append('}')
    if order_clauses is not None:
        clauses.append(f"ORDER BY {order_clauses[1]}({order_clauses[0]}) LIMIT {order_clauses[2]}")
        # # 子查询的 WHERE 对应的 }
        # clauses.append('}')
        # # 子查询对应的 }
        # clauses.append('}')
    
    # EDIT:
    clauses.insert(0, f"{WIKIDATA_PREFIX_LIST}\n")

    return '\n'.join(clauses)

def sexp_to_sparql_wikidata_for_test_suite(lisp_program: str):
    '''
    变量说明:
    - ?x{i}: entity 集合
    - ?y{i}: 中间变量
    - ?lit{i}: string 集合
    - ?v{i}: time / number 集合
    - ?arg{i}: ARGMIN / ARGMAX 的排序目标

    用于Test Suite Consistency 指标的计算，修改之处用 EDIT: 标识
    '''
    clauses = []
    # 仅支持出现一次 ARGMAX 的场景
    order_clauses = None # [变量名, 方向, (LIMIT) n] --> [?arg0, ASC / DESC, 1]
    entities = set()  # collect entites for filtering
    classes = set()
    identical_variables_r = {}  # key should be larger than value
    expression = lisp_to_nested_expression(lisp_program)
    count = False

    sub_programs = _linearize_lisp_expression(expression, [0])
    question_var = len(sub_programs) - 1
    
    def get_root(var: int):
        while var in identical_variables_r:
            var = identical_variables_r[var]

        return var

    for i, subp in enumerate(sub_programs):
        i = str(i)
        if subp[0] == 'JOIN':
            '''
            subp[1] 我认为只有两种选择:
            - relation
            - R relation
            这两者只有 SPARQL 里面三元组方向的区别

            无论 subp[1] 是什么, subp[2] 有如下选择
            - item: entity / class / literal
            - #n: 表示一个嵌套的子结构
            - relation --> 合起来表示一个多跳的关系
            - R relation
            '''
            if isinstance(subp[1], list):  # R relation
                if WikidataConstantForConstruction.get_constant_type(subp[2]) is WIKIDATA_CONSTANT_TYPE.ENTITY: # entity
                    clauses.append(subp[2] + " " + subp[1][1] + " ?x" + i + " .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':  # 嵌套子结构
                    clauses.append(f"?x{get_root(int(subp[2][1:]))} {subp[1][1]} ?x{i} .")
                elif WikidataConstantForConstruction.get_constant_type(subp[2]) in [WIKIDATA_CONSTANT_TYPE.TIME, WIKIDATA_CONSTANT_TYPE.QUANTITY, WIKIDATA_CONSTANT_TYPE.STRING]: # literal
                    if WikidataConstantForConstruction.get_constant_type(subp[2]) == WIKIDATA_CONSTANT_TYPE.STRING:
                        subp[2] = subp[2].replace("@en", "")
                        clauses.append(f"?lit{i} {subp[1][1]} ?x{i} . FILTER (isLiteral( ?lit{i} )) . FILTER (str( ?lit{i} ) = {subp[2]} ) .")
                    # 其他类型的 Literal, 传入 S-expression 时就已经处理好了
                    else:
                        clauses.append(subp[2] + " " + subp[1][1] + " ?x" + i + " .")
                else: 
                    raise Exception(f"subp: {subp}")

            elif isinstance(subp[1], str): # relation
                if WikidataConstantForConstruction.get_constant_type(subp[2]) is WIKIDATA_CONSTANT_TYPE.ENTITY:  # entity
                    clauses.append("?x" + i + " " + subp[1] + " " + subp[2] + " .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':  # variable
                    clauses.append(f"?x{i} {subp[1]} ?x{get_root(int(subp[2][1:]))} .")
                elif WikidataConstantForConstruction.get_constant_type(subp[2]) in [WIKIDATA_CONSTANT_TYPE.TIME, WIKIDATA_CONSTANT_TYPE.QUANTITY, WIKIDATA_CONSTANT_TYPE.STRING]:  # literal
                    if WikidataConstantForConstruction.get_constant_type(subp[2]) is WIKIDATA_CONSTANT_TYPE.STRING:
                        subp[2] = subp[2].replace("@en", "")
                        clauses.append(f"?x{i} {subp[1]} ?lit{i} . FILTER (isLiteral( ?lit{i} )) . FILTER (str( ?lit{i} ) = {subp[2]} )")
                    # 其他类型的 Literal, 传入 S-expression 时就已经处理好了
                    else:
                        clauses.append("?x" + i + " " + subp[1] + " " + subp[2] + " .")
                else: 
                    raise Exception(f"subp: {subp}")
            else:
                raise Exception(f"subp[1]: {subp[1]}; sub_programs: {sub_programs}")
    
        elif subp[0] == 'AND': 
            var1 = int(subp[1][1:])
            rooti = get_root(int(i))
            root1 = get_root(var1)
            if rooti > root1:
                identical_variables_r[rooti] = root1
            else:
                identical_variables_r[root1] = rooti
                root1 = rooti
            var2 = int(subp[2][1:])
            root2 = get_root(var2)
            if root1 > root2:
                identical_variables_r[root1] = root2
            else:
                identical_variables_r[root2] = root1
        elif subp[0] in ['LE', 'LT', 'GE', 'GT', 'EQ']:  
            '''
            subp[1]:
                - 嵌套结构, #n
                - 关系 / 逆关系
            subp[2]:
                - 嵌套结构, #n
                - time / number
            '''
            if subp[0] == 'LE':
                op = "<="
            elif subp[0] == 'LT':
                op = "<"
            elif subp[0] == 'GE':
                op = ">="
            elif subp[0] == 'GT':
                op = ">"
            elif subp[0] == 'EQ':
                op = "="
            else:
                raise Exception(f"op: {op}; sub_programs: {sub_programs}")
            if subp[1].startswith('#'): # 嵌套
                var1 = int(subp[1][1:])
                rooti = get_root(int(i))
                root1 = get_root(var1)
                if rooti > root1:
                    identical_variables_r[rooti] = root1
                else:
                    identical_variables_r[root1] = rooti

                if subp[2].startswith('#'): # 嵌套
                    root2 = get_root(int(subp[2][1:]))
                    # 嵌套的变量应该是以 x 开头的，因为只是单纯去求解一个值
                    clauses.append(f"FILTER ( ?v{root1} {op} ?x{root2} ) .")
                else: # literal, 并且只能是 time / number
                    if WikidataConstantForConstruction.get_constant_type(subp[2]) not in [WIKIDATA_CONSTANT_TYPE.QUANTITY, WIKIDATA_CONSTANT_TYPE.TIME]:
                        raise Exception(f"subp[2]: {subp[2]}, sub_programs: {sub_programs}")
                    clauses.append(f"FILTER ( ?v{root1} {op} {subp[2]} ) .")
            elif isinstance(subp[1], list): # R relation
                if subp[2].startswith('#'): # 嵌套
                    root2 = get_root(int(subp[2][1:]))
                    clauses.append(f"?v{i} {subp[1][1]} ?x{i} . FILTER ( ?v{i} {op} ?v{root2} ) .")
                else: # literal, 并且只能是 time / number
                    if WikidataConstantForConstruction.get_constant_type(subp[2]) not in [WIKIDATA_CONSTANT_TYPE.QUANTITY, WIKIDATA_CONSTANT_TYPE.TIME]:
                        raise Exception(f"subp[2]: {subp[2]}, sub_programs: {sub_programs}")
                    clauses.append(f"?v{i} {subp[1][1]} ?x{i} . FILTER ( ?v{i} {op} {subp[2]} ) .")
            elif isinstance(subp[1], str): # relation
                if subp[2].startswith('#'): # 嵌套
                    root2 = get_root(int(subp[2][1:]))
                    clauses.append(f"?x{i} {subp[1]} ?v{i}. FILTER ( ?v{i} {op} ?x{root2} ) .")
                else: # literal, 并且只能是 time / number
                    if WikidataConstantForConstruction.get_constant_type(subp[2]) not in [WIKIDATA_CONSTANT_TYPE.QUANTITY, WIKIDATA_CONSTANT_TYPE.TIME]:
                        raise Exception(f"subp[2]: {subp[2]}, sub_programs: {sub_programs}")
                    clauses.append(f"?x{i} {subp[1]} ?v{i} . FILTER ( ?v{i} {op} {subp[2]} ) .")
            else:
                raise Exception(f"subp: {subp}; sub_programs: {sub_programs}")

        elif subp[0] in ["ARGMIN", "ARGMAX"]:
            # TODO: 暂时只考虑 Sexp 中只出现一次 ARGMAX 的情况
            '''
            subp[1]: #n
            subp[2]:
                - relation
                - R relation
                - #n
            '''
            if subp[1][0] == '#':
                var1 = int(subp[1][1:])
                rooti = get_root(int(i))
                root1 = get_root(var1)
                if rooti > root1:
                    identical_variables_r[rooti] = root1
                else:
                    identical_variables_r[root1] = rooti
                    root1 = rooti
                
                if subp[2][0] == '#': # 合并变量即可
                    var2 = int(subp[2][1:])
                    root2 = get_root(var2)
                    if root1 > root2:
                        identical_variables_r[root1] = root2
                    else:
                        identical_variables_r[root2] = root1
                        root2 = root1
                    
                    # 标记一下变量，后面加个 order by
                    if subp[0] == 'ARGMIN':
                        order_clauses = [f"?arg{root2}", "ASC", 1] # 目前默认都是 LIMIT 1
                    elif subp[0] == 'ARGMAX':
                        order_clauses = [f"?arg{root2}", "DESC", 1]   
    
                elif isinstance(subp[2], list): # R relation
                    clauses.append(f"?arg{root1} {subp[2][1]} ?x{root1} .")
                    if subp[0] == 'ARGMIN':
                        order_clauses = [f"?arg{root1}", "ASC", 1] # 目前默认都是 LIMIT 1
                    elif subp[0] == 'ARGMAX':
                        order_clauses = [f"?arg{root1}", "DESC", 1]
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?x{root1} {subp[2]} ?arg{root1} .")
                    if subp[0] == 'ARGMIN':
                        order_clauses = [f"?arg{root1}", "ASC", 1] # 目前默认都是 LIMIT 1
                    elif subp[0] == 'ARGMAX':
                        order_clauses = [f"?arg{root1}", "DESC", 1]
            else:  
                raise Exception(f"subp: {subp}; sub_programs: {sub_programs}")


        elif subp[0] == 'COUNT':  # this is easy, since it can only be applied to the quesiton node
            var = int(subp[1][1:])
            root_var = get_root(var)
            identical_variables_r[int(i)] = root_var  # COUNT can only be the outtermost
            count = True
        
        elif subp[0] in ['ARGMIN_JOIN', 'ARGMAX_JOIN']:
            '''
            subp[1]:
            - relation
            - R relation

            subp[2]
            - relation
            - R relation
            '''
            if isinstance(subp[1], list): # R relation
                if isinstance(subp[2], list): # R relation
                    clauses.append(f"?arg{i} {subp[2][1]} ?y{i} . ?y{i} {subp[1][1]} ?x{i} .")
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?y{i} {subp[2]} ?arg{i} . ?y{i} {subp[1][1]} ?x{i} .")
                else:
                    raise Exception(f"subp: {subp}")
            elif isinstance(subp[1], str): # relation
                if isinstance(subp[2], list): # R relation
                    clauses.append(f"?arg{i} {subp[2][1]} ?y{i} . ?x{i} {subp[1]} ?y{i} .")
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?y{i} {subp[2]} ?arg{i} . ?x{i} {subp[1]} ?y{i} .")
                else:
                    raise Exception(f"subp: {subp}")
            else:
                raise Exception(f"subp: {subp}")


        elif subp[0] in ['LT_JOIN', 'LE_JOIN', "GT_JOIN", "GE_JOIN", "EQ_JOIN"]:
            '''
            subp[1]:
            - relation
            - R relation

            subp[2]
            - relation
            - R relation
            '''
            if isinstance(subp[1], list): # R relation
                if isinstance(subp[2], list): # R relation
                    clauses.append(f"?v{i} {subp[2][1]} ?y{i} . ?y{i} {subp[1][1]} ?x{i} .")
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?y{i} {subp[2]} ?v{i} . ?y{i} {subp[1][1]} ?x{i} .")
                else:
                    raise Exception(f"subp: {subp}")
            elif isinstance(subp[1], str):
                if isinstance(subp[2], list): # R relation
                    clauses.append(f"?v{i} {subp[2][1]} ?y{i} . ?x{i} {subp[1]} ?y{i} .")
                elif isinstance(subp[2], str): # relation
                    clauses.append(f"?y{i} {subp[2]} ?v{i} . ?x{i} {subp[1]} ?y{i} .")
                else:
                    raise Exception(f"subp: {subp}")
            else:
                raise Exception(f"subp: {subp}")
        
        else:
            raise Exception(f"subp: {subp}")
    
    #  Merge identical variables
    for i in range(len(clauses)):
        for k in identical_variables_r:
            clauses[i] = clauses[i].replace(f'?x{k} ', f'?x{get_root(k)} ')
            clauses[i] = clauses[i].replace(f'?y{k} ', f'?y{get_root(k)} ')
            clauses[i] = clauses[i].replace(f'?v{k} ', f'?v{get_root(k)} ')
            clauses[i] = clauses[i].replace(f'?lit{k} ', f'?lit{get_root(k)} ')
            clauses[i] = clauses[i].replace(f'?arg{k} ', f'?arg{get_root(k)} ')
    
    # if order_clauses and (order_clauses[0].startswith('?v')):
    #     variable_index = int(order_clauses[0][2:])
    #     order_clauses[0] = f"?v{get_root(variable_index)}"
    if order_clauses is not None:
        for k in identical_variables_r:
            order_clauses[0] = order_clauses[0].replace(f'?arg{k}', f'?arg{get_root(k)}')

    question_var = get_root(question_var)

    for i in range(len(clauses)):
        clauses[i] = clauses[i].replace(f'?x{question_var} ', f'?x ')
    
    # if order_clauses is not None:
    #     arg_clauses = clauses[:]
    
    # TODO: 值得商榷，我感觉没什么必要
    # for entity in entities:
    #     clauses.append(f'FILTER (?x != ns:{entity})')
    clauses.insert(0,
                   f"FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))")
    clauses.insert(0, "WHERE {")
    if count: # EDIT:
        clauses.insert(0, f"SELECT (COUNT (DISTINCT ?x) as ?cnt)")
    # elif order_clauses is not None: # ARGMIN / ARGMAX, 如果存在多个取值相同且都为 top 的元素，SPARQL 需要做特殊处理
    #     clauses.insert(0, "{SELECT " + order_clauses[0])
    #     clauses = arg_clauses + clauses
    #     clauses.insert(0, "WHERE {")
    #     clauses.insert(0, f"SELECT DISTINCT ?x")
    else:
        clauses.insert(0, f"SELECT DISTINCT ?x")

    clauses.append('}')
    if order_clauses is not None:
        clauses.append(f"ORDER BY {order_clauses[1]}({order_clauses[0]}) LIMIT {order_clauses[2]}")
        # # 子查询的 WHERE 对应的 }
        # clauses.append('}')
        # # 子查询对应的 }
        # clauses.append('}')
    
    # EDIT:
    clauses.insert(0, f"{WIKIDATA_PREFIX_LIST}\n")

    return '\n'.join(clauses)
