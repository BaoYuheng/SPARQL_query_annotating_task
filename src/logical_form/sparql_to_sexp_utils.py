import re

class SparqlParserLC2: # LC-QuAD2
    @classmethod
    def preprocess_sparql(cls, query):
        query = query.replace('{', '\n{\n').replace('}', '\n}\n').replace('.', '\n.\n').replace('ORDER', '\nORDER').replace('order', '\norder')
        return query
    
    @classmethod
    def parse_sparql(cls, query):
        """parse a sparql query into a s-expression

        @param query: sparql query
        """
        query = cls.preprocess_sparql(query)
        lines = query.split('\n')
        lines = [x.strip() for x in lines if x]
        count_flag = False # 对于目标变量求 COUNT

        first_line = lines[0]
        first_line_lower = first_line.lower()
        if not first_line_lower.startswith('select'):
            raise NotImplementedError(f"first_line: {first_line}") # ASK 等类型语句，无法处理
        if first_line_lower.startswith('select'):
            if 'distinct' in first_line_lower: # select distinct ?x where
                variable_segment = first_line[(first_line_lower.index('distinct')+8):first_line_lower.index('where')]
                variable = variable_segment.strip().split()
                if len(variable) != 1:
                    raise NotImplementedError(f"first_line: {first_line}, variable: {variable}")
                else:
                    variable = variable[0]
            elif 'count' in first_line_lower:
                variable = re.findall(r"\((\?\w+)\)", first_line)
                if len(variable) != 1:
                    raise NotImplementedError(f"first_line: {first_line}, variable: {variable}")
                else:
                    count_flag = True
                    variable = variable[0]
            else:
                variable_segment = first_line[(first_line_lower.index('select')+6):first_line_lower.index('where')]
                variable = variable_segment.strip().split()
                if len(variable) != 1:
                    raise NotImplementedError(f"first_line: {first_line}, variable: {variable}")
                else:
                    variable = variable[0]
    
        '''
        LC2 里面会出现 ORDER BY ASC(?obj)LIMIT 5,
        但是其自然语言问题往往是对 maximum, largest 等提问，因此我认为这里 LIMIT 5 并没有道理，我们视作 LIMIT 1, 将其转换成 ARGMIN / ARGMAX
        这部分放到后面处理吧
        '''
        body_lines = lines[1:]
        body_lines, spec_condition, filter_lines = cls.normalize_body_lines(body_lines)
        var_dep_list = cls.parse_naive_body(body_lines, filter_lines, variable, spec_condition)
        s_expr = cls.dep_graph_to_s_expr(var_dep_list, variable, spec_condition, count_flag)
        return s_expr

    @classmethod
    def normalize_body_lines(cls, lines):
        """    
        return normalized body lines of sparql, 特别把 ARGMIN / ARGMAX 条件区分出来  
        观察到 LC2 里面的 filter 大部分是我们无法处理的，因此 如果出现 filter line, 就报错  

        @return: (body_lines,
                    spec_condition,
                    # [
                    #     ['SUPERLATIVE', argmax/argmin, arg_var, arg_r], 
                    #     ['COMPARATIVE', gt/lt/ge/le, compare_var, compare_value, compare_rel],
                    #     ['RANGE', range_relation, range_var, range_year],
                    # ]
                    filter_lines
                    )
        """

        spec_condition = []
        body_lines = []
        lines = [x.strip() for x in lines]
        filter_lines = None
        compare_variable = None

        # 1. get literal filter_lines
        
        for line in lines: # 不限定这种 special condition line 的出现位置
            line_lower = line.lower()
            if 'filter' in line_lower:
                raise NotImplementedError(f"filter line: {line}")
            elif re.search(r"limit \d", line_lower):
                direction = 'argmax' if ('desc(' in line_lower) else 'argmin'
                compare_variable = re.findall(r"\((\?\w+)\)", line_lower)
                if len(compare_variable) != 1:
                    raise NotImplementedError(f"line_lower: {line_lower}, variable: {compare_variable}")
                compare_variable = compare_variable[0]
                arg_var, arg_r = None, None
                for (line_idx, other_line) in enumerate(lines):
                    if other_line == line:
                        continue
                    if compare_variable in other_line:
                        if 'FILTER' in other_line: # the return var is also the argmax var, not covered by S-Expression
                            raise NotImplementedError(f"FILTER in {other_line}")
                        arg_var, arg_r = other_line.split(' ')[0], other_line.split(' ')[1]
                        break
                if arg_var is None or arg_r is None:
                    raise NotImplementedError(f"Cannot find arg relation and arg variable, lines: {lines}")
                superlative_cond = ['SUPERLATIVE', direction, arg_var, arg_r]
                spec_condition.append(superlative_cond)
            
            else:
                body_lines.append(line)
        if compare_variable:
            body_lines = [line for line in body_lines if compare_variable not in line]
        return body_lines, spec_condition, filter_lines
        
    @classmethod
    def dep_graph_to_s_expr(cls, var_dep_list, ret_var, spec_condition=None, count_flag=False):
        """Convert dependancy graph to s_expression
        @param var_dep_list: varialbe dependancy list
        @param ret_var: return var
        @param spec_condition: special condition

        @return s_expression
        """
        if not (var_dep_list[0][0] == ret_var):
            raise NotImplementedError(f"var_dep_list[0][0]: {var_dep_list[0][0]}; ret_var: {ret_var}")
        var_dep_list.reverse() # reverse the var_dep_list
        parsed_dict = {}  # dict for parsed variables

        # spec_condition,
        #             # [
        #             #     ['SUPERLATIVE', argmax/argmin, arg_var, arg_r], 
        #             #     ['COMPARATIVE', gt/lt/ge/le, compare_var, compare_value],
        #             #     ['RANGE', range_relation, range_var, range_year],
        #             # ]

        # specical condition var map {spec_var:idx in spec_condition}
        spec_var_map = {cond[2]:i for i,cond in enumerate(spec_condition)} if spec_condition else None
        # spec_var = spec_condition[1] if spec_condition is not None else None

        for var_name, dep_relations in var_dep_list:
            # expr = ''
            clause = cls.triplet_to_clause(
                var_name,  dep_relations[0], parsed_dict)
            for tri in dep_relations[1:]:
                n_clause = cls.triplet_to_clause(var_name, tri, parsed_dict)
                clause = 'AND ({}) ({})'.format(n_clause, clause)
            # if var_name == spec_var:
            if spec_var_map and var_name in spec_var_map: # spec_condition
                cond = spec_condition[spec_var_map[var_name]]
                # if cond[0] == 'argmax' or cond[0] == 'argmin': # superlative
                if cond[0]=='SUPERLATIVE':
                    #relation = spec_condition[2]
                    relation = cond[3]
                    if relation.startswith('wdt:'):
                        # 转成 p: + ps:
                        rel_name = relation[4:]
                        clause = f'{cond[1].upper()} ({clause}) (JOIN p:{rel_name} ps:{rel_name})'
                    else:
                        clause = '{} ({}) {}'.format(
                            cond[1].upper(), clause, relation)
                elif cond[0] == 'COMPARATIVE':
                    op = cond[1]
                    value = cond[3]
                    rel = cond[4]
                    if rel.startswith('wdt:'):
                        # 转成 p: + ps:
                        rel_name = rel[4:]
                        clause = f'{op} (JOIN p:{rel_name} ps:{rel_name}) {value}'
                    else:
                        clause = f'{op} {rel} {value}'
                    clause = 'AND ({}) ({})'.format(n_clause, clause)
                    # pass
            parsed_dict[var_name] = clause
        
        res = '(' + parsed_dict[ret_var] + ')'
        # TODO: 后面统一处理 Constant Serialization 问题
        for suffix in ['dateTime', 'date', 'gYearMonth', 'gYear', 'integer', 'float']: # 顺序很重要，子串应该放在后面
            res = res.replace(f'xsd:{suffix}',f'http://www.w3.org/2001/XMLSchema#{suffix}')
        if count_flag:
            return f"(COUNT {res})"
        return res

    @classmethod
    def triplet_to_clause(cls, tgt_var, triplet, parsed_dict):
        """Convert a triplet to S_expression clause
        @param tgt_var: target variable
        @param triplet: triplet in sparql
        @param parsed_dict: dict for variables already parsed
        """
        if triplet[0] == tgt_var:
            this = triplet[0]
            other = triplet[-1]
            if other in parsed_dict:
                other = '(' + parsed_dict[other] + ')'
            if triplet[1].startswith('wdt:'):
                rel_name = triplet[1][4:]
                return f"JOIN p:{rel_name} (JOIN ps:{rel_name} {other})"
            else:
                return 'JOIN {} {}'.format(triplet[1], other)
        elif triplet[-1] == tgt_var:
            this = triplet[-1]
            other = triplet[0]
            if other in parsed_dict:
                other = '(' + parsed_dict[other] + ')'
            if triplet[1].startswith('wdt:'):
                rel_name = triplet[1][4:]
                return f"JOIN (R ps:{rel_name}) (JOIN (R p:{rel_name}) {other})"
            else:
                return 'JOIN (R {}) {}'.format(triplet[1], other)
        else:
            raise NotImplementedError(f"triplet: {triplet}; tgt_var: {tgt_var}")

    @classmethod
    def parse_naive_body(cls, body_lines, filter_lines, ret_var, spec_condition=None):
        """Parse body lines
        @param body_lines: list of sparql body lines
        @param ret_var: return var, default `?x`
        @param filter_lines: lines that start with `FILTER (str(?`

        @return: variable dependancy list
        """
        triplets = [x.split(' ') for x in body_lines]  # split by '
        triplets = [line for line in triplets if len(line) == 3] # 观察了 LC-QuAD 中的 sparql, 总是恰好为三元组的形式
        # dependancy graph
        triplets_pool = triplets
        # while True:
        # varaible dependancy list, in the form like [(?x,[['?x','ns:aaa.aaa.aaa','?y'],['ns:m.xx','ns:bbb.bbb.bbb','?x''])]
        var_dep_list = []
        successors = []

        # firstly solve the return variable
        dep_triplets, triplets_pool = cls.resolve_dependancy(
            triplets_pool, filter_lines, ret_var, successors)
        var_dep_list.append((ret_var, dep_triplets))
        # handle all the successor variables
        while len(successors):
            tgt_var = successors[0]
            successors = successors[1:]
            dep_triplets, triplets_pool = cls.resolve_dependancy(
                triplets_pool, filter_lines, tgt_var, successors)

            # assert len(dep_triplets) > 0 # at least one dependancy triplets
            if len(dep_triplets) == 0:
                # zero dep_triples, can be a 2-hop constraint
                # e.g.
                # 'ns:m.0d0x8 ns:government.political_district.representatives ?y .'
                # '?y ns:government.government_position_held.office_holder ?x .'
                # '?y ns:government.government_position_held.governmental_body ns:m.07t58 .'
                # '?x ns:government.politician.government_positions_held ?c .'
                
                if spec_condition and any([tgt_var in x for x in spec_condition]):
                    cond = []
                    for x in spec_condition:
                        if tgt_var in x:
                            cond = x
                            break
                    
                    repeat = True
                    while repeat:        
                        # tgt_var is a var in spec_condition
                        for (var, triplets) in var_dep_list:
                            if any([tgt_var in trip for trip in triplets]):
                                head_var = var  # find the real constrained var
                                _temp_triplets = triplets[:]
                                triplets.clear()
                                for trip in _temp_triplets:
                                    if tgt_var not in trip:
                                        triplets.append(trip)
                                    else:
                                        # find the constraint relation
                                        cons_rel = trip[1]
                                        if trip[0] == head_var:
                                            reversed_direction = False
                                        else:
                                            reversed_direction = True
                                        cons_rel = f'(R {cons_rel})' if reversed_direction else cons_rel

                                # modify spec_condition
                                # spec_condition[1] = head_var
                                if cond[0]=='COMPARATIVE':
                                    cond[2] = head_var
                                    if len(cond)<5:
                                        cond.append(cons_rel)
                                    else:
                                        cond[4] = "(JOIN " + cons_rel+" "+ cond[4]+")"
                                else: # SUPERLATIVE
                                    cond[2] = head_var
                                    cond[3] = "(JOIN "+ cons_rel+" "+cond[3]+")"
                                tgt_var = head_var
                        
                        # check whether need to repeat
                        remove_idx=-1
                        for i,(var,triplets) in enumerate(var_dep_list):
                            if var == head_var:
                                if len(triplets)==0:
                                    repeat = True
                                    remove_idx = i
                                else:
                                    repeat = False
                                break
                        
                        if remove_idx>=0:
                            var_dep_list.pop(remove_idx)
                        else:
                            repeat=False
            
                else:
                    # uncovered situation
                    assert 1 == 2
            else:
                """dep_triplets not None"""
                if not (len(dep_triplets) > 0):  # at least dependancy triplets
                    raise NotImplementedError(f"dep_triplets: {dep_triplets}")
                var_dep_list.append((tgt_var, dep_triplets))

        if(len(triplets_pool) != 0):
            print(triplets_pool)

        if not (len(triplets_pool) == 0):
            raise NotImplementedError(f"triplets_pool: {triplets_pool}")
        return var_dep_list

    @classmethod
    def resolve_dependancy(cls, triplets, filter_lines, target_var, successors):
        """resolve dependancy of variables
        @param triplets: all sparql triplet lines
        @param filter_lines: filter lines that start with `Filter (str(`
        @param target_var: target variable
        @param successors: successor variables of target variable

        @return: dependancy triplets of target_var, left triplets (independant of target_var)
        """
        dep = []
        left = []
        if not triplets:  # empty triplets, target_var constrained by filter

            # ns:m.0f9wd ns:influence.influence_node.influenced ?x .
            # ?x ns:government.politician.government_positions_held ?c .
            # ?c ns:government.government_position_held.from ?num .
            # ORDER BY ?num LIMIT 1
            pass
        else:
            for tri in triplets:
                if tri[0] == target_var:  # head is target variable
                    dep.append(tri)  # add to dependancy triplets
                    # tail is variable
                    if tri[-1].startswith('?') and tri[-1] not in successors:
                        successor_var = tri[-1]
                        if filter_lines:  # check filter variable `?sk0`
                            new_filter_lines = []
                            found_filter_variable = False
                            for line in filter_lines:
                                if successor_var in line: # LC-QuAD2 应该走不到这个分支里面
                                    found_filter_variable = True
                                    line = line.replace(
                                        'FILTER (str(', '').replace(')', '')
                                    tuple_list = line.split('=')
                                    var = tuple_list[0].strip()
                                    value = tuple_list[1].strip()

                                    assert successor_var == var
                                    if value.isalpha():
                                        tri[-1] = value+'@en'
                                    else:
                                        tri[-1] = value
                                    # tri[-1] = value+'@en'
                                else:
                                    new_filter_lines.append(line)

                            # remove corresponding filter_lines
                            if not found_filter_variable:  # no filter variable found
                                # add to successor variable
                                successors.append(successor_var)

                            filter_lines = new_filter_lines

                        else:
                            # add to successor variable
                            successors.append(successor_var)
                elif tri[-1] == target_var:  # tail is target variable
                    dep.append(tri)  # add to dependancy triplets
                    # head is variable
                    if tri[0].startswith('?') and tri[0] not in successors:
                        successors.append(tri[0])  # add to successor variable
                else:
                    left.append(tri)  # left triplets
        return dep, left

class SparqlParserLSQ:
    @classmethod
    def get_lines_from_sparql(cls, query):
        query = query.replace('filter', '\nfilter') # 便于按行进行划分
        for _match in re.findall(r"<http://rdf.freebase.com/ns/(.+?)>", query):
            query = query.replace(f"<http://rdf.freebase.com/ns/{_match}>", f"ns:{_match}")
        lines = query.split('\n')
        return lines
    

    @classmethod
    def parse_sparql(cls, query):
        """parse a sparql query into a s-expression

        @param query: sparql query
        """
        lines = cls.get_lines_from_sparql(query)
        lines = [x.strip() for x in lines if x] # 过滤空行

        line_num = 0

        next_line = lines[line_num]
        assert next_line.startswith('select distinct ?uri')
        line_num = line_num + 1
        next_line = lines[line_num]
        assert next_line == 'where'
        line_num = line_num + 1
        next_line = lines[line_num]
        assert next_line == '{'

        lines = lines[line_num:]

        # normalize body lines
        body_lines, spec_condition = cls.normalize_body_lines(
            lines)
        body_lines = [x.strip() for x in body_lines]  # strip spaces
        body_lines = [
            line for line in body_lines
            if (line.startswith('?') or line.startswith('ns:'))
        ]
        body_lines = [
            line[:-1] if line.endswith('.') else line
            for line in body_lines
        ] # LSQ 中，去除三元组末尾的 '.' 

        var_dep_list = cls.parse_naive_body(
            body_lines, '?uri', spec_condition)
        s_expr = cls.dep_graph_to_s_expr(var_dep_list, '?uri', spec_condition)
        return s_expr

    @classmethod
    def normalize_body_lines(cls, lines):
        """return normalized body lines of sparql, specially return filter lines starting with `FILTER (str(`        

        @param lines: sparql lines list
        @param filter_string_flag: flag indicates existence of filter lines


        @return: (body_lines,
                    spec_condition,
                    # [
                    #     ['SUPERLATIVE', argmax/argmin, arg_var, arg_r], 
                    #     ['COMPARATIVE', gt/lt/ge/le, compare_var, compare_value, compare_rel],
                    #     ['RANGE', range_relation, range_var, range_year],
                    # ]
                    filter_lines
                    )
        """

        spec_condition = []
        lines = [x.strip() for x in lines]
        body_lines = []
        
        '''具备对于比较级的处理'''
        # 2. get compare lines
        # 2.1 FILTER (?num > "2009-01-02"^^xsd:dateTime) .
        # 2.2 FILTER (xsd:integer(?num) < 33351310952) . 
        for _line in lines:
            if re.match(r'filter\(\?\w+ (>|<|>=|<=) .*', _line):
                compare_var = re.findall(r'\?\w+',_line)[0]
                compare_operator = re.findall(r'(>=|<=|>|<|=)', _line)[0]
                operator_mapper = {'<':'LT','<=':'LE','>':'GT',">=":"GE", '=':"EQ"} # 和新版保持一致
                compare_value = _line.replace(")","").split(" ")[-1] # 观察 LSQ 里面的 LITERAL, 都是 STRING 类型的
                # print(variable,compare_operator,compare_value)
                compare_condition = ['COMPARATIVE', operator_mapper[compare_operator],compare_var,compare_value]
                spec_condition.append(compare_condition)
            else:
                body_lines.append(_line) # 不是 FILTER line 的，都划分为 body line
    
        return body_lines, spec_condition
        
    @classmethod
    def dep_graph_to_s_expr(cls, var_dep_list, ret_var, spec_condition=None):
        """Convert dependancy graph to s_expression
        @param var_dep_list: varialbe dependancy list
        @param ret_var: return var
        @param spec_condition: special condition

        @return s_expression
        """
        parsed_dict = {}  # dict for parsed variables
        # 允许 var_dep_list 为空，即 SPARQL 中只有一个 LT / LE / GT / GE 的情况
        if len(var_dep_list) == 0:
            if len(spec_condition) != 1:
                raise Exception(f"spec_condition: {len(spec_condition)} {spec_condition}")
            for _spec in spec_condition:
                if _spec[2] != ret_var:
                    raise Exception(f"_spec: {_spec}")
                if _spec[0] == 'COMPARATIVE':
                    op = _spec[1]
                    value = _spec[3]
                    rel = _spec[4]
                    n_clause = '{} {} {}'.format(op, rel, value)
                else:
                    raise NotImplementedError(f"cond[0]: {cond[0]}; cond: {cond}")
                parsed_dict[_spec[2]] = n_clause
            
            res = '(' + parsed_dict[ret_var] + ')'
            return res
        
        else:
            if not (var_dep_list[0][0] == ret_var):
                raise NotImplementedError(f"var_dep_list[0][0]: {var_dep_list[0][0]}; ret_var: {ret_var}")
            var_dep_list.reverse() # reverse the var_dep_list

            # spec_condition,
            #             # [
            #             #     ['SUPERLATIVE', argmax/argmin, arg_var, arg_r], 
            #             #     ['COMPARATIVE', gt/lt/ge/le, compare_var, compare_value],
            #             #     ['RANGE', range_relation, range_var, range_year],
            #             # ]

            # specical condition var map {spec_var:idx in spec_condition}
            spec_var_map = {cond[2]:i for i,cond in enumerate(spec_condition)} if spec_condition else None
            # spec_var = spec_condition[1] if spec_condition is not None else None

            for var_name, dep_relations in var_dep_list:
                # expr = ''
                clause = cls.triplet_to_clause(
                    var_name,  dep_relations[0], parsed_dict)
                for tri in dep_relations[1:]:
                    n_clause = cls.triplet_to_clause(var_name, tri, parsed_dict)
                    clause = 'AND ({}) ({})'.format(n_clause, clause)
                # if var_name == spec_var:
                if spec_var_map and var_name in spec_var_map: # spec_condition
                    cond = spec_condition[spec_var_map[var_name]]
                    # if cond[0] == 'argmax' or cond[0] == 'argmin': # superlative
                    if cond[0] == 'COMPARATIVE':
                        op = cond[1]
                        value = cond[3]
                        rel = cond[4]
                        n_clause = '{} {} {}'.format(op, rel, value)
                        clause = 'AND ({}) ({})'.format(n_clause, clause)
                        # pass
                    else:
                        raise NotImplementedError(f"cond[0]: {cond[0]}; cond: {cond}")
                parsed_dict[var_name] = clause
            
            res = '(' + parsed_dict[ret_var] + ')'
            return res

    @classmethod
    def triplet_to_clause(cls, tgt_var, triplet, parsed_dict):
        """Convert a triplet to S_expression clause
        @param tgt_var: target variable
        @param triplet: triplet in sparql
        @param parsed_dict: dict for variables already parsed
        """
        if triplet[0] == tgt_var:
            this = triplet[0]
            other = triplet[-1]
            if other in parsed_dict:
                other = '(' + parsed_dict[other] + ')'
            return 'JOIN {} {}'.format(triplet[1], other)
        elif triplet[-1] == tgt_var:
            this = triplet[-1]
            other = triplet[0]
            if other in parsed_dict:
                other = '(' + parsed_dict[other] + ')'
            return 'JOIN (R {}) {}'.format(triplet[1], other)
        else:
            raise NotImplementedError(f"triplet: {triplet}; tgt_var: {tgt_var}")

    @classmethod
    def parse_naive_body(cls, body_lines, ret_var, spec_condition=None):
        """Parse body lines
        @param body_lines: list of sparql body lines
        @param ret_var: return var, default `?x`
        @param filter_lines: lines that start with `FILTER (str(?`

        @return: variable dependancy list
        """
        triplets = [x.split() for x in body_lines]  # 太不规整了，空格
        triplets = [x[:2] + [" ".join(x[2:])] if len(x)>4 else x for x in triplets] # avoid error splitting like "2100 Woodward Avenue"@en;
        '''这边我们假设前面已经处理好了，每一行都是三元组的形式'''
        triplets = [line for line in triplets if len(line) == 3]
        
        # remove ns
        triplets = [[x[3:] if x.startswith(
            'ns:') else x for x in tri] for tri in triplets]
        # dependancy graph
        triplets_pool = triplets
        # while True:
        # varaible dependancy list, in the form like [(?x,[['?x','ns:aaa.aaa.aaa','?y'],['ns:m.xx','ns:bbb.bbb.bbb','?x''])]
        var_dep_list = []
        successors = []

        # firstly solve the return variable
        dep_triplets, triplets_pool = cls.resolve_dependancy(
            triplets_pool, ret_var, successors)
        var_dep_list.append((ret_var, dep_triplets))

        # handle all the successor variables
        while len(successors):
            tgt_var = successors[0]
            successors = successors[1:]
            dep_triplets, triplets_pool = cls.resolve_dependancy(
                triplets_pool, tgt_var, successors)

            # assert len(dep_triplets) > 0 # at least one dependancy triplets
            if len(dep_triplets) == 0:
                # zero dep_triples, can be a 2-hop constraint
                # e.g.
                # 'ns:m.0d0x8 ns:government.political_district.representatives ?y .'
                # '?y ns:government.government_position_held.office_holder ?x .'
                # '?y ns:government.government_position_held.governmental_body ns:m.07t58 .'
                # '?x ns:government.politician.government_positions_held ?c .'
                
                if spec_condition and any([tgt_var in x for x in spec_condition]):
                    cond = []
                    for x in spec_condition:
                        if tgt_var in x:
                            cond = x
                            break
                    
                    repeat = True
                    while repeat:        
                        # tgt_var is a var in spec_condition
                        for (var, triplets) in var_dep_list:
                            if any([tgt_var in trip for trip in triplets]):
                                head_var = var  # find the real constrained var
                                _temp_triplets = triplets[:]
                                triplets.clear()
                                for trip in _temp_triplets:
                                    if tgt_var not in trip:
                                        triplets.append(trip)
                                    else:
                                        # find the constraint relation
                                        cons_rel = trip[1]
                                        if trip[0] == head_var:
                                            reversed_direction = False
                                        else:
                                            reversed_direction = True
                                        cons_rel = f'(R {cons_rel})' if reversed_direction else cons_rel

                                # modify spec_condition
                                # spec_condition[1] = head_var
                                if cond[0]=='COMPARATIVE':
                                    cond[2] = head_var
                                    if len(cond)<5:
                                        cond.append(cons_rel)
                                    else:
                                        cond[4] = "(JOIN " + cons_rel+" "+ cond[4]+")"
                                else: # SUPERLATIVE
                                    cond[2] = head_var
                                    cond[3] = "(JOIN "+ cons_rel+" "+cond[3]+")"
                                tgt_var = head_var
                        
                        # check whether need to repeat
                        remove_idx=-1
                        for i,(var,triplets) in enumerate(var_dep_list):
                            if var == head_var:
                                if len(triplets)==0:
                                    repeat = True
                                    remove_idx = i
                                else:
                                    repeat = False
                                break
                        
                        if remove_idx>=0:
                            var_dep_list.pop(remove_idx)
                        else:
                            repeat=False
            
                else:
                    # uncovered situation
                    assert 1 == 2
            else:
                """dep_triplets not None"""
                if not (len(dep_triplets) > 0):  # at least dependancy triplets
                    raise NotImplementedError(f"dep_triplets: {dep_triplets}")
                var_dep_list.append((tgt_var, dep_triplets))

        if(len(triplets_pool) != 0):
            print(triplets_pool)

        if not (len(triplets_pool) == 0):
            raise NotImplementedError(f"triplets_pool: {triplets_pool}")
        return var_dep_list

    @classmethod
    def resolve_dependancy(cls, triplets, target_var, successors):
        """resolve dependancy of variables
        @param triplets: all sparql triplet lines
        @param filter_lines: filter lines that start with `Filter (str(`
        @param target_var: target variable
        @param successors: successor variables of target variable

        @return: dependancy triplets of target_var, left triplets (independant of target_var)
        """
        dep = []
        left = []
        if not triplets:  # empty triplets, target_var constrained by filter

            # ns:m.0f9wd ns:influence.influence_node.influenced ?x .
            # ?x ns:government.politician.government_positions_held ?c .
            # ?c ns:government.government_position_held.from ?num .
            # ORDER BY ?num LIMIT 1
            pass
        else:
            for tri in triplets:
                if tri[0] == target_var:  # head is target variable
                    dep.append(tri)  # add to dependancy triplets
                    # tail is variable
                    if tri[-1].startswith('?') and tri[-1] not in successors:
                        successor_var = tri[-1]
                        successors.append(successor_var)

                elif tri[-1] == target_var:  # tail is target variable
                    dep.append(tri)  # add to dependancy triplets
                    # head is variable
                    if tri[0].startswith('?') and (tri[0] not in successors):
                        successors.append(tri[0])  # add to successor variable
                else:
                    left.append(tri)  # left triplets
        return dep, left
