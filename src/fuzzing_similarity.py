
from components.sparql_utils import (
    pre_process_sparql, SyntaxTreeEditor
)

class FuzzingSimilarity(object):
    def __init__(self):
        pass
    
    @classmethod
    def get_sparql_replace_item_with_variable(cls, sparql_query, topic_entity_rep, topic_entity_var_name):
        """
        @param topic_entity_rep: 主题实体在 sparql_query 中的表示（带上前缀等）
        @param topic_entity_var_name: 主题实体要被替换成的变量名
        """
        tree_editor = SyntaxTreeEditor(sparql_query)
        tree_editor.replace_leaf_with_variable(
            topic_entity_rep, topic_entity_var_name
        )
        return tree_editor.get_sparql_text()
    
    @classmethod
    def get_sparql_replace_item_with_item(cls, sparql_query, item_before, item_after):
        """
        @param topic_entity_rep: 主题实体在 sparql_query 中的表示（带上前缀等）
        @param topic_entity_var_name: 主题实体要被替换成的变量名
        """
        tree_editor = SyntaxTreeEditor(sparql_query)
        tree_editor.update_leaf_value(
            item_before, item_after
        )
        return tree_editor.get_sparql_text()
        
        
