from typing import Optional, List
from transformers import AutoTokenizer
import re
import traceback


class TreeNode():
    def __init__(self,content: list, idx: int):
        self.open_tag: List[str] = content 
        self.end_tag: Optional[List[str]] = None 
        self.self_closing_tag: Optional[List[str]] = None
        self.text = ""
        
        self.name: Optional[str] = None
        self.parent: Optional['TreeNode'] = None  # Use 'TreeNode' as a string for forward reference
        
        self.open_tag_range: Optional[List[int]] = None
        self.end_tag_range: Optional[List[int]] = None
        self.text_range = [-1,-1]
        self.self_closing_tag_range = [-1,-1]
        
        self.idx: int = idx
        self.children: List['TreeNode'] = []  # List of TreeNode instances
    
    
    def partially_open(self):
        if not self.open_tag: return False
        if any('<' in s for s in self.open_tag) and not any('>' in s for s in self.open_tag):
            return True
        return False
    
    def add_child(self,child):
        assert child.parent is None, "Child already has a parent"
        assert child not in self.children, "Child is already in children list"
        child.parent = self
        self.children.append(child)
    
    def get_range(self):
        if self.text:
            return list(range(*self.text_range))
        elif self.self_closing_tag:
            return list(range(*self.self_closing_tag_range))
        else:
            attn_range = []
            if self.open_tag_range:
                attn_range += list(range(*self.open_tag_range))
            if self.end_tag_range:
                attn_range += list(range(*self.end_tag_range))
            return attn_range
    
    def __repr__(self):
        return f"Node(name='{self.open_tag}', idx = {self.idx})"
    
    def print_tree(self, level=0, input_ids = None, tokenizer = None):
        if level == 0:
            print("--------")
        indent = "  " * level
        if self.text:
            print(f"{indent}{tokenizer.convert_tokens_to_string(self.text).strip()}, level = {level} ")
        elif self.self_closing_tag:
            print(f"{indent}{tokenizer.convert_tokens_to_string(self.self_closing_tag).strip()}, level = {level} ")
        elif self.open_tag:
            print(f"{indent}{tokenizer.convert_tokens_to_string(self.open_tag).strip()}, level = {level} ")
            for child in self.children:
                child.print_tree(level + 1, input_ids, tokenizer)
            if self.end_tag:
                print(f"{indent}{tokenizer.convert_tokens_to_string(self.end_tag).strip()}, level = {level} ")
        else:
            for child in self.children:
                child.print_tree(level + 1, input_ids, tokenizer)
        if level == 0:
            print("--------")

    def get_tree(self, level=0, input_ids = None, tokenizer=None):
        tree_str = ""

        indent = "  " * level
        if self.text:
            tree_str+=f"{indent}{tokenizer.convert_tokens_to_string(self.text).strip()} \n"
        elif self.self_closing_tag:
            tree_str+=f"{indent}{tokenizer.convert_tokens_to_string(self.self_closing_tag).strip()} \n"
        elif self.open_tag:
            tree_str+=f"{indent}{tokenizer.convert_tokens_to_string(self.open_tag).strip()} \n"
            for child in self.children:
                tree_str+=child.get_tree(level + 1, input_ids, tokenizer)
            if self.end_tag:
                tree_str+=f"{indent}{tokenizer.convert_tokens_to_string(self.end_tag).strip()} \n"
        else:
            for child in self.children:
                tree_str+=child.get_tree(level + 1, input_ids, tokenizer)

        return tree_str


class TreeBuilder():
    def __init__(self, tokenizer: AutoTokenizer = None, root: TreeNode = None, cur_node: TreeNode = None):
        self.tokenizer = tokenizer
        self.root = TreeNode(None, 0)
        self.cur_node = self.root
        self.buffer = []
        self.buffer_start_index = 0
        self.idx = 0
        self.full_attention_list= None
        self.web_attention_mask = None
        self.input_ids = None
        self.void_elements = [
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr"
        ]

    def is_empty(self):
        return self.root == None

    def in_buffer(self, text):
        if len(self.buffer) == 0:
            return False
        return any(text in s for s in self.buffer)
    
    def find_buffer(self, text):
        # Iterate over the list of strings with their indices
        for index, s in enumerate(self.buffer):
            if text in s:
                return index
        return -1
    
    # Function to extract xxx from <xxx> or <xxx yyy>
    def extract_open_tag_name(self,buffer):
        input_string = self.tokenizer.convert_tokens_to_string(buffer)
        match = re.search(r'<\s*(\w+)(?:\s+[^>]*)?>', input_string)
        if match:
            return match.group(1)
        return None

    def extract_close_tag_name(self,buffer):
        # if isinstance(input_string, list):
        #     input_string = "".join(input_string).replace('Ċ', '\n').replace('Ġ', ' ').replace('ĉ', '\t')
        input_string = self.tokenizer.convert_tokens_to_string(buffer)
        match = re.search(r'</\s*(\w+)(?:\s+[^>]*)?>', input_string)
        if match:
            return match.group(1)
        return None
    
    def is_not_empty_buffer(self):
        return self.tokenizer.convert_tokens_to_string(self.buffer).strip() != ''
    
    def get_parent_and_siblings_attention_range(self):
        attn_range = []
        if self.cur_node.parent:
            parent = self.cur_node.parent
            if parent.open_tag_range:
                attn_range += list(range(*parent.open_tag_range)) 
            for child in parent.children:
                if child is not self.cur_node:
                    if child.open_tag and child.end_tag:
                        attn_range += list(range(*child.open_tag_range)) 
                        attn_range += list(range(*child.end_tag_range)) 
                    elif child.text:
                        attn_range += list(range(*child.text_range))
                    elif child.self_closing_tag:
                        attn_range += list(range(*child.self_closing_tag_range))
                    else:
                        raise Exception(f"??? line 151, get p and s attention range")
                        
        return attn_range
    
    def update_buffer(self, cur_decoded_token):
        # open tag situations
        assert isinstance(cur_decoded_token,list), f"{cur_decoded_token}"
        self.buffer+=cur_decoded_token
        assert isinstance(cur_decoded_token[0],str)
        # print(self.buffer)
        try:
            # dealing with end tag
            if self.in_buffer('</' ) and self.in_buffer('>') and self.find_buffer('</') <= self.find_buffer('>'):                    
                close_tag_name = self.extract_close_tag_name(self.buffer)
                
                if self.cur_node.open_tag and not self.cur_node.end_tag:
                    assert close_tag_name == self.extract_open_tag_name(self.cur_node.open_tag), f"close_tag_name is {close_tag_name}, with buffer: {self.buffer}, open is-----{self.cur_node.open_tag}---"
                elif self.cur_node.text or self.cur_node.self_closing_tag or self.cur_node.end_tag:
                    content = None
                    if self.cur_node.text: content = self.cur_node.text
                    elif self.cur_node.self_closing_tag: content = self.cur_node.self_closing_tag
                    elif self.cur_node.end_tag: content = self.cur_node.end_tag
                    self.root.print_tree(0,None,self.tokenizer)
                    raise Exception(f"This should never happen\n {content}, buffer is {self.buffer}")
                    
                    # assert close_tag_name == extract_open_tag_name(self.cur_node.open_tag), f"close_tag_name is {close_tag_name}, with buffer: {self.buffer}, open is-----{self.cur_node.open_tag}---"
                else:
                    raise Exception(f"having end tag without having an open tag\n {self.cur_node.text}")
                
                self.cur_node.end_tag = self.buffer[:self.find_buffer('>')+1]
                self.cur_node.end_tag_range = [self.buffer_start_index, self.buffer_start_index + self.find_buffer('>')+1]
                self.buffer_start_index += self.find_buffer('>')+1
                self.buffer = self.buffer[self.find_buffer('>')+1:]
            # dealing with open tag
            elif self.in_buffer('</'):
                if self.cur_node.open_tag and not self.cur_node.end_tag:
                    pass
                elif self.cur_node.text or self.cur_node.self_closing_tag or (self.cur_node.open_tag and self.cur_node.end_tag):
                    cur_end_tag_index = self.find_buffer('</')
                    # import pdb;pdb.set_trace()
                    if self.cur_node.text:
                        self.cur_node.text += self.buffer[:cur_end_tag_index]
                        self.cur_node.text_range[1] += len(self.buffer[:cur_end_tag_index])
                    elif self.cur_node.self_closing_tag:
                        self.cur_node.self_closing_tag += self.buffer[:cur_end_tag_index]
                        self.cur_node.self_closing_tag_range[1] += len(self.buffer[:cur_end_tag_index])
                    else:
                        self.cur_node.end_tag += self.buffer[:cur_end_tag_index]
                        self.cur_node.end_tag_range[1] += len(self.buffer[:cur_end_tag_index])
                    self.buffer_start_index += len(self.buffer[:cur_end_tag_index])
                    self.buffer =self.buffer[cur_end_tag_index:]
                    self.cur_node = self.cur_node.parent
                else:
                    raise Exception(f"having end tag without having an open tag\n {self.cur_node.text} {self.cur_node} {self.cur_node.parent.open_tag}")
                
            elif self.in_buffer('<') and self.in_buffer('>'):
                # in the case of self_closing tag
                if self.in_buffer('/>'):
                    self.cur_node.open_tag = None
                    self.cur_node.self_closing_tag = self.buffer[:self.find_buffer(">")+1]
                    self.cur_node.self_closing_tag_range = [self.buffer_start_index, self.buffer_start_index + self.find_buffer('>')+1]
                else:
                    open_tag_name = self.extract_open_tag_name(self.buffer)
                    if open_tag_name in self.void_elements:
                        self.cur_node.open_tag = None
                        self.cur_node.self_closing_tag = self.buffer[:self.find_buffer(">")+1]
                        self.cur_node.self_closing_tag_range = [self.buffer_start_index, self.buffer_start_index + self.find_buffer('>')+1]
                    else:
                        self.cur_node.open_tag = self.buffer[:self.find_buffer(">")+1]
                        self.cur_node.open_tag_range = [self.buffer_start_index, self.buffer_start_index + self.find_buffer('>')+1]
                
                self.buffer_start_index += self.find_buffer('>')+1
                self.buffer = self.buffer[self.find_buffer(">")+1:]
            elif self.in_buffer('<'):
                if self.full_attention_list is None:
                    self.full_attention_list = self.buffer[:-1]
                    self.buffer = self.buffer[-1:]
                    self.buffer_start_index = len(self.full_attention_list)
                else:
                    cur_open_tag_index = self.find_buffer('<')
                    # full open tag, indicating a pair of open and close tags, or a single open tag
                    if not self.cur_node.partially_open() and self.cur_node.open_tag:
                        if self.cur_node.end_tag:
                            self.cur_node.end_tag += self.buffer[:cur_open_tag_index]
                            self.cur_node.end_tag_range[1] += len(self.buffer[:cur_open_tag_index])
                            self.buffer_start_index += len(self.buffer[:cur_open_tag_index])
                            self.buffer =self.buffer[cur_open_tag_index:]
                            child_node = TreeNode(self.buffer, self.idx)
                            if self.cur_node.parent:
                                self.cur_node.parent.add_child(child_node)
                            else:
                                raise Exception(f"This should never happen, a html element with full open tag should have a parent, {self.cur_node.open_tag}")
                            self.idx += 1
                            self.cur_node = child_node
                        else:
                            child_node = TreeNode(self.buffer, self.idx)
                            self.cur_node.add_child(child_node)
                            self.idx += 1
                            self.cur_node = child_node
                    elif self.cur_node.text or self.cur_node.self_closing_tag:
                        if self.cur_node.text:
                            self.cur_node.text += self.buffer[:cur_open_tag_index]
                            self.cur_node.text_range[1] += len(self.buffer[:cur_open_tag_index])
                        elif self.cur_node.self_closing_tag:
                            self.cur_node.self_closing_tag += self.buffer[:cur_open_tag_index]
                            self.cur_node.self_closing_tag_range[1] += len(self.buffer[:cur_open_tag_index])
                        
                        self.buffer_start_index += len(self.buffer[:cur_open_tag_index])
                        self.buffer =self.buffer[cur_open_tag_index:]
                        child_node = TreeNode(self.buffer, self.idx)
                        self.cur_node.parent.add_child(child_node)
                        self.idx += 1
                        self.cur_node = child_node
            # if the current node has an open tag, and we are encountering texts, we create a new text node, and move down a level
            elif (self.cur_node.open_tag or self.cur_node.self_closing_tag) and not self.in_buffer('<') and self.is_not_empty_buffer():
                child_node = TreeNode(None, self.idx)
                child_node.text = self.buffer
                child_node.text_range[0] = self.buffer_start_index
                child_node.text_range[1] = self.buffer_start_index + len(self.buffer)
                
                if self.cur_node.end_tag or self.cur_node.self_closing_tag:
                    self.cur_node.parent.add_child(child_node)
                else:
                    self.cur_node.add_child(child_node)
                
                self.idx += 1
                self.cur_node = child_node
                self.buffer_start_index += len(self.buffer)
                self.buffer = []
            # if the current node does not have an open tag, but we are encountering text, we add to the exisitng text node
            elif self.cur_node.text and not self.in_buffer('<') and self.is_not_empty_buffer():
                self.cur_node.text += self.buffer
                assert self.cur_node.text_range[0] != -1 and self.cur_node.text_range[1] != -1, f"self.cur_node.text_range[0] and [1] should not be -1 but: {self.cur_node.text_range[0]}, {self.cur_node.text_range[1]}"
                self.cur_node.text_range[1] += len(self.buffer)
                self.buffer_start_index += len(self.buffer)
                self.buffer =[]
                
        except Exception as e:
            traceback.format_exc()
            raise Exception(e)
        
        if self.full_attention_list is None:
            attn_range = list(range(len(self.buffer)))
        else:
            attn_range = list(range(len(self.full_attention_list))) + self.get_parent_and_siblings_attention_range() + self.cur_node.get_range() + [i + self.buffer_start_index  for i in list(range(len(self.buffer)))]
        return attn_range
