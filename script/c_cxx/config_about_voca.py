import copy

PAD_token_index = 0
UNK_token_index = 1
real_voca_start_index = 2

c_cxx_unk_max_ratio = 0.5

c_cxx_token_len = 100
c_cxx_comment_len = 50

flag_remove_keyword_for_token_for_deep_code_search = True
c_ast_tree_leaves_voca_size_real_set = 50000
code_voca_size_real_set = 50000
comment_voca_size_real_set = 50000
max_num_node_cfg = 512

ast_unk_index = 0
ast_real_token_start_index = 1
ast_voca_size = 10000

NODE_FIX = '1*NODEFIX'
idx_funcdecl = 2
idx_param = 4
idx_compound = 4

prefix = "same_as_node_name_"

cfg_node_token_line_index_identifier = ["(in line: ", "(ln: ", "\\nln: "]
cfg_node_token_source_file_path_identifier = [" file: ", " fl: "]

cfg_node_prefix = "Node0x"
cfg_color_prefix = "color="
cfg_func_name_prefix = "Fun["

cfg_node_color2index = {"blue": 0, "black": 1, "green": 2, "yellow": 3, "red": 4}
cfg_edge_color2index = {"black": 0, "red": 1, "blue": 2}
cfg_node_feature_dim = len(cfg_node_color2index.keys())
cfg_num_edge_type = len(cfg_edge_color2index.keys())

cfg_default_edge_type = "black"
cfg_default_node_type = "black"



value_list_int_cfg_edge = []
for k, v in cfg_edge_color2index.items():
    value_list_int_cfg_edge.append(v)

print("cfg_node_color2index: ", cfg_node_color2index)
print("cfg_edge_color2index: ", cfg_edge_color2index)




print("config_about_voca.py loaded.....")
