import copy
import random

import dgl
import numpy as np
import torch

import data.Constants
from script.c_cxx import utils_cfg
from utils.util import detectCode


def _pad_cfg(opt, cfg_anno, new_n_node, new_annotation_dim):
    padding = np.zeros((len(cfg_anno), new_n_node, opt.state_dim - new_annotation_dim))
    cfg_init_input = np.concatenate((cfg_anno, padding), 2)
    return cfg_init_input, cfg_anno


def _get_cfg_npy_normal(opt, cfg_json_list, n_node, n_edge_types, annotation_dim):
    len_index_list = len(cfg_json_list)
    node_mask = np.zeros([len_index_list, n_node])

    all_adjmat = np.zeros([len_index_list, n_node, n_node * n_edge_types * 2])
    all_anno = np.zeros([len_index_list, n_node, annotation_dim])
    for cnt in range(len(cfg_json_list)):
        json_dict = copy.deepcopy(cfg_json_list[cnt])
        save_node_feature_node_name_int = json_dict["save_node_feature_digit"]
        save_edge_node_name_int = json_dict["save_edge_digit"]

        exist_node_id_list_str = list(save_node_feature_node_name_int.keys())
        exist_node_id_list_int = [int(k) for k in exist_node_id_list_str]
        exist_node_id_list_int.sort()
        node_mask_tmp1 = copy.deepcopy(exist_node_id_list_int)
        node_mask_tmp2 = [1 if k in node_mask_tmp1 else 0 for k in range(n_node)]

        if sum(node_mask_tmp2) == 0:
            assert False, print("assert False, json_dict:\n", json_dict)

        node_mask_tmp2_array = np.array(node_mask_tmp2)
        node_mask[cnt, :] = node_mask_tmp2_array

        adjmat = utils_cfg.create_adjacency_matrix(save_edge_node_name_int, n_node,
                                                   n_edge_types)
        anno = utils_cfg.create_annotation_mat(n_node, annotation_dim,
                                               save_node_feature_node_name_int)

        all_adjmat[cnt, :, :] = adjmat
        all_anno[cnt, :, :] = anno

    return all_adjmat, all_anno, node_mask


def _pad_seq(seq_data, include_lengths=False):
    lengths = [len(x) for x in seq_data]
    max_length = max(lengths)
    batch = np.zeros((len(seq_data), max_length))
    for i, dat in enumerate(seq_data):
        batch[i, :len(dat)] = dat
    if include_lengths:
        return batch, lengths
    else:
        return batch


def _get_max_len(seq_data):
    lengths = [len(x) for x in seq_data]
    max_len = max(lengths)
    return max_len


def _pad_given_max_len(seq_data, max_len):
    batch = np.zeros((len(seq_data), max_len))
    for i, dat in enumerate(seq_data):
        batch[i, :len(dat)] = dat

    return batch


def _pad_comment(comment_data):
    lengths = [len(x) for x in comment_data]
    max_length = max(lengths)

    comment = np.zeros((len(comment_data), max_length + 1))
    comment_target = np.zeros((len(comment_data), max_length + 1))
    comment[:, 0] = data.Constants.BOS
    for i, dat in enumerate(comment_data):
        comment[i, 1:len(dat) + 1] = dat
        comment_target[i, :len(dat) + 1] = dat + [data.Constants.EOS]
    return comment, comment_target, lengths


def _pad_comment_no_bos(comment_data):
    lengths = [len(x) for x in comment_data]
    max_length = max(lengths)
    comment = np.zeros((len(comment_data), max_length))
    comment_target = np.zeros((len(comment_data), max_length + 1))

    for i, dat in enumerate(comment_data):
        comment[i, 0:len(dat)] = dat
        comment_target[i, :len(dat) + 1] = dat + [data.Constants.EOS]
    return comment, comment_target, lengths


def _get_root_node_info(batch_list):
    list_root_index, list_num_node = [], []

    for tree_dgldigraph in batch_list:
        topological_nodes_list = dgl.topological_nodes_generator(tree_dgldigraph)
        root_id_tree_dgldigraph = topological_nodes_list[-1].item()
        list_root_index.append(root_id_tree_dgldigraph)
        all_num_node_tree_dgldigraph = tree_dgldigraph.number_of_nodes()
        list_num_node.append(all_num_node_tree_dgldigraph)

    root_index_np = np.array(list_root_index)
    num_node_np = np.array(list_num_node)

    return root_index_np, num_node_np


def _get_tree_dgl_graph_batch(batch_list):
    root_index_np, num_node_np = _get_root_node_info(copy.deepcopy(batch_list))
    batch_trees = dgl.batch(batch_list)

    return batch_trees, root_index_np, num_node_np


class CodeRetrievalDataset(object):
    def __init__(self, opt, data_ctg, dict_code, dict_comment, n_node, n_edge_types, annotation_dim, not_test_set=True):
        print("CodeRetrievalDataset __init__ ......")
        self.opt = opt
        self.not_test_set = not_test_set

        self.key_list = data_ctg["key_list"]

        if not self.not_test_set or "seq" in self.opt.modal_type:
            self.code = data_ctg["code"]
            assert len(self.code) == len(self.key_list)

        if not self.not_test_set or "tree" in self.opt.modal_type:
            self.tree_dgl = data_ctg["tree_dgl"]
            assert len(self.tree_dgl) == len(self.key_list)

        if not self.not_test_set or "cfg" in self.opt.modal_type:
            self.cfg = data_ctg['cfg']
            assert len(self.cfg) == len(self.key_list)

        if self.not_test_set:
            self.comment = data_ctg["comment"]
            assert len(self.comment) == len(self.key_list)

        self.n_node = n_node
        self.n_edge_types = n_edge_types
        self.annotation_dim = annotation_dim

        if opt.train_mode == "train":
            self.func_content = torch.load(self.opt.data_pairs_func_content)
        elif opt.train_mode == "test":
            if opt.use_val_as_codebase:
                self.func_content = torch.load(self.opt.data_pairs_func_content)
            else:
                self.func_content = torch.load(self.opt.data_codebase_func_content)

        self.new_n_node = self.n_node
        self.new_n_edge_types = self.n_edge_types

        self.new_annotation_dim = self.annotation_dim

        self.dict_code = dict_code
        self.dict_comment = dict_comment
        if self.not_test_set:
            assert (len(self.key_list) == len(self.comment))

        self._label2idx()

        print("CodeRetrievalDataset_len: ", len(self.key_list))

    def _label2idx(self):
        if not self.not_test_set or "seq" in self.opt.modal_type:
            for i, cod in enumerate(self.code):
                self.code[i] = self.dict_code.convertToIdx(cod, data.Constants.UNK_WORD)

        if self.not_test_set:
            for i, comm in enumerate(self.comment):
                self.comment[i] = self.dict_comment.convertToIdx(comm, data.Constants.UNK_WORD)

        if not self.not_test_set or "cfg" in self.opt.modal_type:
            for i, cfg_dict in enumerate(self.cfg):
                cfg_dict_new = copy.deepcopy(cfg_dict)
                self.cfg[i] = cfg_dict_new

    def __getitem__(self, index):

        if not self.not_test_set:

            code, tree_dgl, cfg = self.code[index], self.tree_dgl[index], self.cfg[index]

            cfg_adjmat, cfg_anno, cfg_node_mask \
                = _get_cfg_npy_normal(self.opt, [cfg], self.n_node, self.n_edge_types, self.annotation_dim)

            cfg_init_input_batch, cfg_anno_batch = _pad_cfg(self.opt, cfg_anno, self.new_n_node,
                                                            self.new_annotation_dim)
            return code, tree_dgl, cfg_init_input_batch, cfg_anno_batch, cfg_adjmat, \
                   cfg_node_mask, index
        else:
            comment = self.comment[index]
            rand_index = random.randint(0, self.__len__() - 1)
            while rand_index == index:
                rand_index = random.randint(0, self.__len__() - 1)
            bad_comment = self.comment[rand_index]

            if "seq" in self.opt.modal_type:
                code = self.code[index]
            if "tree" in self.opt.modal_type:
                tree_dgl = self.tree_dgl[index]
            if "cfg" in self.opt.modal_type:
                cfg = self.cfg[index]

                cfg_adjmat, cfg_anno, cfg_node_mask = _get_cfg_npy_normal(
                    self.opt, [cfg], self.n_node, self.n_edge_types, self.annotation_dim)

                cfg_init_input_batch, cfg_anno_batch = _pad_cfg(self.opt, cfg_anno, self.new_n_node,
                                                                self.new_annotation_dim)

            if "seq" in self.opt.modal_type and "tree" in self.opt.modal_type and "cfg" in self.opt.modal_type:
                if self.not_test_set:
                    return code, tree_dgl, cfg_init_input_batch, cfg_anno_batch, cfg_adjmat, \
                           cfg_node_mask, comment, bad_comment, "seq_tree_cfg"

    def __len__(self):

        return len(self.key_list)


class CodeRetrievalQueryDataset(object):
    def __init__(self, opt, dict_comment):
        print("CodeRetrievalQueryDataset __init__ ......")
        self.opt = opt
        self.dict_comment = dict_comment

        self.raw_query = []
        self.comment = []
        encoding_mode = detectCode(self.opt.data_query_file)
        with open(self.opt.data_query_file, 'r', encoding=encoding_mode, errors="ignore") as this_query_file:

            query_lines = copy.deepcopy(this_query_file.readlines())
        for _i in range(len(query_lines)):

            query_list_ori = copy.deepcopy(query_lines[_i]).split()

            if self.opt.remove_wh_word:
                query_list = copy.deepcopy(query_lines[_i]). \
                    replace("How to", "").replace("How do I", ""). \
                    replace("How do you", "").replace("How do we", ""). \
                    replace("How can I", "").replace("How can we", ""). \
                    replace("What is", "").replace("How are", ""). \
                    replace("How is", "").replace("What are", ""). \
                    replace("Can I", "").replace("Can you", ""). \
                    replace("Can we", ""). \
                    replace("how to", "").replace("how do I", ""). \
                    replace("how do you", "").replace("how do we", ""). \
                    replace("how can I", "").replace("how can we", ""). \
                    replace("what is", "").replace("how are", ""). \
                    replace("how is", "").replace("what are", ""). \
                    replace("can I", "").replace("can you", ""). \
                    replace("can we", "").split()
            else:
                query_list = copy.deepcopy(query_lines[_i]).split()
            if opt.lower:
                query_to_append = [q.lower() for q in query_list]
            else:
                query_to_append = query_list

            print("query_list_ori: \n", query_list_ori)
            print("query_to_append: \n", query_to_append)
            self.raw_query.append(copy.deepcopy(query_list_ori))
            self.comment.append(copy.deepcopy(query_to_append))

        self._label2idx()
        print("CodeRetrievalQueryDataset_len: ", len(self.comment))
        print("--------after self._label2idx(), self.comment: \n")
        for i in range(len(self.comment)):
            print("=={}\n{}\n".format(i, self.comment[i]))

    def _label2idx(self):
        for i, comm in enumerate(self.comment):
            self.comment[i] = self.dict_comment.convertToIdx(comm, data.Constants.UNK_WORD)

    def __getitem__(self, index):
        comment = self.comment[index]
        return comment, index

    def __len__(self):
        return len(self.comment)


def collate_fn_code_retrieval_for_query(batch_data):
    comment_batch, index_batch = zip(*batch_data)

    comment_batch, comment_target_batch, comment_length = _pad_comment_no_bos(comment_batch)

    comment_batch, comment_target_batch = \
        torch.from_numpy(comment_batch).long(), torch.from_numpy(comment_target_batch).long()

    comment_length = torch.from_numpy(np.array(comment_length)).long()

    return comment_batch, comment_target_batch, comment_length, index_batch


def collate_fn_code_retrieval(batch_data):
    opt_data_type = batch_data[0][-1]

    if opt_data_type == "seq_tree_cfg":
        batch_data.sort(key=lambda x: len(x[0]), reverse=True)
        code_batch, tree_dgl_batch, cfg_init_input_batch, cfg_anno_batch, cfg_adjmat_batch, cfg_node_mask, \
        comment_batch, bad_comment_batch, _ = zip(*batch_data)

    if "seq" in opt_data_type:
        code_batch, code_length = _pad_seq(code_batch, include_lengths=True)
        code_batch = torch.from_numpy(code_batch).long()
    else:
        code_batch, code_length = None, None

    if "tree" in opt_data_type:
        tree_dgl_batch, tree_dgl_root_index, tree_dgl_node_num = _get_tree_dgl_graph_batch(tree_dgl_batch)
        tree_dgl_root_index, tree_dgl_node_num = torch.from_numpy(tree_dgl_root_index).long(), torch.from_numpy(
            tree_dgl_node_num).long()
    else:
        tree_dgl_batch, tree_dgl_root_index, tree_dgl_node_num = None, None, None

    if "cfg" in opt_data_type:

        cfg_init_input_batch = np.concatenate(cfg_init_input_batch)
        cfg_anno_batch = np.concatenate(cfg_anno_batch)
        cfg_adjmat_batch = np.concatenate(cfg_adjmat_batch)
        cfg_node_mask = np.concatenate(cfg_node_mask)

        cfg_init_input_batch = torch.from_numpy(cfg_init_input_batch).float()
        cfg_anno_batch = torch.from_numpy(cfg_anno_batch).float()
        cfg_adjmat_batch = torch.from_numpy(cfg_adjmat_batch).float()
        cfg_node_mask = torch.from_numpy(cfg_node_mask).type(torch.ByteTensor)


    else:
        cfg_init_input_batch, cfg_anno_batch, cfg_adjmat_batch, cfg_node_mask, \
            = None, None, None, None

    comment_batch, comment_target_batch, comment_length = _pad_comment_no_bos(comment_batch)
    bad_comment_batch, bad_comment_target_batch, bad_comment_length = _pad_comment_no_bos(bad_comment_batch)

    bad_comment_batch = torch.from_numpy(bad_comment_batch).long()
    bad_comment_target_batch = torch.from_numpy(bad_comment_target_batch).long()
    comment_length = torch.from_numpy(np.array(comment_length)).long()
    bad_comment_length = torch.from_numpy(np.array(bad_comment_length)).long()

    comment_batch, comment_target_batch = torch.from_numpy(comment_batch).long(), torch.from_numpy(
        comment_target_batch).long()

    return code_batch, code_length, tree_dgl_batch, tree_dgl_root_index, tree_dgl_node_num, \
           cfg_init_input_batch, cfg_anno_batch, cfg_adjmat_batch, cfg_node_mask, \
           comment_batch, comment_target_batch, comment_length, \
           bad_comment_batch, bad_comment_target_batch, bad_comment_length


def collate_fn_code_retrieval_for_test_set(batch_data):
    code_seq_len_list_array = np.array([len(x[0]) for x in batch_data])
    index_sort_code_seq_len_list_array = np.argsort(-code_seq_len_list_array)
    batch_data.sort(key=lambda x: len(x[0]), reverse=True)

    code_batch, tree_dgl_batch, cfg_init_input_batch, cfg_anno_batch, cfg_adjmat_batch, cfg_node_mask, \
    index_batch = zip(*batch_data)

    code_batch, code_length = _pad_seq(code_batch, include_lengths=True)

    tree_dgl_batch, tree_dgl_root_index, tree_dgl_node_num = _get_tree_dgl_graph_batch(tree_dgl_batch)

    cfg_init_input_batch = np.concatenate(cfg_init_input_batch)
    cfg_anno_batch = np.concatenate(cfg_anno_batch)
    cfg_adjmat_batch = np.concatenate(cfg_adjmat_batch)
    cfg_node_mask = np.concatenate(cfg_node_mask)

    cfg_init_input_batch = torch.from_numpy(cfg_init_input_batch).float()
    cfg_anno_batch = torch.from_numpy(cfg_anno_batch).float()
    cfg_adjmat_batch = torch.from_numpy(cfg_adjmat_batch).float()
    code_batch, tree_dgl_root_index, tree_dgl_node_num, cfg_node_mask = \
        torch.from_numpy(code_batch).long(), torch.from_numpy(tree_dgl_root_index).long(), torch.from_numpy(
            tree_dgl_node_num).long(), \
        torch.from_numpy(cfg_node_mask).type(torch.ByteTensor)

    return code_batch, code_length, tree_dgl_batch, tree_dgl_root_index, tree_dgl_node_num, \
           cfg_init_input_batch, cfg_anno_batch, cfg_adjmat_batch, cfg_node_mask, \
           index_sort_code_seq_len_list_array, index_batch
