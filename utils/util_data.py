import copy
import os

import dgl
import networkx as nx
import torch

import data.Constants
import data.Constants
from data.Dataset import CodeRetrievalDataset, collate_fn_code_retrieval, collate_fn_code_retrieval_for_test_set, \
    CodeRetrievalQueryDataset, collate_fn_code_retrieval_for_query
from data.Dict import Dict
from model.CodeRetrievalModel import ModelCodeRetrieval
from model.Encoder import Encoder, TreeEncoder_TreeLSTM_dgl, CFGEncoder_GGNN, \
    RetrievalCodeEncoderWrapper, RetrievalCommentEncoderWrapper
from script.c_cxx import config_about_voca


def get_max_node_num_for_one_file(cfgs):
    max_node_num = 0
    for json_dict in cfgs:
        save_node_feature_node_name_int = json_dict["save_node_feature_digit"]
        this_node_num = len(list(save_node_feature_node_name_int.keys()))
        if this_node_num > max_node_num:
            max_node_num = this_node_num
    return max_node_num


def get_max_node_num(train_cfgs, val_cfgs, test_cfgs):
    n_node = 0
    for cfgs in [train_cfgs, val_cfgs, test_cfgs]:
        this_n_node = get_max_node_num_for_one_file(cfgs)
        if this_n_node > n_node:
            n_node = this_n_node

    return n_node


def build_tree(tree_json, dict_code):
    tmp_nodename_list = []
    nodename_int_list = []
    for nodename in tree_json.keys():
        if nodename not in tmp_nodename_list:
            tmp_nodename_list.append(nodename)
            nodename_int_list.append(int(nodename[len(data.Constants.NODE_FIX):]))

    assert len(tree_json) == len(tmp_nodename_list)

    g = nx.DiGraph()

    def _rec_build(nid, idx, t_json):

        children = [c for c in t_json[idx]['children'] if c.startswith(data.Constants.NODE_FIX)]

        if len(children) == 2:

            if nid is None:
                g.add_node(0, x=data.Constants.DGLGraph_PAD_WORD, y=int(idx[len(data.Constants.NODE_FIX):]), mask=0)
                nid = 0

            for c in children:
                cid = g.number_of_nodes()

                y_value = int(c[len(data.Constants.NODE_FIX):])

                c_children = t_json[c]["children"]
                c_children_list = [c_tmp for c_tmp in c_children if c_tmp.startswith(data.Constants.NODE_FIX)]

                if len(c_children_list) == 2:
                    g.add_node(cid, x=data.Constants.DGLGraph_PAD_WORD, y=y_value, mask=0)

                    _rec_build(cid, c, t_json)
                else:
                    assert len(t_json[c]['children']) == 1
                    word_index = dict_code.lookup(t_json[c]['children'][0], data.Constants.UNK)
                    g.add_node(cid, x=word_index, y=y_value, mask=1)

                g.add_edge(cid, nid)
        else:
            assert len(t_json[idx]['children']) == 1
            word_index = dict_code.lookup(t_json[idx]['children'][0], data.Constants.UNK)
            if nid is None:
                cid = 0
            else:
                cid = g.number_of_nodes()
            y_value = int(idx[len(data.Constants.NODE_FIX):])
            g.add_node(cid, x=word_index, y=y_value, mask=1)

            if nid is not None:
                g.add_edge(cid, nid)

    for k, node in tree_json.items():
        if node['parent'] == None:
            root_idx = k

    _rec_build(None, root_idx, tree_json)
    ret = dgl.DGLGraph()

    nx_nodename_int_list = []
    for nx_node_id in g.nodes():
        if g.node[nx_node_id]["y"] not in nx_nodename_int_list:
            nx_nodename_int_list.append(g.node[nx_node_id]["y"])

    nodename_int_not_in_nx = []
    for this_node_name_int in nodename_int_list:
        if this_node_name_int not in nx_nodename_int_list:
            nodename_int_not_in_nx.append(this_node_name_int)

    ret.from_networkx(g, node_attrs=['x', 'y', 'mask'])
    return ret


def get_data_tree_ptb_dgl_graph(trees, dict_code, key_list=None):
    data_tree_dgl_graphs = []
    _cnt_display = 0
    for t_json in trees:
        tree_dgl_graph = build_tree(copy.deepcopy(t_json), dict_code)

        assert tree_dgl_graph.number_of_nodes() == len(t_json), print("tree_dgl_graph.number_of_nodes(): {} " \
                                                                      "\n len(t_json):{} \n t_json:\n{} \n tree_dgl_graph:{} \n tree_dgl_graph.nodes[0]:{} \n tree_dgl_graph.nodes[1]:{}". \
                                                                      format(tree_dgl_graph.number_of_nodes(),
                                                                             len(t_json), t_json, tree_dgl_graph,
                                                                             tree_dgl_graph.nodes[0],
                                                                             tree_dgl_graph.nodes[1]))

        data_tree_dgl_graphs.append(tree_dgl_graph)
        _cnt_display += 1

    return data_tree_dgl_graphs


def load_dict(opt):
    dict_code = Dict(
        [data.Constants.PAD_WORD, data.Constants.UNK_WORD, data.Constants.BOS_WORD, data.Constants.EOS_WORD],
        lower=opt.lower)

    dict_comment = Dict(
        [data.Constants.PAD_WORD, data.Constants.UNK_WORD, data.Constants.BOS_WORD, data.Constants.EOS_WORD],
        lower=opt.lower)
    dict_code.loadFile(opt.dict_code)

    dict_comment.loadFile(opt.dict_comment)
    return_dict = [dict_code, dict_comment]
    if opt.dataset_type == "c":
        dict_leaves = Dict(
            [data.Constants.PAD_WORD, data.Constants.UNK_WORD, data.Constants.BOS_WORD, data.Constants.EOS_WORD],
            lower=opt.lower)
        dict_leaves.loadFile(opt.ast_tree_leaves_dict)
        return_dict.append(dict_leaves)

    return return_dict


def load_data(opt, all_dict):
    dict_code, dict_comment = all_dict[0], all_dict[1]

    if opt.dataset_type == "c":
        dict_leaves = all_dict[2]
    else:
        dict_leaves = dict_code

    print("load_data.....")
    print("opt.data_train_ctg:\n", opt.data_train_ctg)
    print("opt.data_val_ctg:\n", opt.data_val_ctg)
    print("opt.data_test_ctg:\n", opt.data_test_ctg)

    train_ctg, val_ctg, test_ctg = torch.load(opt.data_train_ctg), torch.load(opt.data_val_ctg), \
                                   torch.load(opt.data_test_ctg)

    val_ctg_raw = copy.deepcopy(val_ctg)

    n_node = get_max_node_num(train_ctg['cfg'], val_ctg['cfg'], test_ctg['cfg'])

    if opt.dataset_type == "c":
        n_edge_types = len(list(config_about_voca.cfg_edge_color2index.keys()))
        print("main.py config_about_voca.cfg_edge_color2index: \n", config_about_voca.cfg_edge_color2index)
        print("main.py config_about_voca.cfg_edge_color2index.keys(): ", config_about_voca.cfg_edge_color2index.keys())
        print("list(config_about_voca.cfg_edge_color2index.keys()): ",
              list(config_about_voca.cfg_edge_color2index.keys()))
        print("main.py n_edge_types: ", n_edge_types)
        annotation_dim = len(list(config_about_voca.cfg_node_color2index.keys()))
    print("before_train_data n_node:{} , n_edge_types:{} ,annotation_dim:{}".format(n_node, n_edge_types,
                                                                                    annotation_dim))

    val_ctg_tree_dgl_processed = get_data_tree_ptb_dgl_graph(copy.deepcopy(val_ctg['tree']), dict_leaves,
                                                             copy.deepcopy(val_ctg['key_list']))
    print("finish val tree dgl processing")

    print("validation_with_metric: ", opt.validation_with_metric)
    if opt.train_mode == "train":

        if "tree" in opt.modal_type:
            print("begin  tree dgl processing")
            train_ctg['tree_dgl'] = get_data_tree_ptb_dgl_graph(copy.deepcopy(train_ctg['tree']), dict_leaves,
                                                                copy.deepcopy(train_ctg['key_list']))
            print("finish train tree dgl processing")

        val_ctg['tree_dgl'] = copy.deepcopy(val_ctg_tree_dgl_processed)

        train_dataset = CodeRetrievalDataset(opt, train_ctg, dict_code, dict_comment, n_node, n_edge_types,
                                             annotation_dim)

        val_dataset = CodeRetrievalDataset(opt, val_ctg, dict_code, dict_comment, n_node, n_edge_types, annotation_dim)

        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size,
                                                       shuffle=False,
                                                       num_workers=opt.workers, collate_fn=collate_fn_code_retrieval)
        val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False,
                                                     num_workers=opt.workers, collate_fn=collate_fn_code_retrieval)

    if opt.train_mode == "test" or opt.validation_with_metric:
        if not opt.use_val_as_codebase:
            print("use codebase dataset")

            print("process test dataset dgl tree....")
            test_ctg['tree_dgl'] = get_data_tree_ptb_dgl_graph(copy.deepcopy(test_ctg['tree']), dict_leaves,
                                                               copy.deepcopy(test_ctg['key_list']))

            print("finish test dataset")

            test_dataset = CodeRetrievalDataset(opt, test_ctg, dict_code, dict_comment, n_node, n_edge_types,
                                                annotation_dim, not_test_set=False)
            test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size,
                                                          shuffle=False,
                                                          num_workers=opt.workers, drop_last=False,
                                                          collate_fn=collate_fn_code_retrieval_for_test_set)

        else:
            print("use val set as codebase for debug")
            print("process val dataset dgl tree....")

            val_ctg_raw['tree_dgl'] = copy.deepcopy(val_ctg_tree_dgl_processed)

            test_dataset = CodeRetrievalDataset(opt, val_ctg_raw, dict_code, dict_comment, n_node, n_edge_types,
                                                annotation_dim, not_test_set=False)

            test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size,
                                                          shuffle=False,
                                                          num_workers=opt.workers, drop_last=False,
                                                          collate_fn=collate_fn_code_retrieval_for_test_set)

        query_dataset = CodeRetrievalQueryDataset(opt, dict_comment)
        query_dataloader = torch.utils.data.DataLoader(dataset=query_dataset, batch_size=opt.batch_size,
                                                       shuffle=False,
                                                       num_workers=opt.workers, drop_last=False,
                                                       collate_fn=collate_fn_code_retrieval_for_query)

    if opt.train_mode == "train" and not opt.validation_with_metric:
        print('opt.train_mode == "train" and not opt.validation_with_metric:')
        return train_dataset, train_dataloader, val_dataset, val_dataloader

    elif opt.train_mode == "train" and opt.validation_with_metric:
        print('opt.train_mode == "train" and opt.validation_with_metric:')
        return train_dataset, train_dataloader, val_dataset, val_dataloader, \
               test_dataset, test_dataloader, query_dataset, query_dataloader
    elif opt.train_mode == "test":
        return test_dataset, test_dataloader, query_dataset, query_dataloader


def create_model_code_retrieval(opt, dataset, all_dict):
    def _init_param(opt, model):
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

    dict_code, dict_comment = all_dict[0], all_dict[1]
    if opt.dataset_type == "c":
        dict_leaves = all_dict[2]
    else:
        dict_leaves = dict_code

    if opt.modal_type == "seq8tree8cfg9selfattn":

        seq_encoder = RetrievalCodeEncoderWrapper(opt, Encoder(opt, dict_code), "seq9coattn")
        tree_encoder = RetrievalCodeEncoderWrapper(opt, TreeEncoder_TreeLSTM_dgl(opt, dict_leaves), "tree9coattn")

        if opt.use_outmlp3:
            cfg_encoder = RetrievalCodeEncoderWrapper(opt,
                                                      CFGEncoder_GGNN(opt, dataset.new_annotation_dim,
                                                                      dataset.new_n_edge_types,
                                                                      dataset.new_n_node),
                                                      "cfg")
        else:
            cfg_encoder = RetrievalCodeEncoderWrapper(opt,
                                                      CFGEncoder_GGNN(opt, dataset.new_annotation_dim,
                                                                      dataset.new_n_edge_types,
                                                                      dataset.new_n_node),
                                                      "cfg9coattn")

        code_encoder = RetrievalCodeEncoderWrapper(opt, (seq_encoder, tree_encoder, cfg_encoder), opt.modal_type)
        comment_encoder = RetrievalCommentEncoderWrapper(opt, Encoder(opt, dict_comment))
        _init_param(opt, seq_encoder)
        _init_param(opt, tree_encoder)
        _init_param(opt, comment_encoder)
        if opt.modal_type == "seq8tree8cfg9selfattn":
            model = ModelCodeRetrieval(code_encoder, comment_encoder, opt)

    print("model.state_dict().keys(): \n ", model.state_dict().keys())
    if opt.model_from:
        if os.path.exists(opt.model_from):
            print("Loading from checkpoint at %s" % opt.model_from)
            checkpoint = torch.load(opt.model_from, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint)
        else:
            print("not load pt file")

    print("create_model_code_retrieval, opt.gpus: ", opt.gpus)
    if opt.gpus:
        model.cuda()
        print("model.cuda() ok")
        gpu_list = [int(k) for k in opt.gpus.split(",")]
        gpu_list = list(range(len(gpu_list)))
        if len(gpu_list) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_list)
            print("DataParallel ok , gpu_list: ", gpu_list)

    return model
