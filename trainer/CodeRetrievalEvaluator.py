import collections
import os
from collections import OrderedDict

import numpy as np
import torch

from utils.util import save_json, load_json


class CodeRetrievalEvaluator(object):

    def __init__(self, model, dataset_list, flag_for_val, all_dict, opt):

        self.model = model
        if flag_for_val:
            self.val_dataset = dataset_list[0]
            self.val_dataloader = dataset_list[1]
        else:
            self.test_dataset = dataset_list[0]
            self.test_dataloader = dataset_list[1]
            self.query_dataset = dataset_list[2]
            self.query_dataloader = dataset_list[3]
            self.func_content = self.test_dataset.func_content
            self.func_content_key_list = self.test_dataset.key_list
            self.raw_query = self.query_dataset.raw_query

        self.dict_code = all_dict[0]
        self.dict_comment = all_dict[1]
        self.opt = opt

        self.codebase_vec = None

    def validation(self, criterion):

        self.model.eval()

        total_loss = 0
        total_sample = 0
        iteration = 0

        data_iter = iter(self.val_dataloader)

        while iteration < len(self.val_dataloader):
            batch = data_iter.next()

            code_batch, code_length, tree_dgl_batch, tree_dgl_root_index, tree_dgl_node_num, \
            cfg_init_input_batch, cfg_anno_batch, cfg_adjmat_batch, cfg_node_mask, \
            comment_batch, comment_target_batch, comment_length, \
            bad_comment_batch, bad_comment_target_batch, bad_comment_length = batch

            total_sample += comment_batch.size()[0]

            AST_DGL_Batch = collections.namedtuple('AST_DGL_Batch', ['graph', 'mask', 'wordid', 'label'])

            if self.opt.gpus:

                comment_batch, comment_target_batch, comment_length, \
                bad_comment_batch, bad_comment_target_batch, bad_comment_length \
                    = map(lambda x: x.cuda(),
                          [comment_batch, comment_target_batch, comment_length,
                           bad_comment_batch, bad_comment_target_batch, bad_comment_length])

                if "seq" in self.opt.modal_type:
                    code_batch = code_batch.cuda()
                if "tree" in self.opt.modal_type:
                    tree_dgl_root_index, tree_dgl_node_num = \
                        map(lambda x: x.cuda(), [tree_dgl_root_index, tree_dgl_node_num])
                if "cfg" in self.opt.modal_type:
                    cfg_init_input_batch, cfg_anno_batch, cfg_adjmat_batch, \
                    cfg_node_mask, \
                        = map(lambda x: x.cuda(),
                              [cfg_init_input_batch, cfg_anno_batch, cfg_adjmat_batch,
                               cfg_node_mask])

                if "tree" in self.opt.modal_type:
                    tree_dgl_batch = AST_DGL_Batch(graph=tree_dgl_batch,
                                                   mask=tree_dgl_batch.ndata['mask'].cuda(),
                                                   wordid=tree_dgl_batch.ndata['x'].cuda(),
                                                   label=tree_dgl_batch.ndata['y'].cuda())
            else:
                if "tree" in self.opt.modal_type:
                    tree_dgl_batch = AST_DGL_Batch(graph=tree_dgl_batch,
                                                   mask=tree_dgl_batch.ndata['mask'],
                                                   wordid=tree_dgl_batch.ndata['x'],
                                                   label=tree_dgl_batch.ndata['y'])

            code_feat, good_comment_feat, bad_comment_feat = self.model(code_batch, code_length, tree_dgl_batch,
                                                                        tree_dgl_root_index, tree_dgl_node_num, \
                                                                        cfg_init_input_batch, cfg_anno_batch,
                                                                        cfg_adjmat_batch, cfg_node_mask, \
                                                                        comment_batch, comment_target_batch,
                                                                        comment_length, \
                                                                        bad_comment_batch, bad_comment_target_batch,
                                                                        bad_comment_length)

            loss = criterion(code_feat, good_comment_feat, bad_comment_feat)

            total_loss += (loss.item()) * comment_batch.size()[0]

            iteration += 1

        return total_loss / total_sample, total_sample

    def codebase2vec(self):
        self.model.eval()
        with torch.no_grad():

            codebase_vec_index2dataset_index_new = []

            codebase_vec_list = []

            data_iter = iter(self.test_dataloader)
            iteration = 0

            if not self.opt.validation_with_metric:
                print("len(self.test_dataset): ", len(self.test_dataset))
                print("len(self.test_dataloader): ", len(self.test_dataloader))
                print("self.opt.batch_size: ", self.opt.batch_size)

            while iteration < len(self.test_dataloader):
                batch = data_iter.next()

                code_batch, code_length, tree_dgl_batch, tree_dgl_root_index, tree_dgl_node_num, \
                cfg_init_input_batch, cfg_anno_batch, cfg_adjmat_batch, cfg_node_mask, \
                index_sort_code_seq_len_list_array, index_batch = batch

                codebase_vec_index2dataset_index_new.extend(index_batch)

                AST_DGL_Batch = collections.namedtuple('AST_DGL_Batch', ['graph', 'mask', 'wordid', 'label'])
                if self.opt.gpus:

                    code_batch, cfg_init_input_batch, cfg_anno_batch, cfg_adjmat_batch, \
                    tree_dgl_root_index, tree_dgl_node_num, \
                    cfg_node_mask \
                        = map(lambda x: x.cuda(),
                              [code_batch, cfg_init_input_batch, cfg_anno_batch, cfg_adjmat_batch,
                               tree_dgl_root_index, tree_dgl_node_num,
                               cfg_node_mask])

                    tree_dgl_batch = AST_DGL_Batch(graph=tree_dgl_batch,
                                                   mask=tree_dgl_batch.ndata['mask'].cuda(),
                                                   wordid=tree_dgl_batch.ndata['x'].cuda(),
                                                   label=tree_dgl_batch.ndata['y'].cuda())
                else:
                    tree_dgl_batch = AST_DGL_Batch(graph=tree_dgl_batch,
                                                   mask=tree_dgl_batch.ndata['mask'],
                                                   wordid=tree_dgl_batch.ndata['x'],
                                                   label=tree_dgl_batch.ndata['y'])

                code_feat = self.model.code_encoder(code_batch, code_length, tree_dgl_batch,
                                                    tree_dgl_root_index, tree_dgl_node_num,
                                                    cfg_init_input_batch, cfg_anno_batch,
                                                    cfg_adjmat_batch, cfg_node_mask)

                codebase_vec_list.append(code_feat.detach().cpu().numpy())

                iteration += 1

            codebase_vec = np.concatenate(codebase_vec_list)

            codebase_vec = self.normalize(codebase_vec)

            self.codebase_vec = codebase_vec

            c2d = {}
            for i in range(len(codebase_vec_index2dataset_index_new)):
                c2d[i] = codebase_vec_index2dataset_index_new[i]

            d2c = {}
            for c, d in c2d.items():
                d2c[d] = c

            self.dataset_index2codebase_vec_index_new = d2c
            self.codebase_vec_index2dataset_index_new = c2d

            if not self.opt.validation_with_metric:
                np.save(self.opt.codebase_vec_path, codebase_vec)

                index_json2save = {'dataset_index2codebase_vec_index_new': self.dataset_index2codebase_vec_index_new,
                                   'codebase_vec_index2dataset_index_new': self.codebase_vec_index2dataset_index_new}
                save_json(index_json2save, self.opt.dataset_index_and_codebase_vec_index_path)

    def normalize(self, data):
        """normalize matrix by rows"""
        normalized_data = data / np.linalg.norm(data, axis=1).reshape((data.shape[0], 1))
        return normalized_data

    def dot_np(self, data1, data2):
        """cosine similarity for normalized vectors"""
        return np.dot(data1, np.transpose(data2))

    def retrieval(self, pred_file):

        self.model.eval()
        with torch.no_grad():
            if self.opt.train_mode == 'test':
                if self.codebase_vec is None:
                    print("os.path.exists(self.opt.codebase_vec_path): ", os.path.exists(self.opt.codebase_vec_path))
                    print("self.opt.get_codebase_vec_from_scratch: ", self.opt.get_codebase_vec_from_scratch)
                    if not os.path.exists(
                            self.opt.codebase_vec_path) or self.opt.get_codebase_vec_from_scratch or self.opt.use_val_as_codebase:
                        print("run self.codebase2vec()")
                        self.codebase2vec()
                    else:
                        print("loading:\n", self.opt.codebase_vec_path)
                        self.codebase_vec = np.load(self.opt.codebase_vec_path)

                        codebase_index_json = load_json(self.opt.dataset_index_and_codebase_vec_index_path)
                        self.dataset_index2codebase_vec_index_new = codebase_index_json[
                            "dataset_index2codebase_vec_index_new"]
                        self.codebase_vec_index2dataset_index_new = codebase_index_json[
                            "codebase_vec_index2dataset_index_new"]

            else:
                self.codebase2vec()

            comment_vec_list = []

            data_iter = iter(self.query_dataloader)

            iteration = 0
            cnt_display = 0
            print("len(self.query_dataset): ", len(self.query_dataset))
            print("self.opt.batch_size: ", self.opt.batch_size)
            print("len(self.query_dataloader): ", len(self.query_dataloader))

            comment_vec_index2dataset_index_list = []
            while iteration < len(self.query_dataloader):

                batch = data_iter.next()

                comment_batch, comment_target_batch, comment_length, comment_index_batch = batch
                comment_vec_index2dataset_index_list.extend(comment_index_batch)
                if self.opt.gpus:
                    comment_batch, comment_target_batch, comment_length \
                        = map(lambda x: x.cuda(),
                              [comment_batch, comment_target_batch, comment_length])

                cnt_display += comment_batch.size()[0]

                comment_feat = self.model.comment_encoder(comment_batch, comment_target_batch, comment_length)

                comment_vec_list.append(comment_feat.detach().cpu().numpy())

                iteration += 1
            comment_vec = np.concatenate(comment_vec_list)

            comment_vec = self.normalize(comment_vec)

            evaluation_result = {}

            for _idx in range(comment_vec.shape[0]):

                dataset_index = comment_vec_index2dataset_index_list[_idx]
                print("_idx:{} dataset_index:{} ".format(_idx, dataset_index))

                one_query_vec = comment_vec[_idx].reshape(1, comment_vec.shape[1])

                one_query2codebase_sims = self.dot_np(one_query_vec, self.codebase_vec)[0]
                negsims = np.negative(one_query2codebase_sims)

                if self.opt.use_val_as_codebase:
                    index_sort_negsims = np.argsort(negsims)

                    target_rank = \
                        np.where(index_sort_negsims == self.dataset_index2codebase_vec_index_new[dataset_index])[0][0]

                    evaluation_result[dataset_index] = {"target_rank": target_rank}

            if self.opt.train_mode == 'test':
                print("self.opt.batch_size: ", self.opt.batch_size)
                print("len(evaluation_result): ", len(evaluation_result))
                print("comment_vec.shape: ", comment_vec.shape)
                print("len(dataset_index2codebase_vec_index_new): ", len(self.dataset_index2codebase_vec_index_new))

            self.dump_preds_json(evaluation_result, pred_file)

    def dump_preds_json(self, evaluation_result, pred_file):
        json_pred_file = pred_file.replace("-.re", "-.json")
        result_dict = OrderedDict()
        for i in range(len(evaluation_result)):
            tmp_dict = OrderedDict()
            if self.opt.use_val_as_codebase:
                target_rank = evaluation_result[i]["target_rank"]
                tmp_dict["target_rank"] = int(target_rank)

            result_dict["q_" + str(i)] = tmp_dict
        if self.opt.train_mode == 'test':
            print("result_dict: ", type(result_dict), result_dict)
        save_json(result_dict, json_pred_file)

    def eval_retrieval_json_result(self, pred_file):
        json_file = pred_file.replace("-.re", "-.json")

        result_dict = load_json(json_file)
        rank_all = []
        for k, v in result_dict.items():
            rank_all.append(v["target_rank"] + 1)

        if self.opt.train_mode == 'test':
            print("rank_all: \n", rank_all)
            print("\nrank_all printed \n")
            print("len(rank_all): ", len(rank_all))

        mrr, R1, R5, R10, ave = self.eval_retrieval4validation_with_metric(rank_all)

        print("json_file: \n", json_file)

        print("R1:    R5:   R10:   MRR:   Mean: \n{} {} {} {} {}".format(R1, R5, R10, mrr, ave))

    def eval_retrieval4validation_with_metric(self, rank_all):

        R1 = np.sum(np.array(rank_all) == 1) / float(len(rank_all))
        R5 = np.sum(np.array(rank_all) <= 5) / float(len(rank_all))
        R10 = np.sum(np.array(rank_all) <= 10) / float(len(rank_all))
        ave = np.sum(np.array(rank_all)) / float(len(rank_all))
        mrr = np.sum(1 / (np.array(rank_all, dtype='float'))) / float(len(rank_all))

        return mrr, R1, R5, R10, ave
