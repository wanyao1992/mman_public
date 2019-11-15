import collections
import datetime
import os
import time

import torch

from .CodeRetrievalEvaluator import CodeRetrievalEvaluator


class CodeRetrievalTrainer(object):
    def __init__(self, model, train_dataset, train_dataloader, val_dataset, val_dataloader, all_dict, opt,
                 metric_data=None):
        self.model = model
        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.val_dataset = val_dataset
        self.val_dataloader = val_dataloader
        self.dict_code = all_dict[0]
        self.dict_comment = all_dict[1]

        self.evaluator = CodeRetrievalEvaluator(model=model, dataset_list=[val_dataset, val_dataloader],
                                                flag_for_val=True, all_dict=all_dict, opt=opt)

        if metric_data is not None:
            test_dataset = metric_data[0]
            test_dataloader = metric_data[1]
            query_dataset = metric_data[2]
            query_dataloader = metric_data[3]
            self.metric_evaluator = CodeRetrievalEvaluator(model=model,
                                                           dataset_list=[test_dataset, test_dataloader, query_dataset,
                                                                         query_dataloader],
                                                           flag_for_val=False, all_dict=all_dict, opt=opt)

        self.opt = opt

    def train(self, criterion, optim, pretrain_sl_epoch, start_time=None):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time
        for epoch in range(pretrain_sl_epoch):
            self.model.train()

            model_name = os.path.join(self.opt.save_dir,
                                      "model_xent_rd%s_%s_%s_e%s_bs%s_%s_nsteps%s_withanno_%s_spm%s_dr%s_lr%s_adw%s_tem%s_amf%s_aadw%s_cfgattnmode_%s_ma%s_ut%s_cm%s_tao%s_uo%s_ds_%s.pt" % \
                                      (self.opt.run_id,
                                       self.opt.dataset_type, self.opt.modal_type, epoch, self.opt.batch_size,
                                       self.opt.output_type,
                                       self.opt.n_steps, self.opt.with_anno,
                                       self.opt.supernode_mode, self.opt.dropout,
                                       self.opt.lr, self.opt.attn_delta_loss_weight,
                                       self.opt.transform_every_modal, self.opt.attn_modal_fusion,
                                       self.opt.adap_attn_delta_loss_weight, self.opt.cfg_cfgt_attn_mode,
                                       self.opt.cos_ranking_loss_margin, self.opt.use_tanh, self.opt.cfg_cfgt_mlp,
                                       self.opt.transform_attn_out, self.opt.use_outmlp3,
                                       self.opt.retrieval_train_dataset_split_type))

            train_loss_averaged = self.train_epoch(epoch, criterion, optim)
            print("epoch: {} train_loss_averaged: {} ".format(epoch, train_loss_averaged))

            if self.opt.retrieval_train_dataset_split_type == "train":

                time0 = datetime.datetime.now()

                val_loss_averaged, total_val_sample = self.evaluator.validation(criterion)
                print("epoch: {} val_loss_averaged(origin_ranking_loss,no co-attn): {} total_val_sample:{}". \
                      format(epoch, val_loss_averaged, total_val_sample))

                time1 = datetime.datetime.now()

                print("time for validation: ", (time1 - time0))

                if (epoch + 1) % 1 == 0:
                    retrieval_pred_file = model_name + "-" + self.opt.data_query_file.split('/')[-1] + \
                                          "-uv" + str(self.opt.use_val_as_codebase) + "r" + str(
                        self.opt.remove_wh_word) + "-.re"
                    self.metric_evaluator.retrieval(pred_file=retrieval_pred_file)
                    self.metric_evaluator.eval_retrieval_json_result(pred_file=retrieval_pred_file)

                    time2 = datetime.datetime.now()
                    print("time for metric: ", (time2 - time1))

                print("time for metric: ", (datetime.datetime.now() - time1))

            else:
                print("no validation in train progress, self.opt.retrieval_train_dataset_split_type: ",
                      self.opt.retrieval_train_dataset_split_type)

            if len(self.opt.gpus) == 1:
                torch.save(self.model.state_dict(), model_name)
            elif len(self.opt.gpus) > 1:
                torch.save(self.model.module.state_dict(), model_name)
            print("Save model as %s" % model_name)

    def train_epoch(self, epoch, criterion, optim):
        self.model.train()
        total_train_loss = 0

        total_sample = 0
        iteration = 0

        data_iter = iter(self.train_dataloader)

        while iteration < len(self.train_dataloader):
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
                    cfg_node_mask \
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

            code_feat, good_comment_feat, bad_comment_feat = self.model(
                code_batch, code_length, tree_dgl_batch, tree_dgl_root_index, tree_dgl_node_num,
                cfg_init_input_batch, cfg_anno_batch, cfg_adjmat_batch, cfg_node_mask,
                comment_batch, comment_target_batch, comment_length,
                bad_comment_batch, bad_comment_target_batch, bad_comment_length)

            loss = criterion(code_feat, good_comment_feat, bad_comment_feat)

            self.model.zero_grad()
            loss.backward()
            optim.step()
            total_train_loss += (loss.item()) * comment_batch.size()[0]
            iteration += 1
            if iteration % self.opt.log_interval == 0 and iteration > 0:
                print('Epoch %3d, %6d/%d batches; loss: %14.10f; %s elapsed' % (
                    epoch, iteration, len(self.train_dataloader), loss,
                    str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))

        return total_train_loss / total_sample
