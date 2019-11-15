import torch.nn as nn


class ModelCodeRetrieval(nn.Module):
    def __init__(self, code_encoder, comment_encoder, opt):
        super(ModelCodeRetrieval, self).__init__()
        self.code_encoder = code_encoder
        self.comment_encoder = comment_encoder
        self.opt = opt

    def forward(self, code_batch, code_length, tree_dgl_batch, tree_dgl_root_index, tree_dgl_node_num, \
                cfg_init_input_batch, cfg_anno_batch, cfg_adjmat_batch, cfg_node_mask, \
                comment_batch, comment_target_batch, comment_length, \
                bad_comment_batch, bad_comment_target_batch, bad_comment_length):
        code_feat = self.code_encoder(code_batch, code_length, tree_dgl_batch, tree_dgl_root_index, tree_dgl_node_num, \
                                      cfg_init_input_batch, cfg_anno_batch, cfg_adjmat_batch, cfg_node_mask)

        good_comment_feat = self.comment_encoder(comment_batch, comment_target_batch, comment_length)
        bad_comment_feat = self.comment_encoder(bad_comment_batch, bad_comment_target_batch, bad_comment_length)

        return code_feat, good_comment_feat, bad_comment_feat
