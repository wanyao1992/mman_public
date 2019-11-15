import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import data.Constants
from utils.util import init_xavier_linear


class Encoder(nn.Module):
    def __init__(self, opt, dicts):
        super(Encoder, self).__init__()
        self.opt = opt
        self.nlayers = opt.nlayers
        self.dicts = dicts
        self.rnn_type = opt.rnn_type
        self.nhid = opt.nhid

        self.wemb = nn.Embedding(dicts.size(), opt.ninp, padding_idx=data.Constants.PAD)

        if self.opt.init_type in ["xulu"]:
            init_xavier_linear(opt, self.wemb, init_bias=False)

        self.rnn = getattr(nn, opt.rnn_type)(opt.ninp, opt.nhid, opt.nlayers, dropout=opt.dropout, batch_first=True)



    def forward(self, seq, seq_length, hidden=None):

        seq_emb = self.wemb(seq)

        if seq_length:

            seq_emb_packed = pack_padded_sequence(seq_emb, seq_length, batch_first=True)

            output, hidden = self.rnn(seq_emb_packed, hidden)
        else:
            output, hidden = self.rnn(seq_emb, hidden)

        return output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (weight.new(self.nlayers, bsz, self.nhid).zero_().requires_grad_(),
                    weight.new(self.nlayers, bsz, self.nhid).zero_().requires_grad_())
        else:
            return weight.new(self.nlayers, bsz, self.nhid).zero_().requires_grad_()


class TreeLSTMCell_dgl(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell_dgl, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_cat = nodes.mailbox['h'].reshape(nodes.mailbox['h'].size(0), -1)
        f = torch.sigmoid(self.U_f(h_cat)).reshape(*nodes.mailbox['h'].size())
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return_iou = self.U_iou(h_cat)

        return {'iou': return_iou, 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)

        return {'h': h, 'c': c}


class ChildSumTreeLSTMCell_dgl(nn.Module):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMCell_dgl, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_tild = torch.sum(nodes.mailbox['h'], 1)
        f = torch.sigmoid(self.U_f(nodes.mailbox['h']))
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_tild), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}


class TreeEncoder_TreeLSTM_dgl(nn.Module):
    def __init__(self, opt, dicts):
        super(TreeEncoder_TreeLSTM_dgl, self).__init__()
        self.nlayers = opt.nlayers
        self.opt = opt
        self.dicts = dicts
        self.nhid = opt.nhid
        self.wemb = nn.Embedding(dicts.size(), opt.ninp, padding_idx=data.Constants.PAD)
        if self.opt.init_type in ["xulu"]:
            init_xavier_linear(opt, self.wemb, init_bias=False)
        cell = TreeLSTMCell_dgl if self.opt.tree_lstm_cell_type == 'nary' else ChildSumTreeLSTMCell_dgl
        self.cell = cell(opt.ninp, self.opt.nhid)



    def forward(self, batch, enc_hidden, list_root_index, list_num_node):

        g = batch.graph
        g.register_message_func(self.cell.message_func)
        g.register_reduce_func(self.cell.reduce_func)
        g.register_apply_node_func(self.cell.apply_node_func)

        g.ndata['h'] = enc_hidden[0]
        g.ndata['c'] = enc_hidden[1]
        wemb = self.wemb(batch.wordid * batch.mask)
        g.ndata['iou'] = self.cell.W_iou(wemb) * batch.mask.float().unsqueeze(-1)

        dgl.prop_nodes_topo(g)

        all_node_h_in_batch = g.ndata.pop('h')
        all_node_c_in_batch = g.ndata.pop('c')

        if self.opt.tree_lstm_output_type == "root_node":
            root_node_h_in_batch, root_node_c_in_batch = [], []
            add_up_num_node = 0
            for _i in range(len(list_root_index)):
                if _i - 1 < 0:
                    add_up_num_node = 0
                else:
                    add_up_num_node += list_num_node[_i - 1]
                idx_to_query = list_root_index[_i] + add_up_num_node
                root_node_h_in_batch.append(all_node_h_in_batch[idx_to_query])
                root_node_c_in_batch.append(all_node_c_in_batch[idx_to_query])

            root_node_h_in_batch = torch.cat(root_node_h_in_batch).reshape(1, len(root_node_h_in_batch), -1)
            root_node_c_in_batch = torch.cat(root_node_c_in_batch).reshape(1, len(root_node_c_in_batch), -1)

            return root_node_h_in_batch, root_node_c_in_batch
        elif self.opt.tree_lstm_output_type == "no_reduce":
            return all_node_h_in_batch, all_node_c_in_batch

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        return (weight.new(bsz, self.nhid).zero_().requires_grad_(),
                weight.new(bsz, self.nhid).zero_().requires_grad_())


class AttrProxy(object):
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propogator(nn.Module):
    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Tanh()
        )


    def forward(self, state_in, state_out, state_cur, A):
        A_in = A[:, :, :self.n_node * self.n_edge_types]
        A_out = A[:, :, self.n_node * self.n_edge_types:]

        a_in = torch.bmm(A_in, state_in)
        a_out = torch.bmm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output


class CFGEncoder_GGNN(nn.Module):

    def __init__(self, opt, annotation_dim, n_edge_types, n_node):

        super(CFGEncoder_GGNN, self).__init__()
        self.nlayers = opt.nlayers
        self.opt = opt
        self.nhid = opt.nhid

        self.state_dim = opt.state_dim
        self.annotation_dim = annotation_dim
        self.n_edge_types = n_edge_types
        self.n_node = n_node
        self.n_steps = opt.n_steps
        self.output_type = opt.output_type
        self.batch_size = opt.batch_size
        self.with_anno = opt.with_anno

        self.in_fcs = nn.ModuleList([nn.Linear(self.state_dim, self.state_dim) for _ in range(int(self.n_edge_types))])
        self.out_fcs = nn.ModuleList([nn.Linear(self.state_dim, self.state_dim) for _ in range(int(self.n_edge_types))])

        self.propogator = Propogator(self.state_dim, self.n_node, self.n_edge_types)

        if self.with_anno:
            self.convert_state_anno_dim = nn.Linear(self.state_dim + self.annotation_dim, self.state_dim)




        if opt.output_type in ["no_reduce", "sum"]:
            self.out_mlp = nn.Sequential(
                nn.Dropout(p=self.opt.dropout, inplace=False),
                nn.Linear(self.state_dim + self.state_dim, self.state_dim),
                nn.Tanh()
            )
        else:
            assert False

        self.init_param()

    def init_param(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):

                if self.opt.init_type in ["xulu"]:
                    init_xavier_linear(self.opt, m)


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        return (weight.new(self.nlayers, bsz, self.nhid).zero_().requires_grad_(),
                weight.new(self.nlayers, bsz, self.nhid).zero_().requires_grad_())

    def forward(self, prop_state, annotation, A, node_mask):

        initial_prop_state = prop_state

        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(int(self.n_edge_types)):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))

            in_states = torch.stack(in_states).transpose(0, 1).contiguous()

            in_states = in_states.reshape(-1, self.n_node * self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.reshape(-1, self.n_node * self.n_edge_types, self.state_dim)

            prop_state = self.propogator(in_states, out_states, prop_state, A)




        if self.output_type == "no_reduce":
            prop_state_cat = torch.cat((prop_state, initial_prop_state), 2)

            output = self.out_mlp(prop_state_cat)
        elif self.output_type == "sum":
            prop_state_cat = torch.cat((prop_state, initial_prop_state), 2)

            prop_state_to_sum = self.out_mlp(prop_state_cat)
            output_list = []

            for _i in range(prop_state.size()[0]):
                before_out = torch.masked_select(prop_state_to_sum[_i, :, :].reshape(1, -1, self.state_dim),
                                                 node_mask[_i].reshape(1, -1, 1)).reshape(1, -1, self.state_dim)
                out_to_append = torch.tanh(torch.sum(before_out, 1)).reshape(1, self.state_dim)
                output_list.append(out_to_append)
            output = torch.cat(output_list, 0).reshape(1, -1, self.state_dim)

        return output


class RetrievalCodeEncoderWrapper(nn.Module):
    def __init__(self, opt, encoder, modal):
        super(RetrievalCodeEncoderWrapper, self).__init__()

        self.opt = opt
        self.modal = modal

        if self.opt.save_attn_weight:
            self.attn_weight_torch = []
            self.node_mask_torch = []

        print("RetrievalCodeEncoderWrapper__init__self.opt.transform_every_modal: ", self.opt.transform_every_modal)
        print("RetrievalCodeEncoderWrapper__init__self.opt.attn_modal_fusion: ", self.opt.attn_modal_fusion)

        if self.modal in ["seq", "tree", "cfg"]:
            self.encoder = encoder

            if self.opt.transform_every_modal:
                self.linear_single_modal = nn.Sequential(nn.Linear(self.opt.nhid, self.opt.nhid),
                                                         nn.Tanh(),
                                                         nn.Linear(self.opt.nhid, self.opt.nhid))

            if self.modal in ["cfg"]:
                if self.opt.cfg_cfgt_mlp:
                    self.cfg_mlp_after_sum = nn.Sequential(nn.Linear(self.opt.nhid, self.opt.nhid),
                                                           nn.Linear(self.opt.nhid, self.opt.nhid))

        elif self.modal in ["seq9coattn", "tree9coattn", "cfg9coattn"]:

            self.encoder = encoder

            if self.opt.transform_every_modal:
                self.linear_single_modal = nn.Sequential(nn.Linear(self.opt.nhid, self.opt.nhid),
                                                         nn.Tanh(),
                                                         nn.Linear(self.opt.nhid, self.opt.nhid))
            if self.opt.transform_attn_out:
                self.linear_attn_out = nn.Sequential(nn.Linear(self.opt.nhid, self.opt.nhid),
                                                     nn.Tanh(),
                                                     nn.Linear(self.opt.nhid, self.opt.nhid))

            if self.modal in ["cfg9coattn"]:
                if self.opt.cfg_cfgt_mlp:
                    self.cfg_mlp_after_sum = nn.Sequential(nn.Linear(self.opt.nhid, self.opt.nhid),
                                                           nn.Linear(self.opt.nhid, self.opt.nhid))

            self.self_atten = nn.Linear(self.opt.nhid, self.opt.nhid)
            self.self_atten_scalar = nn.Linear(self.opt.nhid, 1)
            self.dual_atten = nn.Linear(self.opt.nhid, self.opt.nhid)
            self.dual_atten_scalar = nn.Linear(self.opt.nhid, 1)
            self.comment_atten = nn.Linear(self.opt.nhid, self.opt.nhid)



        elif self.modal in ["seq8tree8cfg",  "seq8tree8cfg9selfattn"]:

            self.seq_encoder = encoder[0]
            self.tree_encoder = encoder[1]
            self.cfg_encoder = encoder[2]
            if self.opt.attn_modal_fusion:
                self.W_seq = nn.Linear(self.opt.nhid, self.opt.nhid)
                self.W_tree = nn.Linear(self.opt.nhid, self.opt.nhid)
                self.W_cfg = nn.Linear(self.opt.nhid, self.opt.nhid)
                self.W_a = nn.Linear(self.opt.nhid, 1)
            self.linear = nn.Linear(self.opt.nhid * 3, self.opt.nhid)

    def forward(self, code_batch, code_length, tree_dgl_batch, tree_dgl_root_index, tree_dgl_node_num, \
                cfg_init_input_batch, cfg_anno_batch, cfg_adjmat_batch, cfg_node_mask):

        if "seq" in self.modal:
            batch_size = code_batch.size()[0]
        if "tree" in self.modal:
            batch_size = tree_dgl_root_index.size()[0]
        if "cfg" in self.modal:
            batch_size = cfg_node_mask.size()[0]

        if self.modal == 'seq':

            code_enc_hidden = self.encoder.init_hidden(batch_size)

            code_enc_output, code_enc_hidden = self.encoder(code_batch, code_length, code_enc_hidden)

            if self.opt.rnn_type != "GRU":
                code_enc_hidden = code_enc_hidden[0]
            else:
                code_enc_hidden = code_enc_hidden
            if code_enc_hidden.size()[0] == 1:
                code_enc_hidden = code_enc_hidden.reshape(code_enc_hidden.size()[1], code_enc_hidden.size()[2])
            else:
                assert False, print("对于多层和双向，暂未处理")

            code_feat = code_enc_hidden.reshape(batch_size, self.opt.nhid)

            if self.opt.transform_every_modal:
                code_feat = torch.tanh(
                    self.linear_single_modal(F.dropout(code_feat, self.opt.dropout, training=self.training)))

            elif self.opt.use_tanh:
                code_feat = torch.tanh(code_feat)

            return code_feat

        elif self.modal == "tree":

            code_enc_hidden = self.encoder.init_hidden(tree_dgl_batch.graph.number_of_nodes())
            code_enc_hidden0, code_enc_hidden1 = self.encoder(tree_dgl_batch, code_enc_hidden, tree_dgl_root_index,
                                                              tree_dgl_node_num)

            code_feat = code_enc_hidden0.reshape(tree_dgl_root_index.size()[0], -1)

            if self.opt.transform_every_modal:
                code_feat = torch.tanh(
                    self.linear_single_modal(F.dropout(code_feat, self.opt.dropout, training=self.training)))

            elif self.opt.use_tanh:
                code_feat = torch.tanh(code_feat)

            return code_feat



        elif self.modal == "cfg":
            code_feat = self.encoder(cfg_init_input_batch, cfg_anno_batch, cfg_adjmat_batch, cfg_node_mask)

            code_feat = code_feat.reshape(cfg_node_mask.size()[0], -1)

            if self.opt.transform_every_modal:
                code_feat = torch.tanh(
                    self.linear_single_modal(F.dropout(code_feat, self.opt.dropout, training=self.training)))


            elif self.opt.cfg_cfgt_mlp:
                code_feat = torch.tanh(self.cfg_mlp_after_sum(code_feat))

            return code_feat


        elif self.modal == "tree9coattn":
            code_enc_hidden = self.encoder.init_hidden(tree_dgl_batch.graph.number_of_nodes())
            all_node_h_in_batch, all_node_c_in_batch = self.encoder(tree_dgl_batch, code_enc_hidden,
                                                                    tree_dgl_root_index,
                                                                    tree_dgl_node_num)

            if self.opt.transform_every_modal:
                all_node_h_in_batch = torch.tanh(
                    self.linear_single_modal(F.dropout(all_node_h_in_batch, self.opt.dropout, training=self.training)))




            elif self.opt.use_tanh:
                all_node_h_in_batch = torch.tanh(all_node_h_in_batch)

            self_attn_code_feat = None
            add_up_node_num = 0
            for _i in range(batch_size):
                this_sample_h = all_node_h_in_batch[add_up_node_num:add_up_node_num + tree_dgl_node_num[_i]]
                add_up_node_num += tree_dgl_node_num[_i]
                node_num = tree_dgl_node_num[_i]
                code_sa = self.self_atten(this_sample_h.reshape(-1, self.opt.nhid))
                code_sa_tanh = F.tanh(code_sa)
                code_sa_tanh = F.dropout(code_sa_tanh, self.opt.dropout, training=self.training)

                code_sa_before_softmax = self.self_atten_scalar(code_sa_tanh).reshape(1, node_num)
                self_atten_weight = F.softmax(code_sa_before_softmax, dim=1)

                if self.opt.save_attn_weight:
                    self.attn_weight_torch.append(self_atten_weight.detach().reshape(1, -1).cpu())

                self_attn_this_sample_h = torch.bmm(self_atten_weight.reshape(1, 1, node_num),
                                                    this_sample_h.reshape(1, node_num, self.opt.nhid)).reshape(1,
                                                                                                               self.opt.nhid)
                if self_attn_code_feat is None:
                    self_attn_code_feat = self_attn_this_sample_h
                else:
                    self_attn_code_feat = torch.cat((self_attn_code_feat, self_attn_this_sample_h), 0)

            if self.opt.transform_attn_out:
                self_attn_code_feat = torch.tanh(
                    self.linear_attn_out(
                        F.dropout(self_attn_code_feat, self.opt.dropout, training=self.training)))

            elif self.opt.use_tanh:
                self_attn_code_feat = torch.tanh(self_attn_code_feat)

            return self_attn_code_feat

        elif self.modal in ["cfg9coattn", "seq9coattn"]:
            if self.modal == "cfg9coattn":
                code_feat = self.encoder(cfg_init_input_batch, cfg_anno_batch, cfg_adjmat_batch, cfg_node_mask)

            if self.modal in ["cfg9coattn"]:
                node_num = cfg_node_mask.size()[1]
                code_feat = code_feat.reshape(-1, node_num, self.opt.nhid)
                mask_1forgt0 = cfg_node_mask.byte().reshape(-1, node_num)

            elif self.modal == "seq9coattn":
                code_enc_hidden = self.encoder.init_hidden(batch_size)
                code_enc_output_packed, code_enc_hidden = self.encoder(code_batch, code_length, code_enc_hidden)
                code_feat, unpack_len_list = pad_packed_sequence(code_enc_output_packed, batch_first=True)
                code_feat = code_feat.reshape(batch_size, -1, self.opt.nhid)

                if self.opt.use_tanh:
                    code_feat = torch.tanh(code_feat)

                node_num = code_feat.size()[1]

                if self.opt.gpus:
                    unpack_len_list = unpack_len_list.long().cuda()
                    range_tensor = torch.arange(node_num).cuda()
                    mask_1forgt0 = range_tensor[None, :] < unpack_len_list[:, None]
                else:
                    unpack_len_list = unpack_len_list.long()
                    range_tensor = torch.arange(node_num)
                    mask_1forgt0 = range_tensor[None, :] < unpack_len_list[:, None]

                mask_1forgt0 = mask_1forgt0.byte().reshape(-1, node_num)

            if self.opt.transform_every_modal:
                code_feat = torch.tanh(
                    self.linear_single_modal(F.dropout(code_feat, self.opt.dropout, training=self.training)))

            code_sa_tanh = F.tanh(self.self_atten(code_feat.reshape(-1, self.opt.nhid)))
            code_sa_tanh = F.dropout(code_sa_tanh, self.opt.dropout, training=self.training)
            code_sa_tanh = self.self_atten_scalar(code_sa_tanh).reshape(-1, node_num)

            code_feat = code_feat.reshape(-1, node_num, self.opt.nhid)
            batch_size = code_feat.size()[0]

            self_attn_code_feat = None
            for _i in range(batch_size):
                code_sa_tanh_one = torch.masked_select(code_sa_tanh[_i, :], mask_1forgt0[_i, :]).reshape(1, -1)

                if self.modal in ["cfg9coattn"] \
                        and self.opt.cfg_cfgt_attn_mode == "sigmoid_scalar":

                    attn_w_one = torch.sigmoid(code_sa_tanh_one).reshape(1, 1, -1)


                else:
                    attn_w_one = F.softmax(code_sa_tanh_one, dim=1).reshape(1, 1, -1)

                if self.opt.save_attn_weight:
                    self.attn_weight_torch.append(attn_w_one.detach().reshape(1, -1).cpu())
                    self.node_mask_torch.append(mask_1forgt0[_i, :].detach().reshape(1, -1).cpu())

                attn_feat_one = torch.masked_select(code_feat[_i, :, :].reshape(1, node_num, self.opt.nhid),
                                                    mask_1forgt0[_i, :].reshape(1, node_num, 1)).reshape(1, -1,
                                                                                                         self.opt.nhid)
                out_to_cat = torch.bmm(attn_w_one, attn_feat_one).reshape(1, self.opt.nhid)
                self_attn_code_feat = out_to_cat if self_attn_code_feat is None else torch.cat(
                    (self_attn_code_feat, out_to_cat), 0)

            if self.opt.transform_attn_out:
                self_attn_code_feat = torch.tanh(
                    self.linear_attn_out(
                        F.dropout(self_attn_code_feat, self.opt.dropout, training=self.training)))
            elif self.modal in ["cfg9coattn"] and self.opt.cfg_cfgt_attn_mode \
                    == "sigmoid_scalar":
                self_attn_code_feat = torch.tanh(self_attn_code_feat)

            else:
                if self.modal in ["cfg9coattn"]:
                    if self.opt.cfg_cfgt_mlp:
                        self_attn_code_feat = self.cfg_mlp_after_sum(self_attn_code_feat)
                        if not self.opt.use_tanh:
                            self_attn_code_feat = torch.tanh(self_attn_code_feat)


                elif self.opt.use_tanh:
                    self_attn_code_feat = torch.tanh(self_attn_code_feat)

            return self_attn_code_feat




        elif self.modal in ["seq8tree8cfg", "seq8tree8cfg9selfattn"]:

            seq_feat = self.seq_encoder(code_batch, code_length, tree_dgl_batch, tree_dgl_root_index,
                                        tree_dgl_node_num,
                                        cfg_init_input_batch, cfg_anno_batch, cfg_adjmat_batch, cfg_node_mask)

            tree_feat = self.tree_encoder(code_batch, code_length, tree_dgl_batch, tree_dgl_root_index,
                                          tree_dgl_node_num,
                                          cfg_init_input_batch, cfg_anno_batch, cfg_adjmat_batch, cfg_node_mask)

            cfg_feat = self.cfg_encoder(code_batch, code_length, tree_dgl_batch, tree_dgl_root_index,
                                        tree_dgl_node_num,
                                        cfg_init_input_batch, cfg_anno_batch, cfg_adjmat_batch, cfg_node_mask)

            if self.opt.attn_modal_fusion:

                seq_feat_d = self.W_seq(seq_feat).reshape(-1, self.opt.nhid)
                tree_feat_d = self.W_tree(tree_feat).reshape(-1, self.opt.nhid)
                cfg_feat_d = self.W_cfg(cfg_feat).reshape(-1, self.opt.nhid)
                seq_attn_tanh = torch.tanh(seq_feat_d)
                seq_attn_scalar = self.W_a(
                    F.dropout(seq_attn_tanh, self.opt.dropout, training=self.training).reshape(-1, self.opt.nhid))
                tree_attn_tanh = torch.tanh(tree_feat_d)
                tree_attn_scalar = self.W_a(
                    F.dropout(tree_attn_tanh, self.opt.dropout, training=self.training).reshape(-1, self.opt.nhid))
                cfg_attn_tanh = torch.tanh(cfg_feat_d)
                cfg_attn_scalar = self.W_a(
                    F.dropout(cfg_attn_tanh, self.opt.dropout, training=self.training).reshape(-1, self.opt.nhid))
                attn_catted = torch.cat([seq_attn_scalar, tree_attn_scalar, cfg_attn_scalar], 1)
                atten_weight = F.softmax(attn_catted, dim=1)

                seq_feat_d_attened = torch.bmm(atten_weight[:, 0].reshape(batch_size, 1, 1),
                                               seq_feat.reshape(batch_size, 1, self.opt.nhid))
                tree_feat_d_attened = torch.bmm(atten_weight[:, 1].reshape(batch_size, 1, 1),
                                                tree_feat.reshape(batch_size, 1, self.opt.nhid))
                cfg_feat_d_attened = torch.bmm(atten_weight[:, 2].reshape(batch_size, 1, 1),
                                               cfg_feat.reshape(batch_size, 1, self.opt.nhid))

                concat_feat = torch.cat((seq_feat_d_attened, tree_feat_d_attened, cfg_feat_d_attened), 2)
            else:

                concat_feat = torch.cat((seq_feat, tree_feat, cfg_feat), 1)

            code_feat = torch.tanh(
                self.linear(F.dropout(concat_feat, self.opt.dropout, training=self.training))).reshape(-1,
                                                                                                       self.opt.nhid)

            return code_feat


class RetrievalCommentEncoderWrapper(nn.Module):
    def __init__(self, opt, encoder):
        super(RetrievalCommentEncoderWrapper, self).__init__()
        self.encoder = encoder
        self.opt = opt

        print("RetrievalCommentEncoderWrapper__init__ self.opt.transform_every_modal:  ",
              self.opt.transform_every_modal)

        if self.opt.transform_every_modal:
            self.linear_single_modal = nn.Sequential(nn.Linear(self.opt.nhid, self.opt.nhid),
                                                     nn.Tanh(),
                                                     nn.Linear(self.opt.nhid, self.opt.nhid))

    def forward(self, comment_batch, comment_target_batch, comment_length):

        _, idx_sort = torch.sort(comment_length, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        comment_batch = comment_batch.index_select(0, idx_sort)
        comment_length = list(comment_length[idx_sort])

        comment_enc_hidden = self.encoder.init_hidden(comment_batch.size()[0])

        comment_output, comment_hidden = self.encoder(comment_batch, comment_length, comment_enc_hidden)
        if self.opt.rnn_type != "GRU":
            comment_hidden = comment_hidden[0]
        else:
            comment_hidden = comment_hidden
        if comment_hidden.size()[0] == 1:
            comment_hidden = comment_hidden.reshape(comment_hidden.size()[1], comment_hidden.size()[2])
        else:
            assert False, print("对于多层和双向，暂未处理")
        comment_hidden = comment_hidden.index_select(0, idx_unsort)

        comment_feat = comment_hidden.reshape(comment_batch.size()[0], self.opt.nhid)

        if self.opt.transform_every_modal:
            comment_feat = torch.tanh(
                self.linear_single_modal(F.dropout(comment_feat, self.opt.dropout, training=self.training)))

        elif self.opt.use_tanh:
            comment_feat = torch.tanh(comment_feat)

        return comment_feat
