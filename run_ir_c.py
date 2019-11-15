import argparse
import datetime
import os
import socket
import sys

now = datetime.datetime.now()
month = now.strftime("%b")
day = now.strftime("%d")
parser = argparse.ArgumentParser(description='run')
parser.add_argument('--run_func', default='train', help='run_func')

config = parser.parse_args()

config.task_mode = "code-ir-mm"
config.dataset_type = 'c'

if socket.gethostname() in ["amax-xp", "new-ubuntu", "realdoctor", "amax-new"]:
    config.hostname = 'amax-xp'
    config.data_dir = ''
else:
    config.hostname = 'ubuntu'
    config.data_dir = ''

config.remove_wh_word = 0
train_folder = "train"
config.dict_code = os.path.join(config.data_dir, train_folder, 'processed_all_c.code.dict')
config.dict_comment = os.path.join(config.data_dir, train_folder, 'processed_all_c.comment.dict')
config.data_train_ctg = os.path.join(config.data_dir, train_folder, 'processed_all_c.retrieval.train_ct.pt')
config.data_val_ctg = os.path.join(config.data_dir, train_folder, 'processed_all_c.retrieval.val_ct.pt')
config.data_test_ctg = os.path.join(config.data_dir, train_folder, 'processed_all_c.retrieval.codebase_ct.pt')
config.ast_tree_leaves_dict = os.path.join(config.data_dir, train_folder, 'processed_all_c.ast_tree_leaves_voca.dict')
config.save_dir = os.path.join(config.data_dir, 'result')
config.data_codebase_func_content = os.path.join(config.data_dir, train_folder,
                                                 'processed_all_c.codebase_func_content.pt')
config.data_pairs_func_content = os.path.join(config.data_dir, train_folder, 'processed_all_c.pairs_func_content.pt')
config.embedding_w2v = config.data_dir


def train():
    config.train_mode = 'train'
    config.run_id = 10
    config.pretrain_sl_epoch = 500

    config.gpus = "5"
    config.modal_type = 'seq8tree8cfg9selfattn'
    config.batch_size = 32
    config.dropout = 0
    config.lr = 0.0001
    config.cfg_cfgt_attn_mode = "sigmoid_scalar"
    config.use_tanh = 0
    config.cfg_cfgt_mlp = 0
    config.transform_attn_out = 0
    config.use_outmlp3 = 0

    config.cos_ranking_loss_margin = 0.05
    config.validation_with_metric = 1

    config.query_file_name = _get_val_query()

    config.data_query_file = os.path.join(config.data_dir, train_folder, config.query_file_name)

    if config.validation_with_metric == 1:
        config.use_val_as_codebase = 1
    else:
        config.use_val_as_codebase = 0

    config.adap_attn_delta_loss_weight = 0
    config.attn_delta_loss_weight = 0
    config.transform_every_modal = 0
    config.attn_modal_fusion = 0

    config.output_type = "sum"
    config.supernode_mode = 1
    config.with_anno = 1
    config.n_steps = 5
    config.retrieval_train_dataset_split_type = "train"

    if "attn" in config.modal_type:
        config.output_type = "no_reduce"
        config.tree_lstm_output_type = "no_reduce"
    else:
        config.tree_lstm_output_type = "root_node"



    config.log = \
        '%s/log/log.rd%s_%s_%s_%s-e%s_lr%s_d%s_%s_nstep%s_anno%s_superm%s_bs%s_ds_' \
        '%s_g%s_adw%s_tem%s_amf%s_aadw%s_cfgattnmode_%s_ma%s_ut%s_cm%s_tao%s_uo%s_%s%s' % \
        (config.data_dir,
         config.run_id,
         config.train_mode, config.dataset_type, config.modal_type,
         config.pretrain_sl_epoch, config.lr, config.dropout, config.output_type,
         config.n_steps, config.with_anno, config.supernode_mode, config.batch_size,
         config.retrieval_train_dataset_split_type,
         "_".join(config.gpus.split(",")), config.attn_delta_loss_weight,
         config.transform_every_modal, config.attn_modal_fusion, config.adap_attn_delta_loss_weight,
         config.cfg_cfgt_attn_mode, config.cos_ranking_loss_margin, config.use_tanh, config.cfg_cfgt_mlp,
         config.transform_attn_out, config.use_outmlp3, month, day)

    run = 'python -u main.py --train_mode %s --modal_type %s ' \
          '--dict_code %s --dict_comment %s --data_train_ctg %s --data_val_ctg %s --data_test_ctg %s ' \
          '--save_dir %s --embedding_w2v %s ' \
          '--pretrain_sl_epoch %s  --lr %s --dropout %s ' \
          '--output_type %s --n_steps %d --with_anno %d --supernode_mode %s --batch_size %s --gpus %s' \
          ' --task_mode %s --retrieval_train_dataset_split_type %s --ast_tree_leaves_dict %s --dataset_type %s ' \
          '--data_codebase_func_content %s --data_pairs_func_content %s ' \
          ' --run_id %s --tree_lstm_output_type %s --attn_delta_loss_weight %f ' \
          ' --transform_every_modal %d --attn_modal_fusion %d --adap_attn_delta_loss_weight %d ' \
          ' --cfg_cfgt_attn_mode %s --cos_ranking_loss_margin %s --use_tanh %d --cfg_cfgt_mlp %d ' \
          ' --transform_attn_out %d --validation_with_metric %d --use_outmlp3 %d --use_val_as_codebase %d' \
          ' --data_query_file %s --remove_wh_word %d > %s' \
          % (config.train_mode, config.modal_type,
             config.dict_code, config.dict_comment, config.data_train_ctg, config.data_val_ctg, config.data_test_ctg,
             config.save_dir, config.embedding_w2v,
             config.pretrain_sl_epoch, config.lr, config.dropout,
             config.output_type, config.n_steps, config.with_anno, config.supernode_mode, config.batch_size,
             config.gpus, config.task_mode, config.retrieval_train_dataset_split_type,
             config.ast_tree_leaves_dict, config.dataset_type,
             config.data_codebase_func_content, config.data_pairs_func_content,
             config.run_id, config.tree_lstm_output_type, config.attn_delta_loss_weight,
             config.transform_every_modal, config.attn_modal_fusion, config.adap_attn_delta_loss_weight,
             config.cfg_cfgt_attn_mode, config.cos_ranking_loss_margin, config.use_tanh, config.cfg_cfgt_mlp,
             config.transform_attn_out, config.validation_with_metric, config.use_outmlp3, config.use_val_as_codebase,
             config.data_query_file, config.remove_wh_word,
             config.log)
    print(run)

    a = os.system(run)
    if a == 0:
        print("finished.")
    else:
        print("failed.")
        sys.exit()


def is_name_in_path(name, modelname):
    if name in modelname:
        index = modelname.index(name)
        if modelname[index:index + len(name)] == name:
            return True
        else:
            return False
    else:
        return False


def test():
    config.train_mode = 'test'

    config.modelname = "model_xent_rd10_c_seq8tree8cfg_e99_bs32_sum_nsteps5_withanno_True_spm1_dr0.0_lr0.0001_adw0.0_tem0_amf0_aadw0_cfgattnmode_sigmoid_scalar_ma0.05_ut0_cm0_tao0_uo0_ds_train.pt"

    config.get_codebase_vec_from_scratch = 1
    config.cfg_cfgt_attn_mode = "sigmoid_scalar"
    config.use_val_as_codebase = 1
    config.gpus = "0"
    config.batch_size = 32

    if config.use_val_as_codebase:
        config.query_file_name = _get_val_query()

    print("config.query_file_name: ", config.query_file_name)
    if "val_ct" in config.query_file_name and ".pt" in config.query_file_name:
        short_query_name = config.query_file_name[
                           config.query_file_name.index("val_ct"):config.query_file_name.index(".pt")]
        short_query_name.strip(".json")
    else:
        short_query_name = config.query_file_name
    print("config.query_file_name: \n", config.query_file_name)
    print("short_query_name: \n", short_query_name)

    config.model_from = os.path.join(config.save_dir, config.modelname)
    config.data_query_file = os.path.join(config.data_dir, train_folder, config.query_file_name)
    modified_model_from = config.model_from.replace("model_xent_", "").replace("aseq0_atree0_acfgt0_ad0_", "").rstrip(
        ".pt")
    config.codebase_vec_path = modified_model_from + "_cbuv_" + str(
        config.use_val_as_codebase) + short_query_name + ".npy"
    config.dataset_index_and_codebase_vec_index_path = modified_model_from + "_icbuv_" + str(
        config.use_val_as_codebase) + short_query_name + ".json"
    config.retrieval_pred_file = modified_model_from + "-" + short_query_name + \
                                 "-uv" + str(config.use_val_as_codebase) + "r" + str(config.remove_wh_word) + "-.re"

    config.retrieval_train_dataset_split_type = \
        config.modelname[config.modelname.index("_ds_") + len("_ds_"):].split(".")[0].split("_")[:2]
    if config.retrieval_train_dataset_split_type == ["train", "val"]:
        config.retrieval_train_dataset_split_type = "train_val"
    elif config.retrieval_train_dataset_split_type[0] == "train":
        config.retrieval_train_dataset_split_type = "train"

    config.modal_type = config.modelname[
                        config.modelname.index(config.modelname.split("_")[4]):config.modelname.index("_e")]
    if "nsteps" in config.modelname:
        output_type_tmp1 = config.modelname[config.modelname.index("bs"):config.modelname.index("nsteps") - 1]
        config.output_type = output_type_tmp1[output_type_tmp1.index("_") + 1:]
        config.n_steps = int(config.modelname[config.modelname.index("nsteps") + len("nsteps"):].split("_")[0])

        config.transform_every_modal = int(
            config.modelname[config.modelname.index("tem") + 3:config.modelname.index("tem") + 4])
        config.attn_modal_fusion = int(
            config.modelname[config.modelname.index("amf") + 3:config.modelname.index("amf") + 4])

        with_anno = config.modelname[config.modelname.index("withanno_") + len("withanno_"):].split("_")[0]
    else:
        config.output_type = "sum"
        config.n_steps = 5
        config.adap_attn_delta_loss_weight = 0
        config.attn_delta_loss_weight = 0
        config.transform_every_modal = 0
        config.attn_modal_fusion = 0
        with_anno = "True"

    if with_anno == "True":
        config.with_anno = 1
    elif with_anno == "False":
        config.with_anno = 0
    else:
        assert with_anno.isdigit()
        config.with_anno = int(with_anno)

    if "attn" in config.modal_type:
        config.output_type = "no_reduce"
        config.tree_lstm_output_type = "no_reduce"
    else:
        config.tree_lstm_output_type = "root_node"

    if "ut" in config.modelname:
        config.use_tanh = int(config.modelname[config.modelname.index("ut") + 2:].split("_")[0])
        print("config.use_tanh: ", config.use_tanh)
    else:
        config.use_tanh = 0

    if "cm" in config.modelname:
        config.cfg_cfgt_mlp = int(config.modelname[config.modelname.index("cm") + 2:].split("_")[0])
        print("config.cfg_cfgt_mlp: ", config.cfg_cfgt_mlp)
    else:
        config.cfg_cfgt_mlp = 0

    if "tao" in config.modelname:
        config.transform_attn_out = int(config.modelname[config.modelname.index("tao") + 3:].split("_")[0])
        print("config.transform_attn_out: ", config.transform_attn_out)
    else:
        config.transform_attn_out = 0


    config.use_outmlp3 = 0



    if config.retrieval_train_dataset_split_type == "train_val":
        config.data_train_ctg = os.path.join(config.data_dir, train_folder,
                                             'processed_all_c.retrieval.train_and_val_ct.pt')

    config.log = '%s/log/log.%s-query_file-%s__uv%sr%s_%s%s' % \
                 (config.data_dir,
                  config.modelname.split(".pt")[0].replace("model_xent_", "").replace("aseq0_atree0_acfgt0_ad0_", ""),
                  short_query_name, config.use_val_as_codebase, config.remove_wh_word, month, day)

    run = 'python -u main.py --train_mode %s --modal_type %s --batch_size %s ' \
          '--dict_code %s --dict_comment %s --data_train_ctg %s --data_val_ctg %s --data_test_ctg %s ' \
          '--save_dir %s --embedding_w2v %s --model_from %s ' \
          '--gpus %s ' \
          '--n_steps %d --output_type %s --with_anno %d --task_mode %s ' \
          ' --retrieval_train_dataset_split_type %s --ast_tree_leaves_dict %s --dataset_type %s ' \
          '--data_codebase_func_content %s --data_pairs_func_content %s --data_query_file %s ' \
          '--codebase_vec_path %s --get_codebase_vec_from_scratch %d --use_val_as_codebase %d ' \
          ' --retrieval_pred_file %s --dataset_index_and_codebase_vec_index_path %s ' \
          ' --transform_every_modal %d --attn_modal_fusion %d --cfg_cfgt_attn_mode %s ' \
          ' --tree_lstm_output_type %s --remove_wh_word %d --use_tanh %d --cfg_cfgt_mlp %d' \
          ' --transform_attn_out %d --use_outmlp3 %d  > %s' \
          % (config.train_mode, config.modal_type, config.batch_size,
             config.dict_code, config.dict_comment, config.data_train_ctg, config.data_val_ctg, config.data_test_ctg,
             config.save_dir, config.embedding_w2v, config.model_from,
             str(config.gpus),
             config.n_steps, config.output_type, config.with_anno, config.task_mode,
             config.retrieval_train_dataset_split_type, config.ast_tree_leaves_dict, config.dataset_type,
             config.data_codebase_func_content, config.data_pairs_func_content, config.data_query_file,
             config.codebase_vec_path, config.get_codebase_vec_from_scratch, config.use_val_as_codebase,
             config.retrieval_pred_file, config.dataset_index_and_codebase_vec_index_path,
             config.transform_every_modal, config.attn_modal_fusion, config.cfg_cfgt_attn_mode,
             config.tree_lstm_output_type, config.remove_wh_word, config.use_tanh, config.cfg_cfgt_mlp,
             config.transform_attn_out, config.use_outmlp3,
             config.log)

    print(run)

    a = os.system(run)
    if a == 0:
        print("finished.")
    else:
        print("failed.")
        sys.exit()


def _get_val_query():
    import torch
    print("config.data_val_ctg:\n ", config.data_val_ctg)
    ct = torch.load(config.data_val_ctg)
    comment = ct['comment']
    code = ct["code"]
    key_list = ct['key_list']
    if config.data_val_ctg.split("/")[-1] == "processed_all_c.retrieval.val_ct.pt":
        func_name = "debug_func_content_retrieval.val_ct.txt"
        query_name = "debug_c_query_retrieval.val_ct.txt"

    func_path = os.path.join(config.data_dir, train_folder, func_name)
    query_path = os.path.join(config.data_dir, train_folder, query_name)
    if (not os.path.exists(func_path)) or (not os.path.exists(query_path)):
        with open(func_path, "w",
                  encoding="utf-8") as f_func:
            with open(query_path, "w",
                      encoding="utf-8") as f_comment:
                for i in range(len(comment)):
                    f_comment.write(" ".join(comment[i]) + '\n')
                    f_func.write(
                        "i:" + str(i) + "======idx_to_query_item:" + str(key_list[i]) + "\n" + " ".join(code[i]) + '\n')

    return query_name


if config.run_func == 'train':
    train()

if config.run_func == 'test':
    test()

if config.run_func == "get_val_query":
    _get_val_query()
