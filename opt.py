import argparse


def get_opt():
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('--train_mode', default='pretrain_sl', help='What should train.py do?')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--lower', default=True, type=bool, help='lowercase data')

    parser.add_argument('--dict_code', required=False, help='Path to the *-train.pt file from preprocess.py')
    parser.add_argument('--dict_comment', required=False, help='Path to the *-train.pt file from preprocess.py')
    parser.add_argument('--data_train_ctg', required=False, help='Path to the *-train.pt file from preprocess.py')
    parser.add_argument('--data_val_ctg', required=False, help='Path to the *-train.pt file from preprocess.py')
    parser.add_argument('--data_test_ctg', required=False, help='Path to the *-train.pt file from preprocess.py')
    parser.add_argument('--data_train_adjmat', help='Path to the *.train_cfg_adjmat.npy file from preprocess.py')
    parser.add_argument('--data_train_anno', help='Path to the *.train_cfg_anno.npy file from preprocess.py')
    parser.add_argument('--data_val_adjmat', help='Path to the *.train_cfg_adjmat.npy file from preprocess.py')
    parser.add_argument('--data_val_anno', help='Path to the *.train_cfg_anno.npy file from preprocess.py')
    parser.add_argument('--data_test_adjmat', help='Path to the *.train_cfg_adjmat.npy file from preprocess.py')
    parser.add_argument('--data_test_anno', help='Path to the *.train_cfg_anno.npy file from preprocess.py')
    parser.add_argument('--save_dir', required=True, help='Directory to save models')
    parser.add_argument("--model_from", help="Path to load a pretrained model.")

    parser.add_argument("--pred_file", help="Path to load a pred_file.")
    parser.add_argument('--embedding_w2v', required=False, help='Path to the *-embedding_w2v file from preprocess.py')

    parser.add_argument('--rnn_type', type=str, default='LSTM',
                        help='type of  recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')

    parser.add_argument('--nlayers', type=int, default=1, help='Number of layers in the LSTM encoder/decoder')

    parser.add_argument('--ninp', type=int, default=300, help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=512, help='humber of hidden units per layer')

    parser.add_argument('--brnn', action='store_true', help='Use a bidirectional encoder')
    parser.add_argument('--has_attn', type=int, default=0, help="""attn model or not""")

    parser.add_argument('--state_dim', type=int, default=512, help='GGNN hidden state size')
    parser.add_argument('--n_steps', type=int, default=5, help='propogation steps number of GGNN')
    parser.add_argument("--output_type", type=str, default="outmlp3",
                        help='one of ["avepool","maxpool","supernode","outmlp","outmlp3"]')

    parser.add_argument("--with_anno", type=bool, default=False, help="为True时，最后一个time step后\
        每个节点的hidden state和对应的annotation进行concatenate，再通过MLP转换维度到state_dim,最后再经过self.out; False时,最后一个time step 后每个节点的hidden state直接经过self.out")

    parser.add_argument('--tree_lstm_cell_type', default="nary", help='nary or childsum')
    parser.add_argument('--ast_tree_leaves_dict', default=None, help='dict for c dataset')
    parser.add_argument('--dataset_type', default="python", help='Type of datasetc [c|python]')
    parser.add_argument('--supernode_mode', help='1~12', type=int, default=1)

    parser.add_argument('--modal_type', default='code', help="Type of encoder to use. Options are [text|code].")
    parser.add_argument('--batch_size', type=int, default=32, help='Maximum batch size')

    parser.add_argument("--pretrain_sl_epoch", type=int, default=15, help="Epoch to supervised training.")

    parser.add_argument('--param_init', type=float, default=0.1,
                        help="""Parameters are initialized over uniform distribution with support (-param_init, param_init). Use 0 to not use initialization""")
    parser.add_argument('--optim', default='adam', help="Optimization method. [sgd|adagrad|adadelta|adam]")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")

    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability; applied between LSTM stacks.')

    parser.add_argument('--gpus', default="0", type=str, help="Use CUDA on the listed devices.")

    parser.add_argument('--log_interval', type=int, default=5, help="Print stats at this interval.")
    parser.add_argument('--seed', type=int, default=3435,
                        help="""Random seed used for the experiments reproducibility.""")

    parser.add_argument('--task_mode', default='code-sum-mm', help='code-sum-mm or code-ir-mm')
    parser.add_argument('--cos_ranking_loss_margin', type=float, default=0.05, required=False,
                        help='margin for CosRankingLoss')
    parser.add_argument('--retrieval_train_dataset_split_type', default="train", required=False,
                        help='train_val(train data and val data all for train) or train(train data for train, val data for validation)')
    parser.add_argument('--data_codebase_func_content', required=False)
    parser.add_argument('--data_pairs_func_content', required=False)
    parser.add_argument('--data_query_file', required=False)
    parser.add_argument('--codebase_vec_path', required=False)
    parser.add_argument('--get_codebase_vec_from_scratch', required=False, type=int)
    parser.add_argument('--use_val_as_codebase', required=False, type=int)
    parser.add_argument('--retrieval_pred_file', required=False, type=str)
    parser.add_argument('--dataset_index_and_codebase_vec_index_path', required=False, type=str)
    parser.add_argument('--run_id', required=False, type=str)

    parser.add_argument('--tree_lstm_output_type', default="root_node", type=str)
    parser.add_argument('--attn_delta_loss_weight', required=False, default="1.0", type=float)

    parser.add_argument('--transform_every_modal', default=0, type=int)
    parser.add_argument('--attn_modal_fusion', default=0, type=int)
    parser.add_argument('--adap_attn_delta_loss_weight', required=False, type=int)
    parser.add_argument('--cfg_cfgt_attn_mode', type=str)
    parser.add_argument('--remove_wh_word', required=False, type=int)
    parser.add_argument('--use_tanh', default=0, type=int)
    parser.add_argument('--cfg_cfgt_mlp', default=0, type=int)
    parser.add_argument('--transform_attn_out', default=0, type=int)
    parser.add_argument('--validation_with_metric', default=0, type=int)
    parser.add_argument('--use_outmlp3', default=0, type=int)

    parser.add_argument('--init_type', type=str, default="xulu")
    parser.add_argument('--init_normal_std', type=float, default=1e-4)

    opt = parser.parse_args()
    return opt
