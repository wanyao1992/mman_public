import os
import random

import numpy as np
import torch
import torch.utils.data

from metric.Loss import CosRankingLoss
from opt import get_opt
from trainer.CodeRetrievalEvaluator import CodeRetrievalEvaluator
from trainer.CodeRetrievalTrainer import CodeRetrievalTrainer
from utils.util_data import load_dict, load_data, create_model_code_retrieval


def main():
    print("Start... => main.py PID: %s" % (os.getpid()))
    opt = get_opt()

    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)

    if opt.gpus:
        print("opt.gpus: ", opt.gpus)
        gpu_list = [int(k) for k in opt.gpus.split(",")]
        print("gpu_list: ", gpu_list)
        print("gpu_list[0]: ", gpu_list[0])
        print("type(gpu_list[0]): ", type(gpu_list[0]))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

    all_dict = load_dict(opt)

    if opt.train_mode == "train":
        if opt.validation_with_metric:
            train_dataset, train_dataloader, val_dataset, val_dataloader, \
            test_dataset, test_dataloader, query_dataset, query_dataloader = load_data(opt, all_dict)
        else:
            train_dataset, train_dataloader, val_dataset, val_dataloader = load_data(opt, all_dict)


    elif opt.train_mode == "test":
        test_dataset, test_dataloader, query_dataset, query_dataloader = load_data(opt, all_dict)

    print('Loaded dataset sucessfully.')

    if opt.train_mode == "train":
        model = create_model_code_retrieval(opt, train_dataset, all_dict)

        print('created model..')
        cos_ranking_loss = CosRankingLoss(opt).cuda()
        optim = torch.optim.Adam(model.parameters(), opt.lr)
        print('created criterion and optim')

        if opt.validation_with_metric:

            pretrainer = CodeRetrievalTrainer(model, train_dataset, train_dataloader, val_dataset,
                                              val_dataloader,
                                              all_dict, opt,
                                              metric_data=[test_dataset, test_dataloader, query_dataset,
                                                           query_dataloader])

        else:

            pretrainer = CodeRetrievalTrainer(model, train_dataset, train_dataloader, val_dataset,
                                              val_dataloader,
                                              all_dict, opt)

        pretrainer.train(cos_ranking_loss, optim, opt.pretrain_sl_epoch)

    elif opt.train_mode == "test":
        model = create_model_code_retrieval(opt, test_dataset, all_dict)

        print('created model..')

        evaluator = CodeRetrievalEvaluator(model=model,
                                           dataset_list=[test_dataset, test_dataloader, query_dataset,
                                                         query_dataloader],
                                           flag_for_val=False, all_dict=all_dict, opt=opt)

        evaluator.retrieval(pred_file=opt.retrieval_pred_file)
        if opt.use_val_as_codebase:
            evaluator.eval_retrieval_json_result(pred_file=opt.retrieval_pred_file)


if __name__ == '__main__':
    main()
