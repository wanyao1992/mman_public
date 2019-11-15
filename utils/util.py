import collections
import copy
import json

import chardet

import data.Constants
import data.Constants
import data.Constants as Constants








def detectCode(path):
    with open(path, 'rb') as file:
        data = file.read()
        dicts = chardet.detect(data)
    return dicts["encoding"]


def save_json(output, output_path):
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(output, json_file, ensure_ascii=False)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as json_file:
        aa = json_file.readlines()[0]
        output = json.loads(aa)
    return output



def init_xavier_linear(opt, linear, init_bias=True, gain=1):
    import torch
    torch.nn.init.xavier_uniform_(linear.weight, gain)
    if init_bias:
        if linear.bias is not None:
            linear.bias.data.normal_(std=opt.init_normal_std)




