# coding=utf-8
import json



def load_json(path):
    with open(path, 'r', encoding='utf-8') as json_file:
        aa = json_file.readlines()[0]
        output = json.loads(aa)
    return output


def save_json(path, to_save):
    with open(path, 'w', encoding='utf-8') as json_file:
        json.dump(to_save, json_file, ensure_ascii=False)
