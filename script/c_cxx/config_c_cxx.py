# coding=utf-8
import json
import os

repatt = "^\s*[\w_][\w\d_]*\s*.*\s*[\w_][\w\d_]*\s*\(.*\)\s*$"

remove_fake_funcname = ["for", "if", "do", "while", "switch"]

path_nltk_json = os.path.dirname(os.path.abspath(__file__)) + "/data"  # nltk_name_word.json 位置

data_folder = "code-retrieval-2"
datasetpath = "/data/wanyao/work/ds/codedata"
results_path = datasetpath + "/processed"

flag_filter_func_by_asm = True

first_sent_end_maohao_before_num_words_least = 5

num_words_for_com_star_gang = 15
com_satr_gang_thresh = 50

max_lines_comments = 1000

num_words_comments = [2, 30]

max_num_not_en = 3

path_nltk_json_open = os.path.join(path_nltk_json, "nltk_name_word.json")
with open(path_nltk_json_open, 'r', encoding='utf-8') as json_file:
    aa = json_file.readlines()[0]

    dictaa = json.loads(aa)

    namelist = dictaa["name"]
    enwordlist = dictaa["word"]
