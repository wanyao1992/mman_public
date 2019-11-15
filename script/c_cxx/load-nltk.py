import json
import os

path_nltk_json = "../data"
path_nltk_json_open = os.path.join(path_nltk_json, "nltk_name_word.json")
with open(path_nltk_json_open, 'r', encoding='utf-8') as json_file:
    aa = json_file.readlines()[0]

    dictaa = json.loads(aa)

    namelist = dictaa["name"]
    enwordlist = dictaa["word"]

import json

with open("./cnt_path.json", 'r', encoding='utf-8') as json_file:
    aa = json_file.readlines()[0]
    dictaa = json.loads(aa)
    bigc = dictaa[".CPP"]

bigc = dictaa[".Cc"]
print(len(bigc))
print(bigc[:5])
