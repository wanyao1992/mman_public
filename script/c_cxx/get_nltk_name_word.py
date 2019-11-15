import os

import nltk
import  json

path_nltk_json = "../data"
if not os.path.exists(path_nltk_json):
    os.makedirs(path_nltk_json)
path_nltk_json_open = os.path.join(path_nltk_json, "nltk_name_word.json")

ln = list(set(nltk.corpus.names.words()))
lnlower = [w.lower() for w in ln]

leng = list(set(nltk.corpus.words.words('en')))
lnenglower = [we.lower() for we in leng]

dicta = {'name': lnlower, 'word': lnenglower}

with open(path_nltk_json_open, 'w', encoding='utf-8') as json_file:
    json.dump(dicta, json_file, ensure_ascii=False)

with open(path_nltk_json_open, 'r', encoding='utf-8') as json_file:
    aa = json_file.readlines()[0]

    dictaa = json.loads(aa)

    namelist = dictaa["name"]
    enwordlist = dictaa["word"]
    print("len(namelist): ", len(namelist))
    print("namelist[:5]: ", namelist[:5])
    print("len(enwordlist): ", len(enwordlist))
    print("enwordlist[:5]: ", enwordlist[:5])
