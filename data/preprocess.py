import json
import pandas as pd
import numpy as np
df = pd.read_csv("train_subset_emotion.csv", error_bad_lines=False, warn_bad_lines=False) 

df["prompt_with_emotion"] = df["prompt"] + ", " + df["context"]
df["response_with_emotion"] = df["utterance"] + ", " + df["response_context"]
dff = df[["prompt_with_emotion", "response_with_emotion"]]
grouped_df = dff.groupby("prompt_with_emotion")

grouped_lists = grouped_df["response_with_emotion"].apply(list)
grouped_lists = grouped_lists.reset_index()


compression_opts = dict(method=None,
                         archive_name='grouped_train.csv')  
grouped_lists.to_csv('grouped_train.csv', index=False,
          compression=compression_opts) 

df3 = grouped_lists
result = df3.to_json(orient="split")
parsed = json.loads(result)

emotions = ["sentimental", "afraid", "proud", "faithful", "terrified", "angry", "sad", "joyful", "prepared" , "embarrased", "annoyed", "lonely", 
"ashamed", "guilty", "surprised", "furious", "disgusted", "hopeful", "confident", "excited", "nostalgic", "grateful", "anticipating", "jealous",
"impressed", "caring", "devastated", "apprehensive", "disappointed"]

emo_dict = {0:"sentimental", 1:"afraid", 2:"proud", 3:"faithful", 4:"terrified", 5:"angry", 6:"sad", 7:"joyful", 8:"prepared" , 9:"embarrased", 10:"annoyed", 11:"lonely", 
12:"ashamed", 13:"guilty", 14:"surprised", 15:"furious", 16:"disgusted", 17:"hopeful", 18:"confident", 19:"excited", 20:"nostalgic", 21:"grateful", 22:"anticipating", 23:"jealous",
24:"impressed", 25:"caring", 26:"devastated", 27:"apprehensive", 28:"disappointed", 29:"anxious", 30:"embarrassed", 31:"content",
32:"content", 33:"trusting"}

inv_map = {v: k for k, v in emo_dict.items()}

lst = parsed['data']
output = []
for i in lst:
    prompt = i[0].split(", ")
    prompt[1] = inv_map[prompt[1]]
    responses = []
    for j in i[1]:
        resp = j.split(", ")
        resp[1] = inv_map[resp[1]]
        responses.append(resp)
    lst = [prompt, responses]
    output.append(lst)

pairs =[]
pairs_emotion=[]


# pad short sentences, add tokens, trim sentences of less than 28 words
vocab = []
for i in output:
    question = i[0][0]
    if (len(question) > 2):
    # only get first 28 words, get rid of rest
        split1 = question.split(" ")[:28]
        if len(split1) < 28:
            for i in range(28-len(split1)):
                split1.append(" PAD")
        vocab.append("SOS " + split1 + " EOS")
        question_emo=i[0][1]
        answer = i[1][0][0]
        split2 = answer.split(" ")[:28]
        if len(split2) < 28:
            for i in range(28-len(split2)):
                split2.append(" PAD")
        vocab.append("SOS " + split2 + " EOS")
        ans_emo=i[1][0][1]
        pairs.append([question,answer])
        pairs_emotion.append([question_emo, ans_emo])
        for j in range(1,len(i[1])):
            if j<len(i[1]) -1:
                pairs.append([i[1][j][0], i[1][j+1][0]])
                split3 = i[1][j][0].split(" ")[:28]
                split4 = i[1][j+1][0].split(" ")[:28]
                if len(split3) < 28:
                    for i in range(28-len(split3)):
                        split3.append(" PAD")
                if len(split4) < 28:
                    for i in range(28-len(split4)):
                        split4.append(" PAD")
                vocab.append("SOS " + split3 + " EOS")
                vocab.append("SOS " + split4 + " EOS")
                pairs_emotion.append([i[1][j][1], i[1][j+1][1]])



word2index = {}
index2word = {}
for i, word in enumerate(vocab):
    word2index[word] = i
    index2word[i] = word



class Voc:
    def __init__(self):
        self.word2index = word2index
        self.index2word = index2word
    

def get_data():
    return pairs, pairs_emotion, Voc()