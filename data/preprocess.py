import json
import pandas as pd
import numpy as np
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf


df = pd.read_csv("train_subset_emotion.csv", error_bad_lines=False, warn_bad_lines=False) 

# if (utterance contains "hit:") delete row
df.drop(df[df['utterance'].str.contains("hit:")].index, inplace = True)



# emo=["neutral", "joy", "anger", "sadness", "fear"]

# df["response_context"] = random.choices(emo, k=2854)
# df["context"] = random.choices(emo, k=2854)

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

# emotions = ["sentimental", "afraid", "proud", "faithful", "terrified", "angry", "sad", "joyful", "prepared" , "embarrased", "annoyed", "lonely", 
# "ashamed", "guilty", "surprised", "furious", "disgusted", "hopeful", "confident", "excited", "nostalgic", "grateful", "anticipating", "jealous",
# "impressed", "caring", "devastated", "apprehensive", "disappointed"]


# emo_dict = { 0: 'neutral', 1: 'joy', 2: 'anger', 
#             3: 'sadness',4:'fear'}

# emo2idx = {value:key for key,value in emo_dict.items()}


emo_dict = {0:"sentimental", 1:"afraid", 2:"proud", 3:"faithful", 4:"terrified", 5:"angry", 6:"sad", 7:"joyful", 8:"prepared" , 9:"embarrased", 10:"annoyed", 11:"lonely", 
12:"ashamed", 13:"guilty", 14:"surprised", 15:"furious", 16:"disgusted", 17:"hopeful", 18:"confident", 19:"excited", 20:"nostalgic", 21:"grateful", 22:"anticipating", 23:"jealous",
24:"impressed", 25:"caring", 26:"devastated", 27:"apprehensive", 28:"disappointed", 29:"anxious", 30:"embarrassed", 31:"content",
32:"content", 33:"trusting"}

emo2idx = {v: k for k, v in emo_dict.items()}

lst = parsed['data']
output = []
for i in lst:
    prompt = i[0].split(", ")
    prompt[1] = emo2idx[prompt[1]]
    responses = []
    for j in i[1]:
        resp = j.split(", ")
        resp[1] = emo2idx[resp[1]]
        responses.append(resp)
    lst = [prompt, responses]
    output.append(lst)


pairs =[]
pairs_emotion=[]


# pad short sentences, add tokens, trim sentences of less than 28 words
word_len = 100
vocab = []
for i in output:
    question = i[0][0]
    question.replace("'", " ")
    if (len(question) > 1):
    # only get first 10 words, get rid of rest
        split1 = question.split(" ")[:word_len]
        if len(split1) < word_len:
            for x in range(word_len-len(split1)):
                split1.append("PAD")
        vocab.append(["SOS"] + split1 + ["EOS"])
        question_emo=i[0][1]
        answer = i[1][0][0]
        answer.replace("'", " ")
        split2 = answer.split(" ")[:word_len]
        if len(split2) < word_len:
            for y in range(word_len-len(split2)):
                split2.append("PAD")
        vocab.append(["SOS"] + split2 + ["EOS"])
        ans_emo=i[1][0][1]
        pairs.append([question,answer])
        pairs_emotion.append([question_emo, ans_emo])
        for j in range(1,len(i[1])):
            if j<len(i[1]) -1:
                pairs.append([i[1][j][0], i[1][j+1][0]])
                split3 = i[1][j][0].replace("'", " ").split(" ")[:word_len]
                split4 = i[1][j+1][0].replace("'", " ").split(" ")[:word_len]
                if len(split3) < word_len:
                    for w in range(word_len-len(split3)):
                        split3.append("PAD")
                if len(split4) < word_len:
                    for z in range(word_len-len(split4)):
                        split4.append("PAD")
                vocab.append(["SOS"] + split3 + ["EOS"])
                vocab.append(["SOS"] + split4 + ["EOS"])
                pairs_emotion.append([i[1][j][1], i[1][j+1][1]])


word2index = {}
index2word = {}
for i, sentence in enumerate(vocab):
    for word in sentence:
        word = word.replace("_comma_", "").replace("!", "").replace("?", "").replace(".", "").replace("..", "").replace("...", "").lower().strip()
        word = ''.join([k for k in word if not k.isdigit()]).replace(",", "").replace("$", "").replace("-", "")
        word2index[word] = i
        index2word[i] = word



for i in pairs:
    for word in range(len(i)):
        i[word] = i[word].replace("_comma_", "").replace("!", "").replace("?", "").replace(".", "").replace("..", "").replace("...", "").lower().strip()
        i[word] = ''.join([k for k in i[word] if not k.isdigit()]).replace(",", "").replace("$", "").replace("-", "")
        
class Voc:
    def __init__(self):
        self.name = 'train'
        self.word2index = word2index
        self.index2word = index2word
        self.num_words = len(word2index)
    

def get_data():
    return pairs, pairs_emotion, Voc()

get_data()


import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import time
import json
import numpy as np

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

def indexesFromSentence(voc, sentence):
    # print(sentence)
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = tf.convert_to_tensor([len(indexes) for indexes in indexes_batch], dtype=tf.int32)
    padList = zeroPadding(indexes_batch)
    padVar = tf.convert_to_tensor(padList, dtype=tf.int32)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    mask = tf.convert_to_tensor(mask.numpy())
    padVar = tf.convert_to_tensor(padList, dtype=tf.int32)
    return padVar, mask, max_target_len
# return an emotion tensor
def emotion_tensor(input_list):
    # return torch.LongTensor(input_list)
    return tf.convert_to_tensor(input_list, dtype=tf.int32)

def batch2TrainData(voc, index,pairs,pairs_emotion):
    pair_batch = [pairs[idx] for idx in index]
    pair_batch_emotions =[pairs_emotion[idx] for idx in index]
    keys = [len(x[0].split(' ')) for x in pair_batch]
    sorted_index = np.argsort(keys)[::-1]
    input_batch, output_batch = [], []
    input_batch_emotion, output_batch_emotion = [],[]
    for idx in sorted_index:
        input_batch.append(pair_batch[idx][0])
        input_batch_emotion.append(pair_batch_emotions[idx][0])
        output_batch.append(pair_batch[idx][1])
        output_batch_emotion.append(pair_batch_emotions[idx][1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    input_batch_emotion = emotion_tensor(input_batch_emotion)
    output_batch_emotion = emotion_tensor(output_batch_emotion)
    return inp,input_batch_emotion, lengths, output,output_batch_emotion, mask, max_target_len


small_batch_size = 5
batches = batch2TrainData(Voc(), [random.choice(list(range(len(pairs)))) for _ in range(small_batch_size)],pairs,pairs_emotion)


    