import json
import pandas as pd
import numpy as np
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
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
from data_utils import *



df = pd.read_csv("train_subset_emotion.csv", error_bad_lines=False, warn_bad_lines=False) 

# if (utterance contains "hit:") delete row
df.drop(df[df['utterance'].str.contains("hit:")].index, inplace = True)



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


emo_dict = { 0: 'neutral', 1: 'joy', 2: 'anger', 
            3: 'sadness',4:'fear'}

emo2idx = {value:key for key,value in emo_dict.items()}


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
for i in output:
    question = i[0][0]
    if (len(question) > 1):
    # only get first 10 words, get rid of rest
        split1 = question.split(" ")[:word_len]
        question_emo=i[0][1]
        answer = i[1][0][0]
        split2 = answer.split(" ")[:word_len]
        ans_emo=i[1][0][1]
        pairs.append([question,answer])
        pairs_emotion.append([question_emo, ans_emo])
        for j in range(1,len(i[1])):
            if j<len(i[1]) -1:
                pairs.append([i[1][j][0], i[1][j+1][0]])
                split3 = i[1][j][0].split(" ")[:word_len]
                split4 = i[1][j+1][0].split(" ")[:word_len]
                pairs_emotion.append([i[1][j][1], i[1][j+1][1]])
                
for i in pairs:
    for word in range(len(i)):
        i[word] = unicodeToAscii(i[word])
        i[word] = normalizeString(i[word])
            

class Voc:
    def __init__(self, name,min_count,max_length):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD
        self.min_count = min_count
        self.max_length = max_length
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word]= self.word2count.get(word,0) + 1

    # Remove words below a certain count threshold
    def trim(self):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= self.min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {'PAD':0, 'SOS':1,'EOS':2}
        self.word2count = {'PAD':self.min_count,'SOS':self.min_count,'EOS':self.min_count}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word) 
            
            

def readVocs(min_count,max_length):
    print("Reading lines...")

    voc = Voc('my_voc',min_count,max_length)
    return voc, pairs, pairs_emotion


def filterPair(p,max_length,min_length):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length \
           and len(p[0].split(' ')) >= min_length and len(p[1].split(' ')) >= min_length

# Filter pairs using filterPair condition
def filterPairs(pairs,pairs_emotion,max_length,min_length):
    keep_pairs,keep_pairs_emotion = [], []
    for pair,pair_emotion in zip(pairs,pairs_emotion):
        if filterPair(pair,max_length,min_length):
            keep_pairs.append(pair)
            keep_pairs_emotion.append(pair_emotion)
    return keep_pairs,keep_pairs_emotion



# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(min_count,max_length,drop_num):
    print("Start preparing training data ...")
    voc, pairs, pairs_emotion = readVocs(min_count,max_length)
    # flatten the pairs of sentences
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs,pairs_emotion = filterPairs(pairs,pairs_emotion,voc.max_length,1)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print('Emotions left {}'.format(len(pairs_emotion)))
    print('Under sample one categories:')
    pairs, pairs_emotion = under_sample(pairs, pairs_emotion, 0, drop_num)
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs, pairs_emotion



def trimRareWords(voc, pairs,pairs_emotion, min_count):
    # Trim words used under the MIN_COUNT from the voc

    voc.trim()
    # Filter out pairs with trimmed words
    keep_pairs = []
    keep_emotions = []
    for pair,pair_emotion in zip(pairs,pairs_emotion):
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)
            keep_emotions.append(pair_emotion)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs,keep_emotions


def indexesFromSentence(voc, sentence):
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
    return tf.convert_to_tensor(input_list, dtype=tf.int32)
# Returns all items for a given batch of pairs
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


def get_data(min_count = 1,max_length= 10,drop_num = 30000):
    voc, pairs, pairs_emotion = loadPrepareData(min_count,max_length,drop_num)
    # Print some pairs to validate
    # Trim voc and pairs
    pairs, pairs_emotion = trimRareWords(voc, pairs, pairs_emotion, min_count)
    '''
    dataset = []
    for idx, qa in enumerate(pairs):
        question, response = qa
        emotions = pairs_emotion[idx]
        question_emotion, response_emotion = emotions
        question = [question] + [question_emotion] * 2
        response = [[response] + [response_emotion] * 2]
        dataset.append([question, response])

    train = dataset
    '''
    return voc,pairs,pairs_emotion

def under_sample(pairs,pairs_emotion,emotions_cate = 0,drop_num = 30000):
    '''
    Under sample the response that has emotion category 0 (no emotion)
    :param pairs:
    :param pairs_emotion:
    :param emotions_cate:
    :param drop_num:
    :return:
    '''
    drop_no_emotion = True
    keep_pairs = []
    keep_pairs_emotion = []
    drop_count = 0
    for conversation,emotion in zip(pairs,pairs_emotion):
        post,response = conversation
        if emotion[1] == emotions_cate and drop_no_emotion:
            drop_count += 1
            if drop_count == drop_num:
                drop_no_emotion = False
            continue
        else:
            keep_pairs.append(conversation)
            keep_pairs_emotion.append(emotion)
    print('Under sample non-emotion to {} samples'.format(len(keep_pairs)))
    return keep_pairs,keep_pairs_emotion

voc, pairs, pairs_emotion=get_data()
print(len(pairs_emotion))


test_batch = batch2TrainData(voc,list(range(50)),pairs[-50:],pairs_emotion[-50:])
pairs = pairs[:-50]
pairs_emotion = pairs_emotion[:-50]
test_pairs = pairs[-50:]
test_pairs_emotion = pairs_emotion[-50:]

def get_all():
    return test_batch, pairs, pairs_emotion, test_pairs, test_pairs_emotion