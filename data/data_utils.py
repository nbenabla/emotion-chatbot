from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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
'''
CONSTANT
'''
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

def printLines(file, n=5):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def extractSentencePairs(conversations, is_emotions=True):
    qa_pairs = []
    i = 0
    while i < len(conversations) - 1:
        if is_emotions:
            qa_pairs.append([conversations[i], conversations[i + 1]])
        else:
            qa_pairs.append([normalizeString(conversations[i]), normalizeString(conversations[i + 1])])
        i += 1

    return qa_pairs


def loadLines(fileName):
    lines = {}
    with open(fileName, 'r') as f:
        for idx, line in enumerate(f):
            conversations = [i.strip() for i in line.strip().split('__eou__')[:-1]]
            qa_pairs = extractSentencePairs(conversations, is_emotions=False)
            lines[idx] = qa_pairs
    return lines


def loadEmotions(fileName):
    lines = {}
    with open(fileName, 'r') as f:
        for idx, line in enumerate(f):
            emotions = [int(i) for i in line.strip().split(' ')]

            emotions_pair = extractSentencePairs(emotions, is_emotions=True)
            lines[idx] = emotions_pair

    return lines
