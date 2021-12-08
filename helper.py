import tensorflow as tf
import os
import unicodedata
import re
import itertools
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
USE_CUDA = tf.test.is_gpu_available(cuda_only=True)
device = tf.device("cuda" if USE_CUDA else "cpu")

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
MAX_LENGTH = 10  # Maximum sentence length to consider
MIN_COUNT = 3    # Minimum word count threshold for trimming
save_dir = os.path.join("data", "save")
emo_dict = { 0: 'neutral', 1: 'joy', 2: 'anger', 
            3: 'sadness',4:'fear'}
emo2idx = {value:key for key,value in emo_dict.items()}

model_name = 'emotion_model'
corpus_name = 'ECM10_words_GRU_DailyDialogue'
hidden_size = 500
encoder_n_layers = 4
decoder_n_layers = 4
dropout = 0.2
batch_size = 64
# number of emotion
num_emotions = 7

clip = 50
teacher_forcing_ratio = 0.1
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 20000
print_every = 20
save_every = 100

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
    lengths = tf.Variable([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = tf.Variable(padList, dtype=tf.int64)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = tf.Variable(mask, dtype=tf.uint8)
    padVar = tf.Variable(padList, dtype=tf.int64)
    return padVar, mask, max_target_len
# return an emotion tensor
def emotion_tensor(input_list):
    return tf.Variable(input_list)

    # Returns all items for a given batch of pairs
def batch2TrainData(voc, index, pairs, pairs_emotion):
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

def sentenceFromIdx(idx,voc):
    output = []
    for num,i in enumerate(idx):
        if num > 0 and idx[num] == idx[num - 1] and i == 2:
            continue
        output.append(voc.index2word[i])
        
    return ' '.join(output)