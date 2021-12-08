import tensorflow as tf
import os
import numpy as np

from helper import *


def evaluate(encoder, decoder, searcher, voc, sentence, emotions, max_length=MAX_LENGTH, beam_search = False):
    emotions = int(emotions)
    emotions = tf.Variable([emotions], dtype=tf.int64)
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = tf.convert_to_tensor([len(indexes) for indexes in indexes_batch], dtype=np.int32)
    # Transpose dimensions of batch to match models' expectations
    input_batch = tf.transpose(tf.Variable(indexes_batch, dtype=tf.int64), (0,1))
    

    # indexes -> words
    if beam_search:
        sequences = searcher(input_batch, emotions, lengths, max_length)
        decoded_words = beam_decode(sequences,voc)
    else:
        # Decode sentence with searcher
        tokens, scores = searcher(input_batch, emotions, lengths, max_length)
        decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words

def beam_decode(sequences,voc):
    for each in sequences:
        for idxs in each:
            return [voc.index2word[idx] for idx in idxs[:-1]]
    
def evaluateInput(encoder, decoder, searcher, voc, emotion_dict, beam_search):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            for emotion in range(len(emotion_dict)):
                # Check if it is quit case
                if input_sentence == 'q' or input_sentence == 'quit': break
                # Normalize sentence
                input_sentence = normalizeString(input_sentence)
                # Evaluate sentence
                output_words = evaluate(encoder, decoder, searcher, voc, input_sentence,emotion,beam_search=beam_search)
                # Format and print response sentence
                output=[]
                for word in output_words:
                    if word == 'PAD':
                        continue
                    elif word == 'EOS':
                        break
                    else:
                        output.append(word)
                print('Bot({}):'.format(emotion_dict[emotion]), ' '.join(output))

        except KeyError:
            print("Error: Encountered unknown word.")