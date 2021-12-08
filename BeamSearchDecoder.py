import tensorflow as tf
import numpy as np

from helper import *

class BeamSearchDecoder(tf.Module):
    def __init__(self, encoder, decoder,num_word):
        super(BeamSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_word = num_word

    def forward(self, input_seq,target_emotions,input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = tf.ones((1, 1), device=device, dtype=tf.int64) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = tf.zeros([0], device=device, dtype=tf.int64)
        all_words_order = tf.zeros((1, self.num_word), device=device, dtype=tf.int64)
        all_scores = tf.zeros([0], device=device)
        all_scores_array = tf.zeros((1, self.num_word), device=device, dtype=tf.float32)
        # Set initial context value,last_rnn_output, internal_memory
        context_input = tf.zeros((1, hidden_size), dtype=tf.float32) 
        context_input = context_input.to(device)
        rnn_output = None
        # keep a copy of emotional category for static emotion embedding
        static_emotion = target_emotion
        static_emotion = static_emotion.to(device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden, target_emotions, context_input, rnn_output, g = self.decoder(
                decoder_input,static_emotion,target_emotions, decoder_hidden,
                context_input, encoder_outputs, rnn_output
            )
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = tf.reduce_max(decoder_output, reduction_indices=[1])
            decoder_input_order = tf.argsort(decoder_output, axis=1, direction='DESCENDING')
            # Record token and score
            all_tokens = tf.concat([all_tokens, decoder_input], 0)
            all_scores = tf.concat([all_scores, decoder_scores], 0)
            all_scores_array = tf.concat([all_scores_array, decoder_output], 0)
            all_words_order = tf.concat([all_words_order, decoder_input_order], 0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = tf.expand_dims(decoder_input, axis=0)
        # Return collections of word tokens and scores
        sequences = self.beam_search(all_scores_array, 3)
        return sequences

    def beam_search(self, array, k):
        array = array.tolist()
        sequences = [[list(), 1.0]]
        # walk over each step in sequence
        for row in array:
            all_candidates = list()
            # expand each current candidate
            for i in range(len(sequences)):
                seq, score = sequences[i]
                for j in range(len(row)):
                    candidate = [seq + [j], score - np.log(row[j] + 1e-8)]
                    all_candidates.append(candidate)
            # order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup:tup[1])
            # select k best
            sequences = ordered[:k]
        return sequences