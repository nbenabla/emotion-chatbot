import tensorflow as tf
from tensorflow.keras.layers import Dense

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
USE_CUDA = tf.test.is_gpu_available(cuda_only=True)
device = tf.device("cuda" if USE_CUDA else "cpu")

SOS_token = 1  # Start-of-sentence token
hidden_size = 500


class GreedySearchDecoder(tf.Module):
    '''
    Greedy search decode
    '''

    def __init__(self, encoder, decoder, num_word=None):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, target_emotions, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = tf.ones((1, 1), device=device, dtype=tf.int64) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = tf.zeros([0], device=device, dtype=tf.int64)
        all_scores = tf.zeros([0], device=device)
        # Set initial context value,last_rnn_output, internal_memory
        context_input = tf.zeros((1, hidden_size), dtype=tf.float32, device=self.decoder.device) 
        context_input = context_input.to(device)  
        rnn_output = None
        # keep a copy of emotional category for static emotion embedding
        static_emotion = target_emotions
        static_emotion = static_emotion.to(device)  
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden, target_emotions, context_input, rnn_output, g = self.decoder(
                decoder_input, static_emotion, target_emotions, decoder_hidden,
                context_input, encoder_outputs, rnn_output
            )
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = tf.reduce_max(decoder_output, reduction_indices=[1])
            # Record token and score
            all_tokens = tf.concat([all_tokens, decoder_input], 0)
            all_scores = tf.concat([all_scores, decoder_scores], 0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = tf.expand_dims(decoder_input, axis=0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores
