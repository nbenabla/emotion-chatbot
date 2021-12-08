import tensorflow as tf
from tf.keras.layers import Bidirectional, Dense, GRU
import numpy as np

from helper import *

class EncoderRNN(tf.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        for i in range(n_layers):
            self.gru = tf.keras.layers.Bidirectional(GRU(hidden_size, dropout=(0 if n_layers == 1 else dropout), return_sequences=True))

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = tf.keras.preprocessing.sequence.pad_sequences(embedded, maxlen=input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = tf.keras.preprocessing.sequence.pad_sequences(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


# Luong attention layer
class Attn(tf.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = Dense(hidden_size,input_shape=self.hidden_size)
        elif self.method == 'concat':
            self.attn = Dense(hidden_size,input_shape=self.hidden_size * 2)
            self.v = tf.Variable(tf.Tensor(hidden_size, dtype=np.float32))

    def dot_score(self, hidden, encoder_output):
        return tf.reduce_sum(hidden * encoder_output, axis=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return tf.reduce_sum(hidden * energy, axis=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(tf.concat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return tf.reduce_sum(self.v * energy, axis=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = tf.transpose(attn_energies)

        # Return the softmax normalized probability scores (with added dimension)
        return tf.nn.softmax(attn_energies, axis=1).expand_dims(1)  