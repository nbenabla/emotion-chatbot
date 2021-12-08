import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU
from ECMWrapper import *
import numpy as np
import os


class ECMGRU(tf.keras.Model):
    def __init__(self,hidden_size,static_emo_size,emo_size,n_layers = 1):
        '''
        Single layer GRU.
        '''
        super(ECMGRU,self).__init__()
        # first layer of special GRU
        self.hidden_size = hidden_size
        # these three linear layer compute output from emotion/internal memory 
        self.emotion_u = Dense(hidden_size, input_shape=(static_emo_size + emo_size,))
        self.emotion_r = Dense(hidden_size, input_shape=(static_emo_size + emo_size,))
        self.emotion_c = Dense(hidden_size, input_shape=(static_emo_size + emo_size,))
        # these two are generic GRU output that takes [step_input, last_hidden]
        self.generic_u = Dense(hidden_size, input_shape=(hidden_size + hidden_size + hidden_size + emo_size,))
        self.generic_r = Dense(hidden_size, input_shape=(hidden_size + hidden_size + hidden_size + emo_size,))
        
        # gate value computation
        self.generic_c = Dense(hidden_size, input_shape=(hidden_size + hidden_size + hidden_size + emo_size,))
        self.n_layers = n_layers
        # starting from second layer, using the normal GRU
        # self._cell = tf.keras.layers.GRU(hidden_size, num_layers = n_layers - 1)

        for i in range(n_layers):
            self._cell = tf.keras.layers.GRU(hidden_size, return_sequences=True)
    def call(self,step_input,last_hidden,emotion,internal_memory):
        '''
        step_input: X
        last_hidden: Hidden value from GRU
        emotion: static emotion embedding vector
        internal_memory: decayed emotion embedding vector
        '''
        # compute based on the first layer
        hidden_0 = tf.expand_dims(last_hidden[0], axis= 0)
        internal_memory = tf.squeeze(internal_memory, axis= 0)

        emotion_input = tf.concat([emotion,internal_memory],-1)
        # compute emotion gate value
        _u = self.emotion_u(emotion_input) # update gate
        _r = self.emotion_r(emotion_input) # reset gate
        _c = self.emotion_c(emotion_input) # reset gate vector
        
        # generic GRU gate value
        X = tf.concat([step_input,hidden_0],-1)
        u = self.generic_u(X)
        r = self.generic_r(X)
        
        # gate for this time stamp
        rt = tf.math.sigmoid(r + _r)
        ut = tf.math.sigmoid(u + _u)
        
        # gate vector
        Xc = tf.concat([step_input, hidden_0 * rt], -1)
        ct = _c + self.generic_c(Xc)
        ct = tf.math.tanh(ct)
        # compute new hidden
        hidden = ut * hidden_0 + (1 - ut) * ct
        
        # if it has second layer
        if self.n_layers > 1:
            gru_hidden = last_hidden[1:] # skip the first layer of input
            gru_input = hidden
            rnn_output, gru_hidden = self._cell(gru_input, gru_hidden)
        hidden = tf.concat([hidden,gru_hidden],0)
        return rnn_output, hidden
        