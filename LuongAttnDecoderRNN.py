import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU
from ECMWrapper import *
from ECMGRU import *

from helper import *

class LuongAttnDecoderRNN(tf.Module):
    def __init__(self,embedding,static_emotion_embedding,emotion_embedding, hidden_size, output_size,device,ememory=None, n_layers=1, dropout=0.1,num_emotions = 7,batch_size = 64):
        super(LuongAttnDecoderRNN, self).__init__()
        # Keep for reference
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.dropout = dropout
        self.num_emotions = num_emotions
        self.device = device
        # Define layers
        self.embedding = embedding
        # define emotion embedding
        self.emotion_cat_embedding = static_emotion_embedding
        self.emotion_embedding = emotion_embedding # for internal memory
        self.embedding_dropout = tf.nn.Dropout(dropout)
        #self.emotion_embedding_dropout = nn.Dropout(dropout)
        # dimension
        #self.gru = nn.GRU(hidden_size + hidden_size + hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.gru = ECMGRU(emo_size=hidden_size,hidden_size=hidden_size,static_emo_size=hidden_size,n_layers = n_layers)
        # using in Luong et al. attention mechanism.
        self.internal_memory = ECMWrapper(hidden_size,hidden_size,
                                          hidden_size,self.num_emotions,
                                          self.embedding,self.emotion_embedding,self.gru,device)
        # read external from outside
        self.external_memory = ememory
        # generic output linear layer
        self.generic_word_output_layer = Dense(output_size, input_shape=self.hidden_size)
        # emotional output linear layer 
        self.emotion_word_output_layer = Dense(output_size, input_shape=self.hidden_size)
        # emotional gate/ choice layer
        self.alpha_layer = Dense(1, input_shape=hidden_size)
        # Luong eq 5 layer
        self.concat = Dense(hidden_size, input_shape=hidden_size * 2)
    def forward(self, input_step, input_static_emotion, input_step_emotion, last_hidden
                ,input_context, encoder_outputs,last_rnn_output = None):
        '''
        Decoder with external memory.
        
        '''
        if not input_step_emotion.dtype.is_floating:
            input_step_emotion = self.emotion_embedding(input_step_emotion) # float number for internal memory
        input_static_emotion = self.emotion_cat_embedding(input_static_emotion)
        rnn_output, hidden, new_M_emo,context = self.internal_memory(input_step,last_rnn_output,
                                                                     input_static_emotion,
                                                                     input_step_emotion,
                                                                     input_context,last_hidden,
                                                                     encoder_outputs)
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        concat_input = tf.concat((rnn_output, context), -1)
        concat_output = self.concat(concat_input)
        # concat_output = rnn_output
        # this part is not using inside ECM (?)
        if self.external_memory is not None:
            # Project hidden output to distribution.
            generic_output = self.generic_word_output_layer(concat_output)
            emotion_output = self.emotion_word_output_layer(concat_output)
            generic_output = generic_output.squeeze(0)
            emotion_output = emotion_output.squeeze(0)
            # external memory gate
            g = tf.math.sigmoid(self.alpha_layer(concat_output))
            output_g = tf.nn.softmax(generic_output,dim = 1) * (1 - g)
            output_e = tf.nn.softmax(emotion_output,dim = 1) * g
            output = output_g + output_e # output distribution
            output = output.squeeze(0)
            g = tf.concat([(1 - g),g],dim = -1) # gate distribution
            g = g.squeeze(0)
        else:
            # Predict next word using Luong eq. 6
            output = self.out(concat_output).squeeze(0)
            # generic output
            output = tf.nn.softmax(output, 1)
            output = output.squeeze(0)
            g = None
        # Return output and final hidden state
        return output, hidden, new_M_emo, context,concat_output,g


