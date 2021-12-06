import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np


class ECMWrapper(tf.Module):
    '''
    Internal memory module
    '''
    def __init__(self,hidden_size,state_size,emo_size,num_emotion,embedding,emotion_embedding,gru,device):
        '''
        hidden_size: hidden input dimension
        state_size: state vector size (input a word so hidden size)
        emo_size: emotional embedding size (usually similar to hidden_size)
        num_emotion: number of emotion categories
        '''
        super(ECMWrapper,self).__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.emo_size = emo_size
        self.num_emotion = num_emotion
        self.device = device
        # read gate dimensions (word_embedding + hidden_input + context_input)
        self.read_g = Dense(self.emo_size, input_shape=self.hidden_size + self.hidden_size + self.hidden_size)
        # write gate
        self.write_g = Dense(self.emo_size, input_shape= self.state_size)
        # GRU output input dimensions = state_last + context + emotion emb + internal memory
        self.gru = gru
        self.emotion_embedding = emotion_embedding
        self.embedding = embedding
        # attention layer
        self.attn1 = Dense(self.hidden_size,input_shape=self.hidden_size)
        self.attn2 = Dense(self.hidden_size,input_shape=self.hidden_size)
        self.concat = Dense(1, input_shape=self.hidden_size)
    def forward(self,word_input,decoder_output,static_emotion_input,emotion_input,context_input,last_hidden,memory):
        '''
        Last hidden == prev_cell_state
        last word embedding = word_input
        last hidden input = h
        last_rnn_output = logits before softmax
        memory = encoder_outputs
        emotion_input = internal memory
        static_emotion_input = emotion embedding value
        '''
        # get embedding of input word and emotion
        if decoder_output is None:
            decoder_output = tf.zeros(word_input.shape[1],self.hidden_size,dtype=np.float32,device = self.device)
            decoder_output = tf.expand_dims(decoder_output, axis=0)
            context_input = self._compute_context(decoder_output,memory)
        last_word_embedding = self.embedding(word_input)
        read_inputs = tf.concat((last_word_embedding,decoder_output,context_input), -1)
        # compute read input
        # write to emotion embedding
        emotion_input = self._read_internal_memory(read_inputs,emotion_input)
        # pass everything to GRU
        # decoder_output: logits from last rnn unit
        X = tf.concat([context_input, last_word_embedding, emotion_input],-1)
        rnn_output, hidden = self.gru(X,last_hidden,static_emotion_input,emotion_input)
        # write input
        # update states
        # write to emotion embedding
        new_M_emo = self._write_internal_memory(emotion_input,rnn_output) # new emotion_input
        new_context = self._compute_context(rnn_output,memory)
        return rnn_output, hidden, new_M_emo, new_context
    def _compute_context(self,rnn_output,memory):
        '''
        Compute context
        '''
        rnn_output = rnn_output.unsqueeze(dim=-2).squeeze(0) # make shape (batch,1,hidden_size)
        memory = memory.permute(1,0,2)
        Wq = self.attn1(rnn_output)
        Wm = self.attn2(memory)
        concat = (Wq + Wm).tanh()
        e = self.concat(concat).squeeze(2)
        attn_score = tf.expand_dims(tf.nn.softmax(e,axis = 1),axis=1)
        context = tf.linalg.matmul(attn_score,memory).squeeze(1)
        return tf.expand_dims(context,axis=0)
    def _read_internal_memory(self,read_inputs,emotion_input):
        """
        Read the internal memory
            emotion_input: [batch_size, emo_hidden_size]
            read_inputs: [batch_size, d] d= [last_word_embedding;decoder_output;context_input]
        Returns:
            output: [batch_size, emo__hidden_size]
        """
        read_inputs = self.read_g(read_inputs)
        M_read = tf.math.sigmoid(read_inputs)
        return emotion_input * M_read
    def _write_internal_memory(self,emotion_input,rnn_output):
        """
        Write the internal memory
            emotion_input: [batch_size, emo_hidden_size]
            rnn_output: [batch_size, hidden_size]
        Returns:
            output: [batch_size, emo_hidden_size]
        """
        M_write = tf.math.sigmoid(self.write_g(rnn_output))
        return emotion_input * M_write
    
