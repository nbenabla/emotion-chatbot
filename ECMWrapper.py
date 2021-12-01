import tensorflow as tf
from tensorflow.keras.layers import Dense

class ECMWrapper(tf.Module):
    def __init__(self,hidden_size,state_size,emo_size,num_emotion,embedding,emotion_embedding,gru):
        '''
        hidden_size: hidden input dimension
        state_size: state vector size
        emo_size: emotional embedding size
        num_emotion: number of emotion categories
        '''
        super(ECMWrapper,self).__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.emo_size = emo_size
        self.num_emotion = num_emotion
        # read gate dimensions (word_embedding + hidden_input + context_input)
        self.read_g = Dense(self.emo_size, input_shape=self.hidden_size + self.hidden_size + self.hidden_size)
        # write gate
        self.write_g = Dense(self.emo_size, input_shape= self.state_size)
        # GRU output input dimensions = state_last + context + emotion emb + internal memory
        self.gru = gru
        self.emotion_embedding = emotion_embedding
        self.embedding = embedding
    def forward(self,word_input,emotion_input,last_hidden,context_input,M_emo):
        '''
        Last hidden == prev_cell_state
        last word embedding = word_input
        last hidden input = h
        '''
        # get embedding of input word and emotion
        context_input = context_input.expand_dims(axis = 0)
        emo_embedding = self.emotion_embedding(emotion_input)
        emo_embedding = emo_embedding.expand_dims(axis = 0)
        last_word_embedding = self.embedding(word_input)
        # sum bidirectional hidden input
        last_hidden_sum = tf.reduce_sum(last_hidden,axis = 0).expand_dims(axis=0)
        read_inputs = tf.concat((last_word_embedding,last_hidden_sum,context_input), axis = -1)
        # compute read input
        read_inputs = self.read_g(read_inputs)
        M_read = tf.math.sigmoid(read_inputs)
        # write to emotion embedding
        emo_embedding = emo_embedding * M_read
        # pass everything to GRU
        X = tf.concat([last_word_embedding,last_hidden_sum, context_input, emo_embedding], axis = -1)
        rnn_output, hidden = self.gru(X,last_hidden)
        # write input
        M_write = tf.math.sigmoid(self.write_g(rnn_output))
        # write to emotion embedding
        new_M_emo = emo_embedding * M_write
        return rnn_output, hidden, new_M_emo