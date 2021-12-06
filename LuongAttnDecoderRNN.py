import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU
from ECMWrapper import *


class LuongAttnDecoderRNN(tf.Module):
    def __init__(self, attn_model, embedding, emotion_embedding, hidden_size, output_size, n_layers=1, dropout=0.1, num_emotions=7):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.num_emotions = num_emotions
        # Define layers
        self.embedding = embedding
        # define emotion embedding
        self.emotion_embedding = emotion_embedding
        self.embedding_dropout = tf.nn.Dropout(dropout)
        #self.emotion_embedding_dropout = nn.Dropout(dropout)
        # dimension
        for i in range(n_layers):
            self.gru = tf.keras.layers.GRU(hidden_size, dropout=(
                0 if n_layers == 1 else dropout), return_sequences=True)
        self.concat = Dense(hidden_size, input_shape=hidden_size * 2)
        self.out = Dense(output_size, input_shape=hidden_size)

        self.attn = Attn(attn_model, hidden_size)
        self.internal_memory = ECMWrapper(hidden_size, hidden_size,
                                          hidden_size, self.num_emotions,
                                          self.embedding, self.emotion_embedding, self.gru)

    def forward(self, input_step, input_step_emotion, last_hidden, input_context, last_int_memory, encoder_outputs):
        '''
        First input_context will be a random vectors
        '''
        rnn_output, hidden, new_M_emo = self.internal_memory(input_step, input_step_emotion,
                                                             last_hidden, input_context,
                                                             last_int_memory)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = tf.concat((rnn_output, context), 1)
        concat_output = tf.math.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = tf.nn.softmax(output, axis=1)
        # Return output and final hidden state
        return output, hidden, new_M_emo, context


def maskNLLLoss_IMemory(inp, target, mask,M_emo):
    nTotal = mask.sum()
    crossEntropy = -tf.log(tf.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).sum() + tf.norm(M_emo)
    loss = loss.to(device)
    return loss, nTotal.item()
