from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import tensorflow as tf
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np

import pandas as pd
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import time
import json
import pickle
from queue import PriorityQueue
from preprocess import get_all
from preprocess import get_data



# Define constant
# Default word tokens
# #
torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
MAX_LENGTH = 10  # Maximum sentence length to consider
MIN_COUNT = 3    # Minimum word count threshold for trimming
save_dir = os.path.join("data", "save")
emo_dict = { 0: 'neutral', 1: 'joy', 2: 'anger', 
            3: 'sadness',4:'fear'}
emo2idx = {value:key for key,value in emo_dict.items()}

voc, pairs, pairs_emotion=get_data()
test_batch, pairs, pairs_emotion, test_pairs, test_pairs_emotion = get_all()

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


class ECMWrapper(nn.Module):
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
        self.read_g = nn.Linear(self.hidden_size + self.hidden_size + self.hidden_size,self.emo_size)
        # write gate
        self.write_g = nn.Linear(self.state_size, self.emo_size)
        # GRU output input dimensions = state_last + context + emotion emb + internal memory
        self.gru = gru
        self.emotion_embedding = emotion_embedding
        self.embedding = embedding
        # attention layer
        self.attn1 = nn.Linear(self.hidden_size,self.hidden_size)
        self.attn2 = nn.Linear(self.hidden_size,self.hidden_size)
        self.concat = nn.Linear(self.hidden_size, 1)
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
            decoder_output = torch.zeros(word_input.shape[1],self.hidden_size,dtype=torch.float,device = self.device)
            decoder_output = decoder_output.unsqueeze(0)
            context_input = self._compute_context(decoder_output,memory)
        last_word_embedding = self.embedding(word_input)
        read_inputs = torch.cat((last_word_embedding,decoder_output,context_input), dim = -1)
        # compute read input
        # write to emotion embedding
        emotion_input = self._read_internal_memory(read_inputs,emotion_input)
        # pass everything to GRU
        # decoder_output: logits from last rnn unit
        X = torch.cat([context_input, last_word_embedding, emotion_input], dim = -1)
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
        attn_score = torch.softmax(e,dim = 1).unsqueeze(1)
        context = torch.bmm(attn_score,memory).squeeze(1)
        return context.unsqueeze(0)
    def _read_internal_memory(self,read_inputs,emotion_input):
        """
        Read the internal memory
            emotion_input: [batch_size, emo_hidden_size]
            read_inputs: [batch_size, d] d= [last_word_embedding;decoder_output;context_input]
        Returns:
            output: [batch_size, emo__hidden_size]
        """
        read_inputs = self.read_g(read_inputs)
        M_read = torch.sigmoid(read_inputs)
        return emotion_input * M_read
    def _write_internal_memory(self,emotion_input,rnn_output):
        """
        Write the internal memory
            emotion_input: [batch_size, emo_hidden_size]
            rnn_output: [batch_size, hidden_size]
        Returns:
            output: [batch_size, emo_hidden_size]
        """
        M_write = torch.sigmoid(self.write_g(rnn_output))
        return emotion_input * M_write


class ECMGRU(nn.Module):
    def __init__(self,hidden_size,static_emo_size,emo_size,n_layers = 1):
        '''
        Single layer GRU.
        '''
        super(ECMGRU,self).__init__()
        # first layer of special GRU
        self.hidden_size = hidden_size
        # these three linear layer compute output from emotion/internal memory 
        self.emotion_u = nn.Linear(static_emo_size + emo_size, hidden_size)
        self.emotion_r = nn.Linear(static_emo_size + emo_size, hidden_size)
        self.emotion_c = nn.Linear(static_emo_size + emo_size, hidden_size)
        # these two are generic GRU output that takes [step_input, last_hidden]
        self.generic_u = nn.Linear(hidden_size + hidden_size + hidden_size + emo_size, hidden_size)
        self.generic_r = nn.Linear(hidden_size + hidden_size + hidden_size + emo_size, hidden_size)
        
        # gate value computation
        self.generic_c = nn.Linear(hidden_size + hidden_size + hidden_size + emo_size, hidden_size)
        self.n_layers = n_layers
        # starting from second layer, using the normal GRU
        self._cell = nn.GRU(hidden_size, hidden_size,num_layers = n_layers - 1)
    def forward(self,step_input,last_hidden,emotion,internal_memory):
        '''
        step_input: X
        last_hidden: Hidden value from GRU
        emotion: static emotion embedding vector
        internal_memory: decayed emotion embedding vector
        '''
        # compute based on the first layer
        hidden_0 = last_hidden[0].unsqueeze(dim = 0)
        internal_memory = internal_memory.squeeze(dim = 0)
        emotion_input = torch.cat([emotion,internal_memory],dim=-1)
        # compute emotion gate value
        _u = self.emotion_u(emotion_input) # update gate
        _r = self.emotion_r(emotion_input) # reset gate
        _c = self.emotion_c(emotion_input) # reset gate vector
        
        # generic GRU gate value
        X = torch.cat([step_input,hidden_0],dim = -1)
        u = self.generic_u(X)
        r = self.generic_r(X)
        
        # gate for this time stamp
        rt = torch.sigmoid(r + _r)
        ut = torch.sigmoid(u + _u)
        
        # gate vector
        Xc = torch.cat([step_input, hidden_0 * rt],dim = -1)
        ct = _c + self.generic_c(Xc)
        ct = torch.tanh(ct)
        # compute new hidden
        hidden = ut * hidden_0 + (1 - ut) * ct
        
        # if it has second layer
        if self.n_layers > 1:
            gru_hidden = last_hidden[1:] # skip the first layer of input
            gru_input = hidden
            rnn_output, gru_hidden = self._cell(gru_input, gru_hidden)
        hidden = torch.cat([hidden,gru_hidden],dim = 0)
        return rnn_output, hidden

class LuongAttnDecoderRNN(nn.Module):
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
        self.embedding_dropout = nn.Dropout(dropout)
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
        self.generic_word_output_layer = nn.Linear(self.hidden_size,output_size)
        # emotional output linear layer 
        self.emotion_word_output_layer = nn.Linear(self.hidden_size,output_size)
        # emotional gate/ choice layer
        self.alpha_layer = nn.Linear(hidden_size,1)
        # Luong eq 5 layer
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
    def forward(self, input_step, input_static_emotion, input_step_emotion, last_hidden
                ,input_context, encoder_outputs,last_rnn_output = None):
        '''
        Decoder with external memory.
        
        '''
        if not torch.is_floating_point(input_step_emotion):
            input_step_emotion = self.emotion_embedding(input_step_emotion) # float number for internal memory
        input_static_emotion = self.emotion_cat_embedding(input_static_emotion)
        rnn_output, hidden, new_M_emo,context = self.internal_memory(input_step,last_rnn_output,
                                                                     input_static_emotion,
                                                                     input_step_emotion,
                                                                     input_context,last_hidden,
                                                                     encoder_outputs)
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        concat_input = torch.cat((rnn_output, context), -1)
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
            g = torch.sigmoid(self.alpha_layer(concat_output))
            output_g = torch.softmax(generic_output,dim = 1) * (1 - g)
            output_e = torch.softmax(emotion_output,dim = 1) * g
            output = output_g + output_e # output distribution
            output = output.squeeze(0)
            g = torch.cat([(1 - g),g],dim = -1) # gate distribution
            g = g.squeeze(0)
        else:
            # Predict next word using Luong eq. 6
            output = self.out(concat_output).squeeze(0)
            # generic output
            output = F.softmax(output, dim=1)
            output = output.squeeze(0)
            g = None
        # Return output and final hidden state
        return output, hidden, new_M_emo, context,concat_output,g


gru = ECMGRU(emo_size=500,hidden_size=500,static_emo_size=500,n_layers=2)

inp = torch.zeros((1,64,1500))
last_hidden = torch.ones((2,64,500))
static_emo = torch.randn((64,500))
internal_memory = torch.randn((1,64,500))

rnn_output, hidden = gru(inp,last_hidden,static_emo,internal_memory)


def maskNLLLoss_IMemory(inp, target, mask,M_emo,external_memory,alpha):
    '''
    When external memory input will be a tuple with 4 elements
    '''
    nTotal = mask.sum()
    
    # cross entropy loss
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1) + 1e-12)
    # internal emotional loss
    eos_mask = (target == 2) # 2 is EOS token
    eos_mask = eos_mask.type_as(M_emo)
    internal_memory_loss = torch.norm(M_emo,dim = 2) * eos_mask
    internal_memory_loss = internal_memory_loss.squeeze(0)
    # external
    # find 1,0
    if external_memory is not None:
        qt = torch.gather(external_memory.view(-1,1),0,target.view(-1,1)).type(torch.LongTensor)
        qt = qt.to(device)
        alpha_prob = torch.gather(alpha,1,qt) # if it select emotion word or generic word
        external_memory_loss = (-torch.log(alpha_prob + 1e-12)).reshape(-1) 
    else:
        external_memory_loss = torch.zeros(crossEntropy.shape,dtype=torch.float,device=device)
    #print(crossEntropy.masked_select(mask).mean(),internal_memory_loss.masked_select(mask).mean())
    loss = crossEntropy.masked_select(mask).mean() + external_memory_loss.mean() + internal_memory_loss.mean()
    loss = loss.to(device)
    return loss, nTotal.item(),crossEntropy.masked_select(mask).mean().item()



def compute_perplexity(loss):
    return np.exp(loss)
def train(input_variable, lengths, target_variable,target_variable_emotion,
          mask, max_target_len, encoder, decoder, embedding,emotion_embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # num_samples in this batch
    num_samples = input_variable.shape[1]
    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    target_variable_emotion = target_variable_emotion.to(device)
    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0
    totalCrossEntropy = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(num_samples)]])
    decoder_input = decoder_input.to(device)
    
    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    
    # Set initial context value,last_rnn_output, internal_memory
    context_input = torch.zeros(num_samples,hidden_size,dtype=torch.float,device=device) #torch.FloatTensor(batch_size,hidden_size)
    # Determine if we are using teacher forcing this iteration
    if random.random() < teacher_forcing_ratio:
        use_teacher_forcing = True  
    else:
        use_teacher_forcing = False
    # initialize value for rnn_output
    rnn_output = None
    # keep a copy of emotional category for static emotion embedding
    static_emotion = target_variable_emotion
    static_emotion = static_emotion.to(device)
    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden,target_variable_emotion,context_input,rnn_output,g = decoder(
                decoder_input,static_emotion,target_variable_emotion, decoder_hidden,
                context_input, encoder_outputs,rnn_output
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal,crossEntropy = maskNLLLoss_IMemory(decoder_output, target_variable[t], mask[t],target_variable_emotion,decoder.external_memory,g)
            loss += mask_loss
            totalCrossEntropy += crossEntropy * nTotal
            print_losses.append(mask_loss.item() * nTotal) # print average loss
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden,target_variable_emotion,context_input,rnn_output,g = decoder(
                decoder_input,static_emotion,target_variable_emotion, decoder_hidden,
                context_input,encoder_outputs,rnn_output
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            topi = topi.squeeze(0)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(num_samples)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal,crossEntropy = maskNLLLoss_IMemory(decoder_output, target_variable[t], mask[t],target_variable_emotion,decoder.external_memory,g)
            loss += mask_loss
            totalCrossEntropy += crossEntropy * nTotal
            print_losses.append(mask_loss.item() * nTotal) # print average loss
            n_totals += nTotal

    # Perform backpropatation
    try:
        loss.backward()
    except Exception:
        print(input_variable)
        print(target_variable)

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()
    #print('Total Loss {}; Cross Entropy: {}'.format(sum(print_losses) / n_totals, totalCrossEntropy / n_totals))
    return sum(print_losses) / n_totals,totalCrossEntropy / n_totals
def evaluate_performance(input_variable, lengths, target_variable,target_variable_emotion,
          mask, max_target_len, encoder, decoder):
    # test mode
    
    encoder.eval()
    decoder.eval()
    # num_samples in this batch
    num_samples = input_variable.shape[1]
    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    target_variable_emotion = target_variable_emotion.to(device)
    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0
    totalCrossEntropy = 0
    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(num_samples)]])
    decoder_input = decoder_input.to(device)
    
    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    # Set initial context value,last_rnn_output, internal_memory
    context_input = torch.zeros(num_samples,hidden_size,dtype=torch.float,device=device) #torch.FloatTensor(batch_size,hidden_size)
    # initial value for rnn output
    rnn_output = None
    # keep a copy of emotional category for static emotion embedding
    static_emotion = target_variable_emotion
    static_emotion = static_emotion.to(device)
    # forward pass to generate all sentences
    for t in range(max_target_len):
        decoder_output, decoder_hidden,target_variable_emotion,context_input,rnn_output,g = decoder(
            decoder_input,static_emotion,target_variable_emotion, decoder_hidden,
            context_input,encoder_outputs,rnn_output
        )
        # No teacher forcing: next input is decoder's own current output
        _, topi = decoder_output.topk(1)
        topi = topi.squeeze(0)
        decoder_input = torch.LongTensor([[topi[i][0] for i in range(num_samples)]])
        decoder_input = decoder_input.to(device)
        # Calculate and accumulate loss
        mask_loss, nTotal,crossEntropy = maskNLLLoss_IMemory(decoder_output, target_variable[t], mask[t],target_variable_emotion,decoder.external_memory,g)
        loss += mask_loss
        totalCrossEntropy += (crossEntropy * nTotal)
        print_losses.append(mask_loss.item() * nTotal) # print average loss
        n_totals += nTotal
    # back to train mode
    encoder.train()
    decoder.train()
    return sum(print_losses) / n_totals, totalCrossEntropy / n_totals


def trainIters(model_name, voc, pairs,pairs_emotion, 
               encoder, decoder, encoder_optimizer,
               decoder_optimizer, embedding,emotion_embedding, 
               encoder_n_layers, decoder_n_layers, save_dir, 
               n_iteration, batch_size, print_every, save_every, 
               clip,corpus_name,external_memory,test_pairs,test_pairs_emotion):
    loadFilename=None
    # Load batches for each iteration
    #training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      #for _ in range(n_iteration)]
    print('Loading Training data ...')
    length_pairs = len(pairs)
    #training_batches = [batch2TrainData(voc, [random.choice(range(length_pairs)) for _ in range(batch_size)],
    #                                   pairs,pairs_emotion) for _ in range(n_iteration)]
    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    totalCrossEntropy = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1
    min_test_loss = 1000
    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = batch2TrainData(voc, [random.choice(range(length_pairs)) for _ in range(batch_size)],
                                       pairs,pairs_emotion)
        # to save the data that causes error
        #with open('wrong_data.pickle','rb') as f:
        #    training_batch = pickle.load(f)
        
        # Extract fields from batch
        input_variable,input_variable_emotion, lengths, target_variable,target_variable_emotion, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss,crossEntropy = train(input_variable, lengths, target_variable,target_variable_emotion,
                     mask, max_target_len, encoder,
                     decoder, embedding,emotion_embedding,
                     encoder_optimizer, decoder_optimizer, 
                     batch_size, clip)
        
        print_loss += loss
        totalCrossEntropy += crossEntropy
        # Print progress
        if iteration % print_every == 0 or iteration == 1:
            # directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            # if not os.path.exists(directory):
            #     os.makedirs(directory)
            if iteration == 1:
                print_loss_avg = print_loss / 1
                print_cross_entropy = totalCrossEntropy / 1
            else:
                print_loss_avg = print_loss / print_every
                print_cross_entropy = totalCrossEntropy / print_every
            if print_cross_entropy > 300:
                perplexity = compute_perplexity(300)
            else:
                perplexity = compute_perplexity(print_cross_entropy)
            output1 = "Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}; Perplexity: {:.2f}".format(iteration, iteration / n_iteration * 100, print_loss_avg,perplexity)
            print(output1)
            test_length_pairs = len(test_pairs) 
            test_batch = batch2TrainData(voc, [idx for idx in range(50)],
                                       test_pairs,test_pairs_emotion)
            input_variable,input_emotion, lengths, target_variable,target_emotion, mask, max_target_len = test_batch
            test_loss,testCrossEntropy = evaluate_performance(input_variable,lengths, target_variable,target_emotion,mask,max_target_len,encoder,decoder)
            
            if testCrossEntropy > 300:
                perplexity = compute_perplexity(300)
            else:
                perplexity = compute_perplexity(testCrossEntropy)
            output2 = 'Loss on validation set {:.4f}; Perplexity:{:.2f}'.format(test_loss,perplexity)
            print(output2)
            # with open(os.path.join(directory,'log.txt'),'a+') as f:
            #     f.write(output1 + '\n')
            #     f.write(output2 + '\n')
            print_loss = 0
            totalCrossEntropy = 0

        # Save checkpoint and only save the better perform one,
        if (iteration % save_every == 0) and (testCrossEntropy < min_test_loss):
            min_test_loss = testCrossEntropy
            print('Save the model at checkpoint {}, and test loss is {}'.format(iteration,min_test_loss))
            # directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            # if not os.path.exists(directory):
            #     os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict(),
                'external_memory':external_memory
            }, 'model.pth')
            
def print_param(model):
    for name,param in model.named_parameters():
        print(param)
        print(name,param.grad)


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder,num_word = None):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq,target_emotions,input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Set initial context value,last_rnn_output, internal_memory
        context_input = torch.zeros((1,hidden_size),dtype=torch.float,device=self.decoder.device)
        context_input = context_input.to(device)
        rnn_output = None
        # keep a copy of emotional category for static emotion embedding
        static_emotion = target_emotions
        static_emotion = static_emotion.to(device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden,target_emotions,context_input,rnn_output,g = decoder(
                decoder_input,static_emotion,target_emotions, decoder_hidden,
                context_input, encoder_outputs,rnn_output
            )
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


class BeamSearchDecoder(nn.Module):
    def __init__(self, encoder, decoder,num_word):
        super(BeamSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_word = num_word

    def forward(self, input_seq,target_emotions,input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_words_order = torch.zeros((1,self.num_word),device=decoder.device,dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        all_scores_array = torch.zeros((1,self.num_word),device=decoder.device,dtype=torch.float)
        # Set initial context value,last_rnn_output, internal_memory
        context_input = torch.zeros(1,hidden_size,dtype=torch.float)
        context_input = context_input.to(decoder.device)
        rnn_output = None
        # keep a copy of emotional category for static emotion embedding
        static_emotion = target_emotion
        static_emotion = static_emotion.to(device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden,target_emotions,context_input,rnn_output,g = decoder(
                decoder_input,static_emotion,target_emotions, decoder_hidden,
                context_input, encoder_outputs,rnn_output
            )
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            decoder_input_order = torch.argsort(decoder_output,dim=1,descending=True)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            all_scores_array = torch.cat((all_scores_array,decoder_output),dim = 0)
            all_words_order = torch.cat((all_words_order,decoder_input_order), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        sequences = self.beam_search(all_scores_array,3)
        return sequences
    def beam_search(self,array,k):
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

'''
CONSTANT
'''
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

def printLines(file, n=5):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)


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


def extractSentencePairs(conversations, is_emotions=True):
    qa_pairs = []
    i = 0
    while i < len(conversations) - 1:
        if is_emotions:
            qa_pairs.append([conversations[i], conversations[i + 1]])
        else:
            qa_pairs.append([normalizeString(conversations[i]), normalizeString(conversations[i + 1])])
        i += 1

    return qa_pairs


def loadLines(fileName):
    lines = {}
    with open(fileName, 'r') as f:
        for idx, line in enumerate(f):
            conversations = [i.strip() for i in line.strip().split('__eou__')[:-1]]
            qa_pairs = extractSentencePairs(conversations, is_emotions=False)
            lines[idx] = qa_pairs
    return lines


def loadEmotions(fileName):
    lines = {}
    with open(fileName, 'r') as f:
        for idx, line in enumerate(f):
            emotions = [int(i) for i in line.strip().split(' ')]

            emotions_pair = extractSentencePairs(emotions, is_emotions=True)
            lines[idx] = emotions_pair

    return lines
class Voc:
    def __init__(self, name,min_count,max_length):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD
        self.min_count = min_count
        self.max_length = max_length
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word]= self.word2count.get(word,0) + 1

    # Remove words below a certain count threshold
    def trim(self):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= self.min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {'PAD':0, 'SOS':1,'EOS':2}
        self.word2count = {'PAD':self.min_count,'SOS':self.min_count,'EOS':self.min_count}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)


# Read query/response pairs and return a voc object
import string


def flatten(data):
    pairs = []
    for key, values in data.items():
        for each in values:
            pairs.append([each[0], each[1]])

    return pairs


def readVocs(min_count,max_length):
    print("Reading lines...")
    # Read the file and split into lines
#     conversations = loadLines()
#     emotions = loadEmotions(emotions_data)

    voc = Voc('Movie_Dialogue',min_count,max_length)
    return voc, flatten(conversations), flatten(emotions)


# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p,max_length,min_length):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length \
           and len(p[0].split(' ')) >= min_length and len(p[1].split(' ')) >= min_length


# Filter pairs using filterPair condition
def filterPairs(pairs,pairs_emotion,max_length,min_length):
    keep_pairs,keep_pairs_emotion = [], []
    for pair,pair_emotion in zip(pairs,pairs_emotion):
        if filterPair(pair,max_length,min_length):
            keep_pairs.append(pair)
            keep_pairs_emotion.append(pair_emotion)
    return keep_pairs,keep_pairs_emotion


# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus_data, emotions_data,min_count,max_length,drop_num):
    print("Start preparing training data ...")
    voc, pairs, pairs_emotion = readVocs(corpus_data, emotions_data,min_count,max_length)
    # flatten the pairs of sentences
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs,pairs_emotion = filterPairs(pairs,pairs_emotion,voc.max_length)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print('Emotions left {}'.format(len(pairs_emotion)))
    print('Under sample one categories:')
    pairs, pairs_emotion = under_sample(pairs, pairs_emotion, 0, drop_num)
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs, pairs_emotion

def trimRareWords(voc, pairs,pairs_emotion, min_count):
    # Trim words used under the MIN_COUNT from the voc

    voc.trim()
    # Filter out pairs with trimmed words
    keep_pairs = []
    keep_emotions = []
    for pair,pair_emotion in zip(pairs,pairs_emotion):
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)
            keep_emotions.append(pair_emotion)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs,keep_emotions


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
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len
# return an emotion tensor
def emotion_tensor(input_list):
    return torch.LongTensor(input_list)
# Returns all items for a given batch of pairs
def batch2TrainData(voc, index,pairs,pairs_emotion):
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


def get_data(data_path = '../data/ijcnlp_dailydialog', corpus_name = 'dialogues_text.txt', emotions_file = 'dialogues_emotion.txt',min_count = 1,max_length= 10,drop_num = 30000):
    DATA_PATH = data_path 
    corpus = os.path.join(DATA_PATH, corpus_name)
    emotions = os.path.join(DATA_PATH, emotions_file)

    voc, pairs, pairs_emotion = loadPrepareData(corpus, emotions,min_count,max_length,drop_num)
    # Print some pairs to validate
    # Trim voc and pairs
    pairs, pairs_emotion = trimRareWords(voc, pairs, pairs_emotion, min_count)
    '''
    dataset = []
    for idx, qa in enumerate(pairs):
        question, response = qa
        emotions = pairs_emotion[idx]
        question_emotion, response_emotion = emotions
        question = [question] + [question_emotion] * 2
        response = [[response] + [response_emotion] * 2]
        dataset.append([question, response])

    train = dataset
    '''
    return voc,pairs,pairs_emotion

def under_sample(pairs,pairs_emotion,emotions_cate = 0,drop_num = 30000):
    '''
    Under sample the response that has emotion category 0 (no emotion)
    :param pairs:
    :param pairs_emotion:
    :param emotions_cate:
    :param drop_num:
    :return:
    '''
    drop_no_emotion = True
    keep_pairs = []
    keep_pairs_emotion = []
    drop_count = 0
    for conversation,emotion in zip(pairs,pairs_emotion):
        post,response = conversation
        if emotion[1] == emotions_cate and drop_no_emotion:
            drop_count += 1
            if drop_count == drop_num:
                drop_no_emotion = False
            continue
        else:
            keep_pairs.append(conversation)
            keep_pairs_emotion.append(emotion)
    print('Under sample non-emotion to {} samples'.format(len(keep_pairs)))
    return keep_pairs,keep_pairs_emotion


def get_ememory2(file_path, voc):
    '''
    Get external memory from file. And generate category embedding based on the
    current vocabulary

    :param file_path:
    :param voc:
    :return:
    '''
    emotion_words = [0] * voc.num_words
#     print(emotion_words[3890])

#     print(emotion_words)
    count = 0
    with open(file_path, 'r') as f:
        for each in f:
#             print(each)
            each = each.rstrip()
            if each in voc.word2index:
                if (voc.word2index[each] <= len(voc.word2index) -1):
                    count += 1
            
                    emotion_words[voc.word2index[each]] = 1
    print('Emotion word counts:', count)
    return torch.ByteTensor(emotion_words)


def group_emotions(pairs_emotion, emo_group):
    '''
    Group emotion category based on given dictionary
    :param pairs_emotion: list
    :param emo_group: dict
    :return: another piars_emotions
    '''
    pairs_grouped = []

    for each in pairs_emotion:
        each[0] = emo_group[each[0]]
        each[1] = emo_group[each[1]]
        pairs_grouped.append(each)
    return pairs_grouped

def compute_perplexity(loss):
    '''
    Compute perplexity from loss
    :param loss:
    :return:
    '''
    return np.exp(loss)

try:
    voc
except NameError:
    voc = Voc('a',max_length=MAX_LENGTH,min_count=MIN_COUNT)
# Configure models
model_name = 'emotion_model'
corpus_name = 'ECM10_words_GRU_DailyDialogue'
hidden_size = 500
encoder_n_layers = 4
decoder_n_layers = 4
dropout = 0.2
batch_size = 64
# number of emotion
num_emotions = 7
# load external memory based vocab.

emotion_words = get_ememory2('./ememory2.txt',voc)
# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None #'data/save/emotion_model/ECM10_words_GRU_Large_MINLENGTH5/4-4_500/100_checkpoint.tar'
checkpoint_iter = 120
training = True
if loadFilename:
    training = False
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']
    emotion_words = checkpoint['external_memory']
    


print('Building encoder and decoder ...')
# Initialize word embeddings
if emotion_words is not None:
    emotion_words = emotion_words.to(device)

embedding = nn.Embedding(voc.num_words, hidden_size)
emotion_embedding = nn.Embedding(num_emotions, hidden_size)
emotion_embedding_static = nn.Embedding(num_emotions,hidden_size)

if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(embedding,emotion_embedding_static,emotion_embedding, hidden_size, 
                              voc.num_words,device, emotion_words,decoder_n_layers, dropout,num_emotions=num_emotions,batch_size = batch_size)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')


print(voc.num_words)
print(voc.word2index['afraid'])
emotion_words.sum()


# Configure training/optimization
clip = 50
teacher_forcing_ratio = 0.1
# was 0.0001
learning_rate = 0.0001
decoder_learning_ratio = 5.0
# was 20000
n_iteration = 100
# was 20
print_every = 10
# was 100
save_every = 40


# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# Run training iterations


print("Starting Training!")
trainIters(model_name, voc, pairs,pairs_emotion, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding,emotion_embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip,corpus_name,emotion_words,test_pairs,test_pairs_emotion)
    
    