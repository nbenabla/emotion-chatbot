import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, GRU
from ECMWrapper import *
from ECMGRU import *


def maskNLLLoss_IMemory(inp, target, mask,M_emo,external_memory,alpha):
    '''
    When external memory input will be a tuple with 4 elements
    '''
    nTotal = mask.sum()
    
    # cross entropy loss
    crossEntropy = -tf.log(tf.gather(inp, 1, target.view(-1, 1)).squeeze(1) + 1e-12)
    # internal emotional loss
    eos_mask = (target == 2) # 2 is EOS token
    eos_mask = eos_mask.type_as(M_emo)
    internal_memory_loss = tf.norm(M_emo,dim = 2) * eos_mask
    internal_memory_loss = internal_memory_loss.squeeze(0)
    # external
    # find 1,0
    if external_memory is not None:
        qt = tf.gather(external_memory.view(-1,1),0,target.view(-1,1)).type(tf.Tensor)
        qt = qt.to(device)
        alpha_prob = tf.gather(alpha,1,qt) # if it select emotion word or generic word
        external_memory_loss = (-tf.log(alpha_prob + 1e-12)).reshape(-1) 
    else:
        external_memory_loss = tf.zeros(crossEntropy.shape,dtype=np.float32,device=device)
    #print(crossEntropy.masked_select(mask).mean(),internal_memory_loss.masked_select(mask).mean())
    # TODO
    loss = crossEntropy.masked_select(mask).mean() + external_memory_loss.mean() + internal_memory_loss.mean()
    loss = loss.to(device)
    return loss, nTotal.item(),crossEntropy.masked_select(mask).mean().item()