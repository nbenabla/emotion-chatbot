import tensorflow as tf

from queue import PriorityQueue
from helper import *

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, decoder_input, 
                 logProb, length,static_emotion,emotions_emb,
                 last_rnn_output,context_input,g):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        
        self.hidden_state = hiddenstate
        self.prevNode = previousNode
        self.decoder_input = decoder_input
        self.logp = logProb
        self.leng = length
        self.emotions = emotions_emb
        self.rnn_output = last_rnn_output
        self.context_input = context_input
        self.alpha = g
        self.static_emotion = static_emotion
    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        return self.logp #/ float(self.leng - 1 + 1e-6) + alpha * reward

def beam_decode(encoder,decoder, voc, beam_size, sentence, diversity_penalty, gamma, sentence_length, remove_repeated):
    for emotions in [0,1,2,3,4]:
        #diversity_penalty = False
        emotions = emotions
        #sentence = 'how are you doing ?'
        print('Post({}):{}'.format(emo_dict[emotions],sentence))
        emotions = int(emotions)
        emotions = tf.Variable([emotions], dtype=tf.int64)
        ### Format input sentence as a batch
        # words -> indexes
        indexes_batch = [indexesFromSentence(voc, sentence)]
        # Create lengths tensor
        lengths = tf.Variable([len(indexes) for indexes in indexes_batch])
        # Transpose dimensions of batch to match models' expectations
        input_batch = tf.Variable(indexes_batch, dtype=tf.int64).transpose(0, 1)
        # Use appropriate device
        input_batch = input_batch.to(device)
        lengths = lengths.to(device)
        emotions = emotions.to(device)
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = encoder(input_batch, lengths)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = tf.ones((1, 1), device=device, dtype=tf.int64) * SOS_token
        # Set initial context value,last_rnn_output, internal_memory
        context_input = tf.zeros((1, hidden_size), dtype=tf.float32)
        context_input = context_input.to(decoder.device)
        rnn_output = None
        #
        static_emotion = emotions
        static_emotion = static_emotion.to(device)
        node = BeamSearchNode(hiddenstate=decoder_hidden, decoder_input=decoder_input,
            context_input=context_input, static_emotion=static_emotion, emotions_emb=emotions,
            length=1, logProb=0, last_rnn_output = rnn_output, previousNode=None, g = 0)
        sent_leng = 0
        # beam search
        K = beam_size
        # Iteratively decode one word token at a time
        # Forward pass through decoder
        nodes = PriorityQueue(maxsize=K)
        nodes.put((0,node))
        # diversity rate
        gamma = gamma
        # choice
        g_losses = []
        for i in range(sentence_length):
            #print('Decoder {} word'.format(i + 1))
            choices = []
            while not nodes.empty():
                score,node = nodes.get()
                #print('Last word at position {}'.format(node.leng))
                if node.decoder_input.item() == 2: # decode stop when EOS is met
                     choices.append((score,node))
                     continue
                decoder_output, decoder_hidden,emotions,context_input,rnn_output,g = decoder(
                    node.decoder_input,node.static_emotion,node.emotions, node.hidden_state,
                    node.context_input,encoder_outputs,node.rnn_output
                )
                #print(g)
                # Obtain most likely word token and its softmax score
                # decoder_output = decoder_output.unsqueeze(0)
                decoder_scores, decoder_input = torch.topk(decoder_output, k= K, dim=1) # TODO: Tensorflow's tf.nn.top_k() doesn't have dim parameter
                decoder_scores = tf.math.log(decoder_scores)
                if diversity_penalty and i >= 1:
                    # apply based on rank
                    penalties = tf.range(0, K, dtype=tf.float32,device=device) * gamma
                    # apply penalties on the output
                    decoder_scores = decoder_scores - penalties
                token_choices = [decoder_input[0,i].item() for i in range(K)] 
                token_scores = [decoder_scores[0,i].item() for i in range(K)] 
                #print(voc.index2word[token_choices[0]])
                # for each candidate token, compute loss
                for token,decoder_score in zip(token_choices,token_scores):
                    
                    next_decoder_input = tf.ones((1,1), dtype=tf.int64, device=device) * token
                    #current_score = score + decoder_score
                    if token == node.decoder_input.item() and remove_repeated:
                        decoder_score = -100
                    next_node = BeamSearchNode(decoder_hidden,node,next_decoder_input,
                                          decoder_score,node.leng + 1,static_emotion,emotions,rnn_output,context_input,g)
                    #print('This is {} words'.format(next_node.leng))
                    current_score = (score * node.leng  - next_node.eval()) / next_node.leng
                    choices.append((current_score,next_node))
            choices = sorted(choices, key=lambda x:x[0])
            # choices = choices[:K]
            for choice in choices:
                if not nodes.full():
                    nodes.put(choice)

        #print(nodes.qsize())
        #print('Decode')        
        # decoder    
        sentences = []
        i = 0
        while not nodes.empty():
            #print('Decode {}:'.format(i))
            i += 1 
            sentence_ = []
            score,node = nodes.get()
            while(node.prevNode is not None):
                sentence_.append(node.decoder_input.item())
                node = node.prevNode
            sentence_ = sentence_[::-1]
            #print(sentence,score)
            sentences.append((score,sentence_))
        #print(sentences)
        for sent in sentences[:beam_size]:
            print(sentenceFromIdx(sent[1],voc),sent[0])  