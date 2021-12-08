import tensorflow as tf
import numpy as np
import random
from helper import *
from maskNLLoss import *

def compute_perplexity(loss):
    return np.exp(loss)

def train(input_variable, lengths, target_variable,target_variable_emotion,
          mask, max_target_len, encoder, decoder, embedding, emotion_embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):
    
    # num_samples in this batch
    num_samples = input_variable.shape[1]
    
    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0
    totalCrossEntropy = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = tf.convert_to_tensor([[SOS_token for _ in range(num_samples)]], dtype=tf.int64)
    
    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    
    # Set initial context value,last_rnn_output, internal_memory
    context_input = tf.zeros((num_samples, hidden_size), dtype=tf.float32) 
    # Determine if we are using teacher forcing this iteration
    if random.random() < teacher_forcing_ratio:
        use_teacher_forcing = True  
    else:
        use_teacher_forcing = False
    # initialize value for rnn_output
    rnn_output = None
    # keep a copy of emotional category for static emotion embedding
    static_emotion = target_variable_emotion
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
            mask_loss, nTotal,crossEntropy = maskNLLLoss_IMemory(decoder_output, target_variable[t], mask[t], target_variable_emotion, decoder.external_memory, g)
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
            _, topi = tf.math.top_k(decoder_output, k=1)
            topi = tf.squeeze(topi,axis=0)
            decoder_input = tf.convert_to_tensor([[topi[i][0] for i in range(num_samples)]], dtype=tf.int64)
            # Calculate and accumulate loss
            mask_loss, nTotal,crossEntropy = maskNLLLoss_IMemory(decoder_output, target_variable[t], mask[t], target_variable_emotion, decoder.external_memory, g)
            loss += mask_loss
            totalCrossEntropy += crossEntropy * nTotal
            print_losses.append(mask_loss.item() * nTotal) # print average loss
            n_totals += nTotal

    # Perform backpropagation
    try:
        with tf.GradientTape() as tape:
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    except Exception:
        print(input_variable)
        print(target_variable)

  

    
    #print('Total Loss {}; Cross Entropy: {}'.format(sum(print_losses) / n_totals, totalCrossEntropy / n_totals))
    return sum(print_losses) / n_totals,totalCrossEntropy / n_totals

def evaluate_performance(input_variable, lengths, target_variable,target_variable_emotion,
          mask, max_target_len, encoder, decoder):
    # test mode
    
    encoder.eval()
    decoder.eval()
    # num_samples in this batch
    num_samples = input_variable.shape[1]
   
    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0
    totalCrossEntropy = 0
    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = tf.convert_to_tensor([[SOS_token for _ in range(num_samples)]], dtype=tf.int64)
    
    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    # Set initial context value,last_rnn_output, internal_memory
    context_input = tf.zeros((num_samples,hidden_size), dtype=tf.float32) 
    # initial value for rnn output
    rnn_output = None
    # keep a copy of emotional category for static emotion embedding
    static_emotion = target_variable_emotion
    # forward pass to generate all sentences
    for t in range(max_target_len):
        decoder_output, decoder_hidden ,target_variable_emotion, context_input, rnn_output, g = decoder(
            decoder_input, static_emotion, target_variable_emotion, decoder_hidden,
            context_input, encoder_outputs, rnn_output
        )
        # No teacher forcing: next input is decoder's own current output
        _, topi = tf.math.top_k(decoder_output, k=1)
        topi = tf.squeeze(topi,axis=0)
        decoder_input = tf.convert_to_tensor([[topi[i][0] for i in range(num_samples)]],dtype=np.int32)

        # Calculate and accumulate loss
        mask_loss, nTotal,crossEntropy = maskNLLLoss_IMemory(decoder_output, target_variable[t], mask[t], target_variable_emotion, decoder.external_memory, g)
        loss += mask_loss
        totalCrossEntropy += (crossEntropy * nTotal)
        print_losses.append(mask_loss.item() * nTotal) # print average loss
        n_totals += nTotal
   
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
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
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
            test_batch = batch2TrainData(voc, [idx for idx in range(1000)],
                                       test_pairs,test_pairs_emotion)
            input_variable,input_emotion, lengths, target_variable,target_emotion, mask, max_target_len = test_batch
            test_loss,testCrossEntropy = evaluate_performance(input_variable,lengths, target_variable,target_emotion,mask,max_target_len,encoder,decoder)
            
            if testCrossEntropy > 300:
                perplexity = compute_perplexity(300)
            else:
                perplexity = compute_perplexity(testCrossEntropy)
            output2 = 'Loss on validation set {:.4f}; Perplexity:{:.2f}'.format(test_loss,perplexity)
            print(output2)
            with open(os.path.join(directory,'log.txt'),'a+') as f:
                f.write(output1 + '\n')
                f.write(output2 + '\n')
            print_loss = 0
            totalCrossEntropy = 0

        # Save checkpoint and only save the better perform one,
        if (iteration % save_every == 0) and (testCrossEntropy < min_test_loss):
            min_test_loss = testCrossEntropy
            print('Save the model at checkpoint {}, and test loss is {}'.format(iteration,min_test_loss))
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
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
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))