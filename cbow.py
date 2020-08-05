#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 19:38:49 2020

@author: vyvo
"""
#change directory
# import os
# os.chdir('/Users/vyvo/ML/NLP/codes')

from dependency import *
from cleantext import *
import random


# Generate training data 
def generate_training_data(vocab_size, window_size):  
        
    X = np.zeros((vocab_size, vocab_size))
    Y = np.identity(vocab_size)
    
    for i in range(vocab_size):
        context = list(range(max(0, i - window_size), i))
        context += list(range(i+1, min(vocab_size, i + window_size + 1)))
        X[i,context] = 1/len(context)
    
    return X, Y
                         
# Initialize values for input and output weights
def init_weights(vocab_size, emb_size):
    
    # Word embedding layer input weights
    V = np.random.rand(emb_size, vocab_size)*0.01
    
    # Dense layer output weights
    U = np.random.rand(vocab_size, emb_size)*0.01
    
    return V, U   
        
# Foward Backprogation
def forward_propagation(X, Y, V, U):
                        
    """  
    
    X : average input context word vectors for 1 center word, shape = (N x 1)
    V : input weights, shape = (n x N)
    
    U : output weights, shape = (N x n)
    h: hidden layer vector, shape = (n x 1)
    
    z: linear dense layer vector, shape = (N x 1)
    
    y: predicted output center vector, shape = (N x 1)
    Y: actual output center vector, shape = (N x 1)
    
    
    """
    
    parameters = {'X':X, 'Y':Y, 'V':V, 'U':U}
    
    # hidden layer vector: h, shape = (n x 1) 
    parameters['h'] = V @ X
    
    # linear dense vector: z, shape = (N x 1)
    z = U @ parameters['h']
    parameters['z'] = z
    
    # softmax output vector y, shape = (N x 1) 
    parameters['y'] = np.divide(np.exp(z), np.sum(np.exp(z),axis=0, keepdims=True) + 0.001)
    

    return parameters 

def cross_entropy(parameters):
    
    """ 

    y: predicted output center vector, shape = (N x 1)
    Y: actual output center vector, shape = (N x 1)

    """
    y, Y = parameters['y'], parameters['Y'] 
    cost = -np.sum(Y * np.log(y)+0.01, axis=0, keepdims=True)
    return cost 

def get_gradients(parameters):
    
    """

    dL_dU = dL_dz * dz_dU
    dL_dV = dL_dz * dz_dh * dh_dV

    """
    
    # softmax backward dL_dz, shape = (N x 1)
    dL_dz = parameters['y'] - parameters['Y']
    dL_dz = dL_dz.reshape(-1,1)
    
    # hidden to output weights dL_dU, shape = (N x n)
    
    h = parameters['h'].reshape(-1,1)
    dL_dU =  dL_dz @ h.T
    
    # input to hidden weights dL_dV, shape = (n x N)
    EH = parameters['U'].T @ dL_dz   # shape = (n x 1)
    x = parameters['X'].reshape(1,-1) # shape = (1 x N)
    dL_dV = EH @ x
    
    gradients = (dL_dU, dL_dV)
    return gradients

def backward_propagation(parameters, gradients, learning_rate):
    parameters['U'] -= learning_rate * gradients[0]
    parameters['V'] -= learning_rate * gradients[1]
    

def cbow_stochastic_training(tokens, emb_size, window_size, epochs, learning_rate, print_loss=False, plot_loss=False):
    N = len(tokens) # vocab size
    X, Y = generate_training_data(N, window_size)
    V, U = init_weights(N, emb_size)
    
    loss = []
    
    for epoch in range(epochs):
                
        # select 1 random sample to update
        i = random.randint(0,N-1) 
        X_batch, Y_batch = X[:,i], Y[:,i]
        
        # update parameters
        parameters = forward_propagation(X_batch, Y_batch, V, U)
        gradients = get_gradients(parameters)
        backward_propagation(parameters, gradients, learning_rate)
        
        # calculate loss
        epoch_loss = cross_entropy(parameters)
        loss.append(epoch_loss)
        
        if print_loss:    
            print('Loss after epoch {}: {}'.format(epoch, epoch_loss))
            print('Cummulative loss {}'.format(np.array(loss).sum()))
        
    if plot_loss:
        plt.plot(np.arange(epochs), loss)
        plt.xlabel('Number of epochs')
        plt.ylabel('loss')
        
    return X, Y, parameters['V'], parameters['U']

def model_evaluation(tokens, X, Y, V, U, k):
    """
    Output target words of chosen sets of context words
    
    X: input context word vectors of all words, shape = (N x N)
    Y: all output target word vectors (one hot coding), shape = (N x N)
    U: trained input weights, shape = (n x N)
    k: test sample size
    """
    
    N = len(tokens)
    test_sample = random.sample(range(N),k)
    
    for i in test_sample:
        x_test, y_test = X[i], Y[i]
        actual_word = tokens[np.argwhere(y_test==1).item()]
        print('---- Actual word: {} ----- \n'.format(actual_word))

        context = [tokens[word] for word in np.squeeze(np.argwhere(x_test > 0))]
        parameters = forward_propagation(x_test, y_test, V, U)
        pred_y = np.argmax(parameters['y']) # index of predicted word
        pred_word = tokens[pred_y]
        print('Context words: {} --> Predicted word: {} \n'.format(
            context, pred_word))

#### MODEL TRAINING ####

text = "Victoria's record-breaking day has concerned some experts who say the huge jump in coronavirus case numbers is surprising and shocking."


# Tokenize
clean_text = remove_special_characters(text)
tokens = tokenizer.tokenize(clean_text)

# Define hyperparameters
emb_size = 10           # embedding size n
window_size = 3         # window size
epochs = 10000          # epochs to train model
learning_rate = 0.05    # learning rate
k = 5

X, Y, V, U = cbow_stochastic_training(tokens, emb_size, window_size, epochs, learning_rate)
model_evaluation(tokens, X, Y, V, U, k)




