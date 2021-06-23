#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:09:50 2019
@author: scabini

Sample code to build and train neural network samples according to the dataset
described on the paper "Structure and Performance of Fully-Connected Neural
Networks Through Complex Networks"

Here only the MNIST benchmark is used, but 4 different benchmarks were
considered on the paper

"""
from __future__ import print_function
import sys
import os
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use, "0" is the first GPU, "1" the second, etc;
os.environ["CUDA_VISIBLE_DEVICES"]="0";  
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import keras
sys.stderr = stderr
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

import pickle
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras import backend as K
import time

#experiment parameters
number_of_networks = 1 #number of neural network samples to create and train. Our dataset contains 1000 samples to each dataset
path_save = 'MNIST_deepFeedForward/200x100hidden_30epochs_init09/' #path to store the neural network information
base_seed = 999 #base random seed for rng

#model parameters
hidden1_size = 200
hidden2_size = 100
input_size = 784
num_classes = 10

#training parameters
batch_size = 100
epochs = 30

######################

#data treatment. Opening, split into train and test, regualization between [0,1], conversion of the sample labels
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


x_train = x_train.reshape(60000, input_size)
x_test = x_test.reshape(10000, input_size)
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
###########################################################

import numpy as np
np.random.seed(base_seed) #fixing numpy base rng, so that the training shuffle is the same for all neural network samples
  
 
for i in range(1,number_of_networks+1):
    
    start_time = time.time()
    losses = np.zeros((2,epochs+1)) # store train and test losses from epoch 0 to n
    accuracies = np.zeros((2,epochs+1)) # store train and test accuracy from epoch 0 to n
    
    #synapse weight matrices to be stored
    weights1_2 = np.zeros((epochs+1, input_size, hidden1_size))
    weights2_3 = np.zeros((epochs+1, hidden1_size, hidden2_size))
    weights3_4 = np.zeros((epochs+1, hidden2_size, num_classes))
    
    #checks if actual network ID already exists, then load it
    file = path_save + 'network' + str(i) + '.pickle'                
    exists = os.path.isfile(file)
    if exists:
        with open(file, 'rb') as f:
            weights1_2, weights2_3, weights3_4, losses, accuracies = pickle.load(f)
        model = load_model(path_save + 'network' + str(i) + '.h5')
    else: #if not, then create and train the model
           
        semente = base_seed*i #setting the current rng seed to network id "i"
        
        #the function for random weight initialization
        weight_initializer = keras.initializers.RandomUniform(minval=-0.9, maxval=0.9, seed=semente)

        #the neural network model
        model = Sequential()
        model.add(Dense(hidden1_size, activation='relu',use_bias=False,input_shape=(input_size,), kernel_initializer=weight_initializer,bias_initializer=weight_initializer))
        model.add(Dense(hidden2_size, activation='relu',use_bias=False, kernel_initializer=weight_initializer,bias_initializer=weight_initializer))
        model.add(Dense(num_classes, activation='softmax',use_bias=False, kernel_initializer=weight_initializer,bias_initializer=weight_initializer))
        
        #training hyperparameters
        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False),
                      metrics=['accuracy'])
        
        #getting synapse matrices at epoch 0 (random)
        weights1_2[0][:][:] = np.asarray(model.layers[0].get_weights())
        weights2_3[0][:][:] = np.asarray(model.layers[1].get_weights())
        weights3_4[0][:][:] = np.asarray(model.layers[2].get_weights())
        
        #model performance at epoch 0
        score = model.evaluate(x_train, y_train, verbose=0)
        losses[0][0]= score[0];
        accuracies[0][0]= score[1];         
        
        score = model.evaluate(x_test, y_test, verbose=0)
        losses[1][0]= score[0];
        accuracies[1][0]= score[1];  
        
        ############ TRAINING #####################       
        for epoch in range(1,epochs+1):
            model.fit(x_train, y_train,batch_size=batch_size,epochs=1,shuffle=True,
                                verbose=0,
                                validation_data=(x_test, y_test))
            
            #getting synapse matrices at current epoch
            weights1_2[epoch][:][:] = np.asarray(model.layers[0].get_weights())
            weights2_3[epoch][:][:] = np.asarray(model.layers[1].get_weights())
            weights3_4[epoch][:][:] = np.asarray(model.layers[2].get_weights())
            
            #model performance at current epoch
            score = model.evaluate(x_train, y_train, verbose=0)
            losses[0][epoch]= score[0];
            accuracies[0][epoch]= score[1];         
            
            score = model.evaluate(x_test, y_test, verbose=0)
            losses[1][epoch]= score[0];
            accuracies[1][epoch]= score[1];   
            ###################################################
    
    weights1_2 = weights1_2.astype('float32')
    weights2_3 = weights2_3.astype('float32')
    weights3_4 = weights3_4.astype('float32')
    
    #print INFO: performance and time spent to train and test the  current model 
    print('Network', i, ' ---> Train accuracy:', np.round(accuracies[0][epochs], decimals=3), '  -  Test accuracy:', np.round(accuracies[1][epochs], decimals=3), ' - Spent', np.round(time.time() - start_time, decimals=3), 'seconds')
    
    #stores the model files: architecture (.h5 file); synapse matrices and performance metrics (.pickle file)
    with open(file, 'wb') as f:
        pickle.dump([weights1_2, weights2_3, weights3_4, losses, accuracies], f)
    model.save(path_save + 'network' + str(i) + '.h5') 
    
    #here onwards it is possible to perform operations with the current model
    
    #otherwise, close it
    del model
    K.clear_session()     
    
