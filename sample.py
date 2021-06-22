# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:14:49 2021
@author: Svartox
Sample aplication, getting CN properties of a neural net
"""
import numpy as np
import pickle
from keras.models import load_model

from neural_measures import getNNtopological_measures_top3

#loading a neural network (4-layers, fully-connected)
path = 'MNIST_deepFeedForward/200x100hidden_30epochs_init09/'
model = load_model(path + 'network1.h5')

#getting its synapse matrices
weights1_2 = np.asarray(model.layers[0].get_weights())
weights2_3 = np.asarray(model.layers[1].get_weights())
weights3_4 = np.asarray(model.layers[2].get_weights())

#compupting 3 Complex Network measures for hidden neurons (returns 2 matrices, 1 for each hidden layer)
features_h1, features_h2 = getNNtopological_measures_top3(weights1_2[0], weights2_3[0], weights3_4[0])

#checking performance properties of the given model, stored at the pickle file
file = path + 'network1.pickle'                
with open(file, 'rb') as f:
    _, _, _, losses, accuracies = pickle.load(f)