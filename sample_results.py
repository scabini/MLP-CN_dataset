# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 12:14:49 2021
@author: Scabini
Opening data and some results from our dataset: CN measures and k-means centers 
"""
import pickle

#Opening file containing CN measures for neural networks after training, for
#   a given benchmark. CN measures are not normalized!
file = 'MNIST_CN_measures_allMeasures_lastepoch_allnets.pickle'
with open('results/gathered_measures/'+ file, 'rb') as f:
    hidden1, hidden2,accuracies,losses = pickle.load(f)
    f.close()
    
#Opening kmeans centers obtained from normalized CN measures. Each variable
# corresponds to either one of the hidden layers, or both combined    
file = 'kmeans_fitted_k6_MNIST_lastepoch_allnets.pickle'                
with open('results/kmeans_centers/'+ file, 'rb') as f:
    kmeans_, kmeans_h1, kmeans_h2 = pickle.load(f)
    f.close()
    
print(kmeans_.cluster_centers_)
