#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 09:27:02 2020
@author: scabini

Main script for parallel compputation of Complex Network properties of
Neural Networks

"""
import os
os.environ["KMP_WARNINGS"] = "no"

from multiprocessing import Pool
import scipy.io as sio

from neural_measures import getNNtopological_measures
import numpy as np
import time

#file paths, see script MNIST_training
path_networks = 'MNIST_deepFeedForward/200x100hidden_30epochs_init09/'
path_out =     'MNIST_deepFeedForward/feature_maps_200x100_30epochs_inituniform09/' #this is where measures will be saved
#################

#function for parallel calls
def analyze(i):    
    file = path_out + 'network' + str(i) + '_9measures.mat' 
    exists = os.path.isfile(file)
    #checks if measures of the current network ID already existes
    if not exists: #if not, compute and save them
        fmap_h1, fmap_h2 = getNNtopological_measures(path_networks, i) 
        sio.savemat(file, {'fmap_h1':fmap_h1, 'fmap_h2':fmap_h2})

if __name__ == '__main__':    
    to_run = [1,2,3] #list of network IDs to compute CN measures
    
    #number of parallel threads to be used. It is important to notice that the 
    #   measures are already computed in parallel (see neural_measures), therefore
    #   we recomend using a value equivalent to 10% of the total system's threads
    #   e.g.: 3 for a 30 thread pprocessor.
    processes = 3 
    
    print('Running with', processes, 'threads')
    start_time = time.time()
    pool = Pool(processes)
    index_list = [np.int(i) for i in to_run] 
    pool.map(analyze, iterable=index_list, chunksize=None)
    print('Spent', np.round(time.time() - start_time, decimals=3), 'seconds (using', processes, 'threads)')     