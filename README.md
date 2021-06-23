# MLP-CN_dataset

#### This repository contains data and source code related to the work "Structure and Performance of Fully-Connected Neural Networks Through Complex Networks", which introduces the concept of Bag-Of-Neurons (BON) for analyzing neurons on fully-connected neural networks using Complex Networks.


## Files
We provide a dataset of fully-connected neural networks trained on vision benchmarks, and their corresponding Complex Network (CN) properties. Models are MLPs with sizes 784x200x100x10, ReLu/softmax activations, and trained with stochastic gradient descent.
The whole data is around 150GB, and is stored in a server: http://scg-turing.ifsc.usp.br/data/bases/MLP-CN_dataset/

Data contains all network synapses from random initialization and throughout all training epochs. Complex Network measures are then computed to all hidden neurons.
  * The main directory contains subfolders separated by the target vision benchmarks ("benchmark_deepFeedForward"). Inside the directory of each benchmark, there are two additional subfolders:    
    * "200x100hidden_30epochs_init09" --> Contains .h5 and .pickle files containing the neural network architecture and synapses, filenames are organized by indexes from 1 to 1000. See "MNIST_training.py" for details on how these files are created, and how to open them. 
    * "feature_maps_..." --> .mat files containing matrices of Complex Network measures computed from each neural network's hidden neurons throughout all training epochs. Filenames are organized according to each sample index. See "getCN_measures.py" for details on how they are created.

Notes: folders "EMNISTLETTERS" and "EMNIST" were not used on the paper and did not contain the whole data nor the same structure as the other folders.

## Requirements
* requirements.txt --> Contains the primary package versions used to build, train and store the neural networks, as long as to compute its CN measures.
    `pip install -r requirements.txt`
