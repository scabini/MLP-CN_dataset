# MLP-CN_dataset

This repository corresponds to the work "Structure and Performance of Fully-Connected Neural Networks Through Complex Networks", which introduces the concept of Bag-Of-Neurons (BON) for analyzing neurons on fully-connected neural networks using Complex Networks.

We provide a dataset of fully-connected neural networks trained on vision benchmarks, and their corresponding Complex Network (CN) properties. Models are MLPs with sizes 784x200x100x10, ReLu/softmax activations, and trained with stochastic gradient descent.

Data contains all network synapses from random initialization and throughout all training epochs. Complex Network measures are then computed to all hidden neurons.

Data is stored in a server and can be accessed at http://scg-turing.ifsc.usp.br/data/bases/MLP-CN_dataset/
  - The main directory contains subfolders separated by the target vision benchmarks ("benchmark_deepFeedForward")
  - Inside the directory of each benchmark, there are two additional subfolders:
    - "200x100hidden_30epochs_init09" --> Contains .h5 and .pickle files containing the neural network architecture and synapses, filenames are organized by indexes from 1 to 1000. See "load_neuralnetwork.py" for details on how to open these files. 
    - "feature_maps_..." --> .mat files containing matrices of Complex Network measures computed from each neural network's hidden neurons throughout all training epochs. Filenames are organized according to each sample index. See "load_CNmeasures.py" for details on how to open these files.

Notes: folders "EMNISTLETTERS" and "EMNIST" were not used on the paper and did not contain all files nor the same structure as the other folders.

"requirements.txt" --> Contains the primary package versions used to build, train and store the neural networks, as long as to compute its CN measures.
