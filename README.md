# MLP-CN_dataset
A dataset of fully-connected neural networks trained on vision benchmarks and their corresponding Complex Network (CN) properties. Models are MLPs with sizes 784x200x100x10, ReLu/softmax activations, and trained with stochastic gradient descent.

Data contains all their synapse states, i.e., from random initialization and throughout all training epochs. Complex Network measures are then computed individually to all its hidden neurons.

Data is stored in a server and can be accessed at http://scg-turing.ifsc.usp.br/data/bases/MLP-CN_dataset/
  - The main directory contains subfolders separated by the target vision benchmarks ("benchmark_deepFeedForward")
  - Inside the directory of each benchmark, there are two additional subfolders:
    - "200x100hidden_30epochs_init09" --> Contains .h5 and .pickle files containing the neural network architecture and synapses, filenames are organized by indexes from 1 to 1000. See "load_neuralnetwork.py" for details on how to open these files. 
    - "feature_maps_..." --> .mat files containing matrices of Complex Network measures computed from each neural network's hidden neurons throughout all training epochs. Filenames are organized according to each sample index. See "load_CNmeasures.py" for details on how to open these files.

Notes: folders "EMNISTLETTERS" and "EMNIST" were not used on the paper and did not contain all files nor the same structure as the other folders.

"requirements.txt" --> Contains the primary package versions used to build, train and store the neural networks, as long as to compute its CN measures.
