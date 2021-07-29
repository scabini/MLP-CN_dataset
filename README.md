# MLP-CN_dataset

#### This repository contains data and source code related to the work "Structure and Performance of Fully Connected Neural Networks: Emerging Complex Network Properties", which introduces the concept of Bag-Of-Neurons (BON) for analyzing neurons on neural networks using Complex Networks.


## Files
We provide a dataset of fully connected neural networks trained on vision benchmarks, and their corresponding Complex Network (CN) properties. Models are MLPs with sizes 784x200x100x10, ReLu/softmax activations, and trained with stochastic gradient descent. Data contains all network synapses from random initialization and throughout all training epochs. Complex Network measures are then computed to all hidden neurons. The whole data is around 140GB, and is stored in a server: http://scg-turing.ifsc.usp.br/data/bases/MLP-CN_dataset/
  * The main directory contains subfolders separated by the target vision benchmarks ("benchmark_deepFeedForward"). Inside the directory of each benchmark, there are two additional subfolders:    
    * "200x100hidden_30epochs_init09" --> Contains .h5 and .pickle files containing the neural network architecture and synapses, filenames are organized by indexes from 1 to 1000. See "MNIST_training.py" for details on how these files are created, and how to open them. 
    * "feature_maps_..." --> .mat files containing 3D matrices of shape epochs-by-hidden_size-by-measures, i.e., Complex Network measures computed from each hidden neuron throughout all training epochs. Filenames are organized according to each sample index. The order the measures are stored on the last matrix dimension is: strength, avg neighbor strength, current flow closenness, bipartite clustering, subgraph c., harmonic c., second-order c., and maximum cliques. See "getCN_measures.py" for details on how these files are created.

! folders "EMNISTLETTERS" and "EMNIST" were not used on the paper and did not contain the whole data nor the same structure as the other folders !

* Folder "results" contains data used for the paper's analyzes. The folder here contains files for the MNIST benchmark, others are on the server. See "sample_results.py" for detail on how to open them.
  * "gathered_measures" -> pickle files containing CN measures (not normalized) and model's performance gathered from all neural networks after training.
  * "kmeans_centers" -> the final k-means centroids (k=6) obtained from each dataset.

## Requirements
* requirements.txt --> Contains the primary package versions used to build, train and store the neural networks, as long as to compute its CN measures. Notice that RNG may vary between different combinations of packages and versions. We use Anaconda for package managing:
  *  `conda install --file requirements.txt` 
  * cudnn 7.6.5, cuda10.2_0 -> NVIDIA (along with GPU drivers)
  * scipy 1.2.1             
  * tensorflow 1.14.0    
  * tensorflow-gpu 1.12
  * keras 2.2.4       
  * networkx 2.4       
  * scikit-learn 0.24.1   
  * numpy 1.19.2    

## Cite

If you use this method, please cite our paper:

Scabini, L., and Bruno, O. "Structure and Performance of Fully Connected Neural Networks: Emerging Complex Network Properties", arxiv preprint (2021).

```
@article{scabini2021structure,
  title={Structure and Performance of Fully Connected Neural Networks: Emerging Complex Network Properties},
  author={Scabini, Leonardo FS and Bruno, Odemir M},
  journal={arXiv preprint},
  year={2021}
}
```     
