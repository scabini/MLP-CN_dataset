#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:43:50 2020
@author: scabini

This script contains a ton of functions for calculating CN properties
given neural network synapse matrices. It is importat to notice that some of 
them were not used on the paper. The main functions one needs to refer
are "getNNgraph", which creates the graph object given the neural network; and
"getNNtopological_measures", that describe the used measures and how they were
computed according to the paper
"""
import pickle
import os
import time
import numpy as np
import networkx as nx
os.environ['KMP_WARNINGS'] = 'off'


"""Transforms neural network synapse matrices into NetworkX graphs"""
def getNNgraph(weights1_2, weights2_3, weights3_4, directed=False, threshold=(-99999999,-99999), void_links=False, absolute=False):
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
        
    if absolute:
        weights1_2 = np.absolute(weights1_2)
        weights2_3 = np.absolute(weights2_3)
        weights3_4 = np.absolute(weights3_4)
  
    #this function works only for 4-layer fully-connected models
    layer_1,layer_2 = weights1_2.shape    
    layer_2,layer_3 = weights2_3.shape 
    layer_3,layer_4 = weights3_4.shape
    
    for i in range(0, layer_1 + layer_2 + layer_3 + layer_4):
        G.add_node(i)
 
    for i in range(0, layer_1):
        for j in range(0, layer_2):
            if not ((threshold[0] <= weights1_2[i][j]) and (weights1_2[i][j] <= threshold[1])):
                G.add_edge(i,layer_1 + j,weight = weights1_2[i][j])

    node_count = layer_1 + layer_2
   
    for i in range(0, layer_2):
        for j in range(0, layer_3):
            if not ((threshold[0] <= weights2_3[i][j]) and (weights2_3[i][j] <= threshold[1])):
                G.add_edge(layer_1 + i,node_count + j,weight = weights2_3[i][j])
    
    node_count = node_count + layer_3

    for i in range(0, layer_3):
        for j in range(0, layer_4):
            if not ((threshold[0] <= weights3_4[i][j]) and (weights3_4[i][j] <= threshold[1])):
                G.add_edge(layer_1 + layer_2+i,node_count + j,weight = weights3_4[i][j])
                
    #sorry I dont remember why the heck I was using void links, just dont use it lol    
    if void_links:
        peso=1.0
        for i in range(0, layer_1):
            for j in range(i+1, layer_1):                
                G.add_edge(i, j, weight=peso)
                G.add_edge(j, i, weight=peso)
                
        for i in range(layer_1, layer_1+layer_2):
            for j in range(i+1, layer_1+layer_2):
                G.add_edge(i, j, weight=peso)
                G.add_edge(j, i, weight=peso)
                
        for i in range(layer_1+layer_2, layer_1+layer_2+layer_3):
            for j in range(i+1, layer_1+layer_2+layer_3):
                G.add_edge(i, j, weight=peso)
                G.add_edge(j, i, weight=peso)
                
        for i in range(layer_1+layer_2+layer_3, layer_1+layer_2+layer_3+layer_4):
            for j in range(i+1, layer_1+layer_2+layer_3+layer_4):
                G.add_edge(i, j, weight=peso)
                G.add_edge(j, i, weight=peso)
    
    return G


#For a general definition for all these functions:
    #G is a NetworkX graph, target1 and target2 are subset of nodes from the
        # graph with which one needs to compute the measure. In our case, we
        # consider only hidden neurons, thus target1 = 1st hidden layer, etc

def average_strength_h1xh2(G, target1, target2):

    strength =  G.in_degree(weight='weight', nbunch=target1)
    str_in= np.zeros((G.order()));
    for value in strength:
        str_in[value[0]] = value[1]
    
    str_in = str_in[target1]  
    
    strength =  G.in_degree(weight='weight', nbunch=target2)
    str_in2= np.zeros((G.order()));
    for value in strength:
        str_in2[value[0]] = value[1]
    
    str_in2 = str_in2[target2]  
    
    return [str_in, str_in2, str_in.mean(), str_in2.mean()]

def undirectedstrength_h1xh2(G, target1, target2):

    strength =  G.degree(weight='weight', nbunch=target1)
    str_in= np.zeros((G.order()));
    for value in strength:
        str_in[value[0]] = value[1]
    
    str_in = str_in[target1]  
    
    strength =  G.degree(weight='weight', nbunch=target2)
    str_in2= np.zeros((G.order()));
    for value in strength:
        str_in2[value[0]] = value[1]
    
    str_in2 = str_in2[target2]  
    
    return [str_in, str_in2, str_in.mean(), str_in2.mean()]

#????   
def neural_articulation(G, sizes, target1, target2):

    str_in =  G.in_degree(weight='weight', nbunch=target1)
    str_out =  G.out_degree(weight='weight', nbunch=target1)
    
    articulation1= np.zeros((sizes[1]));
    
    for i in range(0, len(str_in)):
        articulation1[i] = (str_in[target1[i]])*(str_out[target1[i]])
  

    str_in =  G.in_degree(weight='weight', nbunch=target2)
    str_out =  G.out_degree(weight='weight', nbunch=target2)
    
    articulation2= np.zeros((sizes[2]));
    
    for i in range(0, len(str_in)):
        articulation2[i] = (str_in[target2[i]])*(str_out[target2[i]])
        
    return [articulation1, articulation2, articulation1.mean(), articulation2.mean()]
  
def bipartite_clustering_h1xh2(G, target1, target2):
    #mesmo que latapy clustering
    cls =  nx.algorithms.bipartite.cluster.clustering(G, nodes=target1, mode='max')
    clustering = np.zeros((len(target1)))
    for i in range(0, len(cls)):
        clustering[i] = cls[target1[i]]
    
    cls2 =  nx.algorithms.bipartite.cluster.clustering(G, nodes=target2, mode='max')
    clustering2 = np.zeros((len(target2)))
    for i in range(0, len(cls2)):
        clustering2[i] = cls2[target2[i]]    
    
    return [clustering, clustering2, clustering.mean(), clustering2.mean()]

def spectral_bipartivity_h1xh2(G, target1, target2):
    spectral_bipartivity =  nx.algorithms.bipartite.spectral.spectral_bipartivity(G, weight='weight')
    return [spectral_bipartivity[target1], spectral_bipartivity[target2], spectral_bipartivity[target1].mean(), spectral_bipartivity[target2].mean()]

def closeness(G, target1, target2):
    cls =  nx.algorithms.centrality.closeness_centrality(G, distance='weight')
    cls1=np.zeros((len(target1)))
    for i in range(0, len(target1)):
        cls1[i] = cls[target1[i]]
    
    cls2=np.zeros((len(target2)))
    for i in range(0, len(target2)):
        cls2[i] = cls[target2[i]]
    
    return [cls1, cls2, cls1.mean(), cls2.mean()]

def betweenness(G, target1, target2):
    btw =  nx.algorithms.centrality.betweenness_centrality_subset(G, sources=[i for i in range(0, target1[0])], targets=[i for i in range(target2[-1]+1, target2[-1]+11)], weight='weight', normalized=True)
    
    btw1=np.zeros((len(target1)))
    for i in range(0, len(target1)):
        btw1[i] = btw[target1[i]]
    
    btw2=np.zeros((len(target2)))
    for i in range(0, len(target2)):
        btw2[i] = btw[target2[i]]
    
    
    return [btw1, btw2, btw1.mean(), btw2.mean()]

def CF_closeness(G, target1, target2):
    cls =  nx.algorithms.centrality.current_flow_closeness_centrality(G, weight='weight')
    cls1=np.zeros((len(target1)))
    for i in range(0, len(target1)):
        cls1[i] = cls[target1[i]]
    
    cls2=np.zeros((len(target2)))
    for i in range(0, len(target2)):
        cls2[i] = cls[target2[i]]
    
    return [cls1, cls2, cls1.mean(), cls2.mean()]

def CF_betweenness(G, target1, target2):
    btw =  nx.algorithms.centrality.current_flow_betweenness_centrality_subset(G, sources=[i for i in range(0, target1[0])], targets=[i for i in range(target2[-1]+1, target2[-1]+11)], weight='weight', normalized=True)
    
    btw1=np.zeros((len(target1)))
    for i in range(0, len(target1)):
        btw1[i] = btw[target1[i]]
    
    btw2=np.zeros((len(target2)))
    for i in range(0, len(target2)):
        btw2[i] = btw[target2[i]]
    
    
    return [btw1, btw2, btw1.mean(), btw2.mean()]

def subgraph_centrality(G, target1, target2):
    btw =  nx.algorithms.centrality.subgraph_centrality(G)
    
    btw1=np.zeros((len(target1)))
    for i in range(0, len(target1)):
        btw1[i] = btw[target1[i]]
    
    btw2=np.zeros((len(target2)))
    for i in range(0, len(target2)):
        btw2[i] = btw[target2[i]]
    
    
    return [btw1, btw2, btw1.mean(), btw2.mean()]

def harmonic_centrality(G, target1, target2):
    btw =  nx.algorithms.centrality.harmonic_centrality(G, target1, distance='weight')
    
    btw1=np.zeros((len(target1)))
    for i in range(0, len(target1)):
        btw1[i] = btw[target1[i]]
    
    btw = nx.algorithms.centrality.harmonic_centrality(G, target2, distance='weight')
    
    btw2=np.zeros((len(target2)))
    for i in range(0, len(target2)):
        btw2[i] = btw[target2[i]]
  
    return [btw1, btw2, btw1.mean(), btw2.mean()]

def local_reaching_centrality(G, target1, target2):
   
    btw1=np.zeros((len(target1)))
    for i in range(0, len(target1)):
        btw1[i] = nx.algorithms.centrality.local_reaching_centrality(G, target1[i], weight='weight', normalized=True)
        
    btw2=np.zeros((len(target2)))
    for i in range(0, len(target2)):
        btw2[i] = nx.algorithms.centrality.local_reaching_centrality(G, target2[i], weight='weight', normalized=True)
    
    
    return [btw1, btw2, btw1.mean(), btw2.mean()]


def second_order_centrality(G, target1, target2):
    btw =  nx.algorithms.centrality.second_order_centrality(G)
    
    btw1=np.zeros((len(target1)))
    for i in range(0, len(target1)):
        btw1[i] = btw[target1[i]]
        
    btw2=np.zeros((len(target2)))
    for i in range(0, len(target2)):
        btw2[i] = btw[target2[i]]
       
    return [btw1, btw2, btw1.mean(), btw2.mean()]

def communicability(G, target1, target2):
    btw =  nx.algorithms.communicability_alg.communicability(G)
    
    btw1=np.zeros((len(target1)))
    for i in range(0, len(target1)):
        btw1[i] = btw[target1[i]]
        
    btw2=np.zeros((len(target2)))
    for i in range(0, len(target2)):
        btw2[i] = btw[target2[i]]
     
    return [btw1, btw2, btw1.mean(), btw2.mean()]

def average_neighbor_degree(G, target1, target2):
    btw =  nx.algorithms.assortativity.average_neighbor_degree(G, weight='weight')
    
    btw1=np.zeros((len(target1)))
    for i in range(0, len(target1)):
        btw1[i] = btw[target1[i]]
        
    btw2=np.zeros((len(target2)))
    for i in range(0, len(target2)):
        btw2[i] = btw[target2[i]]
    
    
    return [btw1, btw2, btw1.mean(), btw2.mean()]

def number_of_cliques(G, target1, target2):
    btw =  nx.algorithms.clique.number_of_cliques(G, nodes=target1)
    
    btw1=np.zeros((len(target1)))
    for i in range(0, len(target1)):
        btw1[i] = btw[target1[i]]
    
    btw =  nx.algorithms.clique.number_of_cliques(G, nodes=target2)    
    btw2=np.zeros((len(target2)))
    for i in range(0, len(target2)):
        btw2[i] = btw[target2[i]]
      
    return [btw1, btw2, btw1.mean(), btw2.mean()]


def getNNtopological_measures(path_networks, i):
    file = path_networks + 'network' + str(i) + '.pickle' 
    exists = os.path.isfile(file)
    if not exists:
        print('Network', i, 'does not exist')
    else:
        with open(file, 'rb') as f:
            weights1_2, weights2_3, weights3_4, losses, accuracies = pickle.load(f)
            f.close()
        
        shape = weights1_2.shape
        epochs =  shape[0]
        input_size = shape[1]
        hidden1_size = shape[2]
        
        shape = weights2_3.shape
        hidden2_size = shape[2]  
        
        # print(epochs, input_size, hidden1_size, hidden2_size)

        target1 = [i for i in range(input_size, input_size+hidden1_size)]#neurons from 1st hidden
        target2 = [j for j in range(input_size+hidden1_size, input_size+hidden1_size+hidden2_size)]#neurons from 2nd hidden    
         
        fmap_h1 = np.zeros((epochs, 200, 8))
        fmap_h2 = np.zeros((epochs, 100, 8))
        
        for epoch in range(0,epochs):
            print('network ', i, ' epoch ', epoch)        
            start_time = time.time()
            t = (-999999999, -9999999) #nonsense, just giving a large threshold, nothing is removed from the graph
            directed=False #undirected graph
            G = getNNgraph(weights1_2[epoch], weights2_3[epoch], weights3_4[epoch], directed=directed, threshold = t)     
    
            #measure 1-> strength          
            m1, m2, _, _ = undirectedstrength_h1xh2(G, target1, target2)        
            fmap_h1[epoch, :, 0] = m1
            fmap_h2[epoch, :, 0] = m2
            
            #measure 2-> avg neighbor strength
            m1, m2, _, _ = average_neighbor_degree(G, target1, target2)        
            fmap_h1[epoch, :, 1] = m1
            fmap_h2[epoch, :, 1] = m2
                    
            #measure 3-> current flow closenness
            m1, m2, _, _ = CF_closeness(G, target1, target2)        
            fmap_h1[epoch, :, 2] = m1
            fmap_h2[epoch, :, 2] = m2
            
            G.clear()
            
            #next measures: thresholds negative connections, keeps only ppositive
            t = (-99999999, 0)
            directed=False
            G = getNNgraph(weights1_2[epoch], weights2_3[epoch], weights3_4[epoch], directed=directed, threshold = t)
            
            #measure 4-> bipartite local clustering
            m1, m2, _, _ = bipartite_clustering_h1xh2(G, target1, target2)        
            fmap_h1[epoch, :, 3] = m1
            fmap_h2[epoch, :, 3] = m2
            
            #measure 5-> subgraph centrality
            m1, m2, _, _ = subgraph_centrality(G, target1, target2)        
            fmap_h1[epoch, :, 4] = m1
            fmap_h2[epoch, :, 4] = m2
            
            #measure 6-> harmonic centrality
            m1, m2, _, _ = harmonic_centrality(G, target1, target2)        
            fmap_h1[epoch, :, 5] = m1
            fmap_h2[epoch, :, 5] = m2
                    
            #measure 7-> second order centrality
            m1, m2, _, _ = second_order_centrality(G, target1, target2)        
            fmap_h1[epoch, :, 6] = m1
            fmap_h2[epoch, :, 6] = m2
                   
            #measure 8-> number of cliques
            m1, m2, _, _ = number_of_cliques(G, target1, target2)        
            fmap_h1[epoch, :, 7] = m1
            fmap_h2[epoch, :, 7] = m2      
               
                
            G.clear()
            print(np.round(time.time() - start_time, decimals=3), 'seconds')
    
        return fmap_h1, fmap_h2

#this function returns only the top3 CN measures according to the paper
def getNNtopological_measures_top3(weights1_2, weights2_3, weights3_4):    
    shape = weights1_2.shape
    input_size = shape[0]
    hidden1_size = shape[1]
    
    shape = weights2_3.shape
    hidden2_size = shape[1]  

    target1 = [i for i in range(input_size, input_size+hidden1_size)]#neurons from 1st hidden
    target2 = [j for j in range(input_size+hidden1_size, input_size+hidden1_size+hidden2_size)]#neurons from 2nd hidden    
     
    fmap_h1 = np.zeros((200, 3))
    fmap_h2 = np.zeros((100, 3))
    
    G = getNNgraph(weights1_2, weights2_3, weights3_4)     

    #measure -> strength          
    m1, m2, _, _ = undirectedstrength_h1xh2(G, target1, target2)        
    fmap_h1[:, 0] = m1/76.09813499101438 #normalization parameters, obtained from the dataset average
    fmap_h2[:, 0] = m2/39.78247097041458
     
    G.clear()
    
    #next measures: thresholds negative connections, i.e. keeps only ppositive
    t = (-99999999, 0)
    G = getNNgraph(weights1_2, weights2_3, weights3_4, threshold = t)
    
    #measure -> bipartite local clustering
    m1, m2, _, _ = bipartite_clustering_h1xh2(G, target1, target2)
    m1 = m1  -0.40084583925905465
    m2 = m2  -0.32921994061865445       
    fmap_h1[:, 1] = m1/(0.48454784932402406 - 0.40084583925905465)
    fmap_h2[:, 1] = m2/(0.5009741383487208 - 0.32921994061865445)
    
    #measure -> subgraph centrality
    m1, m2, _, _ = subgraph_centrality(G, target1, target2)        
    fmap_h1[:, 2] = m1/5.995302849099099e+89
    fmap_h2[:, 2] = m2/1.5926423017865634e+89            
        
    G.clear()

    return fmap_h1, fmap_h2











































