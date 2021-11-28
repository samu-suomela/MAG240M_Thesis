import networkx as nx
import numpy as np
import pandas as pd
import time
import os

path = os.getcwd()

sizes = ["small", "medium", "large", "xl","xxl"]
#sizes = ["small"]
for size in sizes:

    edgelist = np.load("{}/test_data_collection/test_data_{}/cites_{}.npy".format(path,size,size))
    labels = np.load("{}/test_data_collection/test_data_{}/labels_{}.npy".format(path,size,size))
    papers = np.load("{}/test_data_collection/test_data_{}/papers_{}.npy".format(path,size,size))

    edgelist_reindexed = []
    papers_reindexed = []
    mapping = {}

    # reindex edges and papers to start from 0
    for k in range(len(papers)):
        mapping[papers[k]] = k
        papers_reindexed.append(k)
    papers = np.array(papers_reindexed)

    for m in range(len(edgelist.T)):
        edge = edgelist[:,m]
        edgelist_reindexed.append([mapping[edge[0]],mapping[edge[1]]])
    edgelist = np.array(edgelist_reindexed)

    known_labels = np.where(labels!=-1)[0]
    split_array = np.split(known_labels, [int(0.8*len(known_labels))])
    train_labels = split_array[0]
    test_labels = split_array[1]

    print("Number of unique labels:", len(np.unique(labels))-1) #subtract -1 for the unknown labels
    print("Number of train labels:", train_labels.shape)
    print("Number of unknown labels:", test_labels.shape)
    print("Total number of known labels:", known_labels.shape)

    labels_harmonic = np.copy(labels)
    labels_harmonic.put([test_labels], -1) #this works

    G = nx.Graph()
    G.add_edges_from(edgelist)
    for node in range(G.number_of_nodes()):
        if not labels_harmonic[node] == -1:
            G.nodes[node]["label"] = labels_harmonic[node]
    print("Dataset", size, "has", G.number_of_edges(), "edges and", G.number_of_nodes(), "nodes.")
    
    start_time = time.time()
    predicted = np.array(nx.algorithms.node_classification.harmonic_function(G))
    end_time = time.time() - start_time

    count = 0
    correct = 0

    for i in range(len(labels[test_labels])):
        if labels[test_labels][i] == predicted[test_labels][i]:
            correct += 1
        count += 1

    print("Accuracy for dataset", size, ":", correct/count)
    print("Algorithm took", end_time, "seconds.")
