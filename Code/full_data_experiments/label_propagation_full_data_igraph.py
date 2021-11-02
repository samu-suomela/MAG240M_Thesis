from igraph import *
import numpy as np
import pandas as pd
import time
from ogb.lsc import MAG240MDataset

dataset = MAG240MDataset(root = "/wrk/users/sjsuomel/data/")
paper_to_paper = dataset.edge_index('paper', 'paper').T.tolist()
labels = dataset.paper_label

print("Edges and labels loaded")

start_time = time.time()

G = Graph(edges = paper_to_paper)

print("Graph creation took %s seconds" % (time.time() - start_time))

loop_count = 0
start_time = time.time()

while True:
    loop_count += 1 # safety valve in case we never reach convergence
    print("Loop count:", loop_count)
    converged = True
    for node in G.vs.indices:
        neighbor_labels = {}
        neighbors = list(G.neighbors(node))
        for i in neighbors:
            if not np.isnan(labels[i]) and not labels[i] == -1: # we don't want to propagate nan-values or -1-values
                if labels[i] in neighbor_labels:
                    neighbor_labels[labels[i]] = neighbor_labels[labels[i]] + 1
                else:
                    neighbor_labels[labels[i]] = 1
        if neighbor_labels:  # if dict contains a value
            #prop_label = int(np.random.choice([key for key in neighbor_labels.keys() 
            #                                if neighbor_labels[key]==max(neighbor_labels.values())])) # Select most common label at random
            prop_label = max(neighbor_labels, key = neighbor_labels.get)
            if labels[node] != prop_label: # if propagated label is the same as original, we don't need to do anything
                labels[node] = prop_label
                converged = False # we never reach this point if labels are not updated

    if converged:
        print("Converged.")
        break
    if loop_count == 15:
        print("Loopcount reached.")
        break

print("Label propagation took %s seconds" % (time.time() - start_time))

np.save("/wrk/users/sjsuomel/results/label_propagation_results", labels) # save predicted labels for evaluation

print("File successfully saved!")