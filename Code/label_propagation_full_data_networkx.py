import networkx as nx
import numpy as np
import pandas as pd
import time
from ogb.lsc import MAG240MDataset

dataset = MAG240MDataset(root = "/wrk/users/sjsuomel/data/")
paper_to_paper = dataset.edge_index('paper', 'paper')
labels = dataset.paper_label

print("Edges and labels loaded")

label_dict = {}
for i in range(len(labels)):
    label_dict[i] = labels[i]

G = nx.Graph()
G.add_edges_from(paper_to_paper.T)
nx.set_node_attributes(G,label_dict,name="label")

print("Graph created")

loop_count = 0
start_time = time.time()

while True:
    loop_count += 1 # safety valve in case we never reach convergence
    print("Loop count:", loop_count)
    converged = True
    for node in G.nodes:
        neighbor_labels = {}
        neighbors = list(G.neighbors(node))
        for i in neighbors:
            if not np.isnan(label_dict[i]) and not label_dict[i] == -1: # we don't want to propagate nan-values or -1-values
                if label_dict[i] in neighbor_labels:
                    neighbor_labels[label_dict[i]] = neighbor_labels[label_dict[i]] + 1
                else:
                    neighbor_labels[label_dict[i]] = 1
        if neighbor_labels:  # if dict contains a value
            prop_label = np.random.choice([key for key in neighbor_labels.keys() 
                                            if neighbor_labels[key]==max(neighbor_labels.values())]) # Select most common label at random
            if not labels[node] == prop_label: # if propagated label is the same as original, we don't need to do anything
                G.nodes[node]["label"] = prop_label
                labels[node] = prop_label
                label_dict[node] = prop_label
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