import numpy as np
import pandas as pd
import time
from ogb.lsc import MAG240MDataset

dataset = MAG240MDataset(root = "/wrk/users/sjsuomel/data/")
paper_to_paper = dataset.edge_index('paper', 'paper')
real_labels = dataset.paper_label

labels = np.copy(real_labels)

print("Edges and labels loaded")

loop_count = 0
start_time = time.time()

while True:
    loop_count += 1 # safety valve in case we never reach convergence
    print("Loop count:", loop_count)
    converged = True
    for node in range(len(labels)):
        neighbor_labels = {}
        target_nodes = paper_to_paper[1][paper_to_paper[0]==node]
        source_nodes = paper_to_paper[0][paper_to_paper[1]==node]
        neighbors = np.concatenate((target_nodes,source_nodes),axis = None)       
        for neighbor in neighbors:
            if not np.isnan(labels[neighbor]) and not labels[neighbor] == -1:
                if labels[neighbor] in neighbor_labels:
                    neighbor_labels[labels[neighbor]] = neighbor_labels[labels[neighbor]] + 1
                else:
                    neighbor_labels[labels[neighbor]] = 1
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