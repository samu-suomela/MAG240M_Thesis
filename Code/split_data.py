import numpy as np
import os
import time

path = os.getcwd()

size = "large"

edgelist = np.load("{}/test_data_collection/test_data_{}/cites_{}.npy".format(path,size,size))
labels = np.load("{}/test_data_collection/test_data_{}/labels_{}.npy".format(path,size,size))
papers = np.load("{}/test_data_collection/test_data_{}/papers_{}.npy".format(path,size,size))

for i in range(len(labels)):
    if np.isnan(labels[i]):
        labels[i] = -1

edgelist_reindexed = []
papers_reindexed = []
mapping = {}

# reindex edges and papers to start from 0
for i in range(len(papers)):
    mapping[papers[i]] = i
    papers_reindexed.append(i)
papers = papers_reindexed

for i in range(len(edgelist.T)):
    edge = edgelist[:,i]
    edgelist_reindexed.append([mapping[edge[0]],mapping[edge[1]]])
edgelist = edgelist_reindexed

known_labels = np.where(labels != -1)[0]
split_array = np.split(known_labels, [int(0.8*len(known_labels))]) # doesn't split 80%
train_labels = split_array[0]
test_labels = split_array[1]

labels_propagation = np.copy(labels)
labels_propagation.put([test_labels],-1) #this works


start_time = time.time()

while True:
    for node in papers:
        converged = True
        neighbors = []
        for edge in edgelist:
            if edge[0] == node:
                neighbors.append(edge[1])
            if edge[1] == node:
                neighbors.append(edge[0])
        neighbor_labels = labels_propagation[neighbors]
        d = {}
        for label in neighbor_labels:
            if not label == -1:
                if label in d:
                    d[label] += 1
                else:
                    d[label] = 1
        if d:
            prop_label = max(d, key = d.get)
            if labels_propagation[node] != prop_label: # if propagated label is the same as original, we don't need to do anything
                    labels_propagation[node] = prop_label
                    converged = False # we never reach this point if labels are not updated

    if converged:
        print("Converged.")
        print("Label propagation took %s seconds" % (time.time() - start_time))
        break
    

# Check accuracy

propagated = labels_propagation[test_labels]
original = labels[test_labels]

count = 0
correct = 0
for i in range(len(propagated)):
    if propagated[i] == original[i]:
        correct += 1
    count += 1

print("Accuracy:", correct/count)



            






