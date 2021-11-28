import numpy as np
import os
import time
import random

path = os.getcwd()

#sizes = ["small", "medium", "large", "xl", "xxl"]
sizes = ["xl"]
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

    print("Dataset", size, "has", edgelist.shape, "edges and", papers.shape, "nodes.")

    known_labels = np.where(labels != -1)[0]
    split_array = np.split(known_labels, [int(0.8*len(known_labels))]) # doesn't split 80%
    train_labels = split_array[0]
    test_labels = split_array[1]

    print("Number of unique labels:", len(np.unique(labels))-1) #subtract -1 for the unknown labels
    print("Number of train labels:", train_labels.shape)
    print("Number of unknown labels:", test_labels.shape)
    print("Total number of known labels:", known_labels.shape)

    labels_propagation = np.copy(labels)
    labels_propagation.put([test_labels],-1) #this works
    mask = np.ones(labels_propagation.shape, bool)
    mask[train_labels] = False

    start_time = time.time()
    loop_count = 0 # safety valve if label propagation gets stuck on loop for some reason
    while True:
        #random.shuffle(papers)
        converged = True
        for node in papers[mask]: # we don't want to change the true labels
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
                max_prop_labels = [x for x in d.keys() if d[x] == d[max(d, key=d.get)]]
                #prop_label = random.choice(max_prop_labels)
                prop_label = max_prop_labels[0]
                if labels_propagation[node] != prop_label: # if propagated label is the same as original, we don't need to do anything
                        labels_propagation[node] = prop_label
                        converged = False # we never reach this point if labels are not updated
        loop_count += 1
        if converged:
            print("Converged.")
            print("Label propagation for dataset", size, "took %s seconds" % (time.time() - start_time))
            break
        if loop_count >= 30 and not -1 in labels_propagation:
            print("Loop count reached.")
            print("Label propagation for dataset", size, "took %s seconds" % (time.time() - start_time))
            break

    # Check accuracy

    count = 0
    correct = 0
    for i in range(len(labels[test_labels])):
        if labels[test_labels][i] == labels_propagation[test_labels][i]:
            correct += 1
        count += 1

    print("Accuracy for dataset of size", size, ":", correct/count)



            






