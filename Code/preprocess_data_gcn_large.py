import numpy as np
from scipy.sparse import csr_matrix
import pickle
import os
import pandas as pd

path = os.getcwd()
size = "large"

edgelist = np.load("{}/test_data_collection/test_data_{}/cites_{}.npy".format(path,size,size))
papers = np.load("{}/test_data_collection/test_data_{}/papers_{}.npy".format(path,size,size))
labels = np.load("{}/test_data_collection/test_data_{}/labels_{}.npy".format(path,size,size)) # 153 classes in the data

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
train_test_index_split= int(0.8*len(known_labels))
train_index = []
test_index = []
j = 0
label_matrix = np.zeros((len(labels),153))
for i in range(len(labels)):
    if not labels[i] == -1:
        label_matrix[i,int(labels[i])] = 1
        if j < train_test_index_split:
            train_index.append(i)
            j += 1
        else:
            test_index.append(i)

train_labels = label_matrix[train_index,:]
with open('/wrk/users/sjsuomel/gcn/gcn/data/ind.large.y', 'wb') as handle:
    pickle.dump(train_labels, handle, protocol=4)

test_labels = label_matrix[test_index,:]
with open('/wrk/users/sjsuomel/gcn/gcn/data/ind.large.ty', 'wb') as handle:
    pickle.dump(test_labels, handle, protocol=4)

label_matrix_no_test = np.delete(label_matrix, test_index, axis=0) # remove test index
with open('/wrk/users/sjsuomel/gcn/gcn/data/ind.large.ally', 'wb') as handle:
    pickle.dump(label_matrix_no_test, handle, protocol=4)

with open('/wrk/users/sjsuomel/gcn/gcn/data/ind.large.test.index', 'w') as f:
    for index in test_index:
        f.write(str(index) + "\n")

feature_matrix = np.load("{}/test_data_collection/test_data_{}/features_{}.npy".format(path,size,size))
feature_matrix_no_test = np.delete(feature_matrix, test_index, axis=0) # remove test index
feature_matrix_train = feature_matrix[train_index]
feature_matrix_test = feature_matrix[test_index]

sparse_feature_matrix = csr_matrix(feature_matrix_no_test)
with open('/wrk/users/sjsuomel/gcn/gcn/data/ind.large.allx', 'wb') as handle:
    pickle.dump(sparse_feature_matrix, handle, protocol=4)

sparse_feature_matrix_train = csr_matrix(feature_matrix_train)
with open('/wrk/users/sjsuomel/gcn/gcn/data/ind.large.x', 'wb') as handle:
    pickle.dump(sparse_feature_matrix_train, handle, protocol=4)

sparse_feature_matrix_test = csr_matrix(feature_matrix_test)
with open('/wrk/users/sjsuomel/gcn/gcn/data/ind.large.tx', 'wb') as handle:
    pickle.dump(sparse_feature_matrix_test, handle, protocol=4)

G = dict()
for edge in edgelist:
    if edge[0] in G:
        G[edge[0]].append(int(edge[1]))
    else:
        G[edge[0]] = [int(edge[1])]

    if edge[1] in G: 
        G[edge[1]].append(int(edge[0]))
    else:
        G[edge[1]] = [int(edge[0])]

for no_edge in np.setdiff1d(papers_reindexed, np.unique(edgelist)):
    G[no_edge] = [int(no_edge)]

with open('/wrk/users/sjsuomel/gcn/gcn/data/ind.large.graph', 'wb') as handle:
    pickle.dump(G, handle, protocol=4)

print("Done!")