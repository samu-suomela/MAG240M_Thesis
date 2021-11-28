import numpy as np
import os

path = os.getcwd()
sizes = ["small", "medium", "large", "xl"]

for size in sizes:
    edgelist = np.load("{}/test_data_collection/test_data_{}/cites_{}.npy".format(path,size,size))
    labels = np.load("{}/test_data_collection/test_data_{}/labels_{}.npy".format(path,size,size))
    papers = np.load("{}/test_data_collection/test_data_{}/papers_{}.npy".format(path,size,size))
    print("Size:", size)
    print("Number of nodes:", len(papers))
    print("Number of edges:", edgelist.shape)
    print("Number of labels:", len(np.where(labels!=-1)[0]))
    print("Number of unique labels:", len(np.unique(labels))-1)

