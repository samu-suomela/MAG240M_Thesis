from ogb.lsc import MAG240MDataset
import numpy as np
import os

path = os.getcwd()

dataset = MAG240MDataset(root = "/Volumes/Seagate Backup Plus Drive/Gradu/")
#dataset = MAG240MDataset(root = "{}/data".format(path))

edge_index_cites = dataset.edge_index('paper', 'paper')

starting_paper = np.where(dataset.paper_label==110)[0][0]   # select the first paper with label=110, selected due to
                                                            # neighbors having plenty of labels.
papers = np.array(starting_paper)

citations = edge_index_cites[:,edge_index_cites[0]==starting_paper][1] # select papers that starting_paper cites
papers = np.concatenate((papers, citations), axis=None)

sizes = ["small","medium","large","xl", "xxl"]

n = 6

for i in range(n):
    mask_cites = np.in1d(edge_index_cites[0],papers) # boolean array with True values for citations
    new_papers = edge_index_cites[:,mask_cites][1] # cited papers of the original array
    papers = np.unique(np.concatenate((papers,new_papers), axis=None)) #concatenate new citations to the array of all papers
    if i >= 1:
        paper_to_paper = edge_index_cites[:,mask_cites] # citation array or edgelist
        labels = dataset.paper_label[papers] # labels of selected papers
        count = 0
        for j in range(len(labels)):
            if np.isnan(labels[j]):
                labels[j] = -1 # label nan-values to -1, indicating that they are unknown
            else:
                count += 1
        print(count)
        features = dataset.paper_feat[papers] # features of selected papers 

        edgelist_reindexed = []
        papers_reindexed = []
        mapping = {}

        # reindex edges and papers to start from 0
        for i in range(len(papers)):
            mapping[papers[i]] = i
            papers_reindexed.append(i)
        papers = papers_reindexed

        for i in range(len(paper_to_paper.T)):
            edge = paper_to_paper[:,i]
            edgelist_reindexed.append([mapping[edge[0]],mapping[edge[1]]])
        paper_to_paper = edgelist_reindexed


        np.save('{}/test_data_collection/test_data_{}/papers_{}'.format(path, sizes[i-1], sizes[i-1]), papers)
        np.save('{}/test_data_collection/test_data_{}/cites_{}'.format(path, sizes[i-1], sizes[i-1]), paper_to_paper)
        np.save('{}/test_data_collection/test_data_{}/labels_{}'.format(path, sizes[i-1], sizes[i-1]), labels)
        np.save('{}/test_data_collection/test_data_{}/features_{}'.format(path, sizes[i-1], sizes[i-1]), features)
