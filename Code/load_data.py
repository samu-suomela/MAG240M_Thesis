from ogb.lsc import MAG240MDataset
import numpy as np

#dataset = MAG240MDataset(root = "/wrk/users/sjsuomel/data/")
dataset = MAG240MDataset(root = "/Volumes/Seagate Backup Plus Drive/Gradu/")
# Basic properties

print(dataset.num_papers) # number of paper nodes
print(dataset.num_authors) # number of author nodes
print(dataset.num_institutions) # number of institution nodes
print(dataset.num_paper_features) # dimensionality of paper features
print(dataset.num_classes) # number of subject area classes

# get i-th paper feature
i = 1234
print(dataset.paper_feat[i]) # only i-th data is loaded into memory

# get the feature matrix storing features of papers in idx_arr
idx_arr = np.array([1,10,100,1000,10000])
print(dataset.paper_feat[idx_arr]) # only the 5 data is loaded into memory