from ogb.lsc import MAG240MDataset
import numpy as np

dataset = MAG240MDataset(root = "/Volumes/Seagate Backup Plus Drive/Gradu/")

#edge_index_writes = dataset.edge_index('author', 'paper')
edge_index_cites = dataset.edge_index('paper', 'paper')
#edge_index_affiliated = dataset.edge_index('author', 'institution')




starting_paper = np.where(dataset.paper_label==1)[0][0] # select the first paper with label=1
papers = np.array(starting_paper)

citations = edge_index_cites[:,edge_index_cites[0]==starting_paper][1] # select papers that starting_paper cites
papers = np.concatenate((papers, citations), axis=None)

n = 5
for i in range(n):
    mask_cites = np.in1d(edge_index_cites[0],papers) # boolean array with True values for citations
    new_papers = edge_index_cites[:,mask_cites][1] # cited papers of the original array
    papers = np.concatenate((papers,new_papers), axis=None) #concatenate new citations to the array of all papers
    print("n:", i, "papers", len(papers))









# paper_array = edge_index_writes[1,edge_index_writes[0]==1080565]
# mask_cites = np.in1d(edge_index_cites[0],paper_array)
# cites_array = edge_index_cites[:,mask_cites]

# n = 2

# for i in range(n):
#     paper_array = np.concatenate((paper_array, np.unique(cites_array[1])))
#     mask_author = np.in1d(edge_index_writes[1],paper_array) 
#     author_array = edge_index_writes[:,mask_author][0]
#     mask_cites = np.in1d(edge_index_cites[0],paper_array)
#     cites_array = edge_index_cites[:,mask_cites]
#     print(paper_array)

# mask_cites = np.in1d(edge_index_cites[0],paper_array)
# cites_array = edge_index_cites[:,mask_cites]
# paper_array = np.concatenate((paper_array, np.unique(cites_array[1])))
# mask_author = np.in1d(edge_index_writes[1],paper_array)
# author_array = edge_index_writes[:,mask_author][0]
# print(paper_array)
