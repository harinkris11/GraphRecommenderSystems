from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math, time
from copy import deepcopy
import multiprocessing as mp
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
import warnings
import matplotlib.pyplot as plt
from collections import defaultdict
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csr_matrix.shape

    def __getitem__(self, row_selector):
        indices = np.concatenate(self.indices[row_selector])
        data = np.concatenate(self.data[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))
        shape = [indptr.shape[0] - 1, self.shape[1]]
        return ssp.csr_matrix((data, indices, indptr), shape=shape)


class SparseColIndexer:
    def __init__(self, csc_matrix):
        data = []
        indices = []
        indptr = []

        for col_start, col_end in zip(csc_matrix.indptr[:-1], csc_matrix.indptr[1:]):
            data.append(csc_matrix.data[col_start:col_end])
            indices.append(csc_matrix.indices[col_start:col_end])
            indptr.append(col_end - col_start)

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.shape = csc_matrix.shape

    def __getitem__(self, col_selector):
        indices = np.concatenate(self.indices[col_selector])
        data = np.concatenate(self.data[col_selector])
        indptr = np.append(0, np.cumsum(self.indptr[col_selector]))

        shape = [self.shape[0], indptr.shape[0] - 1]
        return ssp.csc_matrix((data, indices, indptr), shape=shape)

# class to generate the test dataset for the experiment
class MyTestDataset(InMemoryDataset):
    def __init__(self, root, A, links, labels, h, max_nodes_per_hop, 
                  class_values):
        self.Arow = SparseRowIndexer(A)
        self.Acol = SparseColIndexer(A.tocsc())
        self.links = links
        self.labels = labels
        self.h = h
        self.max_nodes_per_hop = max_nodes_per_hop
        self.class_values = class_values
        super(MyTestDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        name = 'data.pt'
        return [name]

    def process(self):
        # Extract enclosing subgraphs to pass to IGMC model.
        data_list = links2subgraphs(self.Arow, self.Acol, self.links, self.labels, self.h, 
                                    self.max_nodes_per_hop, 
                                    self.class_values)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        del data_list

# class to generate the train dataset for the experiment
class MyTrainDataset(Dataset):
    def __init__(self, root, A, links, labels, h, max_nodes_per_hop, class_values):
        super(MyTrainDataset, self).__init__(root)
        self.Arow = SparseRowIndexer(A)
        self.Acol = SparseColIndexer(A.tocsc())
        self.links = links
        self.labels = labels
        self.h = h
        self.max_nodes_per_hop = max_nodes_per_hop
        self.class_values = class_values

    def len(self):
        return self.__len__()

    def __len__(self):
        return len(self.links[0])
    
    # function call to extract the subgraph post labelling and construct pytorch geometric graphs 
    def get(self, idx):
        i, j = self.links[0][idx], self.links[1][idx]
        g_label = self.labels[idx]
        tmp = subgraph_extraction_labeling(
            (i, j), self.Arow, self.Acol, self.h, self.max_nodes_per_hop, self.class_values, g_label )
        return construct_pyg_graph(*tmp)

# function to extract the subgraphs and convert to pytorch graphs for test set.
def links2subgraphs(Arow, 
                    Acol, 
                    links, 
                    labels, 
                    h=1, 
                    max_nodes_per_hop=None, 
                    class_values=None, 
                    ):
    # extract enclosing subgraphs
    print('Enclosing subgraph extraction begins...')
    g_list = []
    start = time.time()
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap_async(
        subgraph_extraction_labeling, 
        [
            ((i, j), Arow, Acol, h, max_nodes_per_hop, class_values, g_label) 
            for i, j, g_label in zip(links[0], links[1], labels)
        ]
    )
    remaining = results._number_left
    pbar = tqdm(total=remaining)
    while True:
        pbar.update(remaining - results._number_left)
        if results.ready(): break
        remaining = results._number_left
        time.sleep(1)
    results = results.get()
    pool.close()
    pbar.close()
    end = time.time()
    print("Time elapsed for subgraph extraction: {}s".format(end-start))
    print("Transforming to pytorch_geometric graphs...")
    g_list = []
    pbar = tqdm(total=len(results))
    while results:
        tmp = results.pop()
        g_list.append(construct_pyg_graph(*tmp))
        pbar.update(1)
    pbar.close()
    end2 = time.time()
    print("Time elapsed for transforming to pytorch_geometric graphs: {}s".format(end2-end))
    return g_list

# function to extract subgraphs using BFS style algorithm
def subgraph_extraction_labeling(ind, Arow, Acol, h=1, max_nodes_per_hop=None, 
                                 class_values=None, y=1):
    # extract the h-hop enclosing subgraph around link 'ind'
    u_nodes, v_nodes = [ind[0]], [ind[1]]
    u_dist, v_dist = [0], [0]
    u_visited, v_visited = set([ind[0]]), set([ind[1]])
    u_fringe, v_fringe = set([ind[0]]), set([ind[1]])
    for dist in range(1, h+1):
        v_fringe, u_fringe = neighbors(u_fringe, Arow), neighbors(v_fringe, Acol)
        u_fringe = u_fringe - u_visited
        v_fringe = v_fringe - v_visited
        u_visited = u_visited.union(u_fringe)
        v_visited = v_visited.union(v_fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(u_fringe):
                u_fringe = random.sample(u_fringe, max_nodes_per_hop)
            if max_nodes_per_hop < len(v_fringe):
                v_fringe = random.sample(v_fringe, max_nodes_per_hop)
        if len(u_fringe) == 0 and len(v_fringe) == 0:
            break
        u_nodes = u_nodes + list(u_fringe)
        v_nodes = v_nodes + list(v_fringe)
        u_dist = u_dist + [dist] * len(u_fringe)
        v_dist = v_dist + [dist] * len(v_fringe)
    subgraph = Arow[u_nodes][:, v_nodes]
    # remove link between target nodes
    subgraph[0, 0] = 0
    
    # prepare pyg graph constructor input
    u, v, r = ssp.find(subgraph)  # r is 1, 2... (rating labels + 1)
    v += len(u_nodes)
    r = r - 1  # transform r back to rating label
    num_nodes = len(u_nodes) + len(v_nodes)
    node_labels = [x*2 for x in u_dist] + [x*2+1 for x in v_dist]
    max_node_label = 2*h + 1
    y = class_values[y]
            
    return u, v, r, node_labels, max_node_label, y

# function to generate pytorch geometric graph for extracted subgraph
def construct_pyg_graph(u, v, r, node_labels, max_node_label, y):
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)  
    edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])], 0)
    edge_type = torch.cat([r, r])
    x = torch.FloatTensor(one_hot(node_labels, max_node_label+1))
    y = torch.FloatTensor([y])
    data = Data(x, edge_index, edge_type=edge_type, y=y)
    return data


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    if not fringe:
        return set([])
    return set(A[list(fringe)].indices)


def one_hot(idx, length):
    idx = np.array(idx)
    x = np.zeros([len(idx), length])
    x[np.arange(len(idx)), idx] = 1.0
    return x

# function to generate subgraph back from pytorch graph
def PyGGraph_to_nx(data):
    edges = list(zip(data.edge_index[0, :].tolist(), data.edge_index[1, :].tolist()))
    g = nx.from_edgelist(edges)
    g.add_nodes_from(range(len(data.x)))  # in case some nodes are isolated
    # transform r back to rating label
    edge_types = {(u, v): data.edge_type[i].item() for i, (u, v) in enumerate(edges)}
    nx.set_edge_attributes(g, name='type', values=edge_types)
    node_types = dict(zip(range(data.num_nodes), torch.argmax(data.x, 1).tolist()))
    nx.set_node_attributes(g, name='type', values=node_types)
    g.graph['rating'] = data.y.item()
    return g

# function to compare various losses as per some hyperparameter
def plot_graph(losses, labels, title, xlabel, ylabel):
    plt.figure(figsize = (10,8))
    epochs = range(1, len(losses[0])+1)
    for loss, label in zip(losses,labels):
        plt.plot(epochs, loss, label = label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
    return

#function to generate average precision@k, recall@k, f1score as per preditions, true labels
def precision_recall_calculation(prediction_ratings, true_ratings, user_indices, topK, threshold=3.5):
    # First map the predictions to each user.
    user_predict_true = defaultdict(list)
    for user_id, true_rating, predicted_rating in zip(user_indices, prediction_ratings, true_ratings):
        user_predict_true[user_id].append((predicted_rating, true_rating))
    precisions = dict()
    recalls = dict()
    for user_id, user_ratings in user_predict_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        # Number of relevant items
        no_of_relevant_items = sum((true_rating >= threshold) for (predicted_rating, true_rating) in user_ratings)
        # Number of recommended items in topK
        no_of_recommended_items = sum((predicted_rating >= threshold) for (predicted_rating, true_rating) in user_ratings[:topK])
        # Number of relevant and recommended items in topK
        no_of_relevant_and_recommended_items = sum(((true_rating >= threshold) and (predicted_rating >= threshold)) for (predicted_rating, true_rating) in user_ratings[:topK])
        # Precision: Proportion of recommended items that are relevant
        precisions[user_id] = no_of_relevant_and_recommended_items / no_of_recommended_items if no_of_recommended_items != 0 else 1
        # Recall: Proportion of relevant items that are recommended
        recalls[user_id] = no_of_relevant_and_recommended_items / no_of_relevant_items if no_of_relevant_items != 0 else 1

    # Averaging the values for all users
    average_precision=sum(precision for precision in precisions.values()) / len(precisions)
    average_recall=sum(recall for recall in recalls.values()) / len(recalls)
    F_score=(2*average_precision*average_recall) / (average_precision + average_recall)
    return [average_precision, average_recall, F_score]
