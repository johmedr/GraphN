from __future__ import print_function

import os

import scipy.sparse as sp
import numpy as np

from keras.utils.data_utils import get_file
from keras.utils import to_categorical


def load_data(fname='DD', test_split=.2, rng_seed=None):
    print("Loading {} dataset".format(fname))

    fpath = get_file(
        fname=fname,
        origin="https://orion.exposed/perso/wp-content/uploads/2018/12/dd.tar.gz",
        untar=True,
        file_hash="3a0ad2dbe5393313dc14690c1c8c6fff",
        hash_algorithm="md5"
    )

    # Get graph index for each node
    gphs_indx = []
    with open(os.path.join(fpath, "DD_graph_indicator.txt"), 'r') as f:
        for line in f:
            gphs_indx.append(int(line) - 1)
    # n: number of nodes
    n = len(gphs_indx)

    # Get graph labels
    gphs_lbls = []
    with open(os.path.join(fpath, "DD_graph_labels.txt"), 'r') as f:
        for line in f:
            gphs_lbls.append(int(line) - 1)
    gphs_lbls = np.array(gphs_lbls, dtype=np.float32)
    # N: number of graphs
    N = len(gphs_lbls)

    # Get edges
    indx = ([], [])
    with open(os.path.join(fpath, "DD_A.txt"), 'r') as f:
        for line in f:
            l = line.split(",")
            indx[0].append(int(l[0]) - 1)
            indx[1].append(int(l[1]) - 1)
    # m: number of edges
    m = len(indx[0])

    # Extract edges for each graph
    cond_indx = [([], []) for _ in range(N)]
    for i in range(n):
        cond_indx[gphs_indx[i]][0].append(indx[0][i])
        cond_indx[gphs_indx[i]][1].append(indx[1][i])
    # Build the corresponding adjacency matrix
    adjs = [sp.csr_matrix((np.ones((len(g[0]),)), g), shape=(n, n))
            for g in cond_indx]

    # Get nodes labels
    lbls = []
    with open(os.path.join(fpath, "DD_node_labels.txt"), 'r') as f:
        for line in f:
            lbls.append(int(line) - 1)
    # Turn labels in one-hot features
    nodes = to_categorical(lbls)

    np.random.seed(rng_seed)
    p = np.random.permutation(range(N))

    train_split = int((1 - test_split) * N)

    train_adjs = [adjs[i] for i in p[:train_split]]
    train_lbls = gphs_lbls[p[:train_split]]

    test_adjs = [adjs[i] for i in p[train_split:]]
    test_lbls = gphs_lbls[p[train_split:]]

    return nodes, (train_adjs, train_lbls), (test_adjs, test_lbls), n, N, m


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm
