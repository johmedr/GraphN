from __future__ import print_function

import os

import scipy.sparse as sp
import numpy as np

from keras.utils.data_utils import get_file
from keras.utils import to_categorical


def load_data(fname='DD'):
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

    return adjs, nodes, gphs_lbls, n, N, m
