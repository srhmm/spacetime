import copy

import networkx as nx
import numpy as np
import math
import itertools
from sklearn import preprocessing


def is_insignificant(gain, alpha=0.05):
    """ Significance by MDL no-hypercompression. """
    return gain < 0 or 2 ** (-gain) > alpha # gain must be over 4.3


def cantor_pairing(x, y):
    return int((x + y) * (x + y + 1) / 2 + y)


def dag_n_edges(adj):
    assert adj.shape[0] == adj.shape[1]
    return sum([len(np.where(adj[i] != 0)[0]) for i in range(len(adj))])


def logg(val):
    return 0 if val == 0 else math.log(val)


def data_scale(y):
    scaler = preprocessing.StandardScaler().fit(y)
    return scaler.transform(y)


def map_to_shifts(mp):
    return [1 if x != y else 0 for k, x in enumerate(mp) for y in mp[k + 1:]]


def shifts_to_map(shifts, n_c):
    mp = [0 for _ in range(n_c)]
    for ci in range(n_c):
        cur_idx = mp[ci]
        # assign all pairs (ci, c2) without a mechanism shift to the same group
        for ind, (c1, c2) in enumerate(itertools.combinations(range(n_c), 2)):
            if c1 != ci:
                continue
            if shifts[ind] == 0:
                mp[c2] = cur_idx
            else:
                mp[c2] = cur_idx + 1
    return mp


def pval_to_map(pval_mat, alpha=0.05):
    n_c = pval_mat.shape[0]
    mp = np.zeros(n_c)
    groups = [[e, e] for e in range(n_c)]
    for c1 in range(n_c):
        for c2 in range(c1+1, n_c):
            if pval_mat[c1][c2] > alpha:
                groups.append([c1, c2])
    G = nx.Graph()
    G.add_edges_from(groups)
    clusters = list(nx.connected_components(G))
    for i, c in enumerate(clusters): mp[list(c)] = i
    return mp


def pi_decrease_naming(map):
    indexes = np.unique(map, return_index=True)[1]
    nms = np.array([map[index] for index in sorted(indexes)])

    renms = [i for i in range(len(nms))]

    nmap = [renms[np.min(np.where(nms == elem))] for elem in map]

    assert (len(np.unique(nmap)) == len(np.unique(map)))
    return nmap

