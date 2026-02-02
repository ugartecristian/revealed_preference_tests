import numpy as np
import networkx as nx
from itertools import product


def GARP(p, x):
    # Returns True if GARP holds, False otherwise
    DRP, DRSP = direct_rev_prefs(p, x)
    RP = rp_graph(DRP)

    for e in RP.edges():
        if DRSP[e[1], e[0]]:
            return False

    return True


def SARP(p, x):
    # Returns True if SARP holds, False otherwise

    DRP, DRSP = direct_rev_prefs(p, x)
    RP = rp_graph(DRP)

    for e in RP.edges():
        if DRP[e[1], e[0]] and (not np.array_equal(x[e[0]], x[e[1]])):
            return False

    return True


def direct_rev_prefs(p, x):
    # Receives matrices of prices and consumption bundles
    #     Matrices are of N x K, where
    #         N is number of observations and
    #         K number of goods
    #
    # Returns N x N Matrices of Direct Revealed Preferences
    #   DRP Direct Revealed Preferences
    #   DRSP Direct Revealed Strict Preferences

    N = p.shape[0]

    e = np.dot(p, x.T)

    DRP = np.ones([N, N], dtype=bool)
    DRSP = np.zeros([N, N], dtype=bool)

    for i, j in product(range(N), range(N)):
        DRP[i, j] = (e[i, j] <= e[i, i])
        DRSP[i, j] = (e[i, j] < e[i, i])

    return DRP, DRSP


def rp_graph(DRP_d):
    # Receives Matrix of Directly Revealed Preferences
    # Returns Revealed Preferences as a nx DiGraph

    DRP = nx.from_numpy_matrix(DRP_d, create_using=nx.DiGraph)

    return nx.transitive_closure(DRP)
