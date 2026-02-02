import numpy as np
import networkx as nx
from itertools import product


# Implements tests for:
#   - Homothetic Axiom of Revealed Preferences HARP and
#   - Cyclical Monotonicity, which is the test for rationalization by a
#     quasilinear utility

# Functions receive as input numpy matrices p and x, where
#   p is matrix of prices
#   x is matrix of consumption bundles
#       both matrices have N rows and K columns
#       p[i, j] is price of good j in observation i
#       x[i, j] is consumption of good j in observation i

#       IMPORTANT: prices must be normalized such that np.dot(p[i], x[i].T) = 1
#       for every observation i in range(N)


def HARP(p, x):  # Check Homothetic Axiom of Revealed Preferences

    e = np.log(np.dot(p, x.T))

    return no_negative_cycles(e)


def cyclical_monotonicity(p, x):  # Check Cyclical Monotonicity

    e = np.dot(p, x.T)

    return no_negative_cycles(e)


def no_negative_cycles(e):
    # Check whether data has negative cycles
    #   For HARP, use as e the matrix with logarithm of expenditures
    #   For cyclical monotonicity, use as e the matrix with expenditures
    # Returns:
    #   True if there are no negative cycles
    #   False otherwise

    N = e.shape[0]

    G = nx.DiGraph()
    G.add_nodes_from(range(N))

    for i, j in product(range(N), range(N)):
        if i != j:
            G.add_edge(i, j, weight=e[i, j] - e[i, i])

    try:
        nx.find_negative_cycle(G)
        return False
    except nx.NetworkXUnbounded:
        return False
    except nx.NetworkXError:
        return True

    return True
