import numpy as np
import networkx as nx
from itertools import product


def homothetic(p, x):  # Check Homothetic Axiom of Revealed Preferences

    e = np.log(np.dot(p, x.T))

    return no_negative_cycles(e)


def quasilinear(p, x):  # Check Cyclical Monotonicity

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
