import numpy as np
from garp_sarp import GARP, SARP, direct_rev_prefs, rp_graph


def smooth_GARP(p, x):
    # Returns True if Smooth GARP holds, False Otherwise

    q = modified_dataset(p, x)

    return GARP(q, x)


def smooth_SARP(p, x):
    #  Returns True if Smooth SARP holds, False Otherwise
    N = p.shape[0]
    q = one_modification(p, x)
    for i in range(N):
        if np.dot(q[i], x[i].T) < 1:
            return False

    return SARP(q, x)


def SSARP(p, x):
    # Returns True is Strong SARP holds, False otherwise

    if not SARP(p, x):
        return False
    else:
        N = p.shape[0]
        for i in range(N):
            for j in range(i + 1, N):
                if np.array_equal(x[i], x[j]) and \
                   not np.array_equal(p[i], p[j]):

                    return False

    return True


def modified_dataset(p, x):
    # Return prices in fixed point of data set modification
    end = False

    q = p.copy()
    while not end:
        q_aux = q.copy()
        q = one_modification(q, x)

        end = np.array_equal(q, q_aux)

    return q


def one_modification(p, x):
    # Return prices in data set modification Gamma (Definition 7)
    N = p.shape[0]
    q = p.copy()

    DRP, DRSP = direct_rev_prefs(q, x)
    RP_G = rp_graph(DRP)

    # Construct indifference sets and compute minimum price
    indif = {}
    for i in range(N):
        indif[i] = []
        for j in range(N):
            if ((i, j) in RP_G.edges()) and ((j, i) in RP_G.edges()):
                indif[i].append(j)

    for i in range(N):
        q[i] = np.minimum.reduce([q[j] for j in indif[i]])

    return q
