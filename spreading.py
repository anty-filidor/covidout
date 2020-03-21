import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io_scr import read_net


def sperad(G: nx.Graph, ill_node, visited: list=[]) -> nx.Graph:
    """
    This method spreads disease from given node to the entire network
    :param G: graph within disease is being spread
    :param ill_node: ill node
    :param visited: taboo list of nodes affected. used in the recursion
    :return: nodes which were affected
    """

    # set status of ill node as 100% infected
    G.nodes[ill_node]['status'] = 1

    # check if neighbour nodes are in taboo list
    nbrs = [*G[ill_node].keys()]
    nbrs = [n for n in nbrs if n not in visited]
    # print(nbrs)

    # take it's fresh neighbours and iterate through them
    for n in nbrs:

        # read weight of connection
        w = G[n][ill_node]
        # print(n, w, G.nodes[n])

        # compute probability of being affected
        prob = w['w'] * G.nodes[n]['status']

        # toss a coin with binomial prob and check if node will become ill
        toss = np.random.choice(np.arange(0, 2), p=[1 - prob, prob])

        # if toss is 1 then node is ill
        if toss == 1:
            # append it to the taboo list
            visited.append(n)
            sperad(G, n, visited)
            # print('tango down', n)

    return visited




g = read_net('nodes.csv', 'edges.csv')

print(g.nodes(data=True))
v = sperad(g, 2)
print(v)
print(g.nodes(data=True))




