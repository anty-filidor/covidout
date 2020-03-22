import networkx as nx
import numpy as np
import pandas as pd


def update_node(G: nx.Graph, node, weight: float):
    """
    This method updates weight of certain node in given graph

    :param G: graph to update
    :param node: node name to update
    :param weight: value of new weight
    """

    assert 0.001 <= weight <= 1, 'Probability must be in 0.001:1'
    G.nodes[node]['w'] = weight


def sperad(G: nx.Graph, node_id, visited: set = None, sure_ill=False):
    """
    This method spreads disease from given node to the entire network

    :param G: graph within disease is being spread
    :param node_id: ill node
    :param visited: taboo list of nodes affected. used in the recursion
    :return: nodes which were affected
    """
    # update visited
    if visited is None:
        visited = set()

    # set status of ill node as 100% infected
    if node_id in visited:
        return visited
    elif sure_ill:
        G.nodes[node_id]['prob'] = 1.0
    else:
        # take neighbours of node
        _n = [*G[node_id].keys()]

        probs = [G.nodes[node_id]['state'] * G[__n][node_id]['w']**2 * G.nodes[__n]['prob'] for __n in _n]
        probs = np.array(probs)

        prob = probs
        prob = 1 - prob
        prob = np.prod(prob)
        # prob is probability of being health of node "n"

        # update new probability computed from external conditions
        G.nodes[node_id]['prob'] =  (1 - prob)

    visited.add(node_id)
    nbrs = [*G[node_id].keys()]
    for n in nbrs:
        sperad(G, n, visited)


    return visited

