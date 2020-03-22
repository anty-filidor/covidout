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


def sperad(G: nx.Graph, ill_node, visited: set = None, sure_ill=False) -> list:
    """
    This method spreads disease from given node to the entire network

    :param G: graph within disease is being spread
    :param start_node: starting poin of simulation
    :param visited: taboo list of nodes affected. used in the recursion
    :return: nodes which were affected
    """

    # set status of ill node as 100% infected

    if sure_ill:
        G.nodes[ill_node]['w'] = 1

    # update visited
    if visited is None:
        visited = set()
    # visited.add(ill_node)

    G.nodes[ill_node]['prob'] = 1

    # check if neighbour nodes are in taboo list
    nbrs = [*G[start_node].keys()]
    nbrs = [n for n in nbrs if n not in visited]
    # print(nbrs)

    # take it's fresh neighbours` and iterate through them
    for n in nbrs:

        # if node is already infected go over it
        if G.nodes[n]['prob'] == 1:
            continue

        # if node has prob = 0.0 dump it to 1e-6
        if G.nodes[n]['prob'] < 1e-6:
            G.nodes[n]['prob'] = 1e-6

        # take neighbours of node
        _n = [*G[n].keys()]

        # take probabilities from neighbouring nodes and mul it by connection
        # weights

        cum_not_infected_prob = 1

        for neigbour in _n:
            neigbour_prob = G.nodes[neigbour]['prob']
            relation_prob = G.edges[(n, neigbour)]['w']
            values_1 = pd.DataFrame.from_dict(G.nodes[n]['state'], orient="index").values
            values_2 = pd.DataFrame.from_dict(G.nodes[n]['state'], orient="index").values
            similarity_prob = np.sum(np.abs(values_1 - values_2))
            similarity_prob = np.clip(similarity_prob, 0, 1)

            cum_not_infected_prob *= (1 - neigbour_prob * relation_prob * similarity_prob)

        G.nodes[n]['prob'] = 1.0 - cum_not_infected_prob

        #probs = [G[__n][n]['prob'] * G.nodes[__n]['prob'] for __n in _n]
        #probs = np.array(probs)

        #prob = probs * G.nodes[n]['prob']
        #prob = 1 - prob
        #prob = np.prod(prob) * 1  # TODO - MAGIC NUMBER to speed up propagation
        # prob is probability of being health of node "n"

        # update new probability computed from external conditions
        #G.nodes[n]['w'] = 1 - prob

        visited.add(n)
        sperad(G, n, visited)

    return visited

