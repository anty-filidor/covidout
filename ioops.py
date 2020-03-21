import networkx as nx
import numpy as np
import pandas as pd
import scipy
import json


def _comp_internal_weights(nodes: str) -> pd.DataFrame:
    """
    reads nodes states and compute resultant function as internal probability
    of being infected
    :param nodes: name of csv with nodes
    :return: data frame with nodes names and their states
    """

    # read csv to pd Data Frame
    n = pd.read_csv(nodes, header=0)
    # print(n)
    n = n.rename(columns={'node_id': 'n'})
    n = n.set_index('n')

    # convert temperature to catrgorical value
    n['temperatura'] = n['temperatura'].apply(lambda x: 1 if x > 39 else x/39)

    '''
    temperatura         2
    dusznosc            10
    zmeczenie           7
    bol_glowy           7
    bol_miesni          7
    bol_gardla          7
    zaburzenie_wechu    7
    zaburzenie_smaku    7
    katar               2
    kichanie            2
    nudnosci            2
    biegunka            2
    bol_brzucha         2
    zawroty_glowy       2
    niepokoj            2
    kolatanie_serca     2
    zime_dreszcze       2
    zaparcia            2
    zgaga               2
    powiekszenie_wezlow_chlonnych    2
    goraczka            10
    wysypka             2
    splatanie           2
    krwioplucie         2 
    '''

    vec = [2, 10, 7, 7, 7, 7, 7, 7,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           10, 2, 2, 2]

    # compute softmax function
    vec = scipy.special.softmax(vec)

    # set probabilities for each parameter
    n = n * vec

    # compute overall probability
    n = n.sum(axis=1)

    return n


def _update_nodes_labels(G: nx.Graph, n:pd.DataFrame):
    """
    updates nodes with given status numbers
    :param G: graph
    :param n: dataframe with nodes and nwe states
    """
    G.update(nodes=n.index.to_list())
    nx.set_node_attributes(G, n, 'w')
    # print(G.nodes(data=True))


def read_net(nodes: str, edges:str) -> nx.Graph:
    """
    Reads network from csv and compute resultant state in each node
    :param nodes: name of csv with nodes
    :param edges: name of edges with nodes
    :return: graph
    """

    # read edges list
    e = pd.read_csv(edges, header=None, names=['n1', 'n2', 'w'])
    G = nx.from_pandas_edgelist(e, 'n1', 'n2', 'w')

    # read nodes list
    n = _comp_internal_weights(nodes)
    _update_nodes_labels(G, n)

    return G


def get_node_states(G: nx.Graph) -> json:
    """
    Method which returns network state as json
    :param G: graph to return
    :return: graph as json
    """

    probs = {}
    _ = [*G.nodes()]
    _.sort()
    for n in _:
        probs[n] = G.nodes[n]['w']

    return json.dumps(probs)
