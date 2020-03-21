import networkx as nx
import numpy as np
import pandas as pd
import scipy

def comp_status(nodes: str) -> pd.DataFrame:
    """
    reads nodes states and compute resultant function
    RESULTANT FUNCTION SHOULD RETURN RANGE [0, 100]
    :param nodes: name of csv with nodes
    :return: data frame with nodes names and their states
    """

    # read csv to pd Data Frame
    n = pd.read_csv(nodes, header=None)
    n = n.rename(columns={0: 'n'})
    n = n.set_index('n')

    #n = n.sum(axis=1)
    #n = n / 10

    # convert temperature to catrgorical value
    n[2] = n[2].apply(lambda x: 2 if x > 38 else (1 if x > 37.5 else 0))

    '''
    - kaszel - 10
    - temperatura - float value
    - dusznosc - 10
    - zmeczenie - 7
    - bol_glowy - 7
    - bol_miesni - 7
    - bol_gardla - 7
    - zaburzenie_wechu - 7
    - zaburzenie_smaku - 7
    - katar - 2
    - kichanie - 2
    - nudnosci - 2
    - biegunka - 2
    - bol_brzucha - 2
    - zawroty_glowy - 2
    - niepokoj - 2
    - kolatanie_serca - 2
    - zime_dreszcze - 2
    '''
    vec = [10, 2, 10, 7, 7, 7, 7, 7, 7, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    # compute softmax
    vec = scipy.special.softmax(vec)

    # set probabilities for each parameter
    n = n * vec

    # compute overall probability
    n = n.sum(axis=1)

    return n


def update_nodes_l(G: nx.Graph, n:pd.DataFrame):
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
    n = comp_status(nodes)
    update_nodes_l(G, n)

    return G
