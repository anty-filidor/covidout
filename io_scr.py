import networkx as nx
import numpy as np
import pandas as pd


def comp_status(nodes: str) -> pd.DataFrame:
    """
    reads nodes states and compute resultant function
    RESULTANT FUNCTION SHOULD RETURN RANGE [0, 100]
    :param nodes: name of csv with nodes
    :return: data frame with nodes names and their states
    """
    n = pd.read_csv(nodes, header=None)
    n = n.rename(columns={0: 'n'})
    n = n.set_index('n')
    n = n.sum(axis=1)
    n = n / 10

    return n


def update_nodes_l(G: nx.Graph, n:pd.DataFrame):
    """
    updates nodes with given status numbers
    :param G: graph
    :param n: dataframe with nodes and nwe states
    """
    G.update(nodes=n.index.to_list())
    nx.set_node_attributes(G, n, 'status')
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
