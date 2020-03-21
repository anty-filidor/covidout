import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io


def filter(G: nx.Graph, node, neighbourhood: int) -> nx.Graph:
    """
    Returns subgraph with node and is't sub neighbourhood
    :param G: garph
    :param node: name of node to create subgraph
    :param neighbourhood: degree of nbrhd
    :return: sub-graph
    """

    # find interesting neighbours of given node
    sp = nx.shortest_path(G, node)
    neighbours = [key for (key, value) in sp.items() if
                  len(value) <= neighbourhood + 1]

    # create subgraph
    g = nx.subgraph(G, neighbours)

    return g


def categorise(G: nx.Graph, what: str) -> (pd.DataFrame, list):
    """
    Method to categorise labels of nodes and edges
    :param what: nodes of edges
    :param G: graph
    :return:
    """

    # initialise params
    if what is 'nodes':
        f = G.nodes
        # categories of nodes
        lcats = ['a', 'b', 'c', 'd']
        lbins = [-.0001, .25, .5, .75, 1.0001]

    elif what is 'edges':
        f = G.edges
        # categories of nodes
        lcats = ['a', 'b']
        lbins = [-.0001, .5, 1.0001]
    else:
        return

    # data frame with characteristics for nodes
    _ = [{'ID': n, 'status': f[n]['w']} for n in f()]

    node_labels = pd.DataFrame(_)
    node_labels = node_labels.set_index('ID')
    node_labels['status'] = pd.cut(node_labels['status'], lbins, labels=lcats)
    # node_labels = pd.DataFrame(node_labels)
    #node_labels = node_labels.set_index('ID')


    # print(node_labels['status'].dtype)
    print(node_labels)
    # print(node_labels.describe())

    '''
    node_labels = pd.DataFrame({'ID': [n for n in f()],
                                'status': [
                                    lcats[np.random.randint(0, len(lcats))]
                                    for m in f()]})
    node_labels = node_labels.set_index('ID')

    # Transform categorical column in a numerical value
    node_labels['status'] = pd.Categorical(node_labels['status'],
                                           categories=lcats)
    print(node_labels['status'].dtype)
    print(node_labels)
    print(node_labels.describe())
    '''
    return node_labels, lcats


def plot(G: nx.Graph, node, neighbourhood: int) -> io.BytesIO:
    """
    Method to visualise subgraph
    :param G: given graph
    :param node: node for which neighbourhood visualise a graph
    :param neighbourhood: degree of neighbourhood
    :return: 
    """

    g = filter(G, node, neighbourhood)
    nlbls, ncats = categorise(g, 'nodes')
    elbls, ecats = categorise(g, 'edges')

    pos = nx.spring_layout(g)

    # Plot a image of this epoch:
    fig, ax = plt.subplots(figsize=(7, 7))
    nx.draw(g, pos=pos, with_labels=True,
            node_color=nlbls['status'].cat.codes,
            node_cmap=plt.cm.Set1, node_size=150,
            edge_color=elbls['status'].cat.codes,
            edge_cmap=plt.cm.Set1, edge_size=10,
            alpha=0.7, font_size=10)

    plt.tight_layout()
    plt.show()




