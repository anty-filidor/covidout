import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import pandas as pd
import io


def _filter(G: nx.Graph, node, neighbourhood: int) -> nx.Graph:
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


def _categorise(G: nx.Graph, what: str) -> (pd.DataFrame, list):
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
        lcats = ['irrelevant symp.', 'low symp.', 'medium symp.', 'serious symp.']
        lbins = [-.0001, .25, .5, .75, 1.0001]

    elif what is 'edges':
        f = G.edges
        # categories of nodes
        lcats = ['tight cont.', 'close cont.']
        lbins = [-.0001, .5, 1.0001]
    else:
        return

    # data frame with characteristics for nodes
    _ = [{'ID': n, 'status': f[n]['w']} for n in f()]

    node_labels = pd.DataFrame(_)
    node_labels = node_labels.set_index('ID')
    node_labels['status'] = pd.cut(node_labels['status'], lbins, labels=lcats)

    # print(node_labels['status'].dtype)
    # print(node_labels)
    # print(node_labels.describe())

    return node_labels, lcats


def plot(G: nx.Graph, node, neighbourhood: int, n_labels: bool = False,
         e_labels: bool = False) -> bytes:
    """
    Method to visualise subgraph. Useful link
    https://stackoverflow.com/questions/22992009/legend-in-python-networkx
    :param G: given graph
    :param node: node for which neighbourhood visualise a graph
    :param neighbourhood: degree of neighbourhood
    :param n_labels: a flag if pring names of nodes
    :param e_labels: a flag if print states of edges
    :return: png image serialised into bytes
    """

    # extract subgraph
    g = _filter(G, node, neighbourhood)

    # if subgraph is 1 element long do sth stupid xD
    if len(g.nodes()) <= 1:
        plt.scatter(0, 0)
        plt.text(0, 0, 'SINGLETON XDDDD')
        plt.show()
        return

    # categorise nodes and edges
    nlbls, ncats = _categorise(g, 'nodes')
    elbls, ecats = _categorise(g, 'edges')

    # init position
    pos = nx.spring_layout(g)

    '''
    # uncomment it to non perfect, but solid visualising
    # plot a image of this epoch
    fig, ax = plt.subplots(figsize=(7, 7))
    nx.draw(g, pos=pos, with_labels=True,
            node_color=nlbls['status'].cat.codes,
            node_cmap=plt.cm.Set1, node_size=150,
            edge_color=elbls['status'].cat.codes,
            edge_cmap=plt.cm.Set1, edge_size=10,
            alpha=0.7, font_size=10)

    plt.tight_layout()
    plt.legend(g.nodes())
    plt.show()
    '''

    # map all possible labels to numerical values
    class_map = {cat: n+1 for n, cat in enumerate(ncats)}
    # print(class_map)

    # map labels in subgraph nodes
    node_map = [class_map.get(node[1]['status'], 0) for
                node in nlbls.iterrows()]
    # print(node_map)

    # color mapping
    jet = plt.get_cmap('jet')
    c_norm = colors.Normalize(vmin=0, vmax=max(class_map.values()))
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=jet)

    # using a figure with legend as a parameter when calling nx.draw_networkx
    f = plt.figure(1)
    ax = f.add_subplot(1, 1, 1)
    for lbl in class_map:
        ax.plot([0], [0], color=scalar_map.to_rgba(class_map[lbl]), label=lbl)

    # plot sub graph nodes
    nx.draw_networkx(g, pos=pos, with_labels=n_labels, cmap=jet,
                     vmin=0, vmax=max(class_map.values()),
                     node_color=node_map, node_size=170, edge_size=10,
                     alpha=0.9, font_size=12, ax=ax)

    # plot sub graph edges
    if e_labels:
        nx.draw_networkx_edge_labels(g, edge_labels=elbls.to_dict()['status'],
                                     pos=pos, ax=ax)

    # make figure more pretty
    plt.axis('off')
    f.set_facecolor('w')
    plt.legend()  # loc='lower right')
    f.tight_layout()
    plt.show()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    return buf.getvalue()
