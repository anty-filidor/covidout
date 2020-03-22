from math import sqrt
from scipy.stats import norm
import numpy as np
import geopy.distance
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import dok_matrix, csc_matrix
import scipy as sp
import abc
from itertools import combinations


class Area(abc.ABC):
    """
    Class representing rectangular-like area on earth coordinates
    """
    def __init__(self, longitude_min, longitude_max, latitude_min, latitude_max):
        self.longitude_min = longitude_min
        self.longitude_max = longitude_max
        self.latitude_min = latitude_min
        self.latitude_max = latitude_max


# Data will be generated roughly for Wroclaw area
WROCLAW = Area(16.918396, 17.134002, 51.078067, 51.165906)


def brownian(x0, n, dt, delta, out=None):
    """
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.
    """

    x0 = np.asarray(x0)

    r = norm.rvs(size=x0.shape + (n,), scale=delta * sqrt(dt))

    if out is None:
        out = np.empty(r.shape)

    np.cumsum(r, axis=-1, out=out)

    out += np.expand_dims(x0, axis=-1)

    return out


def brownian_2d(coords, n: int, dt: float, delta: float):
    """
    Brownian motion for 2d objects
    :param dt: Time delta
    :param coords: list of (x, y) tuples
    :param n: number of steps to simulate
    :param delta: Wiener delta
    :return: Array of shape (n, num_points, 2)
    """

    coords = np.array(coords, dtype=np.float)
    assert (len(coords.shape) == 2)
    coords = coords.transpose()
    assert (coords.shape[0] == 2)

    x_brownian = brownian(coords[0], n, dt, delta)
    y_brownian = brownian(coords[1], n, dt, delta)

    return np.array([x_brownian, y_brownian]).transpose([2, 1, 0])


def state_transition(states: np.array, transition_probs: np.array, n: int) -> np.array:
    """
    A Markovian process step symetrical binomial process
    :param states: array/matrix of binomial state variabless
    :param transition_probs: probability of each element in a states row to transition to opposite state
    :param n: number of transitions to generate
    :return: array of shape (n, *states.shape) representing states in each step of the process
    """

    gen_states = [np.array(states)]
    for i in range(n):
        transitions = np.random.binomial(1, [transition_probs]*len(states), states.shape)
        new_states = np.abs(gen_states[-1] - transitions)
        gen_states.append(new_states)

    return np.array(gen_states)


def gen_city(area: Area, n_people):
    """
    Generate a city-like initial distibution of people on an area.
    :param area: Are representing place where 95% of people will be generated
    :param n_people: Number of people to generate on the area

    :return: (n_people, 2) array of coordinates for each person
    """

    lat_mean = np.mean([area.latitude_min, area.latitude_max])
    lat_std = np.std([area.latitude_min, area.latitude_max])

    long_mean = np.mean([area.longitude_min, area.longitude_max])
    long_std = np.std([area.longitude_min, area.longitude_max])

    people_lat = np.random.normal(lat_mean, lat_std/3, n_people)
    people_long = np.random.normal(long_mean, long_std/3, n_people)

    return np.array([people_long, people_lat], dtype=np.float).transpose()


def calc_distance(pos_a, pos_b):
    """
    Calculates a geodesic distance between 2 points
    :param pos_a: Geodesic coordinates of first point
    :param pos_b: Geodesic coordinates of second point

    :return: Distance in kilometers
    """
    return geopy.distance.geodesic(pos_a, pos_b).km


def weight_func(dist):
    """
    Function that translates distance into node weights
    TODO: Fit to research

    :param dist: distance (in kilometers) between two nodes
    :return: weight connecting 2 nodes (0.0 - 1.0)
    """
    return max(1 - dist**2, 0)


def gen_weighted_connections(nodes_pos: pd.DataFrame) -> csc_matrix:
    """
    Generate connections between nodes from their gedesical positions
    :param nodes_pos: array of shape (n_nodes, 2) representing nodes geodesical position
    :return: sparse half-matrix representing weights between each 2 nodes
    """

    n = max(nodes_pos["user_id"]) + 1
    connections = dok_matrix((n, n), dtype=np.float32)

    for rows in combinations(nodes_pos.values, 2):
        w = weight_func(calc_distance(rows[0][1:], rows[1][1:]))

        if w > 0:
            connections[rows[0][0], rows[1][0]] = w

    return csc_matrix(connections)


def mutate_connections(old_connections: sp.sparse.csc_matrix, new_connections: sp.sparse.csc_matrix, add=0.2, retain=0.8):
    """
    Mutate information of connections between each nodes acoomodationg new location info
    :param old_connections: sparse half-matrix representing weights between each 2 nodes before mutation
                            WARNING - no longer valid after function call!
    :param new_connections: sparse half-matrix representing weights between each 2 nodes on given time
    :param add: how much of new_connecion should be accomodated into weights
    :param retain: how much of old_connections should be left in wegights
    :return: sparse half-matrix representing weights between each 2 nodes after mutation. Clipped between 0.0 and 1.0
    """
    old_connections.data = np.clip(new_connections.data * add + old_connections.data*retain, 0.0, 1.0)

    return old_connections


def save_edges(filename: str, connections: sp.sparse.csc_matrix) -> None:
    """
    Saves connections between nodes in node-analitical friendly csv format
    :param filename: Csv path where connections will be stored
    :param connections: sparse half-matrix representing weights between each 2 nodes
    :return: None
    """

    data = []

    for key, value in connections.todok().items():
        data.append((key[0], key[1], value))

    pd.DataFrame(data).to_csv(filename, index=False, header=False)