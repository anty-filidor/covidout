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


def state_transition(states, transition_probs, n):
    gen_states = [np.array(states)]
    for i in range(n):
        transitions = np.random.binomial(1, [transition_probs]*len(states), states.shape)
        new_states = np.abs(gen_states[-1] - transitions)
        gen_states.append(new_states)

    return np.array(gen_states)


class Area(abc.ABC):
    def __init__(self, longitude_min, longitude_max, latitude_min, latitude_max):
        self.longitude_min = longitude_min
        self.longitude_max = longitude_max
        self.latitude_min = latitude_min
        self.latitude_max = latitude_max


wroclaw = Area(16.918396, 17.134002, 51.078067, 51.165906)


def gen_city(area: Area, n_people):
    lat_mean = np.mean([area.latitude_min, area.latitude_max])
    lat_std = np.std([area.latitude_min, area.latitude_max])

    long_mean = np.mean([area.longitude_min, area.longitude_max])
    long_std = np.std([area.longitude_min, area.longitude_max])

    people_lat = np.random.normal(lat_mean, lat_std/3, n_people)
    people_long = np.random.normal(long_mean, long_std/3, n_people)

    return np.array([people_long, people_lat], dtype=np.float).transpose()


class CitySim:
    def __init__(self, city: Area, num_people):
        self.city = city
        self.num_people = num_people
        self.state = gen_city(city, num_people)

    def simulation(self):
        while True:
            self.state = brownian_2d(self.state, 1, 0.01, 0.001)[-1, :, :]
            yield self.state


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, city_sim: CitySim):
        self.city_sim = city_sim
        self.stream = self.data_stream()

        self.fig, self.ax = plt.subplots()
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5,
                                           init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        data = next(self.stream).transpose()
        self.scat = self.ax.scatter(data[0], data[1],
                                    cmap="jet", edgecolor="k")
        self.ax.axis([
            self.city_sim.city.longitude_min,
            self.city_sim.city.longitude_max,
            self.city_sim.city.latitude_min,
            self.city_sim.city.latitude_max
        ])

        return self.scat,

    def data_stream(self):
        while True:
            yield next(self.city_sim.simulation())

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)

        self.scat.set_offsets(data)

        return self.scat,


def calc_distance(pos_a, pos_b):
    return geopy.distance.geodesic(pos_a, pos_b).km


def weight_func(dist):
    return max(1 - dist**2, 0)


def gen_weighted_connections(nodes_pos: pd.DataFrame):
    n = max(nodes_pos["user_id"]) + 1
    connections = dok_matrix((n, n), dtype=np.float32)

    for rows in combinations(nodes_pos.values, 2):
        w = weight_func(calc_distance(rows[0][1:], rows[1][1:]))

        if w > 0:
            connections[rows[0][0], rows[1][0]] = w

    return csc_matrix(connections)


def mutate_connections(old_connections, new_connections, add=1, retain=1):
    """
    Mutate old_connection to accomodate for new_connections.
    :param old_connections: WARNING - no longer valid after function call!
    :param new_connections:
    :param add:
    :param retain:
    :return:
    """
    old_connections.data = np.clip(new_connections.data * add + old_connections.data*retain, 0.0, 1.0)

    return old_connections


def save_edges(filename, connections: sp.sparse.csc_matrix):
    data = []

    for key, value in connections.todok().items():
        data.append((key[0], key[1], value))

    pd.DataFrame(data).to_csv(filename, index=False, header=False)


#sim = CitySim(wroclaw, 40)

#conn = generate_weighted_conn(next(sim.simulation()))
#conn2 = mutate_connections(conn, conn)
#save_edges("edges.csv", conn)

#data = gen_city(lat_min, lat_max, long_min, long_max, 40)
#result = generate_weighted_conn(data)

#result = pd.DataFrame(result)
#result.to_csv("edges.csv", index=False, header=None)
#data_cechy = np.random.binomial(1, 0.1, (data.shape[0], 18))

#pd.DataFrame(data_cechy).to_csv("nodes.csv", header=None)
