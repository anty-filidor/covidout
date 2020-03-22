from data_generation.simulation import gen_city, Area


class CitySim:
    """
    Class representing a simulation of people movement in the city
    """

    def __init__(self, city: Area, num_people):
        self.city = city
        self.num_people = num_people
        self.state = gen_city(city, num_people)

    def simulation(self):
        while True:
            self.state = brownian_2d(self.state, 1, 0.01, 0.001)[-1, :, :]
            yield self.state


class AnimatedScatter(object):
    """
    Class visualizing movement of points on scatterplot
    source: https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot
    """

    def __init__(self, city_sim: CitySim):
        """An animated scatter plot using matplotlib.animations.FuncAnimation."""
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