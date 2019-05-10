import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import subprocess as sp


class Kohonen:
    def __init__(self, data, iterations=5000, learning_rate=0.01, normalize_data=True, normalize_by_column=False, network_x_dimension=40, network_y_dimension=40):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.normalize_data = normalize_data
        self.normalize_by_column = normalize_by_column
        self.network_dimensions = np.array([network_x_dimension, network_y_dimension])
        self.init_radius = max(self.network_dimensions[0], self.network_dimensions[1]) / 2
        self.time_constant = iterations / np.log(self.init_radius)
        self.raw_data = data
        self.data = None
        self.network = np.random.random((self.network_dimensions[0], self.network_dimensions[1], self.raw_data.shape[0]))

    def _find_bmu(self, t):
        """
        to find best matching unit
        """
        m = self.raw_data.shape[0]
        bmu_idx = np.array([0, 0])
        min_dist = np.iinfo(np.int).max

        for x in range(self.network.shape[0]):
            for y in range(self.network.shape[1]):
                w = self.network[x, y, :].reshape(m, 1)
                sq_dist = np.sum((w - t) ** 2)
                sq_dist = np.sqrt(sq_dist)
                if sq_dist < min_dist:
                    min_dist = sq_dist  # dist
                    bmu_idx = np.array([x, y])  # id

        bmu = self.network[bmu_idx[0], bmu_idx[1], :].reshape(m, 1)
        return bmu, bmu_idx

    @staticmethod
    def decay_radius(init_radius, i, time_constant):
        return init_radius * np.exp(-i / time_constant)

    @staticmethod
    def decay_learning_rate(initial_learning_rate, i, n_iterations):
        return initial_learning_rate * np.exp(-i / n_iterations)

    @staticmethod
    def calculate_influence(distance, radius):
        return np.exp(-distance / (2 * (radius ** 2)))

    def show_percentage(self, i):
        sp.call('clear', shell=True)
        print("Learning in Progress: " + str(i/self.iterations*100) + "%")

    def normalize(self):
        data = self.raw_data
        if self.normalize_data:
            if self.normalize_by_column:
                col_maxes = self.raw_data.max(axis=0)
                data = self.raw_data / col_maxes[np.newaxis, :]
            else:
                data = self.raw_data / data.max()
        self.data = data

    def train(self):
        for i in range(self.iterations + 1):
            self.show_percentage(i)
            # print("Iteration %d Completed" % i)
            # t is reshaped data
            t = self.data[:, np.random.randint(0, self.raw_data.shape[1])].reshape(np.array([self.raw_data.shape[0], 1]))
            bmu, bmu_idx = self._find_bmu(t)
            # r is radius
            r = self.decay_radius(self.init_radius, i, self.time_constant)
            # l is new learning rate
            new_learning_rate = self.decay_learning_rate(self.learning_rate, i, self.iterations)

            for x in range(self.network.shape[0]):
                for y in range(self.network.shape[1]):
                    w = self.network[x, y, :].reshape(self.raw_data.shape[0], 1)
                    w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
                    w_dist = np.sqrt(w_dist)

                    if w_dist <= r:
                        influence = self.calculate_influence(w_dist, r)
                        new_w = w + (new_learning_rate * influence * (t - w))
                        self.network[x, y, :] = new_w.reshape(1, 3)

    def show(self, save=True):
        fig = plt.figure()

        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim((0, self.network.shape[0] + 1))
        ax.set_ylim((0, self.network.shape[1] + 1))
        ax.set_title('Kohonen after %d iterations' % self.iterations)

        # plot
        for x in range(1, self.network.shape[0] + 1):
            for y in range(1, self.network.shape[1] + 1):
                ax.add_patch(patches.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                               facecolor=self.network[x - 1, y - 1, :],
                                               edgecolor='none'))
        if save:
            plt.savefig('Kohonen%d.png' % self.iterations)
        plt.show()




