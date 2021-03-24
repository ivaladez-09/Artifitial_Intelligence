""""""
import numpy as np
from math import exp
import matplotlib.pyplot as plt
from matplotlib import cm


class ArrayLengthError(Exception):
    """"""
    pass


class FuzzyNet:
    """"""

    def __init__(self, x: list = None, y: list = None):
        """"""
        if x and y:
            self.X = x[:]
            self.Y = y[:]
        else:
            self.X = [i for i in range(1, 5)]
            self.Y = [j for j in range(1, 9)]

        self.Z = self.get_base_z()

        x_length = len(self.X)
        y_length = len(self.Y)

        if len(self.Z) != x_length or len(self.Z[0]) != y_length:
            error_message = 'Z must have a shape of (x,y) {}'.format((x_length, y_length))
            raise ArrayLengthError(error_message)

        self.total_parameters = 39

    @staticmethod
    def get_base_z() -> tuple:
        """"""
        z = ((31.00, 35.52, 40.91, 46.26, 55.48, 63.00, 68.78, 75.8),
             (22.57, 25.42, 29.30, 32.74, 36.74, 39.57, 42.96, 54.39),
             (14.87, 17.74, 20.13, 24.36, 26.65, 28.57, 31.65, 37.83),
             (05.87, 07.65, 09.39, 11.78, 14.35, 16.52, 19.26, 23.30))
        return z

    @staticmethod
    def mf_gaussian(x: float, m: float, d: float) -> float:
        """
        Returns a value from x in a gaussian function

        :param x: Number which you want your Gaussian
        :param m: Float with the median of the function
        :param d: Float with the standard deviation
        :return: Float with the result of gaussian(x)
        """
        d = d if d != 0 else 0.2
        return exp(-1 / 2 * pow((x - m) / d, 2))

    def plot(self, matrix_z: list, ciclos: int, bestAptitude: float, aptitudes: list):
        """"""
        # Matrix z must be numpy array
        base_z = np.array(self.Z)
        z = np.array(matrix_z)

        # Data must follow this order (y,x) to generate the matrix 4x8
        Y, X = np.meshgrid(self.Y, self.X)

        fig = plt.figure(1, figsize=(10, 8))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_title("Covid19 Results")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        surf = ax.plot_surface(X, Y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        surf2 = ax.plot_surface(X, Y, base_z, cmap='winter', linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(0, 100)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=20)
        fig.colorbar(surf2, shrink=0.5, aspect=20)

        ax = fig.add_subplot(1, 2, 2)
        ax.plot([i for i in range(ciclos)], aptitudes)
        ax.scatter([i for i in range(ciclos)], aptitudes, color='green')
        ax.grid()
        ax.set_title("Best aptitude: " + str(bestAptitude))
        ax.set_xlabel("Generation")
        ax.set_ylabel("Aptitutde")

        plt.show(block=False)
        plt.pause(2)

        plt.figure(1)
        plt.clf()

    def get_parsed_chromosome(self, chromosome: list) -> dict:
        """"""
        # Guarantee the correct size of the chromosome
        if len(chromosome) != self.total_parameters:
            error_message = 'Size of chromosome {} must be equal to {}'.format(len(chromosome), self.total_parameters)
            raise ArrayLengthError(error_message)

        parsed_chromosome = dict()
        parsed_chromosome['mx'] = chromosome[: 3]
        parsed_chromosome['my'] = chromosome[3: 6]
        parsed_chromosome['dx'] = chromosome[6: 9]
        parsed_chromosome['dy'] = chromosome[9: 12]
        parsed_chromosome['p'] = chromosome[12: 21]
        parsed_chromosome['q'] = chromosome[21: 30]
        parsed_chromosome['r'] = chromosome[30: self.total_parameters]

        return parsed_chromosome

    def get_results(self, chromosome: list) -> (list, float):
        """"""
        parameters = self.get_parsed_chromosome(chromosome)

        # Getting matrix z and aptitude function at the same time to save execution time
        matrix_z = list()
        aptitude_function = 0
        for i, x in enumerate(self.X):
            row_z = list()
            for j, y in enumerate(self.Y):
                # mf stands for 'membership functions'
                mf_x = [self.mf_gaussian(x, m, d) for m, d in zip(parameters['mx'], parameters['dx'])]
                mf_y = [self.mf_gaussian(y, m, d) for m, d in zip(parameters['my'], parameters['dy'])]

                mf_summation = sum(mf_x) + sum(mf_y)
                mf_xy = [mf_xi * mf_yj for mf_xi in mf_x for mf_yj in mf_y]

                inferences = list()
                for mf_xyn, p, q, r in zip(mf_xy, parameters['p'], parameters['q'], parameters['r']):
                    inferences.append(mf_xyn * ((p * x) + (q * y) + r))

                inferences_summation = sum(inferences)

                z = min(inferences_summation // mf_summation, 100)  # Limit the max value to be 100
                row_z.append(z)

                aptitude_function += abs(self.Z[i][j] - z)

            matrix_z.append(row_z)

        aptitude_function /= (len(self.X) * len(self.Y))

        return matrix_z, aptitude_function
