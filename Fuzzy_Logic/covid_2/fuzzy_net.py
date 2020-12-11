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
            self.X = x
            self.X = y
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
        z = ((30, 40, 50, 60, 80, 100, 100, 100),
             (10, 10, 20, 30, 40, 50, 60, 60),
             (0, 0, 10, 10, 20, 20, 25, 30),
             (0, 0, 0, 5, 10, 10, 15, 20))
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
                mf_xy = [i * j for i in mf_x for j in mf_y]

                inferences = list()
                for ij, p, q, r in zip(mf_xy, parameters['p'], parameters['q'], parameters['r']):
                    inferences.append(ij * ((p * x) + (q * y) + r))

                inferences_summation = sum(inferences)

                z = min(inferences_summation // mf_summation, 100)  # Limit the max value to be 100
                row_z.append(z)

                aptitude_function += abs(self.Z[i][j] - z)

            matrix_z.append(row_z)

        aptitude_function /= (len(self.X) * len(self.Y))

        return matrix_z, aptitude_function
