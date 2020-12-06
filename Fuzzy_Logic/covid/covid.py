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

    def __init__(self, z=None):
        """"""
        self.X = np.array([1, 2, 3, 4], dtype=np.uint8)
        self.Y = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8)
        self.WEIGHT = 5  # mx1 = mx1'(0-255) / weight
        if z is None:
            self.Z = self._read_z()  # Output Supervised data
        else:
            self.Z = z  # Output from survey
        # mf stands for Membership function
        self.mfx_number = 3
        self.mfy_number = 3
        self.permutations = self.mfx_number ** self.mfy_number
        self.total_parameters = self.permutations + ((self.mfx_number + self.mfy_number) * 2)

    @staticmethod
    def _read_z() -> np.array:
        """
        Output data 'z' to compare the output from the fuzzy net.
        Note: elements from list are in range of 0-255 (np.uint8)
        :return: Numpy Array (matrix) of MxN results.
        """
        z = np.array([  # 'Z' Debe de estar cambiando dependiendo de la encuesta a cada persona
            [30, 40, 50, 60, 80, 100, 100, 100],
            [10, 10, 20, 30, 40, 50, 60, 60],
            [0, 0, 10, 10, 20, 20, 25, 30],
            [0, 0, 0, 5, 10, 10, 15, 20],
        ], dtype=np.uint8)
        return z

    @staticmethod
    def mf_gaussian(x: float, c: float, r: float) -> float:
        """"""
        r = r if r != 0 else 0.001
        return exp(-1 / 2 * pow((x - c) / r, 2))

    def get_parsed_chromosome(self, chromosome: np.array) -> dict:
        """"""
        # Guarantee the correct size of the chromosome
        if len(chromosome) != self.total_parameters:
            error_message = f'Chromosome length shall not be different from {self.total_parameters}'
            raise ArrayLengthError(error_message)

        chromosome = chromosome / self.WEIGHT  # parameters = chromosome'(0-255) / weight
        parsed_chromosome = dict()

        m_index = self.mfx_number + self.mfy_number
        parsed_chromosome['m'] = chromosome[:m_index]  # [:x] -> x is not included in the returned list

        d_index = (m_index * 2)
        parsed_chromosome['d'] = chromosome[m_index:d_index]

        p_index = int(d_index + (self.permutations / 3))
        parsed_chromosome['p'] = chromosome[d_index:p_index]

        q_index = int(p_index + (self.permutations / 3))
        parsed_chromosome['q'] = chromosome[p_index:q_index]

        r_index = int(q_index + (self.permutations / 3))
        parsed_chromosome['r'] = chromosome[q_index:r_index]

        return parsed_chromosome

    def get_matrix_z(self, parameters: dict) -> np.array:
        """"""
        matrix_z = np.empty((0, self.Y.size), dtype=np.uint16)  # size = 4x8
        for x in self.X:
            row_x = list()
            for y in self.Y:
                membership_functions_x = list()
                membership_functions_y = list()
                for m, d in zip(parameters['m'][:self.mfx_number], parameters['d'][:self.mfx_number]):
                    membership_functions_x.append(self.mf_gaussian(x, m, d))

                for m, d in zip(parameters['m'][self.mfx_number:self.mfx_number + self.mfy_number],
                                parameters['d'][self.mfx_number:self.mfx_number + self.mfy_number]):
                    membership_functions_y.append(self.mf_gaussian(y, m, d))

                mf_summation = sum(membership_functions_x) + sum(membership_functions_y)

                mfx_mfy = list()
                for mfx in membership_functions_x:
                    for mfy in membership_functions_y:
                        mfx_mfy.append(mfx * mfy)

                inferences = list()
                for mfxy, p, q, r in zip(mfx_mfy, parameters['p'], parameters['q'], parameters['r']):
                    inferences.append(mfxy * ((p * x) + (q * y) + r))

                inferences_summation = sum(inferences) / 3  # To avoid numbers way higher than 100

                # Use uint8 values in range of 0-100
                row_x.append(np.uint8(min(inferences_summation / mf_summation, 100)))

            matrix_z = np.append(matrix_z, [row_x], axis=0)

        # Guarantee the correct form of the output matrix Z
        if matrix_z.shape != self.Z.shape:
            error_message = f'Matrix_Z shape {matrix_z.shape} shall be equal to {self.Z.shape}'
            raise ArrayLengthError(error_message)

        return matrix_z

    def plot(self, matrix_z: np.array):
        """"""
        Y, X = np.meshgrid(self.Y, self.X)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        surf = ax.plot_surface(X, Y, matrix_z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(0, 100)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=20)

        plt.show()

    def get_average_matrix_z(self, population_matrix_z: np.array):
        """"""
        average = np.zeros((len(self.X), len(self.Y)), dtype=np.uint16)  # size = 4x8
        for matrix_z in population_matrix_z:
            average = np.add(average, matrix_z)

        return average // len(population_matrix_z)

    def get_FA(self, population: np.array) -> np.uint8:
        """"""
        population_matrix_z = np.empty((0, len(self.X), len(self.Y)), dtype=np.uint16)
        population_aptitude_function = np.empty(0, dtype=np.uint16)

        for chromosome in population:
            parameters = self.get_parsed_chromosome(chromosome)
            # print(parameters.values())

            matrix_z = self.get_matrix_z(parameters=parameters)
            population_matrix_z = np.append(population_matrix_z, [matrix_z], axis=0)
            # print(matrix_z, end='\n\n')

            # print(np.abs(np.subtract(self.Z, matrix_z, dtype=np.int16)), end='\n\n')
            aptitude_function = np.abs(np.subtract(self.Z, matrix_z, dtype=np.int16)).mean()
            population_aptitude_function = np.append(population_aptitude_function, aptitude_function)

        average_matrix_z = self.get_average_matrix_z(population_matrix_z)
        average_aptitude_function = np.abs(np.subtract(self.Z, average_matrix_z, dtype=np.int16)).mean()
        print('Average Z matrix\n', average_matrix_z, average_aptitude_function)

        self.plot(average_matrix_z)

        return population_aptitude_function, population_matrix_z
