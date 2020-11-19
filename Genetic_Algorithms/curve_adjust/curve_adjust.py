"""This file contains the logic to use and create a genetic algorithm"""
import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin


class CurveAdjust:
    """"""

    def __init__(self, constants, weight, population_size, mutation, elitism=False):
        """
        Initialize all the global variables needed.

        :param constants: List with all the constants to use in the equation for the curve
        :param weight: Integer which multiplies the constant to get a chromosome
        :param population_size: Integer with the number of chromosomes in the population
        :param mutation: Float from 0.0 - 1.0 representing the percentage to mutate
        :param elitism: Boolean to enable or disable the elitism
        """
        self.best_chromosome = list()  # [chromosome, aptitude_function]
        self.aptitude_function_history = np.empty(0, dtype=np.float32)
        self.POPULATION_SIZE = abs(int(population_size))
        self.CHROMOSOME_SIZE = 7  # Fixed
        self.WEIGHT = abs(int(weight))
        self.CURVE_CONSTANTS = np.array(constants)
        # Arrays to graph the base curve
        # Chromosome = [A' B' C' D' E' F' G'], Constants = [A B C D E F G], Where A' = A * Weight
        self.CURVE_X = np.arange(start=0.1, stop=100.1, step=0.1)
        self.CURVE_Y = self.get_y(self.CURVE_CONSTANTS * self.WEIGHT, self.CURVE_X)
        self.MUTATION = abs(float(mutation))
        self.ELITISM = elitism if elitism is True or elitism is False else False

    def get_y(self, chromosome, x):
        """
        Convert the values from the current chromosome into an a Numpy Array corresponding to the
        'x' numpy array given.

        :param chromosome: Numpy array with numbers from 0 to 255 and a length of 7
        :param x: Numpy array with n quantity of numbers (Floats 0.1, ... ,100)
        :return: Numpy array with all the y points corresponding to the x points received
        """
        y = np.empty(0, dtype=np.float32)
        constants = np.copy(chromosome) / self.WEIGHT  # Constants = [A B C D E F G]
        constants = np.where(constants == 0, 0.2, constants)  # Avoid divisions by 0
        A, B, C, D, E, F, G = constants

        for i in x:
            expression = (A * (B * sin(i / C) + D * cos(i / E))) + (F * i) - G
            y = np.append(y, expression)

        return y

    def get_y_generator(self, chromosome, x):
        """
        Convert the values from the current chromosome into an a Numpy Array corresponding to the
        'x' numpy array given. It returns the y points one by one due to it is a generator.

        :param chromosome: Numpy array with numbers from 0 to 255 and a length of 7
        :param x: Numpy array with n quantity of numbers (Integers 0.1-100)
        :return: Float with the y point corresponding to the x point
        """
        constants = np.copy(chromosome) / self.WEIGHT  # Constants = [A B C D E F G]
        constants = np.where(constants == 0, 0.2, constants)
        A, B, C, D, E, F, G = constants

        for i in x:
            yield (A * (B * sin(i / C) + D * cos(i / E))) + (F * i) - G

    def get_random_population(self):
        """
        It generates a population with n chromosomes in random order.
        :return: Numpy Array with all the chromosomes [[A',B',C',D',E',F',G'], ... , []]
        """
        return np.random.randint(low=1, high=255, dtype=np.uint8,
                                 size=(self.POPULATION_SIZE, self.CHROMOSOME_SIZE))

    def get_aptitude_function(self, population):
        """
        Calculating the difference between point of the base curve against the chromosome

        :param population: Numpy Array with all the chromosomes
        :return: Numpy Array with all the errors for each chromosome from population
        """
        aptitude_function = np.empty(0, dtype=np.float32)
        for chromosome in population:
            y = self.get_y_generator(chromosome, self.CURVE_X)
            error = 0  # Getting the error from comparing curves
            for point_base_y in self.CURVE_Y:
                error += abs(point_base_y - next(y))
            aptitude_function = np.append(aptitude_function, [error])

        return aptitude_function

    def get_tournament_winner(self, population, aptitude_function):
        """
        Comparing random chromosomes to get the one with the best aptitude function

        :param population: Numpy Array with all the population [[1 ... n], ... ,[1 ... n]]
        :param aptitude_function: Numpy Array with all the aptitude functions for each chromosome
        :return: Numpy Array with the Chromosome with the lower error
        """
        # Getting contender indexes
        percentage = 0.05
        n_contenders = int(self.POPULATION_SIZE * percentage)
        n_contenders = 1 if n_contenders < 1 else n_contenders
        contenders_indexes = np.random.randint(0, self.POPULATION_SIZE, n_contenders)

        # Looking for the winner
        winner_index = contenders_indexes[0]
        for index in contenders_indexes:
            if aptitude_function[winner_index] > aptitude_function[index]:
                winner_index = index

        return population[winner_index]

    def reproduction(self, chromosome_x, chromosome_y):
        """
        Combining the 2 given chromosomes to generate 2 new ones.

        :param chromosome_x: Numpy Array with the Father chromosome
        :param chromosome_y: Numpy Array with the Mother chromosome
        :return: 2 Numpy Arrays with the son and daughter chromosomes
        """
        father = np.copy(chromosome_x)
        mother = np.copy(chromosome_y)
        cut_point = np.random.randint(0, 8 * (self.CHROMOSOME_SIZE - 1), dtype=np.uint8)

        if cut_point % 8 == 0:
            # Getting cut index - [0 ... index ... -1]
            index = int((cut_point / 8) - 1)

            # Adding the chunks - [father ... mother]
            son = np.copy(father[:index + 1])
            son = np.append(son, np.copy(mother[index + 1:]))

            # Adding the chunks - [mother ... father]
            daughter = np.copy(mother[:index + 1])
            daughter = np.append(daughter, np.copy(father[index + 1:]))

            # print("Index {}, Son {}, Daughter {}".format(index, son, daughter))

        else:
            # Getting indexes - [first_part ... cut_part ... second_part]
            first_index = cut_point // 8
            second_index = first_index + 2  # Skipping the cut point

            # Getting cut points - 1 byte - 8 bits - 0...255
            father_cut_point = father[first_index + 1]
            mother_cut_point = mother[first_index + 1]

            # Getting bits from each byte - Example: 1 byte = [2 bits: 6 bits]
            first_cut_index = np.uint8(cut_point % 8)
            second_cut_index = np.uint8(8 - first_cut_index)

            """ Example:
             father_cut_point = 134, mother_cut_point = 203
             first_cut_index = 2, second_cut_index = 6 -> Bits from a Byte(8 bits)
             Son = (left first 2 bits from 134 => 128) + (right first 6 bits from 203 => 11)
             Daughter = (left first 2 bits from 203 => 192) + (right first 6 bits from 134 => 6)
             Son = (128 + 11) = 139, Daughter = (192 + 6) = 198
            """

            # Getting the new cut part for son chromosome
            first_cut_part = np.uint8((father_cut_point >> second_cut_index) << second_cut_index)
            second_cut_part = np.uint8((mother_cut_point << first_cut_index) >> first_cut_index)
            son_cut_point = first_cut_part + second_cut_part

            # Getting the new cut part for daughter chromosome
            first_cut_part = np.uint8((mother_cut_point >> second_cut_index) << second_cut_index)
            second_cut_part = np.uint8((father_cut_point << first_cut_index) >> first_cut_index)
            daughter_cut_point = first_cut_part + second_cut_part

            # Adding the chunks [father ... cut point ... mother]
            son = np.copy(father[:first_index + 1])
            son = np.append(son, [son_cut_point])
            son = np.append(son, np.copy(mother[second_index:]))

            # Adding the chunks [mother ... cut point ... father]
            daughter = np.copy(mother[:first_index + 1])
            daughter = np.append(daughter, [daughter_cut_point])
            daughter = np.append(daughter, np.copy(father[second_index:]))
            """
            print("Cut index {} - Values {}, {}".format(first_index + 1, father[first_index + 1],
                                                        mother[first_index + 1]))
            print("Bits {}, {}".format(first_cut_index, second_cut_index))
            print("Son {}, Daughter {}".format(son_cut_point, daughter_cut_point))
            """

        return son, daughter

    def get_best_from_population(self, population, aptitude_function):
        """
        Getting the best chromosome and its aptitude function from the population

        :param population: Numpy Array with all the population [[1 ... n], ... ,[1 ... n]]
        :param aptitude_function: Numpy Array with all the aptitude functions for each chromosome
        :return : Numpy Arrays with the best chromosome and its aptitude function
        """
        # Finding the best from the current population
        index, best_index = 0, 0

        while index < self.POPULATION_SIZE:
            if aptitude_function[best_index] > aptitude_function[index]:
                best_index = index
            index += 1

        return population[best_index], aptitude_function[best_index]

    def graph(self, population, aptitude_function):
        """
        Graphing the base curve, current best chromosome curve and the aptitude function history.

        :param population: Numpy Array with all the population [[1 ... n], ... ,[1 ... n]]
        :param aptitude_function: Numpy Array with all the aptitude functions for each chromosome
        """
        best_chromosome = self.get_best_from_population(population, aptitude_function)

        plt.figure(1, figsize=(10, 6))
        plt.plot(self.CURVE_X, self.CURVE_Y, color="black", label="Base Curve")
        plt.grid(color="gray", linestyle="--")
        plt.title("Base Curve {} - Current Curve {}, {}".format(self.CURVE_CONSTANTS,
                                                                best_chromosome[0] / self.WEIGHT,
                                                                best_chromosome[1]))
        plt.plot(self.CURVE_X, self.get_y(best_chromosome[0], self.CURVE_X), color="blue",
                 label="Current Curve", linestyle="--", linewidth=0.5)
        plt.legend()
        plt.show(block=False)

        plt.figure(2)
        generation = len(self.aptitude_function_history)
        plt.plot(np.arange(0, generation, dtype=np.uint16), self.aptitude_function_history)
        plt.grid(color="gray", linestyle="--", linewidth=1, alpha=.4)
        plt.title("Closest to 0: {}".format(self.aptitude_function_history[-1]))
        plt.show(block=False)
        plt.pause(1)

        plt.figure(1)
        plt.clf()
        plt.figure(2)
        plt.clf()

    def add_mutation(self, population):
        """
        Modifying the given population applying a mutation single or double for some specific
        percentage of chromosomes from population.

        :param population: Numpy Array with all the population [[1 ... n], ... ,[1 ... n]]
        """
        # Getting mutation_indexes
        mutations = int(self.POPULATION_SIZE * self.MUTATION)
        mutations = 1 if mutations < 1 else mutations
        mutation_indexes = np.random.randint(0, self.POPULATION_SIZE, mutations)
        gen_size = 8  # 8 bits for a number from 0 - 255
        # print("\nMutation indexes from population {}".format(mutation_indexes))
        for population_index in mutation_indexes:
            chromosome = population[population_index]
            chromosome_index = np.random.randint(0, self.CHROMOSOME_SIZE)
            # print("\nChromosome {}, Index {}".format(chromosome, chromosome_index))
            gen = chromosome[chromosome_index]
            gen_index = np.random.randint(1, gen_size + 1)  # You need to start from 1
            # print("Gen {}, Index (bit) {}".format(gen, gen_index))
            gen ^= (1 << (gen_index - 1))
            # print("Gen mutated {}, Mask {}".format(gen, (1 << gen_index)))
            population[population_index][chromosome_index] = gen
            # print("Gen mutated original {}".format(population[population_index][chromosome_index]))
            # print("Chromosome mutated original {}".format(population[population_index]))

            if np.random.randint(3):
                chromosome_index = np.random.randint(0, self.CHROMOSOME_SIZE)
                # print("\nChromosome {}, Index {}".format(chromosome, chromosome_index))
                gen = chromosome[chromosome_index]
                gen_index = np.random.randint(1, gen_size + 1)  # You need to start from 1
                # print("Gen {}, Index (bit) {}".format(gen, gen_index))
                gen ^= (1 << (gen_index - 1))
                # print("Gen mutated {}, Mask {}".format(gen, (1 << gen_index)))
                population[population_index][chromosome_index] = gen
                # print("Gen mutated original {}".format(population[population_index][chromosome_index]))
                # print("Chromosome mutated original {}".format(population[population_index]))

    def get_population_with_elitism(self, population, child_population, aptitude_function,
                                    child_aptitude_function):
        """
        Joining the parent and child population and then getting only the half part with best
        aptitude function.

        :param population: Numpy Array with all the population [[1 ... n], ... ,[1 ... n]]
        :param child_population: Numpy Array with all the child population
        :param aptitude_function: Numpy Array with all the aptitude functions for each chromosome
        :param child_aptitude_function: Numpy Array with the aptitude functions for each chromosome

        :return: Numpy Array with the best chromosomes from parent and child population
        """
        # Join the two populations and aptitude functions
        new_population = np.copy(population)
        new_population = np.append(new_population, child_population, axis=0)
        new_aptitude_function = np.copy(aptitude_function)
        new_aptitude_function = np.append(new_aptitude_function, child_aptitude_function, axis=0)

        # Getting an ordered list of tuples with (index, aptitude function value)
        indexes = np.arange(0, new_population.shape[0])
        mapping_table = [(k, v) for k, v in zip(indexes, new_aptitude_function)]
        mapping_table.sort(key=lambda tup: tup[1])

        # Getting the first half of the new ordered population
        ordered_population = np.empty((0, self.CHROMOSOME_SIZE), dtype=np.uint8)
        for index, aptitude_function_value in mapping_table[:self.POPULATION_SIZE]:
            ordered_population = np.append(ordered_population, [new_population[index]], axis=0)

        return ordered_population

    def get_next_generation(self, population):
        """
        Getting the next generation of chromosomes, saving the best one and its aptitude function
        for plotting it.

        :param population: Numpy Array with all the population [[1 ... n], ... ,[1 ... n]]
        :return: Numpy Arrays with the child population and its aptitude function
        """
        aptitude_function = self.get_aptitude_function(population)
        child_population = np.empty((0, self.CHROMOSOME_SIZE), dtype=np.uint8)
        iterations = self.POPULATION_SIZE // 2

        for i in range(iterations):
            # Getting father and mother chromosomes
            father_chromosome = self.get_tournament_winner(population, aptitude_function)
            mother_chromosome = self.get_tournament_winner(population, aptitude_function)

            # Reproduce parents to get childes
            son, daughter = self.reproduction(father_chromosome, mother_chromosome)
            # print("Parents: {}, {}".format(father_chromosome, mother_chromosome))
            # print("Childes: {}, {}\n".format(son, daughter))

            # Add childes as rows for the new population (new generation)
            child_population = np.append(child_population, [son], axis=0)
            child_population = np.append(child_population, [daughter], axis=0)

        if (iterations * 2) < self.POPULATION_SIZE:  # Repeat one more time with only one child
            father_chromosome = self.get_tournament_winner(population, aptitude_function)
            mother_chromosome = self.get_tournament_winner(population, aptitude_function)

            son, daughter = self.reproduction(father_chromosome, mother_chromosome)
            # print("Parents: {}, {}".format(father_chromosome, mother_chromosome))
            # print("Childes: {}, {}\n".format(son, daughter))

            childes_population = np.append(child_population, [son], axis=0)

        if self.ELITISM:
            child_aptitude_function = self.get_aptitude_function(child_population)
            child_population = self.get_population_with_elitism(population, child_population,
                                                                aptitude_function,
                                                                child_aptitude_function)
            self.add_mutation(child_population)
            child_aptitude_function = self.get_aptitude_function(child_population)
        else:
            # Getting new population with some mutations
            self.add_mutation(child_population)

            # Getting the new aptitude function from new population
            child_aptitude_function = self.get_aptitude_function(child_population)

        # Saving and comparing against the best chromosome from history
        # Example: best_chromosome = [chromosome, aptitude_function]
        best_chromosome = self.get_best_from_population(child_population, child_aptitude_function)
        if self.best_chromosome:
            if self.best_chromosome[1] > best_chromosome[1]:
                self.best_chromosome = best_chromosome
        else:
            self.best_chromosome = best_chromosome

        # Save the best aptitude function from the current population
        self.aptitude_function_history = np.append(self.aptitude_function_history,
                                                   best_chromosome[1])

        return child_population, child_aptitude_function
