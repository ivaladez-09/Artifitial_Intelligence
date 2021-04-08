"""Contains the logic to use and create a genetic algorithm to solve the traveler problem."""
import numpy as np
import math
import matplotlib.pyplot as plt


class Traveler:
    """"""

    def __init__(self, population_size, coordinates):
        """
        Initializing traveler object
        :param population_size: Integer with the size of the population
        :param coordinates: List of Tuples with the cities coordinates [(1,2), ... , (7,12)]
        """
        self.POPULATION_SIZE = abs(int(population_size))
        self.CHROMOSOME_SIZE = len(coordinates)
        cities = np.arange(1, self.CHROMOSOME_SIZE + 1, dtype=np.uint8)
        self.MAPPING_TABLE = {city: coordinate for city, coordinate in zip(cities, coordinates)}
        self.best_chromosome = list()  # [chromosome, aptitude_function]
        self.aptitude_function_history = np.empty(0, dtype=np.float32)

    def get_random_population(self):
        """
        Getting a random population
        :return: Numpy array of specific population and chromosome size [[1 ... n], ... ,[1 ... n]]
        """
        random_population = np.empty((0, self.CHROMOSOME_SIZE), dtype=np.uint8)
        random_chromosome = np.arange(1, self.CHROMOSOME_SIZE + 1, dtype=np.uint8)

        for _ in range(0, self.POPULATION_SIZE):
            np.random.shuffle(random_chromosome)
            random_population = np.append(random_population, [random_chromosome], axis=0)

        return random_population

    def get_aptitude_function(self, population):
        """
        Calculate the summation of distances between points for each chromosome in the population.

        :param population: Numpy Array with all the population [[1 ... n], ... ,[1 ... n]]
        :return: Numpy Array with all the aptitude functions for each chromosome
        """

        def get_distance(p1, p2):
            """Get distance between two given points"""
            return np.float32(math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2)))

        aptitude_function, population_index = np.empty(0, dtype=np.float32), 0
        while population_index < self.POPULATION_SIZE:
            summation, chromosome_index = 0, 0
            while chromosome_index + 1 < self.CHROMOSOME_SIZE:
                summation += get_distance(
                    self.MAPPING_TABLE.get(population[population_index][chromosome_index]),
                    self.MAPPING_TABLE.get(population[population_index][chromosome_index + 1])
                )
                chromosome_index += 1
            aptitude_function = np.append(aptitude_function, [summation], axis=0)
            population_index += 1

        return aptitude_function

    def get_tournament_winner(self, population, aptitude_function):
        """
        Comparing random chromosomes to get the one with the best aptitude function

        :param population: Numpy Array with all the population [[1 ... n], ... ,[1 ... n]]
        :param aptitude_function: Numpy Array with all the aptitude functions for each chromosome
        :return: Numpy Array with the Chromosome with the lower distance
        """
        # Getting contenders indexes
        percentage = 0.05
        n_contenders = int(self.POPULATION_SIZE * percentage)
        n_contenders = 1 if n_contenders < 1 else n_contenders
        contenders_indexes = np.random.choice(self.POPULATION_SIZE, n_contenders)

        # Looking for the lower aptitude function
        winner_index = contenders_indexes[0]
        for index in contenders_indexes:
            if aptitude_function[winner_index] > aptitude_function[index]:
                winner_index = index

        return population[winner_index]

    def reproduction(self, chromosome):
        """
        Modifying a little bit the genes from the chromosome in 1 of 2 possible ways.

        :param chromosome: Numpy Array with the number (tag) of cities [1, ... , 14]
        return: Numpy Array with some genes changes from the original chromosome
        """
        child_chromosome = np.copy(chromosome)
        option = np.random.randint(0, 2, dtype=np.uint8)  # Randomly select a reproduction option

        if option == 0:  # Reversing a chunk from the array -> [1,2,3] -> [3,2,1]
            start_index = np.random.randint(0, self.CHROMOSOME_SIZE - 1, dtype=np.uint8)
            end_index = np.random.randint(start_index, self.CHROMOSOME_SIZE, dtype=np.uint8)

            # Specific logic for Python syntax
            revert_index = start_index - 1
            if revert_index > 0:
                inverted_chunk = np.copy(child_chromosome[end_index:revert_index:-1])
            else:
                inverted_chunk = np.copy(child_chromosome[end_index::-1])

            # print("Inverted chunk {} - ({},{})".format(inverted_chunk, start_index, end_index))

            swap_index = 0
            for idx in range(start_index, end_index + 1):
                child_chromosome[idx] = inverted_chunk[swap_index]
                swap_index += 1

        else:
            # Getting a random index for chunks A and B
            while True:
                start_index_a = np.random.randint(0, self.CHROMOSOME_SIZE - 4, dtype=np.uint8)
                end_index_a = np.random.randint(start_index_a + 1, self.CHROMOSOME_SIZE - 3,
                                                dtype=np.uint8)
                chunk_size = end_index_a - start_index_a
                if end_index_a + 1 >= self.CHROMOSOME_SIZE - chunk_size:
                    continue
                start_index_b = np.random.randint(end_index_a + 1,
                                                  self.CHROMOSOME_SIZE - chunk_size, dtype=np.uint8)
                end_index_b = start_index_b + chunk_size
                if end_index_b >= self.CHROMOSOME_SIZE:
                    continue  # Try again

                break  # Everything went ok

            # Getting chunks
            first_chunk = np.copy(child_chromosome[start_index_a: end_index_a + 1])
            second_chunk = np.copy(child_chromosome[start_index_b: end_index_b + 1])

            # print("1° chunk {}, Indexes ({},{})".format(first_chunk, start_index_a, end_index_a))
            # print("2° chunk {}, Indexes ({},{})".format(second_chunk, start_index_b, end_index_b))

            # Swapping the chunks in the child chromosome
            swap_index = 0
            for idx in range(start_index_a, end_index_a + 1):
                child_chromosome[idx] = second_chunk[swap_index]
                child_chromosome[start_index_b] = first_chunk[swap_index]
                start_index_b += 1
                swap_index += 1

        return child_chromosome

    def get_best_from_population(self, population, aptitude_function):
        """
        Getting the best chromosome from the population based on its aptitude function

        :param population: Numpy Array with all the population [[1 ... n], ... ,[1 ... n]]
        :param aptitude_function: Numpy Array with all the aptitude functions for each chromosome
        :return: Numpy Arrays with the Chromosome with the lower distance and its aptitude function
        """
        # Finding the best from the current population
        index, best_index = 0, 0

        while index < self.POPULATION_SIZE:
            if aptitude_function[best_index] > aptitude_function[index]:
                best_index = index
            index += 1

        return population[best_index], aptitude_function[best_index]

    def get_next_generation(self, population):
        """
        Getting the next population based on a parent one and saving the best chromosome

        :param population: Numpy Array with all the population [[1 ... n], ... ,[1 ... n]]
        """
        aptitude_function = self.get_aptitude_function(population)
        child_population = np.empty((0, self.CHROMOSOME_SIZE), dtype=np.uint8)

        for i in range(self.POPULATION_SIZE):
            parent_chromosome = self.get_tournament_winner(population, aptitude_function)
            child_chromosome = self.reproduction(parent_chromosome)
            child_population = np.append(child_population, [child_chromosome], axis=0)
            # print("Parent: {}, Child: {}\n\n".format(parent_chromosome, child_chromosome))

        child_aptitude_function = self.get_aptitude_function(child_population)

        # Saving and comparing against the best chromosome from history
        best_chromosome = self.get_best_from_population(child_population, child_aptitude_function)
        if self.best_chromosome:
            if self.best_chromosome[1] > best_chromosome[1]:
                self.best_chromosome = best_chromosome
        else:
            self.best_chromosome = best_chromosome

        # Save the best aptitude function from the current population
        self.aptitude_function_history = np.append(self.aptitude_function_history,
                                                   best_chromosome[1])

        # print(best_chromosome[0], self.best_chromosome[0])
        # print(best_chromosome[1], self.best_chromosome[1])

        return child_population, child_aptitude_function

    def graph(self, population, aptitude_function):
        """
        Plotting the best chromosome from generation an history as well as the
        aptitude function history through generations.

        :param population: Numpy Array with all the population [[1 ... n], ... ,[1 ... n]]
        :param aptitude_function: Numpy Array with all the aptitude functions for each chromosome
        """
        generation = len(self.aptitude_function_history)
        best_chromosome = self.get_best_from_population(population, aptitude_function)
        print("Best chromosome from generation #{}:  {}".format(generation, best_chromosome))

        # Getting coordinates for plotting them
        coordinates_x, coordinates_y = [], []
        for gene in best_chromosome[0]:
            coordinates_x.append(self.MAPPING_TABLE[gene][0])
            coordinates_y.append(self.MAPPING_TABLE[gene][1])

        best_coordinates_x, best_coordinates_y = [], []
        for gene in self.best_chromosome[0]:
            best_coordinates_x.append(self.MAPPING_TABLE[gene][0])
            best_coordinates_y.append(self.MAPPING_TABLE[gene][1])

        # Plotting
        plt.figure(1, figsize=(10, 6))
        plt.scatter(coordinates_x, coordinates_y, color="green")
        plt.plot(coordinates_x, coordinates_y, color="blue", label="Generation best chromosome",
                 linestyle="--", linewidth=0.5)
        plt.scatter(best_coordinates_x, best_coordinates_y, color="red")
        plt.plot(best_coordinates_x, best_coordinates_y, color="black",
                 label="History best chromosome")
        plt.grid(color="gray", linestyle="--", linewidth=1, alpha=.4)
        plt.title("Best Chromosome: Generation {} - History {}".format(best_chromosome,
                                                                       self.best_chromosome))
        plt.legend()
        plt.show(block=False)

        plt.figure(2)
        plt.plot(np.arange(0, generation, dtype=np.uint16), self.aptitude_function_history)
        plt.grid(color="gray", linestyle="--", linewidth=1, alpha=.4)
        plt.title("Best Distances {}".format(self.aptitude_function_history[-1]))
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(1)

        plt.figure(1)
        plt.clf()

        plt.figure(2)
        plt.clf()
