""""""
import random
import math
from copy import deepcopy
import matplotlib.pyplot as plt


class Chromosome:
    """"""

    def __init__(self, chromosome_size: int, mapping_table: dict):
        """"""
        self.size = chromosome_size
        self.content = [num for num in range(1, self.size + 1)]
        random.shuffle(self.content)
        self.mapping_table = mapping_table
        self.aptitude_function = self.get_aptitude_function()

    def get_aptitude_function(self) -> float:
        """"""

        def get_distance(p1, p2):
            """Get distance between two given points"""
            return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

        aptitude_function = 0
        for index, gen in enumerate(self.content):
            if index + 1 < self.size:
                aptitude_function += get_distance(
                    self.mapping_table.get(self.content[index]),
                    self.mapping_table.get(self.content[index + 1])
                )

        self.aptitude_function = aptitude_function  # Update AF
        return aptitude_function

    def reproduction(self) -> list:
        """"""
        new_content = list()
        reproduction_method = random.randint(0, 3)

        if reproduction_method == 0:  # Reversing a chunk from the array -> [15, 1,2,3, 10] -> [15, 3,2,1, 10]
            start_index = random.randint(0, self.size - 1)
            end_index = random.randint(start_index + 1, self.size)

            if start_index == 0:
                new_content += self.content[:start_index] + \
                              self.content[end_index::-1] + \
                              self.content[end_index + 1:]
            else:
                new_content += self.content[:start_index] + \
                              self.content[end_index:start_index - 1:-1] + \
                              self.content[end_index + 1:]

        else:  # [15,1, 2, 3,10] -> [3,10, 2, 15,1]
            while True:
                start_index_a = random.randint(0, self.size - 4)
                end_index_a = random.randint(start_index_a + 1, self.size - 3)
                chunk_size = end_index_a - start_index_a

                if end_index_a + 1 >= self.size - chunk_size:
                    continue  # Try again

                start_index_b = random.randint(end_index_a + 1, self.size - chunk_size)
                end_index_b = start_index_b + chunk_size
                if end_index_b >= self.size:
                    continue  # Try again

                break  # Everything went ok

            new_content += self.content[:start_index_a] + \
                           self.content[start_index_b: end_index_b + 1] + \
                           self.content[end_index_a + 1: start_index_b] + \
                           self.content[start_index_a: end_index_a + 1] + \
                           self.content[end_index_b + 1:]

        if len(new_content) != len(self.content):
            raise Exception('The size of the child chromosome is not of the same size that the parent.')

        return new_content


class Population:
    """"""

    def __init__(self, population_size: int, chromosome_size: int, mapping_table: dict):
        """"""
        self.size = population_size
        self.chromosomes = [Chromosome(chromosome_size, mapping_table) for n in range(self.size)]

    def get_tournament_winner(self) -> Chromosome:
        """"""
        # Getting contenders indexes
        percentage = 0.05
        contenders = int(self.size * percentage)
        contenders = 1 if contenders < 1 else contenders
        contenders_indexes = random.sample(range(0, self.size), k=contenders)

        # Looking for the lower aptitude function
        winner_index = contenders_indexes[0]
        for index in contenders_indexes:
            if self.chromosomes[winner_index].get_aptitude_function() > \
                    self.chromosomes[index].get_aptitude_function():
                winner_index = index

        return deepcopy(self.chromosomes[winner_index])

    def get_best_chromosome(self) -> Chromosome:
        """"""
        best_index = 0
        for index, chromosome in enumerate(self.chromosomes):
            if self.chromosomes[best_index].get_aptitude_function() > self.chromosomes[index].get_aptitude_function():
                best_index = index

        return deepcopy(self.chromosomes[best_index])


class Traveler:
    """"""

    def __init__(self, population_size: int, coordinates: list):
        """"""
        cities = [city for city in range(1, len(coordinates) + 2)]
        self.mapping_table = {city: coordinate for city, coordinate in zip(cities, coordinates)}
        self.random_population = Population(population_size=int(population_size),
                                            chromosome_size=len(coordinates),
                                            mapping_table=self.mapping_table)
        self.best_chromosome = None
        self.aptitude_function_history = list()

    def run(self, generations: int):
        """"""
        parent_population = deepcopy(self.random_population)
        for generation in range(1, generations + 1):
            child_population = self.get_next_generation(parent_population)
            parent_population = deepcopy(child_population)

            best_chromosome = child_population.get_best_chromosome()
            self.aptitude_function_history.append(best_chromosome.aptitude_function)
            if self.best_chromosome:
                if self.best_chromosome.aptitude_function > best_chromosome.aptitude_function:
                    self.best_chromosome = best_chromosome
            else:
                self.best_chromosome = best_chromosome

            print(best_chromosome.content, self.best_chromosome.content)
            print(best_chromosome.get_aptitude_function(), self.best_chromosome.get_aptitude_function())

            self.plot(population=child_population, generation=generation)

    def get_next_generation(self, population: Population) -> Population:
        """"""
        next_generation = deepcopy(population)
        for iteration in range(population.size):
            parent_chromosome = population.get_tournament_winner()
            next_generation.chromosomes[iteration].content = parent_chromosome.reproduction()
            # print(parent_chromosome.content, next_generation.chromosomes[iteration].content)

        return next_generation

    def plot(self, population: Population, generation: int):
        """"""
        best_chromosome = population.get_best_chromosome()
        # print("Best chromosome from generation #{}:  {}".format(generation, best_chromosome.content))

        # Getting coordinates for plotting them
        coordinates_x, coordinates_y = [], []
        for gene in best_chromosome.content:
            coordinates_x.append(self.mapping_table[gene][0])
            coordinates_y.append(self.mapping_table[gene][1])

        best_coordinates_x, best_coordinates_y = [], []
        for gene in self.best_chromosome.content:
            best_coordinates_x.append(self.mapping_table[gene][0])
            best_coordinates_y.append(self.mapping_table[gene][1])

        # Plotting
        plt.figure(1, figsize=(10, 6))
        plt.scatter(coordinates_x, coordinates_y, color="green")
        plt.plot(coordinates_x, coordinates_y, color="blue", label="Generation best chromosome",
                 linestyle="--", linewidth=0.5)

        plt.scatter(best_coordinates_x, best_coordinates_y, color="red")
        plt.plot(best_coordinates_x, best_coordinates_y, color="black",
                 label="History best chromosome")
        plt.grid(color="gray", linestyle="--", linewidth=1, alpha=.4)
        plt.title("Chromosome: Generation {} - History {}".format(best_chromosome.content,
                                                                       self.best_chromosome.content))
        plt.legend()
        plt.show(block=False)

        plt.figure(2)
        plt.plot([n for n in range(generation)], self.aptitude_function_history)
        plt.grid(color="gray", linestyle="--", linewidth=1, alpha=.4)
        plt.title("Best Distances {}".format(self.aptitude_function_history[-1]))
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.5)

        plt.figure(1)
        plt.clf()

        plt.figure(2)
        plt.clf()