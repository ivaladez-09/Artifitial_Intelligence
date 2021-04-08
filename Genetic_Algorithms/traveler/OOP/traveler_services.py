"""This file contains the required methods to apply the genetic algorith logic to the traveler problem."""
import matplotlib.pyplot as plt
from population import Population


class TravelerServices:
    """Class to get services for solving the traveler problem."""
    mapping_table = None
    initial_population = None
    best_chromosome = None
    aptitude_function_history = list()

    def __init__(self, population_size: int, coordinates: list):
        """
        param population_size: Integer with the size of population
        param coordinates: List of tuples with the coordinates for each city -> [(p1,p2), ...]
        """
        cities = [city for city in range(1, len(coordinates) + 2)]
        self.mapping_table = {city: coordinate for city, coordinate in zip(cities, coordinates)}
        self.initial_population = Population(size=population_size, chromosome_size=len(coordinates))

    def run(self, generations: int):
        """Run all the processes needed for applying the genetic algorithm to the traveler problem.
        param generations: Integer with all the generation wanted for the genetic algorithm process."""
        parent_population = self.initial_population
        for generation in range(1, generations + 1):
            child_population = self.get_next_generation(parent_population)
            parent_population = child_population

            best_chromosome = child_population.get_best_chromosome(self.mapping_table)
            self.aptitude_function_history.append(best_chromosome.calculate_aptitude_function(self.mapping_table))
            if self.best_chromosome:
                if self.best_chromosome.aptitude_function > best_chromosome.aptitude_function:
                    self.best_chromosome = best_chromosome
            else:
                self.best_chromosome = best_chromosome
                # Calculate for first time the AF of the best chromosome
                self.best_chromosome.calculate_aptitude_function(self.mapping_table)

            # self.plot(population=child_population, generation=generation)

    def get_next_generation(self, population: Population) -> Population:
        """Generate a new population based on the 'population' parameter.

        param population: Population with the content for creating the next generation.
        return: Population equals to the next generation of the population."""
        next_generation = Population(size=population.size, chromosome_size=len(population.chromosomes[0].data))
        for i in range(population.size):
            parent_chromosome = population.get_tournament_winner(self.mapping_table)
            next_generation.chromosomes[i] = parent_chromosome.reproduce()

        return next_generation

    def plot(self, population: Population, generation: int):
        """"""
        best_chromosome = population.get_best_chromosome(self.mapping_table)
        # print("Best chromosome from generation #{}:  {}".format(generation, best_chromosome.content))

        # Getting coordinates for plotting them
        coordinates_x, coordinates_y = [], []
        for gene in best_chromosome.data:
            coordinates_x.append(self.mapping_table[gene][0])
            coordinates_y.append(self.mapping_table[gene][1])

        best_coordinates_x, best_coordinates_y = [], []
        for gene in self.best_chromosome.data:
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
        plt.title("Chromosome: Generation {} - History {}".format(best_chromosome.data,
                                                                  self.best_chromosome.data))
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
