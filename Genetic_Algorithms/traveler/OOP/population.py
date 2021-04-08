"""This file containts the class for creating 'populations'."""
from chromosome import Chromosome
import random


class Population:
    """Class to get basic attributes and methods for a population."""
    size = None
    chromosomes = None

    def __init__(self, size: int, chromosome_size: int):
        """Initialize the object.
        param size: Integer with the size of the population
        param chromosome_size: Integer with the size of the chromosome"""
        self.size = size
        self.chromosomes = [Chromosome(chromosome_size) for _ in range(self.size)]

    def get_tournament_winner(self, mapping_table: dict) -> Chromosome:
        """Identify the Chromosome winner of a tournament. Choosing only the 5% of the population size.

        param mapping_table: Dictionary for mapping the key with its coordinates -> {1:(p1,p2), ... , n:(px,py)}
        return: Chromosome that was identified as the best of the ones that participated on the tournament."""
        # Getting contenders indexes
        percentage = 0.05
        contenders = int(self.size * percentage)
        contenders = 1 if contenders < 1 else contenders
        contenders_indexes = random.sample(range(0, self.size), k=contenders)

        # Looking for the lower aptitude function
        winner_index = contenders_indexes[0]
        for index in contenders_indexes:
            if self.chromosomes[winner_index].calculate_aptitude_function(mapping_table) > \
                    self.chromosomes[index].calculate_aptitude_function(mapping_table):
                winner_index = index

        return self.chromosomes[winner_index]

    def get_best_chromosome(self, mapping_table: dict) -> Chromosome:
        """Identify the current best chromosome on population.

        param mapping_table: Dictionary for mapping the key with its coordinates -> {1:(p1,p2), ... , n:(px,py)}
        return: Chromosome which has the best aptitude function."""
        best_index, index = 0, 0
        while index < self.size:
            if self.chromosomes[best_index].calculate_aptitude_function(mapping_table) > \
                    self.chromosomes[index].calculate_aptitude_function(mapping_table):
                best_index = index
            index += 1

        return self.chromosomes[best_index]
