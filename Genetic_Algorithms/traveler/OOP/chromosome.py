"""This file contains the class for creating 'chromosomes'."""
import random
import math


class Chromosome:
    """Class to get basic attributes and methods for a chromosome."""
    size = None
    data = None
    aptitude_function = None

    def __init__(self, size: int, data=None):
        """Initialize the object.
        param size: Integer with the size of the chromosome
        param data: List with all the data for the chromosome. This parameter is optional.
        """
        self.size = size
        if data is None:
            self.data = [num for num in range(1, self.size + 1)]
            random.shuffle(self.data)
        elif len(data) != size:
            raise ValueError("The size of variable 'data' must be equal to variable 'size'")
        else:
            self.data = data

    def calculate_aptitude_function(self, mapping_table: dict) -> float:
        """Calculate the aptitude function of the current data in the chromosome.

        param mapping_table: Dictionary with the form {1:(p1, p2), ... , n:(p1, p2)}
        return: Float with the current aptitude function"""

        def calculate_distance(p1, p2):
            """Get distance between two given points"""
            return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

        index, aptitude_function = 0, 0
        while index + 1 < self.size:
            aptitude_function += calculate_distance(mapping_table.get(self.data[index]),
                                                    mapping_table.get(self.data[index + 1]))
            index += 1

        self.aptitude_function = aptitude_function
        return self.aptitude_function

    def reproduce(self):
        """Combine genes from a parent Chromosome into a new one.
        return: Chromosome with all the new content (child) from a parent Chromosome"""
        new_data = list()
        reproduction_method = random.randint(0, 3)

        if reproduction_method == 0:
            # Reversing a chunk from the array -> [15, 1,2,3, 10] -> [15, 3,2,1, 10]
            start_index = random.randint(0, self.size - 1)
            end_index = random.randint(start_index + 1, self.size)

            if start_index == 0:
                new_data = self.data[:start_index] + \
                           self.data[end_index::-1] + \
                           self.data[end_index + 1:]
            else:
                new_data = self.data[:start_index] + \
                           self.data[end_index:start_index - 1:-1] + \
                           self.data[end_index + 1:]

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

            new_data = self.data[:start_index_a] + \
                       self.data[start_index_b: end_index_b + 1] + \
                       self.data[end_index_a + 1: start_index_b] + \
                       self.data[start_index_a: end_index_a + 1] + \
                       self.data[end_index_b + 1:]

        if len(new_data) != len(self.data):
            raise Exception('The size of the child chromosome is not of the same size that the parent.')

        return Chromosome(size=self.size, data=new_data)
