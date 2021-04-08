from traveler import Traveler
import time


POPULATION_SIZE = 200
GENERATIONS = 80
COORDINATES = [(1, 7), (2, 5), (4, 4), (2, 3), (3, 2),
               (1, 1), (5, 1), (7, 3), (6, 6), (10, 5),
               (9, 8), (13, 6), (12, 3), (13, 1)]

if __name__ == '__main__':
    t1 = time.time()
    traveler = Traveler(population_size=POPULATION_SIZE, coordinates=COORDINATES)
    parent_population = traveler.get_random_population()

    for generation in range(1, GENERATIONS + 1):
        child_population, aptitude_function = traveler.get_next_generation(parent_population)
        # traveler.graph(child_population, aptitude_function)
        parent_population = child_population
        # print("\n\t =================== {} Generation =================== \n\n".format(generation))

    print(time.time() - t1)
    print("Best from History {}".format(traveler.best_chromosome))
    print("Aptitude function History {}".format(traveler.aptitude_function_history))
    input("Pause")
