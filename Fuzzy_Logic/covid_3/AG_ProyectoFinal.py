import random
import math
import sys
import copy
from fuzzy_net import FuzzyNet as fn

populationLen, chromosomeLen = 200, 39
w = 1.275
elitismo = False
mutation = True
X = [i for i in range(1, 5)]
Y = [j for j in range(1, 9)]
redTS = fn(X, Y)
ciclos = 35


def createPopulation(populationLen, chromosomeLen):
    population = []
    for i in range(populationLen):
        chromosome = random.sample(range(0, 256), chromosomeLen)
        aptitude, matrix = calcAptitude(chromosome)
        population.append({"chromosome": chromosome, "aptitude": aptitude, "matrix": matrix})
    return sorted(population, key=lambda chromosome: chromosome["aptitude"])


def calcAptitude(chromosome):
    matrix_z, aptitude_function = redTS.get_results(decodeChromosome(chromosome))
    return aptitude_function, matrix_z


def decodeChromosome(chromosome):
    dChromosome = []

    for x in chromosome:
        dChromosome.append((x / w) - 80)

    return dChromosome


def chromosomeTournament(population, populationPercent):
    chromosomes = random.sample(range(len(population)), int((len(population) * populationPercent) / 100))

    min = sys.maxsize
    winner = []
    for chromosome in chromosomes:

        if population[chromosome]["aptitude"] < min:
            min = population[chromosome]["aptitude"]
            winner = population[chromosome]

    return winner


def createNextGeneration(population):
    tmpPopulation = []
    for i in range(int(len(population) / 2)):
        winnerChromosome1, winnerChromosome2 = chromosomeTournament(population, 3), chromosomeTournament(population, 3)

        sonChromosome1, sonChromosome2 = copy.deepcopy(
            reproduction(winnerChromosome1["chromosome"], winnerChromosome2["chromosome"]))

        aptitude1, matrix1 = calcAptitude(sonChromosome1)
        aptitude2, matrix2 = calcAptitude(sonChromosome2)

        tmpPopulation.append({"chromosome": sonChromosome1, "aptitude": aptitude1, "matrix": matrix1})
        tmpPopulation.append({"chromosome": sonChromosome2, "aptitude": aptitude2, "matrix": matrix2})

    return sorted(tmpPopulation, key=lambda chromosome: chromosome["aptitude"])


def reproduction(fatherChromosome1, fatherChromosome2):
    cutChromosomePoint = random.randint(0, (len(fatherChromosome1)) * 8) / 8
    cutAleloPoint = cutChromosomePoint % 1
    cutChromosomePoint = math.floor(cutChromosomePoint)
    cutAleloPoint = math.floor(int(cutAleloPoint * 6))
    sonChromosome1 = copy.deepcopy(fatherChromosome1)
    sonChromosome2 = copy.deepcopy(fatherChromosome2)
    if (cutAleloPoint == 0):
        if (cutChromosomePoint == 0):
            sonChromosome1, sonChromosome2 = [sonChromosome1[0]] + sonChromosome2[1:], [
                sonChromosome2[0]] + sonChromosome1[1:]
        elif (cutChromosomePoint == 6):
            sonChromosome1, sonChromosome2 = sonChromosome1[0:-1] + [sonChromosome2[-1]], sonChromosome2[0:-1] + [
                sonChromosome1[-1]]
        else:
            sonChromosome1, sonChromosome2 = sonChromosome1[0:cutChromosomePoint] + sonChromosome2[cutChromosomePoint:], sonChromosome2[0:cutChromosomePoint] + sonChromosome1[cutChromosomePoint:]
    else:
        tempAlelo1, tempAlelo2 = reproductionAlelo(sonChromosome1[cutChromosomePoint],
                                                   sonChromosome2[cutChromosomePoint], cutAleloPoint)
        if (cutChromosomePoint == 0):
            sonChromosome1, sonChromosome2 = [sonChromosome1[0]] + sonChromosome2[1:], [
                sonChromosome2[0]] + sonChromosome1[1:]
        elif (cutChromosomePoint == 6):
            sonChromosome1, sonChromosome2 = sonChromosome1[0:-1] + [sonChromosome2[-1]], sonChromosome2[0:-1] + [
                sonChromosome1[-1]]
        else:
            sonChromosome1, sonChromosome2 = sonChromosome1[0:cutChromosomePoint] + [tempAlelo2] + sonChromosome2[cutChromosomePoint + 1:], sonChromosome2[0:cutChromosomePoint] + [tempAlelo1] + sonChromosome1[cutChromosomePoint + 1:]

    return sonChromosome1, sonChromosome2


def reproductionAlelo(fatherAlelo1, fatherAlelo2, cutAleloPoint):
    sonAlelo1, sonAlelo2 = format(fatherAlelo1, '08b'), format(fatherAlelo2, '08b')
    if (cutAleloPoint == 0):
        sonAlelo1, sonAlelo2 = [sonAlelo1[0]] + sonAlelo2[1:], [sonAlelo2[0]] + sonAlelo1[1:]
    elif (cutAleloPoint == 6):
        sonAlelo1, sonAlelo2 = sonAlelo1[0:-1] + [sonAlelo2[6]], sonAlelo2[0:-1] + [sonAlelo1[6]]
    else:
        sonAlelo1, sonAlelo2 = sonAlelo1[0:cutAleloPoint] + sonAlelo2[cutAleloPoint:], sonAlelo2[0:cutAleloPoint] + sonAlelo1[cutAleloPoint:]

    return int(sonAlelo1, 2), int(sonAlelo2, 2)


def populationMutation(population, populationPercent):
    chromosomes = random.sample(range(len(population)), int((len(population) * populationPercent) / 100))
    for chromosome in chromosomes:
        cutChromosomePoint = random.randint(0, (len(population[chromosome])) * 8) / 8
        cutAleloPoint = cutChromosomePoint % 1
        cutChromosomePoint = math.floor(cutChromosomePoint)
        cutAleloPoint = math.floor(int(cutAleloPoint * 6))

        alelo = format(population[chromosome]["chromosome"][cutChromosomePoint], '08b')
        alelo = alelo[:cutAleloPoint] + ('1' if alelo[cutAleloPoint] == '0' else '0') + alelo[cutAleloPoint + 1:]

        population[chromosome]["chromosome"][cutChromosomePoint] = int(alelo, 2)
        aptitude, matrix = calcAptitude(population[chromosome]["chromosome"])
        population[chromosome]["aptitude"] = aptitude
        population[chromosome]["matrix"] = matrix

    return population


def main():
    population = copy.deepcopy(createPopulation(populationLen, chromosomeLen))

    aptitudes = [0 for i in range(ciclos)]
    aptitudes[0] = population[0]["aptitude"]
    minAptitude = sys.maxsize

    for i in range(ciclos):
        if (elitismo):
            tempPopulation = copy.deepcopy(
                sorted(population + createNextGeneration(population), key=lambda chromosome: chromosome["aptitude"]))
            population = copy.deepcopy(tempPopulation[:len(population)])
        else:
            population = copy.deepcopy(createNextGeneration(population))

        if (mutation):
            population = copy.deepcopy(
                sorted(populationMutation(population, 5), key=lambda chromosome: chromosome["aptitude"]))

        aptitudes[i] = population[0]["aptitude"]
        if (aptitudes[i] < minAptitude):
            minAptitude = aptitudes[i]

        redTS.plot(population[0]["matrix"], ciclos, minAptitude, aptitudes)
        print(decodeChromosome(population[0]["chromosome"]))
        print(population[0]["aptitude"])

    input('End?')


if __name__ == "__main__":
    main()
