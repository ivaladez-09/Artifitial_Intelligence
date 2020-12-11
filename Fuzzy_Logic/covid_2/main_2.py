from fuzzy_net import FuzzyNet as fn
import numpy as np
import time
import random as rd


if __name__ == '__main__':
    # mx1,mx2,mx3,my1,my2,my3, dx1,dx2,dx3,dy1,dy2,dy3, p1,p2,p3,p4,p5,p6,p7,p8,p9, q1,q2,q3,q4,q5,q6,q7,q8,q9, r1,r2,r3,r4,r5,r6,r7,r8,r9
    generations = 1

    for generation in range(generations):
        population = [[round(rd.uniform(-25, 25), 2) for j in range(0, 39)] for i in range(1)]
        # population = np.random.randint(0, 51, size=(5, 39), dtype=np.uint8)
        print('\nPopulation\n', population, end='\n\n')

        for chromosome in population:
            cov19 = fn()
            print('Base Z matrix\n', cov19.Z, end='\n\n')
            matrix_z, aptitude_function = cov19.get_results(list(chromosome))
            print('Matrix Z and aptitude function\n', matrix_z, aptitude_function, end='\n\n')
