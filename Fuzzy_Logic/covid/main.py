from covid import FuzzyNet
import numpy as np
import time


if __name__ == '__main__':
    # mx1,mx2,mx3,my1,my2,my3, dx1,dx2,dx3,dy1,dy2,dy3, p1,p2,p3,p4,p5,p6,p7,p8,p9, q1,q2,q3,q4,q5,q6,q7,q8,q9, r1,r2,r3,r4,r5,r6,r7,r8,r9
    generations = 100

    for generation in range(generations):
        population = np.random.randint(0, 255, size=(200, 39), dtype=np.uint8)
        print('Population\n', population, end='\n\n')

        cov19 = FuzzyNet()
        print('Base Z matrix\n', cov19.Z, end='\n\n')
        population_fa, population_matrix_z, best_FA_index = cov19.get_FA(population=population)

        if population_fa[best_FA_index] < 40:
            print('Best Z matrix\n', population_matrix_z[best_FA_index], population_fa[best_FA_index])
            cov19.plot(population_matrix_z[best_FA_index])
            time.sleep(5)

# xi , yj = zij