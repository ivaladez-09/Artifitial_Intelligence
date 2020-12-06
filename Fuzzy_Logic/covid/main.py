from covid import FuzzyNet
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


if __name__ == '__main__':
    # mx1,mx2,mx3,my1,my2,my3, dx1,dx2,dx3,dy1,dy2,dy3, p1,p2,p3,p4,p5,p6,p7,p8,p9, q1,q2,q3,q4,q5,q6,q7,q8,q9, r1,r2,r3,r4,r5,r6,r7,r8,r8
    population = np.random.randint(1, 254, size=(200, 39), dtype=np.uint8)
    print('Population\n', population, end='\n\n')

    cov19 = FuzzyNet()
    print('Base Z matrix\n', cov19.Z, end='\n\n')
    population_fa, population_matrix_z = cov19.get_FA(population=population)

    for fa, matrix_z in zip(population_fa, population_matrix_z):
        if fa < 20:
            print('Nice sample\n', matrix_z, fa, end='\n\n')
            # cov19.plot(matrix_z)

