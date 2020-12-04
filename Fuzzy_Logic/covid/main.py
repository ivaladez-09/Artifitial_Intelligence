from covid import FuzzyNet
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


if __name__ == '__main__':
    # mx1,mx2,mx3,my1,my2,my3, dx1,dx2,dx3,dy1,dy2,dy3, p1,p2,p3,p4,p5,p6,p7,p8,p9, q1,q2,q3,q4,q5,q6,q7,q8,q9, r1,r2,r3,r4,r5,r6,r7,r8,r8
    population = np.random.randint(1, 254, size=(20000, 39), dtype=np.uint8)
    print(population, end='\n\n')

    cov19 = FuzzyNet()
    print(cov19.Z, end='\n\n')
    for chromosome in population:
        fa, matrix_z = cov19.get_FA(chromosome=chromosome)
        if fa < 15:
            print(matrix_z, fa, end='\n\n')

            Y, X = np.meshgrid(cov19.Y, cov19.X)

            fig = plt.figure()
            ax = plt.axes(projection='3d')
            surf = ax.plot_surface(X, Y, matrix_z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

            # Customize the z axis.
            ax.set_zlim(0, 100)

            # Add a color bar which maps values to colors.
            fig.colorbar(surf, shrink=0.5, aspect=20)

            plt.show()

