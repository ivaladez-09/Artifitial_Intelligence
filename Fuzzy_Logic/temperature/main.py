from temp import FuzzyNet
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Define variables
    c = np.array([2, 6, 13.5])  # Media de los datos
    r = np.array([2.2, 3, 4.5])  # Desviacion estandar
    p = np.array([7, 4, 2])
    q = np.array([22, 2, -3])

    x = np.arange(1, 12.1, 0.1)  # Step = 0.1
    y = np.empty(0, dtype=np.float32)
    temp = FuzzyNet(c, r, p, q)
    for xi in x:
        yi = temp.run(xi)
        y = np.append(y, yi)

    temp.graph(x, y)
    input("Pause")
