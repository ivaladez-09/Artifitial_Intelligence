"""
File to generate memebership functions to use in fuzzy sets.
"""
import numpy as np
from math import exp
import matplotlib.pyplot as plt


class MF:
    """"""

    def __init__(self, x_size):
        """"""
        self.X = np.arange(0, x_size + 1)

    def gaussian(self, c, r):
        """
        (x,50,20)
        """
        get_y = lambda x: exp(-1 / 2 * pow((x - c) / r, 2))
        y = np.empty(0, dtype=np.float32)
        for x in self.X:
            y = np.append(y, get_y(x))
        return y

    def triangle(self, a, b, c):
        """
        (x, 20, 60, 80)
        """
        if a >= b >= c:
            return None
        get_y = lambda x: max(min((x - a)/(b - a), (c - x)/(c - b)), 0)
        y = np.empty(0, dtype=np.float32)
        for x in self.X:
            y = np.append(y, get_y(x))
        return y

    def trapezoid(self, a, b, c, d):
        """
        (x, 10, 20, 60, 95)
        """
        if a >= b > c >= d:
            return None
        get_y = lambda x: max(min((x - a)/(b - a), 1, (d - x)/(d - c)), 0)
        y = np.empty(0, dtype=np.float32)
        for x in self.X:
            y = np.append(y, get_y(x))
        return y

    def sigmoid(self, a, c):
        """
        (x, 2, 5)
        """
        get_y = lambda x: 1/(1 + exp(-a * (x - c)))
        y = np.empty(0, dtype=np.float32)
        for x in self.X:
            try:
                y = np.append(y, get_y(x))
            except Exception as err:
                print(x, a, c, y)
                raise err
        return y


if __name__ == '__main__':
    mf = MF(x_size=250)
    y_gaussian = mf.gaussian(50, 20)
    y_triangle = mf.triangle(40, 80, 100)
    y_trapezoid = mf.trapezoid(90, 100, 140, 175)
    y_sigmoid = mf.sigmoid(2, 150)

    plt.figure(1, figsize=(10, 6))
    plt.plot(mf.X, y_gaussian, color="blue", label='Gaussian')
    plt.plot(mf.X, y_triangle, color="red", label='Triangle')
    plt.plot(mf.X, y_trapezoid, color="black", label='Trapezoid')
    plt.plot(mf.X, y_sigmoid, color="green", label='Sigmoid')

    plt.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.title("Membership function")
    plt.show()



