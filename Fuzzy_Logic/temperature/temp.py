"""Fuzzy net with one input and one output"""
import numpy as np
from math import exp
import matplotlib.pyplot as plt


class FuzzyNet:
    """"""

    def __init__(self, c, r, p, q):
        """"""
        if len(c) == len(r) == len(p) == len(q):
            self.c = c
            self.r = r
            self.p = p
            self.q = q
        else:
            print('The length of all parameters MUST be the same!')
            print('Lengths: c = {}, r = {}, p = {}, q = {}'.format(len(c), len(r), len(p), len(q)))
            raise Exception

    def run(self, x):
        """"""
        # Evaluate all the membership functions with the different parameters - MF(i) = f(x,c,r)
        membership_functions = [self.mf_gaussian(x, self.c[i], self.r[i]) for i in range(len(self.c))]

        # Getting the inference of each rule - I(i) = MF(i) * (pi*x + qi)
        inferences = [membership_functions[i] * (self.p[i] * x + self.q[i]) for i in range(len(self.c))]

        # Return 'Y(i)'
        return sum(inferences) / sum(membership_functions)

    @staticmethod
    def mf_gaussian(x, c, r):
        """"""
        return exp(-1 / 2 * pow((x - c) / r, 2))

    @staticmethod
    def get_gaussian(x, c, r):
        """"""
        get_y = lambda xi: exp(-1 / 2 * pow((xi - c) / r, 2))
        y = np.empty(0, dtype=np.float32)
        for xi in x:
            y = np.append(y, get_y(xi))
        return y

    def graph(self, x, y):
        """"""
        real_x = np.arange(1, 13)
        real_y = np.array([24, 26, 29, 31, 32, 30, 27, 27, 27, 27, 26, 25])

        plt.figure(1).subplots_adjust(hspace=0.0, wspace=1)
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        ax0.grid(color="gray", linestyle="--", linewidth=0.5)
        ax1.grid(color="gray", linestyle="--", linewidth=0.5)

        ax0.plot(x, y, color="red", label="Estimated Curve", linestyle="--")
        ax0.plot(real_x, real_y, color="blue", label="Real Curve")

        colors = ('red', 'green', 'blue')
        labels = ('low', 'medium', 'high')
        for i in range(len(self.c)):
            y_gaussian = self.get_gaussian(x, self.c[i], self.r[i])
            ax1.plot(x, y_gaussian, color=colors[i], label=labels[i])

        ax0.legend()
        ax1.legend()

        plt.figure(1)
        plt.show()
