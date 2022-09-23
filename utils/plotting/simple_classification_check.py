import matplotlib.pyplot as plt
from gmr import GMM, plot_error_ellipses
from itertools import cycle
import numpy as np


def plot_result(Xi_ref, gmm, est_K, Mu, dim):
    if dim == 2:
        plt.figure()
        ax = plt.subplot(111)
        cycol = cycle('bgrcmky')
        ax.scatter(Xi_ref[0], Xi_ref[1], c='g')

        colors = []

        for i in np.arange(0, est_K):
            colors.append(next(cycol))

        plot_error_ellipses(ax, gmm, alpha=0.1, colors=colors)
        for num in np.arange(0, len(Mu[0])):
            plt.text(Mu[0][num], Mu[1][num], str(num+1), fontsize=20)
    else:
        helper = 1

    plt.show()
