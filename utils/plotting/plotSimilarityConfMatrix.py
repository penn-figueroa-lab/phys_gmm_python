import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def plot(S, title_str):
    fig, ax0 = plt.subplots(1, 1)
    c = ax0.pcolor(S, cmap='jet')
    ax0.set_title(title_str)
    fig.tight_layout()
    plt.colorbar(c)
    plt.show()
