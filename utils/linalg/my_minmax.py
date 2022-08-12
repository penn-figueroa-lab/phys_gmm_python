import numpy as np


def my_minmax(x):
    x_sorted = np.sort(x)
    mm = np.array([x_sorted[0], x_sorted[-1]])
    return mm


"""
x = np.array([1,2,3,4])
print(my_minmax(x))
# test code
"""