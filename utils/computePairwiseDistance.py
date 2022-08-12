import numpy as np
import time
from scipy.spatial.distance import pdist, squareform
from scipy import stats


def compute(X, display_hist):
    X = np.transpose(X)
    print(np.shape(X))
    maxSamples = 10000
    if len(X) < maxSamples:
        X_train = X
        hist_distances = 1
    else:
        X_train = X[0:maxSamples, :]
        hist_distances = 10
    start = time.perf_counter()
    D = pdist(X_train, 'euclidean')
    end = time.perf_counter()
    mean_D = np.mean(D)
    max_D = np.max(D)
    hist, bin_edges = np.histogram(D, bins=70)  # caution change bins to adjust result
    max_values_id = np.argmax(hist)
    print("pair calculations take: {}, mean is {}, max is {}".format(start - end, mean_D, max_D))
    return D, bin_edges[max_values_id], mean_D


