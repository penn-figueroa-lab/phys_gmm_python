import numpy as np
from dd_crp.sample.compute_lambdasN import compute_lambdasN
from Structs import Thetas
from scipy.stats import invwishart


def sample_TableParams(Y, Z_C, Lambda, type):

    # Compute lambda_N parameters
    new_lambdas = compute_lambdasN(Y, Z_C, Lambda, type)

    # New Cluster Means \mu = {mu_1, ... \mu_K}
    Theta = Thetas()
    Theta.Mu = new_lambdas.mu_N

    # New Cluster Means \mu = {mu_1, ... \mu_K}
    M = len(Y)
    K = len(Theta.Mu[0])

    Sigma = np.zeros((K, M, M))

    if type == 'diag':
        helper = 1
    else:
        for k in np.arange(0, K):
            kth_sig = invwishart.rvs(new_lambdas.nu_N[k], new_lambdas.Lambda_N[k])
            Sigma[k] = kth_sig

    Theta.Sigma = Sigma

    return Theta
