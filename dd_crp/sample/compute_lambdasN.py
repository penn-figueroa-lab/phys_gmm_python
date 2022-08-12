import numpy as np
from Structs import New_lambdas


def compute_lambdasN(Y, Z_C, Lambda, type):
    # Extract # Clusters and Auxiliary variables
    M = len(Y)
    K = len(np.unique(Z_C))

    # Normal Hyperparameters
    mu_0 = Lambda.mu_0
    kappa_0 = Lambda.kappa_0

    # Compute Parameters for each K tables
    mu_N = np.zeros((M, K))
    kappa_N = np.zeros(K)

    if type == 'diag':
        alpha_N = np.zeros(K)
        beta_N = np.zeros((M,K))
    else:
        nu_N = np.zeros(K)
        Lambda_N = np.zeros((K, M, M))

    for k in np.arange(0, K):
        # Auxiliary Variables per Tables
        k_offset = k + 1
        N_helper = (Z_C == k_offset) + 0
        N = np.sum(N_helper)
        Y_helper = np.argwhere(N_helper == 1)
        helper = Y[:, Y_helper[:, 0]]
        Ybar = np.mean(Y[:, Y_helper[:, 0]], axis=1, keepdims= True)
        YbarN = Ybar * N
        Y_Ybar = Y[:, Y_helper[:, 0]].copy() - Ybar
        Y_YbarN = Y[:, Y_helper[:, 0]].copy() - YbarN
        Ybar_mu = Ybar - mu_0
        helper = mu_N[:, k]
        # Compute Mean Parameters K-th table
        mu_N[:, k] = ((kappa_0 * mu_0 + YbarN) / (kappa_0 + N)).reshape(2)

        # Compute Kappa Parameters K-th table
        kappa_N[k] = kappa_0 + N

        if type == 'diag':
            helper = 1
        else:
            nu_0 = Lambda.nu_0
            Lambda_0 = Lambda.Lambda_0
            nu_N[k] = nu_0 + N
            S = Y_Ybar @ Y_Ybar.T
            Lambda_N[k] = Lambda_0 + S + (kappa_0 * N/kappa_N[k]) * (Ybar - mu_0) @ (Ybar - mu_0).T

    new_lambdas = New_lambdas()
    new_lambdas.mu_N = mu_N
    new_lambdas.kappa_N = kappa_N

    if type == 'diag':
        helper = 1
    else:
        new_lambdas.nu_N = nu_N
        new_lambdas.Lambda_N = Lambda_N

    return new_lambdas



        


