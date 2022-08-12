import numpy as np
from dd_crp.probs.table_logLik import table_logLik


def logPr_spcmCRP(Y, delta, Psi):
    # Current Markov State
    N = len(Y[0])
    C = Psi.C
    Z_C = Psi.Z_C
    clust_ids = np.unique(Z_C)

    # Hyperparameters
    alpha = Psi.alpha
    Lambda = Psi.Lambda
    type = Psi.type

    # Compute Prior Probability on Partition
    prior_LogLik = 0
    for i in np.arange(0, len(C)):
        if i == C[i] - 1:
            prior_LogLik = prior_LogLik + np.log(alpha/(alpha + N))
        else:
            prior_LogLik = prior_LogLik + np.log(delta[i][C[i] - 1] / (alpha + np.sum(delta[i])))

    # Compute Likelihood of Partition
    data_LogLik = 0
    for i in np.arange(0, len(clust_ids)):
        k = clust_ids[i]
        Z_C_helper = (Z_C == k) + 0
        if not(np.sum(Z_C_helper) <= 1):
            Z_C_helper = np.argwhere(Z_C_helper == 1)
            Y_in = Y[:, Z_C_helper[:, 0]]
            data_LogLik = data_LogLik + table_logLik(Y_in, Lambda, type)

    post_LogProb = prior_LogLik + data_LogLik
    return post_LogProb, data_LogLik


