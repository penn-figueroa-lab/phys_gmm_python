import numpy as np
from scipy.special import multigammaln, gammaln


def table_logLik(Y, Lambda, type):
    M = len(Y)  # dimension
    N = len(Y[0])  # number of sample

    mu_0 = Lambda.mu_0
    kappa_0 = Lambda.kappa_0
    kappa_N = kappa_0 + N

    Ybar = np.mean(Y, axis=1, keepdims=True)
    Y_Ybar = Y - Ybar
    Ybar_mu = Ybar - mu_0

    if type == 'diag':
        return
    else:
        nu_0 = Lambda.nu_0
        Lambda_0 = Lambda.Lambda_0
        nu_N = nu_0 + N
        S = Y_Ybar @ Y_Ybar.T
        Lambda_N = Lambda_0 + S + (kappa_0 * N / kappa_N) * (Ybar_mu * Ybar_mu.T)
        # Marginal Likelihood p(Y|\lambda) Eq. 266
        # (page 21 . Conjugate Bayesian analysis of the Gaussian distribution 'Murphy')
        # logLik = log(gamma_ratios) +  (logdet_ratios) + log(kappa_ratios) + log (constant)

        logLik = (multigammaln(0.5 * nu_N, M) - multigammaln(0.5 * nu_0, M) + (nu_0 / 2) *
                   logDet(Lambda_0) - (nu_N / 2) * logDet(Lambda_N) + 0.5 * M *
                  (np.log(kappa_0) - np.log(kappa_N)) - 2 * N * M * np.log(np.pi)
                  )
    return logLik


def logDet(A):
    sign, logdet = np.linalg.slogdet(A)
    return sign * logdet