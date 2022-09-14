import numpy as np


class Est_options:
    type = 0
    # for algo 0(DD-CRP) selected
    maxK = 15
    fixed_K = 0
    # algo 0 or 2 selected:
    samplerIter = 40  # Maximum Sampler Iterations
    # For type 0: 20-50 iter is sufficient
    # For type 2: >100 iter are needed
    do_plots = 1
    sub_sample = 2
    # Metric Hyper-parameters
    estimate_l = 1
    l_sensitivity = 5
    length_scale = []


# in this class, you should initialize type T and alpha
class Options:
    type = ''  # Type of Covariance Matrix: 'full' = NIW or 'Diag' = NIG
    T = 0  # Sampler Iterations
    alpha = 0  # maximum of similarity function
    Lambda = 0
    verbose = 0

    def __init__(self, type, T, alpha):
        self.type = type
        self.T = T
        self.alpha = alpha


# we should assign the value later, so we don't need to initialize parameters to the class
class Lambd:
    alpha_0 = 0
    beta_0 = 0
    nu_0 = 0
    Lambda_0 = 0
    mu_0 = 0
    kappa_0 = 0


class Psi_Stats:
    CompTimes = 0
    PostLogProbs = 0
    LogLiks = 0
    TotalClust = 0
    TableAssign = 0


# for readability, I
class Psis:
    C = np.array([])
    Z_C = np.array([])
    Lambda = None
    alpha = 0
    type = None
    table_members = None
    table_logLiks = None
    MaxLogProb = -np.inf
    Theta = None


# NIW Parameters storage
class New_lambdas:
    mu_N = None
    kappa_N = None
    alpha_N = None
    beta_N = None
    nu_N = None
    Lambda_N = None


# Theta Helper
class Thetas:
    Mu = None
    Sigma = None


