import numpy as np
from gmm_stuff.my_gaussPDF import my_gaussPDF
"""
%MY_GMM_CLUSTER Computes the cluster labels for the data points given the GMM
%
%   input -----------------------------------------------------------------
%   
%       o X      : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o Priors : (1 x K), the set of priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu     : (N x K), an NxK matrix corresponding to the centroids 
%                           mu = {mu^1,...mu^K}
%       o Sigma  : (N x N x K), an NxNxK matrix corresponding to the 
%                           Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%       o type   : string ,{'hard', 'soft'} type of clustering
%
%       o softThresholds: (2 x 1), a vecor for the minimum and maximum of
%                           the threshold for soft clustering in that order
%
%   output ----------------------------------------------------------------
%
%       o labels   : (1 x M), a M dimensional vector with the label of the
%                             cluster for each datapoint
%                             - For hard clustering, the label is the 
%                             cluster number.
%                             - For soft clustering, the label is 0 for 
%                             data points which do not have high confidnce 
%                             in cluster assignment
%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
This function is waited to be checked further (08/11/22)
"""


# is it for re-clustering?
def my_gmm_cluster(X, Priors, Mu, Sigma, type, softThresholds):
    # Auxiliary Variables
    N = len(X)
    M = len(X[0])
    K = len(Mu[0])

    # Initializing variables
    Pk_x = np.zeros((K, M))
    labels = np.zeros(M)
    Pk_x_max = np.zeros(M)
    Px_k = np.zeros((K, M))

    # Find the a posteriori probability for each data point for each cluster
    for ii in np.arange(0, K):
        c2 = Mu[:, ii].reshape((N, 1))
        Px_k[ii] = my_gaussPDF(X, c2, Sigma[ii])

    for ii in np.arange(0, M):
        Priors_trans = Priors.reshape(len(Priors), 1)
        Px_k_trans = Px_k[:, ii].reshape(len(Px_k), 1)
        # helper = (Priors_trans * Px_k_trans)
        Pk_x[:, ii] = (Priors * Px_k[:, ii]) / np.sum(Priors_trans * Px_k_trans)

    for ii in np.arange(0, M):
        if type == 'hard':
            labels[ii] = np.argmax(Pk_x[:, ii]) + 1
            Pk_x_max[ii] = np.max(Pk_x[:, ii])

    return labels

