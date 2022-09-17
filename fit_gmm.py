import numpy as np
from utils import computePairwiseDistance
from utils.plotting import plotSimilarityConfMatrix
from Structs import Options, Lambd
from dd_crp.run_ddCRP_sampler import run_ddCRP_sampler
from gmm_stuff.my_gmm_cluster import my_gmm_cluster
from gmr import GMM
from utils.plotting.simple_classification_check import plot_result
from utils.plotting.plot_ellopsoid import plot_result_3D


def fig_gmm(Xi_ref, Xi_dot_ref, est_options):
    est_type = est_options.type
    do_plots = est_options.do_plots
    M = len(Xi_ref)
    N = len(Xi_ref[0])

    # in this program we use 0 to replace empty
    if est_options.sub_sample != 0:
        sub_sample = est_options.sub_sample
        Xi_ref = Xi_ref[:, ::sub_sample]
        Xi_dot_ref = Xi_dot_ref[:, ::sub_sample]

    if est_type != 1:
        if est_options.samplerIter != 0:
            if est_type == 0:
                samplerIter = 20
            if est_type == 2:
                samplerIter = 200
        else:
            samplerIter = est_options.samplerIter

        if est_options.l_sensitivity != 0:
            l_sensitivity = est_options.l_sensitivity
        else:
            l_sensitivity = 2

    if est_type == 0:
        # Option1: Non-parametric Clustering with Pos-Vel-cos-sim prior
        if est_options.estimate_l == 1:
            D, mode_hist_D, mean_D = computePairwiseDistance.compute(Xi_ref, 1)
            if mode_hist_D == 0:
                mode_hist_D = mean_D
            sigma = np.sqrt(mode_hist_D / l_sensitivity)  # warning because I haven't implemented full functionality
            l = 1 / (2 * (sigma ** 2))
        else:
            l = est_options.length_scale
        print('Computed length-scale l= {}'.format(l))

        # Compute element-wise cosine similarities
        len_of_Xi_dot = len(Xi_dot_ref[0])
        S = np.zeros((len_of_Xi_dot,len_of_Xi_dot))
        for i in np.arange(0, len_of_Xi_dot):
            for j in np.arange(0, len_of_Xi_dot):
                cos_angle = (np.dot(Xi_dot_ref[:, i], Xi_dot_ref[:, j])) / (np.linalg.norm(Xi_dot_ref[:, i]) * np.linalg.norm(Xi_dot_ref[:, j]))
                if np.isnan(cos_angle):
                    cos_angle = 0
                s = 1 + cos_angle

                # Compute Position component
                xi_i = Xi_ref[:, i]
                xi_j = Xi_ref[:, j]

                # Euclidean pairwise position-kernel
                p = np.exp(-l * np.linalg.norm(xi_i - xi_j))

                # Shifted Cosine Similarity of velocity vectors
                S[i][j] = p * s

        # Plot Similarity matrix
        if do_plots:
            title_str = 'Physically-Consistent Similarity Confusion Matrix'
            plotSimilarityConfMatrix.plot(S, title_str)

        # Setting sampler/model options (i.e. hyper-parameters, alpha, Covariance matrix)
        Xi_ref_mean = np.mean(Xi_ref, axis=1, keepdims=True)  # develop note: check dimension
        options = Options('full', samplerIter, 2)
        Lambda = Lambd()
        if options.type == 'diag':
            Lambda.alpha_0 = M
            Lambda.beta_0 = sum(np.diag())  # develop note: wait to be implement
        else:
            Lambda.nu_0 = M
            Lambda.Lambda_0 = np.diag(np.diag(np.cov(Xi_ref)))  # why here we do not need to transpose? solved
        Lambda.mu_0 = Xi_ref_mean  # hyper for N(mu_k|mu_0,kappa_0)
        Lambda.kappa_0 = 1  # hyper for N(mu_k|mu_0,kappa_0)

        # Run Collapsed Gibbs Sampler
        options.Lambda = Lambda
        options.verbose = 1
        # here is the code
        Psi, Psi_Stats = run_ddCRP_sampler(Xi_ref, S, options)
        est_labels = Psi.Z_C

        # Extract Learnt cluster parameters
        N = N / sub_sample  # change with sub_sample
        unique_labels = np.unique(est_labels)
        est_K = len(unique_labels)
        Priors = np.zeros(est_K)
        singletons = np.zeros(est_K)
        for k in np.arange(0, est_K):
            assigned_k = np.sum(est_labels == unique_labels[k] + 0)
            Priors[k] = assigned_k / N
            singletons[k] = assigned_k < 2  # find cluster with only 1 member

        Mu = Psi.Theta.Mu
        Sigma = Psi.Theta.Sigma

        # these code below should be tested further
        if any(singletons == 1):
            est_labels = my_gmm_cluster(Xi_ref, Priors, Mu, Sigma, 'hard', [])
            # plot_cluster(Xi_ref, est_labels)
            unique_labels = np.unique(est_labels)
            est_K = len(unique_labels)
            singletons_idx = np.argwhere(singletons == 1)
            singletons_idx = singletons_idx.reshape(len(singletons_idx), 1)
            Mu = np.delete(Mu, singletons_idx, 1)
            Sigma = np.delete(Sigma, singletons_idx, 0)
            Priors = np.zeros(est_K)
            for k in np.arange(0, est_K):
                assigned_k = np.sum(est_labels == unique_labels[k] + 0)
                Priors[k] = assigned_k / N
            print(np.sum(Priors))

        if len(Xi_ref) == 2:
            gmm = GMM(est_K, Priors, Mu.T, Sigma)
            plot_result(Xi_ref, gmm, est_K, Mu, len(Xi_ref))
        else:
            plot_result_3D(Mu, Sigma, Xi_ref)
            np.save('Mu_3D.npy', Mu)
            np.save('Sigma_3D.npy', Sigma)
        return Priors, Mu, Sigma









