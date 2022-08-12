import numpy as np
from Structs import Lambd, Options, Psi_Stats, Psis
from dd_crp.helper.extract_TableIds import extract_TableIds
from dd_crp.probs.table_logLik import table_logLik
from dd_crp.sample.sample_ddCRPMM import sample_ddCRPMM
from dd_crp.probs.logPr_spcmCRP import logPr_spcmCRP
from dd_crp.sample.sample_TableParams import sample_TableParams
"""
 Distance Dependent Chinese Restaurant Process Mixture Model.
 **Inputs**
          Y: projected M-dimensional points  Y (y1,...,yN) where N = dim(S),
          S: Similarity Matrix where s_ij=1 is full similarity and
          s_ij=0 no similarity between observations
         
 **Outputs**
          Psi (MAP Markov Chain State)
          Psi.LogProb:
          Psi.Z_C:
          Psi.clust_params:
          Psi.iter:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 Copyright (C) 2018 Learning Algorithms and Systems Laboratory,          %
 EPFL, Switzerland                                                       %
 Author:  Nadia Figueroa                                                 % 
 email:   nadia.figueroafernandez@epfl.ch                                %
 website: http://lasa.epfl.ch                                            %
                                                                         %
 This work was supported by the EU project Cogimon H2020-ICT-23-2014.    %
                                                                         %
 Permission is granted to copy, distribute, and/or modify this program   %
 under the terms of the GNU General Public License, version 2 or any     %
 later version published by the Free Software Foundation.                %
                                                                         %
 This program is distributed in the hope that it will be useful, but     %
 WITHOUT ANY WARRANTY; without even the implied warranty of              %
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General%
 Public License for more details                                         %
                                                                         %
 If you use this code in your research please cite:                      %
 "A Physically-Consistent Bayesian Non-Parametric Mixture Model for      %
   Dynamical System Learning."; N. Figueroa and A. Billard; CoRL 2018    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


# Y is Xi_ref(2x385), S is similarity matrix, options contain the lambda
def run_ddCRP_sampler(Y, S, options):
    # Data Dimensionality
    M = len(Y)
    N = len(Y[0])
    C = np.arange(1, N + 1)
    # Default Hyper-parameters
    if options is not None:
        T = options.T
        alpha = options.alpha
        type = options.type
        Lambda = options.Lambda
        # C = options.init_clust (not implemented yet)
    else:
        T = 100
        alpha = 1
        type = 'full'
        Lambda = Lambd()
        Lambda.mu_0 = 0
        Lambda.kappa_0 = 1
        Lambda.nu_0 = M
        Lambda.Lambda_0 = np.eye(M) * M * 0.5

    # Initialize Stats Variables
    psi_Stats = Psi_Stats()
    psi_Stats.CompTimes = np.zeros(T, dtype='float')
    psi_Stats.PostLogProbs = np.zeros(T, dtype='float')
    psi_Stats.LogLiks = np.zeros(T, dtype='float')
    psi_Stats.TotalClust = np.zeros(T,dtype='int')
    psi_Stats.TableAssign = []

    ##############################################
    # Define Initial Markov Chain State Psi^{t-1}#
    ##############################################

    # Augment Similarity Matrix with alpha on diagonal
    S = S + np.eye(N) * alpha
    # S_alpha = num2cell(S,2)

    # Compute Initial Customer/Table Assignments and Likelihoods
    table_members = []
    table_logLiks = []
    Z_C = extract_TableIds(C)
    K = np.max(Z_C)
    # k+1
    for k in np.arange(1, K+1):
        members = np.argwhere(Z_C == k)
        table_members.append(members[:, 0]) # store python style index
        Y_in = Y[:, members[:, 0]]  # Y_in is customer who sit together represented by column vector
        current_table_LogLik = table_logLik(Y_in, Lambda, type)
        table_logLiks.append(current_table_LogLik)

    # Load initial variables
    Psi = Psis()
    Psi.C = C
    Psi.Z_C = Z_C
    Psi.Lambda = Lambda
    Psi.alpha = alpha
    Psi.type = type
    Psi.table_members = table_members
    Psi.table_logLiks = table_logLiks
    Psi.MaxLogProb = -np.inf

    ################################
    # Run Gibbs Sampler for dd-CRP #
    ################################
    if options.verbose == 1:
        print('*** Initialized with {} clusters out of {} observations ***'.format(K, N))
        print('Running dd-CRP Mixture Sampler...')
        # tic

    for i in np.arange(0, T):
        # Draw Sample dd(SPCM)-CRP
        Psi.C, Psi.Z_C, Psi.table_members, Psi.table_logLiks = sample_ddCRPMM(Y, S, Psi, type)
        # Compute the Posterior Conditional Probability of current Partition
        LogProb, data_LogLik = logPr_spcmCRP(Y, S, Psi)
        if options.verbose == 1:
            print('Iteration {}: Started with {} clusters '.format(i, np.max(Psi.Z_C)))
            print('--> moved to {} clusters with logprob = {}'.format(max(Psi.Z_C), LogProb))

        # Store Stats
        # wait to be implement
        psi_Stats.PostLogProbs[i] = LogProb
        psi_Stats.LogLiks[i] = data_LogLik
        psi_Stats.TotalClust[i] = np.max(Psi.Z_C)
        psi_Stats.TableAssign.append(Psi.Z_C.copy())

        # If current posterior is higher than previous update MAP estimate
        if psi_Stats.PostLogProbs[i] > Psi.MaxLogProb:
            Psi.MaxLogProb = psi_Stats.PostLogProbs[i]
            Psi.Maxiter = i

    Psi.Z_C = psi_Stats.TableAssign[Psi.Maxiter]
    Psi.Theta = sample_TableParams(Y, Psi.Z_C, Lambda, type)

    if options.verbose == 1:
        # toc
        print('***********************************************************************')

    return Psi, Psi_Stats










