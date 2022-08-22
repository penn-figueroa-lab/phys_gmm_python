import numpy as np
from dd_crp.helper.get_Connections import get_Connections
from dd_crp.probs.table_logLik import table_logLik
from utils.linalg.my_minmax import my_minmax

# Y is Xi_ref(2x385), S is similarity matrix
# This is algorithm 1 of the Paper (Page 14)

"""
% Gibbs Sampling of the dd-CRP
% **Inputs** 
%   o Y:          Data points in Spectral Space
%   o S_alpha:    Similarities in Original Space (self-simiarity = alpha)
%   o Psi:        Current Markov Chain State
%
% **Outputs** 
%   o C:              Customer Assignments
%   o Z_C:            Table Assignments
%   o table_members:  Cluster Members
%   o table_logLiks:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2018 Learning Algorithms and Systems Laboratory,          %
% EPFL, Switzerland                                                       %
% Author:  Nadia Figueroa                                                 % 
% email:   nadia.figueroafernandez@epfl.ch                                %
% website: http://lasa.epfl.ch                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

# Note: Z_C and C store matlab style index, which means that C[0] = 1 means the point 1, referred by index 0
# sit with itself. Z_C[0] = 1 means the first point sit at table 1.
# table_Logliks[0] means first table so if you want to get prob in 1st table you should code
# table_Logliks[Z_C[0] - 1] <- add a -1 offset
def sample_ddCRPMM(Y, S_alpha, Psi, type):
    # Data Dimensionality
    N = len(Y[0])  # number of observation

    # Extracting current markov state
    C = Psi.C  # C use same index as matlab, if the first entry sit by itself it will be 1 not 0
    Z_C = Psi.Z_C  # contains table assignment which not affected by different index rule
    alpha = Psi.alpha
    Lambda = Psi.Lambda  # lambda contains 4 NIW parameters
    type = Psi.type
    table_members = Psi.table_members
    ##############################################################
    # each entry contains current table, caution: the entry start
    # from 0, which means the member of table 1 will be at entry of index 0
    ##############################################################
    table_logLiks = Psi.table_logLiks
    K = np.max(Z_C)

    # Sample random permutation of observations \tau of [1,..,N]
    tau = np.random.permutation(N)  # generate python style index, permutation 0 to N - 1
    # For every i-th randomly sampled observation sample a new cluster
    # assignment c_i
    # 注意，C与Z_C使用的都是标准matlab index，C表示你和谁坐，用matlab的index表示，
    # Z_C是你在哪张桌子上
    for i in np.arange(0, N):
        c_i = tau[i]
        ##################################################################
        # Step 1: "Remove" the c_i, i.e. the outgoing link of customer i #
        # to do this, set its cluster to c_i, and set its connected customers to c_i
        ##################################################################
        ci_old = C[c_i]
        # document who it sits with, if obs 1 sit itself it will be ci_old = C[0] = 1
        helper = int(Z_C[c_i][0])  # get the table belonging
        old_conn_customers = table_members[helper - 1]  # for exp: if Z_C[0] = 1, the it is related to index 0

        # new assigned connected customer (assign to itself)
        C[c_i] = c_i + 1  # for exp: C[0] = 0 + 1 = 1
        # new connected customers (table) considering the removal of c_i from C; i.e. C_{-i}
        new_conn_customers = get_Connections(C, c_i)
        # return <index> of the connection
        if len(new_conn_customers) != len(old_conn_customers):
            # Increase number of tables
            K = K + 1

            # Adding new customer cycle as new table and removing other
            # linked customers
            table_members.append(new_conn_customers)  # create new table
            indexs = np.in1d(old_conn_customers, new_conn_customers)  # return array length equal to old, check old in
            # new, note 老的在新的里的东西，移除了老table里被分出去的部分，也就是完成了顾客的分流
            indexs = indexs + 0  # convert Boolean array to Integer array
            indexs = np.argwhere(indexs == 1)
            indexs = indexs.reshape(len(indexs))
            table_members[Z_C[c_i][0] - 1] = np.delete(table_members[Z_C[c_i][0] - 1], indexs.reshape(len(indexs)))

            # Creating new table
            Z_C[new_conn_customers] = K

            # Likelihood of old table without c_i
            # (recompute likelihood of customers sitting without c_i)
            old_table_id = Z_C[ci_old - 1]  # if old table ID is 1 then you should refer to index 0
            members = np.argwhere(Z_C == old_table_id)
            Y_in = Y[:, members[:, 0]]
            table_logLiks[old_table_id[0] - 1] = table_logLik(Y_in, Lambda, type)

            # Likelihood of new table created by c_i
            new_table_id = K
            # (compute likelihood of new customers sitting with c_i)
            members = np.argwhere(Z_C == new_table_id)
            Y_in = Y[:, members[:, 0]]
            table_logLiks.append(table_logLik(Y_in, Lambda, type))

        #########################################
        # Compute priors p(c_i = j | S, \alpha) #
        #########################################
        assign_priors = S_alpha[c_i]
        # assign_Priors = S_alpha[:, c_i]
        assign_logPriors = np.log(assign_priors)  # in matlab we transpose to 385 x 1
        assign_logPriors = assign_logPriors.reshape((len(assign_priors), 1))

        ########################################
        # Compute the conditional distribution #
        ########################################

        # Current table assignment
        table_id_curr = Z_C[c_i]

        # Current clusters / tables
        table_ids = np.unique(Z_C)
        tables = len(table_ids)

        # Compute log-likelihood of clusters given data
        new_logLiks = np.zeros((tables, 1))
        sum_logLiks = np.zeros((tables, 1))
        old_logLiks = table_logLiks[0:tables]

        # TODO: THESE LINE CAN BE MADE MORE EFFICIENT ------>>
        # Eq. 30 Likelihood of Partition
        for j in np.arange(0, tables):
            k = table_ids[j]
            if table_id_curr == k:
                sum_logLiks[j] = np.sum(old_logLiks)
            else:
                others = np.ones(tables, dtype=int)
                others[j] = 0
                others[table_ids == table_id_curr] = 0
                Z_C_helper = Z_C.reshape(len(Z_C))
                Y_in = np.append(Y[:, Z_C_helper == table_id_curr], Y[:, Z_C_helper == k], axis=1)
                new_logLiks[j] = table_logLik(Y_in, Lambda, type)
                old_logLiks_helper = np.array(old_logLiks, dtype='float')
                sum_logLiks[j] = np.sum(old_logLiks_helper[others == 1]) + new_logLiks[j]

        # Compute Cluster LogLikes
        data_logLik = np.zeros((N, 1))
        for ii in np.arange(0, N):
            data_logLik[ii] = sum_logLiks[table_ids == Z_C[ii]]

        data_logLik = data_logLik - np.max(data_logLik)

        ###################################################################
        # STEP 2: Sample new customer assignment from updated conditional #
        ###################################################################
        # Compute log cond prob of all possible cluster assignments
        # Eq. 31
        log_cond_prob = assign_logPriors + data_logLik

        # Compute conditional distribution
        # convert to probability sans log
        cond_prob = np.exp(log_cond_prob)
        # normalize
        cond_prob = cond_prob / np.sum(cond_prob)
        rand_values = np.random.rand()
        cond_prob = np.cumsum(cond_prob)  # from pdf to cdf
        helper = np.argwhere(cond_prob > rand_values)
        c_i_sample = np.argwhere(cond_prob > rand_values)[0][0]
        # caution: c_i_sample return the <index> of sitting with
        # if c_i_sample = 0 it will sit with data point 1

        ##########################################################
        # Adjust customer seating, table assignments and LogLiks #
        ##########################################################
        # If sampled customer assignment leaves table assignments intact do
        # nothing, otherwise it means that it joined another table
        # Update table parameters and likelihoods
        tab_not_change = np.in1d(c_i_sample, new_conn_customers)
        if not tab_not_change:
            # Table id for sampled cluster assign
            table_id_sample = Z_C[c_i_sample]

            # Update Table Members
            helper = np.array([table_id_curr[0], table_id_sample[0]])
            table_swap = my_minmax(helper)

            # new way
            tab_curr = table_members[table_id_curr[0] - 1]
            tab_sample = table_members[table_id_sample[0] - 1]
            new_table_members = np.append(tab_curr, tab_sample)

            # swap
            table_members[table_swap[0] - 1] = new_table_members
            table_members.pop(table_swap[1] - 1)

            # REWRITE THESE LINES ------>>
            # Update Table Assignments
            Z_C[Z_C == table_swap[1]] = table_swap[0]
            Z_C[Z_C > table_swap[1]] = Z_C[Z_C > table_swap[1]] - 1

            # Update Table LogLiks
            table_logLiks[table_swap[0] - 1] = new_logLiks[table_ids == Z_C[c_i_sample]]
            table_logLiks[(table_swap[1] - 1):(K - 1)] = table_logLiks[table_swap[1]:K]

            # Reduce Number of Tables
            K = K - 1

        # Update Customer assignment
        C[c_i] = c_i_sample + 1

    table_logLiks = table_logLiks[0:K]
    return C, Z_C, table_members, table_logLiks
