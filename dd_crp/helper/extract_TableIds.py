import numpy as np


def extract_TableIds(C):
    C = np.concatenate((np.array([-1]), C))
    N = len(C)
    K = 0
    Z_C = np.zeros((N,1),dtype=int)
    # because the problem caused by difference between matlab and python
    # we add a dummy head in C, thus we should return an array with no 1st element
    # when using this function we should set first element of C as -1
    for i in np.arange(1, N):
        if Z_C[i] == 0:
            K = K + 1
            Piold = Z_C.copy()
            curr = i
            Z_C[curr] = K
            while not all(Z_C == Piold):
                Piold = Z_C.copy()
                curr = C[curr]
                if curr > 0:
                    if Z_C[curr] == 0:
                        Z_C[curr] = K
                    elif Z_C[curr] < K:
                        k = Z_C[curr]
                        Z_C[Z_C == K] = k
                        K = K - 1
                        break

    return Z_C[1:, :]


