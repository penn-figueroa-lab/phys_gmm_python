import numpy as np

# This function returns customers that are connected to i, directly or indirectly
# does the job of the sitbehind(i) function from the Socher11 paper.
def get_Connections(C, i):
    nn = 0
    customers = i
    n = 1
    indexs = np.arange(0, len(C))
    C1 = C.copy() - 1
    while n > nn:
        nn = n
        back = indexs[np.in1d(C1, customers)]  # find all cus that relate to current
        customers = np.sort(np.append(np.append(back, C1[customers]), i))  # merge
        customers = np.unique(customers) # deduplicate
        n = len(customers)

    return customers




