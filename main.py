import numpy
import numpy as np
from dd_crp.helper.extract_TableIds import extract_TableIds
from dd_crp.helper.get_Connections import get_Connections
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from itertools import cycle

"""C = np.array([1, 2, 3, 4])  # could not handle when sit by itself
D = [1, 2, 3]
D = np.array(([1, 2, 3], [1, 2, 3]))
Z_C = extract_TableIds(C)
print(Z_C)
K = np.argwhere(Z_C == 2)
K1 = K[:, 0]
print(K1)
D = D[:, K1]
print(D)
V1 = np.array([[1, 3], [-5, -9]])
V2 = np.array([[1], [2]])
print(V1 - V2)
Y_bar = np.mean(V1, axis=1, keepdims=True)
print(Y_bar)
print(Y_bar * Y_bar.T)"""
# C1 = np.zeros((2,2,3))
"""A = np.array([1, 5, 3, 4])
print(all(sorted(A) == A))
colors = []
cycol = cycle('bgrcmky')
for i in np.arange(0, 4):
    colors.append(next(cycol))"""
p1 = np.array([[1, 2, 3],[1, 2, 3]])
print(np.repeat(p1[0, :].reshape(1, len(p1[0])), 2, axis=0))
print(np.repeat(np.sum(p1, axis=0, keepdims=True), 5, axis=0))
p1 = []
p1.append(2)
p1.append(3)
print(np.array(p1))

y = np.array([1, 2, 4, 3])
indx = y.argsort()
print(indx)
y.sort()
print(y)

A = np.array([[1, 3, 2],
              [4, 2, 5],
              [6, 1, 4]])
A = np.delete(A, 1, 1)
print(A)
#print(np.sum(A,axis = 1,keepdims=True))
# print(np.argmax(A[0]))

print(np.linspace(0, 5, 10, endpoint=False))
C = np.array([[1],[1]]) + 2
C1 = np.array([[2],[3]])
print(C1 + C)
C = np.array([[5, 1, 1, 1, 4, 4],[5, 1, 1, 1, 4, 4]])
C_bar = np.array([[1], [2]])
C_mean = np.mean(C, axis=1, keepdims=True)
print(C_mean - C_bar)
print((C == 1) + 0)
print(np.unique(C))
print(np.sort(C))
C2 = np.argwhere(C == 1)
print(C2.reshape(len(C2)))
np.delete()
"""
customers = 0
customers = get_Connections(C, customers)
print(customers)
print(np.unique(C))
sb = [1, 2, 3, 4]
test = np.array([1, 2, 3])
sb1 = sb[1:3]
V1 = np.array([[1, 3], [-5, -9]])
V2 = np.array([[1, 3], [-5, -9]])
print(np.append(V1,V2,axis=1))
"""
"""det = np.linalg.det(V1)
print(det)
logDet = np.linalg.slogdet(V1)
print(logDet)
print(-np.inf > 1)"""
"""print(Z_C)
K = np.argwhere(Z_C == 1)
K1 = K[:, 0]
print(K1)"""
"""Mylist = [None] * 10
Mylist[1] = 250
print(Mylist)"""
