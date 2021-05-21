""" 
    Processamento Paralelo e Distribuído
    Exercício 03 - Introdução ao CUDA

    Trabalhando com matrizes
    
 """

from numba import cuda
import numpy as np
import math

@cuda.jit
def somaMatrizes(X, Y, Z):
    x, y = cuda.grid(2)
    print(x)
    if x < X.shape[0] and y < X.shape[1]:
            Z[x][y] = X[x][y] + Y[x][y]


N = 30
X = np.zeros((N, N))
Y = np.zeros((N, N))
Z = np.zeros((N, N))

for i in range(N):
  X[i] = 1.0
  Y[i] = 2.0

threadsPerBlock = (16, 16)
blocksPerGrid_x = math.ceil(X.shape[0] / threadsPerBlock[0])
blocksPerGrid_y = math.ceil(X.shape[1] / threadsPerBlock[1])
blocksPerGrid = (blocksPerGrid_x, blocksPerGrid_y)
print(blocksPerGrid)

print(X.shape[0])
print(X.shape[1])

somaMatrizes[blocksPerGrid, threadsPerBlock](X, Y, Z)

print(Z)
