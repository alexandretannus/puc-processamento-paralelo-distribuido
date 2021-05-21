""" 
    Processamento Paralelo e Distribuído
    Exercício 01 - Introdução ao CUDA
    
 """

from numba import cuda
import numpy as np
import math

@cuda.jit
def somaMatrizes(X, Y, Z):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    blockSize = cuda.blockDim.x

    i = tx + ty * blockSize    

    if i < X.size:
        Z[i] = X[i] + Y[i]


N = 22
X = np.zeros(N)
Y = np.zeros(N)
Z = np.zeros(N)

for i in range(N):
  X[i] = 1.0
  Y[i] = 2.0

threadsPerBlock = 8
blocksPerGrid = math.ceil((X.size + (threadsPerBlock-1)) / threadsPerBlock)

somaMatrizes[blocksPerGrid, threadsPerBlock](X, Y, Z)

print(Z)
