""" 
    Processamento Paralelo e Distribuído
    Exercício 02 - Introdução ao CUDA

    Utilização de funções de transferência de dados de/para memória
    
 """

from numba import cuda
import numpy as np

@cuda.jit
def somaMatrizes(X, Y, Z):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    blockSize = cuda.blockDim.x

    i = tx + ty * blockSize    

    if i < X.size:
        Z[i] = X[i] + Y[i]


N = 20
X = np.zeros(N)
Y = np.zeros(N)
# Z = np.zeros(N)

for i in range(N):
  X[i] = 1.0
  Y[i] = 2.0

x_device = cuda.to_device(X)
y_device = cuda.to_device(Y)
z_device = cuda.device_array_like(X)

threadsPerBlock = 8
blocksPerGrid = (X.size + (threadsPerBlock-1))//threadsPerBlock

somaMatrizes[blocksPerGrid, threadsPerBlock](x_device, y_device, z_device)

Z = z_device.copy_to_host()


print(Z)