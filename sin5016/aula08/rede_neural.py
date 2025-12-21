import numpy as np


def relu( valor : np.ndarray ):
    return np.maximum(valor, np.zeros_like(valor))

def sigmoid( valor : np.ndarray):
    return  1 / (1 + np.exp(-valor))


# Definir tamanhos
N = 10
ne = 2 # numero de features
ns = 2 # Numero de classes de saida
h = 2 # Numero de neuronios na camada escondida

X = np.zeros( (N, ne + 1) )
Yd = np.zeros( (N, ns))

# Calcular pesos
A = np.random.randint(-2, 2, ne + 1) * 0.10
B = np.random.randint(-2, 2, ne + 1) * 0.10
print("A: ", A)
print("B: ", B)

# Calcular a saida
print("X Shape => ", X.shape)
print("A.T Shape => ", (A.T).shape)
Zin = X @ A.T
print("Zin => ", Zin)
Z = relu( Zin )
print("Z Shape => ", Z.shape)
print("B.T Shape => ", (B.T).shape)
Yin = Z @ B.T
Y = sigmoid( Yin )
fl = (1-Z) * Z
gl = (1-Y) * Y

print("Y: ", Y)

# Entradas


