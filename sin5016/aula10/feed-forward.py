import numpy as np
from random import random

def normalizacao_gradiente(grad, eps=1e-8):
    norm = np.linalg.norm(grad)
    if norm < eps:
        return grad  # evita dividir por quase zero
    return grad / norm

def calcula_gradiente( X, Yd, A, B):
    N = X.shape[0]
    X_Bias = np.column_stack((X, np.ones((N, 1))))
    Zin = X_Bias * A
    Z = 1. / (1 + np.exp(-Zin)) # sigmoid
    Z_Bias = np.column_stack((Z, np.ones((N, 1))))
    Yin = Z_Bias * B
    Y = 1. / (1 + np.exp(-Yin))
    erro = Y-Yd
    g1 = (1 - Y) * Y
    f1 = (1 - Z) * Z
    dJdB = 1 / N * (erro * g1).T * np.column_stack((np.ones((N, 1)), Z))
    dJdZ = (erro * g1) * B[:, 2:]
    dJdA = 1 / N * (dJdZ * f1).T * np.column_stack((np.ones((N, 1)), X))
    return dJdA, dJdB




def rna( X, Yd, h, num_epocas_max):
    N, ne = X.shape
    ns = Yd.shape[1]
    A = np.random.rand(h, ne + 1) / 5
    B = np.random.rand(ns, h + 1) / 5

    X = np.array([np.ones((N, 1)), X]) # Criar matriz np.ones x X
    Yr = calc_saida( X, A, B)
    erro = Yr - Yd
    EQM = 1 / N * np.sum(erro * erro)

    nepocas = 0
    dJdA, dJdB = calcula_gradiente(X, Yd, A, B)
    alfa = 0.1
    grad = np.concat([dJdA.flatten(), dJdB.flatten()])
    vetEQM = [ EQM ]

    while nepocas < num_epocas_max and normalizacao_gradiente(grad) > 1e-4:
        
        A = A - alfa * dJdA
        B = B - alfa * dJdB
        Yr = calc_saida(X, A, B)
        erro = Yr - Yd
        EQM = 1 / N * np.sum(erro * erro)
        dJdA, dJdB = calcula_gradiente(X, Yd, A, B)
        grad = np.concat([dJdA.flatten(), dJdB.flatten()])
        vetEQM.append( EQM )



def calc_saida( X, A, B):
    X_Bias = np.column_stack((X, np.ones((X.shape[0], 1))))
    Zin = X * A
    Z = 1. / (1 + np.exp(-Zin)) # sigmoid
    Yin = Z * B
    Y_hat = 1. / (1 + np.exp(-Yin))
    return Y_hat

