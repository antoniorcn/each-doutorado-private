import numpy as np
from typing import Tuple
from math import cos, sin
PontoVetor = np.ndarray[Tuple[int], np.dtype[np.float64]]

class Ponto2d:

    def __init__(self, pos : PontoVetor) -> None:
        self.pos : PontoVetor = pos

    def __copy__(self, **kvargs) -> Ponto2d:
        # Retorna uma nova instância com o mesmo valor
        return Ponto2d(**kvargs)

    def translacao(self, delta : PontoVetor ) -> Ponto2d:
        return Ponto2d(pos = self.pos + delta)

    def rotacao(self, angulo : float) -> Ponto2d:
        matriz_rotacao = np.array( [
                (cos(angulo), -sin(angulo)),
                (sin(angulo), cos(angulo))
        ])
        return Ponto2d(pos = self.pos * matriz_rotacao)

