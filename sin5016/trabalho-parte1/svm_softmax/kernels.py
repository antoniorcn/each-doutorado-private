"""
Transformações de features que permitem trocar o "kernel" utilizado.

Para kernels lineares basta usar :class:`LinearFeatureMap`, enquanto kernels
não lineares podem ser aproximados via features explícitas (ex.: Fourier
aleatório para RBF) sem depender de bibliotecas externas de ML.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class FeatureMap(ABC):
    """
    Interface para transformações de features.

    Implementações devem gerar representações no espaço desejado (linear ou
    aproximado) sem expor detalhes ao classificador.
    """

    def __init__(self) -> None:
        self._initialized: bool = False
        self._output_dim: Optional[int] = None

    def ensure_initialized(self, input_dim: int, xp_module) -> None:
        if not self._initialized:
            self._initialize(input_dim, xp_module)
            self._initialized = True

    @abstractmethod
    def _initialize(self, input_dim: int, xp_module) -> None:
        """Inicializa parâmetros internos da transformação."""

    @abstractmethod
    def transform(self, X, xp_module):
        """Aplica a transformação ao batch X."""

    @property
    def output_dim(self) -> Optional[int]:
        return self._output_dim


class LinearFeatureMap(FeatureMap):
    """
    Transformação identidade: mantém o kernel linear tradicional.
    """

    def _initialize(self, input_dim: int, xp_module) -> None:
        self._output_dim = input_dim

    def transform(self, X, xp_module):
        return X


class RandomFourierFeatureMap(FeatureMap):
    """
    Aproxima o kernel RBF usando Random Fourier Features.

    Args:
        gamma: Parâmetro do kernel RBF (1 / (2 * sigma^2)).
        n_components: Número de componentes na projeção.
        random_state: Semente opcional para reprodutibilidade.
    """

    def __init__(self, gamma: float = 0.01, n_components: int = 512, random_state: Optional[int] = None) -> None:
        super().__init__()
        if n_components <= 0:
            raise ValueError("n_components deve ser > 0")
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state
        self._omega = None
        self._bias = None

    def _initialize(self, input_dim: int, xp_module) -> None:
        rng = np.random.default_rng(self.random_state)
        omega_np = rng.normal(
            loc=0.0,
            scale=math.sqrt(2 * self.gamma),
            size=(input_dim, self.n_components),
        )
        bias_np = rng.uniform(0.0, 2 * math.pi, size=(self.n_components,))

        # Copia para o backend desejado
        self._omega = xp_module.asarray(omega_np)
        self._bias = xp_module.asarray(bias_np)
        self._output_dim = self.n_components

    def transform(self, X, xp_module):
        if self._omega is None or self._bias is None:
            raise RuntimeError("RandomFourierFeatureMap não inicializado. Chame ensure_initialized primeiro.")
        projections = X @ self._omega  # (batch, n_components)
        projections += self._bias
        return math.sqrt(2.0 / self.n_components) * xp_module.cos(projections)
