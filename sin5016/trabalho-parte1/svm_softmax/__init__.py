"""
Pacote com componentes para treinar um classificador softmax inspirado em SVM
usando apenas NumPy/CuPy e pipelines de dados customizáveis.
"""

from .backend import get_array_module, to_device
from .model import SoftmaxClassifier, TrainingConfig
from . import kernels
from .data_sources.base import DataSource

__all__ = [
    "get_array_module",
    "to_device",
    "SoftmaxClassifier",
    "TrainingConfig",
    "kernels",
    "DataSource",
]
