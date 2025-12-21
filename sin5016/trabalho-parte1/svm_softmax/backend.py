"""
Funções utilitárias para selecionar o backend NumPy/CuPy dinamicamente.
"""
from __future__ import annotations

from typing import Any
import warnings

import numpy as _np


def get_array_module(use_gpu: bool = False):
    """
    Retorna o módulo de arrays a ser usado (numpy ou cupy).

    Args:
        use_gpu: Se True, tenta usar CuPy. Se não estiver disponível,
                 volta automaticamente para NumPy emitindo um aviso.
    """
    if use_gpu:
        try:
            import cupy as cp  # type: ignore

            return cp
        except Exception as exc:  # pragma: no cover - apenas fallback
            warnings.warn(
                f"Não foi possível carregar CuPy ({exc}). "
                "Continuando com NumPy.",
                RuntimeWarning,
                stacklevel=2,
            )
    return _np


def is_gpu_backend(xp_module: Any) -> bool:
    """Retorna True se o backend fornecido for o CuPy."""
    return xp_module.__name__ == "cupy"  # type: ignore[attr-defined]


def to_device(array: Any, xp_module: Any):
    """
    Converte um array para o backend informado.

    - Se backend for CuPy e o array estiver no host (NumPy), copia para GPU.
    - Se backend for NumPy e o array estiver na GPU, traz de volta (usa .get()).
    - Caso contrário, apenas retorna o array original.
    """
    if is_gpu_backend(xp_module):
        import numpy as np

        if isinstance(array, np.ndarray):
            return xp_module.asarray(array)
        return array

    # Backend CPU: garante que objetos CuPy sejam convertidos para NumPy
    if hasattr(array, "get"):
        return array.get()
    return _np.asarray(array)
