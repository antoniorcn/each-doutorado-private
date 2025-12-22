"""
Interfaces para provedores de dados usados no treinamento em mini-batches.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Tuple

import numpy as np


@dataclass(frozen=True)
class DataBatch:
    features: np.ndarray
    labels: np.ndarray


class DataSource(ABC):
    """
    Fonte de dados abstrata: implementações podem manter tudo em memória
    ou fazer leitura incremental do disco.
    """

    def __init__(self) -> None:
        self._num_samples: int = 0
        self._num_features: int = 0
        self._num_classes: int = 0

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def num_features(self) -> int:
        return self._num_features

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @abstractmethod
    def iter_batches(self, batch_size: int, shuffle: bool = True) -> Iterator[DataBatch]:
        """
        Itera sobre o dataset produzindo lotes.
        """

    def as_epoch_iterator(self, batch_size: int, shuffle: bool = True) -> Iterable[DataBatch]:
        return self.iter_batches(batch_size=batch_size, shuffle=shuffle)

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Consolida todas as amostras em arrays NumPy."""
        if self.num_samples == 0:
            raise ValueError("DataSource vazio, nada para converter em arrays.")

        features: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        batch_size = max(self.num_samples, 1)
        for batch in self.iter_batches(batch_size=batch_size, shuffle=False):
            features.append(batch.features)
            labels.append(batch.labels)

        return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

    def _update_metadata(self, num_samples: int, num_features: int, num_classes: int) -> None:
        self._num_samples = num_samples
        self._num_features = num_features
        self._num_classes = num_classes


class ArrayDataSource(DataSource):
    """Implementação simples baseada em arrays já carregados."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        rng_seed: int | None = None,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()
        if features.shape[0] != labels.shape[0]:
            raise ValueError("features e labels devem ter o mesmo número de amostras.")

        self._features = np.asarray(features)
        self._labels = np.asarray(labels, dtype=np.int32)
        if num_classes is None:
            num_classes = 0 if self._labels.size == 0 else int(self._labels.max()) + 1
        self._rng = np.random.default_rng(rng_seed)
        self._update_metadata(
            num_samples=self._features.shape[0],
            num_features=self._features.shape[1] if self._features.ndim == 2 else 0,
            num_classes=num_classes,
        )

    def iter_batches(self, batch_size: int, shuffle: bool = True) -> Iterator[DataBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size deve ser > 0")

        num_samples = self.num_samples
        if num_samples == 0:
            return

        order = np.arange(num_samples)
        if shuffle:
            self._rng.shuffle(order)

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            idx = order[start:end]
            yield DataBatch(features=self._features[idx], labels=self._labels[idx])
