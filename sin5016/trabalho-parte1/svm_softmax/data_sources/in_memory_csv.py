"""Implementação simples que carrega um CSV inteiro na memória."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .base import DataBatch, DataSource


class LabelEncoder:
    def __init__(self) -> None:
        self._to_index = {}
        self._from_index: List[str] = []

    def encode(self, raw_value: str) -> int:
        value = raw_value.strip()
        if value not in self._to_index:
            self._to_index[value] = len(self._from_index)
            self._from_index.append(value)
        return self._to_index[value]

    @property
    def num_classes(self) -> int:
        return len(self._from_index)

    @property
    def classes(self) -> List[str]:
        return list(self._from_index)


class InMemoryCSVDataSource(DataSource):
    """
    Lê todas as amostras de um CSV para memória. Permite ignorar colunas
    (ex.: caminho da imagem) e escolher a coluna de rótulo.
    """

    def __init__(
        self,
        csv_path: str | Path,
        *,
        label_column: int,
        feature_slice: Tuple[int, Optional[int]] | None = None,
        feature_columns: Optional[Sequence[int]] = None,
        ignored_columns: Optional[Sequence[int]] = None,
        skip_header: bool = True,
        delimiter: str = ",",
        decimal_separator: str = ".",
        dtype: np.dtype = np.float32,
        rng_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.csv_path = Path(csv_path)
        self.label_column = label_column
        self.feature_slice = feature_slice
        self.feature_columns = list(feature_columns) if feature_columns is not None else None
        self.ignored_columns = set(ignored_columns or [])
        self.skip_header = skip_header
        self.delimiter = delimiter
        self.decimal_separator = decimal_separator
        self.dtype = dtype
        self._rng = np.random.default_rng(rng_seed)

        self._features: np.ndarray
        self._labels: np.ndarray
        self._load()

    def _load(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)

        features: List[List[float]] = []
        labels: List[int] = []
        encoder = LabelEncoder()

        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter=self.delimiter)
            if self.skip_header:
                next(reader, None)

            for row in reader:
                if not row:
                    continue
                label_value = row[self.label_column]
                labels.append(encoder.encode(label_value))
                feature_values = self._extract_feature_values(row)
                features.append(feature_values)

        self._features = np.asarray(features, dtype=self.dtype)
        self._labels = np.asarray(labels, dtype=np.int32)
        self._update_metadata(
            num_samples=self._features.shape[0],
            num_features=self._features.shape[1],
            num_classes=encoder.num_classes,
        )

    def _extract_feature_values(self, row: Sequence[str]) -> List[float]:
        indices = self._resolve_feature_indices(len(row))
        return [self._parse_float(row[idx]) for idx in indices]

    def _parse_float(self, value: str) -> float:
        if self.decimal_separator != ".":
            value = value.replace(self.decimal_separator, ".")
        return float(value)

    def _resolve_feature_indices(self, row_length: int) -> List[int]:
        if self.feature_columns is not None:
            return [idx for idx in self.feature_columns if idx != self.label_column]

        start, end = (0, row_length) if self.feature_slice is None else self.feature_slice
        if end is None:
            end = row_length

        indices: List[int] = []
        for idx in range(start, end):
            if idx == self.label_column or idx in self.ignored_columns:
                continue
            indices.append(idx)
        return indices

    def iter_batches(self, batch_size: int, shuffle: bool = True):
        num_samples = self.num_samples
        if batch_size <= 0:
            raise ValueError("batch_size deve ser > 0")

        order = np.arange(num_samples)
        if shuffle:
            self._rng.shuffle(order)

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_idx = order[start:end]
            yield DataBatch(
                features=self._features[batch_idx],
                labels=self._labels[batch_idx],
            )
