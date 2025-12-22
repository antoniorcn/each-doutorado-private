"""Leitor de CSV que mantém apenas buffers menores em memória."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .base import DataBatch, DataSource


class StreamingCSVDataSource(DataSource):
    """
    DataSource que percorre o CSV sob demanda e mantém apenas um buffer
    de mini-batches em memória.
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
        dtype: np.dtype = np.float32,
        decimal_separator: str = ".",
        buffer_size: int = 8192,
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
        self.buffer_size = buffer_size
        self._rng = np.random.default_rng(rng_seed)

        self._label_to_index: dict[str, int] = {}
        self._feature_indices: Optional[List[int]] = None
        self._scan_metadata()

    def _scan_metadata(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)

        num_samples = 0
        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter=self.delimiter)
            if self.skip_header:
                next(reader, None)

            for row in reader:
                if not row:
                    continue
                if self._feature_indices is None:
                    self._feature_indices = self._resolve_feature_indices(len(row))
                self._encode_label(row[self.label_column])
                num_samples += 1

        if self._feature_indices is None:
            raise ValueError("CSV não contém linhas de dados válidas.")

        self._update_metadata(
            num_samples=num_samples,
            num_features=len(self._feature_indices),
            num_classes=len(self._label_to_index),
        )

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

    def _encode_label(self, label_value: str) -> int:
        value = label_value.strip()
        if value not in self._label_to_index:
            self._label_to_index[value] = len(self._label_to_index)
        return self._label_to_index[value]

    def _decode_row(self, row: Sequence[str]) -> Tuple[np.ndarray, int]:
        if self._feature_indices is None:
            raise RuntimeError("Feature indices não definidos.")
        label = self._encode_label(row[self.label_column])
        features = np.asarray(
            [self._parse_float(row[idx]) for idx in self._feature_indices],
            dtype=self.dtype,
        )
        return features, label

    def _parse_float(self, value: str) -> float:
        if self.decimal_separator != ".":
            value = value.replace(self.decimal_separator, ".")
        return float(value)
    def iter_batches(self, batch_size: int, shuffle: bool = True):
        if batch_size <= 0:
            raise ValueError("batch_size deve ser > 0")

        buffer_features: List[np.ndarray] = []
        buffer_labels: List[int] = []

        def flush_buffer():
            if not buffer_features:
                return
            feats = np.stack(buffer_features, axis=0)
            labs = np.asarray(buffer_labels, dtype=np.int32)
            order = np.arange(len(labs))
            if shuffle:
                self._rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                end = min(start + batch_size, len(order))
                idx = order[start:end]
                yield DataBatch(features=feats[idx], labels=labs[idx])

        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter=self.delimiter)
            if self.skip_header:
                next(reader, None)

            for row in reader:
                if not row:
                    continue
                features, label = self._decode_row(row)
                buffer_features.append(features)
                buffer_labels.append(label)

                if len(buffer_features) >= self.buffer_size:
                    yield from flush_buffer()
                    buffer_features.clear()
                    buffer_labels.clear()

        # Último buffer
        yield from flush_buffer()
