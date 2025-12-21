"""Implementação do classificador Softmax com suporte a kernels e paralelismo."""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .backend import get_array_module, is_gpu_backend, to_device
from .kernels import FeatureMap, LinearFeatureMap


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    num_workers: int = 1
    use_gpu: bool = False
    log_every: int = 20
    history_log_path: Optional[str] = None


class SoftmaxClassifier:
    """
    Classificador linear/multiclasse treinado com entropia cruzada e suporte a
    transformações de features (kernels) e paralelismo em mini-batches.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        *,
        feature_map: Optional[FeatureMap] = None,
        dtype: np.dtype = np.float32,
        seed: Optional[int] = None,
        use_gpu: bool = False,
    ) -> None:
        self.dtype = dtype
        self.xp = get_array_module(use_gpu)
        self.using_gpu = is_gpu_backend(self.xp)
        self.rng = np.random.default_rng(seed)

        self.feature_map = feature_map or LinearFeatureMap()
        self.feature_map.ensure_initialized(num_features, self.xp)
        feature_dim = self.feature_map.output_dim or num_features

        weight_scale = 0.01
        weight_init = self.rng.standard_normal((num_classes, feature_dim)).astype(dtype) * weight_scale
        bias_init = np.zeros((num_classes,), dtype=dtype)

        self.W = self.xp.asarray(weight_init, dtype=dtype)
        self.b = self.xp.asarray(bias_init, dtype=dtype)
        self.num_classes = num_classes

    def fit(self, data_source, config: TrainingConfig):
        history: List[Dict[str, float]] = []
        for epoch in range(1, config.epochs + 1):
            stats = self._run_epoch(data_source, config, epoch=epoch)
            stats["epoch"] = epoch
            history.append(stats)
            print(f"[Época {epoch}] loss={stats['loss']:.4f} accuracy={stats['accuracy']:.4f}")

        if config.history_log_path:
            self._write_history(config.history_log_path, history)
        return history

    def fit_with_validation(
        self,
        train_source,
        val_source,
        config: TrainingConfig,
    ) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
        train_history: List[Dict[str, float]] = []
        val_history: List[Dict[str, float]] = []

        for epoch in range(1, config.epochs + 1):
            train_stats = self._run_epoch(train_source, config, epoch=epoch)
            train_stats["epoch"] = epoch
            train_history.append(train_stats)

            val_metrics = self.evaluate(val_source, batch_size=config.batch_size)
            val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
            val_metrics["epoch"] = epoch
            val_history.append(val_metrics)

            print(
                f"[Época {epoch}] train_loss={train_stats['loss']:.4f} train_acc={train_stats['accuracy']:.4f} "
                f"val_loss={val_metrics['val_loss']:.4f} val_acc={val_metrics['val_accuracy']:.4f}"
            )

        combined_history = []
        for train_stats, val_stats in zip(train_history, val_history):
            combined_history.append({**train_stats, **val_stats})

        if config.history_log_path:
            self._write_history(config.history_log_path, combined_history)

        return train_history, val_history

    def _run_epoch(self, data_source, config: TrainingConfig, epoch: Optional[int] = None) -> Dict[str, float]:
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        step = 0

        for batch in data_source.iter_batches(config.batch_size, shuffle=True):
            X_batch, y_batch = self._prepare_batch(batch)
            loss, grad_W, grad_b = self._compute_gradients(
                X_batch,
                y_batch,
                weight_decay=config.weight_decay,
                num_workers=config.num_workers,
            )
            self._apply_gradients(grad_W, grad_b, config.learning_rate)

            batch_size = int(y_batch.shape[0])
            epoch_loss += loss * batch_size
            epoch_samples += batch_size
            epoch_correct += self._batch_correct_predictions(X_batch, y_batch)

            step += 1
            if config.log_every and step % config.log_every == 0:
                avg_loss = epoch_loss / max(epoch_samples, 1)
                acc = epoch_correct / max(epoch_samples, 1)
                epoch_label = f"{epoch}/{config.epochs}" if epoch is not None else "?"
                print(
                    f"[Época {epoch_label}] Step {step} - "
                    f"loss {avg_loss:.4f} - acc {acc:.4f}"
                )

        avg_loss = epoch_loss / max(epoch_samples, 1)
        acc = epoch_correct / max(epoch_samples, 1)
        return {"loss": avg_loss, "accuracy": acc}

    def _write_history(self, path: str, history: List[Dict[str, float]]) -> None:
        import json

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)

    def predict(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        preds: List[np.ndarray] = []
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            chunk = self.xp.asarray(X[start:end], dtype=self.dtype)
            chunk = self.feature_map.transform(chunk, self.xp)
            scores = self._scores(chunk)
            pred = self.xp.argmax(scores, axis=1)
            preds.append(to_device(pred, self.xp))
        return np.concatenate(preds, axis=0)

    def evaluate(self, data_source, batch_size: int = 1024) -> Dict[str, float]:
        total_correct = 0
        total = 0
        total_loss = 0.0
        for batch in data_source.iter_batches(batch_size, shuffle=False):
            X_batch, y_batch = self._prepare_batch(batch, requires_grad=False)
            scores = self._scores(X_batch)
            probs = self._softmax(scores)
            preds = self.xp.argmax(probs, axis=1)
            correct_batch = to_device(self.xp.sum(preds == y_batch), self.xp)
            total_correct += int(correct_batch)
            batch_size = int(y_batch.shape[0])
            total += batch_size
            loss_value = to_device(self._cross_entropy_loss(probs, y_batch), self.xp)
            total_loss += float(loss_value) * batch_size
        return {
            "loss": total_loss / max(total, 1),
            "accuracy": total_correct / max(total, 1),
        }

    def save(self, path: str) -> None:
        """
        Salva os pesos no formato NPZ (sempre em CPU).
        """
        np.savez(
            path,
            W=to_device(self.W, self.xp),
            b=to_device(self.b, self.xp),
            num_classes=self.num_classes,
        )

    @classmethod
    def load(
        cls,
        path: str,
        *,
        feature_map: Optional[FeatureMap] = None,
        use_gpu: bool = False,
    ) -> "SoftmaxClassifier":
        data = np.load(path)
        W = data["W"]
        b = data["b"]
        num_classes = int(data["num_classes"])
        model = cls(
            num_features=W.shape[1],
            num_classes=num_classes,
            feature_map=feature_map,
            use_gpu=use_gpu,
        )
        model.W = model.xp.asarray(W, dtype=model.dtype)
        model.b = model.xp.asarray(b, dtype=model.dtype)
        return model

    def _prepare_batch(self, batch, requires_grad: bool = True):
        X = self.xp.asarray(batch.features, dtype=self.dtype)
        X = self.feature_map.transform(X, self.xp)
        label_dtype = self.xp.int32 if self.using_gpu else np.int32
        y = self.xp.asarray(batch.labels, dtype=label_dtype)
        if not requires_grad and not self.using_gpu:
            y = batch.labels
        return X, y

    def _compute_gradients(self, X, y, weight_decay: float, num_workers: int):
        if self.using_gpu or num_workers <= 1:
            return self._loss_and_gradients_single(X, y, weight_decay)
        return self._loss_and_gradients_parallel(X, y, weight_decay, num_workers)

    def _loss_and_gradients_single(self, X, y, weight_decay: float):
        xp = self.xp
        batch_size = X.shape[0]
        scores = self._scores(X)
        probs = self._softmax(scores)
        log_likelihood = -xp.log(probs[xp.arange(batch_size), y] + 1e-12)
        loss = xp.mean(log_likelihood) + 0.5 * weight_decay * float(xp.sum(self.W * self.W))

        dscores = probs
        dscores[xp.arange(batch_size), y] -= 1
        dscores /= batch_size

        grad_W = dscores.T @ X + weight_decay * self.W
        grad_b = xp.sum(dscores, axis=0)
        return float(loss), grad_W, grad_b

    def _loss_and_gradients_parallel(self, X, y, weight_decay: float, num_workers: int):
        if not isinstance(X, np.ndarray):
            X_cpu = to_device(X, self.xp)
        else:
            X_cpu = X
        if not isinstance(y, np.ndarray):
            y_cpu = to_device(y, self.xp)
        else:
            y_cpu = y

        indices = np.array_split(np.arange(X_cpu.shape[0]), num_workers)
        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)
        loss_sum = 0.0
        total = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _chunk_softmax_gradients,
                    self.W,
                    self.b,
                    X_cpu[idx],
                    y_cpu[idx],
                )
                for idx in indices
                if idx.size > 0
            ]
            for future in as_completed(futures):
                chunk_loss, chunk_grad_W, chunk_grad_b, chunk_size = future.result()
                loss_sum += chunk_loss
                grad_W += chunk_grad_W
                grad_b += chunk_grad_b
                total += chunk_size

        if total == 0:
            raise RuntimeError("Nenhum dado encontrado para calcular gradientes.")

        grad_W /= total
        grad_b /= total
        loss = loss_sum / total
        loss += 0.5 * weight_decay * float(np.sum(self.W * self.W))
        grad_W += weight_decay * self.W
        return float(loss), grad_W, grad_b

    def _apply_gradients(self, grad_W, grad_b, learning_rate: float):
        self.W = self.W - learning_rate * grad_W
        self.b = self.b - learning_rate * grad_b

    def _scores(self, X):
        return X @ self.W.T + self.b

    def _softmax(self, scores):
        xp = self.xp
        scores = scores - xp.max(scores, axis=1, keepdims=True)
        exp_scores = xp.exp(scores)
        return exp_scores / xp.sum(exp_scores, axis=1, keepdims=True)

    def _batch_correct_predictions(self, X, y) -> int:
        probs = self._softmax(self._scores(X))
        preds = self.xp.argmax(probs, axis=1)
        correct = self.xp.sum(preds == y)
        return int(to_device(correct, self.xp))

    def _cross_entropy_loss(self, probs, y):
        xp = self.xp
        log_likelihood = -xp.log(probs[xp.arange(probs.shape[0]), y] + 1e-12)
        return xp.mean(log_likelihood)


def _chunk_softmax_gradients(W, b, X_chunk, y_chunk):
    if X_chunk.size == 0:
        return 0.0, np.zeros_like(W), np.zeros_like(b), 0

    logits = X_chunk @ W.T + b
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    log_likelihood = -np.log(probs[np.arange(len(y_chunk)), y_chunk] + 1e-12)
    dscores = probs
    dscores[np.arange(len(y_chunk)), y_chunk] -= 1

    grad_W = dscores.T @ X_chunk
    grad_b = np.sum(dscores, axis=0)

    return float(np.sum(log_likelihood)), grad_W, grad_b, len(y_chunk)
