#!/usr/bin/env python3
"""
Treina um classificador linear multiclasses com hinge loss (Crammer & Singer).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from svm_softmax.data_sources import InMemoryCSVDataSource


class LinearMulticlassSVM:
    def __init__(self, num_features: int, num_classes: int, *, learning_rate: float, reg: float, seed: int) -> None:
        self.lr = learning_rate
        self.reg = reg
        rng = np.random.default_rng(seed)
        self.W = rng.normal(scale=0.01, size=(num_classes, num_features)).astype(np.float32)
        self.b = np.zeros(num_classes, dtype=np.float32)

    def _scores(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W.T + self.b

    def _loss_and_grads(self, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        num_samples = X.shape[0]
        scores = self._scores(X)
        correct_scores = scores[np.arange(num_samples), y][:, None]
        margins = scores - correct_scores + 1.0
        margins[np.arange(num_samples), y] = 0.0

        positive = margins > 0
        loss = np.sum(margins[positive]) / num_samples
        loss += 0.5 * self.reg * float(np.sum(self.W * self.W))

        grad_scores = positive.astype(np.float32)
        row_sum = np.sum(positive, axis=1, dtype=np.float32)
        grad_scores[np.arange(num_samples), y] -= row_sum
        grad_scores /= num_samples

        grad_W = grad_scores.T @ X + self.reg * self.W
        grad_b = np.sum(grad_scores, axis=0)
        return loss, grad_W, grad_b

    def step(self, X: np.ndarray, y: np.ndarray) -> float:
        loss, grad_W, grad_b = self._loss_and_grads(X, y)
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b
        return loss

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        scores = self._scores(X)
        preds = np.argmax(scores, axis=1)
        accuracy = float(np.mean(preds == y))

        num_samples = X.shape[0]
        correct_scores = scores[np.arange(num_samples), y][:, None]
        margins = scores - correct_scores + 1.0
        margins[np.arange(num_samples), y] = 0.0
        margins = np.maximum(margins, 0.0)
        loss = np.sum(margins) / num_samples + 0.5 * self.reg * float(np.sum(self.W * self.W))
        return loss, accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina um SVM linear multiclasses com hinge loss.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", required=True, help="Arquivo CSV com features e rótulos.")
    parser.add_argument("--label-column", type=int, required=True, help="Índice da coluna de rótulo.")
    parser.add_argument(
        "--feature-slice",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Faixa de colunas usada como features. END exclusivo.",
    )
    parser.add_argument(
        "--feature-columns",
        type=int,
        nargs="+",
        help="Lista explícita de colunas para usar como features (substitui feature-slice).",
    )
    parser.add_argument(
        "--ignored-columns",
        type=int,
        nargs="+",
        default=[0],
        help="Colunas descartadas (ex.: caminho da imagem).",
    )
    parser.add_argument("--no-header", action="store_true", help="Indica que o CSV não possui cabeçalho.")
    parser.add_argument("--delimiter", default=";", help="Delimitador do CSV.")
    parser.add_argument("--decimal-sep", default=",", help="Separador decimal ao ler o CSV.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--reg", type=float, default=1e-4, help="Força da regularização L2.")
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--stratify-split", action="store_true", help="Mantém proporção dos rótulos nos splits.")
    parser.add_argument("--history-log", type=str, help="Arquivo JSON para salvar histórico de treinamento.")
    args = parser.parse_args()

    total = args.train_size + args.val_size + args.test_size
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        parser.error("train-size + val-size + test-size deve ser igual a 1.0")
    if len(args.delimiter) != 1:
        parser.error("--delimiter deve ter exatamente um caractere.")
    if len(args.decimal_sep) != 1:
        parser.error("--decimal-sep deve ter exatamente um caractere.")
    return args


def load_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, InMemoryCSVDataSource]:
    data_source = InMemoryCSVDataSource(
        csv_path=args.csv,
        label_column=args.label_column,
        feature_slice=tuple(args.feature_slice) if args.feature_slice else None,
        feature_columns=args.feature_columns,
        ignored_columns=args.ignored_columns,
        skip_header=not args.no_header,
        delimiter=args.delimiter,
        decimal_separator=args.decimal_sep,
    )
    features, labels = data_source.as_arrays()
    features = features.astype(np.float32)
    labels = labels.astype(np.int32)
    return features, labels, data_source


def split_dataset(args, X, y):
    stratify = y if args.stratify_split else None
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.split_seed,
        stratify=stratify,
    )
    val_ratio = args.val_size / (args.train_size + args.val_size)
    stratify_trainval = y_trainval if args.stratify_split else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_ratio,
        random_state=args.split_seed,
        stratify=stratify_trainval,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def train():
    args = parse_args()
    X, y, data_source = load_dataset(args)
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(args, X, y)

    print(
        f"Dataset completo: {data_source.num_samples} amostras | "
        f"{data_source.num_features} features | {data_source.num_classes} classes"
    )
    print(
        f"Split -> treino: {X_train.shape[0]} | validação: {X_val.shape[0]} | teste: {X_test.shape[0]}"
    )

    model = LinearMulticlassSVM(
        num_features=data_source.num_features,
        num_classes=data_source.num_classes,
        learning_rate=args.learning_rate,
        reg=args.reg,
        seed=args.split_seed,
    )

    history: list[dict[str, float]] = []
    rng = np.random.default_rng(args.split_seed)
    num_train = X_train.shape[0]

    for epoch in range(1, args.epochs + 1):
        indices = rng.permutation(num_train)
        epoch_loss = 0.0
        steps = 0
        for start in range(0, num_train, args.batch_size):
            end = min(start + args.batch_size, num_train)
            batch_idx = indices[start:end]
            batch_loss = model.step(X_train[batch_idx], y_train[batch_idx])
            epoch_loss += batch_loss
            steps += 1

        train_loss, train_acc = model.evaluate(X_train, y_train)
        val_loss, val_acc = model.evaluate(X_val, y_val)
        history.append(
            {
                "epoch": epoch,
                "loss": train_loss,
                "accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )
        print(
            f"[Época {epoch:03d}] loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"[Teste] loss={test_loss:.4f} accuracy={test_acc:.4f}")

    if args.history_log:
        Path(args.history_log).parent.mkdir(parents=True, exist_ok=True)
        with open(args.history_log, "w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)
        print(f"Histórico salvo em {args.history_log}")


if __name__ == "__main__":
    train()
