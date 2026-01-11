"""
MLP simples (ReLU + Softmax) para classificação multiclasse.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from svm_softmax.data_sources import InMemoryCSVDataSource

class MLPClassifier:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_classes: int,
        *,
        learning_rate: float,
        weight_decay: float,
        seed: int,
    ) -> None:
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.rng = np.random.default_rng(seed)
        dims = [input_dim] + hidden_dims + [num_classes]

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            limit = math.sqrt(2.0 / in_dim)
            W = self.rng.normal(0.0, limit, size=(in_dim, out_dim)).astype(np.float32)
            b = np.zeros(out_dim, dtype=np.float32)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X: np.ndarray):
        activations = X
        cache = []
        for idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = activations @ W + b
            cache.append((activations, Z, idx))
            if idx < len(self.weights) - 1:
                activations = np.maximum(0.0, Z)
            else:
                activations = Z # logits
        return activations, cache

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits, dtype=np.float32)
        return exp / exp.sum(axis=1, keepdims=True)

    def loss_and_grads(self, X: np.ndarray, y: np.ndarray):
        batch_size = X.shape[0]
        logits, cache = self.forward(X)
        probs = self.softmax(logits)
        loss = -np.log(probs[np.arange(batch_size), y] + 1e-12).mean()
        loss += 0.5 * self.weight_decay * sum(float(np.sum(W * W)) for W in self.weights)

        grads_W = [np.zeros_like(W) for W in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        dlogits = probs
        dlogits[np.arange(batch_size), y] -= 1.0
        dlogits /= batch_size

        upstream = dlogits
        for idx in reversed(range(len(self.weights))):
            A_prev, _, _ = cache[idx]
            grads_W[idx] = A_prev.T @ upstream + self.weight_decay * self.weights[idx]
            grads_b[idx] = upstream.sum(axis=0)

            if idx > 0:
                dA_prev = upstream @ self.weights[idx].T
                relu_grad = (cache[idx - 1][1] > 0).astype(np.float32)
                upstream = dA_prev * relu_grad
        return loss, grads_W, grads_b

    def step(self, X: np.ndarray, y: np.ndarray) -> float:
        loss, grads_W, grads_b = self.loss_and_grads(X, y)
        for idx in range(len(self.weights)):
            self.weights[idx] -= self.learning_rate * grads_W[idx]
            self.biases[idx] -= self.learning_rate * grads_b[idx]
        return loss

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        logits, _ = self.forward(X)
        probs = self.softmax(logits)
        preds = np.argmax(probs, axis=1)
        accuracy = float(np.mean(preds == y))
        loss = -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()
        loss += 0.5 * self.weight_decay * sum(float(np.sum(W * W)) for W in self.weights)
        return loss, accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina uma MLP (ReLU + Softmax) para classificação multiclasse.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", required=True, help="Arquivo CSV com features e rótulos.")
    parser.add_argument("--label-column", type=int, required=True,
                        help="Índice da coluna de rótulo.")
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
    parser.add_argument("--no-header", action="store_true",
                        help="Indica que o CSV não possui cabeçalho.")
    parser.add_argument("--delimiter", default=";",
                        help="Delimitador do CSV.")
    parser.add_argument("--decimal-sep", default=",",
                        help="Separador decimal ao ler o CSV.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Decaimento do peso")
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 128],
        help="Lista com o número de neurônios de cada camada escondida.",
    )

    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--stratify-split", action="store_true",
                        help="Mantém proporção dos rótulos nos splits.")
    parser.add_argument("--history-log", type=str,
                        help="Arquivo JSON para salvar histórico de treinamento.")
    args = parser.parse_args()
    total = args.train_size + args.val_size + args.test_size
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        parser.error("train-size + val-size + test-size deve ser igual a 1.0")
    if len(args.delimiter) != 1:
        parser.error("--delimiter deve ter exatamente um caractere.")
    if len(args.decimal_sep) != 1:
        parser.error("--decimal-sep deve ter exatamente um caractere.")
    if not args.hidden_dims:
        parser.error("--hidden-dims precisa ter pelo menos um valor.")
    return args


def load_dataset(args: argparse.Namespace):
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
    return features.astype(np.float32), labels.astype(np.int32), data_source


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


def train() -> None:
    args = parse_args()
    X, y, data_source = load_dataset(args)
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(args, X, y)

    print(
        f"Dataset completo: {data_source.num_samples} amostras | ",
        f"{data_source.num_features} features | {data_source.num_classes} classes"
    )
    print(f"Split -> treino: {X_train.shape[0]} | validação: {X_val.shape[0]} | ",
          f"teste: {X_test.shape[0]}")

    start_time = time.perf_counter()

    model = MLPClassifier(
        input_dim=data_source.num_features,
        hidden_dims=args.hidden_dims,
        num_classes=data_source.num_classes,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.split_seed,
    )

    history: list[dict[str, float]] = []
    rng = np.random.default_rng(args.split_seed)
    num_train = X_train.shape[0]

    for epoch in range(1, args.epochs + 1):
        indices = rng.permutation(num_train)
        for start in range(0, num_train, args.batch_size):
            end = min(start + args.batch_size, num_train)
            batch_idx = indices[start:end]
            model.step(X_train[batch_idx], y_train[batch_idx])

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
            f"[Época {epoch:03d}] loss={train_loss:.4f} acc={train_acc:.4f} | ",
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"[Teste] loss={test_loss:.4f} accuracy={test_acc:.4f}")

    elapsed = time.perf_counter() - start_time
    print(f"Treinamento concluído em {elapsed:.2f} segundos")

    if args.history_log:
        print(f"Salvando o arquivo de log: {args.history_log}")
        Path(args.history_log).parent.mkdir(parents=True, exist_ok=True)
        with open(args.history_log, "w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)
        print(f"Histórico salvo em {args.history_log}")


if __name__ == "__main__":
    train()
