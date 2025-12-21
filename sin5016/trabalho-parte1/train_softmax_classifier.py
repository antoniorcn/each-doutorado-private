#!/usr/bin/env python3
"""
CLI para treinar o classificador softmax multiclasses a partir de um CSV de features.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from sklearn.model_selection import train_test_split

from svm_softmax.data_sources import ArrayDataSource, InMemoryCSVDataSource, StreamingCSVDataSource
from svm_softmax.kernels import LinearFeatureMap, RandomFourierFeatureMap
from svm_softmax.model import SoftmaxClassifier, TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treina um classificador softmax multi-classe usando features HOG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", required=True, help="Arquivo CSV com features e rótulos.")
    parser.add_argument("--label-column", type=int, required=True, help="Índice da coluna de rótulo.")
    parser.add_argument(
        "--data-source",
        choices=["in-memory", "streaming"],
        default="in-memory",
        help="Seleciona a camada de dados.",
    )
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
        help="Lista explícita de colunas para usar como features.",
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
    parser.add_argument("--buffer-size", type=int, default=8192, help="Tamanho do buffer na fonte streaming.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=1, help="Threads para gradientes paralelos (CPU).")
    parser.add_argument("--use-gpu", action="store_true", help="Tenta usar CuPy.")
    parser.add_argument(
        "--kernel",
        choices=["linear", "rbf"],
        default="linear",
        help="Transformação de features aplicada antes do classificador.",
    )
    parser.add_argument("--rbf-components", type=int, default=512, help="Componentes RFF para kernel RBF.")
    parser.add_argument("--rbf-gamma", type=float, default=0.01, help="Parâmetro gamma do kernel RBF.")
    parser.add_argument("--save-model", type=str, help="Caminho para salvar pesos (npz).")
    parser.add_argument("--train-size", type=float, default=0.7, help="Percentual destinado ao treinamento.")
    parser.add_argument("--val-size", type=float, default=0.15, help="Percentual destinado à validação.")
    parser.add_argument("--test-size", type=float, default=0.15, help="Percentual destinado ao teste.")
    parser.add_argument("--split-seed", type=int, default=42, help="Semente usada pelo train_test_split.")
    parser.add_argument(
        "--stratify-split",
        action="store_true",
        help="Mantém a proporção dos rótulos em todas as divisões.",
    )
    parser.add_argument(
        "--history-log",
        type=str,
        help="Arquivo JSON onde o histórico de treino/validação será salvo.",
    )
    args = parser.parse_args()

    total_ratio = args.train_size + args.val_size + args.test_size
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        parser.error("train-size + val-size + test-size deve ser igual a 1.0")
    for name, value in (("train-size", args.train_size), ("val-size", args.val_size), ("test-size", args.test_size)):
        if value <= 0 or value >= 1:
            parser.error(f"{name} deve estar entre 0 e 1 (exclusivo)")
    if len(args.delimiter) != 1:
        parser.error("--delimiter deve conter exatamente um caractere")
    if len(args.decimal_sep) != 1:
        parser.error("--decimal-sep deve conter exatamente um caractere")

    return args


def build_data_source(args):
    common_kwargs = {
        "csv_path": args.csv,
        "label_column": args.label_column,
        "feature_slice": tuple(args.feature_slice) if args.feature_slice else None,
        "feature_columns": args.feature_columns,
        "ignored_columns": args.ignored_columns,
        "skip_header": not args.no_header,
        "delimiter": args.delimiter,
        "decimal_separator": args.decimal_sep,
    }
    if args.data_source == "in-memory":
        return InMemoryCSVDataSource(**common_kwargs)
    return StreamingCSVDataSource(buffer_size=args.buffer_size, **common_kwargs)


def build_feature_map(args):
    if args.kernel == "linear":
        return LinearFeatureMap()
    return RandomFourierFeatureMap(gamma=args.rbf_gamma, n_components=args.rbf_components)


def split_dataset(args, base_source):
    features, labels = base_source.as_arrays()
    stratify_labels = labels if args.stratify_split else None

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        features,
        labels,
        test_size=args.test_size,
        random_state=args.split_seed,
        stratify=stratify_labels,
    )

    relative_val = args.val_size / (args.train_size + args.val_size)
    stratify_trainval = y_trainval if args.stratify_split else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=relative_val,
        random_state=args.split_seed,
        stratify=stratify_trainval,
    )

    num_classes = base_source.num_classes
    class_names = base_source.class_names
    train_source = ArrayDataSource(
        X_train,
        y_train,
        rng_seed=args.split_seed,
        num_classes=num_classes,
        class_names=class_names,
    )
    val_source = ArrayDataSource(
        X_val,
        y_val,
        rng_seed=args.split_seed,
        num_classes=num_classes,
        class_names=class_names,
    )
    test_source = ArrayDataSource(
        X_test,
        y_test,
        rng_seed=args.split_seed,
        num_classes=num_classes,
        class_names=class_names,
    )
    return train_source, val_source, test_source


def print_class_summary(data_source):
    if not data_source.class_names:
        return
    print("Classes detectadas/alvo:")
    for idx, name in enumerate(data_source.class_names):
        print(f"  [{idx}] {name}")


def main():
    args = parse_args()
    data_source = build_data_source(args)
    feature_map = build_feature_map(args)
    train_source, val_source, test_source = split_dataset(args, data_source)

    print(
        f"Dataset: {data_source.num_samples} amostras | "
        f"{data_source.num_features} features | "
        f"{data_source.num_classes} classes"
    )
    print_class_summary(data_source)
    print(
        f"Split -> treino: {train_source.num_samples} | "
        f"validação: {val_source.num_samples} | "
        f"teste: {test_source.num_samples}"
    )

    model = SoftmaxClassifier(
        num_features=data_source.num_features,
        num_classes=data_source.num_classes,
        feature_map=feature_map,
        use_gpu=args.use_gpu,
    )

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        use_gpu=args.use_gpu,
        history_log_path=args.history_log,
    )

    model.fit_with_validation(train_source, val_source, config)

    test_metrics = model.evaluate(test_source, batch_size=config.batch_size)
    print(f"[Teste] loss={test_metrics['loss']:.4f} accuracy={test_metrics['accuracy']:.4f}")

    if args.save_model:
        Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
        model.save(args.save_model)
        print(f"Pesos salvos em {args.save_model}")


if __name__ == "__main__":
    main()
