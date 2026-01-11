import argparse
import logging
import csv
import math
import time
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import optuna

try:
    from sklearn.metrics import accuracy_score, f1_score
except ImportError as exc:
    raise RuntimeError(
        "Please install scikit-learn to compute accuracy and F1 metrics (`pip install scikit-learn`)."
    ) from exc

LOGGER = logging.getLogger(__name__)


def configure_logger(log_file: Path) -> None:
    """Prepare console/file logging with a consistent format."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def resize_to_height(image: Image.Image, target_height: int) -> Image.Image:
    """Resize the image so that its height matches target_height while keeping aspect ratio."""
    width, height = image.size
    if height == target_height:
        return image
    new_width = max(1, math.floor(width * target_height / height))
    return image.resize((new_width, target_height), Image.Resampling.LANCZOS)


class HeightResizer:
    """Callable used within transform pipelines to keep height fixed without lambdas."""

    def __init__(self, target_height: int) -> None:
        self._target_height = target_height

    def __call__(self, image: Image.Image) -> Image.Image:
        return resize_to_height(image, self._target_height)


def build_transforms(height: int, width: int) -> transforms.Compose:
    """Return preprocessing pipeline that enforces height + center width crop + normalization."""
    return transforms.Compose(
        [
            HeightResizer(height),
            transforms.CenterCrop((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


class FaceClassifier(nn.Module):
    """Simple CNN block followed by a two-layer classifier ending in softmax logits."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        dropout: float = 0.3,
        feature_dim: int = 128,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.cnn(x))


class CsvImageDataset(Dataset):
    """Dataset that reads image paths and labels from a CSV annotation file."""

    def __init__(
        self,
        annotations_file: Path,
        image_dir: Path,
        transform: transforms.Compose = None,
    ) -> None:
        self.transform = transform
        self.image_dir = image_dir
        self.samples: list[tuple[Path, int]] = []
        self.class_to_idx: dict[str, int] = {}
        LOGGER.info("Lendo arquivo: %s...", annotations_file)
        with annotations_file.open(newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile, delimiter=";")
            LOGGER.info("Lendo arquivo Reader: %s", reader)
            for row_index, row in enumerate(reader):
                if row_index == 0 or not row or len(row) < 2:
                    continue
                filename, label = row[0].strip(), row[1].strip()
                LOGGER.debug("Filename: %s\tLabel: %s", filename, label)
                if not filename or not label:
                    continue
                image_path = image_dir / filename
                if not image_path.is_file():
                    raise FileNotFoundError(f"Annotation references missing image {image_path!r}.")
                if label not in self.class_to_idx:
                    self.class_to_idx[label] = len(self.class_to_idx)
                self.samples.append((image_path, self.class_to_idx[label]))

        if not self.samples:
            raise ValueError("No valid samples found in the annotations CSV.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label_idx = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label_idx, dtype=torch.long)


def compute_metrics(outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    """Turn logits into predictions and compute accuracy + macro F1."""
    preds = torch.argmax(outputs, dim=1).cpu().tolist()
    labels = targets.cpu().tolist()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return acc, f1


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Single epoch training that reports loss, accuracy, and F1 score."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for batch_index, batch in enumerate(loader):
        LOGGER.info("Carregando batch: %d", batch_index)
        LOGGER.debug("Carregando batch do loader: %s", batch)
        inputs, labels = batch
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_preds.append(outputs.detach())
        all_labels.append(labels.detach())
    logits = torch.cat(all_preds)
    targets = torch.cat(all_labels)
    acc, f1 = compute_metrics(logits, targets)
    average_loss = total_loss / len(loader)
    return average_loss, acc, f1


def create_data_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool = False,
) -> DataLoader:
    """Build a DataLoader with common options shared by training and evaluation."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def run_training_epochs(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    log_metrics: bool = True,
    prefix: str = "",
) -> Tuple[float, float, float]:
    """Run the main epoch loop, optionally logging per-epoch metrics."""
    last_loss = last_acc = last_f1 = 0.0
    prefix_text = f"{prefix} " if prefix else ""
    for epoch in range(1, epochs + 1):
        last_loss, last_acc, last_f1 = train_epoch(model, loader, optimizer, criterion, device)
        if log_metrics:
            LOGGER.info(
                "%sEpoch %d/%d | loss=%.4f | accuracy=%.4f | f1=%.4f",
                prefix_text,
                epoch,
                epochs,
                last_loss,
                last_acc,
                last_f1,
            )
    return last_loss, last_acc, last_f1


def measure_inference_latency(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    """Run inference over the loader and report min/max/avg per-sample latency (seconds)."""
    model.eval()
    batch_latencies: list[float] = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            batch_time = end - start
            if inputs.shape[0]:
                batch_latencies.append(batch_time / inputs.shape[0])
    if not batch_latencies:
        return 0.0, 0.0, 0.0
    return min(batch_latencies), max(batch_latencies), sum(batch_latencies) / len(batch_latencies)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a face recognition CNN on CelebA-style folders."
    )
    parser.add_argument("--input-file", type=Path, help="CSV File containing image names and labels.")
    parser.add_argument("--image-dir", type=Path, help="Root folder where images are located.")
    parser.add_argument(
        "--height", type=int, default=128, help="Target height after resizing the images."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="Target width to center-crop after height normalization.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Samples per batch.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 regularization.")
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=0,
        help="Number of Optuna trials to run; set >0 to enable tuning.",
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout in classifier.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Override the number of classes; defaults to CSV-derived label count.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("training.log"),
        help="File that also collects training logs.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("face_model.pt"),
        help="Path where the final checkpoint is saved.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Reproducible randomness seed for torch/cuda."
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU via CUDA if available.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Batch size used for inference timing; defaults to --batch-size when unset.",
    )
    return parser.parse_args()


def run_optuna_trials(
    args: argparse.Namespace,
    dataset: Dataset,
    target_num_classes: int,
    device: torch.device,
    eval_batch_size: int,
) -> optuna.study.Study:
    """Use Optuna to search over batch size, learning rate, dropout, and weight decay."""
    study = optuna.create_study(direction="maximize")

    def objective(trial: optuna.trial.Trial) -> float:
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 1e-2)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        loader = create_data_loader(dataset, batch_size, args.num_workers, device, shuffle=True)
        model = FaceClassifier(num_classes=target_num_classes, dropout=dropout).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        _, accuracy, _ = run_training_epochs(
            model, loader, optimizer, criterion, device, args.epochs, log_metrics=False
        )

        inference_loader = create_data_loader(
            dataset, eval_batch_size, args.num_workers, device, shuffle=False
        )
        min_latency, max_latency, avg_latency = measure_inference_latency(
            model, inference_loader, device
        )
        trial.set_user_attr("min_latency", min_latency)
        trial.set_user_attr("max_latency", max_latency)
        trial.set_user_attr("avg_latency", avg_latency)

        LOGGER.info(
            "Optuna trial %d | accuracy=%.4f | batch_size=%d | dropout=%.2f | lr=%.5f | "
            "weight_decay=%.5f | lat(min/avg/max)=%.6f/%.6f/%.6f",
            trial.number,
            accuracy,
            batch_size,
            dropout,
            lr,
            weight_decay,
            min_latency,
            avg_latency,
            max_latency,
        )
        return accuracy

    study.optimize(objective, n_trials=args.optuna_trials)
    return study


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def main() -> None:
    args = parse_args()
    if not args.input_file.exists():
        raise FileNotFoundError(f"{args.input_file} does not exist.")
    if not args.image_dir.exists():
        raise FileNotFoundError(f"{args.image_dir} does not exist.")
    configure_logger(args.log_file)
    set_seed(args.seed)
    device = (
        torch.device("cuda")
        if args.gpu and torch.cuda.is_available()
        else torch.device("cpu")
    )
    LOGGER.info("Running on %s", device)

    preprocessing = build_transforms(args.height, args.width)
    dataset = CsvImageDataset(args.input_file, args.image_dir, transform=preprocessing)
    dataset_classes = len(dataset.class_to_idx)
    LOGGER.info("Found %d samples across %d classes.", len(dataset), dataset_classes)
    if args.num_classes is not None and args.num_classes != dataset_classes:
        LOGGER.warning(
            "CSV defines %d classes but --num-classes=%d was passed; using CSV-derived count.",
            dataset_classes,
            args.num_classes,
        )

    target_num_classes = dataset_classes
    eval_batch_size = args.eval_batch_size or args.batch_size

    selected_batch_size = args.batch_size
    selected_dropout = args.dropout
    selected_lr = args.lr
    selected_weight_decay = args.weight_decay

    if args.optuna_trials > 0:
        study = run_optuna_trials(args, dataset, target_num_classes, device, eval_batch_size)
        best_trial = study.best_trial
        best_params = best_trial.params
        selected_batch_size = best_params.get("batch_size", selected_batch_size)
        selected_dropout = best_params.get("dropout", selected_dropout)
        selected_lr = best_params.get("lr", selected_lr)
        selected_weight_decay = best_params.get("weight_decay", selected_weight_decay)
        LOGGER.info(
            "Optuna best trial %d | accuracy=%.4f | dropout=%.3f | lr=%.5f | weight_decay=%.5f",
            best_trial.number,
            best_trial.value,
            selected_dropout,
            selected_lr,
            selected_weight_decay,
        )
        LOGGER.info(
            "Best trial inference latency (s): min=%.6f | avg=%.6f | max=%.6f",
            best_trial.user_attrs.get("min_latency", 0.0),
            best_trial.user_attrs.get("avg_latency", 0.0),
            best_trial.user_attrs.get("max_latency", 0.0),
        )

    loader = create_data_loader(
        dataset, selected_batch_size, args.num_workers, device, shuffle=True
    )
    model = FaceClassifier(num_classes=target_num_classes, dropout=selected_dropout).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=selected_lr, weight_decay=selected_weight_decay
    )

    final_loss, final_acc, final_f1 = run_training_epochs(
        model, loader, optimizer, criterion, device, args.epochs
    )
    LOGGER.info(
        "Final epoch metrics | loss=%.4f | accuracy=%.4f | f1=%.4f",
        final_loss,
        final_acc,
        final_f1,
    )

    inference_loader = create_data_loader(
        dataset, eval_batch_size, args.num_workers, device, shuffle=False
    )
    min_latency, max_latency, avg_latency = measure_inference_latency(
        model, inference_loader, device
    )
    torch.save(model.state_dict(), args.model_out)
    LOGGER.info(
        "Inference latency (s) | min=%.6f | max=%.6f | avg=%.6f",
        min_latency,
        max_latency,
        avg_latency,
    )
    LOGGER.info("Training complete; weights written to %s", args.model_out)


if __name__ == "__main__":
    main()
