import argparse
import logging
import csv
import math
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import json

# try:
#     import optuna
# except ImportError as exc:
#     raise RuntimeError("Optuna is required for hyperparameter tuning (`pip install optuna`).") from exc

LOGGER = logging.getLogger(__name__)

# Store metric history across epochs/evaluations.
obj_metrics = {
    "training_loss": [],
    "validation_loss": [],
    "acc": [],
    "f1": [],
}


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
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )


class FaceClassifier(nn.Module):
    """Simple CNN block followed by a two-layer classifier ending in softmax logits."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        dropout: float = 0.3,
        feature_dim: int = 16536,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(8),
            # nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),
            # nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32768, feature_dim),
            nn.ReLU(),
            # nn.Linear(feature_dim, num_classes * 2),
            # nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        return self.classifier( x )


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
        image = Image.open(image_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label_idx, dtype=torch.long)


def compute_metrics_from_logits(
    outputs: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> Tuple[float, float]:
    """Compute accuracy + macro F1 from logits using PyTorch ops."""
    if targets.numel() == 0:
        return 0.0, 0.0
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == targets).sum().item()
    accuracy = correct / targets.numel()
    LOGGER.info("Targets ==> %s", targets)
    LOGGER.info("Predicts ==> %s", preds)
    LOGGER.info("Corrects ==> %s", correct)

    preds_onehot = F.one_hot(preds, num_classes=num_classes).to(dtype=torch.float64)
    targets_onehot = F.one_hot(targets, num_classes=num_classes).to(dtype=torch.float64)
    true_positives = (preds_onehot * targets_onehot).sum(dim=0)
    pred_totals = preds_onehot.sum(dim=0)
    target_totals = targets_onehot.sum(dim=0)
    precision = torch.where(pred_totals == 0, torch.zeros_like(pred_totals), true_positives / pred_totals)
    recall = torch.where(target_totals == 0, torch.zeros_like(target_totals), true_positives / target_totals)
    denom = precision + recall
    f1 = torch.where(denom == 0, torch.zeros_like(denom), 2 * precision * recall / denom)
    macro_f1 = f1.mean().item()
    return accuracy, macro_f1


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float, float]:
    global obj_metrics
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
    acc, f1 = compute_metrics_from_logits(logits, targets, num_classes)
    average_loss = total_loss / len(loader)
    obj_metrics["training_loss"].append(average_loss)
    obj_metrics["acc"].append(acc)
    obj_metrics["f1"].append(f1)
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
    num_classes: int,
    log_metrics: bool = True,
    prefix: str = "",
    val_loader: Optional[DataLoader] = None,
    val_num_classes: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Run the main epoch loop, optionally logging per-epoch metrics."""
    last_loss = last_acc = last_f1 = 0.0
    prefix_text = f"{prefix} " if prefix else ""
    for epoch in range(1, epochs + 1):
        last_loss, last_acc, last_f1 = train_epoch(
            model, loader, optimizer, criterion, device, num_classes
        )
        if val_loader is not None and val_num_classes is not None:
            val_loss, val_acc, val_f1 = evaluate_model(
                model, val_loader, device, val_num_classes, criterion
            )
            obj_metrics["validation_loss"].append(val_loss)
            if log_metrics:
                LOGGER.info(
                    "%sEpoch %d/%d | val_loss=%.4f | val_acc=%.4f | val_f1=%.4f",
                    prefix_text,
                    epoch,
                    epochs,
                    val_loss,
                    val_acc,
                    val_f1,
                )
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


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    criterion: Optional[nn.Module] = None,
) -> Tuple[float, float, float]:
    """Evaluate average loss, accuracy, and macro-F1 without modifying weights."""
    model.eval()
    outputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    total_loss = 0.0
    total_samples = 0
    criterion = criterion or nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(inputs)
            outputs.append(logits)
            targets.append(labels)
            batch_loss = criterion(logits, labels)
            total_loss += batch_loss.item() * labels.size(0)
            total_samples += labels.size(0)
    if not outputs or total_samples == 0:
        return 0.0, 0.0, 0.0
    logits = torch.cat(outputs)
    collected_targets = torch.cat(targets)
    average_loss = total_loss / total_samples
    acc, f1 = compute_metrics_from_logits(logits, collected_targets, num_classes)
    return average_loss, acc, f1


def stratified_split_dataset(
    dataset: Dataset,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[Dataset, Dataset, Dataset]:
    """Return train/val/test subsets that preserve label distribution."""
    if not math.isclose(train_frac + val_frac + test_frac, 1.0, rel_tol=1e-4):
        raise ValueError("Train/val/test split fractions must sum to 1.0.")
    total_samples = len(dataset)
    if total_samples < 3:
        raise ValueError("Need at least 3 samples to stratify splits.")

    label_to_indices: dict[int, list[int]] = {}
    for idx, record in enumerate(dataset.samples):
        _, label = record
        label_to_indices.setdefault(label, []).append(idx)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for label, indices in label_to_indices.items():
        generator = torch.Generator().manual_seed(seed + label)
        permutation = torch.randperm(len(indices), generator=generator)
        shuffled = [indices[i] for i in permutation.tolist()]
        LOGGER.debug("Label %d Shuffled %s", label, shuffled)
        train_count = int(len(shuffled) * train_frac)
        val_count = int(len(shuffled) * val_frac)
        # rest_count = int(len(shuffled) * (1.0 - train_frac))
        # test_count = rest_count - val_count

        LOGGER.debug("Sizes: Trainning %d    Validation %d    Tests %d", train_count, val_count, len(shuffled) - train_count - val_count)
        splits = [
            shuffled[:train_count],
            shuffled[train_count : train_count + val_count],
            shuffled[train_count + val_count :],
        ]
        train_indices.extend(splits[0])
        val_indices.extend(splits[1])
        test_indices.extend(splits[2])
    LOGGER.info("Result Sizes: Trainning %d    Validation %d    Tests %d", len(train_indices), len(val_indices), len(test_indices))
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    if len(train_subset) == 0 or len(val_subset) == 0 or len(test_subset) == 0:
        raise ValueError("Stratified split produced an empty subset; adjust split fractions.")
    return train_subset, val_subset, test_subset


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
    # parser.add_argument(
    #     "--optuna-trials",
    #     type=int,
    #     default=0,
    #     help="Number of Optuna trials to run; set >0 to enable tuning.",
    # )
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
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Fraction of samples assigned to the training split.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Fraction of samples assigned to validation; train+val+test must equal 1.",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.15,
        help="Fraction of samples assigned to test; train+val+test must equal 1.",
    )


    parsed_args = parser.parse_args()
    return parsed_args


# def run_optuna_trials(
#     args: argparse.Namespace,
#     train_dataset: Dataset,
#     val_dataset: Dataset,
#     test_dataset: Dataset,
#     target_num_classes: int,
#     device: torch.device,
#     eval_batch_size: int,
# ) -> optuna.study.Study:
#     """Use Optuna to search over batch size, learning rate, dropout, and weight decay."""
#     study = optuna.create_study(direction="maximize")

#     def objective(trial: optuna.trial.Trial) -> float:
#         dropout = trial.suggest_float("dropout", 0.1, 0.5)
#         lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
#         weight_decay = trial.suggest_float("weight_decay", 0.0, 1e-2)
#         # batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
#         batch_size = args.batch_size

#         train_loader = create_data_loader(
#             train_dataset, batch_size, args.num_workers, device, shuffle=True
#         )
#         model = FaceClassifier(num_classes=target_num_classes, dropout=dropout).to(device)
#         LOGGER.info("Model Architecture: %s", model)
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#         run_training_epochs(
#             model,
#             train_loader,
#             optimizer,
#             criterion,
#             device,
#             args.epochs,
#             target_num_classes,
#             log_metrics=False,
#         )

#         val_loader = create_data_loader(
#             val_dataset, eval_batch_size, args.num_workers, device, shuffle=False
#         )
#         val_loss, val_accuracy, val_f1 = evaluate_model(
#             model, val_loader, device, target_num_classes, criterion
#         )

#         inference_loader = create_data_loader(
#             test_dataset, eval_batch_size, args.num_workers, device, shuffle=False
#         )
#         min_latency, max_latency, avg_latency = measure_inference_latency(
#             model, inference_loader, device
#         )
#         trial.set_user_attr("val_accuracy", val_accuracy)
#         trial.set_user_attr("val_f1", val_f1)
#         trial.set_user_attr("val_loss", val_loss)
#         trial.set_user_attr("min_latency", min_latency)
#         trial.set_user_attr("max_latency", max_latency)
#         trial.set_user_attr("avg_latency", avg_latency)

#         LOGGER.info(
#             "Optuna trial %d | val_loss=%.4f | val_acc=%.4f | val_f1=%.4f | batch_size=%d | dropout=%.2f | lr=%.5f | "
#             "weight_decay=%.5f | lat(min/avg/max)=%.6f/%.6f/%.6f",
#             trial.number,
#             val_loss,
#             val_accuracy,
#             val_f1,
#             batch_size,
#             dropout,
#             lr,
#             weight_decay,
#             min_latency,
#             avg_latency,
#             max_latency,
#         )
#         return val_accuracy

#     study.optimize(objective, n_trials=args.optuna_trials)
#     return study


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def main() -> None:
    inicio = time.time()
    args = parse_args()
    if not args.input_file.exists():
        raise FileNotFoundError(f"{args.input_file} does not exist.")
    if not args.image_dir.exists():
        raise FileNotFoundError(f"{args.image_dir} does not exist.")
    configure_logger(args.log_file)
    set_seed(args.seed)
    LOGGER.info("Horario de inicio: %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    args_dict = vars(args)
    LOGGER.info("Args: %s", args_dict)
    # Method C: Iterate over the dictionary for formatted output
    LOGGER.info("Formatted arguments:")
    for arg_name, arg_value in args_dict.items():
        LOGGER.info("Args: %s => %s", arg_name, arg_value)
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

    train_dataset, val_dataset, test_dataset = stratified_split_dataset(
        dataset,
        args.train_split,
        args.val_split,
        args.test_split,
        args.seed,
    )

    LOGGER.info("Dataset splitted samples: (%d trainning; %d validation; %d tests).", len(train_dataset), len(val_dataset), len(test_dataset))

    selected_batch_size = args.batch_size
    selected_dropout = args.dropout
    selected_lr = args.lr
    selected_weight_decay = args.weight_decay

    # if args.optuna_trials > 0:
    #     study = run_optuna_trials(
    #         args,
    #         train_dataset,
    #         val_dataset,
    #         test_dataset,
    #         target_num_classes,
    #         device,
    #         eval_batch_size,
    #     )
    #     best_trial = study.best_trial
    #     best_params = best_trial.params
    #     selected_batch_size = best_params.get("batch_size", selected_batch_size)
    #     selected_dropout = best_params.get("dropout", selected_dropout)
    #     selected_lr = best_params.get("lr", selected_lr)
    #     selected_weight_decay = best_params.get("weight_decay", selected_weight_decay)
    #     LOGGER.info(
    #         "Optuna best trial %d | accuracy=%.4f | dropout=%.3f | lr=%.5f | weight_decay=%.5f",
    #         best_trial.number,
    #         best_trial.value,
    #         selected_dropout,
    #         selected_lr,
    #         selected_weight_decay,
    #     )
    #     LOGGER.info(
    #         "Best trial inference latency (s): min=%.6f | avg=%.6f | max=%.6f",
    #         best_trial.user_attrs.get("min_latency", 0.0),
    #         best_trial.user_attrs.get("avg_latency", 0.0),
    #         best_trial.user_attrs.get("max_latency", 0.0),
    #     )
    #     LOGGER.info(
    #         "Best trial validation metrics | loss=%.4f | accuracy=%.4f | f1=%.4f",
    #         best_trial.user_attrs.get("val_loss", 0.0),
    #         best_trial.user_attrs.get("val_accuracy", 0.0),
    #         best_trial.user_attrs.get("val_f1", 0.0),
    #     )

    train_loader = create_data_loader(
        train_dataset, selected_batch_size, args.num_workers, device, shuffle=True
    )
    val_loader = create_data_loader(
        val_dataset, eval_batch_size, args.num_workers, device, shuffle=False
    )
    test_loader = create_data_loader(
        test_dataset, eval_batch_size, args.num_workers, device, shuffle=False
    )
    model = FaceClassifier(num_classes=target_num_classes, dropout=selected_dropout).to(
        device
    )
    LOGGER.info("Model Architecture: %s", model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=selected_lr, weight_decay=selected_weight_decay
    )

    final_loss, final_acc, final_f1 = run_training_epochs(
        model,
        train_loader,
        optimizer,
        criterion,
        device,
        args.epochs,
        target_num_classes,
        val_loader=val_loader,
        val_num_classes=target_num_classes,
    )
    LOGGER.info(
        "Final training metrics | loss=%.4f | accuracy=%.4f | f1=%.4f",
        final_loss,
        final_acc,
        final_f1,
    )

    val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, device, target_num_classes, criterion)
    obj_metrics["validation_loss"].append(val_loss)
    LOGGER.info("Validation metrics | loss=%.4f | accuracy=%.4f | f1=%.4f", val_loss, val_acc, val_f1)
    test_loss, test_acc, test_f1 = evaluate_model(model, test_loader, device, target_num_classes, criterion)
    LOGGER.info("Test metrics | loss=%.4f | accuracy=%.4f | f1=%.4f", test_loss, test_acc, test_f1)

    min_latency, max_latency, avg_latency = measure_inference_latency(
        model, test_loader, device
    )
    torch.save(model.state_dict(), args.model_out)
    LOGGER.info(
        "Inference latency (s) | min=%.6f | max=%.6f | avg=%.6f",
        min_latency,
        max_latency,
        avg_latency,
    )
    LOGGER.info("Training complete; weights written to %s", args.model_out)
    termino = time.time()
    LOGGER.info("Metrics History: %s", json.dumps(obj_metrics))
    LOGGER.info("Horario de termino: %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    LOGGER.info("Tempo gasto: %6.2f segundos", (termino - inicio))


if __name__ == "__main__":
    main()
