import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "Datasets"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "output" / "results"
DEFAULT_SAVE_PATH = PROJECT_ROOT / "models" / "plant_model.pth"

INPUT_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the plant interaction classifier.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--save-path", type=Path, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class PlantDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_path, label = self.records[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def clean_class_name(folder_name: str) -> str:
    parts = folder_name.split(" - ")
    return parts[-1].strip() if len(parts) > 1 else folder_name.strip()


def load_records(data_dir: Path):
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    class_dirs = sorted([path for path in data_dir.iterdir() if path.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class folders found in {data_dir}")

    records = []
    class_names = []
    for class_index, class_dir in enumerate(class_dirs):
        class_name = clean_class_name(class_dir.name)
        class_names.append(class_name)
        image_paths = sorted(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png")))
        for image_path in image_paths:
            records.append((str(image_path), class_index))

    if not records:
        raise RuntimeError(f"No images found in {data_dir}")

    return records, class_names


def stratified_split(records, val_ratio=0.2, seed=42):
    rng = random.Random(seed)
    grouped = {}
    for index, (_, label) in enumerate(records):
        grouped.setdefault(label, []).append(index)

    train_indices = []
    val_indices = []
    for label, indices in grouped.items():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * val_ratio)))
        if len(shuffled) - val_count < 1:
            val_count = max(0, len(shuffled) - 1)
        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])

    train_records = [records[index] for index in train_indices]
    val_records = [records[index] for index in val_indices]
    return train_records, val_records


def build_transforms(input_size=INPUT_SIZE):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            input_size,
            scale=(0.75, 1.0),
            ratio=(0.9, 1.1),
            interpolation=InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.03),
        ], p=0.7),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    return train_transform, val_transform


def build_model(num_classes: int):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        if name.startswith("fc."):
            parameter.requires_grad = True
        else:
            parameter.requires_grad = trainable


def create_weighted_sampler(labels):
    counts = Counter(labels)
    sample_weights = [1.0 / counts[label] for label in labels]
    weights_tensor = torch.DoubleTensor(sample_weights)
    return WeightedRandomSampler(weights_tensor, num_samples=len(weights_tensor), replacement=True)


@torch.no_grad()
def evaluate(model, dataloader, criterion, class_names):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    preds = []
    labels = []

    for images, targets in dataloader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        batch_preds = torch.argmax(outputs, dim=1)
        preds.extend(batch_preds.detach().cpu().tolist())
        labels.extend(targets.detach().cpu().tolist())

    avg_loss = total_loss / max(1, total_samples)
    accuracy = float(np.mean(np.array(preds) == np.array(labels))) if labels else 0.0
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    macro_f1 = float(np.mean(f1)) if len(f1) else 0.0
    weighted_f1 = float(np.average(f1, weights=support)) if np.sum(support) else 0.0
    balanced_acc = float(np.mean(recall)) if len(recall) else 0.0

    per_class = []
    for index, class_name in enumerate(class_names):
        per_class.append({
            "class_name": class_name,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        })

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_acc,
        "confusion_matrix": cm,
        "per_class": per_class,
    }


def save_plots(history, cm, class_names, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss", marker="o")
    plt.plot(history["val_loss"], label="Val Loss", marker="o")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "loss_curve.png", dpi=140, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history["train_acc"], label="Train Acc", marker="o")
    plt.plot(history["val_acc"], label="Val Acc", marker="o")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "accuracy_curve.png", dpi=140, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=140, bbox_inches="tight")
    plt.close()


def freeze_backbone_for_warmup(model: nn.Module, freeze: bool) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = True
        if freeze and not name.startswith("fc."):
            parameter.requires_grad = False


def build_optimizer(model, lr, weight_decay, backbone_scale=1.0):
    backbone_params = []
    head_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("fc."):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)

    param_groups = []
    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr": lr * backbone_scale,
            "weight_decay": weight_decay,
        })
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": lr,
            "weight_decay": weight_decay,
        })

    return torch.optim.AdamW(param_groups)


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, use_amp):
    model.train()
    total_loss = 0.0
    total_samples = 0
    preds = []
    labels = []
    skipped_batches = 0

    for images, targets in dataloader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if not torch.isfinite(loss):
            skipped_batches += 1
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        batch_preds = torch.argmax(outputs.detach(), dim=1)
        preds.extend(batch_preds.cpu().tolist())
        labels.extend(targets.cpu().tolist())

    if total_samples == 0:
        return float("inf"), 0.0, skipped_batches

    avg_loss = total_loss / max(1, total_samples)
    accuracy = float(np.mean(np.array(preds) == np.array(labels))) if labels else 0.0
    return float(avg_loss), float(accuracy), skipped_batches


def write_history_csv(history, results_dir: Path):
    history_path = results_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "val_macro_f1"])
        writer.writeheader()
        for index in range(len(history["train_loss"])):
            writer.writerow({
                "epoch": index + 1,
                "train_loss": history["train_loss"][index],
                "val_loss": history["val_loss"][index],
                "train_acc": history["train_acc"][index],
                "val_acc": history["val_acc"][index],
                "val_macro_f1": history["val_macro_f1"][index],
            })


def save_summary(save_path: Path, summary: dict):
    summary_path = save_path.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary_path


def main():
    args = parse_args()
    seed_everything(args.seed)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    print("Loading dataset...")
    records, class_names = load_records(args.data_dir)
    train_records, val_records = stratified_split(records, val_ratio=args.val_ratio, seed=args.seed)

    train_counts = Counter(label for _, label in train_records)
    val_counts = Counter(label for _, label in val_records)

    print(f"Found {len(class_names)} classes: {class_names}")
    print(f"Total images: {len(records)}")
    print(f"Train: {len(train_records)}, Val: {len(val_records)}")

    train_transform, val_transform = build_transforms(INPUT_SIZE)
    train_dataset = PlantDataset(train_records, transform=train_transform)
    val_dataset = PlantDataset(val_records, transform=val_transform)

    train_sampler = create_weighted_sampler([label for _, label in train_records])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=2 if DEVICE.type == "cuda" else 0,
        pin_memory=DEVICE.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 if DEVICE.type == "cuda" else 0,
        pin_memory=DEVICE.type == "cuda",
    )

    print("Loading ResNet18...")
    model = build_model(num_classes=len(class_names)).to(DEVICE)
    freeze_backbone_for_warmup(model, freeze=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay, backbone_scale=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    use_amp = args.amp == "on" or (args.amp == "auto" and DEVICE.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
    }

    best_metric = -math.inf
    best_val_acc = 0.0
    best_val_loss = math.inf
    best_epoch = -1
    patience_counter = 0

    for epoch in range(args.epochs):
        if epoch == args.freeze_epochs:
            freeze_backbone_for_warmup(model, freeze=False)
            optimizer = build_optimizer(model, lr=args.lr * 0.5, weight_decay=args.weight_decay, backbone_scale=0.2)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
            print("Unfroze backbone for fine-tuning.")

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc, skipped_batches = train_one_epoch(model, train_loader, criterion, optimizer, scaler, use_amp)
        val_stats = evaluate(model, val_loader, criterion, class_names)

        if skipped_batches > 0:
            print(f"Skipped {skipped_batches} non-finite training batches.")

        if not math.isfinite(train_loss) or not math.isfinite(val_stats["loss"]):
            print("Non-finite loss detected; stopping this run early to avoid corrupt checkpoints.")
            break

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_stats["loss"])
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_stats["accuracy"])
        history["val_macro_f1"].append(val_stats["macro_f1"])

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['accuracy']:.4f}, Macro F1: {val_stats['macro_f1']:.4f}")
        scheduler.step(val_stats["loss"])

        improved = (val_stats["macro_f1"] > best_metric + 1e-6) or (
            abs(val_stats["macro_f1"] - best_metric) <= 1e-6 and val_stats["accuracy"] > best_val_acc
        )

        if improved:
            best_metric = val_stats["macro_f1"]
            best_val_acc = val_stats["accuracy"]
            best_val_loss = val_stats["loss"]
            best_epoch = epoch
            patience_counter = 0

            checkpoint = {
                "epoch": int(epoch),
                "best_epoch": int(best_epoch),
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "display_class_names": class_names,
                "input_size": int(INPUT_SIZE),
                "normalization": {
                    "mean": [float(value) for value in NORM_MEAN],
                    "std": [float(value) for value in NORM_STD],
                },
                "architecture": "resnet18",
                "best_val_acc": float(best_val_acc),
                "best_val_loss": float(best_val_loss),
                "best_macro_f1": float(best_metric),
                "train_counts": {str(key): int(value) for key, value in train_counts.items()},
                "val_counts": {str(key): int(value) for key, value in val_counts.items()},
                "seed": int(args.seed),
            }
            torch.save(checkpoint, args.save_path)
            print(f"Checkpoint saved (macro_f1: {best_metric:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print("\nLoading best model...")
    best_checkpoint = torch.load(args.save_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_checkpoint["state_dict"])
    final_stats = evaluate(model, val_loader, criterion, class_names)

    print("\nFinal Results:")
    print(f"Val Acc: {final_stats['accuracy']:.4f}, Val Loss: {final_stats['loss']:.4f}")
    print(f"Macro F1: {final_stats['macro_f1']:.4f}")
    print(f"Balanced Acc: {final_stats['balanced_accuracy']:.4f}")
    for row in final_stats["per_class"]:
        print(f"{row['class_name']}: P={row['precision']:.3f}, R={row['recall']:.3f}, F1={row['f1']:.3f}, N={row['support']}")

    save_plots(history, final_stats["confusion_matrix"], class_names, args.results_dir)
    write_history_csv(history, args.results_dir)

    metrics = {
        "best_epoch": int(best_epoch + 1),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "best_macro_f1": float(best_metric),
        "final_val_acc": float(final_stats["accuracy"]),
        "final_val_loss": float(final_stats["loss"]),
        "macro_f1": float(final_stats["macro_f1"]),
        "weighted_f1": float(final_stats["weighted_f1"]),
        "balanced_accuracy": float(final_stats["balanced_accuracy"]),
        "class_names": class_names,
        "train_counts": {class_names[index]: int(count) for index, count in train_counts.items()},
        "val_counts": {class_names[index]: int(count) for index, count in val_counts.items()},
        "per_class": final_stats["per_class"],
        "history": history,
        "config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "val_ratio": float(args.val_ratio),
            "patience": int(args.patience),
            "freeze_epochs": int(args.freeze_epochs),
            "label_smoothing": float(args.label_smoothing),
            "seed": int(args.seed),
            "input_size": int(INPUT_SIZE),
            "architecture": "resnet18",
            "device": str(DEVICE),
        },
        "model_path": str(args.save_path),
    }
    summary_path = save_summary(args.save_path, metrics)

    print(f"\nResults saved to {args.results_dir}")
    print(f"Model checkpoint saved to {args.save_path}")
    print(f"Model summary saved to {summary_path}")


if __name__ == "__main__":
    main()
