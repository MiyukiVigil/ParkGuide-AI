import os
from pathlib import Path
import yaml
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

# -- Configuration ------------------------------------------
# Accuracy-focused default. Use yolo11s.pt if training is too slow or runs out of memory.
MODEL_BASE   = "yolo11m.pt"

DATASET_YAML = BASE_DIR / "dataset.yaml"
EPOCHS       = 120
IMG_SIZE     = 768
BATCH_SIZE   = -1
PROJECT_DIR  = BASE_DIR / "runs/train"
RUN_NAME     = "park_activity_yolo11m"
# -----------------------------------------------------------

def _resolve_dataset_path(dataset_config: dict, yaml_path: Path) -> Path:
    dataset_root = Path(dataset_config.get("path", "."))
    if not dataset_root.is_absolute():
        dataset_root = yaml_path.parent / dataset_root
    return dataset_root.resolve()


def _resolve_split_path(dataset_root: Path, split_value: str) -> Path:
    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = dataset_root / split_path
    return split_path.resolve()


def load_dataset_config():
    yaml_path = Path(DATASET_YAML)
    if not yaml_path.exists():
        print(f"\n[ERROR] Dataset config not found: {DATASET_YAML}")
        return None

    with yaml_path.open("r", encoding="utf-8") as file:
        dataset_config = yaml.safe_load(file) or {}

    missing_keys = [key for key in ("train", "val", "names") if key not in dataset_config]
    if missing_keys:
        print("\n[ERROR] Missing required entries in dataset.yaml:")
        for key in missing_keys:
            print(f"  - {key}")
        return None

    return dataset_config, yaml_path


def dataset_folders_from_config(dataset_config: dict, yaml_path: Path):
    dataset_root = _resolve_dataset_path(dataset_config, yaml_path)
    train_images = _resolve_split_path(dataset_root, dataset_config["train"])
    val_images = _resolve_split_path(dataset_root, dataset_config["val"])

    train_labels = Path(str(train_images).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))
    val_labels = Path(str(val_images).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))

    return dataset_root, {
        "training images": train_images,
        "validation images": val_images,
        "training labels": train_labels,
        "validation labels": val_labels,
    }


def class_names_from_config(dataset_config: dict):
    names = dataset_config["names"]
    if isinstance(names, dict):
        return [name for _, name in sorted(names.items(), key=lambda item: int(item[0]))]
    return list(names)


def main():
    print("=" * 60)
    print("  Sarawak Park Guide - Activity Detection Training")
    print("=" * 60)

    loaded_config = load_dataset_config()
    if loaded_config is None:
        return

    dataset_config, yaml_path = loaded_config
    dataset_root, required_dirs = dataset_folders_from_config(dataset_config, yaml_path)
    class_names = class_names_from_config(dataset_config)

    missing = [label for label, path in required_dirs.items() if not path.is_dir()]
    if missing:
        print("\n[ERROR] Missing dataset folders:")
        for label in missing:
            print(f"  - {label}: {required_dirs[label]}")
        print("\nPlease check dataset.yaml and your dataset folders.")
        return

    print(f"\n[1/3] Loading base model: {MODEL_BASE}")
    model = YOLO(MODEL_BASE)

    print(f"[2/3] Starting training for {EPOCHS} epochs...")
    print(f"      Dataset : {DATASET_YAML}")
    print(f"      Root    : {dataset_root}")
    print(f"      Classes : {len(class_names)} ({', '.join(class_names)})")
    print(f"      Image sz: {IMG_SIZE}px")
    print(f"      Batch   : {BATCH_SIZE}")
    print(f"      Output  : {PROJECT_DIR}/{RUN_NAME}/\n")

    results = model.train(
        data      = str(DATASET_YAML),
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = BATCH_SIZE,
        project   = str(PROJECT_DIR),
        name      = RUN_NAME,

        # Accuracy-focused augmentation: enough variation without distorting actions too much.
        augment   = True,
        flipud    = 0.0,       # People/plants should stay upright
        fliplr    = 0.5,       # Horizontal flip 50%
        mosaic    = 0.7,       # Mosaic helps generalization, but too much can distort actions
        close_mosaic = 15,     # Finish training on natural-looking images
        degrees   = 5.0,       # Small camera tilt variation
        translate = 0.08,
        scale     = 0.4,
        hsv_h     = 0.015,     # Hue shift (handles lighting variation)
        hsv_s     = 0.5,       # Saturation shift
        hsv_v     = 0.3,       # Brightness shift

        # Early stopping — stops training if validation does not improve
        patience  = 20,

        # Save best model weights automatically
        save      = True,
        save_period = 10,      # Also save checkpoint every 10 epochs
    )

    print("\n[3/3] Training complete!")
    best_weights = PROJECT_DIR / RUN_NAME / "weights/best.pt"
    print(f"      Best model saved to: {best_weights}")
    print("\nNext step: run  python evaluate.py  to check accuracy.")
    print("           run  python detect.py    to test on video/camera.")


if __name__ == "__main__":
    main()
