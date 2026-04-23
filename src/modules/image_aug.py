"""Data augmentation utilities and dataset expansion script."""
import argparse
import random
from pathlib import Path

from PIL import Image
from torchvision import transforms
import torch

def get_train_transforms(input_size=224):
    """Training augmentation (random crops, flips, color jittering)"""
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.15),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

def get_val_transforms(input_size=224):
    """Validation transforms (no augmentation)"""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

def get_test_transforms(input_size=224):
    """Test transforms (same as validation)"""
    return get_val_transforms(input_size)


def get_export_transforms(input_size=224):
    """Augmentation pipeline for writing image files (no tensor normalization)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.15),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
    ])


def parse_args():
    parser = argparse.ArgumentParser(description="Duplicate dataset images with random augmentation.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "Datasets",
        help="Input dataset root containing class folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dataset root. Defaults to <data-dir>_augmented.",
    )
    parser.add_argument(
        "--copies-per-image",
        type=int,
        default=12,
        help="How many augmented copies to create for each original image.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Target size used for random resized crop.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible augmentation.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Write augmented images into the existing class folders.",
    )
    return parser.parse_args()


def list_images(folder: Path):
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    images = []
    for pattern in patterns:
        images.extend(folder.glob(pattern))
    return sorted(images)


def duplicate_dataset(data_dir: Path, output_dir: Path, copies_per_image: int, input_size: int, seed: int):
    if copies_per_image < 1:
        raise ValueError("copies_per_image must be at least 1")
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    random.seed(seed)
    torch.manual_seed(seed)
    transform = get_export_transforms(input_size=input_size)

    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class folders found in {data_dir}")

    total_original = 0
    total_augmented = 0

    for class_dir in class_dirs:
        class_out = output_dir / class_dir.name
        class_out.mkdir(parents=True, exist_ok=True)

        images = list_images(class_dir)
        total_original += len(images)

        for image_path in images:
            with Image.open(image_path) as image:
                image = image.convert("RGB")

                if output_dir != data_dir:
                    original_out = class_out / image_path.name
                    if not original_out.exists():
                        image.save(original_out, quality=95)

                stem = image_path.stem
                for i in range(copies_per_image):
                    aug_image = transform(image)
                    out_name = f"{stem}__aug_{i + 1:03d}.jpg"
                    out_path = class_out / out_name
                    aug_image.save(out_path, quality=95)
                    total_augmented += 1

    return total_original, total_augmented

if __name__ == "__main__":
    args = parse_args()

    destination = args.data_dir if args.inplace else (args.output_dir or Path(f"{args.data_dir}_augmented"))
    destination.mkdir(parents=True, exist_ok=True)

    originals, augmented = duplicate_dataset(
        data_dir=args.data_dir,
        output_dir=destination,
        copies_per_image=args.copies_per_image,
        input_size=args.input_size,
        seed=args.seed,
    )
    print(f"Done. Original images: {originals}")
    print(f"Generated augmented images: {augmented}")
    print(f"Output dataset: {destination}")
