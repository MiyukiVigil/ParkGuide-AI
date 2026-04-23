"""Split an image dataset into train/test folders with group-aware sampling.

Groups are based on original image stem so augmented siblings (e.g. name__aug_001)
stay in the same split to reduce data leakage.
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset into train/test per class.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Dataset root with class subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output root. Creates train/ and test/ class folders.",
    )
    parser.add_argument(
        "--train-per-class",
        type=int,
        default=1000,
        help="Target train images per class.",
    )
    parser.add_argument(
        "--test-per-class",
        type=int,
        default=500,
        help="Target test images per class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output-dir first if it already exists.",
    )
    return parser.parse_args()


def image_files(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def base_key(file_path: Path) -> str:
    stem = file_path.stem
    if "__aug_" in stem:
        return stem.split("__aug_", 1)[0]
    return stem


def choose_groups(groups, target_count, rng):
    shuffled = list(groups)
    rng.shuffle(shuffled)

    selected = []
    total = 0
    for group in shuffled:
        if total >= target_count:
            break
        selected.append(group)
        total += len(group)

    if total < target_count:
        raise RuntimeError(
            f"Not enough images to satisfy requested split target={target_count}. Have={total}."
        )

    return selected


def flatten(groups):
    files = []
    for group in groups:
        files.extend(group)
    return files


def sample_exact(files, target_count, rng):
    copied = list(files)
    rng.shuffle(copied)
    return copied[:target_count]


def prepare_output_dir(output_dir: Path, overwrite: bool):
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_files(file_paths, destination_dir: Path):
    destination_dir.mkdir(parents=True, exist_ok=True)
    for src in file_paths:
        shutil.copy2(src, destination_dir / src.name)


def split_class(class_dir: Path, train_target: int, test_target: int, rng):
    files = image_files(class_dir)
    if not files:
        raise RuntimeError(f"No images found in class folder: {class_dir}")

    grouped = defaultdict(list)
    for file_path in files:
        grouped[base_key(file_path)].append(file_path)

    groups = list(grouped.values())
    train_groups = choose_groups(groups, train_target, rng)

    train_group_ids = {id(group) for group in train_groups}
    test_groups = [group for group in groups if id(group) not in train_group_ids]

    test_total = sum(len(group) for group in test_groups)
    if test_total < test_target:
        raise RuntimeError(
            f"Not enough remaining images for test in {class_dir.name}. "
            f"Need={test_target}, remaining={test_total}."
        )

    train_files = sample_exact(flatten(train_groups), train_target, rng)
    test_files = sample_exact(flatten(test_groups), test_target, rng)

    return train_files, test_files, len(files)


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input_dir}")

    class_dirs = sorted([p for p in args.input_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class folders in input dataset: {args.input_dir}")

    prepare_output_dir(args.output_dir, args.overwrite)

    train_root = args.output_dir / "train"
    test_root = args.output_dir / "test"

    print("Split summary")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print("")

    for class_dir in class_dirs:
        train_files, test_files, total = split_class(
            class_dir,
            train_target=args.train_per_class,
            test_target=args.test_per_class,
            rng=rng,
        )

        copy_files(train_files, train_root / class_dir.name)
        copy_files(test_files, test_root / class_dir.name)

        print(
            f"{class_dir.name}: total={total}, "
            f"train={len(train_files)}, test={len(test_files)}"
        )


if __name__ == "__main__":
    main()
