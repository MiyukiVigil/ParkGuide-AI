from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

BASE_DIR = Path(__file__).resolve().parent

# -- Configuration ------------------------------------------
MODEL_PATH   = BASE_DIR / "runs/train/park_activity_v2/weights/best.pt"
DATASET_YAML = BASE_DIR / "dataset.yaml"
IMG_SIZE     = 640
# -----------------------------------------------------------


def find_model_path():
    configured_path = Path(MODEL_PATH)
    if configured_path.exists():
        return configured_path

    search_roots = [
        BASE_DIR / "runs/train",
        Path("/opt/homebrew/runs/detect/runs/train"),
    ]

    trained_models = []
    for search_root in search_roots:
        if search_root.exists():
            trained_models.extend(search_root.glob("*/weights/best.pt"))

    trained_models = sorted(
        trained_models,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if trained_models:
        return trained_models[0]

    return None


def load_class_names():
    yaml_path = Path(DATASET_YAML)
    if not yaml_path.exists():
        return []

    if yaml is None:
        names = []
        in_names_block = False
        for line in yaml_path.read_text(encoding="utf-8").splitlines():
            if line.strip() == "names:":
                in_names_block = True
                continue
            if in_names_block and ":" in line:
                _, name = line.split(":", 1)
                names.append(name.strip())
        return names

    with yaml_path.open("r", encoding="utf-8") as file:
        dataset_config = yaml.safe_load(file) or {}

    names = dataset_config.get("names", [])
    if isinstance(names, dict):
        return [name for _, name in sorted(names.items(), key=lambda item: int(item[0]))]
    return list(names)


def main():
    print("=" * 60)
    print("  Sarawak Park Guide - Model Evaluation")
    print("=" * 60)

    model_path = find_model_path()
    if model_path is None:
        print(f"\n[ERROR] Model not found: {MODEL_PATH}")
        print("I also checked:")
        print(f"  {BASE_DIR / 'runs/train/*/weights/best.pt'}")
        print("  /opt/homebrew/runs/detect/runs/train/*/weights/best.pt")
        print("but no trained model was found.")
        print("\nRun training first:")
        print("  python3 training.py")
        print("\nNote: yolov8n.pt is only the pretrained base model.")
        print("After training finishes, YOLO should create:")
        print(f"  {BASE_DIR / 'runs/train/park_activity_v2/weights/best.pt'}")
        return

    print(f"\nLoading model : {model_path}")
    from ultralytics import YOLO

    model = YOLO(model_path)

    print(f"Evaluating on : {DATASET_YAML} (validation split)\n")

    metrics = model.val(
        data  = str(DATASET_YAML),
        imgsz = IMG_SIZE,
        split = "val",   
    )

    # ── Print summary ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)

    map50    = metrics.box.map50        # mAP at IoU threshold 0.5
    map50_95 = metrics.box.map          # mAP averaged over 0.5–0.95
    precision = metrics.box.mp          # Mean precision
    recall    = metrics.box.mr          # Mean recall

    print(f"\n  mAP@0.5         : {map50:.4f}   (main accuracy metric)")
    print(f"  mAP@0.5:0.95    : {map50_95:.4f}   (stricter accuracy)")
    print(f"  Precision       : {precision:.4f}   (how precise detections are)")
    print(f"  Recall          : {recall:.4f}   (how many violations caught)")

    # ── Per-class breakdown ────────────────────────────────
    class_names = load_class_names()

    print("\n  Per-class AP@0.5:")
    print("  " + "-" * 40)
    if hasattr(metrics.box, "ap50"):
        for i, cls_name in enumerate(class_names):
            if i < len(metrics.box.ap50):
                ap = metrics.box.ap50[i]
                bar = "█" * int(ap * 20)
                print(f"  {cls_name:22s}  {ap:.4f}  |{bar}")
    else:
        print("  (Per-class breakdown not available in this YOLO version)")

    # ── Interpretation guide ───────────────────────────────
    print("\n  Interpretation:")
    if map50 >= 0.85:
        print("  ✅ Excellent — model is ready for deployment testing.")
    elif map50 >= 0.70:
        print("  ✅ Good — consider collecting more images for weak classes.")
    elif map50 >= 0.50:
        print("  ⚠️  Fair — collect more data and re-train.")
    else:
        print("  ❌ Poor — check labels, increase data, review augmentation.")

    print("\n  Full metrics saved to: runs/val/")
    print("=" * 60)


if __name__ == "__main__":
    main()
