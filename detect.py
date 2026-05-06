from ultralytics import YOLO
import argparse
from datetime import datetime
from pathlib import Path
import os

# -- Configuration ------------------------------------------
MODEL_PATH = "latest_training/best.pt"
CONFIDENCE = 0.35
ALERT_LOG = "alerts/alert_log.txt"
OUTPUT_DIR = "runs/detect"

VIOLATION_CLASSES = {
    "plant_plucking",
    "animal_touching",
}

RISK_CLASSES = {
    "plant_approaching",
}

EXPECTED_CLASSES = RISK_CLASSES | VIOLATION_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description="Run park activity detection on one image or video.")
    parser.add_argument(
        "--source",
        required=True,
        help="Image or video file path to test.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=CONFIDENCE,
        help="Minimum detection confidence. Default: 0.35",
    )
    return parser.parse_args()


def class_names_from_model(model):
    names = model.names
    if isinstance(names, dict):
        return [names[index] for index in sorted(names)]
    return list(names)


def alert_level_for(class_name: str):
    if class_name in RISK_CLASSES:
        return "RISK"
    if class_name in VIOLATION_CLASSES:
        return "VIOLATION"
    return None


def format_detection_message(class_name: str, alert_level: str, confidence: float, frame_number: int):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"[{timestamp}] {alert_level} | "
        f"Activity: {class_name.upper():20s} | "
        f"Confidence: {confidence:.2%} | "
        f"Frame: {frame_number}"
    )


def log_violation_alert(class_name: str, confidence: float, frame_number: int):
    os.makedirs("alerts", exist_ok=True)
    message = format_detection_message(class_name, "VIOLATION", confidence, frame_number)
    print(f"\n{message}")
    with open(ALERT_LOG, "a", encoding="utf-8") as file:
        file.write(message + "\n")


def show_risk_notice(class_name: str, confidence: float, frame_number: int):
    message = format_detection_message(class_name, "RISK", confidence, frame_number)
    print(f"\n{message} | Guide notice only")


def process_result(result, class_names, frame_number):
    detections_count = 0

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = class_names[class_id]
        alert_level = alert_level_for(class_name)
        if alert_level is None:
            continue

        detections_count += 1
        if alert_level == "VIOLATION":
            log_violation_alert(class_name, confidence, frame_number)
        elif alert_level == "RISK":
            show_risk_notice(class_name, confidence, frame_number)

    return detections_count


def main():
    args = parse_args()
    source = Path(args.source)

    print("=" * 60)
    print("  Sarawak Park Guide - Activity Detection")
    print("=" * 60)

    if not Path(MODEL_PATH).exists():
        print(f"\n[ERROR] Model not found at: {MODEL_PATH}")
        print("Please run python3 training.py first.")
        return

    if not source.exists():
        print(f"\n[ERROR] Source file not found: {source}")
        return

    print(f"\nLoading model : {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    class_names = class_names_from_model(model)
    missing_classes = sorted(EXPECTED_CLASSES - set(class_names))

    if missing_classes:
        print("\n[WARNING] This model does not include all 3 expected classes:")
        for class_name in missing_classes:
            print(f"  - {class_name}")
        print("Retrain with an updated dataset.yaml if you need these classes detected.")

    print(f"Source        : {source}")
    print(f"Confidence    : {args.confidence}")
    print(f"Alert log     : {ALERT_LOG}")
    print(f"Classes       : {', '.join(class_names)}")
    print()

    results = model.predict(
        source=str(source),
        conf=args.confidence,
        save=True,
        project=OUTPUT_DIR,
        name=source.stem,
        exist_ok=True,
        verbose=False,
        stream=True,
    )

    total_detections = 0
    for frame_number, result in enumerate(results, start=1):
        total_detections += process_result(result, class_names, frame_number)

    if total_detections == 0:
        print("Result: no risk/violation detected")

    print(f"\nAnnotated result saved under: {OUTPUT_DIR}/{source.stem}/")
    print(f"Detection complete. Admin violation alerts saved to: {ALERT_LOG}")


if __name__ == "__main__":
    main()
