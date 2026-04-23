import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from numpy._core.multiarray import scalar as numpy_scalar


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULTS_DIR = OUTPUT_DIR / "results"
RUNS_DIR = RESULTS_DIR / "ui_runs"

MODEL_PATH = MODELS_DIR / "plant_model.pth"
SUMMARY_PATH = MODELS_DIR / "plant_model.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MODEL = None
_MODEL_CLASSES = None
_MODEL_TRANSFORM = None


def load_model_once():
    global _MODEL, _MODEL_CLASSES, _MODEL_TRANSFORM

    if _MODEL is not None:
        return _MODEL, _MODEL_CLASSES, _MODEL_TRANSFORM

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"plant_model.pth not found at {MODEL_PATH}. Please train first.")

    torch.serialization.add_safe_globals([numpy_scalar])
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        classes = checkpoint.get("display_class_names") or checkpoint.get("class_names")
        input_size = checkpoint.get("input_size", 224)
        norm_cfg = checkpoint.get("normalization", {})
        norm_mean = norm_cfg.get("mean", [0.485, 0.456, 0.406])
        norm_std = norm_cfg.get("std", [0.229, 0.224, 0.225])
    else:
        state_dict = checkpoint
        classes = ["NoInteraction", "Approaching", "Touching", "Plucking"]
        input_size = 224
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]

    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    _MODEL = model
    _MODEL_CLASSES = classes
    _MODEL_TRANSFORM = transform
    return _MODEL, _MODEL_CLASSES, _MODEL_TRANSFORM


def run_training(epochs, batch_size, lr, val_ratio, patience):
    cmd = [
        sys.executable,
        str(BASE_DIR / "training.py"),
        "--data-dir",
        str(PROJECT_ROOT / "Datasets"),
        "--epochs",
        str(int(epochs)),
        "--batch-size",
        str(int(batch_size)),
        "--lr",
        str(float(lr)),
        "--val-ratio",
        str(float(val_ratio)),
        "--patience",
        str(int(patience)),
        "--results-dir",
        str(RESULTS_DIR),
        "--save-path",
        str(MODEL_PATH),
    ]

    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")

    global _MODEL
    _MODEL = None

    status = "Training finished successfully." if proc.returncode == 0 else "Training failed."
    return status, combined.strip()


def refresh_results():
    if not SUMMARY_PATH.exists():
        return "No model summary found yet.", None, None, None

    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)

    lines = [
        "## Latest Metrics",
        f"- Best Val Loss: {summary.get('best_val_loss', 'n/a')}",
        f"- Best Val Accuracy: {summary.get('best_val_acc', 'n/a')}",
        f"- Final Val Accuracy: {summary.get('final_val_acc', 'n/a')}",
        f"- Macro F1: {summary.get('macro_f1', 'n/a')}",
    ]

    per_class = summary.get("per_class", [])
    if per_class:
        lines.append("\n### Per-Class")
        for row in per_class:
            lines.append(
                "- "
                f"{row['class_name']}: "
                f"P={row['precision']:.3f}, "
                f"R={row['recall']:.3f}, "
                f"F1={row['f1']:.3f}, "
                f"N={row['support']}"
            )

    loss_img = str(RESULTS_DIR / "loss_curve.png") if (RESULTS_DIR / "loss_curve.png").exists() else None
    acc_img = str(RESULTS_DIR / "accuracy_curve.png") if (RESULTS_DIR / "accuracy_curve.png").exists() else None
    cm_img = str(RESULTS_DIR / "confusion_matrix.png") if (RESULTS_DIR / "confusion_matrix.png").exists() else None

    return "\n".join(lines), loss_img, acc_img, cm_img


def predict_image(image_path):
    if not image_path:
        return "Please upload an image.", None

    model, classes, transform = load_model_once()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())
        conf = float(probs[0, pred].item() * 100.0)

    return f"Prediction: {classes[pred]} ({conf:.2f}%)", image


def process_video(video_path, alert_threshold):
    if isinstance(video_path, dict):
        video_path = video_path.get("video") or video_path.get("path")
    if isinstance(video_path, (list, tuple)) and video_path:
        video_path = video_path[0]

    if not video_path:
        return None, [], "Please upload a video file."

    model, classes, transform = load_model_once()

    critical_labels = {"Touching", "Plucking"}
    warning_label = "Approaching"
    threshold = float(alert_threshold)
    warning_threshold = min(0.40, max(0.25, threshold * 0.5))

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    evidence_dir = run_dir / "evidence_frames"
    run_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(video_path)
    output_path = run_dir / f"processed_{input_path.stem}.mp4"
    alerts_csv = run_dir / "alerts.csv"

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return None, [], "Could not read video."

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    alerts_rows = []
    frame_idx = 0
    class_index = {name: idx for idx, name in enumerate(classes)}
    approaching_idx = class_index.get(warning_label)
    critical_indices = {name: class_index.get(name) for name in critical_labels if name in class_index}

    with open(alerts_csv, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["status", "timestamp", "frame", "label", "confidence", "evidence_path"])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(tensor)
                probs = F.softmax(output, dim=1)
                top_pred = int(torch.argmax(probs, dim=1).item())

            top_label = classes[top_pred]
            top_conf = float(probs[0, top_pred].item())

            critical_hits = []
            for label_name, idx in critical_indices.items():
                if idx is None:
                    continue
                score = float(probs[0, idx].item())
                if score >= threshold:
                    critical_hits.append((label_name, score))

            warning_score = float(probs[0, approaching_idx].item()) if approaching_idx is not None else 0.0
            is_warning = warning_score >= warning_threshold
            is_critical = len(critical_hits) > 0

            if critical_hits:
                label, conf = max(critical_hits, key=lambda item: item[1])
                status = "ALERT"
            elif is_warning:
                label, conf = warning_label, warning_score
                status = "WARNING"
            else:
                label, conf = top_label, top_conf
                status = "OK"

            if is_critical or is_warning:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                evidence_name = f"frame_{frame_idx:06d}_{label}.jpg"
                evidence_path = evidence_dir / evidence_name
                cv2.imwrite(str(evidence_path), frame)

                row = [status, ts, frame_idx, label, f"{conf * 100.0:.2f}", str(evidence_path)]
                csv_writer.writerow(row)
                alerts_rows.append(row)

            if status == "WARNING":
                color = (0, 200, 255)
                prefix = "WARN"
            elif status == "ALERT":
                color = (0, 0, 255)
                prefix = "ALERT"
            else:
                color = (0, 255, 0)
                prefix = "OK"
            text = f"{prefix}: {label} ({conf * 100.0:.1f}%)"
            cv2.putText(frame, text, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            writer.write(frame)

    cap.release()
    writer.release()

    message = (
        f"Processed video saved to: {output_path}\n"
        f"Alerts saved to: {alerts_csv}\n"
        f"Total warnings/alerts: {len(alerts_rows)}"
    )
    return str(output_path), alerts_rows, message


def build_ui():
    with gr.Blocks(title="ParkGuideAI Unified Console") as demo:
        gr.Markdown("# ParkGuideAI Unified Console")
        gr.Markdown("Train model, view results, run image inference, and process videos from one UI.")

        with gr.Tab("Train"):
            epochs = gr.Slider(1, 50, value=12, step=1, label="Epochs")
            batch_size = gr.Slider(4, 64, value=16, step=4, label="Batch Size")
            lr = gr.Number(value=0.001, label="Learning Rate")
            val_ratio = gr.Slider(0.1, 0.4, value=0.2, step=0.05, label="Validation Ratio")
            patience = gr.Slider(2, 10, value=4, step=1, label="Early Stop Patience")
            train_btn = gr.Button("Start Training")
            train_status = gr.Textbox(label="Status")
            train_log = gr.Textbox(label="Training Log", lines=16)

            train_btn.click(
                fn=run_training,
                inputs=[epochs, batch_size, lr, val_ratio, patience],
                outputs=[train_status, train_log],
            )

        with gr.Tab("Results"):
            refresh_btn = gr.Button("Refresh Results")
            summary_md = gr.Markdown()
            with gr.Row():
                loss_plot = gr.Image(label="Loss Curve")
                acc_plot = gr.Image(label="Accuracy Curve")
            cm_plot = gr.Image(label="Confusion Matrix")

            refresh_btn.click(fn=refresh_results, outputs=[summary_md, loss_plot, acc_plot, cm_plot])

        with gr.Tab("Image Prediction"):
            image_input = gr.Image(type="filepath", label="Upload Image")
            predict_btn = gr.Button("Predict")
            pred_text = gr.Textbox(label="Prediction")
            preview = gr.Image(label="Preview")

            predict_btn.click(fn=predict_image, inputs=[image_input], outputs=[pred_text, preview])

        with gr.Tab("Video Monitoring"):
            video_input = gr.Video(label="Upload Video", sources=["upload"], format="mp4")
            alert_threshold = gr.Slider(0.5, 0.99, value=0.85, step=0.01, label="Alert Confidence Threshold")
            process_btn = gr.Button("Process Video")
            processed_video = gr.Video(label="Processed Video")
            alerts_table = gr.Dataframe(
                headers=["status", "timestamp", "frame", "label", "confidence", "evidence_path"],
                datatype=["str", "str", "number", "str", "str", "str"],
                label="Alerts",
            )
            video_msg = gr.Textbox(label="Run Summary", lines=4)

            process_btn.click(
                fn=process_video,
                inputs=[video_input, alert_threshold],
                outputs=[processed_video, alerts_table, video_msg],
            )

    return demo


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Models dir: {MODELS_DIR}")
    print(f"Results will be saved to: {RESULTS_DIR}")
    app = build_ui()
    app.launch()
