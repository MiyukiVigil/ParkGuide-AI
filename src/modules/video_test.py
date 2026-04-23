import torch
import cv2
import csv
from pathlib import Path
from datetime import datetime
from PIL import Image
from torchvision import models, transforms
import numpy as np
from numpy._core.multiarray import scalar as numpy_scalar

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "plant_model.pth"
OUTPUT_DIR = PROJECT_ROOT / "output" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALERT_THRESHOLD = 0.85
ALERT_CLASSES = ["Touching", "Plucking"]

def load_model_and_transform():
    """Load model checkpoint"""
    if not MODEL_PATH.exists():
        return None, None, None
    
    torch.serialization.add_safe_globals([numpy_scalar])
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        class_names = checkpoint.get("display_class_names") or checkpoint.get("class_names")
        norm_mean = checkpoint.get("norm_mean", [0.485, 0.456, 0.406])
        norm_std = checkpoint.get("norm_std", [0.229, 0.224, 0.225])
    else:
        state_dict = checkpoint
        class_names = ["NoInteraction", "Approaching", "Touching", "Plucking"]
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
    
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, len(class_names))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    
    return model, class_names, transform

def process_video(video_path, output_video_path, threshold=0.85):
    """Process video with frame-by-frame detection"""
    model, class_names, transform = load_model_and_transform()
    
    if model is None:
        print("Model not found. Please train first.")
        return
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    alerts = []
    frame_count = 0
    
    print(f"Processing video: {video_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Predict on frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = probs.argmax(dim=1).item()
            confidence = probs.max().item()
        
        class_name = class_names[pred_class]
        
        # Check for alert
        is_alert = class_name in ALERT_CLASSES and confidence >= threshold
        
        if is_alert:
            alerts.append({
                'frame': frame_count,
                'timestamp': frame_count / fps,
                'class': class_name,
                'confidence': confidence,
            })
            
            # Save evidence frame
            evidence_path = OUTPUT_DIR / f"alert_frame_{frame_count}.jpg"
            cv2.imwrite(str(evidence_path), frame)
        
        # Draw on frame
        color = (0, 0, 255) if is_alert else (0, 255, 0)
        text = f"{class_name} ({confidence:.2f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    # Save alerts to CSV
    if alerts:
        alert_csv = OUTPUT_DIR / "alerts.csv"
        with open(alert_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['frame', 'timestamp', 'class', 'confidence'])
            writer.writeheader()
            writer.writerows(alerts)
        
        print(f"Found {len(alerts)} alerts. Saved to {alert_csv}")
    else:
        print("No alerts detected.")
    
    print(f"Output video saved to {output_video_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python video_test.py <video_path> [threshold] [output_path]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.85
    output_path = sys.argv[3] if len(sys.argv) > 3 else str(OUTPUT_DIR / "output_video.mp4")
    
    process_video(video_path, output_path, threshold)
