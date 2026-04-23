import torch
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
import sys
from numpy._core.multiarray import scalar as numpy_scalar

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "plant_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_transform():
    """Load model checkpoint with metadata"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    torch.serialization.add_safe_globals([numpy_scalar])
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    # Extract metadata
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        class_names = checkpoint.get("display_class_names") or checkpoint.get("class_names")
        norm_mean = checkpoint.get("norm_mean", [0.485, 0.456, 0.406])
        norm_std = checkpoint.get("norm_std", [0.229, 0.224, 0.225])
    else:
        # Old checkpoint format
        state_dict = checkpoint
        class_names = ["NoInteraction", "Approaching", "Touching", "Plucking"]
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
    
    # Build model
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, len(class_names))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    
    return model, class_names, transform

def predict_single_image(image_path):
    """Predict class for single image"""
    model, class_names, transform = load_model_and_transform()
    
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs.max().item()
    
    return class_names[pred_class], confidence, class_names

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    class_name, confidence, _ = predict_single_image(image_path)
    print(f"Predicted: {class_name} (Confidence: {confidence:.4f})")
