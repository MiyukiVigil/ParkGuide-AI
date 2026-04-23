# ParkGuideAI: Plant Interaction Detection Prototype

AI-powered monitoring system to detect abnormal plant interactions in park guiding scenarios.

## Project Structure

```
ParkGuideAI/
├── src/                          # Source code
│   ├── app.py                   # Main Gradio UI launcher
│   ├── training.py              # Training pipeline
│   └── modules/
│       ├── test.py              # Single image inference
│       ├── video_test.py        # Video monitoring with alerts
│       ├── show_results.py      # Result visualization
│       └── image_aug.py         # Data augmentation utilities
├── models/                      # Trained model checkpoints
│   ├── plant_model.pth
│   └── plant_model.json
├── data/                        # Datasets and samples
│   ├── samples/
│   │   └── test.jpg
│   └── Datasets/ (from repo)
├── docs/                        # Documentation
│   ├── SCOPE_ALIGNMENT.md
│   └── Project Scope.pdf
├── output/                      # Training and inference outputs
│   └── results/
│       ├── loss_curve.png
│       ├── accuracy_curve.png
│       ├── confusion_matrix.png
│       ├── metrics.json
│       └── ui_runs/
├── Datasets/                    # Original training data
├── results/ (legacy)            # Legacy results directory
├── run.py                       # Main launcher
└── venv/                        # Python virtual environment
```

## Quick Start

### 1. Launch the UI
```bash
python run.py
```
Open the URL shown in terminal (typically http://localhost:7860)

### 2. Tabs Available

**Train Tab:**
- Configure hyperparameters (epochs, batch size, learning rate, etc.)
- Click "Start Training" to train the model
- View real-time training log

**Results Tab:**
- View latest metrics (accuracy, F1, per-class performance)
- See training curves (loss and accuracy over epochs)
- View confusion matrix

**Image Prediction Tab:**
- Upload a single image
- Get real-time prediction with confidence score

**Video Monitoring Tab:**
- Upload a video file
- Set alert confidence threshold
- Process frames with real-time detection
- Download annotated video with alerts
- View alert events in table format
- Access evidence frames

## Model Performance

**Latest Run Metrics:**
- Overall Accuracy: 93.39%
- Macro F1: 0.9328
- Classes:
  - NoInteraction: P=0.968, R=1.000, F1=0.984
  - Approaching: P=0.882, R=0.968, F1=0.923
  - Touching: P=1.000, R=0.800, F1=0.889
  - Plucking: P=0.906, R=0.967, F1=0.935

## Classes Detected

1. **NoInteraction** - No interaction with plant
2. **Approaching** - Moving toward the plant
3. **Touching** - Making contact with plant
4. **Plucking** - Attempting to pick/damage plant

## Training Data

- Total: 605 images
- Distribution:
  - NoInteraction: 150 images
  - Approaching: 155 images
  - Touching: 150 images
  - Plucking: 150 images

## Technical details

- **Model**: ResNet18 (pretrained on ImageNet)
- **Input Size**: 224x224
- **Optimization**: AdamW with ReduceLROnPlateau scheduler
- **Loss**: CrossEntropyLoss with label smoothing
- **Class Balancing**: Weighted random sampler
- **Validation**: Stratified train/val split (80/20)
- **Device**: CUDA GPU (falls back to CPU)

## Scope Alignment

This prototype implements the AI-based abnormal activity detection module of the broader Digital Park Guide Training Platform. See [SCOPE_ALIGNMENT.md](docs/SCOPE_ALIGNMENT.md) for mapping to full project scope.

## Requirements

- Python 3.14+
- PyTorch with CUDA support (GPU recommended)
- gradio, opencv-python, torchvision, matplotlib, numpy
- See venv/... for full dependency list

## File Organization Benefits

- **src/** - All executable code in one place
- **models/** - Centralized checkpoint storage
- **data/** - Clean separation of datasets and samples
- **docs/** - Documentation and project scope
- **output/** - All results neatly organized
- **Datasets/** - Original training data (outside src)
