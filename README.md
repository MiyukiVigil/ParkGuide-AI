# Sarawak Park Guide - AI Activity Detection Module

This module trains, evaluates, and runs a YOLO activity detection model for park guide monitoring.

## Current Classes

The current dataset has **4 classes**:

| Class ID | Class | Alert Type |
|---|---|---|
| 0 | `plant_plucking` | Violation |
| 1 | `animal_touching` | Violation |
| 2 | `plant_approaching` | Risk |
| 3 | `normal_activity` | No alert |


## Project Structure

```text
ParkGuide-AI/
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── alerts/
├── latest_training/
│   ├── best.pt
│   └── v2_results/
│       ├── results.png
│       └── confusion_matrix.png
├── runs/
│   └── train/
│       └── park_activity_yolo11m/
│           ├── weights/
│           │   └── best.pt
│           ├── results.png
│           └── confusion_matrix.png
├── dataset.yaml
├── training.py
├── evaluate.py
├── detect.py
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Dataset is on google drive
Link: https://drive.google.com/file/d/1oFQGFHFl1LNuRAC3OGkw7rPSsRJGbg1Z/view?usp=sharing

The dataset is configured in `dataset.yaml`:

```yaml
path: dataset

train: images/train
val: images/val

names:
  0: plant_plucking
  1: animal_touching
  2: plant_approaching
  3: normal_activity
```

Dataset folder structure:

```text
dataset/images/train/
dataset/images/val/
dataset/labels/train/
dataset/labels/val/
```

Each image needs a matching YOLO `.txt` label file with the same filename.

YOLO label format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Example:

```text
0 0.512 0.430 0.280 0.350
```

Class `0` is `plant_plucking`.

## Training

Run:

```bash
python3 training.py
```

Current training settings:

```python
MODEL_BASE = "yolo11m.pt"
EPOCHS = 120
IMG_SIZE = 768
BATCH_SIZE = -1
RUN_NAME = "park_activity_yolo11m"
```

Best model output:

```text
runs/train/park_activity_yolo11m/weights/best.pt
latest_training/best.pt
```

## Evaluation

Run:

```bash
python3 evaluate.py
```

## Detection

Run detection on one uploaded image or video file:

```bash
python3 detect.py --source path/to/image.jpg
```

To test a video file:

```bash
python3 detect.py --source path/to/video.mp4
```

To change confidence:

```bash
python3 detect.py --source path/to/image.jpg --confidence 0.50
```

Annotated results are saved under:

```text
runs/detect/
```

Admin violation alerts are saved to:

```text
alerts/alert_log.txt
```

Risk detections are printed in the terminal for the park guide only.
They are not written to the admin alert log since it is just a risk.

Alert behavior:

| Class | Output Behavior |
|---|---|
| `plant_approaching` | Risk notice for park guide only |
| `plant_plucking` | Violation alert sent to admin log |
| `animal_touching` | Violation alert sent to admin log |
| `normal_activity` | No alert |

## For Teammates

To use this project:

```bash
git clone <repo-url>
cd ParkGuide-AI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 training.py
python3 evaluate.py
python3 detect.py
```

If the trained `best.pt` is shared in the repo, skip training and run `evaluate.py` or `detect.py` directly.

`detect.py` and `evaluate.py` are linked to `latest_training/best.pt`. After training, `training.py` copies the newest best weights there automatically.

`last.pt` is only for resuming training if needed.



