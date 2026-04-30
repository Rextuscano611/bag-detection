# Abandoned Bag Detection System

A real-time CCTV surveillance system that detects abandoned bags using a custom-trained YOLOv8 model and intelligent tracking logic. When a person leaves a bag unattended, the system triggers a visual alert with a red bounding box.

---

## Demo

| State | Bounding Box Color | Description |
|-------|--------------------|-------------|
| Bag detected | Grey | No owner assigned yet |
| Owner assigned | Blue | Person associated with bag |
| Person walked away | Orange (WARNING) | Countdown timer started |
| Bag abandoned | Red (ABANDONED) | Alert triggered |

---

## Features

- Real-time person and bag detection using YOLOv8m
- Owner assignment — links a bag to the nearest person
- Placement detection — avoids false alerts when person is removing bag from shoulder
- Smooth bounding boxes using EMA (Exponential Moving Average) — no flickering
- Supports video files, webcam, and RTSP IP camera streams
- Configurable thresholds via command line arguments

---

## Model

- Architecture: YOLOv8m (medium)
- Classes: `bag` (0), `person` (1)
- Training dataset: ~16,600 images (merged from 3 Roboflow datasets)
- Training: 50 epochs on Google Colab T4 GPU
- Final metrics:
  - mAP50: 0.864
  - Precision: 0.915
  - Recall: 0.838
  - Bag mAP50: 0.843
  - Person mAP50: 0.886

---

## Project Structure

```
bag_detection/
├── abandoned_bag_owner_tracking.py   # main detection + tracking script
├── best.pt                           # trained YOLOv8m model weights
├── requirements.txt                  # dependencies
└── README.md
```

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Rextuscano611/bag-detection.git
cd bag-detection
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Requirements

```
ultralytics
opencv-python
```

---

## Usage

### Test on a video file
```bash
python abandoned_bag_owner_tracking.py --model best.pt --source video.mp4
```

### Test on webcam
```bash
python abandoned_bag_owner_tracking.py --model best.pt --source 0
```

### Test on IP camera (RTSP)
```bash
python abandoned_bag_owner_tracking.py --model best.pt --source "rtsp://username:password@ip:554/stream1"
```

### Save output video
```bash
python abandoned_bag_owner_tracking.py --model best.pt --source video.mp4 --save-video output.mp4
```

---

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `best.pt` | Path to YOLOv8 model weights |
| `--source` | `0` | Video file, webcam index, or RTSP URL |
| `--conf` | `0.45` | Detection confidence threshold |
| `--near-person-dist` | `80.0` | Distance (px) to consider person near bag |
| `--bag-static-sec` | `2.0` | Seconds bag must be still before timers start |
| `--bag-alone-sec` | `3.0` | Seconds bag alone before abandoned alert |
| `--resize` | `640` | Inference frame width |
| `--skip-frames` | `1` | Process every Nth frame (use 2-3 for RTSP) |
| `--save-video` | `None` | Output video save path |

---

## How It Works

```
Frame input (video / webcam / RTSP)
        ↓
YOLOv8m detects persons and bags
        ↓
Tracker assigns stable IDs to each object
        ↓
Bag confirmed after appearing in 5+ frames
        ↓
Bag must be stationary for placed_sec before logic starts
        ↓
Owner assigned — nearest person linked to bag
        ↓
If owner walks away → orange WARNING box
        ↓
If bag stays alone for bag_alone_sec → red ABANDONED box
        ↓
If person returns near bag → back to normal
```

---

## Training Details

The model was trained on a merged dataset built from:
- Custom bag detection dataset (Roboflow)
- Person detection dataset (Roboflow)
- Luggage/CCTV bag dataset (Roboflow)

Training was done on Google Colab T4 GPU using the following config:
- Model: YOLOv8m pretrained on COCO
- Epochs: 50
- Image size: 640
- Batch size: 16
- Optimizer: AdamW
- Augmentations: HSV, mosaic, flip, rotation, scale

---

## Known Limitations

- Bag detection may be less accurate on real CCTV cameras with different lighting or angles compared to training data
- Performance on CPU is limited (~4 FPS) — GPU recommended for deployment
- Model works best when camera is at a moderate angle (not extreme overhead)

---

## Future Improvements

- Add sound/email alert on abandoned bag detection
- Multi-camera support
- Fine-tune model on real CCTV footage for better deployment accuracy
- Add a web dashboard for monitoring

---

## Author

Rex — Computer Science Engineering  
| Mumbai, India