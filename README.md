# Real-Time Multi-Object Detection System
### YOLO11m · 5 Classes · ByteTrack · ONNX Export

![Demo](assets/demo.gif)

A real-time object detection pipeline built on YOLO11m, trained on a 
custom dataset collected and annotated from scratch. The system detects 
five object classes in video streams with a threaded capture pipeline, 
structured logging, and ONNX export support.

---

## Classes
`jug` · `ball` · `tennis-ball` · `pen` · `book`

---

## Results

### Ablation Study — v1 vs v2

| Version | mAP50 | mAP50-95 | Precision | Recall | Training Time |
|---------|-------|----------|-----------|--------|---------------|
| v1      | 0.914 | 0.820    | 0.825     | 0.879  | 49 min        |
| v2      | 0.942 | 0.830    | 0.899     | 0.936  | 49 min        |

**What changed between v1 and v2:**
- Augmentation: blur increased from 2.5px → 3.1px, 
  brightness from ±15% → ±43%
- Dataset split rebalanced from 86/8/6 → 60/20/20 
  (train/val/test) for more reliable validation signal
- Result: mAP50 +2.8%, Precision +7.4%, Recall +5.7%

**Key finding:** Pen detection on same-color backgrounds 
remained the hardest case (~30% confidence). Targeted 
data collection on matching backgrounds is the identified 
next iteration.

### Confusion Matrices
| v1 | v2 |
|----|----|
| ![v1 confusion matrix](benchmarks/results/v1%20confusion_matrix.png) | ![v2 confusion matrix](benchmarks/results/v2%20confusion_matrix.png) |

---

## ONNX Benchmark (CPU)

| Format  | ms/frame | FPS   |
|---------|----------|-------|
| PyTorch | 26.19    | 38.19 |
| ONNX    | 37.57    | 26.61 |

> Tested on CPU. ONNX showed higher latency than PyTorch in 
> this configuration — a known behaviour on certain CPU 
> architectures where PyTorch's optimized CPU kernels 
> outperform ONNX Runtime's default execution provider. 
> GPU inference would reverse this gap.

---

## Dataset

- **Total images:** 729 across 5 classes
- **Collection:** Photos taken manually with mobile camera 
  under varied lighting, angles, backgrounds, and occlusion
- **Annotation:** First 200 images annotated manually in 
  Roboflow. Remaining images annotated using pseudo-labeling 
  with the v1 model, then manually verified.
- **Augmentations:** Horizontal/vertical flip, 90° rotation, 
  crop, shear, saturation, brightness, exposure, blur, noise

---

## Architecture

The pipeline is structured into three components:

- **Camera** — handles `cv2.VideoCapture`, frame reading, 
  and safe release with error handling
- **Detector** — loads YOLO11m weights, runs inference, 
  draws bounding boxes and confidence scores
- **Pipeline** — wires Camera and Detector together, manages 
  the threaded capture loop, and handles logging

Logging is handled via Python's `logging` module. FPS is 
calculated as a rolling average over 30 frames and rendered 
on screen.

---

## Model Weights

Model weights (~50MB) are hosted on Google Drive due to 
GitHub's file size limits.

**Download:** [best.pt — Google Drive] https://drive.google.com/file/d/1oXDeS3hk0QflpzkZXC5DEqmbr8N2qNhS/view?usp=sharing

Place the downloaded file in the project root before running.

---

## How to Run

```bash
git clone https://github.com/zadif/CV-Capstone
cd CV-Capstone
pip install ultralytics opencv-python
python videoProcessing.py
```

To run on a specific video file, edit the `source` path 
in `videoProcessing.py`.

---

## Training

Trained on Google Colab with GPU. To retrain:

```bash
from ultralytics import YOLO
model = YOLO('yolo11m.pt')
model.train(data='data.yaml', epochs=50, imgsz=640, batch=16)
```

---

## Project Context

Built as part of a structured 5-week computer vision 
learning plan covering object detection, segmentation, 
pose estimation, optical flow, multi-object tracking, 
and real-time pipeline engineering.
