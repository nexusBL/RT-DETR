# 🚦 RT-DETR Traffic Sign Detection

Real-time traffic sign detection using **RT-DETR (Real-Time Detection Transformer)** — detects 31 classes of traffic signs via live webcam with 93.4% mAP50 accuracy.

---

## 📊 Model Performance

| Version | mAP50 | Precision | Recall | mAP50-95 | Notes |
|---------|-------|-----------|--------|----------|-------|
| traffic_v15 | 91.7% | 90.8% | 94.2% | 78.1% | Baseline — 50 epochs |
| traffic_final | 92.6% | 90.6% | 94.3% | 63.9% | Full dataset — 100 epochs |
| traffic_v2 | **93.4%** | **94.2%** | **92.7%** | **67.4%** | Extra data + weak class fix |
| traffic_v3 | 🔄 Training... | — | — | — | fliplr fix + scale augmentation |

> **Best model:** `runs/traffic_v2/weights/best.pt`

---

## 🎯 Detected Classes (31 Total)

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | Road narrows on right | 16 | Overtaking by trucks is prohibited |
| 1 | 50 mph speed limit | 17 | Pedestrian Crossing |
| 2 | Attention Please | 18 | Round About |
| 3 | Beware of children | 19 | Slippery Road Ahead |
| 4 | Cycle Route Ahead Warning | 20 | Speed Limit 20 KMPh |
| 5 | Dangerous Left Curve Ahead | 21 | Speed Limit 30 KMPh |
| 6 | Dangerous Right Curve Ahead | 22 | Stop Sign ✅ |
| 7 | End of all speed and passing limits | 23 | Straight Ahead Only |
| 8 | Give Way | 24 | Traffic Signal |
| 9 | Go Straight or Turn Right | 25 | Truck traffic is prohibited |
| 10 | Go Straight or Turn Left | 26 | Turn Left Ahead ✅ |
| 11 | Keep Left | 27 | Turn Right Ahead ✅ |
| 12 | Keep Right | 28 | Uneven Road |
| 13 | Left Zig Zag Traffic | 29 | Green Light ✅ |
| 14 | No Entry | 30 | Red Light ✅ |
| 15 | No Over Taking | | |

---

## 🗂️ Dataset

| Split | Images | Notes |
|-------|--------|-------|
| Train | 21,486 | + grayscale augmentation + 44,790 small-scale copies |
| Validation | 2,578 | Used for mAP scoring |
| Test | 1,275 | Held-out evaluation |
| **Total** | **25,339** | **31 classes** |

### Source Datasets (merged from 5 sources)
- [Traffic and Road Signs v1 — Roboflow](https://universe.roboflow.com)
- [Road Signs v3 — Roboflow](https://universe.roboflow.com)
- [Traffic and Road Signs — usmanchaudhry622](https://universe.roboflow.com/usmanchaudhry622-gmail-com/traffic-and-road-signs)
- Detection Thesis 50-images — Roboflow
- Traffic Sign Detection v11 — Roboflow

### Augmentation Strategy
| Technique | Value | Purpose |
|-----------|-------|---------|
| Grayscale copies | 30% of train | Detect signs on any background color |
| HSV saturation shift | 0.9 | Handles different lighting |
| HSV value shift | 0.5 | Bright / dark environments |
| Mosaic | 1.0 | Combines 4 images per batch |
| Copy-paste | 0.1 | Signs on random backgrounds |
| Random erasing | 0.4 | Partial sign occlusion |
| Small-scale synthesis | ×3 scales (8%, 15%, 30%) | Far-distance detection |
| `fliplr` | **0.0** | Disabled — prevents Left/Right label confusion |

---

## 🖥️ Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU (8188 MiB) |
| Framework | PyTorch 2.5.1 + CUDA 12.1 |
| Library | Ultralytics 8.3.207 |
| Python | 3.11.4 |
| OS | Windows |

---

## ⚙️ Installation

```bash
# Install Python dependencies
python -m pip install ultralytics
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install opencv-python
```

---

## 🚀 Quick Start

### Run Live Webcam Detection

```bash
python webcam_v2.py
```

**Keyboard controls:**

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save screenshot |
| `+` / `=` | Increase confidence threshold |
| `-` | Decrease confidence threshold |
| `G` | Toggle grayscale display mode |

### Run Training

```bash
# Step 1 — add grayscale augmentation
python add_grayscale.py

# Step 2 — train model
python train_v3.py
```

### Analyse Dataset
```bash
python check_classes_count.py   # class distribution
python bbox_analysis.py         # bounding box size analysis
```

---

## 📁 Project Structure

```
RT-DETR/
│
├── MERGED/                         # Final merged training dataset
│   ├── images/
│   │   ├── train/                  # 21,486 training images
│   │   ├── valid/                  # 2,578 validation images
│   │   └── test/                   # 1,275 test images
│   ├── labels/
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   └── data.yaml                   # Dataset config — 31 classes
│
├── runs/
│   ├── traffic_v15/weights/        # Baseline model  (91.7% mAP50)
│   ├── traffic_final/weights/      # V2 model        (92.6% mAP50)
│   ├── traffic_v2/weights/         # Best model ⭐   (93.4% mAP50)
│   └── traffic_v3/weights/         # Current training (in progress)
│
├── extra/                          # Extra datasets downloaded
│   ├── Detection_Thesis-50-images.v1-50-images.yolov8/
│   └── traffic-sign-detection-yolov8.v11i.yolov8/
│
├── backups/                        # Auto-saved best.pt backups
├── BEST_MODEL.pt                   # Always points to latest best model
│
├── train_v3.py                     # Current training script
├── webcam_v2.py                    # Live webcam detection
├── add_grayscale.py                # Grayscale augmentation
├── fix_scale.py                    # Small-scale image synthesis
├── boost_weak.py                   # Boost under-represented classes
├── merge_extra.py                  # Merge new datasets into MERGED/
├── check_classes_count.py          # Dataset class distribution
├── bbox_analysis.py                # Bounding box size analysis
└── collect_data.py                 # Collect your own webcam training data
```

---

## 🔬 Key Technical Findings

### Problem: Left/Right Sign Confusion
- **Cause:** `fliplr=0.5` horizontally flips Turn Left signs during training, making them look like Turn Right — but the label stays as Turn Left. Model learns wrong associations.
- **Fix:** Set `fliplr=0.0` in training config (implemented in traffic_v3).

### Problem: Far-Distance Detection Failure
- **Cause:** 64.7% of training bounding boxes occupied more than 15% of the image — model trained almost exclusively on close-up signs.
- **Fix:** `fix_scale.py` generated 44,790 synthetic images with signs at 8%, 15%, and 30% of image size.

### Problem: Wrong Detection on Colored Backgrounds
- **Cause:** Model learned color patterns, not sign shapes.
- **Fix:** Grayscale augmentation (30% of training data) + high HSV saturation shifts.

### Problem: Class Imbalance
- **Cause:** No Over Taking had 0 training images; Traffic Signal had only 76.
- **Fix:** Targeted dataset collection + class duplication (4x–10x for weakest classes).

---

## 📈 Training Configuration (traffic_v3)

```python
model.train(
    data          = 'MERGED/data.yaml',
    epochs        = 60,
    imgsz         = 640,
    batch         = 8,
    optimizer     = 'AdamW',
    lr0           = 0.00001,       # Low LR — fine-tuning already-trained model
    amp           = True,          # Mixed precision for RTX 4070
    fliplr        = 0.0,           # CRITICAL — prevents left/right confusion
    hsv_s         = 0.9,           # Color invariance
    hsv_v         = 0.5,
    mosaic        = 1.0,
    copy_paste    = 0.1,
    erasing       = 0.4,
    scale         = 0.5,
    patience      = 20,
)
```

---

## 🔮 Future Work

- [ ] Collect real webcam photos of person-held signs (50–100 images per class)
- [ ] Expand Speed Limit 30 dataset (currently only 135 images, mAP50 = 20.8%)
- [ ] Expand Traffic Signal dataset (currently 468 images, mAP50 = 70.4%)
- [ ] Evaluate traffic_v3 far-distance detection improvements
- [ ] Add support for video file inference
- [ ] Export model to ONNX for deployment on edge devices

---

## 📄 License

This project uses publicly available datasets from [Roboflow Universe](https://universe.roboflow.com). Model weights are for research and educational use only.
