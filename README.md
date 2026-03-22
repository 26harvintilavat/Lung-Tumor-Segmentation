# 🫁 Lung Tumor Segmentation

Automated lung tumor segmentation from CT scans using
Attention U-Net deep learning architecture, trained on
the LIDC-IDRI dataset.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green)
![Val Dice](https://img.shields.io/badge/Val%20Dice-0.7545-brightgreen)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API](#api)

---

## 🔍 Overview

This project implements an end-to-end pipeline for
lung tumor segmentation from CT scans:

- **Input**: Raw DICOM CT scan files
- **Output**: Binary tumor segmentation masks
- **Model**: Attention U-Net with 3-slice context window
- **API**: FastAPI REST endpoint for inference
- **Performance**: Val Dice Score of 0.7545

---

## 🧠 Architecture

### Attention U-Net

- **Input channels**: 3 (previous + current + next slice)
- **Output channels**: 1 (binary tumor mask)
- **Input size**: 256 × 256
- **Key features**:
  - Attention gates for focusing on tumor regions
  - Skip connections for preserving spatial information
  - 3-slice context window for 3D awareness

### Loss Function

- **TverskyFocal Loss**
  - Tversky loss (α=0.3, β=0.7) for class imbalance
  - Focal loss (γ=2.0) for hard examples
  - Combined weight: 70% Tversky + 30% Focal

---

## 📦 Dataset

**LIDC-IDRI** (Lung Image Database Consortium)

- 150+ patients
- Raw DICOM CT scans
- JSON annotations converted to binary masks
- Patient-level train/val split (75/25)

### Preprocessing Pipeline

1. HU conversion from raw DICOM
2. Isotropic resampling to 1mm spacing
3. Lung window clipping (-1000 to 400 HU)
4. Z-score normalization → rescaled to [0,1]
5. Lung bounding box cropping
6. Resize to 256×256

---

## 📊 Results

| Metric           | Value           |
| ---------------- | --------------- |
| Val Dice Score   | **0.7545**      |
| Train Dice Score | **0.7865**      |
| Val Loss         | **0.1682**      |
| Best Epoch       | **38/50**       |
| Architecture     | Attention U-Net |
| Dataset          | LIDC-IDRI       |

### Training Details

| Parameter      | Value                    |
| -------------- | ------------------------ |
| Batch Size     | 2                        |
| Learning Rate  | 3e-4                     |
| Scheduler      | Warmup + CosineAnnealing |
| Optimizer      | Adam                     |
| Epochs         | 50                       |
| Early Stopping | Patience 15              |
| Image Size     | 256×256                  |
| GPU            | NVIDIA RTX 3050 4GB      |

---

## 📁 Project Structure

```
Lung-Tumor-Segmentation/
│
├── api/
│   └── main.py              ← FastAPI endpoint
│
├── configs/
│   └── config.py            ← Training configuration
│
├── data/
│   ├── raw/                 ← DICOM files (not tracked)
│   ├── masks/               ← Generated masks (not tracked)
│   ├── cache/               ← Preprocessed cache (not tracked)
│   └── annotations/         ← JSON annotations
│
├── scripts/
│   ├── train.py             ← Training script
│   ├── prepare_dataloaders.py ← Cache preprocessing
│   ├── json_to_mask.py      ← Mask generation
│   └── evaluate.py          ← Model evaluation
│
├── src/
│   ├── model.py             ← Attention U-Net
│   ├── losses.py            ← TverskyFocal Loss
│   ├── preprocessing.py     ← CT preprocessing
│   ├── dataset.py           ← Base dataset
│   └── train_dataset.py     ← Training dataset
│
├── checkpoints/             ← Saved models (not tracked)
├── test_viewer.html         ← Web UI for visualization
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/Lung-Tumor-Segmentation.git
cd Lung-Tumor-Segmentation
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1 — Convert XML Annotations to JSON

```bash
python scripts/parse_lidc_annotation.py
```

### Step 2 — Generate Masks from JSON

```bash
python scripts/json_to_mask.py
```

### Step 3 — Cache Preprocessed Data

```bash
python scripts/prepare_dataloaders.py
```

### Step 4 — Train Model

```bash
python scripts/train.py
```

### Step 5 — Evaluate Model

```bash
python scripts/evaluate.py
```

### Step 6 — Run API

```bash
python api/main.py
```

### Step 7 — Open Web UI

```bash
# New terminal
python -m http.server 3000

# Open browser
http://localhost:3000/test_viewer.html
```

---

## 🌐 API

### Endpoints

| Method | Endpoint      | Description      |
| ------ | ------------- | ---------------- |
| GET    | `/health`     | API health check |
| GET    | `/model-info` | Model details    |
| POST   | `/predict`    | Run segmentation |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@patient_dicoms.zip"
```

### Example Response

```json
{
  "status": "success",
  "total_slices": 280,
  "tumor_slices": 13,
  "tumor_slice_ids": [184, 185, 186, 187],
  "total_tumor_volume": 4523.0,
  "overlays": [
    {
      "slice_index": 184,
      "image_base64": "...",
      "tumor_pixels": 54
    }
  ]
}
```

### Swagger UI

```
http://localhost:8000/docs
```

---

## 🔧 Configuration

Edit `configs/config.py`:

```python
EPOCHS          = 50
BATCH_SIZE      = 2
LR              = 3e-4
VAL_SPLIT       = 0.25
IMG_SIZE        = 256
PATIENCE        = 15
WARMUP_EPOCHS   = 5
COSINE_EPOCHS   = 45
```

---

## 📚 References

- LIDC-IDRI Dataset: [https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
- Attention U-Net: [Oktay et al., 2018](https://arxiv.org/abs/1804.03999)
- Tversky Loss: [Salehi et al., 2017](https://arxiv.org/abs/1706.05721)

---

## 📬 Author

**Harvin Tilavat**
(Computer Science | Medical Imaging | Machine Learning)

---

> ⭐ If you find this project interesting, consider starring the repository and following the progress!
