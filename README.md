# 🫁 LungSeg AI — Lung Tumor Segmentation

Automated lung tumor segmentation from CT scans using an Attention U-Net deep learning architecture, trained on the LIDC-IDRI dataset. Includes a full-stack web interface for uploading DICOM scans and visualising predictions.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green)
![Val Dice](https://img.shields.io/badge/Val%20Dice-0.7545-brightgreen)

> ⚠️ **Research use only** — not intended for clinical diagnosis or medical decision-making.

---

## 📋 Table of Contents
- [Quickstart](#quickstart)
- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API](#api)
- [Configuration](#configuration)
- [References](#references)

---

## ⚡ Quickstart

```bash
git clone https://github.com/26harvintilavat/Lung-Tumor-Segmentation.git
cd Lung-Tumor-Segmentation
python -m venv venv
pip install -r requirements.txt
# Place best_model.pth in checkpoints/ (ask the author)
```

**macOS / Linux:**
```bash
source venv/bin/activate
./start.sh
```

**Windows:**
```bat
start.bat
```

Both scripts start the API, serve the frontend, and open your browser automatically.  
Visit **http://localhost:3000/index.html** — landing page with project details.  
Visit **http://localhost:3000/tool.html** — the segmentation tool.

---

## 🔍 Overview

End-to-end pipeline for lung tumor segmentation from CT scans:

- **Input**: Raw DICOM CT scan files (uploaded as a ZIP)
- **Output**: Binary tumor segmentation masks + annotated overlays
- **Model**: Attention U-Net with 2.5D (3-slice) context window
- **API**: FastAPI REST endpoint for inference
- **Frontend**: Interactive web interface with landing page and tool
- **Performance**: Val Dice Score of 0.7545

---

## 🧠 Architecture

### Attention U-Net
- **Input channels**: 3 (previous + current + next slice — 2.5D context)
- **Output channels**: 1 (binary tumor mask)
- **Input size**: 256 × 256
- **Encoder depth**: 5 levels (32 → 64 → 128 → 256 → 512 filters)
- **Key features**:
  - Attention gates on every skip connection
  - 2.5D input strategy for volumetric context without 3D memory cost
  - Dropout (0.1) for regularisation

### Loss Function — TverskyFocal
- **Tversky loss** (α=0.3, β=0.7) — penalises False Negatives (missed tumors) more heavily
- **Focal loss** (γ=2.0) — focuses training on hard boundary pixels
- **Combined weight**: 70% Tversky + 30% Focal

---

## 📦 Dataset

**LIDC-IDRI** (Lung Image Database Consortium and Image Database Resource Initiative)
- 1,010 patients, ~240,000 CT slices
- Raw DICOM CT scans with XML annotations from 4 radiologists
- Patient-level train/val split (75/25)
- Extreme class imbalance — tumor voxels < 0.5% of total

### Preprocessing Pipeline
1. HU conversion from raw DICOM pixel values
2. Isotropic resampling to 1 mm spacing (SimpleITK)
3. Lung window clipping (−1000 to 400 HU)
4. Z-score normalisation → rescaled to [0, 1]
5. Automatic lung bounding box cropping
6. Resize to 256 × 256

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Val Dice Score | **0.7545** |
| Train Dice Score | **0.7865** |
| Val Loss | **0.1682** |
| Best Epoch | **38 / 50** |
| Architecture | Attention U-Net |
| Dataset | LIDC-IDRI |

### Training Details
| Parameter | Value |
|-----------|-------|
| Batch Size | 2 |
| Learning Rate | 3e-4 |
| Scheduler | Warmup (5 ep) + CosineAnnealing |
| Optimizer | Adam |
| Epochs | 50 |
| Early Stopping | Patience 15 |
| Image Size | 256 × 256 |
| GPU | NVIDIA RTX 3050 4 GB |

---

## 📁 Project Structure

```
Lung-Tumor-Segmentation/
│
├── frontend/
│   ├── index.html           ← Landing page (project showcase)
│   ├── tool.html            ← Segmentation tool UI
│   ├── app.js               ← Frontend logic
│   └── style.css            ← Styles
│
├── api/
│   └── main.py              ← FastAPI inference endpoint
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
│   ├── model.py             ← Attention U-Net (MONAI)
│   ├── losses.py            ← TverskyFocal Loss
│   ├── preprocessing.py     ← CT preprocessing utilities
│   ├── dataset.py           ← Base dataset
│   └── train_dataset.py     ← Training dataset
│
├── checkpoints/             ← Saved model weights (not tracked)
├── start.sh                 ← One-command launcher
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/26harvintilavat/Lung-Tumor-Segmentation.git
cd Lung-Tumor-Segmentation
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Get Model Weights
The trained weights are not stored in the repo (file size). Contact the author for `best_model.pth` and place it at:
```
checkpoints/best_model.pth
```

---

## 🚀 Usage

### Option A — One command (recommended)

**macOS / Linux:**
```bash
./start.sh
```

**Windows:**
```bat
start.bat
```

Starts the API, serves the frontend, and opens the browser automatically.

### Option B — Manual

**Terminal 1 — API:**
```bash
source venv/bin/activate
uvicorn api.main:app --host localhost --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
python -m http.server 3000
```

Then open **http://localhost:3000/index.html**

### Training from scratch
```bash
# 1. Download LIDC-IDRI dataset
python scripts/lidc_downloader.py

# 2. Generate masks from annotations
python scripts/json_to_mask.py

# 3. Cache preprocessed data
python scripts/prepare_dataloaders.py

# 4. Train
python scripts/train.py
```

---

## 🌐 API

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API health check + model status |
| GET | `/model-info` | Architecture and training details |
| POST | `/predict` | Run segmentation on a DICOM ZIP |

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

### Swagger UI (interactive docs)
```
http://localhost:8000/docs
```

---

## 🔧 Configuration

Edit `configs/config.py` to change training parameters:
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

- LIDC-IDRI Dataset: [Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
- Attention U-Net: [Oktay et al., 2018](https://arxiv.org/abs/1804.03999)
- Tversky Loss: [Salehi et al., 2017](https://arxiv.org/abs/1706.05721)
- MONAI Framework: [monai.io](https://monai.io)

---

## 👨‍💻 Author

Built by **Harvin Tilavat** as a deep learning project for medical image segmentation.  
GitHub: [@26harvintilavat](https://github.com/26harvintilavat)
