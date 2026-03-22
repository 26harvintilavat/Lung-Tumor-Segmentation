# 🫁 LungSeg AI: End-to-End Lung Tumor Segmentation

[![Project Status: Finished](https://img.shields.io/badge/Project%20Status-Finished-brightgreen.svg)](https://github.com/26harvintilavat/Lung-Tumor-Segmentation)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com/)

**LungSeg AI** is a comprehensive, medical-grade deep learning pipeline for the automated segmentation of lung tumors from CT scans. It leverages the **LIDC-IDRI** dataset and implements an **Attention U-Net** architecture with a **2.5D spatial context** approach.

---

## 🌟 Key Features

*   **🧠 Advanced Architecture**: Implements an **Attention U-Net** (via MONAI) to focus on small, irregular lung nodules while suppressing irrelevant background features.
*   **🔬 2.5D Spatial Context**: Uses a multi-slice input strategy (3-slice stack: previous, current, next) to provide the model with local 3D spatial information in a 2D framework.
*   **⚖️ Robust Loss Function**: Employs a hybrid **Tversky + Focal Loss** specifically designed to handle extreme class imbalance and focus on hard-to-segment tumor boundaries.
*   **🖥️ Interactive Web Dashboard**: A premium Vanilla JS frontend featuring a medical-grade DICOM viewer powered by **Cornerstone.js**, providing real-time mask overlays and volumetric metrics.
*   **⚡ Automated Pipeline**: Full end-to-end automation from raw TCIA data downloading and XML annotation parsing to mask generation and model training.
*   **📏 Volumetric Analytics**: Automatically calculates tumor volume ($cm^3$), maximum diameter ($mm$), and AI confidence scores.

---

## 🛠️ Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Deep Learning** | PyTorch, MONAI, Torchvision |
| **Data Processing** | NumPy, SimpleITK, Pydicom, SciPy, Pandas |
| **Backend API** | FastAPI, Uvicorn, Python-Multipart |
| **Frontend UI** | Vanilla JS (ES6+), CSS Grid/Flexbox, HTML5 Canvas |
| **Medical Imaging** | Cornerstone.js, dicomParser, TCIA-utils |

---

## 📂 Project Structure

```
Lung-Tumor-Segmentation/
├── api/                    # FastAPI backend implementation
│   └── main.py             # Inference API and model serving
├── checkpoints/            # Saved model weights (.pth)
├── configs/                # Hyperparameter & path configurations
├── data/                   # Dataset storage (Raw, Masks, Annotations)
├── evaluation/             # Metrics and evaluation scripts
├── frontend/               # Web dashboard (SPA)
│   ├── index.html          # Modern UI layout
│   ├── app.js              # Frontend logic & Cornerstone integration
│   └── style.css           # Premium aesthetics & dark mode
├── models/                 # Model architecture definitions
├── notebooks/              # Exploration & LIDC analysis
├── scripts/                # Pipeline automation scripts
│   ├── train.py            # Model training & validation
│   ├── lidc_downloader.py  # TCIA dataset downloader
│   └── evaluate.py         # Performance assessment
└── src/                    # Core library code
    ├── model.py            # Attention U-Net implementation
    ├── dataset.py          # Medical dataset loaders
    ├── losses.py           # Tversky & Focal loss definitions
    └── preprocessing.py    # HU conversion & volume resampling
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/26harvintilavat/Lung-Tumor-Segmentation.git
cd Lung-Tumor-Segmentation
pip install -r requirements.txt
```

### 2. Data Preparation
Download and preprocess the LIDC-IDRI dataset:
```bash
python scripts/lidc_downloader.py
python scripts/parse_lidc_annotations.py
python scripts/prepare_dataloaders.py
```

### 3. Training
Train the Attention U-Net model:
```bash
python scripts/train.py
```

### 4. Running the Dashboard
Launch the backend and open the frontend:
```bash
# Terminal 1: Backend
uvicorn api.main:app --reload

# Terminal 2: Frontend
cd frontend
python -m http.server 3000
```
Visit `http://localhost:3000` to interact with the AI.

---

## 📊 Methodology

### 2.5D Training Approach
Unlike standard 2D U-Nets, our model receives **three consecutive slices** as input. This allows the network to learn the "depth" of the tumor, significantly reducing false positives caused by circular structures like blood vessels or bronchi that lack the vertical consistency of a tumor.

### Hybrid Loss Strategy
We use a weighted combination of:
*   **Tversky Loss**: Maximizes the Dice coefficient while allowing us to penalize False Negatives (missed tumors) more heavily than False Positives.
*   **Focal Loss**: Down-weights "easy" background pixels and focuses the model's attention on the difficult "hard" pixels at the tumor boundaries.

---

## 🤝 Contributions

Developed by **Harvin Tilavat**. Contributions, issues, and feature requests are welcome!

---

> [!NOTE]
> This project is intended for research and educational purposes. Always consult a medical professional for diagnostic interpretations.
