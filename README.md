# Lung Tumor Segmentation using U-Net (LIDC-IDRI)

This project focuses on **lung tumor (nodule) segmentation from CT scans** using a **U-Net architecture**.  
A **subset of the LIDC-IDRI dataset** is used, and the codebase is designed with a **clean, modular structure** suitable for reproducible experimentation and future deployment.

Training and data processing are intended to be executed on **Google Colab**, while this repository contains **only code** (no datasets or model weights).

---

## 📌 Project Highlights
- Medical image segmentation using **U-Net**
- CT scan preprocessing (DICOM-based)
- Modular, production-style project structure
- Colab-friendly execution
- Clean GitHub repository (code-only)

## 📊 Dataset
- **Dataset**: LIDC-IDRI (Lung Image Database Consortium)
- **Source**: The Cancer Imaging Archive (TCIA)
- **Usage**: A **small subset of CT series** is downloaded programmatically
- **Format**: DICOM CT scans

> ⚠️ The dataset is **not included** in this repository.

---

## ⬇️ Download Dataset (Run in Google Colab)

```bash
python scripts/download_lidc_subset.py