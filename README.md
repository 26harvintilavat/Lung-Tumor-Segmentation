# 🫁 Lung Tumor Segmentation (End-to-End)

## 🚧 Project Status: Under Construction

> **This project is currently in active development.**
> The repository is being built step by step as part of an end-to-end medical image segmentation pipeline.
> New modules (preprocessing, modeling, training, and evaluation) will be added incrementally.

---

## 📌 Overview

This repository aims to build an **end-to-end lung tumor segmentation pipeline** using **CT scans from The Cancer Imaging Archive (TCIA)**. The project covers the complete workflow starting from **data ingestion** to **model training and evaluation** using deep learning.

The focus is on building a **clean, modular, and reproducible pipeline**, similar to what is expected in real-world medical imaging research and machine learning engineering.

---

## 🎯 Project Goals

* Download and manage lung CT datasets from TCIA
* Perform preprocessing on CT scan data
* Build a deep learning model for lung tumor segmentation
* Evaluate model performance using appropriate metrics
* Extend to inference APIs and deployment

---

## 🧩 Current Progress

✅ LIDC-IDRI dataset selection  
✅ TCIA dataset downloader (robust & resumable)  
✅ Raw CT series download and verification  
✅ DICOM loading using pydicom  
✅ Slice ordering and 3D volume construction  
✅ Hounsfield Unit (HU) conversion  
✅ Lung windowing and normalization  
✅ Dataset abstraction for CT volumes  

🚧 Annotation parsing (LIDC XML)  
🚧 Segmentation mask generation  
🚧 Model training (U-Net / variants)  
🚧 Evaluation and metrics  

---

## 📂 Project Structure

```
lung-tumor-segmentation/
│
├── configs/
│ └── config.py # Centralized configuration
│
├── data/
│ └── raw/lung_data/ # Downloaded LIDC-IDRI CT scans (gitignored)
│ ├── LIDC-IDRI-0001/
│ ├── LIDC-IDRI-0005/
│ └── download_log.json
│
├── notebooks/
│ ├── archive/ # Old dataset experiments
│ └── 02_lidc_notebook.ipynb # LIDC data exploration
│
├── scripts/
│ └── lidc_downloader.py # TCIA downloader for LIDC-IDRI
│
├── src/
│ ├── preprocessing.py # HU conversion & windowing
│ └── dataset.py # CT dataset abstraction
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

* **Source:** The Cancer Imaging Archive (TCIA)
* **Collection:** LIDC-IDRI (Lung Image Dataset Consortium)
* **Modality:** CT
* **Data Type:** DICOM (.dcm)

At the current stage, the project uses **raw CT scan series only**.
Segmentation masks are **not yet generated** and will be derived from LIDC
annotations in a later phase of the project.

---

## ▶️ How to Run (Current Stage)

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Download LIDC-IDRI CT data:

   ```
   python scripts/lidc_downloader.py --num-series 2
   ```

---

## 🛠 Technologies Used

* Python
* Pandas
* TCIA Utils (`tcia-utils`)
* Medical Imaging (DICOM)
* Deep Learning (planned: PyTorch)

---

## 🧠 Learning Objectives

This project is also a **learning-focused implementation**, emphasizing:

* Real-world dataset handling
* Defensive programming for unstable APIs
* Modular ML pipeline design
* Medical image segmentation workflows

---

## 🚀 Roadmap (Upcoming)

*

---

## ⚠️ Disclaimer

This project is **under active development**. Code structure, APIs, and implementations may change as the pipeline evolves.

---

## 🤝 Contributions

Suggestions, issues, and discussions are welcome. Since this project is still evolving, feedback is highly appreciated.

---

## 📬 Author

**Harvin Tilavat**
(Computer Science | Medical Imaging | Machine Learning)

---

> ⭐ If you find this project interesting, consider starring the repository and following the progress!

