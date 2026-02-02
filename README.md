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

✅ TCIA dataset ingestion pipeline (modular)
✅ Partial dataset download support (single or multiple CT series)
✅ Metadata logging for reproducibility

🚧 DICOM preprocessing (in progress)
🚧 CT slice visualization
🚧 Dataset class for training
🚧 Model training (U-Net / variants)
🚧 Evaluation and metrics

---

## 📂 Project Structure

```
lung-tumor-segmentation/
│
├── data/
│   |── lung_data/            # Downloaded data (gitignored)
|   ├── config.py
|   ├── download_lung_data.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

* **Source:** The Cancer Imaging Archive (TCIA)
* **Collection:** NSCLC-Radiomics
* **Modality:** CT
* **Data Type:** DICOM

The dataset downloader is configurable to fetch a **small subset for testing** or a **larger subset for training**, enabling safe and incremental experimentation.

---

## ▶️ How to Run (Current Stage)

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Configure dataset settings in `config.py`:

   ```python
   NUM_SERIES_TO_DOWNLOAD = 20 # increase later
   ```

3. Run the downloader:

   ```bash
   python data/download_lung_data.py
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
