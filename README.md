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

## 🧠 Pipeline Overview

The dataset processing pipeline is fully automated and consists of the following steps:

1. **XML Parsing**
   - LIDC XML annotation files are parsed
   - Patient ID and SeriesInstanceUID are extracted
   - Slice-level contours and metadata are stored in JSON format

2. **Annotation Serialization**
   - XML annotations are converted into structured JSON files
   - Each JSON file contains:
     - patient ID
     - annotated CT series UID
     - nodule-wise slice contours

3. **CT Series Resolution**
   - The pipeline checks whether the annotated CT series exists locally
   - Missing CT series are automatically downloaded from TCIA

4. **Mask Generation**
   - Polygon contours are rasterized into binary masks
   - Slice alignment is done using SOPInstanceUID
   - Final output is a 3D tumor mask per patient

This design ensures correctness, reproducibility, and scalability across multiple patients.


## 🎯 Project Goals

* Download and manage lung CT datasets from TCIA
* Perform preprocessing on CT scan data
* Build a deep learning model for lung tumor segmentation
* Evaluate model performance using appropriate metrics
* Extend to inference APIs and deployment

---

## 🧩 Current Progress

✅ TCIA CT series downloader  
✅ LIDC XML annotation parsing  
✅ Structured JSON annotation format  
✅ Automatic CT series resolution  
✅ Tumor mask generation (3D)  
✅ CT + mask visualization and validation  

🚧 Dataset class for training  
🚧 Model architecture (U-Net / variants)  
🚧 Training pipeline  
🚧 Evaluation metrics  
🚧 Inference and deployment

---

## 📂 Project Structure

```
lung-tumor-segmentation/
│
├── configs/
│ └── config.py # Centralized configuration
│
├── data/
│   ├── raw/                # Downloaded CT series (DICOM)
│   ├── annotations/        # XML + parsed JSON annotations
│   ├── masks/              # Generated 3D tumor masks (.npy)
│
├── notebooks/
│ ├── archive/ # Old dataset experiments
│ └── 02_lidc_notebook.ipynb # LIDC data exploration
│
├── scripts/
│ └── lidc_downloader.py         # TCIA downloader for LIDC-IDRI
│ ├── parse_lidc_annotations.py   # XML → JSON
│ ├── json_to_mask.py             # JSON → mask (auto-download CT)
|
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

This project uses the **LIDC-IDRI (Lung Image Database Consortium Image Collection)** dataset from  
**The Cancer Imaging Archive (TCIA)**.

### Dataset Details
- **Modality:** CT
- **Format:** DICOM
- **Annotations:** XML (LIDC radiologist annotations)
- **Task:** Lung nodule (tumor) segmentation
- **Granularity:** Slice-level polygon annotations

## ⚠️ Dataset Caveats

- LIDC-IDRI annotations are provided by multiple radiologists
- Nodules may be very small and sparsely distributed
- Most CT slices do not contain tumors
- Class imbalance is significant and handled in later stages

### Important Notes
- Each patient may contain **multiple CT series**
- LIDC XML annotations correspond to **one specific SeriesInstanceUID**
- This pipeline automatically aligns annotations with the correct CT series
- Tumor nodules are often **very small (3–6 mm)** and appear in only a few slices


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

