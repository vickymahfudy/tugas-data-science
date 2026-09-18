# Replikasi: Gradient Boosting Classifier for Purchase Intention

Replikasi metodologi dan hasil eksperimen dari paper:

> Abdullah-All-Tanvir, Khandokar, I. A., Islam, A.K.M.M., Islam, S., & Shatabda, S. (2023). *A gradient boosting classifier for purchase intention prediction of online shoppers*. Heliyon, 9(4), e15163. https://doi.org/10.1016/j.heliyon.2023.e15163

Project ini dikerjakan sebagai bagian dari tugas kuliah Data Science (Program Studi Magister Ilmu Komputer, Universitas Gadjah Mada).

## Dataset

Online Shoppers Purchasing Intention Dataset (UCI ML Repository, id=468), 12.330 sesi pengunjung e-commerce, 18 kolom (10 fitur numerik, 7 fitur kategorikal, 1 label `Revenue`).

- Sumber: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

## Struktur Project

```
Replikasi_Paper/
├── data/            # Dataset (diunduh otomatis dari UCI)
├── notebooks/       # Notebook per tahap: 01_download_data, 02_eda,
│                    # 03_pure_replication (replikasi murni tanpa koreksi, utk pembanding),
│                    # 04_preprocessing_split, 05_modeling_evaluation (Tugas #2, terkoreksi),
│                    # 06_pipeline_improvements (Tugas #3)
├── src/             # Kode reusable (data_utils.py, preprocessing.py, dst)
├── results/
│   ├── tables/      # Hasil tabel (CSV)
│   └── figures/     # Hasil visualisasi (PNG)
└── requirements.txt
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Di macOS, XGBoost membutuhkan runtime OpenMP:

```bash
brew install libomp
```

## Menjalankan Notebook

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_download_data.ipynb
```

## Lisensi

Untuk keperluan akademik.
