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
│                    # 06_pipeline_improvements (Tugas #3),
│                    # 07_data_centric (Tugas #4: imbalance, Cleanlab, SHAP/LIME)
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

## Reproduksi Tugas #4 (Data-Centric Improvement and Model Interpretation)

Semua eksperimen Tugas #4 ada di `notebooks/07_data_centric.ipynb`, dengan helper di `src/data_centric.py`.
Notebook ini memuat data, melakukan split, dan menjalankan semua langkah sendiri. Notebook Tugas #2/#3 tidak perlu
dijalankan lebih dulu, cukup dataset di `data/` (diunduh otomatis kalau belum ada).

```bash
source .venv/bin/activate
cd notebooks
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1500 07_data_centric.ipynb
```

Waktu run sekitar 30 detik di Apple M2 (8 core). Environment yang dipakai untuk hasil di laporan: Python
3.11.16, scikit-learn 1.9.0, xgboost 3.2.0, imbalanced-learn 0.14.2, pandas 3.0.5, numpy 2.4.6, cleanlab 2.9.0,
shap 0.51.0, lime 0.2.0.1.

Setup yang dijaga tetap sama dengan Tugas #3:

- Split 70/30 stratified (`random_state=42`). Semua CV hanya di data train.
- `StratifiedKFold(5, shuffle=True, random_state=42)`, SMOTE `random_state=42`, XGBoost `max_depth=2`.
- Notebook berisi dua sanity check (`assert`): MCC CV D0 = 0,6257 dan MCC test D0 = 0,5955 (sama dengan Tugas #3).
- Test set hanya dipakai di Bagian 8 dan tidak diubah.

Output utama (semua di `results/`):

| File | Isi |
|---|---|
| `tables/tugas4_data_quality_issues.csv` | 7 masalah kualitas data, termasuk yang tidak diubah |
| `tables/tugas4_imbalance_comparison.csv` | Tanpa treatment vs `scale_pos_weight` vs SMOTE |
| `tables/tugas4_cleanlab_review_top20.csv`, `tugas4_cleanlab_counts.csv` | Review label Cleanlab dan jumlah per keputusan |
| `tables/tugas4_removed_observations.csv` | Dokumentasi 2 observasi train yang dihapus |
| `tables/tugas4_d0_d3_comparison.csv`, `tugas4_smote_seed_sensitivity.csv` | Eksperimen D0 sampai D3 (CV) |
| `tables/tugas4_shap_*.csv`, `figures/tugas4_shap_*.png`, `figures/tugas4_lime_fp_fn.png` | Interpretasi SHAP dan LIME |
| `tables/tugas4_final_test_evaluation.csv`, `tugas4_final_test_seed_sensitivity.csv` | Evaluasi final test set D0 vs D2 |

Catatan: Cleanlab dipanggil dengan `n_jobs=1`. Multiprocessing default Cleanlab bisa macet kalau kode dijalankan
dari stdin atau di luar `if __name__ == '__main__':`.

## Lisensi

Untuk keperluan akademik.
