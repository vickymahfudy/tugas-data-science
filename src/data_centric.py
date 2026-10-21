"""Helper untuk Tugas #4 (Data-Centric Improvement and Model Interpretation).

Memakai ulang pipeline Tugas #3 (CorrectedPreprocessor -> SMOTE -> XGBoost max_depth=2)
tanpa mengubah model maupun preprocessing. Yang berubah di Tugas #4 hanya DATA train
(imbalance treatment, koreksi label hasil review Cleanlab), sehingga setiap selisih
performa dapat diatributkan ke perubahan data.

Prinsip yang dijaga:
- Semua CV hanya di data train hasil split_raw() (70/30, stratified, random_state=42).
- Fold CV identik dengan Tugas #3: StratifiedKFold(5, shuffle=True, random_state=42).
- SMOTE dan semua preprocessing berada di dalam pipeline, jadi di-fit ulang per fold.
- Test set tidak disentuh sampai evaluasi akhir.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from pipeline_experiments import SCORING as T3_SCORING, RANDOM_STATE

CV_FOLDS = 5

# Primary metric tetap MCC (dikunci sejak Tugas #3). Macro F1 ditambahkan sebagai kolom
# pendamping karena dipakai pada contoh tabel di brief Tugas #4.
SCORING = dict(T3_SCORING)
SCORING["Macro F1"] = make_scorer(f1_score, average="macro", zero_division=0)

PRIMARY_METRIC = "MCC"


def get_cv(random_state=RANDOM_STATE):
    """Splitter CV yang sama persis dengan Tugas #3."""
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)


def cv_evaluate(pipeline, X, y, cv=None):
    """Jalankan CV dan kembalikan (summary mean/std per metrik, skor per fold).

    summary: DataFrame index=metrik, kolom Mean dan Std (std populasi, sama dengan Tugas #3).
    per_fold: DataFrame baris=fold, kolom=metrik.
    """
    cv = cv or get_cv()
    res = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=1, error_score="raise")
    per_fold = pd.DataFrame({m: res[f"test_{m}"] for m in SCORING})
    per_fold.index = [f"Fold {i + 1}" for i in range(len(per_fold))]
    summary = pd.DataFrame({"Mean": per_fold.mean().round(4), "Std": per_fold.std(ddof=0).round(4)})
    return summary, per_fold


def summary_row(summary, name):
    """Ubah summary (Mean/Std) menjadi satu baris 'mean ± std' untuk tabel perbandingan."""
    return pd.Series(
        {m: f"{summary.loc[m, 'Mean']:.4f} ± {summary.loc[m, 'Std']:.4f}" for m in summary.index},
        name=name,
    )


def neg_pos_ratio(y):
    """n_negatif / n_positif, dipakai sebagai scale_pos_weight XGBoost."""
    y = np.asarray(y)
    return float((y == 0).sum() / (y == 1).sum())
