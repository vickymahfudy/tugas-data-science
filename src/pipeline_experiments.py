"""Pipeline components untuk Tugas #3 (Improving the Data Science Pipeline).

Membungkus ulang logika preprocessing.py (Tugas #2) sebagai transformer sklearn
yang bisa di-fit per-fold di dalam stratified k-fold CV, supaya semua estimator
(encoder, seleksi fitur chi2, scaler) tetap hanya belajar dari fold train --
konsisten dengan koreksi anti-leakage di AUDIT_METODOLOGI.md dan requirement
Tugas #3 (fit hanya di training data).

Dipakai untuk membandingkan skenario E0 (baseline Tugas #2), E1 (Improvement A:
log-transform numerik, salah satu dari 8 kategori improvement brief -- "numerical
transformations"), E2 (Improvement B: feature pruning, kategori "removal of
irrelevant/redundant features"), E3 (kombinasi), semua lewat `cross_validate_pipeline`.

Catatan revisi (17 Sep 2026): Improvement A SEBELUMNYA berupa perubahan rasio SMOTE,
tapi itu bukan salah satu dari 8 kategori resmi brief (missing-value handling,
categorical encoding, numerical transformations, scaling/normalization, outlier
handling, feature engineering, feature selection, removal of irrelevant/redundant
features) -- SMOTE adalah teknik imbalance handling, bukan preprocessing. Diganti ke
log-transform numerik agar konsisten dengan daftar 8 opsi brief. SMOTE tetap dipakai
di semua eksperimen dengan sampling_strategy='auto' konstan (bukan lagi variabel yang
dieksplorasi).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (
    make_scorer, accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

NUMERIC_FEATURES = [
    "Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]
RAW_CODE_FEATURES = ["OperatingSystems", "Browser", "Region", "TrafficType"]
BINARY_FEATURE = "Weekend"
ONEHOT_FEATURES = ["Month", "VisitorType"]
CHI2_K = 20
RANDOM_STATE = 42

# Pasangan fitur berkorelasi tinggi (dari eda_high_correlation_pairs.csv), untuk Improvement B.
# Fitur kedua di tiap pasangan yang di-drop (dipertahankan yang skor chi2-nya lebih tinggi
# di notebook 04, lihat AUDIT_METODOLOGI.md / hasil seleksi fitur Tugas #2).
REDUNDANT_FEATURES_TO_DROP = ["ExitRates", "ProductRelated"]


class CorrectedPreprocessor(BaseEstimator, TransformerMixin):
    """Replikasi pipeline preprocessing terkoreksi Tugas #2 (encode -> [log-transform] -> chi2 ->
    scale), dibungkus jadi transformer yang fit/transform-nya kompatibel dengan cross-validation.

    Parameter
    ---------
    k : int
        Jumlah fitur yang dipilih SelectKBest(chi2).
    drop_features : list[str] atau None
        Fitur mentah yang di-drop SEBELUM encoding (untuk Improvement B: feature pruning).
        Jika None, tidak ada fitur yang di-drop (perilaku identik pipeline baseline Tugas #2).
    log_transform : bool
        Jika True, terapkan log1p() pada semua fitur numerik SEBELUM seleksi chi2 dan scaling
        (Improvement A: "numerical transformations" -- brief kategori #3 dari 8 opsi). log1p
        dipakai (bukan log biasa) karena banyak fitur numerik bernilai 0 (lihat EDA: pct_zero
        44-90% di beberapa fitur), dan semua fitur numerik di dataset ini non-negatif secara
        alami sehingga log1p selalu terdefinisi. Default False (perilaku identik baseline
        Tugas #2, tanpa transformasi tambahan).
    """

    def __init__(self, k=CHI2_K, drop_features=None, log_transform=False):
        self.k = k
        self.drop_features = drop_features
        self.log_transform = log_transform

    def _numeric_features(self):
        if self.drop_features:
            return [c for c in NUMERIC_FEATURES if c not in self.drop_features]
        return list(NUMERIC_FEATURES)

    def _encode(self, X):
        ohe_vals = self.encoder_.transform(X[ONEHOT_FEATURES])
        ohe = pd.DataFrame(
            ohe_vals, columns=self.encoder_.get_feature_names_out(ONEHOT_FEATURES), index=X.index
        )
        passthrough_cols = self._numeric_features() + RAW_CODE_FEATURES + [BINARY_FEATURE]
        pas = X[passthrough_cols].copy()
        pas[BINARY_FEATURE] = pas[BINARY_FEATURE].astype(int)
        X_enc = pd.concat([pas.reset_index(drop=True), ohe.reset_index(drop=True)], axis=1)
        if self.log_transform:
            numeric_present = [c for c in self._numeric_features() if c in X_enc.columns]
            X_enc[numeric_present] = np.log1p(X_enc[numeric_present])
        return X_enc

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        y = np.asarray(y)

        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.encoder_.fit(X[ONEHOT_FEATURES])

        X_enc = self._encode(X)

        assert (X_enc.values >= 0).all(), "Semua fitur harus non-negatif untuk chi2"
        k_eff = min(self.k, X_enc.shape[1])
        self.selector_ = SelectKBest(score_func=chi2, k=k_eff)
        self.selector_.fit(X_enc, y)
        self.selected_cols_ = X_enc.columns[self.selector_.get_support()].tolist()

        X_sel = X_enc[self.selected_cols_]
        numeric_present = self._numeric_features()
        self.numeric_cols_present_ = [c for c in numeric_present if c in self.selected_cols_]

        self.scaler_ = StandardScaler()
        if self.numeric_cols_present_:
            self.scaler_.fit(X_sel[self.numeric_cols_present_])
        return self

    def transform(self, X):
        X = X.reset_index(drop=True)
        X_enc = self._encode(X)
        X_sel = X_enc[self.selected_cols_].copy()
        if self.numeric_cols_present_:
            X_sel[self.numeric_cols_present_] = self.scaler_.transform(X_sel[self.numeric_cols_present_])
        return X_sel.values


def _tnr_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else float("nan")


SCORING = {
    "Accuracy": make_scorer(accuracy_score),
    "Precision": make_scorer(precision_score, zero_division=0),
    "TPR (Recall)": make_scorer(recall_score, zero_division=0),
    "F1-Score": make_scorer(f1_score, zero_division=0),
    "TNR": make_scorer(_tnr_score),
    "MCC": make_scorer(matthews_corrcoef),
    "auROC": make_scorer(roc_auc_score, response_method="predict_proba"),
    "auPR": make_scorer(average_precision_score, response_method="predict_proba"),
    # Weighted-average varian (kedua kelas), dipakai HANYA untuk konteks "seberapa dekat ke
    # paper" (paper diduga melapor weighted average, bukan skor kelas positif murni -- lihat
    # AUDIT_METODOLOGI.md / notebook 05). BUKAN kriteria seleksi rasio/hyperparameter/improvement
    # -- itu tetap MCC (revisi final 17 Sep 2026, setelah didiskusikan: F1-weighted sempat
    # dipakai sebagai primary metric, tapi itu bikin trade-off precision-recall sepihak (mis. E2
    # tampak "netral" di F1-weighted tapi tampak "menang" di Precision biasa) -- MCC dipilih balik
    # karena satu skalar, dihitung langsung dari confusion matrix, robust terhadap imbalance, dan
    # tidak punya varian "weighted vs positif" yang bisa dipilih-pilih sesudah lihat hasil).
    "Precision (weighted)": make_scorer(precision_score, average="weighted", zero_division=0),
    "F1-Score (weighted)": make_scorer(f1_score, average="weighted", zero_division=0),
}


def build_pipeline(model, k=CHI2_K, drop_features=None, log_transform=False, use_smote=True,
                    smote_sampling_strategy="auto", use_class_weight=False,
                    scale_pos_weight=None):
    """Rakit imblearn Pipeline: CorrectedPreprocessor -> (opsional SMOTE) -> model.

    log_transform=True mengaktifkan Improvement A (numerical transformations, log1p pada fitur
    numerik sebelum chi2+scaling). smote_sampling_strategy dibiarkan 'auto' (1:1) secara konstan
    di semua eksperimen Tugas #3 -- SMOTE bukan lagi variabel improvement yang dieksplorasi
    (lihat catatan revisi di docstring modul).

    use_class_weight=True akan set_params(class_weight='balanced') pada model sklearn
    yang mendukungnya. XGBoost TIDAK punya parameter class_weight (parameter itu diam-diam
    diabaikan oleh XGBoost native trainer, hanya muncul warning) -- untuk XGBClassifier,
    pass scale_pos_weight secara eksplisit (mis. rasio n_negatif/n_positif dari fold train)
    sebagai pengganti class_weight.
    """
    steps = [("prep", CorrectedPreprocessor(k=k, drop_features=drop_features, log_transform=log_transform))]

    model = model.__class__(**model.get_params())  # fresh clone tiap panggilan
    if use_class_weight:
        if isinstance(model, XGBClassifier):
            if scale_pos_weight is None:
                raise ValueError(
                    "XGBClassifier tidak punya class_weight; berikan scale_pos_weight eksplisit"
                )
            model.set_params(scale_pos_weight=scale_pos_weight)
        else:
            try:
                model.set_params(class_weight="balanced")
            except (ValueError, TypeError):
                raise ValueError(
                    f"Model {model.__class__.__name__} tidak mendukung class_weight maupun scale_pos_weight"
                )

    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy=smote_sampling_strategy)))

    steps.append(("clf", model))
    return ImbPipeline(steps)


def cross_validate_pipeline(pipeline, X, y, cv_folds=5, random_state=RANDOM_STATE):
    """Jalankan stratified k-fold CV, kembalikan DataFrame ringkasan (mean +- std) per metrik.

    Semua langkah pipeline (encoder, chi2, scaler, SMOTE) di-fit ulang di setiap fold train,
    tidak pernah melihat fold validasi -- konsisten dengan requirement anti-leakage Tugas #3.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        pipeline, X, y, cv=skf, scoring=SCORING, n_jobs=1, error_score="raise",
    )
    rows = []
    for metric in SCORING:
        scores = cv_results[f"test_{metric}"]
        rows.append({
            "Metric": metric,
            "Mean": round(scores.mean(), 4),
            "Std": round(scores.std(), 4),
        })
    return pd.DataFrame(rows).set_index("Metric"), cv_results
