"""Pipeline preprocessing terkoreksi untuk replikasi paper (Tugas #2).

Perbedaan kunci vs paper (lihat notebooks/AUDIT_METODOLOGI.md):
- Split dilakukan LEBIH DULU, sebelum semua langkah yang mengestimasi
  parameter dari data (encoding kolom, seleksi fitur chi2, scaling).
- Semua estimator (OneHotEncoder, SelectKBest, StandardScaler) di-fit
  HANYA pada data train, lalu diterapkan (transform) ke train dan test.
- SMOTE hanya diterapkan pada data train (paper juga menyatakan hal ini
  secara eksplisit, jadi bukan koreksi, hanya verifikasi implementasi).
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2

NUMERIC_FEATURES = [
    "Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]
RAW_CODE_FEATURES = ["OperatingSystems", "Browser", "Region", "TrafficType"]
BINARY_FEATURE = "Weekend"
ONEHOT_FEATURES = ["Month", "VisitorType"]
TARGET = "Revenue"

RANDOM_STATE = 42
TEST_SIZE = 0.3
CHI2_K = 20


def split_raw(df: pd.DataFrame):
    """Split dataset mentah 70/30, stratified by target. Dilakukan SEBELUM preprocessing apa pun."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), \
        y_train.reset_index(drop=True), y_test.reset_index(drop=True)


def encode_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """One-hot encode Month & VisitorType (fit di train saja); fitur lain dipakai apa adanya.

    Menggunakan OneHotEncoder(handle_unknown='ignore') agar kategori yang mungkin hanya
    muncul di salah satu split tidak menyebabkan error, dan agar kolom hasil encoding
    konsisten antara train dan test tanpa perlu melihat isi test saat fit.
    """
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[ONEHOT_FEATURES])

    train_ohe = pd.DataFrame(
        encoder.transform(X_train[ONEHOT_FEATURES]),
        columns=encoder.get_feature_names_out(ONEHOT_FEATURES),
        index=X_train.index,
    )
    test_ohe = pd.DataFrame(
        encoder.transform(X_test[ONEHOT_FEATURES]),
        columns=encoder.get_feature_names_out(ONEHOT_FEATURES),
        index=X_test.index,
    )

    passthrough_cols = NUMERIC_FEATURES + RAW_CODE_FEATURES + [BINARY_FEATURE]

    train_pass = X_train[passthrough_cols].copy()
    test_pass = X_test[passthrough_cols].copy()
    train_pass[BINARY_FEATURE] = train_pass[BINARY_FEATURE].astype(int)
    test_pass[BINARY_FEATURE] = test_pass[BINARY_FEATURE].astype(int)

    X_train_enc = pd.concat([train_pass.reset_index(drop=True), train_ohe.reset_index(drop=True)], axis=1)
    X_test_enc = pd.concat([test_pass.reset_index(drop=True), test_ohe.reset_index(drop=True)], axis=1)

    assert list(X_train_enc.columns) == list(X_test_enc.columns), "Kolom train dan test harus identik"
    return X_train_enc, X_test_enc, encoder


def select_features_chi2(X_train_enc: pd.DataFrame, y_train: pd.Series, X_test_enc: pd.DataFrame, k: int = CHI2_K):
    """Seleksi fitur chi-square, fit HANYA pada data train (mencegah label test bocor ke seleksi fitur).

    Chi-square dihitung SEBELUM standard scaling (asumsi replikasi, lihat AUDIT_METODOLOGI.md poin 3),
    karena chi2 membutuhkan nilai non-negatif dan seluruh fitur hasil encoding di sini memang
    non-negatif secara alami (fitur numerik asli, kode kategori, dan dummy 0/1).
    """
    assert (X_train_enc.values >= 0).all(), "Semua fitur harus non-negatif untuk chi2"

    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(X_train_enc, y_train)

    selected_cols = X_train_enc.columns[selector.get_support()].tolist()
    scores = pd.Series(selector.scores_, index=X_train_enc.columns).sort_values(ascending=False)

    X_train_sel = X_train_enc[selected_cols].copy()
    X_test_sel = X_test_enc[selected_cols].copy()
    return X_train_sel, X_test_sel, selected_cols, scores


def scale_numeric(X_train_sel: pd.DataFrame, X_test_sel: pd.DataFrame):
    """Standard scaling pada fitur numerik yang lolos seleksi, fit HANYA di train."""
    numeric_cols_present = [c for c in NUMERIC_FEATURES if c in X_train_sel.columns]

    scaler = StandardScaler()
    X_train_scaled = X_train_sel.copy()
    X_test_scaled = X_test_sel.copy()

    if numeric_cols_present:
        scaler.fit(X_train_sel[numeric_cols_present])
        X_train_scaled[numeric_cols_present] = scaler.transform(X_train_sel[numeric_cols_present])
        X_test_scaled[numeric_cols_present] = scaler.transform(X_test_sel[numeric_cols_present])

    return X_train_scaled, X_test_scaled, scaler, numeric_cols_present


def save_processed(X_train, X_test, y_train, y_test, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    X_train.to_csv(os.path.join(out_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(out_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(out_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(out_dir, "y_test.csv"), index=False)
