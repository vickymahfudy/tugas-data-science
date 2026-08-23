"""Utility functions untuk memuat dataset Online Shoppers Purchasing Intention."""
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CSV_PATH = os.path.join(DATA_DIR, "online_shoppers_intention.csv")


def download_and_save():
    """Unduh dataset dari UCI ML Repository (id=468) dan simpan sebagai CSV lokal."""
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=468)
    df = dataset.data.original
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    return df


def load_data():
    """Muat dataset dari CSV lokal. Jika belum ada, unduh dulu."""
    if not os.path.exists(CSV_PATH):
        return download_and_save()
    return pd.read_csv(CSV_PATH)
