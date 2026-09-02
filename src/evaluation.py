"""Metrik evaluasi sesuai yang dilaporkan paper: Accuracy, Precision, TPR (Recall),
F1, TNR, MCC, auROC, auPR."""
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, roc_auc_score, average_precision_score,
)


def evaluate_model(y_true, y_pred, y_proba):
    """Hitung 8 metrik evaluasi paper. y_proba adalah probabilitas kelas positif (1)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tnr = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "TPR (Recall)": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "TNR": tnr,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "auROC": roc_auc_score(y_true, y_proba),
        "auPR": average_precision_score(y_true, y_proba),
    }
