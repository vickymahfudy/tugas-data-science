"""Definisi 5 model klasifikasi sesuai parameter yang dilaporkan paper.

Parameter diambil dari Ringkasan_Paper.md (bukan hasil tuning ulang), sesuai instruksi
brief Tugas #2: tidak melakukan hyperparameter tuning tambahan untuk sekadar meningkatkan
performa pada tahap ini.
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42


def get_models():
    """Kembalikan dict {nama_model: estimator} dengan parameter sesuai paper."""
    return {
        "Decision Tree": DecisionTreeClassifier(
            criterion="entropy", max_depth=5, random_state=RANDOM_STATE
        ),
        "SVM": SVC(
            kernel="rbf", C=7, probability=True, random_state=RANDOM_STATE
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(28, 56, 28),
            activation="relu",
            solver="sgd",
            learning_rate_init=0.00005,
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            max_depth=20, n_estimators=100, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            max_depth=2, random_state=RANDOM_STATE, eval_metric="logloss"
        ),
    }
