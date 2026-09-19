from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_PATH = RESULTS_DIR / "evaluation.json"


# =========================
# Load data
# =========================

def load_data():

    df = pd.read_csv(DATA_PATH)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df = df.drop(columns=["customerID"])

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y


# =========================
# Calculate metrics
# =========================

def calculate_metrics(y_true, y_proba, threshold):

    y_pred = [
        "Yes" if probability >= threshold else "No"
        for probability in y_proba
    ]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            pos_label="Yes"
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            pos_label="Yes"
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            pos_label="Yes"
        ),
    }


# =========================
# Evaluate model
# =========================

def evaluate():

    print("Loading model...")

    model = joblib.load(MODEL_PATH)

    print("Loading dataset...")

    X, y = load_data()

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Test samples: {len(X_test)}")

    # Churn probabilities
    y_proba = model.predict_proba(X_test)[:, 1]

    # ROC-AUC
    roc_auc = roc_auc_score(
        (y_test == "Yes").astype(int),
        y_proba
    )

    # Thresholds
    metrics_050 = calculate_metrics(
        y_test,
        y_proba,
        threshold=0.50
    )

    metrics_035 = calculate_metrics(
        y_test,
        y_proba,
        threshold=0.35
    )

    # Final selected threshold
    final_threshold = 0.35

    final_metrics = {
        "threshold": final_threshold,
        "accuracy": metrics_035["accuracy"],
        "precision": metrics_035["precision"],
        "recall": metrics_035["recall"],
        "f1": metrics_035["f1"],
        "roc_auc": roc_auc,
    }

    # Comparison
    comparison = {
        "threshold_0.50": {
            **metrics_050,
            "roc_auc": roc_auc,
        },
        "threshold_0.35": {
            **metrics_035,
            "roc_auc": roc_auc,
        },
    }

    results = {
        "final_model": "Logistic Regression",
        "selected_threshold": final_threshold,
        "final_metrics": final_metrics,
        "comparison": comparison,
    }

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("\n===== Final Model =====")

    print("Model     : Logistic Regression")
    print("Threshold : 0.35")

    print(f"Accuracy  : {final_metrics['accuracy']:.4f}")
    print(f"Precision : {final_metrics['precision']:.4f}")
    print(f"Recall    : {final_metrics['recall']:.4f}")
    print(f"F1 Score  : {final_metrics['f1']:.4f}")
    print(f"ROC-AUC   : {final_metrics['roc_auc']:.4f}")

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    evaluate()