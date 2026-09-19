from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"


def test_dataset_exists():

    assert DATA_PATH.exists()


def test_model_exists():

    assert MODEL_PATH.exists()


def test_model_can_load():

    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_model_can_predict():

    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df = df.drop(columns=["customerID"])

    X = df.drop(columns=["Churn"])

    predictions = model.predict(X.head(5))

    assert len(predictions) == 5


def test_model_can_predict_probability():

    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df = df.drop(columns=["customerID"])

    X = df.drop(columns=["Churn"])

    probabilities = model.predict_proba(X.head(5))

    assert probabilities.shape == (5, 2)