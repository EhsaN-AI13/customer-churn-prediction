from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"


def create_sample_customer():
    return pd.DataFrame([
        {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 55.0,
            "TotalCharges": 660.0,
        }
    ])


def test_model_exists():
    assert MODEL_PATH.exists()


def test_model_can_load():
    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_model_can_predict():
    model = joblib.load(MODEL_PATH)

    X = create_sample_customer()

    predictions = model.predict(X)

    assert len(predictions) == 1


def test_model_can_predict_probability():
    model = joblib.load(MODEL_PATH)

    X = create_sample_customer()

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (1, 2)