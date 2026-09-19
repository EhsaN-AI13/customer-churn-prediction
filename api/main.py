from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"


# =========================
# Load model
# =========================

model = joblib.load(MODEL_PATH)


# =========================
# FastAPI
# =========================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn probability.",
    version="1.0.0",
)


# =========================
# Request schema
# =========================

class CustomerData(BaseModel):

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# =========================
# Health check
# =========================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# =========================
# Prediction
# =========================

@app.post("/predict")
def predict(customer: CustomerData):

    data = pd.DataFrame(
        [customer.model_dump()]
    )

    probability = model.predict_proba(data)[0][1]

    threshold = 0.35

    prediction = (
        "Yes"
        if probability >= threshold
        else "No"
    )

    return {
        "prediction": prediction,
        "churn_probability": round(
            float(probability),
            4
        ),
        "threshold": threshold
    }