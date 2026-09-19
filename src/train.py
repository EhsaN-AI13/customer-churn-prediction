from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"


# =========================
# Load data
# =========================

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


# =========================
# Prepare data
# =========================

def prepare_data(df):

    df = df.copy()

    # Convert TotalCharges from object to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Customer ID is not a useful feature
    df = df.drop(columns=["customerID"])

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y


# =========================
# Build preprocessing
# =========================

def build_preprocessor(X):

    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        exclude=["object", "string"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


# =========================
# Build model
# =========================

def build_model(preprocessor):

    model = LogisticRegression(
        max_iter=1000
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


# =========================
# Train model
# =========================

def train():

    print("Loading dataset...")

    df = load_data()

    print(f"Dataset shape: {df.shape}")

    X, y = prepare_data(df)

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    preprocessor = build_preprocessor(X_train)

    pipeline = build_model(preprocessor)

    print("Training model...")

    pipeline.fit(X_train, y_train)

    print("Training completed.")

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()