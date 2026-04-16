"""
Train a Logistic Regression classifier for Parkinson's Disease detection.

Uses 22 vocal acoustic features from the UCI Parkinson's dataset to classify
subjects as having Parkinson's disease (status=1) or being healthy (status=0).

Outputs
-------
models/model.pkl   – fitted LogisticRegression
models/scaler.pkl  – fitted StandardScaler
metrics/metrics.json – accuracy, ROC AUC, F1, sample counts
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def preprocess(
    df: pd.DataFrame,
    target: str = "status",
    drop_cols: list = None,
) -> tuple:
    """Return (X, y) after dropping identifier and target columns."""
    drop_cols = drop_cols or ["name"]
    X = df.drop(columns=drop_cols + [target])
    y = df[target]
    return X, y



# Training
def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict,
) -> tuple:
    """Scale features and fit a logistic regression. Returns (model, scaler)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        C=params["model"]["C"],
        max_iter=params["model"]["max_iter"],
        random_state=params["model"]["random_state"],
    )
    model.fit(X_scaled, y_train)
    return model, scaler


# Evaluation
def evaluate_model(
    model,
    scaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Return a dict of scalar performance metrics on the held-out test set."""
    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "roc_auc":  round(float(roc_auc_score(y_test, y_prob)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "n_test":   int(len(y_test)),
    }

# Persistence
def save_artifacts(model, scaler, metrics: dict) -> None:
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    joblib.dump(model,  "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


# Entry point
if __name__ == "__main__":
    params = load_params()

    df = load_data(params["data"]["filepath"])
    X, y = preprocess(
        df,
        target=params["data"]["target"],
        drop_cols=params["data"]["drop_cols"],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
        stratify=y,
    )

    model, scaler = train_model(X_train, y_train, params)

    metrics = evaluate_model(model, scaler, X_test, y_test)
    metrics["n_train"] = int(len(y_train))

    save_artifacts(model, scaler, metrics)

    print("=" * 45)
    print("  Training complete")
    print("=" * 45)
    for k, v in metrics.items():
        print(f"  {k:<12}: {v}")
    print()
    print("  model  → models/model.pkl")
    print("  scaler → models/scaler.pkl")
    print("  metrics→ metrics/metrics.json")
