"""
Run inference on new data using the trained model.

Usage
-----
    python src/predict.py --input path/to/new_data.csv
    python src/predict.py --input path/to/new_data.csv --output results.csv

The input CSV must have the same acoustic feature columns as the training data.
The 'name' and 'status' columns are optional.
"""
import argparse
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from train import preprocess


def predict(
    input_path: str,
    model_path: str = "models/model.pkl",
    scaler_path: str = "models/scaler.pkl",
) -> pd.DataFrame:
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    df = pd.read_csv(input_path)

    drop_cols = ["name"] if "name" in df.columns else []
    if "status" in df.columns:
        X = df.drop(columns=drop_cols + ["status"])
    else:
        X = df.drop(columns=drop_cols)

    X_scaled = scaler.transform(X)
    predictions   = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    results = df.copy()
    results["predicted_status"]       = predictions
    results["parkinsons_probability"] = probabilities.round(4)
    results["prediction_label"] = results["predicted_status"].map(
        {1: "Parkinson's", 0: "Healthy"}
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict Parkinson's disease status from acoustic features."
    )
    parser.add_argument("--input",  required=True,             help="Path to input CSV")
    parser.add_argument("--output", default="predictions.csv", help="Path to save predictions")
    args = parser.parse_args()

    results = predict(args.input)
    results.to_csv(args.output, index=False)

    print(f"Predictions saved to {args.output}")
    print("\nPrediction summary:")
    print(results["prediction_label"].value_counts().to_string())
