"""
Generate evaluation plots and a markdown performance report.

Run this after train.py. Produces:
  metrics/roc_curve.png
  metrics/confusion_matrix.png
  metrics/report.md

Usage
-----
    python src/evaluate.py
"""
import json
import os
import sys

import joblib
import matplotlib
import yaml

matplotlib.use("Agg")  # headless — no display needed in CI
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from train import load_data, load_params, preprocess



# Plots
def plot_roc_curve(model, scaler, X_test, y_test, output_path: str) -> None:
    X_scaled = scaler.transform(X_test)
    y_prob   = model.predict_proba(X_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc  = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--",
            label="Random classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Parkinson's Disease Detection", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"ROC curve saved → {output_path}")


def plot_confusion_matrix(model, scaler, X_test, y_test, output_path: str) -> None:
    X_scaled = scaler.transform(X_test)
    y_pred   = model.predict(X_scaled)
    cm       = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Healthy", "Parkinson's"],
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Parkinson's Disease Detection", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"Confusion matrix saved → {output_path}")


# Report
def write_markdown_report(metrics: dict, output_path: str) -> None:
    lines = [
        "## Model Performance Report",
        "",
        "**Dataset:** UCI Parkinson's Disease  ",
        "**Model:** Logistic Regression  ",
        "",
        "| Metric        | Value  |",
        "|---------------|--------|",
        f"| Accuracy      | {metrics['accuracy']:.4f} |",
        f"| ROC AUC       | {metrics['roc_auc']:.4f} |",
        f"| F1 Score      | {metrics['f1_score']:.4f} |",
        f"| Train samples | {metrics.get('n_train', '—')} |",
        f"| Test samples  | {metrics['n_test']} |",
        "",
        "### ROC Curve",
        "![ROC Curve](metrics/roc_curve.png)",
        "",
        "### Confusion Matrix",
        "![Confusion Matrix](metrics/confusion_matrix.png)",
        "",
    ]
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Markdown report saved → {output_path}")


# Entry point
if __name__ == "__main__":
    os.makedirs("metrics", exist_ok=True)

    params = load_params()

    df = load_data(params["data"]["filepath"])
    X, y = preprocess(
        df,
        target=params["data"]["target"],
        drop_cols=params["data"]["drop_cols"],
    )

    # Use the same split as train.py (identical seed + stratify)
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
        stratify=y,
    )

    model  = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    with open("metrics/metrics.json") as f:
        metrics = json.load(f)

    plot_roc_curve(model, scaler, X_test, y_test, "metrics/roc_curve.png")
    plot_confusion_matrix(model, scaler, X_test, y_test, "metrics/confusion_matrix.png")
    write_markdown_report(metrics, "metrics/report.md")
