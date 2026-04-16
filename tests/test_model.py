"""
Model artifact and prediction quality tests.

These run after train.py to verify that the saved model meets
minimum performance thresholds before it reaches production.
"""
import os

import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from train import load_data, load_params, preprocess

MODEL_PATH   = "models/model.pkl"
SCALER_PATH  = "models/scaler.pkl"
MIN_ACCURACY = 0.80
MIN_ROC_AUC  = 0.85


# Fixtures
@pytest.fixture(scope="module")
def params():
    return load_params()


@pytest.fixture(scope="module")
def model():
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model not found: {MODEL_PATH}. Run: python src/train.py")
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def scaler():
    if not os.path.exists(SCALER_PATH):
        pytest.skip(f"Scaler not found: {SCALER_PATH}. Run: python src/train.py")
    return joblib.load(SCALER_PATH)


@pytest.fixture(scope="module")
def test_split(params):
    data_path = params["data"]["filepath"]
    if not os.path.exists(data_path):
        pytest.skip(f"Data not found: {data_path}. Run: python data/download_data.py")
    df = load_data(data_path)
    X, y = preprocess(
        df,
        target=params["data"]["target"],
        drop_cols=params["data"]["drop_cols"],
    )
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
        stratify=y,
    )
    return X_test, y_test

# Artifact tests
def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), f"Model artifact missing: {MODEL_PATH}"


def test_scaler_file_exists():
    assert os.path.exists(SCALER_PATH), f"Scaler artifact missing: {SCALER_PATH}"


def test_model_is_logistic_regression(model):
    assert isinstance(model, LogisticRegression), (
        f"Expected LogisticRegression, got {type(model).__name__}"
    )


def test_model_is_fitted(model):
    assert hasattr(model, "coef_"), "Model has not been fitted (missing coef_)"


def test_feature_count_consistency(model, scaler):
    assert model.n_features_in_ == scaler.n_features_in_, (
        f"Feature count mismatch: model={model.n_features_in_}, "
        f"scaler={scaler.n_features_in_}"
    )


# Performance tests
def test_model_accuracy(model, scaler, test_split):
    from sklearn.metrics import accuracy_score
    X_test, y_test = test_split
    y_pred = model.predict(scaler.transform(X_test))
    acc = accuracy_score(y_test, y_pred)
    assert acc >= MIN_ACCURACY, (
        f"Accuracy {acc:.4f} is below the minimum threshold {MIN_ACCURACY}"
    )


def test_model_roc_auc(model, scaler, test_split):
    from sklearn.metrics import roc_auc_score
    X_test, y_test = test_split
    y_prob  = model.predict_proba(scaler.transform(X_test))[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    assert roc_auc >= MIN_ROC_AUC, (
        f"ROC AUC {roc_auc:.4f} is below the minimum threshold {MIN_ROC_AUC}"
    )


# Output sanity tests
def test_predictions_are_binary(model, scaler, test_split):
    X_test, _ = test_split
    preds = model.predict(scaler.transform(X_test))
    assert set(preds).issubset({0, 1}), (
        f"Unexpected prediction values: {set(preds)}"
    )


def test_probabilities_in_unit_interval(model, scaler, test_split):
    X_test, _ = test_split
    probs = model.predict_proba(scaler.transform(X_test))
    assert probs.min() >= 0.0 and probs.max() <= 1.0, (
        "Predicted probabilities fall outside [0, 1]"
    )
