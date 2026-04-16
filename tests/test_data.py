"""
Data integrity and schema validation tests.

These run before any model training to catch broken downloads,
schema drift, or corrupt files early in the CI pipeline.
"""
import os

import pandas as pd
import pytest

DATA_PATH        = "data/parkinsons.csv"
EXPECTED_TARGET  = "status"
EXPECTED_ID_COL  = "name"
MIN_ROWS         = 100
EXPECTED_CLASSES = {0, 1}


@pytest.fixture(scope="module")
def df():
    if not os.path.exists(DATA_PATH):
        pytest.skip(
            f"Data file not found at {DATA_PATH}. "
            "Run:  python data/download_data.py"
        )
    return pd.read_csv(DATA_PATH)

def test_data_file_exists():
    assert os.path.exists(DATA_PATH), (
        f"Data file missing: {DATA_PATH}. Run: python data/download_data.py"
    )
def test_target_column_exists(df):
    assert EXPECTED_TARGET in df.columns, (
        f"Target column '{EXPECTED_TARGET}' not found in dataset"
    )


def test_id_column_exists(df):
    assert EXPECTED_ID_COL in df.columns, (
        f"ID column '{EXPECTED_ID_COL}' not found in dataset"
    )


def test_minimum_row_count(df):
    assert len(df) >= MIN_ROWS, (
        f"Expected at least {MIN_ROWS} rows, got {len(df)}"
    )

def test_target_is_binary(df):
    unique_vals = set(df[EXPECTED_TARGET].unique())
    assert unique_vals.issubset(EXPECTED_CLASSES), (
        f"Expected binary target {{0, 1}}, got {unique_vals}"
    )


def test_no_missing_values(df):
    n_missing = df.isnull().sum().sum()
    assert n_missing == 0, f"Found {n_missing} missing values in dataset"


def test_features_are_numeric(df):
    feature_cols = [
        c for c in df.columns if c not in [EXPECTED_ID_COL, EXPECTED_TARGET]
    ]
    non_numeric = [
        c for c in feature_cols
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    assert not non_numeric, f"Non-numeric feature columns: {non_numeric}"


def test_no_duplicate_feature_rows(df):
    numeric_df = df.drop(columns=[EXPECTED_ID_COL])
    n_dupes = numeric_df.duplicated().sum()
    assert n_dupes == 0, f"Found {n_dupes} duplicate rows (excluding name column)"
