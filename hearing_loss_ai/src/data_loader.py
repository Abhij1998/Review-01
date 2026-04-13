"""
data_loader.py
==============
Utilities for loading and preprocessing the audiology dataset.

The dataset contains simulated patient records with features commonly
collected during an audiological evaluation, including:
- Pure Tone Average (PTA) for each ear
- Age and occupational noise exposure history
- Speech Recognition Score (SRS)
- Presence of tinnitus

These features map directly to real-world audiological assessments and
mirror the kind of data that hearing-health AI systems are trained on.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Default path to the bundled CSV dataset
_DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "hearing_data.csv"
)

# Feature columns used for model training
FEATURE_COLUMNS = [
    "age",
    "noise_exposure_years",
    "left_ear_pta_dB",
    "right_ear_pta_dB",
    "speech_recognition_score_pct",
    "tinnitus",
]

TARGET_COLUMN = "hearing_loss_category"

# Consistent label ordering used across the project
CATEGORY_ORDER = ["normal", "mild", "moderate", "severe"]


def load_data(filepath: str = _DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the hearing-loss CSV dataset into a DataFrame.

    Parameters
    ----------
    filepath:
        Path to the CSV file.  Defaults to the bundled dataset.

    Returns
    -------
    pd.DataFrame
        Raw dataset with all columns intact.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    return pd.read_csv(filepath)


def preprocess(df: pd.DataFrame):
    """Extract features (X) and encode target labels (y).

    Parameters
    ----------
    df:
        Raw DataFrame returned by :func:`load_data`.

    Returns
    -------
    X : pd.DataFrame
        Numeric feature matrix ready for scikit-learn estimators.
    y : np.ndarray
        Integer-encoded target labels.
    encoder : LabelEncoder
        Fitted encoder so predictions can be converted back to strings.
    """
    X = df[FEATURE_COLUMNS].copy()
    encoder = LabelEncoder()
    encoder.classes_ = np.array(CATEGORY_ORDER)
    y = encoder.transform(df[TARGET_COLUMN].to_numpy())
    return X, y, encoder


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Split data into training and test sets.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Target array.
    test_size:
        Fraction of data to reserve for testing (default 20 %).
    random_state:
        Seed for reproducibility.

    Returns
    -------
    Tuple of (X_train, X_test, y_train, y_test).
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
