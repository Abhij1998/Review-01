"""
model.py
========
Machine-learning model for hearing loss severity classification.

A Random Forest classifier is trained on audiological patient features to
predict one of four WHO-aligned hearing loss categories:
  normal  → PTA < 26 dB HL
  mild    → 26–40 dB HL
  moderate→ 41–60 dB HL
  severe  → > 60 dB HL

This mirrors real-world clinical decision-support tools where ML models
assist audiologists in triaging patients or identifying at-risk individuals
from routine screening data.
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Default model hyperparameters chosen for interpretability on small data
DEFAULT_N_ESTIMATORS = 100
DEFAULT_MAX_DEPTH = 6
DEFAULT_RANDOM_STATE = 42


def build_model(
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> RandomForestClassifier:
    """Instantiate a Random Forest classifier.

    Parameters
    ----------
    n_estimators:
        Number of decision trees in the forest.
    max_depth:
        Maximum depth of each tree.  Limits over-fitting on small datasets.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    RandomForestClassifier
        An unfitted estimator ready for :func:`train_model`.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight="balanced",
    )


def train_model(model: RandomForestClassifier, X_train, y_train) -> RandomForestClassifier:
    """Fit the model on training data.

    Parameters
    ----------
    model:
        Unfitted ``RandomForestClassifier``.
    X_train:
        Training feature matrix.
    y_train:
        Training target array (integer-encoded labels).

    Returns
    -------
    RandomForestClassifier
        The same estimator, now fitted.
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: RandomForestClassifier, X_test, y_test, label_names=None) -> dict:
    """Evaluate a fitted model on a held-out test set.

    Parameters
    ----------
    model:
        Fitted ``RandomForestClassifier``.
    X_test:
        Test feature matrix.
    y_test:
        True labels for the test set.
    label_names:
        Human-readable class names in label-index order (optional).

    Returns
    -------
    dict
        ``{"accuracy": float, "report": str}``
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_names, zero_division=0)
    return {"accuracy": accuracy, "report": report}


def get_feature_importances(model: RandomForestClassifier, feature_names) -> dict:
    """Return feature importances as a name → importance mapping.

    Parameters
    ----------
    model:
        Fitted ``RandomForestClassifier``.
    feature_names:
        Sequence of feature column names matching the training order.

    Returns
    -------
    dict
        ``{feature_name: importance_score}`` sorted by importance descending.
    """
    importances = model.feature_importances_
    paired = dict(zip(feature_names, importances))
    return dict(sorted(paired.items(), key=lambda kv: kv[1], reverse=True))


def save_model(model: RandomForestClassifier, filepath: str) -> None:
    """Persist a fitted model to disk using joblib.

    Parameters
    ----------
    model:
        Fitted estimator to save.
    filepath:
        Destination file path (e.g. ``"model.pkl"``).
    """
    joblib.dump(model, filepath)


def load_model(filepath: str) -> RandomForestClassifier:
    """Load a previously saved model from disk.

    Parameters
    ----------
    filepath:
        Path to a file created by :func:`save_model`.

    Returns
    -------
    RandomForestClassifier
        The fitted estimator.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    """
    import os
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    return joblib.load(filepath)


def predict(model: RandomForestClassifier, X, encoder) -> np.ndarray:
    """Run inference and return human-readable category labels.

    Parameters
    ----------
    model:
        Fitted estimator.
    X:
        Feature matrix (same column order as training).
    encoder:
        ``LabelEncoder`` from :func:`~src.data_loader.preprocess`.

    Returns
    -------
    np.ndarray
        Array of category strings, e.g. ``["mild", "normal", ...]``.
    """
    return encoder.inverse_transform(model.predict(X))
