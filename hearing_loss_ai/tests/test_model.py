"""
tests/test_model.py
===================
Unit tests for the hearing_loss_ai project.

Run from the hearing_loss_ai/ directory:
    python -m pytest tests/ -v
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Make sure the package root is on the path so imports work from any CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import (
    CATEGORY_ORDER,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_data,
    preprocess,
    split_data,
)
from src.model import (
    build_model,
    evaluate_model,
    get_feature_importances,
    load_model,
    predict,
    save_model,
    train_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_df():
    """Load the bundled CSV dataset once for the whole module."""
    return load_data()


@pytest.fixture(scope="module")
def preprocessed(raw_df):
    X, y, encoder = preprocess(raw_df)
    return X, y, encoder


@pytest.fixture(scope="module")
def trained_model(preprocessed):
    X, y, encoder = preprocessed
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = build_model()
    train_model(model, X_train, y_train)
    return model, X_train, X_test, y_train, y_test, encoder


# ---------------------------------------------------------------------------
# data_loader tests
# ---------------------------------------------------------------------------

class TestLoadData:
    def test_loads_csv(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_row_count(self, raw_df):
        assert len(raw_df) == 200

    def test_expected_columns(self, raw_df):
        for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
            assert col in raw_df.columns, f"Missing column: {col}"

    def test_no_missing_values(self, raw_df):
        assert raw_df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum().sum() == 0

    def test_target_values(self, raw_df):
        assert set(raw_df[TARGET_COLUMN].unique()).issubset(set(CATEGORY_ORDER))

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_data("/non/existent/path/data.csv")


class TestPreprocess:
    def test_X_shape(self, preprocessed, raw_df):
        X, y, _ = preprocessed
        assert X.shape == (len(raw_df), len(FEATURE_COLUMNS))

    def test_y_length(self, preprocessed, raw_df):
        _, y, _ = preprocessed
        assert len(y) == len(raw_df)

    def test_y_integer_encoded(self, preprocessed):
        _, y, _ = preprocessed
        assert y.dtype in (np.int32, np.int64, int)

    def test_encoder_classes(self, preprocessed):
        _, _, encoder = preprocessed
        assert list(encoder.classes_) == CATEGORY_ORDER

    def test_encoder_inverse(self, preprocessed):
        _, y, encoder = preprocessed
        labels = encoder.inverse_transform(y)
        assert set(labels).issubset(set(CATEGORY_ORDER))


class TestSplitData:
    def test_split_sizes(self, preprocessed):
        X, y, _ = preprocessed
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        total = len(X)
        assert len(X_test) == pytest.approx(total * 0.2, abs=2)
        assert len(X_train) + len(X_test) == total

    def test_no_overlap(self, preprocessed):
        X, y, _ = preprocessed
        X_train, X_test, _, _ = split_data(X, y)
        train_idx = set(X_train.index)
        test_idx = set(X_test.index)
        assert train_idx.isdisjoint(test_idx)


# ---------------------------------------------------------------------------
# model tests
# ---------------------------------------------------------------------------

class TestBuildModel:
    def test_returns_rf(self):
        from sklearn.ensemble import RandomForestClassifier
        model = build_model()
        assert isinstance(model, RandomForestClassifier)

    def test_hyperparameters(self):
        model = build_model(n_estimators=50, max_depth=4, random_state=7)
        assert model.n_estimators == 50
        assert model.max_depth == 4
        assert model.random_state == 7


class TestTrainModel:
    def test_model_is_fitted(self, trained_model):
        model, *_ = trained_model
        # sklearn fitted models expose estimators_
        assert hasattr(model, "estimators_") and len(model.estimators_) > 0


class TestEvaluateModel:
    def test_accuracy_range(self, trained_model):
        model, _, X_test, _, y_test, _ = trained_model
        results = evaluate_model(model, X_test, y_test, label_names=CATEGORY_ORDER)
        assert 0.0 <= results["accuracy"] <= 1.0

    def test_report_contains_categories(self, trained_model):
        model, _, X_test, _, y_test, _ = trained_model
        results = evaluate_model(model, X_test, y_test, label_names=CATEGORY_ORDER)
        for cat in CATEGORY_ORDER:
            assert cat in results["report"]

    def test_high_accuracy(self, trained_model):
        """Model should achieve at least 80 % accuracy on the test set."""
        model, _, X_test, _, y_test, _ = trained_model
        results = evaluate_model(model, X_test, y_test)
        assert results["accuracy"] >= 0.80, (
            f"Accuracy {results['accuracy']:.1%} is below the 80 % threshold"
        )


class TestFeatureImportances:
    def test_returns_dict(self, trained_model):
        model, *_ = trained_model
        fi = get_feature_importances(model, FEATURE_COLUMNS)
        assert isinstance(fi, dict)

    def test_all_features_present(self, trained_model):
        model, *_ = trained_model
        fi = get_feature_importances(model, FEATURE_COLUMNS)
        assert set(fi.keys()) == set(FEATURE_COLUMNS)

    def test_importances_sum_to_one(self, trained_model):
        model, *_ = trained_model
        fi = get_feature_importances(model, FEATURE_COLUMNS)
        assert sum(fi.values()) == pytest.approx(1.0, abs=1e-6)

    def test_sorted_descending(self, trained_model):
        model, *_ = trained_model
        fi = get_feature_importances(model, FEATURE_COLUMNS)
        values = list(fi.values())
        assert values == sorted(values, reverse=True)


class TestSaveLoadModel:
    def test_roundtrip(self, trained_model):
        model, X_train, X_test, y_train, y_test, encoder = trained_model
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            save_model(model, path)
            loaded = load_model(path)
            original_preds = model.predict(X_test)
            loaded_preds = loaded.predict(X_test)
            np.testing.assert_array_equal(original_preds, loaded_preds)
        finally:
            os.unlink(path)

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_model("/tmp/does_not_exist_xyz.pkl")


class TestPredict:
    def test_output_length(self, trained_model):
        model, _, X_test, _, _, encoder = trained_model
        preds = predict(model, X_test, encoder)
        assert len(preds) == len(X_test)

    def test_output_values_valid(self, trained_model):
        model, _, X_test, _, _, encoder = trained_model
        preds = predict(model, X_test, encoder)
        assert set(preds).issubset(set(CATEGORY_ORDER))

    def test_single_patient(self, trained_model):
        model, _, _, _, _, encoder = trained_model
        patient = pd.DataFrame(
            [[25, 0, 12.0, 10.5, 97.0, 0]],
            columns=FEATURE_COLUMNS,
        )
        result = predict(model, patient, encoder)
        assert len(result) == 1
        assert result[0] in CATEGORY_ORDER
