"""
main.py
=======
Entry point for the Hearing Loss AI demonstration project.

Running this script will:
  1. Load the bundled audiology dataset.
  2. Preprocess features and encode labels.
  3. Split data into train / test sets.
  4. Train a Random Forest classifier.
  5. Evaluate the model and print a classification report.
  6. Display the most important audiological features.
  7. Demonstrate prediction on three example patients.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import CATEGORY_ORDER, FEATURE_COLUMNS, load_data, preprocess, split_data
from src.model import (
    build_model,
    evaluate_model,
    get_feature_importances,
    predict,
    train_model,
)

import pandas as pd


def main():
    print("=" * 60)
    print("  AI in Healthcare – Hearing Loss Severity Classifier")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Load and preprocess data                                          #
    # ------------------------------------------------------------------ #
    print("\n[1] Loading dataset …")
    df = load_data()
    print(f"    Loaded {len(df)} patient records with {len(df.columns)} columns.")

    X, y, encoder = preprocess(df)

    category_counts = df["hearing_loss_category"].value_counts()
    print("\n    Class distribution:")
    for cat in CATEGORY_ORDER:
        print(f"      {cat:<10} {category_counts.get(cat, 0):>4} samples")

    # ------------------------------------------------------------------ #
    # 2. Train / test split                                                #
    # ------------------------------------------------------------------ #
    print("\n[2] Splitting into 80 % train / 20 % test …")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"    Train: {len(X_train)} samples   Test: {len(X_test)} samples")

    # ------------------------------------------------------------------ #
    # 3. Build and train the model                                         #
    # ------------------------------------------------------------------ #
    print("\n[3] Training Random Forest classifier …")
    model = build_model()
    train_model(model, X_train, y_train)
    print("    Training complete.")

    # ------------------------------------------------------------------ #
    # 4. Evaluate                                                          #
    # ------------------------------------------------------------------ #
    print("\n[4] Evaluation on held-out test set:")
    results = evaluate_model(model, X_test, y_test, label_names=CATEGORY_ORDER)
    print(f"\n    Accuracy: {results['accuracy']:.1%}")
    print("\n    Classification report:")
    for line in results["report"].splitlines():
        print("   ", line)

    # ------------------------------------------------------------------ #
    # 5. Feature importances                                               #
    # ------------------------------------------------------------------ #
    print("\n[5] Feature importances (most predictive → least):")
    importances = get_feature_importances(model, FEATURE_COLUMNS)
    for feat, score in importances.items():
        bar = "█" * int(score * 40)
        print(f"    {feat:<38} {score:.3f}  {bar}")

    # ------------------------------------------------------------------ #
    # 6. Example predictions                                               #
    # ------------------------------------------------------------------ #
    print("\n[6] Example patient predictions:")
    examples = pd.DataFrame(
        [
            # age, noise_exp, left_pta, right_pta, srs, tinnitus
            [25,  0, 12.0, 10.5, 97.0, 0],   # young, no exposure → normal
            [52, 20, 38.5, 41.0, 75.0, 1],   # mid-age, industrial noise → mild/moderate
            [72,  5, 65.0, 68.0, 45.0, 1],   # elderly, age-related → severe
        ],
        columns=FEATURE_COLUMNS,
    )
    descriptions = [
        "25 y/o, no noise exposure, good SRS",
        "52 y/o, 20 yrs industrial noise, tinnitus",
        "72 y/o, age-related loss, poor SRS, tinnitus",
    ]

    categories = predict(model, examples, encoder)
    for desc, cat in zip(descriptions, categories):
        print(f"    [{desc}]  →  {cat.upper()}")

    print("\n" + "=" * 60)
    print("  Done. See README.md for full documentation.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
