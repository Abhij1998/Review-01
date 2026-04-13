"""
predict.py
==========
Command-line interface for single-patient hearing loss prediction.

Usage
-----
From the ``hearing_loss_ai/`` directory:

    python src/predict.py --age 55 --noise_exposure_years 20 \
        --left_ear_pta_dB 42.5 --right_ear_pta_dB 38.0 \
        --speech_recognition_score_pct 72.0 --tinnitus 1

The script will train a fresh model on the bundled dataset (unless a
pre-trained model file is provided via --model_path) and then output
a predicted hearing loss category together with a probability breakdown.
"""

import argparse
import sys
import os

import pandas as pd

# Allow running as a script from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import FEATURE_COLUMNS, CATEGORY_ORDER, load_data, preprocess, split_data
from src.model import build_model, train_model, predict


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict hearing loss category for a single patient."
    )
    parser.add_argument("--age", type=float, required=True, help="Patient age in years.")
    parser.add_argument(
        "--noise_exposure_years",
        type=float,
        required=True,
        help="Years of significant occupational/recreational noise exposure.",
    )
    parser.add_argument(
        "--left_ear_pta_dB",
        type=float,
        required=True,
        help="Left-ear Pure Tone Average in dB HL (500–4000 Hz).",
    )
    parser.add_argument(
        "--right_ear_pta_dB",
        type=float,
        required=True,
        help="Right-ear Pure Tone Average in dB HL (500–4000 Hz).",
    )
    parser.add_argument(
        "--speech_recognition_score_pct",
        type=float,
        required=True,
        help="Speech Recognition Score as a percentage (0–100).",
    )
    parser.add_argument(
        "--tinnitus",
        type=int,
        choices=[0, 1],
        required=True,
        help="Tinnitus present? 1 = yes, 0 = no.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional path to a pre-trained model .pkl file.",
    )
    return parser.parse_args(argv)


def run(args):
    """Train (or load) a model and predict for the given patient values."""
    df = load_data()
    X, y, encoder = preprocess(df)
    X_train, _, y_train, _ = split_data(X, y)

    if args.model_path:
        from src.model import load_model
        model = load_model(args.model_path)
    else:
        model = build_model()
        train_model(model, X_train, y_train)

    patient = pd.DataFrame(
        [[
            args.age,
            args.noise_exposure_years,
            args.left_ear_pta_dB,
            args.right_ear_pta_dB,
            args.speech_recognition_score_pct,
            args.tinnitus,
        ]],
        columns=FEATURE_COLUMNS,
    )

    category = predict(model, patient, encoder)[0]
    proba = model.predict_proba(patient)[0]

    print("\n=== Hearing Loss Prediction ===")
    print(f"  Predicted category : {category.upper()}")
    print("\n  Probability breakdown:")
    for label, prob in zip(CATEGORY_ORDER, proba):
        bar = "█" * int(prob * 30)
        print(f"    {label:<10} {prob:.1%}  {bar}")
    print()


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
