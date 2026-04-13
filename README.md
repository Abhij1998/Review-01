# AI in Healthcare – Hearing Loss Severity Classifier

> A self-contained Python project demonstrating how machine learning can
> assist audiologists and hearing-health professionals in classifying the
> severity of a patient's hearing loss from routine clinical measurements.

---

## Table of Contents

1. [Concept & Clinical Background](#concept--clinical-background)
2. [Project Structure](#project-structure)
3. [Dataset](#dataset)
4. [Model](#model)
5. [Quick Start](#quick-start)
6. [Usage](#usage)
   - [Full Demo (`main.py`)](#full-demo-mainpy)
   - [Single-Patient Prediction CLI](#single-patient-prediction-cli)
7. [Running the Tests](#running-the-tests)
8. [Results](#results)
9. [How This Relates to Audiology & Hearing Science](#how-this-relates-to-audiology--hearing-science)
10. [Future Improvements](#future-improvements)
11. [License](#license)

---

## Concept & Clinical Background

**Hearing loss** affects more than 1.5 billion people worldwide (WHO, 2021)
and is the most prevalent sensory disability. Early detection and severity
classification are critical for timely intervention.

In a standard audiological assessment, clinicians collect:

| Measurement | Description |
|---|---|
| **Pure Tone Average (PTA)** | Mean hearing threshold across 500–4000 Hz; the gold-standard metric for hearing loss severity |
| **Speech Recognition Score (SRS)** | Percentage of spoken words correctly identified |
| **Noise exposure history** | Years of occupational or recreational noise exposure |
| **Tinnitus** | Ringing/buzzing in ears; a common early indicator of cochlear damage |

The **WHO grading** used in this project maps average binaural PTA to four
categories:

| Category | PTA range |
|---|---|
| Normal | < 26 dB HL |
| Mild | 26–40 dB HL |
| Moderate | 41–60 dB HL |
| Severe | > 60 dB HL |

A machine-learning model that can reliably assign these categories from
screening data could help:
- **Triage** large populations in low-resource settings.
- **Prioritise** patients for detailed diagnostic follow-up.
- **Track** individual progression over time.

---

## Project Structure

```
hearing_loss_ai/
├── data/
│   └── hearing_data.csv       # Synthetic dataset (200 patients, 7 features)
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # CSV loading, feature extraction, train/test split
│   ├── model.py               # Random Forest build / train / evaluate / save / load
│   └── predict.py             # Command-line single-patient inference tool
├── tests/
│   ├── __init__.py
│   └── test_model.py          # 28 unit tests covering all modules
├── main.py                    # End-to-end demo script
└── requirements.txt           # Python dependencies
```

---

## Dataset

`data/hearing_data.csv` contains **200 synthetic patient records** generated
with realistic statistical relationships between age, noise exposure, and
audiological measurements.

| Column | Type | Description |
|---|---|---|
| `patient_id` | string | Unique patient identifier (P001–P200) |
| `age` | int | Patient age (18–85 years) |
| `noise_exposure_years` | int | Years of significant noise exposure |
| `left_ear_pta_dB` | float | Left-ear Pure Tone Average (dB HL) |
| `right_ear_pta_dB` | float | Right-ear Pure Tone Average (dB HL) |
| `speech_recognition_score_pct` | float | Speech Recognition Score (0–100 %) |
| `tinnitus` | int | Tinnitus present: 1 = yes, 0 = no |
| `hearing_loss_category` | string | **Target**: normal / mild / moderate / severe |

> **Note:** This is a synthetic demonstration dataset. Do not use it for
> clinical decisions or published research.

---

## Model

A **Random Forest classifier** (scikit-learn) is used because it:

- Handles small datasets well without extensive hyperparameter tuning.
- Provides interpretable **feature importances**.
- Is robust to outliers and skewed class distributions.
- Requires no feature scaling.

Key settings: `n_estimators=100`, `max_depth=6`, `class_weight="balanced"`.

The `class_weight="balanced"` option compensates for the imbalance between
the common "normal" category and the rarer "severe" category.

---

## Quick Start

```bash
# 1. Clone / navigate to the project
cd hearing_loss_ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the end-to-end demo
python main.py
```

---

## Usage

All commands should be run from the `hearing_loss_ai/` directory.

### Full Demo (`main.py`)

```bash
python main.py
```

This script:
1. Loads and summarises the dataset.
2. Trains the Random Forest on 80 % of the data.
3. Prints a full classification report on the held-out 20 %.
4. Shows feature importances in a visual bar format.
5. Demonstrates predictions for three archetypal patients.

**Sample output (abridged):**

```
============================================================
  AI in Healthcare – Hearing Loss Severity Classifier
============================================================

[1] Loading dataset …
    Loaded 200 patient records with 8 columns.

    Class distribution:
      normal        82 samples
      mild          60 samples
      moderate      53 samples
      severe         5 samples

[4] Evaluation on held-out test set:

    Accuracy: 95.0%

[5] Feature importances (most predictive → least):
    right_ear_pta_dB        0.377  ███████████████
    left_ear_pta_dB         0.246  █████████
    speech_recognition …    0.227  █████████
    noise_exposure_years    0.073  ██
    age                     0.073  ██
    tinnitus                0.005

[6] Example patient predictions:
    [25 y/o, no noise exposure, good SRS]  →  NORMAL
    [52 y/o, 20 yrs industrial noise, tinnitus]  →  MILD
    [72 y/o, age-related loss, poor SRS, tinnitus]  →  SEVERE
```

---

### Single-Patient Prediction CLI

```bash
python src/predict.py \
  --age 55 \
  --noise_exposure_years 20 \
  --left_ear_pta_dB 42.5 \
  --right_ear_pta_dB 38.0 \
  --speech_recognition_score_pct 72.0 \
  --tinnitus 1
```

**Output:**

```
=== Hearing Loss Prediction ===
  Predicted category : MILD

  Probability breakdown:
    normal      1.8%
    mild       82.1%  ████████████████████████
    moderate   15.1%  ████
    severe      1.0%
```

All flags are required. Run `python src/predict.py --help` for descriptions.

---

## Running the Tests

```bash
cd hearing_loss_ai
python -m pytest tests/ -v
```

The test suite comprises **28 unit tests** across five areas:

| Test class | What is covered |
|---|---|
| `TestLoadData` | CSV loading, schema validation, missing-value check, error handling |
| `TestPreprocess` | Feature/target shapes, integer encoding, encoder class names |
| `TestSplitData` | Train/test sizes, non-overlapping indices |
| `TestBuildModel` | Estimator type, custom hyperparameters |
| `TestTrainModel` | Model is fitted after training |
| `TestEvaluateModel` | Accuracy range, report content, ≥ 80 % accuracy gate |
| `TestFeatureImportances` | Dict structure, completeness, sums to 1, sorted |
| `TestSaveLoadModel` | Round-trip serialisation, missing-file error |
| `TestPredict` | Output length, valid label values, single-patient inference |

Expected result: **28 passed**.

---

## Results

On a stratified 80/20 split of the 200-record dataset:

| Metric | Value |
|---|---|
| Overall accuracy | **95 %** |
| Macro-average F1 | **0.96** |

Top predictive features (by Random Forest importance):

1. `right_ear_pta_dB` – 37.7 %
2. `left_ear_pta_dB` – 24.6 %
3. `speech_recognition_score_pct` – 22.7 %
4. `noise_exposure_years` – 7.3 %
5. `age` – 7.3 %
6. `tinnitus` – 0.5 %

PTA is the dominant signal, which aligns with clinical practice where PTA is
the primary diagnostic criterion for hearing loss severity.

---

## How This Relates to Audiology & Hearing Science

### Noise-Induced Hearing Loss (NIHL)

Occupational and recreational noise is the leading *preventable* cause of
hearing loss. `noise_exposure_years` is a known risk factor, and its
importance in the model reflects that. An AI screening tool could flag
industrial workers at risk before their loss becomes clinically significant.

### Age-Related Hearing Loss (Presbycusis)

High-frequency hearing declines with age. The positive correlation between
`age` and PTA in the dataset reflects real epidemiological data. A deployed
model could integrate with GP electronic health records to proactively invite
older patients for audiological review.

### Tinnitus as an Early Warning Signal

Tinnitus often precedes measurable hearing loss. Including it as a feature
shows how AI models can incorporate patient-reported outcomes alongside
objective measurements—an approach validated in real hearing-health AI
research (e.g., work by the British Society of Audiology on digital tools).

### Speech Recognition & Quality of Life

Poor SRS is a strong predictor of hearing-aid candidacy and quality-of-life
impact. The model's reliance on SRS aligns with clinical priorities and
supports automated prioritisation of patients for hearing-aid fitting.

---

## Future Improvements

| Area | Idea |
|---|---|
| **Data** | Replace synthetic data with de-identified real audiograms (e.g., from NHS or VA databases) |
| **Features** | Add frequency-specific thresholds (250, 500, 1000, 2000, 4000, 8000 Hz) for richer audiogram representation |
| **Model** | Experiment with gradient boosting (XGBoost/LightGBM) or a small neural network for larger datasets |
| **Explainability** | Integrate SHAP values for per-patient feature attribution, aiding clinician trust |
| **Calibration** | Apply probability calibration (Platt scaling) to improve reliability of predicted probabilities |
| **Deployment** | Wrap in a REST API (FastAPI) or a simple web interface (Streamlit) for clinician use |
| **Continuous learning** | Implement online learning so the model improves as new patient data arrives |
| **Multi-label** | Extend to bilateral asymmetric loss (different severity per ear) |
| **Longitudinal** | Add time-series modelling to track hearing progression per patient |

---

## License

This project is released under the [MIT License](../LICENSE).
