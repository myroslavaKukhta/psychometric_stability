# Psychometric Stability: Synthetic Data and Model Explainability

This repository contains the reproducibility code for:

> Kukhta M., Statkevych V. Controlled Synthetic Data for Analysis of Stability and Explainability of Classification Models in Social and Psychometric Measurements. Submitted manuscript.

The code generates synthetic psychometric data, evaluates classification stability, estimates SHAP-rank stability, and performs an out-of-distribution synthetic transferability check.

## Repository contents

| File | Purpose |
|---|---|
| `01_classification_stability.py` | RQ1: classification performance and stability across 30 independent synthetic-data runs |
| `02_shap_stability.py` | RQ2: SHAP feature-importance rank stability across four controlled variation scenarios |
| `03_ood_transferability.py` | RQ3: out-of-distribution synthetic transferability check for Random Forest |
| `requirements.txt` | Pinned Python dependencies tested for this release |
| `run_all.sh` | Convenience runner for all three experiments |
| `MANIFEST.md` | File list, expected outputs, and reproducibility notes |
| `LICENSE` | MIT license |

## Installation

Use Python 3.10 or 3.11. A virtual environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

Run all experiments:

```bash
bash run_all.sh
```

Or run scripts individually:

```bash
python 01_classification_stability.py
python 02_shap_stability.py
python 03_ood_transferability.py
```

Outputs are written to:

- `results/` for CSV tables
- `figures/` for PNG figures

## Research questions and outputs

| RQ | Script | Main output |
|---|---|---|
| RQ1 | `01_classification_stability.py` | `results/metrics_rq1.csv`; Figures 4.1–4.3 |
| RQ2 | `02_shap_stability.py` | `results/shap_rank_correlations.csv`; Figures 4.4–4.6 |
| RQ3 | `03_ood_transferability.py` | `results/rq3_transferability.csv`; Figure 4.7 |

## Synthetic data design

The dataset includes five continuous features and five categorical features.

| Feature | Meaning | Range / categories |
|---|---|---|
| `gad7_score` | GAD-7 score | 0–21 |
| `phq4_score` | PHQ-4 score | 0–12 |
| `nstf_score` | Neurosensitivity and Stress Triggers Form score | 0–29 |
| `brs_score` | Brief Resilience Scale score | 0–29 |
| `age` | Age | 18–64 |
| `gender` | Gender category | generated categorical |
| `education` | Education level | generated categorical |
| `marital_status` | Marital status | generated categorical |
| `therapy_experience` | Therapy experience | generated categorical |
| `chronic_disease` | Chronic disease indicator | generated categorical |

The target variable `risk_group` is derived from a synthetic risk index:

```text
S = 0.4 * GAD7 + 0.3 * PHQ4 + 0.2 * NSTF - 0.3 * BRS + ε
```

where `ε ~ N(0, σ²)`. The `age` feature is included as an input to classifiers but has zero direct weight in the risk-index construction.

Risk groups are obtained by quantising `S` at the 33rd and 66th percentiles into three classes: low, medium, and high.

## Variation scenarios

| Scenario | Description |
|---|---|
| `s0` | Baseline synthetic data |
| `s1` | Increased noise, `noise_sigma = 6.0` |
| `s2` | Rotated continuous-feature weight vector |
| `s3` | Shifted categorical-feature distribution |
| `s4` | Increased overlap between class boundaries |

## Models

| Model | Implementation |
|---|---|
| Logistic Regression | `LogisticRegression(C=1.0, max_iter=1000)` |
| SVM | `SVC(kernel="rbf", C=1.0, gamma="scale")` |
| Random Forest | `RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced")` |

Internal model random states are fixed where applicable to isolate variability caused by synthetic data generation and train/test splitting.

## Expected key results

### RQ1: baseline macro-F1 over 30 runs

| Model | Macro-F1 mean ± 95% CI |
|---|---:|
| Logistic Regression | 0.660 ± 0.006 |
| SVM | 0.645 ± 0.006 |
| Random Forest | 0.625 ± 0.007 |

### RQ2: SHAP-rank stability

Script 02 evaluates SHAP-rank stability for Logistic Regression and Random Forest. SVM is excluded from this SHAP analysis because a fast model-specific SHAP explainer is not available for kernel SVMs at the experimental scale.

Expected mean Spearman correlations by scenario:

| Model | s1 | s2 | s3 | s4 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.658 | 0.574 | 0.879 | 0.660 |
| Random Forest | 0.705 | 0.652 | 0.927 | 0.686 |

### RQ3: OOD synthetic transferability

Script 03 evaluates Random Forest on a held-out out-of-distribution synthetic sample with `N = 1000`, `noise_sigma = 6.0`, and `seed = 999`.

Expected result:

| Metric | Value |
|---|---:|
| Accuracy on OOD sample | 0.515 |
| Macro-F1 on OOD sample | 0.505 |
| SHAP Spearman correlation, baseline vs OOD | 0.995 |

The RQ3 result is a deterministic single-pair check, not a confidence-interval estimate.

## Reproducibility notes

- RQ1 uses 30 independent data-generation seeds.
- RQ2 uses 10 independent data-generation seeds per scenario.
- RQ3 uses a single deterministic OOD sample.
- Package versions are pinned in `requirements.txt`.
- The scripts do not require real psychometric or participant-level data.

## License

MIT License. See `LICENSE`.
