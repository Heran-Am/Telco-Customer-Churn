# Telco Customer Churn Prediction (End-to-End ML Mini Project)

This project builds a complete churn prediction pipeline (data cleaning → EDA → modeling → threshold tuning → interpretation → model comparison).
The goal is to predict whether a customer will churn and to understand the main factors associated with churn.

**Target variable:** `Churn Value`  
- `1` = customer churned  
- `0` = customer did not churn

---

## What this project delivers

- ✅ Cleaned dataset ready for modeling
- ✅ Exploratory Data Analysis (EDA) with saved plots
- ✅ Baseline model (Logistic Regression) with a preprocessing pipeline
- ✅ Threshold tuning (precision/recall trade-off)
- ✅ Feature importance / driver interpretation (Logistic Regression coefficients)
- ✅ Model comparison (Logistic Regression vs Random Forest)
- ✅ Metrics saved to JSON for reproducibility

---

## Repository structure

Scripts:
- `src/01_load_and_inspect.py` — load raw Excel + inspect schema, types, missing values
- `src/02_clean_data.py` — clean data + export processed CSV
- `src/03_eda_plots.py` — generate EDA plots into `reports/figures/`
- `src/04_train_baseline.py` — train baseline Logistic Regression + evaluate at tuned threshold
- `src/05_threshold_tuning.py` — evaluate multiple thresholds (precision/recall/F1)
- `src/06_feature_importance.py` — interpret coefficients (top churn-increasing/decreasing features)
- `src/07_model_comparison.py` — compare Logistic Regression vs Random Forest (same pipeline, same threshold)

Key folders:
- `data/raw/` — raw dataset (not committed)
- `data/processed/` — processed dataset (not committed)
- `reports/figures/` — plots generated from EDA
- `reports/metrics_baseline.json` — baseline model metrics
- `reports/metrics_model_comparison.json` — model comparison metrics

---

## Dataset

Raw dataset file (stored locally):
`data/raw/Telco_customer_churn.xlsx`

> Note: raw/processed data is ignored via `.gitignore` (not pushed to GitHub).

---

## Setup

Create & activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

## Run the pipeline 
python src/01_load_and_inspect.py
python src/02_clean_data.py
python src/03_eda_plots.py
python -u src/04_train_baseline.py
python -u src/05_threshold_tuning.py
python -u src/06_feature_importance.py
python -u src/07_model_comparison.py

## Results (held-out test set)
Threshold was tuned to improve churn recall (catch more churners).

Metrics table (threshold = 0.35)

Model	Threshold	Accuracy	Precision (churn=1)	Recall (churn=1)	F1 (churn=1)	ROC-AUC
Logistic Regression	0.35	0.777	0.563	0.719	0.631	0.848
Random Forest	0.35	0.774	0.561	0.679	0.614	0.845

Conclusion: Logistic Regression slightly outperformed Random Forest on this split (higher recall/F1 and slightly higher ROC-AUC), while staying more interpretable.

Baseline confusion matrix (LogReg @ 0.35): [[826, 209], [105, 269]]
Format: [[TN, FP], [FN, TP]]