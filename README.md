# Telco Customer Churn Prediction

Machine learning project using the IBM Telco Customer Churn dataset.

This project builds an end-to-end churn prediction pipeline using a Telco customer dataset.
Goal: predict whether a customer will churn (`Churn Value` = 1) and understand the drivers behind churn.

## Project structure

- `src/01_load_and_inspect.py` — load the raw Excel file and inspect columns/types/missing values
- `src/02_clean_data.py` — clean data and save a processed CSV (`data/processed/telco_clean.csv`)
- `src/03_eda_plots.py` — generate EDA plots into `reports/figures/`
- `src/04_train_baseline.py` — baseline Logistic Regression model with preprocessing + threshold tuning
- `src/05_threshold_tuning.py` — compare precision/recall/F1 for different probability thresholds
- `src/06_feature_importance.py` — interpret logistic regression coefficients (top churn drivers)

Folders:
- `data/raw/` — raw dataset (not committed)
- `data/processed/` — processed dataset (not committed)
- `reports/figures/` — saved plots
- `reports/metrics_baseline.json` — saved baseline metrics


## Dataset

Telco churn dataset (Excel). The raw file is stored locally under:
`data/raw/Telco_customer_churn.xlsx`

> Note: Raw/processed data is not committed to GitHub (see `.gitignore`).

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

## License
MIT License (see `LICENSE`).

