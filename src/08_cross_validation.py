from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

DATA_PATH = Path("data/processed/telco_clean.csv")
OUT_PATH = Path("reports/metrics_cross_validation.json")

def build_pipeline(X: pd.DataFrame) -> Pipeline:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])

    model = LogisticRegression(max_iter=2000)

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run src/02_clean_data.py first.")

    df = pd.read_csv(DATA_PATH)
    target = "Churn Value"

    X = df.drop(columns=[target])
    y = df[target]

    clf = build_pipeline(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    scores = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    summary = {}
    for k, v in scores.items():
        if k.startswith("test_"):
            metric = k.replace("test_", "")
            summary[metric] = {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "folds": [float(x) for x in v],
            }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))

    print("\n=== 5-Fold Cross-Validation (Logistic Regression) ===")
    for metric, stats in summary.items():
        print(f"{metric:9s}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    print(f"\nSaved: {OUT_PATH}")

if __name__ == "__main__":
    main()
