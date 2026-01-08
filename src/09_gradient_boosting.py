from pathlib import Path
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix

DATA_PATH = Path("data/processed/telco_clean.csv")
OUT_PATH = Path("reports/metrics_hist_gb.json")

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run src/02_clean_data.py first.")

    df = pd.read_csv(DATA_PATH)
    target = "Churn Value"

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # HistGradientBoosting only accepts numeric data -> one-hot encode categoricals
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])

    model = HistGradientBoostingClassifier(random_state=42)

    clf = Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])

    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    threshold = 0.35
    pred = (proba >= threshold).astype(int)

    metrics = {
        "model": "HistGradientBoostingClassifier",
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(metrics, indent=2))

    print("\n=== HIST GRADIENT BOOSTING ===")
    for k in ["threshold", "accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{k}: {metrics[k]}")
    print("confusion_matrix:", metrics["confusion_matrix"])
    print(f"\nSaved: {OUT_PATH}")

if __name__ == "__main__":
    main()
