from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

DATA_PATH = Path("data/processed/telco_clean.csv")
OUT_PATH = Path("reports/metrics_model_comparison.json")

def make_preprocessor(X):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    num = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("num", num, numeric_features),
        ("cat", cat, categorical_features),
    ])

def eval_at_threshold(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
    }

def main():
    df = pd.read_csv(DATA_PATH)
    target = "Churn Value"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = make_preprocessor(X)

    models = {
        "logreg": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),
    }

    threshold = 0.35  # use your tuned threshold

    results = {}
    for name, model in models.items():
        clf = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ])
        clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_test)[:, 1]
        results[name] = eval_at_threshold(y_test, proba, threshold)

        print(f"\n=== {name.upper()} ===")
        for k, v in results[name].items():
            print(f"{k}: {v}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved comparison metrics to: {OUT_PATH}")

if __name__ == "__main__":
    main()
