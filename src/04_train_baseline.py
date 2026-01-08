from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

DATA_PATH = Path("data/processed/telco_clean.csv")
METRICS_PATH = Path("reports/metrics_baseline.json")


def main():
    # 0) Load data
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed file not found at {DATA_PATH}. Run src/02_clean_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    target = "Churn Value"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {df.columns.tolist()}")

    X = df.drop(columns=[target])
    y = df[target]

    # 1) Train/test split (stratify keeps churn ratio similar)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 2) Identify feature types
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # 3) Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    # 4) Model
    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=2000)),
    ])

    # 5) Train
    clf.fit(X_train, y_train)

    # 6) Predict + probabilities
    threshold =0.35 

    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # 7) Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== BASELINE RESULTS (Logistic Regression) ===")
    print("Accuracy:", accuracy)
    print("ROC-AUC:", roc_auc)
    print("\nConfusion matrix [ [TN FP], [FN TP] ]:\n", cm)
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    print("Threshold:", threshold)
    
    # 8) Save metrics to JSON
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),                          
        "confusion_matrix": cm.tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate_overall": float(y.mean()),
        "churn_rate_test": float(y_test.mean()),
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print("\nSaved metrics to:", METRICS_PATH)


if __name__ == "__main__":
    main()
