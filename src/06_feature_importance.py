from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

DATA_PATH = Path("data/processed/telco_clean.csv")

def main():
    df = pd.read_csv(DATA_PATH)
    target = "Churn Value"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

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

    model = LogisticRegression(max_iter=2000)

    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    clf.fit(X_train, y_train)

    # Get feature names after preprocessing
    ohe = clf.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(categorical_features)

    feature_names = list(numeric_features) + list(cat_feature_names)

    # Coefficients (positive -> increases churn odds, negative -> decreases)
    coefs = clf.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    coef_df = coef_df.sort_values("coef", ascending=False)

    print("\n=== Top 15 features that INCREASE churn risk ===")
    print(coef_df.head(15).to_string(index=False))

    print("\n=== Top 15 features that DECREASE churn risk ===")
    print(coef_df.tail(15).sort_values("coef").to_string(index=False))

if __name__ == "__main__":
    main()

