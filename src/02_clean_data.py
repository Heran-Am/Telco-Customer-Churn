from pathlib import Path
import pandas as pd
RAW_PATH = Path("data/raw/Telco_customer_churn.xlsx")
OUT_PATH = Path("data/processed/telco_clean.csv")

def main():
    df = pd.read_excel(RAW_PATH)
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

    # 2) Choose target (what we predict)
    target = "Churn Value"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {df.columns.tolist()}")
    
    # 3) Drop leakage columns (these contain post-churn info)
    leakage_cols = ["Churn Label", "Churn Score", "Churn Reason"]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

    # 4) Drop ID/useless columns
    drop_cols = ["CustomerID", "Count"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # 3) Drop leakage columns (these contain post-churn info)
    leakage_cols = ["Churn Label", "Churn Score", "Churn Reason"]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

    # 4) Drop ID/useless columns
    drop_cols = ["CustomerID", "Count"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
     # 5) (Optional for v1) Drop high-cardinality geo columns to keep model simple
    geo_cols = ["Country", "State", "City", "Zip Code", "Lat Long"]
    df = df.drop(columns=[c for c in geo_cols if c in df.columns])

    # 6) Save processed data
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    # 7) Print a clean summary
    print("Saved:", OUT_PATH)
    print("Shape:", df.shape)
    print("Churn rate:", df[target].mean())
    print("Total Charges NaNs:", df["Total Charges"].isna().sum())

if __name__ == "__main__":
    main()
