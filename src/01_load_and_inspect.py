from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/raw/Telco_customer_churn.xlsx")

SHEET_NAME = "Telco_Churn"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {DATA_PATH.resolve()}")
    # Load the dataset
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    
    # ✅ FIX TYPES RIGHT AFTER LOADING (correct column name!)
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
        print("Total Charges NaNs after conversion:", df["Total Charges"].isna().sum())

    # Inspect the dataset
    print("First 5 rows of the dataset:")
    print(df.head())
    
    print("\nDataset information:")
    print(df.info())
    
    print("\nStatistical summary of numerical columns:")
    print(df.describe())
    
    print("\n=== MISSING VALUES(Top 15)===")
    missing = df.isna().sum().sort_values(ascending=False).head(15)
    print(missing)
    
    if "Churn" in df.columns:
        print("\n=== TARGET DISTRIBUTION (Churn) ===")
        print(df["Churn"].value_counts(dropna=False))
        print(df["Churn Value"].value_counts())
        print("Churn rate:", df["Churn Value"].mean())

        churn_rate = (df["Churn"].astype(str).str.lower() == "yes").mean()
        print("Churn rate:", churn_rate)

    # Common issue: TotalCharges may be text / blanks
    if "TotalCharges" in df.columns:
        tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
        print("\n=== TotalCharges conversion check ===")
        print("Converted-to-NaN count:", tc.isna().sum())

if __name__ == "__main__":
    main()
