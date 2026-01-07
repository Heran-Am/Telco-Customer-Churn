from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/Telco_customer_churn.xlsx")

def list_sheets():
    with pd.ExcelFile(DATA_PATH) as xls:
        sheet_names = xls.sheet_names
        print("Available sheets:")
        for sheet in sheet_names:
            print(f" - {sheet}")

if __name__ == "__main__":
    list_sheets()