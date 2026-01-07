from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path("data/processed/telco_clean.csv")
FIG_DIR = Path("reports/figures")

def churn_rate_by(df, col):
    rates = df.groupby(col)["Churn Value"].mean().sort_values(ascending=False)
    ax = rates.plot(kind="bar")
    ax.set_title(f"Churn rate by {col}")
    ax.set_ylabel("Churn rate")
    ax.set_xlabel(col)
    plt.tight_layout()
    return ax

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing processed file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loaded:", DATA_PATH, "shape:", df.shape)
    print("Churn rate:", df["Churn Value"].mean())

    # 1) Churn rate by Contract
    plt.figure()
    churn_rate_by(df, "Contract")
    plt.savefig(FIG_DIR / "churn_by_contract.png")
    plt.close()

    # 2) Churn rate by Internet Service
    plt.figure()
    churn_rate_by(df, "Internet Service")
    plt.savefig(FIG_DIR / "churn_by_internet_service.png")
    plt.close()

    # 3) Churn rate by Payment Method
    plt.figure()
    churn_rate_by(df, "Payment Method")
    plt.savefig(FIG_DIR / "churn_by_payment_method.png")
    plt.close()

    # 4) Tenure distribution (churn vs non-churn)
    plt.figure()
    df[df["Churn Value"] == 0]["Tenure Months"].hist(alpha=0.6, bins=30, label="No churn")
    df[df["Churn Value"] == 1]["Tenure Months"].hist(alpha=0.6, bins=30, label="Churn")
    plt.title("Tenure Months distribution (churn vs non-churn)")
    plt.xlabel("Tenure Months")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "tenure_hist_churn_vs_not.png")
    plt.close()

    # 5) Monthly Charges distribution (churn vs non-churn)
    plt.figure()
    df[df["Churn Value"] == 0]["Monthly Charges"].hist(alpha=0.6, bins=30, label="No churn")
    df[df["Churn Value"] == 1]["Monthly Charges"].hist(alpha=0.6, bins=30, label="Churn")
    plt.title("Monthly Charges distribution (churn vs non-churn)")
    plt.xlabel("Monthly Charges")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "monthly_charges_hist_churn_vs_not.png")
    plt.close()

    # 6) Correlation heatmap for numeric columns (simple)
    num_cols = df.select_dtypes(include="number").columns
    corr = df[num_cols].corr()

    plt.figure(figsize=(8, 6))
    plt.imshow(corr, aspect="auto")
    plt.xticks(range(len(num_cols)), num_cols, rotation=90)
    plt.yticks(range(len(num_cols)), num_cols)
    plt.title("Correlation (numeric features)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_numeric.png")
    plt.close()

    print("Saved plots to:", FIG_DIR)

if __name__ == "__main__":
    main()
