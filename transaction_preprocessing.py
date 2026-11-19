
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "datasets" / "creditcard.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# 2. Load transaction dataset
print("Loading credit card dataset...")

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")

# 3. Display dataset information
print("\n========== DATASET INFORMATION ==========")

print("Total transactions:", len(df))
print("Total columns:", len(df.columns))

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 transactions:")
print(df.head())

# 4. Check missing values
print("\n========== MISSING VALUE ANALYSIS ==========")

print(df.isnull().sum().sum())

# 5. Check duplicate records
print("\nDuplicate transactions:")

print(df.duplicated().sum())

# 6. Check fraud distribution
print("\n========== FRAUD DISTRIBUTION ==========")

class_counts = df["Class"].value_counts().sort_index()

print("Normal transactions:", class_counts.get(0, 0))
print("Fraudulent transactions:", class_counts.get(1, 0))

# 7. Calculate fraud percentage
fraud_percentage = (
    class_counts.get(1, 0) / len(df)
) * 100

print("\nFraud percentage:", round(fraud_percentage, 3), "%")

# 8. Transaction amount analysis
print("\n========== TRANSACTION AMOUNT ANALYSIS ==========")

print(df["Amount"].describe())

# 9. Visualize normal and fraudulent transactions
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Class"
)

plt.title("Normal vs Fraudulent Transactions")
plt.xlabel("Transaction Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "transaction_class_distribution.png"
)

plt.close()

# 10. Visualize transaction amounts
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Amount",
    bins=50
)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "transaction_amount_distribution.png"
)

plt.close()

# 11. Save dataset summary
summary = pd.DataFrame({
    "Metric": [
        "Total Transactions",
        "Normal Transactions",
        "Fraudulent Transactions",
        "Fraud Percentage",
        "Missing Values",
        "Duplicate Records"
    ],
    "Value": [
        len(df),
        class_counts.get(0, 0),
        class_counts.get(1, 0),
        round(fraud_percentage, 3),
        df.isnull().sum().sum(),
        df.duplicated().sum()
    ]
})

summary.to_csv(
    OUTPUT_DIR / "transaction_summary.csv",
    index=False
)

print("\nTransaction preprocessing analysis completed!")

print("Summary saved in outputs folder.")
