
import pandas as pd
from pathlib import Path

# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "outputs" / "financial_news_processed.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "financial_news_categorized.csv"

# 2. Load financial news
df = pd.read_csv(INPUT_PATH)

# 3. Define financial news categories
categories = {
    "Banking": [
        "bank", "loan", "lending", "deposit",
        "interest rate", "credit"
    ],

    "Stock Market": [
        "stock", "share price", "equity",
        "stock market", "investor", "shares"
    ],

    "Economy": [
        "inflation", "economy", "economic",
        "GDP", "unemployment", "recession"
    ],

    "Corporate": [
        "company", "corporation", "profit",
        "revenue", "sales", "business"
    ],

    "Cryptocurrency": [
        "bitcoin", "crypto", "cryptocurrency",
        "blockchain", "ethereum"
    ],

    "Fraud": [
        "fraud", "scam", "money laundering",
        "financial crime", "embezzlement"
    ]
}

# 4. Function to classify news
def classify_news(news):

    text = str(news).lower()

    for category, keywords in categories.items():

        for keyword in keywords:

            if keyword.lower() in text:
                return category

    return "Other"


# 5. Apply classification
df["category"] = df["news"].apply(classify_news)

# 6. Display results
print("\n===== FINANCIAL NEWS CATEGORY ANALYSIS =====")

print("\nFirst 10 records:")
print(df[["news", "category", "sentiment"]].head(10))

# 7. Category distribution
print("\nCategory Distribution:")
print(df["category"].value_counts())

# 8. Save categorized dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\nCategorization completed!")
print("Saved at:", OUTPUT_PATH)
