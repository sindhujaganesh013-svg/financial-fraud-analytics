
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

NEWS_PATH = BASE_DIR / "outputs" / "financial_news_categorized.csv"

TRANSACTION_PATH = BASE_DIR / "outputs" / "anomaly_predictions.csv"

OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# 2. Load both datasets
print("Loading financial news and transaction results...")

news = pd.read_csv(NEWS_PATH)

transactions = pd.read_csv(TRANSACTION_PATH)

print("News records:", len(news))
print("Transaction records:", len(transactions))

# 3. Financial news category analysis
print("\n========== NEWS CATEGORY ANALYSIS ==========")

category_counts = news["category"].value_counts()

print(category_counts)

# 4. Sentiment analysis
print("\n========== SENTIMENT ANALYSIS ==========")

sentiment_counts = news["sentiment"].value_counts()

print(sentiment_counts)

# 5. Category and sentiment relationship
print("\n========== CATEGORY VS SENTIMENT ==========")

category_sentiment = pd.crosstab(
    news["category"],
    news["sentiment"]
)

print(category_sentiment)

category_sentiment.to_csv(
    OUTPUT_DIR / "category_sentiment_analysis.csv"
)

# 6. Transaction anomaly analysis
print("\n========== ANOMALY ANALYSIS ==========")

anomaly_counts = transactions[
    "Anomaly_Prediction"
].value_counts().sort_index()

print("Normal transactions:",
      anomaly_counts.get(0, 0))

print("Detected anomalies:",
      anomaly_counts.get(1, 0))

# 7. Analyse actual fraud among detected anomalies
if "Actual_Class" in transactions.columns:

    actual_fraud = transactions["Actual_Class"].sum()

    detected_fraud = (
        (transactions["Actual_Class"] == 1) &
        (transactions["Anomaly_Prediction"] == 1)
    ).sum()

    print("\nActual fraud cases:", actual_fraud)

    print("Fraud cases detected:", detected_fraud)

# 8. Transaction amount analysis
print("\n========== ANOMALOUS TRANSACTION AMOUNTS ==========")

anomalies = transactions[
    transactions["Anomaly_Prediction"] == 1
].copy()

if "Amount" in transactions.columns:

    print(anomalies["Amount"].describe())

# 9. Transaction time pattern
# Time is elapsed seconds, not calendar time.

if "Time" in transactions.columns:

    transactions["Elapsed_Hour"] = (
        transactions["Time"] // 3600
    )

    hourly_anomalies = (
        transactions[
            transactions["Anomaly_Prediction"] == 1
        ]
        .groupby("Elapsed_Hour")
        .size()
    )

    hourly_anomalies.to_csv(
        OUTPUT_DIR / "hourly_anomaly_patterns.csv"
    )

# 10. Visualize news categories
plt.figure(figsize=(10, 5))

sns.countplot(
    data=news,
    y="category",
    order=news["category"].value_counts().index
)

plt.title("Financial News Category Distribution")
plt.xlabel("Number of Articles")
plt.ylabel("News Category")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "news_category_analysis.png"
)

plt.close()

# 11. Visualize sentiment distribution
plt.figure(figsize=(8, 5))

sns.countplot(
    data=news,
    x="sentiment",
    order=["positive", "negative", "neutral"]
)

plt.title("Financial News Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Articles")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "news_sentiment_analysis.png"
)

plt.close()

# 12. Visualize category vs sentiment
category_sentiment.plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("News Category vs Sentiment")
plt.xlabel("News Category")
plt.ylabel("Number of Articles")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "category_sentiment_chart.png"
)

plt.close()

# 13. Visualize normal vs anomalous transactions
plt.figure(figsize=(8, 5))

sns.countplot(
    data=transactions,
    x="Anomaly_Prediction"
)

plt.title("Normal vs Anomalous Transactions")
plt.xlabel("0 = Normal, 1 = Anomaly")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "anomaly_distribution.png"
)

plt.close()

# 14. Visualize anomalous transaction amounts
if len(anomalies) > 0 and "Amount" in anomalies.columns:

    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=anomalies,
        x="Amount",
        bins=30
    )

    plt.title("Anomalous Transaction Amount Distribution")
    plt.xlabel("Transaction Amount")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "anomaly_amount_analysis.png"
    )

    plt.close()

# 15. Save summary
summary = pd.DataFrame({
    "Metric": [
        "Total News Articles",
        "Total Transactions Analysed",
        "Total News Categories",
        "Total Sentiment Types",
        "Detected Anomalies"
    ],
    "Value": [
        len(news),
        len(transactions),
        news["category"].nunique(),
        news["sentiment"].nunique(),
        int((transactions["Anomaly_Prediction"] == 1).sum())
    ]
})

summary.to_csv(
    OUTPUT_DIR / "pattern_discovery_summary.csv",
    index=False
)

# 16. Explain cross-dataset limitation
print("\n========== CROSS-DATASET ANALYSIS ==========")

print(
    "News and transaction patterns have been analysed separately."
)

print(
    "Direct temporal or company-level correlation cannot be "
    "established because the news dataset lacks dates and "
    "company identifiers."
)

print("\nPattern discovery completed successfully!")

print("All reports saved in outputs folder.")
