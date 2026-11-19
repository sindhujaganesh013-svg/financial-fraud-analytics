
import pandas as pd
from pathlib import Path

# Get project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATASET_PATH = BASE_DIR / "datasets" / "Sentences_AllAgree.txt"

# Output path
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Read financial news dataset
news_data = []

with open(DATASET_PATH, "r", encoding="utf-8", errors="replace") as file:

    for line in file:

        line = line.strip()

        if not line or "@" not in line:
            continue

        # Separate news sentence and sentiment
        sentence, sentiment = line.rsplit("@", 1)

        sentence = sentence.strip()
        sentiment = sentiment.strip().lower()

        if sentiment in ["positive", "negative", "neutral"]:
            news_data.append({
                "news": sentence,
                "sentiment": sentiment
            })

# Convert into DataFrame
df = pd.DataFrame(news_data)

# Remove missing values and duplicates
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Save processed dataset
output_file = OUTPUT_DIR / "financial_news_processed.csv"
df.to_csv(output_file, index=False)

# Display results
print("\n========== FINANCIAL NEWS PREPROCESSING ==========")

print("\nDataset Shape:", df.shape)

print("\nFirst 5 News Records:")
print(df.head())

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nProcessed dataset saved successfully!")
print("Output Location:", output_file)
