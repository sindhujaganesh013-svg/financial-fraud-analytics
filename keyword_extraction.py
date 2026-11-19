
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "outputs" / "financial_news_categorized.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

# 2. Load categorized news
df = pd.read_csv(INPUT_PATH)

print("Financial news dataset loaded!")
print("Total articles:", len(df))

# 3. Initialize TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)

# 4. Convert news into numerical features
tfidf_matrix = vectorizer.fit_transform(df["news"].fillna(""))

# 5. Extract keywords
feature_names = vectorizer.get_feature_names_out()

# Calculate average TF-IDF score
scores = tfidf_matrix.mean(axis=0).A1

# Create keyword dataframe
keyword_df = pd.DataFrame({
    "keyword": feature_names,
    "score": scores
})

# 6. Sort keywords by importance
keyword_df = keyword_df.sort_values(
    by="score",
    ascending=False
)

# 7. Display top 20 keywords
print("\n========== TOP 20 FINANCIAL KEYWORDS ==========")

print(keyword_df.head(20).to_string(index=False))

# 8. Save keyword results
keyword_df.to_csv(
    OUTPUT_DIR / "financial_keywords.csv",
    index=False
)

# 9. Visualize top 15 keywords
top_keywords = keyword_df.head(15).sort_values(
    by="score"
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_keywords["keyword"],
    top_keywords["score"]
)

plt.xlabel("Average TF-IDF Score")
plt.ylabel("Keywords")

plt.title("Top 15 Financial News Keywords")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "financial_keywords.png"
)

plt.close()

print("\nKeyword extraction completed!")
print("Results saved in outputs folder.")
