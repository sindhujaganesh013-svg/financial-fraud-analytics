
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "outputs" / "financial_news_processed.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# 2. Load processed news dataset
df = pd.read_csv(DATA_PATH)

X = df["news"]
y = df["sentiment"]

print("Dataset loaded successfully!")
print("Total records:", len(df))


# 3. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# 4. Create TF-IDF and Logistic Regression pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2)
    )),

    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])


# 5. Train the model
print("\nTraining the model...")

model.fit(X_train, y_train)

print("Model training completed!")


# 6. Predict sentiment
y_pred = model.predict(X_test)


# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))


# 8. Generate confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["negative", "neutral", "positive"]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Neutral", "Positive"]
)

display.plot(cmap="Blues")

plt.title("Financial News Sentiment Confusion Matrix")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "sentiment_confusion_matrix.png")

plt.close()


# 9. Save trained model
model_path = MODEL_DIR / "news_sentiment_model.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Location:", model_path)


# 10. Test with new financial news
test_news = [
    "The company reported a significant increase in profits.",
    "The company suffered a major financial loss.",
    "The company announced its annual general meeting."
]

predictions = model.predict(test_news)

print("\n========== SAMPLE PREDICTIONS ==========")

for news, sentiment in zip(test_news, predictions):

    print("\nNews:", news)
    print("Predicted Sentiment:", sentiment)

print("\nModule 2 completed successfully!")
