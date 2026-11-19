
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "datasets" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# 2. Load transaction dataset
print("Loading credit card dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Total transactions:", len(df))

# 3. Check missing values
print("\nMissing values:", df.isnull().sum().sum())

if df.isnull().sum().sum() > 0:
    raise ValueError("Dataset contains missing values. Clean it first.")

# 4. Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"].astype(int)

print("\nFeature count:", X.shape[1])

# 5. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 6. Create Random Forest model
print("\nInitializing Random Forest...")

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

# 7. Train model
print("\nTraining started...")

model.fit(X_train, y_train)

print("Training completed!")

# 8. Predict test transactions
y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

# 9. Evaluate model
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

roc_auc = roc_auc_score(y_test, y_probability)

pr_auc = average_precision_score(y_test, y_probability)

print("\n========== FRAUD DETECTION RESULTS ==========")

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))
print("Average Precision:", round(pr_auc, 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Fraud"],
    zero_division=0
))

# 10. Confusion matrix
cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Fraud"]
)

display.plot(cmap="Blues")

plt.title("Random Forest Fraud Detection")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "random_forest_confusion_matrix.png"
)

plt.close()

# 11. Save model
model_path = MODEL_DIR / "fraud_detection_model.pkl"

joblib.dump(model, model_path)

# 12. Save test predictions
results = X_test.copy()

results["Actual_Class"] = y_test
results["Predicted_Class"] = y_pred
results["Fraud_Probability"] = y_probability

results.to_csv(
    OUTPUT_DIR / "fraud_predictions.csv",
    index=False
)

# 13. Save evaluation metrics
metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "Average Precision"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        pr_auc
    ]
})

metrics.to_csv(
    OUTPUT_DIR / "fraud_model_metrics.csv",
    index=False
)

print("\nModel saved:", model_path)

print("\nPrediction results saved successfully!")

print("\nRandom Forest Fraud Detection Completed!")
