
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "datasets" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# 2. Load dataset
print("Loading transaction dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)

# 3. Separate features and actual labels
X = df.drop("Class", axis=1)
y = df["Class"].astype(int)

# 4. Split dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Use only normal training transactions
X_train_normal = X_train[y_train == 0]

print("\nNormal training records:", len(X_train_normal))

# 6. Create Isolation Forest pipeline
model = Pipeline([
    ("scaler", StandardScaler()),

    ("isolation_forest", IsolationForest(
        n_estimators=150,
        contamination=0.00173,
        random_state=42,
        n_jobs=-1
    ))
])

# 7. Train the model
print("\nTraining Isolation Forest...")

model.fit(X_train_normal)

print("Training completed!")

# 8. Predict test transactions
raw_predictions = model.predict(X_test)

# Isolation Forest:
#  1 = Normal
# -1 = Anomaly

# Convert predictions into:
#  0 = Normal
#  1 = Anomaly

y_pred = np.where(raw_predictions == -1, 1, 0)

# 9. Calculate anomaly scores
anomaly_scores = -model.decision_function(X_test)

# 10. Evaluate model
print("\n========== ISOLATION FOREST RESULTS ==========")

print("Actual fraud transactions:", int(y_test.sum()))
print("Detected anomalies:", int(y_pred.sum()))

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

average_precision = average_precision_score(
    y_test, anomaly_scores
)

print("\nPrecision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("Average Precision:", round(average_precision, 4))

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Anomaly"],
    zero_division=0
))

# 11. Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

plt.imshow(cm, cmap="Blues")

plt.title("Isolation Forest Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks([0, 1], ["Normal", "Anomaly"])
plt.yticks([0, 1], ["Normal", "Fraud"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center")

plt.colorbar()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "isolation_forest_confusion_matrix.png"
)

plt.close()

# 12. Save anomaly predictions
results = X_test.copy()

results["Actual_Class"] = y_test
results["Anomaly_Prediction"] = y_pred
results["Anomaly_Score"] = anomaly_scores

results.to_csv(
    OUTPUT_DIR / "anomaly_predictions.csv",
    index=False
)

# 13. Save model
joblib.dump(
    model,
    MODEL_DIR / "isolation_forest_model.pkl"
)

print("\nModel saved successfully!")

print("Isolation Forest module completed!")
