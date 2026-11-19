# Financial Fraud Analytics: Big Data Laboratory Project

## 1. Abstract

This laboratory project implements an end-to-end financial analytics system using Python, scikit-learn, pandas, Streamlit, and Matplotlib/Seaborn. It processes financial news, extracts sentiment and keywords, classifies news into financial categories, detects credit-card fraud with Random Forest, identifies unusual transactions with Isolation Forest, and presents the generated reports in an interactive dashboard.

The project uses two independent datasets:

- `datasets/Sentences_AllAgree.txt` for financial news.
- `datasets/creditcard.csv` for credit-card transactions.

The news and transaction analyses are deliberately kept separate. The available datasets do not provide a common date or company identifier, so the project does not claim that news sentiment directly causes transaction anomalies.

## 2. Laboratory Objectives

1. Preprocess and clean financial news text.
2. Perform three-class sentiment analysis: positive, negative, and neutral.
3. Classify financial news using keyword rules.
4. Extract important financial keywords with TF-IDF.
5. Analyze credit-card transaction data for class imbalance and missing values.
6. Train a supervised Random Forest fraud classifier.
7. Train an unsupervised Isolation Forest anomaly detector.
8. Compare model predictions and evaluation metrics.
9. Produce reusable CSV reports and visualizations.
10. Build a Streamlit dashboard for results and data exploration.

## 3. System Architecture

```text
datasets/
    Sentences_AllAgree.txt
    creditcard.csv
          |
          v
src/ preprocessing and analysis scripts
          |
          +--> models/ trained machine-learning models
          |
          +--> outputs/ CSV reports and chart images
                          |
                          v
                       app.py
                          |
                          v
                  Streamlit dashboard
```

The project follows this workflow:

1. News text is cleaned and converted into structured records.
2. A TF-IDF model represents text numerically.
3. Logistic Regression predicts the news sentiment.
4. Keyword rules assign a broad financial category.
5. Transaction features and fraud labels are prepared for model training.
6. Random Forest performs supervised fraud classification.
7. Isolation Forest performs unsupervised anomaly detection.
8. CSV and image outputs are loaded by the dashboard.

## 4. Project Structure

| Path | Purpose |
|---|---|
| `app.py` | Streamlit analytics dashboard |
| `requirements.txt` | Python dependencies |
| `datasets/Sentences_AllAgree.txt` | Financial news sentences and sentiment labels |
| `datasets/creditcard.csv` | Credit-card transaction records and fraud label |
| `src/news_preprocessing.py` | Parses and cleans financial news |
| `src/news_classification.py` | Trains the sentiment model |
| `src/news_category_classification.py` | Applies rule-based news categories |
| `src/keyword_extraction.py` | Extracts TF-IDF keywords |
| `src/transaction_preprocessing.py` | Summarizes and plots transactions |
| `src/fraud_detection.py` | Trains and evaluates Random Forest |
| `src/isolation_forest.py` | Trains and evaluates Isolation Forest |
| `src/pattern_discovery.py` | Generates cross-module summary reports |
| `models/` | Serialized trained models |
| `outputs/` | Generated datasets, metrics, and visualizations |

## 5. Dataset Description

### 5.1 Financial News Dataset

The source file contains a news sentence and a sentiment label separated by `@`. Valid labels are:

- `positive`
- `negative`
- `neutral`

The preprocessing module removes empty records and duplicates and saves `outputs/financial_news_processed.csv`.

The processed data contains 2,259 records:

| Sentiment | Record count |
|---|---:|
| Neutral | 1,386 |
| Positive | 570 |
| Negative | 303 |

### 5.2 Credit-Card Transaction Dataset

The transaction dataset contains 284,807 rows and 31 features. The `Class` column is the target:

- `0` indicates a normal transaction.
- `1` indicates a fraudulent transaction.

The preprocessing summary reports:

| Measure | Value |
|---|---:|
| Total transactions | 284,807 |
| Normal transactions | 284,315 |
| Fraudulent transactions | 492 |
| Fraud percentage | 0.173% |
| Missing values | 0 |
| Duplicate records | 1,081 |

The small fraud percentage makes accuracy alone insufficient. Precision, recall, F1, ROC-AUC, and average precision are therefore used for evaluation.

## 6. Experimental Modules

### 6.1 News Preprocessing

Run:

```powershell
python src/news_preprocessing.py
```

The module parses each line, separates the sentence from the label, normalizes the label, removes duplicates and missing values, and creates `financial_news_processed.csv`.

### 6.2 News Sentiment Classification

Run:

```powershell
python src/news_classification.py
```

This module uses:

- TF-IDF text vectorization.
- Unigrams and bigrams.
- English stop-word removal.
- A maximum of 10,000 features.
- Logistic Regression with balanced class weights.

The data is split using 80% for training and 20% for testing. A fixed random seed provides reproducibility.

The generated outputs include:

- `outputs/sentiment_confusion_matrix.png`
- `models/news_sentiment_model.pkl`

### 6.3 News Category Classification

Run:

```powershell
python src/news_category_classification.py
```

Category names and example keywords are:

| Category | Example keywords |
|---|---|
| Banking | bank, loan, lending, deposit, credit |
| Stock Market | stock, share price, equity, investor |
| Economy | inflation, economy, GDP, recession |
| Corporate | company, profit, revenue, sales, business |
| Cryptocurrency | bitcoin, crypto, blockchain, ethereum |
| Fraud | fraud, scam, money laundering, embezzlement |

The classification is keyword based and assigns `Other` when no rule matches. The output is `outputs/financial_news_categorized.csv`.

### 6.4 Financial Keyword Extraction

Run:

```powershell
python src/keyword_extraction.py
```

TF-IDF is applied to all news text. The average TF-IDF score is calculated for every unigram and bigram, and terms are ranked in descending order.

Generated files:

- `outputs/financial_keywords.csv`
- `outputs/financial_keywords.png`

Frequently prominent terms in the current generated data include `eur`, `mn`, `company`, `profit`, `sales`, `year`, `million`, `net`, `operating`, and `eur mn`.

### 6.5 Transaction Preprocessing

Run:

```powershell
python src/transaction_preprocessing.py
```

This module calculates dataset shape, missing values, duplicates, fraud distribution, fraud percentage, and transaction amount statistics. It produces:

- `outputs/transaction_summary.csv`
- `outputs/transaction_class_distribution.png`
- `outputs/transaction_amount_distribution.png`

### 6.6 Random Forest Fraud Detection

Run:

```powershell
python src/fraud_detection.py
```

The module:

1. Loads `creditcard.csv`.
2. Removes `Class` to form feature matrix `X`.
3. Uses `Class` as target `y`.
4. Splits the data 80/20 using stratification.
5. Trains 150 balanced Random Forest trees.
6. Predicts normal or fraud for test transactions.
7. Saves probabilities and evaluation metrics.

The test split contains 56,962 transactions. Current generated metrics are:

| Metric | Result |
|---|---:|
| Accuracy | 99.95% |
| Precision | 0.9494 |
| Recall | 0.7653 |
| F1 Score | 0.8475 |
| ROC-AUC | 0.9467 |
| Average Precision | 0.8535 |

The model generated 79 fraud alerts in the test predictions.

Generated files:

- `models/fraud_detection_model.pkl`
- `outputs/fraud_predictions.csv`
- `outputs/fraud_model_metrics.csv`
- `outputs/random_forest_confusion_matrix.png`

For an imbalanced dataset, recall is especially important because it measures how many actual fraud cases the model detects. Precision indicates how reliable each fraud alert is.

### 6.7 Isolation Forest Anomaly Detection

Run:

```powershell
python src/isolation_forest.py
```

This module:

1. Trains only with normal transactions.
2. Standardizes all features through a pipeline.
3. Uses 150 Isolation Forest estimators.
4. Converts raw `-1/1` outputs to `1/0` anomaly flags.
5. Saves anomaly scores and predictions.

The configured contamination value is `0.00173`. On the test split, the model generated 129 anomaly alerts.

Generated files:

- `models/isolation_forest_model.pkl`
- `outputs/anomaly_predictions.csv`
- `outputs/isolation_forest_confusion_matrix.png`

### 6.8 Pattern Discovery

Run:

```powershell
python src/pattern_discovery.py
```

This module combines the generated news and transaction summaries:

- Counts news categories and sentiments.
- Creates category-versus-sentiment tables.
- Counts normal and anomalous transactions.
- Calculates anomalous transaction amount statistics.
- Converts elapsed transaction time into elapsed-hour groups.
- Produces category, sentiment, anomaly, and amount visualizations.

It saves `outputs/pattern_discovery_summary.csv` and `outputs/category_sentiment_analysis.csv`, as well as hourly anomaly and chart outputs.

Cross-dataset correlation is not reported because the two datasets do not share timestamps or company identifiers.

## 7. Model Evaluation Metrics

For binary fraud detection:

- **Accuracy** = correct predictions / total predictions.
- **Precision** = true fraud / (true fraud + false fraud alerts).
- **Recall** = true fraud / (true fraud + missed fraud).
- **F1** = harmonic mean of precision and recall.
- **ROC-AUC** = model ranking quality across classification thresholds.
- **Average Precision** = precision-recall summary for an imbalanced classification task.

Accuracy may appear high even when the model misses fraud cases. Recall and average precision provide stronger evidence in this laboratory.

## 8. Installation

### 8.1 Requirements

- Windows 10/11
- Python 3.9 or later
- PowerShell
- Internet access only if packages are not already installed

Required packages are listed in `requirements.txt`:

- `joblib`
- `matplotlib`
- `numpy`
- `pandas`
- `plotly`
- `scikit-learn`
- `seaborn`
- `streamlit`

### 8.2 Create the Environment

Open PowerShell in the project root:

```powershell
cd C:\BigData
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script, enable scripts for the current user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## 9. Complete Execution Procedure

Run the scripts from the repository root in this order:

```powershell
python src/news_preprocessing.py
python src/news_classification.py
python src/news_category_classification.py
python src/keyword_extraction.py
python src/transaction_preprocessing.py
python src/fraud_detection.py
python src/isolation_forest.py
python src/pattern_discovery.py
```

The generated dashboard can then be started:

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, typically `http://localhost:8501`.

The dashboard also works from the existing files in `outputs/`, so it is not necessary to retrain the models before viewing the stored reports.

## 10. Dashboard Features

The sidebar provides six pages:

| Page | Available features |
|---|---|
| Dashboard Overview | News, transaction, fraud, and anomaly totals; sentiment and anomaly charts |
| Financial News Analytics | Category distribution, sentiment distribution, category filtering, article table |
| Keyword Analysis | Top TF-IDF terms and detailed keyword table |
| Fraud Detection | Random Forest counts, fraud chart, suspicious transaction records, CSV download |
| Anomaly Detection | Isolation Forest counts, anomaly chart, anomaly records, CSV download |
| Model Evaluation | Fraud metrics, confusion-matrix images |

## 11. Generated Output Index

| File | Description |
|---|---|
| `financial_news_processed.csv` | Cleaned news records |
| `financial_news_categorized.csv` | News with category and sentiment |
| `financial_keywords.csv` | Ranked TF-IDF keywords |
| `financial_keywords.png` | Top keyword chart |
| `transaction_summary.csv` | Transaction dataset summary |
| `fraud_predictions.csv` | Test predictions and fraud probabilities |
| `fraud_model_metrics.csv` | Random Forest metrics |
| `anomaly_predictions.csv` | Test anomalies and anomaly scores |
| `category_sentiment_analysis.csv` | Category-versus-sentiment counts |
| `hourly_anomaly_patterns.csv` | Anomalies grouped by elapsed hour |
| `pattern_discovery_summary.csv` | Combined project summary |
| `random_forest_confusion_matrix.png` | Random Forest confusion matrix |
| `isolation_forest_confusion_matrix.png` | Isolation Forest confusion matrix |
| `sentiment_confusion_matrix.png` | News sentiment confusion matrix |

The remaining PNG files contain charts for news sentiment, category distributions, transaction distributions, anomalous amounts, and category/sentiment relationships.

## 12. Experimental Results and Interpretation

The generated results show 2,259 processed financial news records and 56,962 test transactions in the saved transaction-analysis data. Neutral is the largest sentiment class. Corporate and Stock Market are the largest explicitly matched news categories in the generated rule-based output.

The Random Forest model achieved 99.95% accuracy and 94.67% ROC-AUC. Precision was 0.9494 and recall was 0.7653, meaning most alerts were correct but some actual fraud cases were not detected. The Isolation Forest produced 129 anomaly predictions in the test split. It is an unsupervised screening approach and is not directly equivalent to the supervised fraud labels.

These figures describe the current generated outputs. Re-running scripts with a different dataset, Python environment, library version, or data split may change the results.

## 13. Limitations

- The news category classifier uses fixed keywords and may produce incorrect or `Other` categories.
- The news sentiment model is based on one historical labeled dataset.
- Credit-card feature names `V1` to `V28` are anonymized.
- Isolation Forest detects general anomalies, not necessarily fraud.
- High accuracy may hide poor performance on rare fraud cases.
- The news and transaction datasets have no shared entity or timestamp for direct correlation.
- The stored transaction results should not be interpreted as real-time fraud monitoring.

## 14. Conclusion

The project demonstrates a complete financial analytics pipeline from raw data to trained models, generated reports, and an interactive dashboard. Random Forest provides a supervised fraud classification baseline, while Isolation Forest provides an unsupervised anomaly-screening baseline. The news modules add text preprocessing, sentiment prediction, category classification, and keyword extraction. The dashboard connects these results into one accessible laboratory application.

## 15. Troubleshooting

### Missing credit-card dataset

Place a compatible file containing a `Class` column at:

```text
datasets\creditcard.csv
```

### Missing news dataset

Ensure `datasets\Sentences_AllAgree.txt` exists in the project.

### Missing model or output files

Run the relevant script from the project root. For all files, execute the complete script sequence in Section 9.

### Streamlit cannot find `app.py`

Run the command while the terminal working directory is `C:\BigData`:

```powershell
streamlit run app.py
```
