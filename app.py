
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Financial Fraud Analytics",
    page_icon="💳",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

# ==========================================
# 2. CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD DATASETS
# ==========================================

def load_data(filename):
    file_path = OUTPUT_DIR / filename

    if file_path.exists():
        return pd.read_csv(file_path)

    return pd.DataFrame()


news = load_data("financial_news_categorized.csv")

keywords = load_data("financial_keywords.csv")

fraud = load_data("fraud_predictions.csv")

anomalies = load_data("anomaly_predictions.csv")

metrics = load_data("fraud_model_metrics.csv")

# ==========================================
# 4. SIDEBAR
# ==========================================

st.sidebar.title("💳 Financial Analytics")

st.sidebar.caption("Big Data Analytics Project")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Financial News Analytics",
        "Keyword Analysis",
        "Fraud Detection",
        "Anomaly Detection",
        "Model Evaluation"
    ]
)

# ==========================================
# 5. DASHBOARD OVERVIEW
# ==========================================

if page == "Dashboard Overview":

    st.title("Financial Fraud Analytics Dashboard")

    st.write(
        "Monitor financial news, sentiment, "
        "transaction patterns and suspicious activities."
    )

    total_news = len(news)

    total_transactions = len(fraud)

    total_anomalies = (
        int((anomalies["Anomaly_Prediction"] == 1).sum())
        if "Anomaly_Prediction" in anomalies.columns
        else 0
    )

    total_alerts = (
        int((fraud["Predicted_Class"] == 1).sum())
        if "Predicted_Class" in fraud.columns
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Financial News", f"{total_news:,}")

    c2.metric("Test Transactions", f"{total_transactions:,}")

    c3.metric("Fraud Alerts", f"{total_alerts:,}")

    c4.metric("Anomalies Detected", f"{total_anomalies:,}")

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Financial News Sentiment")

        if not news.empty and "sentiment" in news.columns:

            sentiment = (
                news["sentiment"]
                .value_counts()
                .reset_index()
            )

            sentiment.columns = ["Sentiment", "Count"]

            fig = px.pie(
                sentiment,
                names="Sentiment",
                values="Count",
                hole=0.45,
                title="Sentiment Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.warning("News data unavailable.")

    with right:

        st.subheader("Transaction Anomalies")

        if not anomalies.empty:

            counts = (
                anomalies["Anomaly_Prediction"]
                .map({0: "Normal", 1: "Anomaly"})
                .value_counts()
                .reset_index()
            )

            counts.columns = ["Status", "Count"]

            fig = px.bar(
                counts,
                x="Status",
                y="Count",
                color="Status",
                title="Normal vs Anomalous Transactions"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.warning("Anomaly data unavailable.")

    st.info(
        "News and transaction datasets are analysed separately. "
        "A direct causal relationship is not established."
    )

# ==========================================
# 6. FINANCIAL NEWS ANALYTICS
# ==========================================

elif page == "Financial News Analytics":

    st.title("📰 Financial News Analytics")

    if not news.empty:

        if "category" in news.columns:

            st.subheader("News Category Distribution")

            category_counts = (
                news["category"]
                .value_counts()
                .reset_index()
            )

            category_counts.columns = [
                "Category", "Articles"
            ]

            fig = px.bar(
                category_counts,
                x="Category",
                y="Articles",
                color="Category",
                title="Financial News Categories"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        if "sentiment" in news.columns:

            st.subheader("Sentiment Analysis")

            sentiment_counts = (
                news["sentiment"]
                .value_counts()
                .reset_index()
            )

            sentiment_counts.columns = [
                "Sentiment", "Count"
            ]

            fig = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Count",
                hole=0.4
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader("Financial News Records")

        filtered_news = news.copy()

        if "category" in news.columns:

            categories = [
                "All"
            ] + sorted(
                news["category"].dropna().unique().tolist()
            )

            selected = st.selectbox(
                "Select News Category",
                categories
            )

            if selected != "All":

                filtered_news = filtered_news[
                    filtered_news["category"] == selected
                ]

        st.dataframe(
            filtered_news,
            use_container_width=True
        )

    else:
        st.warning("Financial news dataset not found.")

# ==========================================
# 7. KEYWORD ANALYSIS
# ==========================================

elif page == "Keyword Analysis":

    st.title("🔍 Financial Keyword Analysis")

    if not keywords.empty:

        st.subheader("Top 20 Important Keywords")

        top = keywords.head(20).sort_values("score")

        fig = px.bar(
            top,
            x="score",
            y="keyword",
            orientation="h",
            title="Important Financial Keywords"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Keyword Details")

        st.dataframe(
            keywords.head(50),
            use_container_width=True
        )

    else:
        st.warning("Keyword dataset not found.")

# ==========================================
# 8. RANDOM FOREST FRAUD DETECTION
# ==========================================

elif page == "Fraud Detection":

    st.title("🚨 Random Forest Fraud Detection")

    if not fraud.empty:

        predicted_fraud = (
            fraud["Predicted_Class"] == 1
        ).sum()

        predicted_normal = (
            fraud["Predicted_Class"] == 0
        ).sum()

        c1, c2 = st.columns(2)

        c1.metric(
            "Normal Predictions",
            f"{predicted_normal:,}"
        )

        c2.metric(
            "Fraud Alerts",
            f"{predicted_fraud:,}"
        )

        st.subheader("Fraud Prediction Distribution")

        chart_data = pd.DataFrame({
            "Status": ["Normal", "Fraud"],
            "Count": [
                predicted_normal,
                predicted_fraud
            ]
        })

        fig = px.bar(
            chart_data,
            x="Status",
            y="Count",
            color="Status",
            title="Random Forest Predictions"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Suspicious Transactions")

        suspicious = fraud[
            fraud["Predicted_Class"] == 1
        ]

        display_columns = [
            col for col in [
                "Time",
                "Amount",
                "Actual_Class",
                "Predicted_Class",
                "Fraud_Probability"
            ]
            if col in suspicious.columns
        ]

        st.dataframe(
            suspicious[display_columns],
            use_container_width=True
        )

        st.download_button(
            "Download Fraud Alerts",
            suspicious.to_csv(index=False),
            "fraud_alerts.csv",
            "text/csv"
        )

    else:
        st.warning("Run fraud_detection.py first.")

# ==========================================
# 9. ISOLATION FOREST ANOMALY DETECTION
# ==========================================

elif page == "Anomaly Detection":

    st.title("⚠️ Anomaly Detection")

    if not anomalies.empty:

        anomaly_records = anomalies[
            anomalies["Anomaly_Prediction"] == 1
        ]

        normal_records = anomalies[
            anomalies["Anomaly_Prediction"] == 0
        ]

        c1, c2 = st.columns(2)

        c1.metric(
            "Normal Transactions",
            f"{len(normal_records):,}"
        )

        c2.metric(
            "Detected Anomalies",
            f"{len(anomaly_records):,}"
        )

        st.subheader("Anomaly Transaction Distribution")

        chart_data = pd.DataFrame({
            "Status": ["Normal", "Anomaly"],
            "Count": [
                len(normal_records),
                len(anomaly_records)
            ]
        })

        fig = px.pie(
            chart_data,
            names="Status",
            values="Count",
            hole=0.4
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Detected Anomaly Records")

        st.dataframe(
            anomaly_records,
            use_container_width=True
        )

        st.download_button(
            "Download Anomaly Alerts",
            anomaly_records.to_csv(index=False),
            "anomaly_alerts.csv",
            "text/csv"
        )

    else:
        st.warning("Run isolation_forest.py first.")

# ==========================================
# 10. MODEL EVALUATION
# ==========================================

elif page == "Model Evaluation":

    st.title("📊 Machine Learning Model Evaluation")

    if not metrics.empty:

        st.subheader("Random Forest Performance")

        st.dataframe(
            metrics,
            use_container_width=True
        )

        fig = px.bar(
            metrics,
            x="Metric",
            y="Value",
            title="Model Performance Metrics"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.warning("Model metrics file not found.")

    st.divider()

    st.subheader("Confusion Matrix Visualizations")

    images = {
        "Random Forest":
            "random_forest_confusion_matrix.png",

        "Isolation Forest":
            "isolation_forest_confusion_matrix.png",

        "News Sentiment":
            "sentiment_confusion_matrix.png"
    }

    for title, filename in images.items():

        image_path = OUTPUT_DIR / filename

        if image_path.exists():

            st.markdown(f"**{title}**")

            st.image(
                str(image_path),
                use_container_width=True
            )

# ==========================================
# 11. FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div style="text-align:center">
        <p>
        Financial News Analytics and Fraud Pattern Discovery
        </p>
        <p>Big Data Analytics Academic Project</p>
    </div>
    """,
    unsafe_allow_html=True
)
