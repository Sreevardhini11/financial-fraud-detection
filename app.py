# app.py
import os
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import random
import time
from utils import preprocess

# --- Dataset & Model paths ---
DATASET_PATHS = {
    "Credit Card": "datasets/creditcard.csv",
    "Insurance Claims": "datasets/insurance_data.csv",
    "Loan Application": "datasets/loan_applications.csv",
    "Online Payment": "datasets/onlinefraud.csv"
}

MODEL_PATHS = {
    "Credit Card": "models/creditcard_model.pkl",
    "Insurance Claims": "models/insurance_model.pkl",
    "Loan Application": "models/loan_model.pkl",
    "Online Payment": "models/online_payment_model.pkl"
}

# --- Load datasets safely ---
datasets = {}
for name, path in DATASET_PATHS.items():
    if os.path.exists(path):
        try:
            datasets[name] = pd.read_csv(path)
        except Exception as e:
            st.warning(f"⚠️ Failed to load {name} dataset: {e}")
    else:
        st.warning(f"⚠️ Skipping {name} - dataset not found at {path}")

# --- Load models safely ---
models = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        try:
            models[name] = joblib.load(path)
        except Exception as e:
            st.warning(f"⚠️ Failed to load {name} model: {e}")
    else:
        st.warning(f"⚠️ Skipping {name} - model not found at {path}")

# --- Streamlit UI ---
st.title("🚨 Financial Fraud Detection Dashboard")

if not datasets:
    st.error("❌ No datasets available! Please check your 'datasets/' folder.")
    st.stop()

transaction_type = st.selectbox("Select Transaction Type", list(datasets.keys()))

df = datasets[transaction_type].copy()
model = models.get(transaction_type)

if model is None:
    st.error(f"❌ No trained model available for {transaction_type}. Please run train_models.py first.")
    st.stop()

# Drop target column if exists
target_col = "Class" if "Class" in df.columns else None
if target_col:
    df = df.drop(target_col, axis=1)

# Preprocess dataset
try:
    df_processed = preprocess(df)
except Exception as e:
    st.error(f"❌ Preprocessing failed: {e}")
    st.stop()

# Dataset overview
st.subheader("📊 Dataset Overview")
st.dataframe(df.head(10))
st.write(f"Shape: {df.shape}")

# Heatmap (numeric only)
st.subheader("🔥 Feature Correlation Heatmap")
numeric_df = df_processed.select_dtypes(include=["number"])
if not numeric_df.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)
else:
    st.info("No numeric columns available for correlation heatmap.")

# Summary stats
st.subheader("📈 Summary Statistics")
st.write(df.describe(include="all"))

# --- Real-time simulation ---
st.subheader("🚀 Real-Time Transaction Simulation")

if st.button("Start Simulation"):
    progress_text = st.empty()
    chart_placeholder = st.empty()
    table_placeholder = st.empty()
    bar_placeholder = st.empty()
    fraud_history = []
    suspicious_transactions = []

    total, fraud_count = 0, 0

    for _ in range(20):  # simulate 20 transactions
        idx = random.randint(0, len(df_processed) - 1)
        transaction = df_processed.iloc[idx]
        sample = pd.DataFrame([transaction.values], columns=df_processed.columns)

        # Prediction
        try:
            prediction = model.predict(sample)[0]
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            break

        total += 1
        if prediction == 1:
            fraud_count += 1
            suspicious_transactions.append(sample)

        fraud_percent = fraud_count / total * 100
        fraud_history.append(fraud_percent)

        display_df = sample.copy()
        display_df["Prediction"] = "🚨 Fraud" if prediction == 1 else "✅ Legit"
        table_placeholder.dataframe(display_df)

        chart_placeholder.line_chart(pd.DataFrame(fraud_history, columns=["Fraud %"]))
        bar_placeholder.bar_chart(pd.Series([prediction]).value_counts())

        progress_text.text(f"Processed: {total} | Fraud: {fraud_count} | Fraud %: {fraud_percent:.2f}")
        time.sleep(1)

    # Download suspicious transactions
    if suspicious_transactions:
        all_suspicious = pd.concat(suspicious_transactions)
        st.download_button(
            label="💾 Download Suspicious Transactions",
            data=all_suspicious.to_csv(index=False),
            file_name=f"suspicious_{transaction_type.lower().replace(' ','_')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No fraud detected in simulation.")
