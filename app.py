import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =============================
# Load saved artifacts
# =============================
model = joblib.load("profitability_model.joblib")
model_features = joblib.load("profitability_model_features.joblib")

cluster_scaler = joblib.load("trader_cluster_scaler.joblib")
cluster_model = joblib.load("trader_cluster_model.joblib")

# ⚠️ HARD-CODED cluster features (DO NOT LOAD FROM FILE)
CLUSTER_FEATURES = [
    "avg_trades_per_day",
    "avg_leverage",
    "avg_long_ratio",
    "avg_win_rate",
    "pnl_volatility"
]

CLUSTER_NAMES = {
    0: "High-Leverage Active Trader",
    1: "Extreme Directional Trader",
    2: "High-Frequency Controlled Trader"
}

st.set_page_config(page_title="Trader Analytics Dashboard", layout="centered")
st.title("📊 Trader Analytics Dashboard")

# =============================
# SECTION 1: Profitability Prediction
# =============================
st.header("🔮 Predict Next-Day Profitability")
st.markdown("Enter **today's trading behavior** and **market sentiment**.")

input_data = {}

for col in model_features:
    if col.startswith("classification_"):
        input_data[col] = st.selectbox(
            label=col.replace("classification_", "Sentiment (0=Fear, 1=Greed):"),
            options=[0, 1],
            key=f"pred_{col}"
        )
    else:
        input_data[col] = st.number_input(
            label=col,
            value=0.0,
            key=f"pred_{col}"
        )

if st.button("Predict Next Day Outcome"):
    X_input = pd.DataFrame([input_data], columns=model_features)

    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][1]

    if pred == 1:
        st.success(f"✅ Likely PROFITABLE tomorrow (confidence: {prob:.2f})")
    else:
        st.error(f"⚠️ Likely NOT profitable tomorrow (confidence: {1 - prob:.2f})")

# =============================
# SECTION 2: Trader Behavior Clustering
# =============================
st.header("🧠 Identify Trader Behavior Type")
st.markdown("Enter **average behavior metrics** for a trader.")

cluster_input = {}

for col in CLUSTER_FEATURES:
    cluster_input[col] = st.number_input(
        label=col,
        value=0.0,
        key=f"cluster_{col}"
    )

if st.button("Identify Trader Type"):
    # IMPORTANT: enforce exact column order
    X_cluster = pd.DataFrame([cluster_input], columns=CLUSTER_FEATURES)

    X_scaled = cluster_scaler.transform(X_cluster)
    cluster_id = cluster_model.predict(X_scaled)[0]

    st.info(f"🧩 Trader Archetype: **{CLUSTER_NAMES[cluster_id]}**")
