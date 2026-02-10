# Trader Profitability and Behavioral Clustering Dashboard

## Overview
This project analyzes trader behavior and performance using historical trading data combined with market sentiment (Fear & Greed Index). The goal is to understand how sentiment influences trading outcomes, identify distinct trader behavior patterns, and translate these insights into an interactive dashboard.

Live App:  
https://traderprofitabilityandclusteringmodel-qzexkse5jtxwoer2xgbz8f.streamlit.app/

---

## Objectives
- Analyze performance differences between Fear and Greed market conditions.
- Study how trader behavior (frequency, leverage, direction) changes with sentiment.
- Segment traders into behavioral archetypes using clustering.
- Provide an interactive dashboard for prediction and interpretation.

---

## Dashboard Functionality

### 1. Next-Day Profitability Prediction
Predicts whether a trader is likely to be profitable on the next trading day using:
- Trade frequency
- Leverage usage
- Win rate
- Directional bias
- Market sentiment (Fear or Greed)

A logistic regression model is used to learn general behavior-to-outcome patterns rather than memorizing individual traders. The output includes a binary prediction and a confidence score.

---

### 2. Trader Behavioral Archetypes
Traders are classified into one of three behavioral archetypes based on average trading behavior:

- **High-Leverage Active Trader**  
  Trades regularly with very high leverage and elevated risk.

- **Extreme Directional Trader**  
  Takes highly directional positions with extreme leverage and high volatility.

- **High-Frequency Controlled Trader**  
  Trades very frequently while maintaining comparatively better risk control and stability.

KMeans clustering is applied to scaled behavioral features to derive these groups.

---

## Models and Artifacts
The dashboard loads the following trained artifacts using joblib:
- Logistic regression model for next-day profitability prediction
- Feature list used by the prediction model
- KMeans clustering model for trader archetypes
- Scaler used to normalize trader behavior features before clustering

---

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## Key Insights
- Greed periods are associated with higher downside risk without improved average performance.
- Fear periods tend to show more disciplined and selective trading behavior.
- Higher leverage and higher activity do not guarantee better outcomes.
- Behavioral segmentation provides clearer insight than performance metrics alone.

---

## Notes
- This project is intended for analytical and educational purposes only.
- Model predictions are probabilistic and should not be considered financial advice.
- Emphasis is placed on interpretability and sound analytical reasoning rather than production-grade forecasting.

---

## Author
Souhardya Das
