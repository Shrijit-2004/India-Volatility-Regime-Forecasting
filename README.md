# 📈 India Volatility Regime Forecasting

An end-to-end financial data science project that forecasts stock market volatility and identifies market regimes using both classical econometric models and modern machine learning techniques.

## 📌 Project Overview

This project analyses the volatility of the Indian stock market (NIFTY 50 and SENSEX) by comparing traditional volatility models with machine learning approaches. It also detects Bull and Bear market regimes using Hidden Markov Models (HMM) to better understand market dynamics.

The objective is to evaluate which modelling approach provides the most accurate volatility forecasts while statistically analysing market behaviour.

---

## 🚀 Features

- Historical market data collection and preprocessing
- Exploratory Data Analysis (EDA)
- Volatility modelling using:
  - GARCH
  - EGARCH
- Machine Learning models:
  - XGBoost
  - LSTM Neural Network
- Hidden Markov Model (HMM) for market regime detection
- Model comparison using multiple evaluation metrics
- Visualisation of volatility forecasts and market regimes

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Plotly
- Scikit-learn
- TensorFlow / Keras
- XGBoost
- arch (GARCH Models)
- hmmlearn
- yfinance
- Streamlit

---

## 📊 Models Used

### Econometric Models
- GARCH(1,1)
- EGARCH

### Machine Learning Models
- XGBoost Regressor
- Long Short-Term Memory (LSTM)

### Regime Detection
- Hidden Markov Model (HMM)

---

## 📈 Evaluation Metrics

The forecasting models are compared using:

- RMSE
- MAE
- MAPE
- R² Score


---

## 📷 Sample Outputs

The repository contains:

- Volatility Forecast Graphs
- Market Regime Detection Plots
- Model Performance Comparison
- Dashboard Visualisations

---

## 💡 Key Findings

- Hidden Markov Models effectively classify Bull and Bear market regimes.
- EGARCH captures asymmetric market shocks (leverage effect).
- Machine Learning models (XGBoost and LSTM) improve forecasting accuracy compared with traditional econometric models.
- Volatility clustering is clearly observed in Indian equity markets.

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/Shrijit-2004/India-Volatility-Regime-Forecasting.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

---

## 🎯 Future Improvements

- Transformer-based Time Series Models
- Real-time Data Pipeline
- Portfolio Risk Analysis
- Options Implied Volatility
- Explainable AI (SHAP)

---

## 👨‍💻 Author

**Shrijit Ghosh**

B.Sc. Economics (Hons.) 

Interested in Financial Data Science, Quantitative Finance, Risk Analytics, Machine Learning, and Time Series Forecasting.

---

## ⭐ If you found this project useful, consider giving it a Star!
