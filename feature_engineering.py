import pandas as pd
import numpy as np

data = pd.read_csv("raw_market_data.csv", index_col="Date", parse_dates=True)

egarch_vol = pd.read_csv("garch_egarch_volatility_nsei.csv", index_col="Date", parse_dates=True)

data["NSEI_ret"] = 100 * np.log(data["NSEI"] / data["NSEI"].shift(1))

data["realized_vol_5d"] = data["NSEI_ret"].rolling(window=5).std()
data["target_realized_vol"] = data["realized_vol_5d"].shift(-5)

data["target_vix"] = data["INDIAVIX"].shift(-1)

for lag in [1, 2, 3, 5]:
    data[f"ret_lag{lag}"] = data["NSEI_ret"].shift(lag)
    data[f"realized_vol_lag{lag}"] = data["realized_vol_5d"].shift(lag)
    data[f"vix_lag{lag}"] = data["INDIAVIX"].shift(lag)

data = data.join(egarch_vol[["EGARCH_vol"]])
data["egarch_vol_lag1"] = data["EGARCH_vol"].shift(1)

feature_cols = [c for c in data.columns if "lag" in c]
model_data = data[feature_cols + ["target_realized_vol", "target_vix"]].dropna()

print("Feature set shape:", model_data.shape)
print("\nColumns:", list(model_data.columns))
print("\nFirst few rows:")
print(model_data.head())

model_data.to_csv("ml_features.csv")
print("\nSaved to ml_features.csv")