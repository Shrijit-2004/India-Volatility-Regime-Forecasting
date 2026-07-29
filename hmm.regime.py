import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

data = pd.read_csv("raw_market_data.csv", index_col="Date", parse_dates=True)
data["NSEI_ret"] = 100 * np.log(data["NSEI"] / data["NSEI"].shift(1))
data["realized_vol_5d"] = data["NSEI_ret"].rolling(window=5).std()
data = data.dropna()

X = data[["NSEI_ret", "realized_vol_5d"]].values

model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
model.fit(X)

hidden_states = model.predict(X)
data["regime"] = hidden_states

print("=== Regime Characteristics (mean return, mean volatility) ===")
regime_stats = data.groupby("regime")[["NSEI_ret", "realized_vol_5d"]].mean()
regime_stats["days_in_regime"] = data["regime"].value_counts().sort_index()
print(regime_stats)

vol_rank = regime_stats["realized_vol_5d"].rank(ascending=False)
crisis_state = vol_rank.idxmin()  
remaining = regime_stats.drop(crisis_state)
bull_state = remaining["NSEI_ret"].idxmax()
bear_state = remaining["NSEI_ret"].idxmin()

label_map = {
    crisis_state: "High Volatility / Crisis",
    bull_state: "Bull (Strong Uptrend)",
    bear_state: "Low-Momentum / Sideways"
}
data["regime_label"] = data["regime"].map(label_map)

print("\n=== Regime Labels Assigned ===")
print(label_map)

print("\n=== Days per regime ===")
print(data["regime_label"].value_counts())

plt.figure(figsize=(14, 6))
colors = {"Bull (Strong Uptrend)": "green", "Low-Momentum / Sideways": "red", "High Volatility / Crisis": "orange"}
for label, color in colors.items():
    mask = data["regime_label"] == label
    plt.scatter(data.index[mask], data["NSEI"][mask], color=color, label=label, s=8)

plt.title("Nifty 50 Price Colored by Detected Regime (HMM)")
plt.legend()
plt.tight_layout()
plt.savefig("hmm_regimes.png", dpi=150)
plt.show()

data[["NSEI", "NSEI_ret", "realized_vol_5d", "regime", "regime_label"]].to_csv("hmm_regime_labeled.csv")
print("\nSaved to hmm_regime_labeled.csv")