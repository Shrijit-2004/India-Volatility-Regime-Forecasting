import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("raw_market_data.csv", index_col="Date", parse_dates=True)

returns = np.log(data[["NSEI", "BSESN"]] / data[["NSEI", "BSESN"]].shift(1))
returns = returns.dropna()
returns.columns = ["NSEI_ret", "BSESN_ret"]

print("=== Summary Statistics (Log Returns) ===")
print(returns.describe())

print("\n=== Skewness & Kurtosis ===")
print("Skewness:\n", returns.skew())
print("\nKurtosis:\n", returns.kurt())

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(data.index, data["NSEI"], label="NSEI (Nifty 50)", color="navy")
axes[0].set_title("Nifty 50 - Price Level")
axes[0].legend()

axes[1].plot(returns.index, returns["NSEI_ret"], color="darkred", linewidth=0.7)
axes[1].set_title("Nifty 50 - Daily Log Returns (look for volatility clustering)")

axes[2].plot(data.index, data["INDIAVIX"], color="darkgreen")
axes[2].set_title("India VIX - Implied Volatility Index")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150)
plt.show()

print("\nSaved plots to eda_plots.png")