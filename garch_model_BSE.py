import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt

data = pd.read_csv("raw_market_data.csv", index_col="Date", parse_dates=True)
returns = 100 * np.log(data["NSEI"] / data["NSEI"].shift(1)).dropna()
garch = arch_model(returns, vol="GARCH", p=1, q=1, dist="normal")
garch_fit = garch.fit(disp="off")

print("=========== GARCH(1,1) Results ===========")
print(garch_fit.summary())

egarch = arch_model(returns, vol="EGARCH", p=1, o=1, q=1, dist="normal")
egarch_fit = egarch.fit(disp="off")

print("\n=========== EGARCH(1,1) Results ===========")
print(egarch_fit.summary())

print("\n=========== Model Comparison ===========")
print(f"GARCH  AIC: {garch_fit.aic:.2f}   BIC: {garch_fit.bic:.2f}")
print(f"EGARCH AIC: {egarch_fit.aic:.2f}   BIC: {egarch_fit.bic:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(garch_fit.conditional_volatility, label="GARCH(1,1) Volatility", alpha=0.8)
plt.plot(egarch_fit.conditional_volatility, label="EGARCH(1,1) Volatility", alpha=0.8)
plt.title("Estimated Conditional Volatility - Nifty 50")
plt.legend()
plt.tight_layout()
plt.savefig("garch_egarch_volatility.png", dpi=150)
plt.show()

vol_df = pd.DataFrame({
    "GARCH_vol": garch_fit.conditional_volatility,
    "EGARCH_vol": egarch_fit.conditional_volatility
})
vol_df.to_csv("garch_egarch_volatility_nsei.csv")
print("\nSaved volatility series to garch_egarch_volatility_nsei.csv")