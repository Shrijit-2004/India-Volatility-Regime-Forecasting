import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = pd.read_csv("raw_market_data.csv", index_col="Date", parse_dates=True)
returns = 100 * np.log(data["NSEI"] / data["NSEI"].shift(1)).dropna()

realized_vol_5d = returns.rolling(window=5).std()
target_realized_vol = realized_vol_5d.shift(-5)
target_vix = data["INDIAVIX"].shift(-1)

ml_data = pd.read_csv("ml_features.csv", index_col="Date", parse_dates=True)
test_dates = ml_data.index[int(len(ml_data)*0.8):]

print(f"Test period: {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} days)")

def rolling_egarch_forecast(returns_series, test_dates):
    """
    For each day in the test period, refit EGARCH using only data available
    up to that point, and forecast 1-day-ahead volatility.
    This is SLOW (refitting hundreds of times) but it's the only honest way
    to get genuine out-of-sample GARCH forecasts.
    """
    forecasts = []
    for date in test_dates:
        train_data = returns_series[returns_series.index < date]
        model = arch_model(train_data, vol="EGARCH", p=1, o=1, q=1, dist="normal")
        fit = model.fit(disp="off")
        fc = fit.forecast(horizon=1)
        vol_forecast = np.sqrt(fc.variance.values[-1, 0])
        forecasts.append(vol_forecast)
    return pd.Series(forecasts, index=test_dates)

print("\nRunning rolling EGARCH forecast (this will take a few minutes -- refitting per day)...")
egarch_forecast = rolling_egarch_forecast(returns, test_dates)
egarch_forecast_annualized = egarch_forecast * np.sqrt(252)

y_true_rv = target_realized_vol.loc[test_dates].dropna()
common_idx = egarch_forecast.index.intersection(y_true_rv.index)

rmse_rv = np.sqrt(mean_squared_error(y_true_rv.loc[common_idx], egarch_forecast.loc[common_idx]))
mae_rv = mean_absolute_error(y_true_rv.loc[common_idx], egarch_forecast.loc[common_idx])

print(f"\n=== EGARCH Rolling Forecast vs Realized Volatility ===")
print(f"RMSE: {rmse_rv:.4f}")
print(f"MAE:  {mae_rv:.4f}")

y_true_vix = target_vix.loc[test_dates].dropna()
common_idx_vix = egarch_forecast_annualized.index.intersection(y_true_vix.index)

rmse_vix = np.sqrt(mean_squared_error(y_true_vix.loc[common_idx_vix], egarch_forecast_annualized.loc[common_idx_vix]))
mae_vix = mean_absolute_error(y_true_vix.loc[common_idx_vix], egarch_forecast_annualized.loc[common_idx_vix])

print(f"\n=== EGARCH Rolling Forecast vs India VIX ===")
print(f"RMSE: {rmse_vix:.4f}")
print(f"MAE:  {mae_vix:.4f}")

summary = pd.DataFrame({
    "Model": ["EGARCH", "EGARCH"],
    "Target": ["Realized Volatility", "India VIX"],
    "RMSE": [rmse_rv, rmse_vix],
    "MAE": [mae_rv, mae_vix]
})
summary.to_csv("egarch_forecast_results_summary.csv", index=False)
print("\nSaved to egarch_forecast_results_summary.csv")