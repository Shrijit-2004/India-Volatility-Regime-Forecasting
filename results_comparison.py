import pandas as pd

xgb = pd.read_csv("xgboost_results_summary.csv")
lstm = pd.read_csv("lstm_results_summary.csv")
egarch = pd.read_csv("egarch_forecast_results_summary.csv")

master = pd.concat([egarch, xgb, lstm], ignore_index=True)
master = master.sort_values(["Target", "RMSE"])
print(master.to_string(index=False))

master.to_csv("master_comparison_results.csv", index=False)
print("\nSaved to master_comparison_results.csv")