import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

data = pd.read_csv("ml_features.csv", index_col="Date", parse_dates=True)

feature_cols = [c for c in data.columns if "target" not in c]

def train_xgb(target_col, label):
    X = data[feature_cols]
    y = data[target_col]

    split_idx = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    print(f"\n=== XGBoost — {label} ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    print("\nTop 5 most important features:")
    print(importances.head(5))

    plt.figure(figsize=(12, 5))
    plt.plot(y_test.index, y_test.values, label="Actual", alpha=0.8)
    plt.plot(y_test.index, preds, label="XGBoost Predicted", alpha=0.8)
    plt.title(f"XGBoost — {label} (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"xgboost_{label.lower().replace(' ', '_')}.png", dpi=150)
    plt.show()

    return {"model": model, "rmse": rmse, "mae": mae, "preds": preds, "y_test": y_test}

results_realized_vol = train_xgb("target_realized_vol", "Realized Volatility")
results_vix = train_xgb("target_vix", "India VIX")

summary = pd.DataFrame({
    "Model": ["XGBoost", "XGBoost"],
    "Target": ["Realized Volatility", "India VIX"],
    "RMSE": [results_realized_vol["rmse"], results_vix["rmse"]],
    "MAE": [results_realized_vol["mae"], results_vix["mae"]]
})
summary.to_csv("xgboost_results_summary.csv", index=False)
print("\nSaved results summary to xgboost_results_summary.csv")