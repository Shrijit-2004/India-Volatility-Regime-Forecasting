import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

data = pd.read_csv("ml_features.csv", index_col="Date", parse_dates=True)
feature_cols = [c for c in data.columns if "target" not in c]

SEQ_LEN = 10  

def create_sequences(X, y, seq_len):
    """
    Converts flat (rows, features) data into 3D sequences:
    (samples, timesteps, features) -- the shape LSTM requires.
    """
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

def train_lstm(target_col, label):
    X_raw = data[feature_cols].values
    y_raw = data[target_col].values.reshape(-1, 1)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_scaled = x_scaler.fit_transform(X_raw)
    y_scaled = y_scaler.fit_transform(y_raw)

    X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN)

    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, X_seq.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=0  
    )

    preds_scaled = model.predict(X_test, verbose=0)
    preds = y_scaler.inverse_transform(preds_scaled)
    y_test_actual = y_scaler.inverse_transform(y_test)

    rmse = np.sqrt(mean_squared_error(y_test_actual, preds))
    mae = mean_absolute_error(y_test_actual, preds)

    print(f"\n=== LSTM — {label} ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")

    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"LSTM Training Loss — {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"lstm_loss_{label.lower().replace(' ', '_')}.png", dpi=150)
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(y_test_actual, label="Actual", alpha=0.8)
    plt.plot(preds, label="LSTM Predicted", alpha=0.8)
    plt.title(f"LSTM — {label} (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"lstm_{label.lower().replace(' ', '_')}.png", dpi=150)
    plt.show()

    return {"rmse": rmse, "mae": mae}

results_realized_vol = train_lstm("target_realized_vol", "Realized Volatility")
results_vix = train_lstm("target_vix", "India VIX")

summary = pd.DataFrame({
    "Model": ["LSTM", "LSTM"],
    "Target": ["Realized Volatility", "India VIX"],
    "RMSE": [results_realized_vol["rmse"], results_vix["rmse"]],
    "MAE": [results_realized_vol["mae"], results_vix["mae"]]
})
summary.to_csv("lstm_results_summary.csv", index=False)
print("\nSaved results summary to lstm_results_summary.csv")