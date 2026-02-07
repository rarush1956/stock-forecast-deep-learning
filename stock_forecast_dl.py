import numpy as np
import pandas as pd
import yfinance as yf
import ta
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping

from datetime import datetime


# USER INPUT
ticker = input("Enter stock ticker (e.g., RELIANCE.NS, TCS.NS, AAPL): ").strip().upper()
target_date = input("Enter target date (YYYY-MM-DD): ").strip()
target_date = datetime.strptime(target_date, "%Y-%m-%d")


# DOWNLOAD DATA
df_raw = yf.download(ticker, start="2015-01-01")

if df_raw.empty:
    raise ValueError("Invalid ticker or no data found.")

close = df_raw["Close"]
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]

df = pd.DataFrame({"Close": close})
df.dropna(inplace=True)

print(f"Data downloaded: {len(df)} rows")


# DATE HANDLING
last_date = df.index[-1]
days_ahead = (target_date - last_date.to_pydatetime()).days

if days_ahead <= 0:
    raise ValueError("Target date must be AFTER the last available date.")

print(f"Forecasting {days_ahead} days into the future")


# FEATURE ENGINEERING
df["rsi"] = ta.momentum.RSIIndicator(close=df["Close"]).rsi()
df["sma"] = ta.trend.SMAIndicator(close=df["Close"], window=20).sma_indicator()
df["returns"] = df["Close"].pct_change()
df.dropna(inplace=True)


# SCALING
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)


# SEQUENCE CREATION
def make_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = make_sequences(scaled)
split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# LSTM MODEL
lstm = Sequential([
    LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])

lstm.compile(optimizer="adam", loss="mse")

print("\nTraining LSTM...")
lstm.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[EarlyStopping(patience=3)],
    verbose=1
)


# GRU MODEL
gru = Sequential([
    GRU(32, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])

gru.compile(optimizer="adam", loss="mse")

print("\nTraining GRU...")
gru.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[EarlyStopping(patience=3)],
    verbose=1
)


# EVALUATION
def inverse_transform(pred):
    dummy = np.zeros((len(pred), scaled.shape[1]))
    dummy[:, 0] = pred.flatten()
    return scaler.inverse_transform(dummy)[:, 0]

lstm_pred = inverse_transform(lstm.predict(X_test))
gru_pred = inverse_transform(gru.predict(X_test))
actual = inverse_transform(y_test.reshape(-1, 1))

lstm_rmse = np.sqrt(mean_squared_error(actual, lstm_pred))
gru_rmse = np.sqrt(mean_squared_error(actual, gru_pred))

lstm_mae = mean_absolute_error(actual, lstm_pred)
gru_mae = mean_absolute_error(actual, gru_pred)

print("\nLSTM Results")
print(f"RMSE: {lstm_rmse:.2f}")
print(f"MAE : {lstm_mae:.2f}")

print("\nGRU Results")
print(f"RMSE: {gru_rmse:.2f}")
print(f"MAE : {gru_mae:.2f}")


# ============================
# SELECT BEST MODEL
# ============================
if gru_rmse < lstm_rmse:
    best_model = gru
    best_name = "GRU"
else:
    best_model = lstm
    best_name = "LSTM"

print(f"\nBetter performing model based on RMSE: {best_name}")


# FUTURE FORECASTING
def forecast_future(model, last_seq, steps):
    seq = last_seq.copy()
    preds = []

    for _ in range(steps):
        p = model.predict(seq.reshape(1, seq.shape[0], seq.shape[1]), verbose=0)
        preds.append(p[0, 0])
        seq = np.roll(seq, -1, axis=0)
        seq[-1, 0] = p[0, 0]

    return np.array(preds)

last_sequence = X[-1]
future_scaled = forecast_future(best_model, last_sequence, days_ahead)

dummy = np.zeros((len(future_scaled), scaled.shape[1]))
dummy[:, 0] = future_scaled
future_prices = scaler.inverse_transform(dummy)[:, 0]

predicted_price = future_prices[-1]
last_price = df["Close"].iloc[-1]


# CONCLUSION
trend = "upward" if predicted_price > last_price else "downward"

print("\n--- FORECAST CONCLUSION ---")
print(f"Stock: {ticker}")
print(f"Model used: {best_name}")
print(f"Last known price: {last_price:.2f}")
print(f"Predicted price on {target_date.date()}: {predicted_price:.2f}")
print(f"Inferred trend based on historical patterns: {trend}")


# OPTIONAL PLOT
plt.figure(figsize=(10, 5))
plt.plot(actual, label="Actual")
plt.plot(lstm_pred, label="LSTM")
plt.plot(gru_pred, label="GRU")
plt.title(f"{ticker} Prediction Comparison")
plt.legend()
plt.show()
