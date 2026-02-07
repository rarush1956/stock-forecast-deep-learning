# Deep Learning–Based Stock Price Forecasting (LSTM & GRU)

This project implements a deep learning–based stock price forecasting system using **LSTM** and **GRU** neural networks.  
The system automatically selects the better-performing model and forecasts future stock prices based on historical trends.

---

## 🔹 Features
- User-defined stock ticker (NSE / global stocks)
- User-defined future date
- LSTM and GRU model comparison
- Automatic best-model selection using RMSE
- Multi-step future price forecasting
- Clear trend-based conclusion

---

## 🔹 Tech Stack
- Python
- NumPy, Pandas
- TensorFlow / Keras
- Scikit-learn
- yFinance
- TA-Lib (technical indicators)
- Matplotlib

---

## 🔹 How It Works
1. Downloads historical stock price data
2. Generates technical indicators (RSI, SMA, returns)
3. Trains LSTM and GRU models
4. Evaluates models using RMSE and MAE
5. Selects the better-performing model
6. Forecasts future prices based on learned patterns

---

## 🔹 How to Run

### Install dependencies
```bash
pip install -r requirements.txt
