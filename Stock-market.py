import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import datetime, timedelta

model = load_model("Stock Prediction Model.keras")
if model is None:
    st.error("Failed to load the model. Please check the model file.")
    st.stop()

st.header("Stock Market Predictor")
stock = st.text_input("Enter the stock symbol", "GOOG")
start = "2012-01-01"
end = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

data = yf.download(tickers=stock, start=start, end=end)

if data is None or data.empty:
    st.error("Unable to download stock data. Please check the stock symbol and try again.")
    st.stop()

st.subheader("Stock Data")
st.write(data.sort_index(ascending=False))

ma_50_days = data.Close.rolling(50).mean()
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()

data_train = pd.DataFrame(data.Close[0:int(len(data)*0.8)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.8):len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

scaler.fit(data_train)

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.transform(data_test)


st.subheader("Price vs MA50")
fig1 = plt.figure(figsize=(10, 8))
plt.plot(ma_50_days, "r", label="MA 50")
plt.plot(data.Close, "b", label="Actual")
plt.legend()
st.pyplot(fig1)
plt.close(fig1)

st.subheader("Price vs MA50 vs MA100")
fig2 = plt.figure(figsize=(10, 8))
plt.plot(ma_50_days, "r", label="MA 50")
plt.plot(ma_100_days, "g", label="MA 100")
plt.plot(data.Close, "b", label="Actual")
plt.legend()
st.pyplot(fig2)
plt.close(fig2)

st.subheader("Price vs MA100 vs MA200")
fig3 = plt.figure(figsize=(10, 8))
plt.plot(ma_100_days, "r", label="MA 100")
plt.plot(ma_200_days, "g", label="MA 200")
plt.plot(data.Close, "b", label="Actual")
plt.legend()
st.pyplot(fig3)
plt.close(fig3)

x, y = [], []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x = np.array(x)
y = np.array(y)

predict = model.predict(x)  # type: ignore

predict = scaler.inverse_transform(predict)
y = scaler.inverse_transform(y.reshape(-1, 1)).ravel()

test_dates = data.index[-len(y):]

st.subheader("Original Price vs Predicted Price (aligned)")
fig4 = plt.figure(figsize=(10, 6))
plt.plot(test_dates, y,       label="Original")
plt.plot(test_dates, predict, label="Predicted")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
st.pyplot(fig4)
plt.close(fig4)

st.header("Future predictions")
n_future = st.number_input("Enter the number of future days", min_value=1, max_value=365, value=5, step=1)
last_100 = data_test_scale[-100:].flatten()

future_preds = []

for _ in range(1, n_future + 1):
    X = last_100[-100:].reshape(1, 100, 1)
    p = model.predict(X)[0, 0]  # type: ignore
    last_100 = np.append(last_100, p)
    future_preds.append(p)

future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
last_date = data.index[-1]
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=int(n_future))


st.subheader(f"Predicted price for the next {int(n_future)} days")
fig5 = plt.figure(figsize=(10, 6))
plt.plot(future_dates, future_preds, label="Predicted")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig5)
plt.close(fig5)

st.write(pd.DataFrame({"Date": future_dates, "Predicted_Price": future_preds}))

