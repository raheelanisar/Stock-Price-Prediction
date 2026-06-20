import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load Model
model = load_model(r"C:\Python\Stock\Stock Predictions Model.keras")

# Title
st.title("Stock Market Predictor")

# User Input
stock = st.text_input("Enter Stock Symbol", "GOOG")

start = "2012-01-01"
end = "2022-12-31"

# Download Data
data = yf.download(stock, start=start, end=end)

# Check if data exists
if data.empty:
    st.error("Invalid Stock Symbol or No Data Found!")
    st.stop()

# Display Data
st.subheader("Stock Data")
st.write(data)

# Split Data
data_train = pd.DataFrame(data["Close"][:int(len(data) * 0.80)])
data_test = pd.DataFrame(data["Close"][int(len(data) * 0.80):])

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scale = scaler.fit_transform(data_train)

# Last 100 Days + Test Data
past_100_days = data_train.tail(100)

data_test = pd.concat([past_100_days, data_test], ignore_index=True)

data_test_scale = scaler.transform(data_test)

# Moving Average 50
st.subheader("Price vs MA50")

ma_50_days = data["Close"].rolling(50).mean()

fig1 = plt.figure(figsize=(10, 5))
plt.plot(data["Close"], label="Close Price")
plt.plot(ma_50_days, label="MA50")
plt.legend()
st.pyplot(fig1)

# Moving Average 100
st.subheader("Price vs MA50 vs MA100")

ma_100_days = data["Close"].rolling(100).mean()

fig2 = plt.figure(figsize=(10, 5))
plt.plot(data["Close"], label="Close Price")
plt.plot(ma_50_days, label="MA50")
plt.plot(ma_100_days, label="MA100")
plt.legend()
st.pyplot(fig2)

# Moving Average 200
st.subheader("Price vs MA100 vs MA200")

ma_200_days = data["Close"].rolling(200).mean()

fig3 = plt.figure(figsize=(10, 5))
plt.plot(data["Close"], label="Close Price")
plt.plot(ma_100_days, label="MA100")
plt.plot(ma_200_days, label="MA200")
plt.legend()
st.pyplot(fig3)

# Prepare Test Data
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x = np.array(x)
y = np.array(y)

# Prediction
predictions = model.predict(x)

# Reverse Scaling
scale_factor = 1 / scaler.scale_[0]

predictions = predictions * scale_factor
y = y * scale_factor

# Plot Prediction
st.subheader("Actual Price vs Predicted Price")

fig4 = plt.figure(figsize=(10, 5))

plt.plot(y, label="Actual Price")
plt.plot(predictions, label="Predicted Price")

plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()

st.pyplot(fig4)