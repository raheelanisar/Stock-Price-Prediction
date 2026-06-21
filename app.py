import streamlit as st
import pandas as pd
import yfinance as yf
import pickle
import matplotlib.pyplot as plt

# Load Trained Model
with open(r"D:\Stock\stock_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# App Title
st.title("Stock Price Prediction App")

# User Input
stock = st.text_input("Enter Stock Symbol", "AAPL")

if st.button("Predict"):

    try:
        # Download Stock Data
        data = yf.download(stock, start="2020-01-01", end="2025-01-01")

        if data.empty:
            st.error("Invalid Stock Symbol or No Data Found")
            st.stop()

        # Show Data
        st.subheader("Stock Data")
        st.write(data.tail())

        # Features (same as training)
        X = data[["Open", "High", "Low", "Volume"]]

        # Prediction
        predictions = model.predict(X)

        # Add Prediction Column
        data["Predicted_Close"] = predictions

        # Graph
        st.subheader("Actual vs Predicted Closing Price")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(data.index, data["Close"], label="Actual Close Price")
        ax.plot(data.index, data["Predicted_Close"], label="Predicted Close Price")

        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()

        st.pyplot(fig)

        # Latest Prediction
        latest_price = predictions[-1]

        try:
            latest_price = float(latest_price)
        except:
            latest_price = float(latest_price[0])

        st.subheader("Latest Predicted Price")
        st.success(f"${latest_price:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")