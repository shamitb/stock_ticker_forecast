import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import plotly.graph_objs as go

scaler = MinMaxScaler(feature_range=(0, 1))

# Helper function to create LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to prepare the data and create datasets
def prepare_data(data, lookback):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(lookback, len(data_scaled)):
        X.append(data_scaled[i-lookback:i, 0])
        y.append(data_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Streamlit app
def main():
    st.title("Stock Price Prediction with LSTM")

    # User input for ticker symbol
    ticker = st.text_input("Enter the stock ticker symbol", "AAPL")
    
    if st.button("Get Data and Predict"):
        st.write(f"Fetching data for {ticker}...")

        # Calculate the date range for the past year
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365)

        # Fetching the data
        data = yf.download(ticker, start=start_date, end=end_date)
        st.write(f"Data for {ticker}")
        st.line_chart(data['Close'])

        # Preparing data for LSTM
        data_close = data[['Close']].values

        lookbacks = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        predictions = {}

        for lookback in lookbacks:
            st.write(f"Training LSTM model with lookback period: {lookback}")
            
            # Prepare the dataset
            X_train, y_train, scaler = prepare_data(data_close, lookback)
            
            # Create and train the model
            model = create_lstm_model((X_train.shape[1], 1))
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
            
            # Predicting the next 15 days
            last_days = data_close[-lookback:]
            last_days_scaled = scaler.transform(last_days)
            X_test = np.reshape(last_days_scaled, (1, last_days_scaled.shape[0], 1))

            prediction = []
            for _ in range(15):
                pred = model.predict(X_test)
                prediction.append(pred[0][0])
                
                # Reshape the prediction to have the same shape as a single timestep (1, 1, 1)
                pred_reshaped = np.reshape(pred, (1, 1, 1))
                
                # Update the X_test array by removing the first element and adding the new prediction
                X_test = np.append(X_test[:, 1:, :], pred_reshaped, axis=1)

            prediction = scaler.inverse_transform(np.array(prediction).reshape(-1, 1))
            predictions[f"Lookback {lookback}"] = prediction.flatten()
        
        # Plotting all predictions on the same graph using Plotly
        fig = go.Figure()

        for key, value in predictions.items():
            fig.add_trace(go.Scatter(
                y=value,
                mode='lines',
                name=key
            ))

        fig.update_layout(
            title="LSTM Stock Price Predictions for Different Lookback Periods",
            xaxis_title="Days",
            yaxis_title="Predicted Price",
            legend_title="Lookback Period",
            height=600,
            width=1000,
            template="plotly_white"
        )

        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
