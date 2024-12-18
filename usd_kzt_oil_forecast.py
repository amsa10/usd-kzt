
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.model_selection import train_test_split

# Function to calculate RSI
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# Fetch historical data for USD/KZT and Brent Crude Oil prices
ticker_usd_kzt = 'USDKZT=X'
ticker_oil = 'BZ=F'  # Brent Crude Oil ticker
data_usd_kzt = yf.download(ticker_usd_kzt, start='2020-01-01', end='2024-12-31')
data_oil = yf.download(ticker_oil, start='2020-01-01', end='2024-12-31')

# Calculate RSI for USD/KZT
data_usd_kzt['RSI'] = calculate_rsi(data_usd_kzt['Close'], period=14)

# Calculate SMA and EMA for USD/KZT
data_usd_kzt['SMA_50'] = data_usd_kzt['Close'].rolling(window=50).mean()
data_usd_kzt['SMA_200'] = data_usd_kzt['Close'].rolling(window=200).mean()
data_usd_kzt['EMA_50'] = data_usd_kzt['Close'].ewm(span=50, adjust=False).mean()
data_usd_kzt['EMA_200'] = data_usd_kzt['Close'].ewm(span=200, adjust=False).mean()

# Calculate MACD for USD/KZT
data_usd_kzt['MACD'] = data_usd_kzt['EMA_50'] - data_usd_kzt['EMA_200']
data_usd_kzt['Signal_Line'] = data_usd_kzt['MACD'].ewm(span=9, adjust=False).mean()

# Calculate ATR for USD/KZT
data_usd_kzt['True_Range'] = np.maximum(data_usd_kzt['High'] - data_usd_kzt['Low'],
                                        np.maximum(abs(data_usd_kzt['High'] - data_usd_kzt['Close'].shift(1)),
                                                   abs(data_usd_kzt['Low'] - data_usd_kzt['Close'].shift(1))))
data_usd_kzt['ATR'] = data_usd_kzt['True_Range'].rolling(window=14).mean()

# Calculate Stochastic Oscillator for USD/KZT
low_14 = data_usd_kzt['Low'].rolling(window=14).min()
high_14 = data_usd_kzt['High'].rolling(window=14).max()
data_usd_kzt['Stochastic_K'] = 100 * (data_usd_kzt['Close'] - low_14) / (high_14 - low_14)
data_usd_kzt['Stochastic_D'] = data_usd_kzt['Stochastic_K'].rolling(window=3).mean()

# Shift the 'Close' price to create a target variable (next day's price)
data_usd_kzt['Close_Shifted'] = data_usd_kzt['Close'].shift(-1)

# Drop rows with missing values (due to RSI, moving averages, ATR, etc.)
data_usd_kzt.dropna(inplace=True)

# Merge the USD/KZT data with Brent Crude Oil prices based on Date
data_oil['Oil_Price'] = data_oil['Close']
data_oil = data_oil[['Oil_Price']]

# Merge the dataframes using the index
data = pd.merge(data_usd_kzt, data_oil, left_index=True, right_index=True, how='inner')

# Features (X) - Including RSI, USD/KZT closing price, Oil Price, and other indicators
X = data[['RSI', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'MACD', 'Signal_Line', 'ATR', 'Stochastic_K', 'Stochastic_D', 'Oil_Price', 'Volume']]

# Target (y) - Next day's closing price
y = data['Close_Shifted']

# Scale the features and target to the range [0, 1] using MinMaxScaler
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Create sequences of data for LSTM input (X: indicators, y: Next day's closing price)
def create_sequences(X, y, time_step=10):
    X_seq, y_seq = [], []
    for i in range(time_step, len(X)):
        X_seq.append(X[i-time_step:i, :])  # Use the last `time_step` days' data as features
        y_seq.append(y[i, 0])  # Next day's price
    return np.array(X_seq), np.array(y_seq)

# Create sequences for training
time_step = 10  # Number of previous days to use for prediction
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_step)

# Reshape data for LSTM input: [samples, time_steps, features]
X_seq = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], X_seq.shape[2])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# Build the LSTM model
model = Sequential()

# Add LSTM layers
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Add a Dense layer for output
model.add(Dense(units=1))  # Predicting one value (next day's price)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Predict the next day's closing prices for the test set
y_pred_scaled = model.predict(X_test)

# Inverse transform the predicted values and the actual values to get the original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Evaluate the model
mse = np.mean((y_pred - y_test_actual) ** 2)
print(f"Mean Squared Error (MSE): {mse}")
