import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

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

# Fetch historical data for USD/KZT and Brent Crude Oil (BZ=F for Brent)
ticker_usd_kzt = 'USDKZT=X'
ticker_oil = 'BZ=F'  # Updated symbol for Brent Crude Oil Futures
data_usd_kzt = yf.download(ticker_usd_kzt, start='2020-01-01', end='2024-12-31')
data_oil = yf.download(ticker_oil, start='2020-01-01', end='2024-12-31')

# Calculate RSI for USD/KZT
data_usd_kzt['RSI'] = calculate_rsi(data_usd_kzt['Close'], period=14)

# Shift the 'Close' price to create a target variable (120 days ahead price)
data_usd_kzt['Close_Shifted'] = data_usd_kzt['Close'].shift(-1)

# Drop rows with missing values (due to RSI and shifted 'Close' columns)
data_usd_kzt.dropna(inplace=True)

# Reset the index to ensure both dataframes can be merged on the same column
data_usd_kzt.reset_index(inplace=True)
data_oil.reset_index(inplace=True)

# Merge the USD/KZT data with the Brent Crude Oil prices data based on Date
# Include Close_Shifted in the merge
data = pd.merge(data_usd_kzt, data_oil[['Date', 'Close']], left_on='Date', right_on='Date', how='inner')

# Flatten the MultiIndex columns after merge
data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]

# Rename columns for clarity
data.rename(columns={'Close_USDKZT=X': 'USD_KZT_Close', 'Close_BZ=F': 'Oil_Price'}, inplace=True)

# Ensure RSI and Close_Shifted columns are present
if 'RSI' not in data.columns:
    data['RSI'] = data_usd_kzt['RSI']

#Added to ensure that Close_Shifted is present
if 'Close_Shifted' not in data.columns:
    data['Close_Shifted'] = data_usd_kzt['Close_Shifted']

# Ensure 'Date' column is retained after merge and rename operations
if 'Date' not in data.columns:
    data['Date'] = data_usd_kzt['Date']


# Features (X) - Including RSI, USD/KZT closing price, and Oil Price
X = data[['RSI', 'USD_KZT_Close', 'Oil_Price']]  # Use RSI, Close price, and Oil price as features

# Target (y) - 120 days ahead closing price
y = data['Close_Shifted']

# Scale the features and target to range [0, 1] using MinMaxScaler
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Create sequences of data for LSTM input (X: RSI, Close price, Oil price, y: 120 days ahead closing price)
def create_sequences(X, y, time_step=60):
    X_seq, y_seq = [], []
    for i in range(time_step, len(X)):
        X_seq.append(X[i-time_step:i, :])  # Use RSI, Close, and Oil price for the last `time_step` days
        y_seq.append(y[i, 0])  # 120 days ahead price
    return np.array(X_seq), np.array(y_seq)

# Create sequences for training
time_step = 120 # Number of previous days to use for prediction
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
model.add(Dense(units=1))  # Predicting one value (120 days ahead price)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Predict the 120 days ahead closing prices for the test set
y_pred_scaled = model.predict(X_test)

# Inverse transform the predicted values and the actual values to get the original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Evaluate the model
mse = np.mean((y_pred - y_test_actual) ** 2)
print(f"Mean Squared Error (MSE): {mse}")

import matplotlib.pyplot as plt
%matplotlib inline
# Plot the actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual Price (120 days ahead)', color='blue')
plt.plot(y_pred, label='Predicted Price (120 days ahead)', color='orange')
plt.title('KZT/USD Price Forecasting using LSTM, RSI, and Oil Price (120 Days Ahead)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Visualize the RSI and Oil Price alongside the actual and predicted price data
plt.figure(figsize=(12, 6))

# Plot Close price
plt.subplot(3, 1, 1)
plt.plot(data['Date'][-len(y_test_actual):], y_test_actual, label='Actual Close Price', color='blue')
plt.plot(data['Date'][-len(y_pred):], y_pred, label='Predicted Close Price', color='orange')
plt.title(f'{ticker_usd_kzt} Actual vs Predicted Close Price (120 Days Ahead)')
plt.legend()

# Plot RSI
plt.subplot(3, 1, 2)
plt.plot(data['Date'][-len(y_test_actual):], data['RSI'][-len(y_test_actual):], label='RSI', color='green')
plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
plt.title('RSI')
plt.legend()

# Plot Oil Price
plt.subplot(3, 1, 3)
plt.plot(data['Date'][-len(y_test_actual):], data['Oil_Price'][-len(y_test_actual):], label='Oil Price (Brent)', color='purple')
plt.title('Oil Price (Brent)')
plt.legend()

plt.tight_layout()
plt.show()
