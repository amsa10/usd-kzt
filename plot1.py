plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual Price', color='blue')
plt.plot(y_pred, label='Predicted Price', color='orange')
plt.title('KZT/USD Price Forecasting using LSTM, RSI, and USDKZT Close')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Visualize the RSI alongside the actual and predicted price data
plt.figure(figsize=(12, 6))

# Plot Close price
plt.subplot(2, 1, 1)
plt.plot(data.index[-len(y_test_actual):], y_test_actual, label='Actual Close Price', color='blue')
plt.plot(data.index[-len(y_pred):], y_pred, label='Predicted Close Price', color='orange')
plt.title(f'{ticker_usd_kzt} Actual vs Predicted Close Price') # Replace ticker with ticker_usd_kzt
plt.legend()

# Plot RSI
plt.subplot(2, 1, 2)
plt.plot(data['RSI'][-len(y_test_actual):], label='RSI', color='green')
plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
plt.title('RSI')
plt.legend()



plt.tight_layout()
plt.show()
