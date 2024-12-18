plt.figure(figsize=(15, 18))

# Plot each feature in a separate subplot
features = ['RSI', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'MACD', 'Signal_Line', 'ATR', 'Stochastic_K', 'Stochastic_D', 'Oil_Price']
feature_titles = [
    'RSI (Relative Strength Index) - Measures overbought/oversold conditions',
    'SMA_50 (50-period Simple Moving Average) - Trend indicator',
    'SMA_200 (200-period Simple Moving Average) - Long-term trend indicator',
    'EMA_50 (50-period Exponential Moving Average) - Short-term trend indicator',
    'EMA_200 (200-period Exponential Moving Average) - Long-term trend indicator',
    'MACD (Moving Average Convergence Divergence) - Momentum indicator',
    'Signal Line (MACD Signal Line) - Buy/Sell signal based on MACD crossovers',
    'ATR (Average True Range) - Volatility indicator',
    'Stochastic_K (Stochastic Oscillator %K) - Measures momentum',
    'Stochastic_D (Stochastic Oscillator %D) - 3-day moving average of %K',
    'Oil_Price - Brent Crude Oil price, used as a macroeconomic factor'
]

for i, feature in enumerate(features):
    plt.subplot(6, 2, i+1)
    plt.plot(data.index[-len(y_test_actual):], data[feature][-len(y_test_actual):], label=feature)
    plt.title(feature_titles[i])
    plt.legend()

plt.tight_layout()
