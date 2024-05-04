import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_technical_indicators(df, period='5m'):
    # Calculate moving averages
    if period == '5m':
        short_window, long_window = 5, 8
    elif period == '15m':
        short_window, long_window = 10, 15
    else:
        raise ValueError("Unsupported period. Use '5m' or '15m'.")

    df['MA_Short'] = df['Close'].rolling(window=short_window).mean()
    df['MA_Long'] = df['Close'].rolling(window=long_window).mean()

    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Calculate MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    return df

def plot_technical_indicators(df, ticker):
    plt.figure(figsize=(14, 7))

    # Plot closing price and moving averages
    plt.subplot(311)
    plt.title(f'Stock Price and Moving Averages for {ticker}')
    plt.plot(df['Close'], label='Close')
    plt.plot(df['MA_Short'], label=f'{df["MA_Short"].rolling(window=2).mean().iloc[0]}-Period MA')
    plt.plot(df['MA_Long'], label=f'{df["MA_Long"].rolling(window=2).mean().iloc[0]}-Period MA')
    plt.legend()

    # Plot RSI
    plt.subplot(312)
    plt.title('RSI')
    plt.plot(df['RSI'])
    plt.axhline(70, linestyle='--', alpha=0.5, color='red')
    plt.axhline(30, linestyle='--', alpha=0.5, color='green')

    # Plot MACD
    plt.subplot(313)
    plt.title('MACD')
    plt.plot(df['MACD'], label='MACD')
    plt.plot(df['Signal'], label='Signal')
    plt.legend()

    plt.tight_layout()
    plt.show()

def analyze_stock(ticker, period='5m'):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1mo", interval=period)
    df = calculate_technical_indicators(df, period)
    plot_technical_indicators(df, ticker)

# Example usage for a highly liquid stock, e.g., "AAPL"
analyze_stock('AAPL', '5m')
