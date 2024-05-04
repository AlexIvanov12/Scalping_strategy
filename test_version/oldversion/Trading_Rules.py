import yfinance as yf
import pandas as pd
import numpy as np

def calculate_technical_indicators(df):
    # Define windows for moving averages
    short_window, long_window = 5, 8  # 5m chart settings; adjust if using 15m
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

def identify_trading_signals(df):
    # Initialize columns for Buy and Sell signals
    df['Buy'] = 0
    df['Sell'] = 0

    # Loop through DataFrame and apply trading logic
    for i in range(1, len(df)):
        # Buy conditions
        if (df['MA_Short'][i] > df['MA_Long'][i] and
            df['MA_Short'][i - 1] <= df['MA_Long'][i - 1] and
            df['RSI'][i] < 30 and
            df['MACD'][i] > df['Signal'][i] and
            df['MACD'][i - 1] <= df['Signal'][i - 1]):
            df['Buy'][i] = 1

        # Sell conditions
        if (df['MA_Short'][i] < df['MA_Long'][i] and
            df['MA_Short'][i - 1] >= df['MA_Long'][i - 1] and
            df['RSI'][i] > 70 and
            df['MACD'][i] < df['Signal'][i] and
            df['MACD'][i - 1] >= df['Signal'][i - 1]):
            df['Sell'][i] = 1

    return df

def backtest_strategy(df):
    buying_price = None
    selling_price = None
    stop_loss = 0.002  # 0.2%
    profit_target = 0.005  # 0.5%
    positions = []
    trades = []

    for i in range(len(df)):
        if df['Buy'][i] == 1:
            buying_price = df['Close'][i]
            continue

        if buying_price is not None and (df['Sell'][i] == 1 or
            df['Close'][i] >= buying_price * (1 + profit_target) or
            df['Close'][i] <= buying_price * (1 - stop_loss)):
            selling_price = df['Close'][i]
            trades.append((buying_price, selling_price))
            buying_price = None  # Reset buying price for next trade

    return trades

def analyze_stock(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1mo", interval='5m')
    df = calculate_technical_indicators(df)
    df = identify_trading_signals(df)
    trades = backtest_strategy(df)
    return trades

# Example usage for a highly liquid stock, e.g., "AAPL"
trades = analyze_stock('AAPL')
print(trades)
