import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

# Helper function to calculate the Exponential Moving Average
def calculate_ema(prices, period):
    return prices.ewm(span=period, adjust=False).mean()

# Helper function to calculate the Stochastic Oscillator
def stochastic_oscillator(high, low, close, k_window=14, d_window=3):
    min_low = low.rolling(window=k_window).min()
    max_high = high.rolling(window=k_window).max()
    k = 100 * ((close - min_low) / (max_high - min_low))
    d = k.rolling(window=d_window).mean()
    return k, d

# Define the strategy implementation
def trading_strategy(data):
    conn = sqlite3.connect('trade_data.db')
    setup_database(conn)
    
    data['EMA50'] = calculate_ema(data['Close'], 50)
    data['EMA200'] = calculate_ema(data['Close'], 200)
    data['%K'], data['%D'] = stochastic_oscillator(data['High'], data['Low'], data['Close'])

    buy_signals = []
    sell_signals = []
    transaction_data = []
    
    for i in range(1, len(data)):
        if data['EMA50'][i] > data['EMA200'][i] and data['%K'][i] < 20 and data['%D'][i] < 20:
            buy_signals.append(data['Close'][i])
            sell_signals.append(np.nan)
            entry_data = (data.index[i], data['Close'][i], data['EMA50'][i], data['EMA200'][i], data['%K'][i], data['%D'][i], data['Volume'][i])
            transaction_data.append(entry_data)
        elif len(buy_signals) > 0 and buy_signals[-1] is not np.nan and (data['Close'][i] > buy_signals[-1] * 1.002 or data['Close'][i] < buy_signals[-1] * 0.998):
            sell_signals.append(data['Close'][i])
            buy_signals.append(np.nan)
            exit_data = transaction_data.pop()
            profit_loss = (data['Close'][i] - exit_data[1]) * 10000  # profit or loss in pips
            store_transaction(conn, exit_data, (data.index[i], data['Close'][i], data['EMA50'][i], data['EMA200'][i], data['%K'][i], data['%D'][i], data['Volume'][i], profit_loss))
        else:
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)
    
    data['Buy'] = buy_signals
    data['Sell'] = sell_signals
    plot_signals(data)
    conn.close()

# Function to set up the database for storing transactions
def setup_database(conn):
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_time DATETIME,
        exit_time DATETIME,
        entry_price REAL,
        exit_price REAL,
        profit_loss REAL,
        entry_ema50 REAL,
        exit_ema50 REAL,
        entry_ema200 REAL,
        exit_ema200 REAL,
        entry_k REAL,
        exit_k REAL,
        entry_d REAL,
        exit_d REAL,
        entry_volume INTEGER,
        exit_volume INTEGER
    )''')
    conn.commit()

# Function to store a transaction in the database
def store_transaction(conn, entry_data, exit_data):
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO transactions (
        entry_time, exit_time, entry_price, exit_price, profit_loss,
        entry_ema50, exit_ema50, entry_ema200, exit_ema200,
        entry_k, exit_k, entry_d, exit_d, entry_volume, exit_volume
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
    (entry_data[0], exit_data[0], entry_data[1], exit_data[1], exit_data[7],
     entry_data[2], exit_data[2], entry_data[3], exit_data[3],
     entry_data[4], exit_data[4], entry_data[5], exit_data[5], entry_data[6], exit_data[6]))
    conn.commit()

# Function to plot the trading signals
def plot_signals(data):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['EMA50'], label='EMA 50')
    plt.plot(data['EMA200'], label='EMA 200')
    plt.scatter(data.index[data['Buy'].notna()], data['Buy'][data['Buy'].notna()], color='green', marker='^', alpha=1)
    plt.scatter(data.index[data['Sell'].notna()], data['Sell'][data['Sell'].notna()], color='red', marker='v', alpha=1)
    plt.title('Trading Strategy Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Example usage:
# Assume data is loaded into a DataFrame named 'data' with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
# data = pd.read_csv('path_to_your_minute5_data.csv')
# data['Date'] = pd.to_datetime(data['Date'])
# data.set_index('Date', inplace=True)
# trading_strategy(data)
