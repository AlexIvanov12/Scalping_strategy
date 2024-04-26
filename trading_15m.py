import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from ta.volatility import BollingerBands
from ta.trend import ADXIndicator

def read_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return df

def calculate_indicators(df):
    indicator_bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()
    
    adx_i = ADXIndicator(df['High'], df['Low'], df['Close'], window=14, fillna=True)
    df['adx'] = adx_i.adx()

    return df

def setup_database(conn):
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_time DATETIME,
        exit_time DATETIME,
        entry_price REAL,
        exit_price REAL,
        profit_loss REAL,
        entry_adx REAL,
        exit_adx REAL,
        entry_bbm REAL,
        entry_bbh REAL,
        entry_bbl REAL,
        exit_bbm REAL,
        exit_bbh REAL,
        exit_bbl REAL,
        entry_volume INTEGER,
        exit_volume INTEGER
    )''')
    conn.commit()

def trading_logic(df, conn):
    cursor = conn.cursor()
    in_position = False
    for i, row in df.iterrows():
        if not in_position and row['Close'] > row['bb_bbh'] and row['adx'] > 25:
            # Condition to enter a trade
            entry_data = (i, row['Close'], row['adx'], row['bb_bbm'], row['bb_bbh'], row['bb_bbl'], row['Volume'])
            in_position = True
        elif in_position and (row['Close'] < row['bb_bbl'] or row['adx'] < 20):
            # Condition to exit a trade
            exit_data = (i, row['Close'], row['adx'], row['bb_bbm'], row['bb_bbh'], row['bb_bbl'], row['Volume'])
            profit_loss = (exit_data[1] - entry_data[1]) * 10000  # Calculating profit/loss in pips
            cursor.execute('''INSERT INTO trades (
                entry_time, exit_time, entry_price, exit_price, profit_loss,
                entry_adx, exit_adx, entry_bbm, entry_bbh, entry_bbl,
                exit_bbm, exit_bbh, exit_bbl, entry_volume, exit_volume
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
            (entry_data[0], exit_data[0], entry_data[1], exit_data[1], profit_loss,
             entry_data[2], exit_data[2], entry_data[3], entry_data[4], entry_data[5],
             exit_data[3], exit_data[4], exit_data[5], entry_data[6], exit_data[6]))
            conn.commit()
            in_position = False

def plot_signals(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df['Close'], label='Close Price', alpha=0.5)
    plt.plot(df['bb_bbm'], label='Bollinger Mid Band', linestyle='--')
    plt.plot(df['bb_bbh'], label='Bollinger High Band', color='red', alpha=0.5)
    plt.plot(df['bb_bbl'], label='Bollinger Low Band', color='blue', alpha=0.5)
    plt.scatter(df.index[df['Buy']], df['bb_bbh'][df['Buy']], color='green', label='Buy Signal', marker='^', alpha=1)
    plt.scatter(df.index[df['Sell']], df['bb_bbl'][df['Sell']], color='red', label='Sell Signal', marker='v', alpha=1)
    plt.title('Bollinger Bands and ADX Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def main(file_path):
    df = read_data(file_path)
    df = calculate_indicators(df)
    conn = sqlite3.connect('trading_data.db')
    setup_database(conn)
    trading_logic(df, conn)
    plot_signals(df)
    conn.close()

# Replace 'your_data_file.csv' with the path to your data file
# main('your_data_file.csv')
