import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import sqlite3

# Function to read data from a CSV file
def read_data(file_path):
    df = pd.read_csv(file_path, delimiter=',')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df




def compute_indicators(df):
    # Calculations for MACD and RSI
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    df['MACD'] = macd
    df['Signal'] = signal
    df['RSI'] = rsi
    
    return df

def signal_generation(df, conn):
    cursor = conn.cursor()
    buy_signals = [np.nan] * len(df)
    sell_signals = [np.nan] * len(df)
    transaction_data = []

    take_profit_threshold = 3
    stop_loss_threshold = -0.15

    for i in range(1, len(df)):
        # Buy condition
        if df['MACD'].iloc[i] > df['Signal'].iloc[i] and df['MACD'].iloc[i-1] <= df['Signal'].iloc[i-1] and df['RSI'].iloc[i] > 40:
            buy_signals[i] = df['Close'].iloc[i]
            entry = {
                'time_entry': df.index[i].strftime('%Y-%m-%d %H:%M:%S'),
                'price_entry': df['Close'].iloc[i],
                'macd_entry': df['MACD'].iloc[i],
                'signal_entry': df['Signal'].iloc[i],
                'rsi_entry': df['RSI'].iloc[i],
                'volume_entry': df['Volume'].iloc[i],
                'take_profit_price': df['Close'].iloc[i] + take_profit_threshold,
                'stop_loss_price': df['Close'].iloc[i] + stop_loss_threshold
            }
            transaction_data.append(entry)

        # Sell condition and transaction processing
        if transaction_data:
            current_price = df['Close'].iloc[i]
            last_transaction = transaction_data[-1]
            if current_price >= last_transaction['take_profit_price'] or current_price <= last_transaction['stop_loss_price']:
                sell_signals[i] = current_price
                exit_data = transaction_data.pop()
                exit_data['time_exit'] = df.index[i].strftime('%Y-%m-%d %H:%M:%S')
                exit_data['price_exit'] = current_price
                exit_data['profit_loss'] = current_price - exit_data['price_entry']
                exit_data['macd_exit'] = df['MACD'].iloc[i]
                exit_data['signal_exit'] = df['Signal'].iloc[i]
                exit_data['rsi_exit'] = df['RSI'].iloc[i]
                exit_data['volume_exit'] = df['Volume'].iloc[i]  # Set Volume at exit
                is_profitable = exit_data['profit_loss'] > 0

                # ... inside your signal_generation function, just before the database operation
                print(f"Preparing to insert transaction: Entry Volume - {exit_data['volume_entry']}, Exit Volume - {exit_data['volume_exit']}")
                exit_data['volume_entry'] = int(exit_data['volume_entry'])
                exit_data['volume_exit'] = int(exit_data['volume_exit'])

                # Insert transaction details into the transactions table
                cursor.execute('''INSERT INTO transactions (
                    time_entry, time_exit, price_entry, price_exit, profit_loss, macd_entry, macd_exit, signal_entry, signal_exit, rsi_entry, rsi_exit, volume_entry, volume_exit
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                    exit_data['time_entry'], exit_data['time_exit'], exit_data['price_entry'], exit_data['price_exit'], 
                    exit_data['profit_loss'], exit_data['macd_entry'], exit_data['macd_exit'], exit_data['signal_entry'], 
                    exit_data['signal_exit'], exit_data['rsi_entry'], exit_data['rsi_exit'], exit_data['volume_entry'], exit_data['volume_exit']
                ))

                if is_profitable:
                    cursor.execute('''UPDATE transaction_summary SET 
                                      profitable_count = profitable_count + 1, 
                                      total_profit = total_profit + ?, 
                                      net_result = total_profit - total_loss
                                      WHERE id = 1''', (exit_data['profit_loss'],))
                else:
                    cursor.execute('''UPDATE transaction_summary SET 
                                      non_profitable_count = non_profitable_count + 1, 
                                      total_loss = total_loss + ?, 
                                      net_result = total_profit - total_loss
                                      WHERE id = 1''', (-exit_data['profit_loss'],))
                conn.commit()
                # ... right after the commit() in your signal_generation function:
                cursor.execute("SELECT id, volume_entry, volume_exit FROM transactions ORDER BY id DESC LIMIT 1;")
                last_inserted_row = cursor.fetchone()
                print(f"Last inserted row: {last_inserted_row}")


    df['Buy_Signal_Price'] = buy_signals
    df['Sell_Signal_Price'] = sell_signals




def setup_database(conn):
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY,
        time_entry DATETIME,
        time_exit DATETIME,
        price_entry REAL,
        price_exit REAL,
        profit_loss REAL,
        macd_entry REAL,
        macd_exit REAL,
        signal_entry REAL,
        signal_exit REAL,
        rsi_entry REAL,
        rsi_exit REAL,
        volume_entry INTEGER,
        volume_exit INTEGER
    )''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS transaction_summary (
        id INTEGER PRIMARY KEY,
        profitable_count INTEGER DEFAULT 0,
        non_profitable_count INTEGER DEFAULT 0,
        total_profit REAL DEFAULT 0.0,
        total_loss REAL DEFAULT 0.0,
        net_result REAL DEFAULT 0.0
    )''')

    # Add new columns safely (ignoring errors if they already exist)
    try:
        cursor.execute('ALTER TABLE transaction_summary ADD COLUMN total_profit REAL DEFAULT 0.0')
        cursor.execute('ALTER TABLE transaction_summary ADD COLUMN total_loss REAL DEFAULT 0.0')
        cursor.execute('ALTER TABLE transaction_summary ADD COLUMN net_result REAL DEFAULT 0.0')
    except sqlite3.OperationalError:
        pass  # If the column already exists, do nothing

    cursor.execute('''
    INSERT OR IGNORE INTO transaction_summary (id, profitable_count, non_profitable_count, total_profit, total_loss, net_result)
    VALUES (1, 0, 0, 0.0, 0.0, 0.0)
    ''')
    conn.commit()



def trading_strategy(file_path):
    conn = sqlite3.connect('trading_data.db')
    setup_database(conn)
    df = read_data(file_path)
    df = compute_indicators(df)
    signal_generation(df, conn)
    
    plt.figure(figsize=(14,7))
    plt.plot(df['Close'], label='Close Price', alpha=0.3)
    plt.scatter(df.index, df['Buy_Signal_Price'], label='Buy Signal', marker='^', color='g', s=100)
    plt.scatter(df.index, df['Sell_Signal_Price'], label='Sell Signal', marker='v', color='r', s=100)
    plt.title('Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.show()
    
    def check_volumes(conn):
        cursor = conn.cursor()
        cursor.execute("SELECT id, volume_entry, volume_exit FROM transactions LIMIT 10;")
        rows = cursor.fetchall()
        for row in rows:
            print(row)

# Call this function after your trading_strategy function to print the check
    check_volumes(conn)

    conn.close()
    


# Example usage
# Example usage
file_path = r'C:\Program VC\scalping_strategy\APPLE_data5M.csv'
trading_strategy(file_path)

import csv

def export_transactions_to_csv(conn, csv_file_path):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM transactions;")
    transactions = cursor.fetchall()

    # Column headers as they appear in the SQLite database
    headers = [description[0] for description in cursor.description]

    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(transactions)

# Usage
conn = sqlite3.connect('trading_data.db')
export_csv_file_path = 'transactions_export.csv'
export_transactions_to_csv(conn, export_csv_file_path)
conn.close()







