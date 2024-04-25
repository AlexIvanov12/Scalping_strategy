import yfinance as yf
import pandas as pd

def calculate_technical_indicators(df):
    # Calculate indicators as previously defined
    df['MA_Short'] = df['Close'].rolling(window=5).mean()
    df['MA_Long'] = df['Close'].rolling(window=8).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def identify_trading_signals(df):
    df['Buy'] = 0
    df['Sell'] = 0
    for i in range(1, len(df)):
        if (df['MA_Short'][i] > df['MA_Long'][i] and df['RSI'][i] < 30 and df['MACD'][i] > df['Signal'][i]):
            df['Buy'][i] = 1
        if (df['MA_Short'][i] < df['MA_Long'][i] and df['RSI'][i] > 70 and df['MACD'][i] < df['Signal'][i]):
            df['Sell'][i] = 1
    return df

# Placeholder functions for API integration
def place_limit_order(ticker, quantity, price, order_type):
    print(f"Placing {order_type} Limit Order for {quantity} shares of {ticker} at ${price}")
    # Here you would integrate with your brokerage API to place an order.

def place_market_order(ticker, quantity, order_type):
    print(f"Placing {order_type} Market Order for {quantity} shares of {ticker}")
    # Here you would integrate with your brokerage API to place an order.

def execute_trades(df, ticker):
    for i in range(len(df)):
        if df['Buy'][i] == 1:
            # Assuming you decide the quantity and the limit price based on some strategy
            quantity = 10  # example quantity
            price = df['Close'][i] * 0.995  # slightly below the close price for limit buy
            place_limit_order(ticker, quantity, price, 'Buy')
        elif df['Sell'][i] == 1:
            quantity = 10  # example quantity
            place_market_order(ticker, quantity, 'Sell')

def analyze_and_trade(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1mo", interval='5m')
    df = calculate_technical_indicators(df)
    df = identify_trading_signals(df)
    execute_trades(df, ticker)

# Example usage
analyze_and_trade('AAPL')


#verson 2

import yfinance as yf
import pandas as pd
import numpy as np
import datetime

# Initialize a DataFrame to log trades
trade_log = pd.DataFrame(columns=['Date', 'Ticker', 'Action', 'Quantity', 'Price', 'Stop Loss', 'Profit Target', 'Outcome'])

def place_limit_order(ticker, quantity, price, order_type):
    # Simulated API call
    print(f"Placing {order_type} Limit Order for {quantity} shares of {ticker} at ${price}")
    # Log the trade
    global trade_log
    trade_log = trade_log.append({
        'Date': datetime.datetime.now(),
        'Ticker': ticker,
        'Action': 'Buy',
        'Quantity': quantity,
        'Price': price,
        'Stop Loss': price * 0.998,  # Example stop-loss
        'Profit Target': price * 1.005,  # Example profit target
        'Outcome': None  # Outcome to be updated post-trade
    }, ignore_index=True)

def place_market_order(ticker, quantity, order_type):
    # Simulated API call
    print(f"Placing {order_type} Market Order for {quantity} shares of {ticker}")
    # Log the trade
    global trade_log
    trade_log = trade_log.append({
        'Date': datetime.datetime.now(),
        'Ticker': ticker,
        'Action': 'Sell',
        'Quantity': quantity,
        'Price': None,  # Price determined at market
        'Stop Loss': None,
        'Profit Target': None,
        'Outcome': None  # Outcome to be updated post-trade
    }, ignore_index=True)

def review_trades():
    # Simulate reviewing trades and updating the outcome
    global trade_log
    for index, row in trade_log.iterrows():
        if row['Action'] == 'Sell':
            entry = trade_log[(trade_log['Ticker'] == row['Ticker']) & (trade_log['Action'] == 'Buy')].iloc[-1]
            if entry['Profit Target'] <= row['Price']:
                trade_log.at[index, 'Outcome'] = 'Profit Target Hit'
            elif entry['Stop Loss'] >= row['Price']:
                trade_log.at[index, 'Outcome'] = 'Stop Loss Hit'
            else:
                trade_log.at[index, 'Outcome'] = 'Normal Exit'

def analyze_performance():
    global trade_log
    profits = trade_log['Outcome'].value_counts()
    print("Trade Outcomes:\n", profits)

    # Example of adjusting strategy
    if profits.get('Stop Loss Hit', 0) > profits.get('Profit Target Hit', 0):
        print("Consider widening the stop-loss or reducing the profit target.")
    else:
        print("Strategy appears balanced.")

def analyze_and_trade(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1mo", interval='5m')
    df = calculate_technical_indicators(df)
    df = identify_trading_signals(df)
    execute_trades(df, ticker)
    review_trades()
    analyze_performance()

# Example usage
analyze_and_trade('AAPL')

# To export the trade log to a CSV file for further analysis:
trade_log.to_csv('trade_log.csv')

