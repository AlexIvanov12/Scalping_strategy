import yfinance as yf
import pandas as pd

def fetch_highly_liquid_stocks(index_ticker, min_volume):
    """
    Fetches and returns highly liquid stocks from a given stock index.
    
    Args:
    index_ticker (str): Ticker symbol of the stock index.
    min_volume (int): Minimum average trading volume to consider for liquidity.

    Returns:
    DataFrame: List of highly liquid stocks.
    """
    # Fetch index components
    index_components = yf.Ticker(index_ticker).components
    # Initialize a DataFrame to store stock data
    stocks_data = pd.DataFrame()
    
    # Loop over each component in the index
    for ticker in index_components:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        # Fetch historical market data
        hist = stock.history(period="1mo")  # 1 month of data
        # Calculate average volume
        avg_volume = hist['Volume'].mean()
        
        if avg_volume >= min_volume:
            # If average volume meets the threshold, add to DataFrame
            stocks_data = stocks_data.append({
                'Ticker': ticker,
                'Average Volume': avg_volume,
                'Current Price': hist['Close'][-1]
            }, ignore_index=True)
    
    return stocks_data

# Example usage
highly_liquid_stocks = fetch_highly_liquid_stocks('SPY', 5000000)  # S&P 500 stocks with at least 5 million average volume
print(highly_liquid_stocks)
