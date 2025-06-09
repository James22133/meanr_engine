import yfinance as yf
import pandas as pd


class DataFetchError(Exception):
    """Custom exception for data fetching errors."""
    pass

def fetch_etf_data(tickers, start_date, end_date):
    """
    Fetches historical adjusted close price data for a list of ETF tickers.

    Args:
        tickers (list): A list of ticker symbols.
        start_date (str): The start date in "YYYY-MM-DD" format.
        end_date (str): The end date in "YYYY-MM-DD" format.

    Returns:
        pandas.DataFrame: A DataFrame with adjusted close prices indexed by date.
    """
    # Download data with auto_adjust=True to get adjusted prices
    try:
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    except Exception as e:
        raise DataFetchError(f"Failed to download data for {tickers}: {e}")

    if data.empty:
        raise DataFetchError("Downloaded data is empty. Please check ticker symbols or network connection.")
    
    # If multiple tickers, data will have a MultiIndex columns
    if isinstance(tickers, list) and len(tickers) > 1:
        # Get the 'Close' prices (which are now adjusted)
        return data['Close']
    else:
        # For single ticker, return the Close prices
        return data['Close']

# Example Usage (for demonstration)
# if __name__ == "__main__":
#     etf_tickers = ["SPY", "QQQ", "IWM", "EFA", "EMB", "GLD", "SLV", "USO", "TLT", "IEF"]
#     start = "2020-01-01"
#     end = "2023-01-01"
#     etf_data = fetch_etf_data(etf_tickers, start, end)
#     print(etf_data.head()) 