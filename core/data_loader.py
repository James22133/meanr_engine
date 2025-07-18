"""
Data loader module for fetching and processing market data.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
from .cache import save_to_cache, load_from_cache

class DataLoader:
    """Class for loading and preprocessing market data."""
    
    def __init__(self, config):
        """Initialize the data loader with configuration."""
        self.config = config
        self.data = None
        self.ohlc_data = None
        self.logger = logging.getLogger(__name__)

    def fetch_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch market data for a list of tickers."""
        try:
            self.logger.info(f"Fetching market data for {len(tickers)} tickers...")
            
            # Fetch data for all tickers
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)
            
            # Check for missing tickers
            missing_tickers = []
            for ticker in tickers:
                if ticker not in data['Close'].columns:
                    missing_tickers.append(ticker)
                    self.logger.warning(f"Ticker {ticker} not found in data")
            
            if missing_tickers:
                self.logger.warning(f"Missing tickers: {', '.join(missing_tickers)}")
            
            # Get closing prices
            close_prices = data['Close']

            # Store OHLC data for ATR calculation
            self.ohlc_data = data

            # Liquidity filter based on average dollar volume
            avg_dollar_vol = (data['Close'] * data['Volume']).mean()
            liquid_tickers = avg_dollar_vol[avg_dollar_vol >= 5_000_000].index.tolist()
            removed = set(tickers) - set(liquid_tickers)
            if removed:
                self.logger.info(f"Removing illiquid tickers (<$5M volume): {sorted(removed)}")
            close_prices = close_prices[liquid_tickers]
            if isinstance(self.ohlc_data.columns, pd.MultiIndex):
                self.ohlc_data = self.ohlc_data.loc[:, pd.IndexSlice[:, liquid_tickers]]
            else:
                self.ohlc_data = self.ohlc_data[liquid_tickers]
            
            # Log data shape and date range
            self.logger.info(f"Data shape: {close_prices.shape}")
            self.logger.info(f"Date range: {close_prices.index[0]} to {close_prices.index[-1]}")
            
            # Check for missing values
            missing_values = close_prices.isnull().sum()
            if missing_values.any():
                self.logger.warning("Missing values per column:")
                self.logger.warning(missing_values[missing_values > 0])
            
            # Store the fetched data in self.data
            self.data = close_prices
            
            return close_prices
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            raise

    def get_ohlc_data(self, ticker: str) -> pd.DataFrame:
        """Get OHLC data for a specific ticker."""
        try:
            if self.ohlc_data is not None:
                # Extract OHLC data for the specific ticker
                ohlc = pd.DataFrame({
                    'open': self.ohlc_data['Open'][ticker],
                    'high': self.ohlc_data['High'][ticker],
                    'low': self.ohlc_data['Low'][ticker],
                    'close': self.ohlc_data['Close'][ticker]
                })
                return ohlc
            else:
                self.logger.warning(f"No OHLC data available for {ticker}")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error getting OHLC data for {ticker}: {e}")
            return pd.DataFrame()

    def _log_data_quality(self):
        """Log data quality metrics."""
        if self.data is not None:
            self.logger.info(f"Data shape: {self.data.shape}")
            self.logger.info(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
            self.logger.info(f"Missing values per column:\n{self.data.isnull().sum()}")

    def get_pair_data(self, pair):
        """Retrieve data for a specific pair of tickers, with robust error handling."""
        if self.data is None:
            self.logger.error(f"No data loaded. Cannot get pair data for {pair}.")
            return None
        asset1, asset2 = pair
        # Log available columns for debugging
        self.logger.debug(f"Available columns: {self.data.columns}")
        missing = [a for a in (asset1, asset2) if a not in self.data.columns]
        if missing:
            self.logger.warning(f"Missing tickers in data: {missing} for pair {pair}")
            return None
        pair_data = self.data[[asset1, asset2]].copy()
        # Check for all-NaN columns
        for a in (asset1, asset2):
            if pair_data[a].isna().all():
                self.logger.warning(f"All data is NaN for {a} in pair {pair}")
                return None
        # Forward/backward fill if needed
        pair_data = pair_data.ffill().bfill()
        if pair_data.isna().any().any():
            self.logger.warning(f"Still missing values after fill for pair {pair}")
            return None
        return pair_data 

    def get_all_pairs(self) -> List[Tuple[str, str]]:
        """Generate all possible pairs from the loaded tickers."""
        if self.data is None:
            self.logger.error("No data loaded. Cannot generate pairs.")
            return []
        
        tickers = list(self.data.columns)
        pairs = []
        
        # Generate all possible combinations
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                pairs.append((tickers[i], tickers[j]))
        
        self.logger.info(f"Generated {len(pairs)} pairs from {len(tickers)} tickers")
        return pairs 