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
