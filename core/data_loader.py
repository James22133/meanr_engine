"""
Data loader module for fetching and processing market data.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional
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

    def fetch_data(self) -> pd.DataFrame:
        """Fetch market data for all tickers."""
        try:
            self.logger.info("Fetching market data...")

            cache_file = os.path.join(self.config.data_dir, "market_data.pkl")
            if self.config.use_cache and os.path.exists(cache_file) and not self.config.force_refresh:
                self.logger.info("Loading data from cache")
                self.data = load_from_cache(cache_file)
                self._log_data_quality()
                return self.data

            data = {}

            raw_data = yf.download(
                self.config.etf_tickers,
                start=self.config.start_date,
                end=self.config.end_date,
                group_by="ticker",
                progress=False,
            )

            if raw_data.empty:
                raise ValueError("No data fetched for any tickers")

            if isinstance(raw_data.columns, pd.MultiIndex):
                level0 = raw_data.columns.get_level_values(0)
                if "Close" in level0:
                    for ticker in self.config.etf_tickers:
                        col = ("Close", ticker)
                        if col in raw_data.columns:
                            data[ticker] = raw_data[col]
                        else:
                            self.logger.warning(f"No 'Close' data found for {ticker}")
                else:
                    for ticker in self.config.etf_tickers:
                        col = (ticker, "Close")
                        if col in raw_data.columns:
                            data[ticker] = raw_data[col]
                        else:
                            self.logger.warning(f"No 'Close' data found for {ticker}")
            else:
                if len(self.config.etf_tickers) == 1 and "Close" in raw_data.columns:
                    ticker = self.config.etf_tickers[0]
                    data[ticker] = raw_data["Close"]
                else:
                    for ticker in self.config.etf_tickers:
                        if ticker in raw_data.columns:
                            data[ticker] = raw_data[ticker]
                        else:
                            self.logger.warning(f"No data found for {ticker}")

            if not data:
                raise ValueError("No data fetched for any tickers")

            # Convert to DataFrame
            self.data = pd.DataFrame(data)
            
            # Log data quality metrics
            self._log_data_quality()

            if self.config.use_cache:
                save_to_cache(self.data, cache_file)

            return self.data

        except Exception as e:
            self.logger.error(f"Error in fetch_data: {str(e)}")
            raise

    def _log_data_quality(self):
        """Log data quality metrics."""
        if self.data is not None:
            self.logger.info(f"Data shape: {self.data.shape}")
            self.logger.info(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
            self.logger.info(f"Missing values per column:\n{self.data.isnull().sum()}")

    def get_pair_data(self, pair: List[str]) -> Optional[pd.DataFrame]:
        """Get data for a specific pair of tickers."""
        if self.data is None:
            raise ValueError("Data not loaded. Call fetch_data() first.")
        
        try:
            pair_data = self.data[pair].copy()
            if pair_data.isnull().any().any():
                self.logger.warning(f"Missing values found in pair {pair}")
                pair_data = pair_data.dropna()
            
            if len(pair_data) < self.config.pair_selection.min_data_points:
                self.logger.warning(f"Insufficient data points for pair {pair}")
                return None
                
            return pair_data
        except Exception as e:
            self.logger.error(f"Error getting pair data for {pair}: {str(e)}")
            return None 
