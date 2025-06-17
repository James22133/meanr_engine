"""
Data loader module for fetching and processing market data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional
import logging
from tqdm import tqdm

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
            data = {}
            
            for ticker in tqdm(self.config.etf_tickers, desc="Fetching data"):
                try:
                    ticker_data = yf.download(
                        ticker,
                        start=self.config.start_date,
                        end=self.config.end_date,
                        progress=False
                    )
                    if not ticker_data.empty and ('Close', ticker) in ticker_data.columns:
                        close_series = ticker_data[('Close', ticker)]
                        if isinstance(close_series, pd.Series):
                            data[ticker] = close_series
                        else:
                            self.logger.warning(f"'Close' for {ticker} is not a Series. Skipping.")
                            self.logger.debug(f"DataFrame structure for {ticker}:\n{ticker_data.info()}")
                            self.logger.debug(f"DataFrame head for {ticker}:\n{ticker_data.head()}")
                    else:
                        self.logger.warning(f"No 'Close' data found for {ticker}")
                except Exception as e:
                    self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
                    continue

            if not data:
                raise ValueError("No data fetched for any tickers")

            # Convert to DataFrame
            self.data = pd.DataFrame(data)
            
            # Log data quality metrics
            self._log_data_quality()
            
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