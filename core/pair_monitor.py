import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.stattools import adfuller

from .enhanced_pair_selection import hurst


class PairHealthMonitor:
    """Monitor spread health using rolling ADF and Hurst checks."""

    def __init__(self, adf_threshold: float = 0.15, hurst_threshold: float = 0.65, window: int = 30):
        self.adf_threshold = adf_threshold
        self.hurst_threshold = hurst_threshold
        self.window = window
        self.logger = logging.getLogger(__name__)

    def evaluate(self, spread: pd.Series) -> pd.DataFrame:
        """Return DataFrame with rolling ADF p-values, Hurst exponent and health flag."""
        adf_series = spread.rolling(self.window).apply(
            lambda x: adfuller(x)[1] if x.notna().sum() == self.window else np.nan,
            raw=False,
        )
        hurst_series = spread.rolling(self.window).apply(lambda x: hurst(x.values), raw=False)
        health = (adf_series < self.adf_threshold) & (hurst_series < self.hurst_threshold)
        return pd.DataFrame({
            "adf_pvalue": adf_series,
            "hurst": hurst_series,
            "healthy": health,
        })
