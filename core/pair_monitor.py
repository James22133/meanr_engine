import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.stattools import adfuller
# conflict markers removed here  f3usdw-codex/modify-backtest-engine-with-slippage-and-filters
from datetime import datetime
=======
# conflict markers removed here  main

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
# conflict markers removed here  f3usdw-codex/modify-backtest-engine-with-slippage-and-filters

    def log_pair_health(self, pair_name: str, adf_p: float, hurst_val: float, adv: float, is_healthy: bool):
        """Append pair health statistics to CSV log."""
        try:
            with open("pair_health_log.csv", "a") as f:
                f.write(
                    f"{datetime.now().isoformat()},{pair_name},{adf_p:.4f},{hurst_val:.4f},{adv:.2f},{is_healthy}\n"
                )
        except Exception as e:
            self.logger.error(f"Error logging pair health: {e}")

# conflict markers removed here  main
