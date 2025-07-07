import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.stattools import adfuller
#conflict resolved here  wkkj7n-codex/modify-backtest-engine-with-slippage-and-filters
from datetime import datetime
#conflict resolved here 
# conflict markers removed here  f3usdw-codex/modify-backtest-engine-with-slippage-and-filters
from datetime import datetime
#conflict resolved here 
# conflict markers removed here  main
#conflict resolved here  main

from .enhanced_pair_selection import hurst


class PairHealthMonitor:
    """Monitor spread health using rolling ADF and Hurst checks."""

#conflict resolved here  wkkj7n-codex/modify-backtest-engine-with-slippage-and-filters
    def __init__(
        self,
        adf_threshold: float = 0.15,
        hurst_threshold: float = 0.65,
        window: int = 30,
    ):
#conflict resolved here 
    def __init__(self, adf_threshold: float = 0.15, hurst_threshold: float = 0.65, window: int = 30):
#conflict resolved here  main
        self.adf_threshold = adf_threshold
        self.hurst_threshold = hurst_threshold
        self.window = window
        self.logger = logging.getLogger(__name__)

    def evaluate(self, spread: pd.Series) -> pd.DataFrame:
#conflict resolved here  wkkj7n-codex/modify-backtest-engine-with-slippage-and-filters
        """Return DataFrame with rolling ADF, Hurst, volatility z-score and flags."""
#conflict resolved here 
        """Return DataFrame with rolling ADF p-values, Hurst exponent and health flag."""
#conflict resolved here  main
        adf_series = spread.rolling(self.window).apply(
            lambda x: adfuller(x)[1] if x.notna().sum() == self.window else np.nan,
            raw=False,
        )
        hurst_series = spread.rolling(self.window).apply(lambda x: hurst(x.values), raw=False)
#conflict resolved here  wkkj7n-codex/modify-backtest-engine-with-slippage-and-filters
        spread_std = spread.rolling(self.window).std()
        vol_z = (spread_std - spread_std.rolling(self.window).mean()) / spread_std.rolling(self.window).std()

        unstable = adf_series >= self.adf_threshold
        excessive_vol = vol_z > 2
        health = (~unstable) & (hurst_series < self.hurst_threshold) & (~excessive_vol)

        return pd.DataFrame({
            "adf_pvalue": adf_series,
            "hurst": hurst_series,
            "vol_zscore": vol_z,
            "spread_std": spread_std,
            "healthy": health,
            "unstable_cointegration": unstable,
            "excessive_volatility": excessive_vol,
        })

    def log_pair_health(
        self,
        pair_name: str,
        adf_p: float,
        hurst_val: float,
        adv: float,
        is_healthy: bool,
        vol_z: float = np.nan,
        spread_std: float = np.nan,
        unstable: bool = False,
        excessive_vol: bool = False,
    ):
#conflict resolved here 
        health = (adf_series < self.adf_threshold) & (hurst_series < self.hurst_threshold)
        return pd.DataFrame({
            "adf_pvalue": adf_series,
            "hurst": hurst_series,
            "healthy": health,
        })
# conflict markers removed here  f3usdw-codex/modify-backtest-engine-with-slippage-and-filters

    def log_pair_health(self, pair_name: str, adf_p: float, hurst_val: float, adv: float, is_healthy: bool):
#conflict resolved here  main
        """Append pair health statistics to CSV log."""
        try:
            with open("pair_health_log.csv", "a") as f:
                f.write(
#conflict resolved here  wkkj7n-codex/modify-backtest-engine-with-slippage-and-filters
                    f"{datetime.now().isoformat()},{pair_name},{adf_p:.4f},{hurst_val:.4f},{adv:.2f},{vol_z:.2f},{spread_std:.4f},{unstable},{excessive_vol},{is_healthy}\n"
#conflict resolved here 
                    f"{datetime.now().isoformat()},{pair_name},{adf_p:.4f},{hurst_val:.4f},{adv:.2f},{is_healthy}\n"
#conflict resolved here main
                )
        except Exception as e:
            self.logger.error(f"Error logging pair health: {e}")

#conflict resolved here  wkkj7n-codex/modify-backtest-engine-with-slippage-and-filters
#conflict resolved here 
# conflict markers removed here  main
#conflict resolved here >>>>>>> main
