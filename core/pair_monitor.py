import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.stattools import adfuller
from datetime import datetime

from .enhanced_pair_selection import hurst


class PairHealthMonitor:
    """Monitor spread health using rolling ADF and Hurst checks."""

    def __init__(
        self,
        adf_threshold: float = 0.15,
        hurst_threshold: float = 0.65,
        window: int = 30,
    ):
        self.adf_threshold = adf_threshold
        self.hurst_threshold = hurst_threshold
        self.window = window
        self.logger = logging.getLogger(__name__)

    def evaluate(self, spread: pd.Series) -> pd.DataFrame:
        """Return DataFrame with rolling ADF, Hurst, volatility z-score and flags."""
        adf_series = spread.rolling(self.window).apply(
            lambda x: adfuller(x)[1] if x.notna().sum() == self.window else np.nan,
            raw=False,
        )
        hurst_series = spread.rolling(self.window).apply(lambda x: hurst(x.values), raw=False)
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
        """Append pair health statistics to CSV log."""
        try:
            with open("pair_health_log.csv", "a") as f:
                f.write(
                    f"{datetime.now().isoformat()},{pair_name},{adf_p:.4f},{hurst_val:.4f},{adv:.2f},{vol_z:.2f},{spread_std:.4f},{unstable},{excessive_vol},{is_healthy}\n"
                )
        except Exception as e:
            self.logger.error(f"Error logging pair health: {e}")

