import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd


class SignalGenerator:
    """Generate trading signals using adaptive thresholds."""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def get_dynamic_thresholds(pair_atr: float, atr_mean: float) -> dict:
        """Return entry/exit thresholds scaled by pair-specific volatility."""
        base_entry = 2.0
        base_exit = 0.5

        if np.isnan(pair_atr) or atr_mean == 0 or np.isnan(atr_mean):
            vol_ratio = 1.0
        else:
            vol_ratio = pair_atr / atr_mean

        return {"entry": base_entry * vol_ratio, "exit": base_exit * vol_ratio}

    @staticmethod
    def should_halt_signals(vix: float, spy_ret_5d: float) -> bool:
        return (vix > 35) or (spy_ret_5d < -0.07)

    def generate_signals(
        self,
        pair_data: pd.DataFrame,
        pair_name: Tuple[str, str],
        vix_series: pd.Series,
        spy_series: Optional[pd.Series] = None,
        pair_health: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if pair_data is None or pair_data.empty:
            return pd.DataFrame()

        price1, price2 = pair_data.iloc[:, 0], pair_data.iloc[:, 1]
        spread = price1 - price2
        lookback = self.config.get("signals", {}).get("lookback", 20)
        z = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()
        atr = spread.diff().abs().rolling(14).mean()
        vix = vix_series.reindex(z.index).fillna(method="ffill")

        if spy_series is not None:
            spy_ret_5d = spy_series.pct_change(5).reindex(z.index).fillna(0)
        else:
            spy_ret_5d = pd.Series(0, index=z.index)

        atr_mean = atr.mean()
        thresh_entry = []
        thresh_exit = []
        for dt in z.index:
            thr = self.get_dynamic_thresholds(atr.loc[dt], atr_mean)
            thresh_entry.append(thr["entry"])
            thresh_exit.append(thr["exit"])
        thresh_entry = pd.Series(thresh_entry, index=z.index)
        thresh_exit = pd.Series(thresh_exit, index=z.index)

        entries = pd.Series(0, index=z.index)
        entries[z < -thresh_entry] = 1
        entries[z > thresh_entry] = -1
        if self.config.get('backtest', {}).get('stress_filtering', True):
            halt_mask = [self.should_halt_signals(vix.loc[dt], spy_ret_5d.loc[dt]) for dt in z.index]
            entries[pd.Series(halt_mask, index=z.index)] = 0
        exits = (z.abs() <= thresh_exit)

        if pair_health is not None and "healthy" in pair_health.columns:
            health = pair_health.reindex(z.index)["healthy"].fillna(False)
            strict = self.config.get('backtest', {}).get('health_strict_mode', False)
            if strict:
                entries[~health] = 0
            else:
                unhealthy_days = (~health).sum()
                if unhealthy_days > 0:
                    self.logger.warning(
                        f"{pair_name} has {unhealthy_days} unhealthy days - signals not filtered due to strict mode off"
                    )

        return pd.DataFrame({"entries": entries, "exits": exits, "z_score": z})
