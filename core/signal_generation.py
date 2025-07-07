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
    def get_dynamic_thresholds(vix: float, atr: float, close_mean: float) -> dict:
        base_entry = 2.0
        base_exit = 0.5
        entry_adj = 0.25 * (vix / 20) + 0.5 * (atr / close_mean)
        exit_adj = 0.1 * (vix / 20)
        return {"entry": base_entry + entry_adj, "exit": base_exit + exit_adj}

# conflict markers removed here  f3usdw-codex/modify-backtest-engine-with-slippage-and-filters
    @staticmethod
    def should_halt_signals(vix: float, spy_ret_5d: float) -> bool:
        return (vix > 35) or (spy_ret_5d < -0.07)

# conflict markers removed here  main
    def generate_signals(
        self,
        pair_data: pd.DataFrame,
        pair_name: Tuple[str, str],
        vix_series: pd.Series,
# conflict markers removed here  f3usdw-codex/modify-backtest-engine-with-slippage-and-filters
        spy_series: Optional[pd.Series] = None,
=======
# conflict markers removed here  main
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
        close_mean = ((price1 + price2) / 2).reindex(z.index)
# conflict markers removed here  f3usdw-codex/modify-backtest-engine-with-slippage-and-filters
        if spy_series is not None:
            spy_ret_5d = spy_series.pct_change(5).reindex(z.index).fillna(0)
        else:
            spy_ret_5d = pd.Series(0, index=z.index)
=======
# conflict markers removed here  main

        thresh_entry = []
        thresh_exit = []
        for dt in z.index:
            thr = self.get_dynamic_thresholds(vix.loc[dt], atr.loc[dt], close_mean.loc[dt])
            thresh_entry.append(thr["entry"])
            thresh_exit.append(thr["exit"])
        thresh_entry = pd.Series(thresh_entry, index=z.index)
        thresh_exit = pd.Series(thresh_exit, index=z.index)

        entries = pd.Series(0, index=z.index)
        entries[z < -thresh_entry] = 1
        entries[z > thresh_entry] = -1
# conflict markers removed here  f3usdw-codex/modify-backtest-engine-with-slippage-and-filters
        if self.config.get('backtest', {}).get('stress_filtering', True):
            halt_mask = [self.should_halt_signals(vix.loc[dt], spy_ret_5d.loc[dt]) for dt in z.index]
            entries[pd.Series(halt_mask, index=z.index)] = 0
# conflict markers removed here  main
        exits = (z.abs() <= thresh_exit)

        if pair_health is not None and "healthy" in pair_health.columns:
            health = pair_health.reindex(z.index)["healthy"]
            entries[~health] = 0

        return pd.DataFrame({"entries": entries, "exits": exits, "z_score": z})
