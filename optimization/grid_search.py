import itertools
from dataclasses import replace
from typing import Iterable, Dict, Tuple

import pandas as pd

from backtest.backtest import PairsBacktest, BacktestConfig


def grid_search(
    prices: pd.DataFrame,
    signals: Dict[Tuple[str, str], pd.DataFrame],
    regimes: pd.Series,
    entry_thresholds: Iterable[float],
    exit_thresholds: Iterable[float],
    stop_loss_ks: Iterable[float],
    base_config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Run a grid search over the provided parameter ranges.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data for the assets in each pair.
    signals : dict
        Mapping of pair tuples to DataFrame signals.
    regimes : pd.Series
        Regime series used by the backtester.
    entry_thresholds, exit_thresholds, stop_loss_ks : iterable of float
        Parameter ranges to evaluate.
    base_config : BacktestConfig, optional
        Base configuration used as a template. Defaults to ``BacktestConfig()``.

    Returns
    -------
    pd.DataFrame
        DataFrame containing performance metrics for each parameter
        combination, sorted by Sharpe ratio.
    """
    if base_config is None:
        base_config = BacktestConfig()

    results = []
    for entry_t, exit_t, sl_k in itertools.product(
        entry_thresholds, exit_thresholds, stop_loss_ks
    ):
        cfg = replace(
            base_config,
            zscore_entry_threshold=entry_t,
            zscore_exit_threshold=exit_t,
            stop_loss_k=sl_k,
        )
        bt = PairsBacktest(cfg)
        bt.run_backtest(prices, signals, regimes)
        metrics = bt.get_performance_metrics()
        metrics.update(
            {
                "entry_threshold": entry_t,
                "exit_threshold": exit_t,
                "stop_loss_k": sl_k,
            }
        )
        results.append(metrics)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    if "sharpe_ratio" in df.columns:
        df = df.sort_values("sharpe_ratio", ascending=False)
    return df.reset_index(drop=True)
