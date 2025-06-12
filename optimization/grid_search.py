import itertools
from dataclasses import replace
from typing import Iterable, Dict, Tuple, Callable

import pandas as pd

from backtest.backtest import PairsBacktest, BacktestConfig


def generate_signals(
    prices: pd.DataFrame,
    regimes: pd.Series,
    entry_threshold: float,
    exit_threshold: float,
    stop_loss_k: float,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Generate simple trading signals for a single pair using Z-score."""
    asset1, asset2 = prices.columns[:2]
    spread = prices[asset1] - prices[asset2]
    rolling_mean = spread.rolling(5).mean()
    rolling_std = spread.rolling(5).std()
    z_score = (spread - rolling_mean) / rolling_std
    z_score = z_score.fillna(0.0)

    entry_long = z_score < -entry_threshold
    entry_short = z_score > entry_threshold
    exit_signal = z_score.abs() < exit_threshold

    signals = pd.DataFrame(
        {
            "entry_long": entry_long,
            "entry_short": entry_short,
            "exit": exit_signal,
            "stop_loss_k": stop_loss_k,
        },
        index=prices.index,
    )

    return {(asset1, asset2): signals}


def grid_search(
    prices: pd.DataFrame,
    signal_func: Callable[
        [pd.DataFrame, pd.Series, float, float, float],
        Dict[Tuple[str, str], pd.DataFrame],
    ],
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
    signal_func : callable
        Function used to generate signals. It should accept ``prices``,
        ``regimes``, ``entry_threshold``, ``exit_threshold`` and ``stop_loss_k``
        and return a mapping of pair tuples to signal DataFrames.
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
        generated_signals = signal_func(
            prices, regimes, entry_t, exit_t, sl_k
        )
        bt.run_backtest(prices, generated_signals, regimes)
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
