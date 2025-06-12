import pandas as pd
import numpy as np

from backtest.backtest import BacktestConfig
from optimization.grid_search import grid_search, generate_signals


def make_data():
    dates = pd.date_range("2020-01-01", periods=25)
    prices = pd.DataFrame(
        {
            "A": np.linspace(100, 120, 25),
            "B": np.linspace(101, 121, 25) + np.random.normal(0, 0.5, 25),
        },
        index=dates,
    )
    regimes = pd.Series(index=dates, data=0)
    return prices, regimes


def test_grid_search_runs():
    prices, regimes = make_data()
    result = grid_search(
        prices,
        generate_signals,
        regimes,
        entry_thresholds=[1.5, 2.0],
        exit_thresholds=[0.1],
        stop_loss_ks=[2.0],
        base_config=BacktestConfig(slippage_bps=0.0, commission_bps=0.0),
    )
    assert not result.empty
    assert {"entry_threshold", "exit_threshold", "stop_loss_k"}.issubset(result.columns)


def test_parameters_affect_sharpe():
    prices, regimes = make_data()
    result = grid_search(
        prices,
        generate_signals,
        regimes,
        entry_thresholds=[0.5, 2.0],
        exit_thresholds=[0.1],
        stop_loss_ks=[2.0],
        base_config=BacktestConfig(slippage_bps=0.0, commission_bps=0.0),
    )
    sharpe_ratios = result["sharpe_ratio"]
    assert len(sharpe_ratios.unique()) > 1
