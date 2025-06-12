import pandas as pd
import numpy as np

from backtest.backtest import BacktestConfig
from optimization.grid_search import grid_search


def make_data():
    dates = pd.date_range("2020-01-01", periods=25)
    prices = pd.DataFrame(
        {
            "A": np.linspace(100, 120, 25),
            "B": np.linspace(101, 121, 25) + np.random.normal(0, 0.5, 25),
        },
        index=dates,
    )
    signals = {
        ("A", "B"): pd.DataFrame(
            {
                "entry_long": [False] * 20 + [True] + [False] * 4,
                "entry_short": [False] * 25,
                "exit": [False] * 21 + [True] + [False] * 3,
                "stop_loss_k": [2.0] * 25,
            },
            index=dates,
        )
    }
    regimes = pd.Series(index=dates, data=0)
    return prices, signals, regimes


def test_grid_search_runs():
    prices, signals, regimes = make_data()
    result = grid_search(
        prices,
        signals,
        regimes,
        entry_thresholds=[1.5, 2.0],
        exit_thresholds=[0.1],
        stop_loss_ks=[2.0],
        base_config=BacktestConfig(slippage_bps=0.0, commission_bps=0.0),
    )
    assert not result.empty
    assert {"entry_threshold", "exit_threshold", "stop_loss_k"}.issubset(result.columns)
