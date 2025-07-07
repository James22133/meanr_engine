import numpy as np
import pandas as pd


def monte_carlo_spread_simulation(spread: pd.Series, mu: float, sigma: float, days: int = 30, n_paths: int = 1000) -> pd.DataFrame:
    """Generate Monte Carlo simulated spread paths using GBM."""
    dt = 1.0
    paths = np.zeros((days, n_paths))
    paths[0] = spread.iloc[-1]
    for t in range(1, days):
        noise = np.random.normal(0, np.sqrt(dt), n_paths)
        paths[t] = paths[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * noise)
    return pd.DataFrame(paths, index=pd.RangeIndex(days))


def summarize_simulation(paths: pd.DataFrame) -> dict:
    drawdowns = paths.div(paths.iloc[0]).min() - 1
    ttm = paths.apply(lambda x: (x > x[0]).argmax(), axis=0)
    return {
        "expected_drawdown": drawdowns.mean(),
        "worst_drawdown": drawdowns.min(),
        "median_time_to_mean": ttm.median(),
    }
