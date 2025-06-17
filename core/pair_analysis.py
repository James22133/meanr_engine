"""
Pair analysis module with utility functions for analyzing pairs.
"""

import numpy as np
import pandas as pd
import logging
from statsmodels.tsa.stattools import adfuller, coint
from pykalman import KalmanFilter
from typing import Optional, Tuple

def calculate_hurst_exponent(time_series: pd.Series, lags: Optional[list] = None) -> float:
    """
    Calculate the Hurst exponent of a time series.
    
    Args:
        time_series: The time series to analyze
        lags: List of lags to use in the calculation. If None, uses default lags.
        
    Returns:
        float: The Hurst exponent (H)
        - H < 0.5: Mean-reverting
        - H = 0.5: Random walk
        - H > 0.5: Trending
    """
    if lags is None:
        lags = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    # Calculate price changes
    returns = np.log(time_series / time_series.shift(1))
    returns = returns.dropna()
    
    # Calculate variance of returns
    tau = []
    lagged_var = []
    
    for lag in lags:
        # Calculate variance of lagged differences
        tau.append(lag)
        lagged_var.append(np.sqrt(returns.rolling(lag).var().mean()))
    
    # Linear fit to get Hurst exponent
    m = np.polyfit(np.log(tau), np.log(lagged_var), 1)
    hurst = m[0] / 2.0
    
    return hurst

def calculate_adf(time_series: pd.Series) -> float:
    """
    Calculate the Augmented Dickey-Fuller test p-value.
    
    Args:
        time_series: The time series to test for stationarity
        
    Returns:
        float: The p-value from the ADF test
        - p-value < 0.05: Stationary
        - p-value >= 0.05: Non-stationary
    """
    try:
        adf_result = adfuller(time_series.dropna())
        return adf_result[1]  # Return p-value
    except Exception as e:
        print(f"Error in ADF calculation: {str(e)}")
        return 1.0  # Return high p-value to indicate non-stationarity

def calculate_rolling_correlation(series1: pd.Series, series2: pd.Series, 
                                window: int = 60) -> pd.Series:
    """
    Calculate rolling correlation between two series.
    
    Args:
        series1: First time series
        series2: Second time series
        window: Rolling window size
        
    Returns:
        pd.Series: Rolling correlation values
    """
    return series1.rolling(window).corr(series2)

def calculate_spread_stability(spread: pd.Series, window: int = 60) -> float:
    """
    Calculate spread stability score.
    
    Args:
        spread: The spread series
        window: Window size for calculations
        
    Returns:
        float: Stability score between 0 and 1
    """
    try:
        # Calculate rolling statistics
        rolling_std = spread.rolling(window).std()
        rolling_mean = spread.rolling(window).mean()
        
        # Calculate stability metrics
        std_stability = 1 - (rolling_std / rolling_std.max())
        mean_stability = 1 - (rolling_mean.diff().abs() / rolling_mean.diff().abs().max())
        
        # Combine metrics
        stability = (std_stability.mean() + mean_stability.mean()) / 2
        
        return stability
    except Exception as e:
        print(f"Error calculating spread stability: {str(e)}")
        return 0.0

def calculate_zscore_volatility(zscore: pd.Series, window: int = 60) -> float:
    """
    Calculate Z-score volatility.
    
    Args:
        zscore: The Z-score series
        window: Window size for calculations
        
    Returns:
        float: Z-score volatility
    """
    try:
        return zscore.rolling(window).std().mean()
    except Exception as e:
        print(f"Error calculating Z-score volatility: {str(e)}")
        return float('inf')


def calculate_cointegration(series1: pd.Series, series2: pd.Series) -> Tuple[float, float, Tuple[float, float, float]]:
    """Return Engle-Granger cointegration test statistics.

    Parameters
    ----------
    series1, series2 : pd.Series
        Time series to test.

    Returns
    -------
    tuple
        ``(t_statistic, p_value, critical_values)``. ``(None, None, None)`` is
        returned if the input is invalid.
    """

    if series1.isnull().all() or series2.isnull().all():
        return None, None, None
    try:
        aligned = pd.concat([series1, series2], axis=1).dropna()
        if len(aligned) < 20:
            return None, None, None
        t_stat, p_val, crit = coint(aligned.iloc[:, 0], aligned.iloc[:, 1])
        return t_stat, p_val, crit
    except Exception as e:  # pragma: no cover - log and return NaNs
        logging.getLogger(__name__).error("Error calculating cointegration: %s", e)
        return None, None, None


def apply_kalman_filter(y: pd.Series, X: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate dynamic hedge ratio using a Kalman filter."""

    aligned = pd.concat([y, X], axis=1).dropna()
    y_a = aligned.iloc[:, 0].values
    X_a = aligned.iloc[:, 1].values

    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mats = np.array([[1, x] for x in X_a]).reshape(-1, 1, 2)

    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mats,
        observation_covariance=1.0,
        transition_covariance=trans_cov,
    )

    states, covs = kf.filter(y_a)
    return states, covs


def apply_kalman_filter_rolling(y: pd.Series, X: pd.Series, window: int = 60) -> pd.DataFrame:
    """Estimate Kalman filter parameters using a rolling window."""

    aligned = pd.concat([y, X], axis=1).dropna()
    n = len(aligned)
    states = np.full((n, 2), np.nan)

    for i in range(window, n + 1):
        sub_y = aligned.iloc[i - window:i, 0]
        sub_X = aligned.iloc[i - window:i, 1]
        kf_states, _ = apply_kalman_filter(sub_y, sub_X)
        states[i - 1] = kf_states[-1]

    return pd.DataFrame(states, index=aligned.index, columns=["alpha", "beta"])


def calculate_spread_and_zscore(y: pd.Series, X: pd.Series, states: np.ndarray, rolling_window: int = 20) -> pd.Series:
    """Return rolling Z-score of the Kalman filter spread."""

    aligned = pd.concat([y, X], axis=1).dropna()
    y_a = aligned.iloc[:, 0]
    X_a = aligned.iloc[:, 1]
    beta = pd.Series(states[:, 1], index=aligned.index)
    spread = y_a - beta * X_a
    mean = spread.rolling(window=rolling_window).mean()
    std = spread.rolling(window=rolling_window).std()
    return (spread - mean) / std


def rolling_cointegration(series1: pd.Series, series2: pd.Series, window: int = 60) -> pd.Series:
    """Compute rolling Engle-Granger cointegration p-value."""

    pvals = []
    idxs = []
    s1 = series1.values
    s2 = series2.values
    index = series1.index
    for i in range(window - 1, len(series1)):
        w1 = s1[i - window + 1 : i + 1]
        w2 = s2[i - window + 1 : i + 1]
        if np.isnan(w1).any() or np.isnan(w2).any():
            pvals.append(np.nan)
        else:
            try:
                pval = coint(w1, w2)[1]
            except Exception:
                pval = np.nan
            pvals.append(pval)
        idxs.append(index[i])
    return pd.Series(pvals, index=idxs)


def hurst_exponent(ts: pd.Series, min_lag: int = 2, max_lag: int = 20) -> float:
    """Estimate the Hurst exponent of a time series."""

    lags = range(min_lag, max_lag)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0


def adf_pvalue(series: pd.Series) -> float:
    """Return the p-value from the Augmented Dickey-Fuller test."""

    try:
        return adfuller(series.dropna())[1]
    except Exception:
        return np.nan


def rolling_hurst(series: pd.Series, window: int = 60) -> pd.Series:
    """Compute rolling Hurst exponent for a time series."""

    def hurst_win(x: np.ndarray) -> float:
        lags = range(2, 20)
        tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    return series.rolling(window=window, min_periods=window).apply(hurst_win, raw=True)


def rolling_adf(series: pd.Series, window: int = 60) -> pd.Series:
    """Compute rolling ADF p-value for a time series."""

    def adf_win(x: np.ndarray) -> float:
        try:
            return adfuller(x)[1]
        except Exception:
            return np.nan

    return series.rolling(window=window, min_periods=window).apply(adf_win, raw=True)
