"""
Pair analysis module with utility functions for analyzing pairs.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from typing import Tuple, Optional

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