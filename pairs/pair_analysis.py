import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts
from pykalman import KalmanFilter
import logging

logger = logging.getLogger(__name__)

def calculate_cointegration(series1, series2):
    """
    Calculate the Engle-Granger cointegration test result.

    Args:
        series1 (pd.Series): The first time series.
        series2 (pd.Series): The second time series.

    Returns:
        tuple: (cointegration_t_stat, p_value, critical_values) or (None, None, None) if input is invalid.
    """
    if series1.isnull().all() or series2.isnull().all():
        return None, None, None
    try:
        # Drop NaNs to ensure equal length and valid input for cointegration test
        aligned_series = pd.concat([series1, series2], axis=1).dropna()
        if len(aligned_series) < 20: # Minimum observations required for cointegration test
             return None, None, None
        result = ts.coint(aligned_series.iloc[:, 0], aligned_series.iloc[:, 1])
        return result[0], result[1], result[2]
    except Exception as e:
        logger.error("Error calculating cointegration: %s", e)
        return None, None, None

def apply_kalman_filter(y, X):
    """
    Apply a Kalman Filter to estimate the dynamic hedge ratio (beta).

    Args:
        y (pd.Series): The dependent variable (series1).
        X (pd.Series): The independent variable (series2).

    Returns:
        tuple: (states, covs) where states are the estimated parameters (alpha, beta) and covs are their covariances.
    """
    # Ensure data is aligned and no NaNs
    aligned_data = pd.concat([y, X], axis=1).dropna()
    y_aligned = aligned_data.iloc[:, 0].values
    X_aligned = aligned_data.iloc[:, 1].values
    
    # Initialize Kalman Filter parameters
    delta = 1e-5  # tune this
    trans_cov = delta / (1 - delta) * np.eye(2)
    
    # Create observation matrix for each time step
    observation_matrices = np.array([[1, x] for x in X_aligned]).reshape(-1, 1, 2)
    
    # Initialize Kalman Filter
    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=observation_matrices,
        observation_covariance=1.0,
        transition_covariance=trans_cov
    )
    
    # Run Kalman Filter
    states_pred, covs_pred = kf.filter(y_aligned)
    
    # Convert states back to pandas Series with original index
    states_series = pd.Series(states_pred[:, 1], index=aligned_data.index)
    
    return states_pred, covs_pred

def apply_kalman_filter_rolling(y: pd.Series, X: pd.Series, window: int = 60) -> pd.DataFrame:
    """Estimate Kalman filter parameters using a rolling window.

    Parameters
    ----------
    y : pd.Series
        Dependent variable.
    X : pd.Series
        Independent variable.
    window : int, optional
        Number of observations to use for each re-estimation, by default ``60``.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the filtered ``alpha`` and ``beta`` for each time
        step. The first ``window - 1`` rows are ``NaN`` because there is not
        enough history to estimate the filter yet.
    """

    aligned = pd.concat([y, X], axis=1).dropna()
    n = len(aligned)
    states = np.full((n, 2), np.nan)

    for i in range(window, n + 1):
        sub_y = aligned.iloc[i - window:i, 0]
        sub_X = aligned.iloc[i - window:i, 1]
        kf_states, _ = apply_kalman_filter(sub_y, sub_X)
        states[i - 1] = kf_states[-1]

    return pd.DataFrame(states, index=aligned.index, columns=["alpha", "beta"])

def calculate_spread_and_zscore(y, X, states, rolling_window=20):
    """
    Calculate the spread (y - beta*X) and its rolling Z-score.

    Args:
        y (pd.Series): The first time series.
        X (pd.Series): The second time series.
        states (np.ndarray): The estimated states (alpha, beta) from Kalman Filter.
        rolling_window (int, optional): Window size for rolling statistics used
            when computing the Z-score. Defaults to 20.

    Returns:
        pd.Series: The rolling Z-score of the spread.
    """
    # Ensure indices are aligned
    aligned_data = pd.concat([y, X], axis=1).dropna()
    aligned_y = aligned_data.iloc[:, 0]
    aligned_X = aligned_data.iloc[:, 1]
    
    # Get beta values for aligned data
    aligned_states_beta = pd.Series(states[:, 1], index=aligned_data.index)
    
    # Calculate spread
    spread = aligned_y - aligned_states_beta * aligned_X
    
    # Calculate rolling mean and standard deviation of the spread
    rolling_mean = spread.rolling(window=rolling_window).mean()
    rolling_std = spread.rolling(window=rolling_window).std()
    
    # Calculate Z-score
    z_score = (spread - rolling_mean) / rolling_std
    
    return z_score

# Example Usage (for demonstration)
# if __name__ == "__main__":
#     # Assume data is fetched and available as pandas Series price_y, price_X
#     # Dummy data for demonstration
#     dates = pd.date_range(start='2023-01-01', periods=100)
#     price_y = pd.Series(np.random.rand(100).cumsum() + 50, index=dates)
#     price_X = pd.Series(np.random.rand(100).cumsum() + 52, index=dates)

#     # Assuming coint_p_value < 0.05 for this dummy data
#     # coint_t, coint_p, coint_crit = calculate_cointegration(price_y, price_X)
#     # print(f"Cointegration p-value: {coint_p}")

#     # Apply Kalman Filter
#     states, covs = apply_kalman_filter(price_y, price_X)

#     # Calculate Spread and Z-score
#     z_score = calculate_spread_and_zscore(price_y, price_X, states)
#     print(z_score.head())
#     z_score.plot(title="Spread Z-score")
#     import matplotlib.pyplot as plt
#     plt.show()

# Example: if you expect prices to be a DataFrame
# dates = prices.index 

# --- Enhancement: Rolling Cointegration, Hurst, and ADF Filters

def rolling_cointegration(series1, series2, window=60):
    """
    Compute rolling Engle-Granger cointegration p-value for each window.
    Returns a Series of p-values aligned to the right edge of each window.
    """
    pvals = []
    idxs = []
    s1 = series1.values
    s2 = series2.values
    index = series1.index
    for i in range(window - 1, len(series1)):
        window_s1 = s1[i - window + 1:i + 1]
        window_s2 = s2[i - window + 1:i + 1]
        if np.isnan(window_s1).any() or np.isnan(window_s2).any():
            pvals.append(np.nan)
        else:
            try:
                pval = ts.coint(window_s1, window_s2)[1]
            except Exception:
                pval = np.nan
            pvals.append(pval)
        idxs.append(index[i])
    return pd.Series(pvals, index=idxs)

# --- Enhancement: Hurst Exponent

def hurst_exponent(ts, min_lag=2, max_lag=20):
    """
    Estimate the Hurst exponent of a time series.
    """
    lags = range(min_lag, max_lag)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

# --- Enhancement: ADF Test Utility

def adf_pvalue(series):
    """
    Return the p-value from the Augmented Dickey-Fuller test.
    """
    try:
        return ts.adfuller(series.dropna())[1]
    except Exception:
        return np.nan 

# --- Enhancement: Rolling Hurst and Rolling ADF
# Add rolling Hurst and rolling ADF calculation utilities for dynamic filtering.
def rolling_hurst(series, window=60):
    """
    Compute rolling Hurst exponent for a time series.
    Returns a Series aligned to the right edge of each window.
    """
    def hurst_win(x):
        lags = range(2, 20)
        tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    return series.rolling(window=window, min_periods=window).apply(hurst_win, raw=True)

def rolling_adf(series, window=60):
    """
    Compute rolling ADF p-value for a time series.
    Returns a Series aligned to the right edge of each window.
    """
    import statsmodels.tsa.stattools as tsastat
    def adf_win(x):
        try:
            return tsastat.adfuller(x)[1]
        except Exception:
            return np.nan
    return series.rolling(window=window, min_periods=window).apply(adf_win, raw=True)
