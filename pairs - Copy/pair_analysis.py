import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts
from pykalman import KalmanFilter

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
        print(f"Error calculating cointegration: {e}")
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

def calculate_spread_and_zscore(y, X, states):
    """
    Calculate the spread (y - beta*X) and its rolling Z-score.

    Args:
        y (pd.Series): The first time series.
        X (pd.Series): The second time series.
        states (np.ndarray): The estimated states (alpha, beta) from Kalman Filter.

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
    rolling_window = 20
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
