import numpy as np
import pandas as pd
from hmmlearn import hmm

def calculate_volatility(prices, window=20):
    """
    Calculates rolling historical volatility.

    Args:
        prices (pd.Series): Time series of prices.
        window (int): Rolling window size for volatility calculation.

    Returns:
        pd.Series: Rolling volatility.
    """
    returns = prices.pct_change().dropna()
    # Annualize volatility for daily data (assuming 252 trading days)
    volatility = returns.rolling(window=window).std() * np.sqrt(252)
    return volatility.dropna()

def train_hmm(features, n_components=2, n_iter=100):
    """
    Trains a Hidden Markov Model on the given features.

    Args:
        features (np.ndarray): Array of features for the HMM (e.g., volatility).
        n_components (int): Number of hidden states (regimes).
        n_iter (int): Number of iterations for the training algorithm.

    Returns:
        hmmlearn.hmm.GaussianHMM: Trained HMM model.
    """
    # Ensure features are in the correct shape (samples, features)
    if isinstance(features, pd.Series):
        features = features.values
    if features.ndim == 1:
        features = features.reshape(-1, 1)

    # Use GaussianHMM for continuous features
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=n_iter, random_state=42)
    model.fit(features)
    return model

def predict_regimes(model, features):
    """
    Predicts the hidden states (regimes) using the trained HMM.

    Args:
        model (hmmlearn.hmm.GaussianHMM): Trained HMM model.
        features (np.ndarray): Array of features used for prediction.

    Returns:
        np.ndarray: Array of predicted hidden states (regimes).
    """
     # Ensure features are in the correct shape (samples, features)
    if isinstance(features, pd.Series):
        features = features.values
    if features.ndim == 1:
        features = features.reshape(-1, 1)

    regimes = model.predict(features)
    return regimes

class RegimeDetector:
    """Wrapper for volatility/HMM-based regime detection."""
    def __init__(self, config=None):
        self.config = config or {}
        self.n_components = self.config.get('hmm_n_components', 3)
        self.window = self.config.get('regime_volatility_window', 20)
        self.n_iter = self.config.get('n_iter', 100)

    def detect_regimes(self, price_data: pd.DataFrame) -> dict:
        """Detect regimes for each pair or for the market as a whole."""
        regimes = {}
        # If price_data is a DataFrame of multiple assets, process each pair/column
        for col in price_data.columns:
            prices = price_data[col]
            vol = calculate_volatility(prices, window=self.window)
            if len(vol) < self.n_components:
                continue
            hmm_model = train_hmm(vol, n_components=self.n_components, n_iter=self.n_iter)
            regime_states = predict_regimes(hmm_model, vol)
            regime_series = pd.Series(regime_states, index=vol.index)
            regimes[(col,)] = regime_series
        return regimes

# Example Usage (for demonstration)
# if __name__ == "__main__":
#     # Assume 'price_data' is a pandas DataFrame with prices of multiple assets
#     # Dummy data for demonstration
#     dates = pd.date_range(start='2023-01-01', periods=100)
#     prices = pd.DataFrame({'Asset1': np.random.rand(100).cumsum() + 50, 'Asset2': np.random.rand(100).cumsum() + 52}, index=dates)

#     # Calculate a market-wide volatility proxy (e.g., mean volatility of selected assets)
#     volatilities = prices.apply(calculate_volatility, axis=0)
#     market_volatility = volatilities.mean(axis=1).dropna()

#     if not market_volatility.empty:
#         # Train HMM
#         hmm_model = train_hmm(market_volatility.values)

#         # Predict Regimes
#         regimes = predict_regimes(hmm_model, market_volatility.values)

#         # Align regimes with the original data index
#         regime_series = pd.Series(regimes, index=market_volatility.index)
#         print(regime_series.head())

#         # Plot regimes (simple scatter plot)
#         # plt.figure(figsize=(12, 6))
#         # plt.scatter(regime_series.index, market_volatility, c=regime_series, cmap='viridis')
#         # plt.title("Market Regimes based on Volatility")
#         # plt.xlabel("Date")
#         # plt.ylabel("Volatility")
#         # plt.colorbar(label='Regime')
#         # plt.show()
#     else:
#         print("Not enough data to calculate volatility and train HMM.") 
