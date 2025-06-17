"""
Pair selection module for analyzing and selecting trading pairs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from statsmodels.tsa.stattools import coint
from pykalman import KalmanFilter
from .pair_analysis import calculate_hurst_exponent, calculate_adf

class PairSelector:
    """Class for analyzing and selecting trading pairs."""
    
    def __init__(self, config):
        """Initialize the pair selector with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)

    def calculate_pair_metrics(self, pair_data: pd.DataFrame) -> Dict:
        """Calculate metrics for a pair of assets."""
        try:
            # Calculate correlation
            correlation = pair_data.corr().iloc[0, 1]
            
            # Calculate cointegration
            coint_pvalue = self._calculate_cointegration(pair_data)
            
            # Calculate Kalman filter parameters
            kf_params = self._calculate_kalman_params(pair_data)
            
            # Calculate spread
            spread = self._calculate_spread(pair_data, kf_params)
            
            # Calculate stability metrics
            stability = self._calculate_stability_metrics(spread)
            
            # Calculate Z-score
            zscore = self._calculate_zscore(spread)
            
            # Calculate composite score
            score = self._calculate_composite_score(
                correlation=correlation,
                coint_pvalue=coint_pvalue,
                stability=stability,
                zscore_vol=zscore.std()
            )
            
            return {
                'correlation': correlation,
                'coint_pvalue': coint_pvalue,
                'spread_stability': stability,
                'zscore_volatility': zscore.std(),
                'score': score,
                'spread': spread,
                'zscore': zscore,
                'kalman_params': kf_params
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating pair metrics: {str(e)}")
            return None

    def _calculate_cointegration(self, pair_data: pd.DataFrame) -> float:
        """Calculate cointegration p-value for a pair."""
        try:
            _, pvalue, _ = coint(pair_data.iloc[:, 0], pair_data.iloc[:, 1])
            return pvalue
        except Exception as e:
            self.logger.error(f"Error calculating cointegration: {str(e)}")
            return 1.0

    def _calculate_kalman_params(self, pair_data: pd.DataFrame) -> Dict:
        """Calculate Kalman filter parameters."""
        try:
            # Initialize Kalman filter
            kf = KalmanFilter(
                n_dim_obs=1,
                n_dim_state=2,
                initial_state_mean=np.zeros(2),
                initial_state_covariance=np.eye(2),
                transition_matrices=np.array([[1, 0], [0, 1]]),
                observation_matrices=np.array([[1, 0]]),
                transition_covariance=np.eye(2) * 0.01,
                observation_covariance=1.0
            )

            # Fit Kalman filter parameters using EM
            observations = pair_data.iloc[:, 1].values.reshape(-1, 1)
            kf = kf.em(observations)

            # Run the filter to obtain state estimates
            filtered_state_means, filtered_state_covariances = kf.filter(observations)

            return {
                'filtered_state_means': filtered_state_means,
                'filtered_state_covariances': filtered_state_covariances,
                'transition_matrix': kf.transition_matrices,
                'observation_matrix': kf.observation_matrices
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Kalman parameters: {str(e)}")
            return None

    def _calculate_spread(self, pair_data: pd.DataFrame, kf_params: Dict) -> pd.Series:
        """Calculate the spread between two assets."""
        try:
            if kf_params is None:
                return None

            # Use the beta component from the filtered state means as hedge ratio
            hedge_ratio = kf_params['filtered_state_means'][:, 1]
            spread = pair_data.iloc[:, 1] - hedge_ratio * pair_data.iloc[:, 0]
            
            return spread
            
        except Exception as e:
            self.logger.error(f"Error calculating spread: {str(e)}")
            return None

    def _calculate_stability_metrics(self, spread: pd.Series) -> float:
        """Calculate stability metrics for the spread."""
        try:
            if spread is None or len(spread) < self.config.pair_selection.stability_lookback:
                return 0.0
                
            # Calculate rolling metrics
            hurst = calculate_hurst_exponent(spread)
            adf_pvalue = calculate_adf(spread)
            
            # Calculate stability score
            stability = (1 - hurst) * (1 - adf_pvalue)
            
            return stability
            
        except Exception as e:
            self.logger.error(f"Error calculating stability metrics: {str(e)}")
            return 0.0

    def _calculate_zscore(self, spread: pd.Series) -> pd.Series:
        """Calculate Z-score of the spread."""
        try:
            if spread is None:
                return None
                
            return (spread - spread.mean()) / spread.std()
            
        except Exception as e:
            self.logger.error(f"Error calculating Z-score: {str(e)}")
            return None

    def _calculate_composite_score(self, correlation: float, coint_pvalue: float,
                                 stability: float, zscore_vol: float) -> float:
        """Calculate composite score for pair selection."""
        try:
            # Normalize metrics
            norm_correlation = max(0, correlation)
            norm_coint = 1 - coint_pvalue
            norm_stability = max(0, stability)
            norm_vol = 1 - min(1, zscore_vol / self.config.pair_selection.max_zscore_volatility)
            
            # Calculate weighted score
            score = (
                0.3 * norm_correlation +
                0.3 * norm_coint +
                0.2 * norm_stability +
                0.2 * norm_vol
            )
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating composite score: {str(e)}")
            return 0.0

    def select_pairs(self, pair_metrics: Dict[str, Dict]) -> List[Tuple[str, str]]:
        """Select pairs based on metrics and configuration."""
        try:
            selected_pairs = []
            
            for pair, metrics in pair_metrics.items():
                if metrics is None:
                    continue
                    
                # Check if pair meets criteria
                if (metrics['correlation'] >= self.config.pair_selection.min_correlation and
                    metrics['spread_stability'] >= self.config.pair_selection.min_spread_stability and
                    metrics['zscore_volatility'] <= self.config.pair_selection.max_zscore_volatility):
                    
                    selected_pairs.append((pair, metrics['score']))
            
            # Sort by score and select top pairs
            selected_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # If no pairs meet criteria, select top 3 by score
            if not selected_pairs:
                self.logger.warning("No pairs met filtering criteria. Selecting top 3 by score.")
                selected_pairs = sorted(
                    [(p, m['score']) for p, m in pair_metrics.items() if m is not None],
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
            
            return [p[0] for p in selected_pairs]
            
        except Exception as e:
            self.logger.error(f"Error selecting pairs: {str(e)}")
            return [] 