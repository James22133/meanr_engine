"""
Pair selection module for analyzing and selecting trading pairs.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from multiprocessing import Pool
from functools import partial
from statsmodels.tsa.stattools import coint
from pykalman import KalmanFilter
from .pair_analysis import calculate_hurst_exponent, calculate_adf
from .cache import save_to_cache, load_from_cache

class PairSelector:
    """Class for analyzing and selecting trading pairs."""
    
    def __init__(self, config):
        """Initialize the pair selector with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)

    def calculate_pair_metrics(self, pair_data: pd.DataFrame) -> Dict:
        """Calculate metrics for a pair of assets."""
        try:
            pair_key = f"{pair_data.columns[0]}_{pair_data.columns[1]}"
            # Calculate correlation
            correlation = pair_data.corr().iloc[0, 1]
            
            # Calculate cointegration
            coint_pvalue = self._calculate_cointegration(pair_data)
            
            # Calculate Kalman filter parameters
            kf_params = self._calculate_kalman_params(pair_data, pair_key)
            
            # Calculate spread
            spread = self._calculate_spread(pair_data, kf_params)
            
            # Calculate stability metrics
            stability = self._calculate_stability_metrics(spread)
            
            # Calculate Z-score
            zscore = self._calculate_zscore(spread, pair_key)
            
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

    def _calculate_kalman_params(self, pair_data: pd.DataFrame, pair_key: str = "") -> Dict:
        """Calculate Kalman filter parameters."""
        try:
            cache_file = os.path.join(getattr(self.config, 'data_dir', 'data'), f"kalman_{pair_key}.pkl")
            if getattr(self.config, 'use_cache', False) and os.path.exists(cache_file) and not getattr(self.config, 'force_refresh', False):
                return load_from_cache(cache_file)

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

            result = {
                'filtered_state_means': filtered_state_means,
                'filtered_state_covariances': filtered_state_covariances,
                'transition_matrix': kf.transition_matrices,
                'observation_matrix': kf.observation_matrices
            }

            if getattr(self.config, 'use_cache', False):
                save_to_cache(result, cache_file)

            return result
            
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
            if spread is None or len(spread) < getattr(self.config.pair_selection, 'stability_lookback', 90):
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

    def _calculate_zscore(self, spread: pd.Series, pair_key: str = "") -> pd.Series:
        """Calculate Z-score of the spread."""
        try:
            if spread is None:
                return None

            cache_file = os.path.join(getattr(self.config, 'data_dir', 'data'), f"zscore_{pair_key}.pkl")
            if getattr(self.config, 'use_cache', False) and os.path.exists(cache_file) and not getattr(self.config, 'force_refresh', False):
                return load_from_cache(cache_file)

            z = (spread - spread.mean()) / spread.std()

            if getattr(self.config, 'use_cache', False):
                save_to_cache(z, cache_file)

            return z
            
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
            max_zscore_volatility = getattr(self.config.pair_selection, 'max_zscore_volatility', 1.0)
            norm_vol = 1 - min(1, zscore_vol / max_zscore_volatility)
            weights = getattr(self.config, 'pair_scoring', None)
            if weights and hasattr(weights, 'weights'):
                weights = weights.weights
            else:
                weights = {
                    'correlation': 0.25,
                    'coint_p': 0.25,
                    'hurst': 0.25,
                    'zscore_vol': 0.25,
                }
            corr_w = weights['correlation']
            coint_w = weights['coint_p']
            hurst_w = weights['hurst']
            vol_w = weights['zscore_vol']

            score = (
                corr_w * norm_correlation +
                coint_w * norm_coint +
                hurst_w * norm_stability +
                vol_w * norm_vol
            )
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating composite score: {str(e)}")
            return 0.0

    def validate_pair_stability(
        self,
        pair_data: pd.DataFrame,
        train_window: int = 90,
        test_window: int = 30,
    ) -> Tuple[bool, List[Dict[str, float]]]:
        """Perform rolling out-of-sample validation and flag instability."""
        metrics = []
        pair_key = f"{pair_data.columns[0]}_{pair_data.columns[1]}"
        for start in range(0, len(pair_data) - train_window - test_window + 1, test_window):
            train = pair_data.iloc[start : start + train_window]
            test = pair_data.iloc[start + train_window : start + train_window + test_window]
            kf_train = self._calculate_kalman_params(train, pair_key + f"_tr{start}")
            spread_test = self._calculate_spread(test, kf_train)
            if spread_test is None or spread_test.std() == 0:
                continue
            returns = spread_test.diff().dropna()
            sharpe = (np.sqrt(252) * returns.mean() / returns.std()) if returns.std() != 0 else 0
            cum = returns.cumsum()
            dd = (cum - cum.cummax()).min()
            metrics.append({'sharpe': sharpe, 'max_drawdown': dd})

        unstable = any(m['sharpe'] < 0 or m['max_drawdown'] < -0.1 for m in metrics)
        return unstable, metrics

    def _score_single(self, pair: Tuple[str, str], data_loader) -> Tuple[Tuple[str, str], Optional[Dict]]:
        """Helper for parallel scoring."""
        try:
            pair_data = data_loader.get_pair_data(list(pair))
            if pair_data is None:
                return pair, None
            metrics = self.calculate_pair_metrics(pair_data)
            return pair, metrics
        except Exception as e:
            self.logger.error(f"Error scoring pair {pair}: {e}")
            return pair, None

    def score_pairs_parallel(self, pairs: List[Tuple[str, str]], data_loader) -> Dict[Tuple[str, str], Dict]:
        """Score multiple pairs in parallel."""
        pair_metrics: Dict[Tuple[str, str], Dict] = {}
        with Pool() as pool:
            func = partial(self._score_single, data_loader=data_loader)
            results = pool.map(func, pairs)
        for pair, metrics in results:
            if metrics is not None:
                pair_metrics[pair] = metrics
        return pair_metrics

    def select_pairs(self, pair_metrics: Dict) -> List[Tuple[str, str]]:
        """Select pairs based on calculated metrics and configuration criteria."""
        if not pair_metrics:
            self.logger.warning("No pairs met filtering criteria. Selecting top 3 by score.")
            return []
        
        # Sort pairs by score
        sorted_pairs = sorted(pair_metrics.items(), key=lambda x: x[1]['score'], reverse=True)
        
        # Select top pairs
        selected_pairs = [pair for pair, _ in sorted_pairs[:3]]
        
        self.logger.info(f"Selected {len(selected_pairs)} pairs: {selected_pairs}")
        return selected_pairs

    def generate_signals(self, pair_data: pd.DataFrame, pair_name: Tuple[str, str]) -> pd.DataFrame:
        """Generate trading signals for a pair."""
        try:
            if pair_data is None or pair_data.empty or len(pair_data.columns) < 2:
                return pd.DataFrame()
            
            asset1_col, asset2_col = pair_data.columns[0], pair_data.columns[1]
            price1, price2 = pair_data[asset1_col], pair_data[asset2_col]
            
            # Calculate spread and z-score
            spread = price1 - price2
            lookback = 20
            z_score = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()
            
            # Get thresholds from config (using dict access instead of attribute access)
            entry_threshold = self.config.get('backtest', {}).get('zscore_entry_threshold', 1.5)
            exit_threshold = self.config.get('backtest', {}).get('zscore_exit_threshold', 0.5)
            
            # Generate signals
            long_signals = z_score < -entry_threshold
            short_signals = z_score > entry_threshold
            exit_signals = abs(z_score) < exit_threshold
            
            # Create signals DataFrame
            signals = pd.DataFrame({
                'long': long_signals,
                'short': short_signals,
                'exit': exit_signals,
                'z_score': z_score
            }, index=pair_data.index)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return pd.DataFrame()

    def _log_signal_stats(
        self,
        pair_data: pd.DataFrame,
        z_score: pd.Series,
        entry_threshold: float,
        exit_threshold: float,
        signals: pd.DataFrame,
    ) -> None:
        """Log detailed signal generation statistics."""
        try:
            long_signals = signals["entry_long"].sum()
            short_signals = signals["entry_short"].sum()
            exit_signals = signals["exit"].sum()
            total_signals = long_signals + short_signals

            self.logger.info("=== SIGNAL GENERATION DEBUG ===")
            self.logger.info(
                f"Pair: {pair_data.columns[0]}-{pair_data.columns[1]}"
            )
            self.logger.info(f"Data points: {len(pair_data)}")
            self.logger.info(
                f"Z-score range: {z_score.min():.3f} to {z_score.max():.3f}"
            )
            self.logger.info(
                f"Z-score mean: {z_score.mean():.3f}, std: {z_score.std():.3f}"
            )
            self.logger.info(
                f"Thresholds - Entry: {entry_threshold}, Exit: {exit_threshold}"
            )
            self.logger.info(
                f"Signals generated - Long: {long_signals}, Short: {short_signals}, Exit: {exit_signals}"
            )
            self.logger.info(f"Total entry signals: {total_signals}")

            recent_zscore = z_score.tail(10)
            self.logger.info(f"Recent z-scores: {recent_zscore.values}")

            extreme_positive = (z_score > entry_threshold).sum()
            extreme_negative = (z_score < -entry_threshold).sum()
            self.logger.info(
                f"Extreme z-scores > {entry_threshold}: {extreme_positive}"
            )
            self.logger.info(
                f"Extreme z-scores < -{entry_threshold}: {extreme_negative}"
            )

            if total_signals == 0:
                self.logger.warning(
                    "No entry signals generated! Consider lowering entry threshold."
                )
                near_threshold = z_score[abs(z_score) >= entry_threshold * 0.8]
                if not near_threshold.empty:
                    self.logger.info(
                        f"Z-scores near threshold: {near_threshold.head(5).values}"
                    )
        except Exception as e:
            self.logger.error(f"Error logging signal stats: {e}")
