"""
Enhanced pair selection module with statistical rigor using statsmodels.
Provides cointegration testing, statistical significance, and robust pair filtering.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import statsmodels.api as sm

@dataclass
class StatisticalThresholds:
    """Statistical thresholds for pair selection."""
    adf_pvalue_max: float = 0.05  # Maximum p-value for ADF test
    coint_pvalue_max: float = 0.05  # Maximum p-value for cointegration test
    r_squared_min: float = 0.7  # Minimum R-squared for linear relationship
    correlation_min: float = 0.8  # Minimum correlation coefficient
    hurst_threshold: float = 0.5  # Maximum Hurst exponent for mean reversion
    min_observations: int = 252  # Minimum number of observations

class EnhancedPairSelector:
    """Enhanced pair selector with statistical rigor."""
    
    def __init__(self, config, statistical_thresholds: Optional[StatisticalThresholds] = None):
        """Initialize the enhanced pair selector."""
        self.config = config
        self.statistical_thresholds = statistical_thresholds or StatisticalThresholds()
        self.logger = logging.getLogger(__name__)
        
    def calculate_hurst_exponent(self, series: pd.Series) -> float:
        """Calculate Hurst exponent to test for mean reversion."""
        try:
            # Calculate price changes
            price_changes = series.pct_change().dropna()
            
            # Calculate cumulative sum
            cumsum = price_changes.cumsum()
            
            # Calculate range and standard deviation for different lags
            lags = range(10, min(len(price_changes) // 4, 100))
            tau = []
            
            for lag in lags:
                # Split series into chunks
                chunks = len(price_changes) // lag
                if chunks < 2:
                    continue
                    
                rs_values = []
                for i in range(chunks):
                    chunk = price_changes[i * lag:(i + 1) * lag]
                    if len(chunk) < lag:
                        continue
                        
                    # Calculate R/S statistic
                    mean_chunk = chunk.mean()
                    dev = chunk - mean_chunk
                    cumdev = dev.cumsum()
                    
                    r = cumdev.max() - cumdev.min()  # Range
                    s = chunk.std()  # Standard deviation
                    
                    if s > 0:
                        rs_values.append(r / s)
                
                if rs_values:
                    tau.append(np.mean(rs_values))
            
            if len(tau) < 2:
                return 0.5
            
            # Calculate Hurst exponent
            reg = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
            hurst = reg[0]
            
            return hurst
            
        except Exception as e:
            self.logger.error(f"Error calculating Hurst exponent: {e}")
            return 0.5
    
    def test_cointegration(self, price1: pd.Series, price2: pd.Series) -> Dict:
        """Test for cointegration using Engle-Granger test."""
        try:
            # Remove any NaN values
            price1_clean = price1.dropna()
            price2_clean = price2.dropna()
            
            # Align series
            common_index = price1_clean.index.intersection(price2_clean.index)
            if len(common_index) < self.statistical_thresholds.min_observations:
                return {'is_cointegrated': False, 'pvalue': 1.0, 'statistic': 0}
            
            price1_aligned = price1_clean.loc[common_index]
            price2_aligned = price2_clean.loc[common_index]
            
            # Perform Engle-Granger cointegration test
            coint_result = coint(price1_aligned, price2_aligned)
            coint_stat, pvalue, critical_values = coint_result
            
            # Calculate spread and test for stationarity
            spread = price1_aligned - price2_aligned
            adf_result = adfuller(spread, regression='ct', autolag='AIC')
            adf_stat, adf_pvalue, adf_critical_values = adf_result[0], adf_result[1], adf_result[4]
            
            # Linear regression for R²
            X = sm.add_constant(price2_aligned)
            model = OLS(price1_aligned, X).fit()
            
            # Test for heteroscedasticity
            bp_result = het_breuschpagan(model.resid, model.model.exog)
            bp_stat, bp_pvalue = bp_result[0], bp_result[1]
            
            # Calculate correlation
            correlation = price1_aligned.corr(price2_aligned)
            
            # Calculate Hurst exponent of spread
            hurst = self.calculate_hurst_exponent(spread)
            
            return {
                'is_cointegrated': pvalue < self.statistical_thresholds.coint_pvalue_max,
                'coint_statistic': coint_stat,
                'coint_pvalue': pvalue,
                'coint_critical_values': critical_values,
                'adf_statistic': adf_stat,
                'adf_pvalue': adf_pvalue,
                'adf_critical_values': adf_critical_values,
                'r_squared': model.rsquared,
                'r_squared_adj': model.rsquared_adj,
                'correlation': correlation,
                'hurst_exponent': hurst,
                'heteroscedasticity_pvalue': bp_pvalue,
                'spread_mean': spread.mean(),
                'spread_std': spread.std(),
                'observations': len(common_index)
            }
            
        except Exception as e:
            self.logger.error(f"Error testing cointegration: {e}")
            return {'is_cointegrated': False, 'coint_pvalue': 1.0, 'coint_statistic': 0}
    
    def calculate_pair_metrics_enhanced(self, pair_data: pd.DataFrame) -> Optional[Dict]:
        """Calculate enhanced pair metrics with statistical rigor."""
        try:
            if pair_data is None or pair_data.empty:
                return None
            
            # Extract price series
            asset1_col = pair_data.columns[0]
            asset2_col = pair_data.columns[1]
            
            price1 = pair_data[asset1_col]
            price2 = pair_data[asset2_col]
            
            # Test cointegration
            coint_results = self.test_cointegration(price1, price2)
            
            # Calculate basic metrics
            returns1 = price1.pct_change().dropna()
            returns2 = price2.pct_change().dropna()
            
            # Align returns
            common_index = returns1.index.intersection(returns2.index)
            if len(common_index) < 50:
                return None
                
            returns1_aligned = returns1.loc[common_index]
            returns2_aligned = returns2.loc[common_index]
            
            # Calculate spread metrics
            spread = price1 - price2
            spread_returns = spread.pct_change().dropna()
            
            # Rolling statistics
            lookback = 20
            rolling_mean = spread.rolling(lookback).mean()
            rolling_std = spread.rolling(lookback).std()
            z_score = (spread - rolling_mean) / rolling_std
            
            # Volatility metrics
            vol1 = returns1_aligned.std() * np.sqrt(252)
            vol2 = returns2_aligned.std() * np.sqrt(252)
            vol_spread = spread_returns.std() * np.sqrt(252)
            
            # Correlation metrics
            correlation = returns1_aligned.corr(returns2_aligned)
            correlation_rolling = returns1_aligned.rolling(60).corr(returns2_aligned)
            
            # Statistical significance tests
            t_stat, p_value = stats.pearsonr(returns1_aligned, returns2_aligned)
            
            # Check if pair meets statistical criteria
            meets_criteria = (
                coint_results['is_cointegrated'] and
                coint_results['r_squared'] >= self.statistical_thresholds.r_squared_min and
                abs(coint_results['correlation']) >= self.statistical_thresholds.correlation_min and
                coint_results['hurst_exponent'] <= self.statistical_thresholds.hurst_threshold and
                coint_results['observations'] >= self.statistical_thresholds.min_observations
            )
            
            return {
                'asset1': asset1_col,
                'asset2': asset2_col,
                'is_cointegrated': coint_results['is_cointegrated'],
                'coint_pvalue': coint_results['coint_pvalue'],
                'adf_pvalue': coint_results['adf_pvalue'],
                'r_squared': coint_results['r_squared'],
                'correlation': coint_results['correlation'],
                'hurst_exponent': coint_results['hurst_exponent'],
                'heteroscedasticity_pvalue': coint_results['heteroscedasticity_pvalue'],
                'volatility_asset1': vol1,
                'volatility_asset2': vol2,
                'volatility_spread': vol_spread,
                'correlation_t_stat': t_stat,
                'correlation_p_value': p_value,
                'spread_mean': coint_results['spread_mean'],
                'spread_std': coint_results['spread_std'],
                'z_score_mean': z_score.mean(),
                'z_score_std': z_score.std(),
                'observations': coint_results['observations'],
                'meets_criteria': meets_criteria,
                'statistical_score': self._calculate_statistical_score(coint_results)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced pair metrics: {e}")
            return None
    
    def _calculate_statistical_score(self, coint_results: Dict) -> float:
        """Calculate a composite statistical score for pair ranking."""
        try:
            score = 0.0
            
            # Cointegration score (40% weight)
            if coint_results['is_cointegrated']:
                score += 0.4 * (1 - coint_results['coint_pvalue'])
            
            # R-squared score (25% weight)
            score += 0.25 * coint_results['r_squared']
            
            # Correlation score (20% weight)
            score += 0.20 * abs(coint_results['correlation'])
            
            # Mean reversion score (15% weight)
            hurst_score = max(0, 1 - coint_results['hurst_exponent'])
            score += 0.15 * hurst_score
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating statistical score: {e}")
            return 0.0
    
    def select_pairs_enhanced(self, pair_metrics: Dict) -> List[Tuple[str, str]]:
        """Select pairs based on enhanced statistical criteria."""
        try:
            # Filter pairs that meet criteria
            qualified_pairs = []
            
            for pair, metrics in pair_metrics.items():
                if metrics and metrics.get('meets_criteria', False):
                    qualified_pairs.append((pair, metrics))
            
            # Sort by statistical score
            qualified_pairs.sort(key=lambda x: x[1]['statistical_score'], reverse=True)
            
            # Select top pairs
            max_pairs = getattr(self.config, 'max_pairs', 10)
            selected_pairs = [pair for pair, _ in qualified_pairs[:max_pairs]]
            
            # Log selection results
            self.logger.info(f"Enhanced pair selection results:")
            self.logger.info(f"Total pairs analyzed: {len(pair_metrics)}")
            self.logger.info(f"Pairs meeting criteria: {len(qualified_pairs)}")
            self.logger.info(f"Pairs selected: {len(selected_pairs)}")
            
            # Log detailed statistics for selected pairs
            for i, (pair, metrics) in enumerate(qualified_pairs[:len(selected_pairs)]):
                self.logger.info(f"Pair {i+1}: {pair[0]}-{pair[1]}")
                self.logger.info(f"  Statistical Score: {metrics['statistical_score']:.3f}")
                self.logger.info(f"  Cointegration p-value: {metrics['coint_pvalue']:.4f}")
                self.logger.info(f"  R-squared: {metrics['r_squared']:.3f}")
                self.logger.info(f"  Correlation: {metrics['correlation']:.3f}")
                self.logger.info(f"  Hurst Exponent: {metrics['hurst_exponent']:.3f}")
            
            return selected_pairs
            
        except Exception as e:
            self.logger.error(f"Error in enhanced pair selection: {e}")
            return []
    
    def generate_statistical_report(self, pair_metrics: Dict) -> str:
        """Generate comprehensive statistical report."""
        try:
            report = f"""
{'='*80}
ENHANCED STATISTICAL PAIR ANALYSIS REPORT
{'='*80}

SUMMARY STATISTICS:
{'-'*40}
Total Pairs Analyzed: {len(pair_metrics)}
Pairs Meeting Criteria: {sum(1 for m in pair_metrics.values() if m and m.get('meets_criteria', False))}
Pairs Not Meeting Criteria: {sum(1 for m in pair_metrics.values() if m and not m.get('meets_criteria', False))}

DETAILED PAIR ANALYSIS:
{'-'*40}
"""
            
            # Sort pairs by statistical score
            sorted_pairs = sorted(
                [(pair, metrics) for pair, metrics in pair_metrics.items() if metrics],
                key=lambda x: x[1]['statistical_score'],
                reverse=True
            )
            
            for i, (pair, metrics) in enumerate(sorted_pairs[:20]):  # Top 20 pairs
                report += f"""
Pair {i+1}: {pair[0]}-{pair[1]}
  Statistical Score: {metrics['statistical_score']:.3f}
  Meets Criteria: {'Yes' if metrics['meets_criteria'] else 'No'}
  Cointegration Test:
    p-value: {metrics['coint_pvalue']:.4f}
    ADF p-value: {metrics['adf_pvalue']:.4f}
    R-squared: {metrics['r_squared']:.3f}
  Correlation Analysis:
    Correlation: {metrics['correlation']:.3f}
    t-statistic: {metrics['correlation_t_stat']:.3f}
    p-value: {metrics['correlation_p_value']:.4f}
  Mean Reversion:
    Hurst Exponent: {metrics['hurst_exponent']:.3f}
  Volatility:
    Asset 1: {metrics['volatility_asset1']:.2%}
    Asset 2: {metrics['volatility_asset2']:.2%}
    Spread: {metrics['volatility_spread']:.2%}
  Observations: {metrics['observations']}
"""
            
            report += f"\n{'='*80}\n"
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating statistical report: {e}")
            return f"Error generating report: {e}" 