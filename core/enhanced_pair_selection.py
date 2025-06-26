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
from collections import defaultdict

@dataclass
class StatisticalThresholds:
    """Statistical thresholds for pair selection."""
    adf_pvalue_max: float = 0.05  # Maximum p-value for ADF test
    coint_pvalue_max: float = 0.05  # Maximum p-value for cointegration test
    r_squared_min: float = 0.7  # Minimum R-squared for linear relationship
    correlation_min: float = 0.8  # Minimum correlation coefficient
    hurst_threshold: float = 0.5  # Maximum Hurst exponent for mean reversion
    min_observations: int = 252  # Minimum number of observations
    max_volatility_spread: float = 2.0  # Maximum volatility spread ratio (new)
    max_sector_pairs: int = 3  # Maximum pairs per sector (new)
    min_sector_diversification: int = 3  # Minimum number of sectors (new)

@dataclass
class SectorMapping:
    """Sector mapping for diversification."""
    energy: List[str] = None
    tech: List[str] = None
    financials: List[str] = None
    industrials: List[str] = None
    consumer: List[str] = None
    materials: List[str] = None
    utilities: List[str] = None
    healthcare: List[str] = None
    
    def __post_init__(self):
        if self.energy is None:
            self.energy = ['XOM', 'CVX', 'SLB', 'HAL', 'MPC', 'VLO', 'COP', 'EOG', 'DVN', 'ET', 'EPD', 'OXY', 'XLE', 'UNG', 'BNO', 'USO']
        if self.tech is None:
            self.tech = ['QQQ', 'XLK', 'SMH']
        if self.financials is None:
            self.financials = ['XLF', 'KBE', 'KRE']
        if self.industrials is None:
            self.industrials = ['XLI', 'VIS']
        if self.consumer is None:
            self.consumer = ['XLP', 'VDC', 'XLY']
        if self.materials is None:
            self.materials = ['XLB', 'GLD', 'SLV', 'PPLT']
        if self.utilities is None:
            self.utilities = ['XLU']
        if self.healthcare is None:
            self.healthcare = ['XLV', 'IHI']

class EnhancedPairSelector:
    """Enhanced pair selector with statistical rigor and risk management."""
    
    def __init__(self, config, statistical_thresholds: Optional[StatisticalThresholds] = None):
        """Initialize the enhanced pair selector."""
        self.config = config
        self.statistical_thresholds = statistical_thresholds or StatisticalThresholds()
        self.sector_mapping = SectorMapping()
        self.logger = logging.getLogger(__name__)
        
        # Risk management settings
        self.max_volatility_spread = getattr(self.statistical_thresholds, 'max_volatility_spread', 2.0)
        self.max_sector_pairs = getattr(self.statistical_thresholds, 'max_sector_pairs', 3)
        self.min_sector_diversification = getattr(self.statistical_thresholds, 'min_sector_diversification', 3)
        
    def get_asset_sector(self, asset: str) -> str:
        """Get the sector for a given asset."""
        for sector, assets in self.sector_mapping.__dict__.items():
            if assets and asset in assets:
                return sector
        return 'other'
    
    def calculate_sector_diversification(self, pairs: List[Tuple[str, str]]) -> Dict[str, int]:
        """Calculate sector distribution for selected pairs."""
        sector_counts = defaultdict(int)
        for pair in pairs:
            sector1 = self.get_asset_sector(pair[0])
            sector2 = self.get_asset_sector(pair[1])
            sector_counts[sector1] += 1
            sector_counts[sector2] += 1
        return dict(sector_counts)
    
    def check_volatility_spread_risk(self, vol1: float, vol2: float, vol_spread: float) -> bool:
        """Check if volatility spread is within acceptable limits."""
        # Calculate volatility spread ratio
        if vol2 > 0:
            vol_ratio = vol1 / vol2
        else:
            vol_ratio = float('inf')
        
        # Check if spread volatility is excessive
        spread_risk = vol_spread > (max(vol1, vol2) * self.max_volatility_spread)
        
        # Check if individual asset volatility ratio is excessive
        ratio_risk = vol_ratio > self.max_volatility_spread or vol_ratio < (1 / self.max_volatility_spread)
        
        return not (spread_risk or ratio_risk)
    
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
                coint_results['observations'] >= self.statistical_thresholds.min_observations and
                self.check_volatility_spread_risk(vol1, vol2, vol_spread)
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
        """Select pairs based on enhanced statistical criteria with risk management."""
        try:
            # Filter pairs that meet criteria
            qualified_pairs = []
            
            for pair, metrics in pair_metrics.items():
                if metrics and metrics.get('meets_criteria', False):
                    qualified_pairs.append((pair, metrics))
            
            # Sort by statistical score
            qualified_pairs.sort(key=lambda x: x[1]['statistical_score'], reverse=True)
            
            # Apply sector diversification
            selected_pairs = self._apply_sector_diversification(qualified_pairs)
            
            # Log selection results
            self.logger.info(f"Enhanced pair selection results:")
            self.logger.info(f"Total pairs analyzed: {len(pair_metrics)}")
            self.logger.info(f"Pairs meeting criteria: {len(qualified_pairs)}")
            self.logger.info(f"Pairs selected after diversification: {len(selected_pairs)}")
            
            # Log sector distribution
            sector_dist = self.calculate_sector_diversification(selected_pairs)
            self.logger.info(f"Sector distribution: {sector_dist}")
            
            # Log detailed statistics for selected pairs
            for i, (pair, metrics) in enumerate(qualified_pairs[:len(selected_pairs)]):
                self.logger.info(f"Pair {i+1}: {pair[0]}-{pair[1]}")
                self.logger.info(f"  Statistical Score: {metrics['statistical_score']:.3f}")
                self.logger.info(f"  Cointegration p-value: {metrics['coint_pvalue']:.4f}")
                self.logger.info(f"  R-squared: {metrics['r_squared']:.3f}")
                self.logger.info(f"  Correlation: {metrics['correlation']:.3f}")
                self.logger.info(f"  Hurst Exponent: {metrics['hurst_exponent']:.3f}")
                self.logger.info(f"  Volatility Spread: {metrics['volatility_spread']:.2%}")
                self.logger.info(f"  Sectors: {self.get_asset_sector(pair[0])}-{self.get_asset_sector(pair[1])}")
            
            return selected_pairs
            
        except Exception as e:
            self.logger.error(f"Error in enhanced pair selection: {e}")
            return []
    
    def _apply_sector_diversification(self, qualified_pairs: List[Tuple[Tuple[str, str], Dict]]) -> List[Tuple[str, str]]:
        """Apply sector diversification to selected pairs."""
        selected_pairs = []
        sector_counts = defaultdict(int)
        
        for pair, metrics in qualified_pairs:
            sector1 = self.get_asset_sector(pair[0])
            sector2 = self.get_asset_sector(pair[1])
            
            # Check if adding this pair would exceed sector limits
            if (sector_counts[sector1] < self.max_sector_pairs and 
                sector_counts[sector2] < self.max_sector_pairs):
                
                selected_pairs.append(pair)
                sector_counts[sector1] += 1
                sector_counts[sector2] += 1
                
                # Stop if we have enough pairs
                if len(selected_pairs) >= getattr(self.config, 'max_pairs', 10):
                    break
        
        # Ensure minimum sector diversification
        unique_sectors = set()
        for pair in selected_pairs:
            unique_sectors.add(self.get_asset_sector(pair[0]))
            unique_sectors.add(self.get_asset_sector(pair[1]))
        
        if len(unique_sectors) < self.min_sector_diversification:
            self.logger.warning(f"Only {len(unique_sectors)} sectors represented, minimum is {self.min_sector_diversification}")
        
        return selected_pairs
    
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