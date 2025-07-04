"""
Enhanced metrics calculator for institutional-quality performance analysis.
Provides comprehensive risk and return metrics for strategy evaluation.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Union
import logging
from dataclasses import dataclass

@dataclass
class EnhancedMetricsConfig:
    """Configuration for enhanced metrics calculation."""
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    benchmark_returns: Optional[pd.Series] = None
    confidence_level: float = 0.95  # For VaR calculations
    periods_per_year: int = 252  # Trading days per year

class EnhancedMetricsCalculator:
    """Enhanced metrics calculator with custom implementations."""
    
    def __init__(self, config: EnhancedMetricsConfig):
        """Initialize the enhanced metrics calculator."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_comprehensive_metrics(self, returns: pd.Series, 
                                      equity_curve: Optional[pd.Series] = None) -> Dict:
        """Calculate comprehensive performance metrics."""
        try:
            if returns.empty:
                return {}
            
            # Ensure returns are clean
            returns_clean = returns.dropna()
            if returns_clean.empty:
                return {}
            
            # Basic return metrics
            total_return = self._cumulative_return(returns_clean)
            annual_return = self._annual_return(returns_clean)
            annual_volatility = self._annual_volatility(returns_clean)
            
            # Risk-adjusted return metrics
            sharpe_ratio = self._sharpe_ratio(returns_clean)
            sortino_ratio = self._sortino_ratio(returns_clean)
            calmar_ratio = self._calmar_ratio(returns_clean)
            
            # Drawdown metrics
            max_drawdown = self._max_drawdown(returns_clean)
            avg_drawdown = self._avg_drawdown(returns_clean)
            
            # Risk metrics
            var_95 = self._value_at_risk(returns_clean, 0.05)
            cvar_95 = self._conditional_value_at_risk(returns_clean, 0.05)
            var_99 = self._value_at_risk(returns_clean, 0.01)
            cvar_99 = self._conditional_value_at_risk(returns_clean, 0.01)
            
            # Distribution metrics
            tail_ratio = self._tail_ratio(returns_clean)
            stability = self._stability_of_timeseries(returns_clean)
            downside_risk = self._downside_risk(returns_clean)
            upside_risk = self._upside_risk(returns_clean)
            
            # Trade statistics
            win_rate = self._win_rate(returns_clean)
            profit_factor = self._profit_factor(returns_clean)
            avg_win = self._avg_win(returns_clean)
            avg_loss = self._avg_loss(returns_clean)
            
            # Additional metrics
            skewness = returns_clean.skew()
            kurtosis = returns_clean.kurtosis()
            information_ratio = None
            beta = None
            alpha = None
            treynor_ratio = None
            
            # Calculate benchmark-relative metrics if benchmark provided
            if self.config.benchmark_returns is not None:
                try:
                    benchmark_clean = self.config.benchmark_returns.dropna()
                    common_index = returns_clean.index.intersection(benchmark_clean.index)
                    
                    if len(common_index) > 0:
                        returns_aligned = returns_clean.loc[common_index]
                        benchmark_aligned = benchmark_clean.loc[common_index]
                        
                        information_ratio = self._information_ratio(returns_aligned, benchmark_aligned)
                        beta = self._beta(returns_aligned, benchmark_aligned)
                        alpha = self._alpha(returns_aligned, benchmark_aligned)
                        treynor_ratio = self._treynor_ratio(returns_aligned, benchmark_aligned)
                except Exception as e:
                    self.logger.warning(f"Error calculating benchmark metrics: {e}")
            
            # Calculate rolling metrics
            try:
                rolling_sharpe = self._rolling_sharpe(returns_clean)
                rolling_volatility = self._rolling_volatility(returns_clean)
            except Exception as e:
                self.logger.warning(f"Error calculating rolling metrics: {e}")
                rolling_sharpe = pd.Series(dtype=float)
                rolling_volatility = pd.Series(dtype=float)
            
            # Calculate regime-specific metrics
            try:
                regime_metrics = self._calculate_regime_metrics(returns_clean)
            except Exception as e:
                self.logger.warning(f"Error calculating regime metrics: {e}")
                regime_metrics = {}
            
            # Calculate additional statistics with proper error handling
            try:
                positive_days = (returns_clean > 0).sum()
                negative_days = (returns_clean < 0).sum()
                zero_days = (returns_clean == 0).sum()
            except Exception as e:
                self.logger.warning(f"Error calculating day statistics: {e}")
                positive_days = 0
                negative_days = 0
                zero_days = 0
            
            return {
                # Basic metrics
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                
                # Risk-adjusted returns
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'information_ratio': information_ratio,
                'treynor_ratio': treynor_ratio,
                
                # Drawdown metrics
                'max_drawdown': max_drawdown,
                'avg_drawdown': avg_drawdown,
                
                # Risk metrics
                'var_95': var_95,
                'cvar_95': cvar_95,
                'var_99': var_99,
                'cvar_99': cvar_99,
                'tail_ratio': tail_ratio,
                'stability': stability,
                'downside_risk': downside_risk,
                'upside_risk': upside_risk,
                
                # Trade statistics
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                
                # Distribution metrics
                'skewness': skewness,
                'kurtosis': kurtosis,
                
                # Benchmark metrics
                'beta': beta,
                'alpha': alpha,
                
                # Rolling metrics
                'rolling_sharpe_mean': rolling_sharpe.mean().item() if not rolling_sharpe.empty else 0,
                'rolling_sharpe_std': rolling_sharpe.std().item() if not rolling_sharpe.empty else 0,
                'rolling_volatility_mean': rolling_volatility.mean().item() if not rolling_volatility.empty else 0,
                
                # Regime metrics
                'regime_metrics': regime_metrics,
                
                # Additional statistics
                'observations': len(returns_clean),
                'positive_days': positive_days,
                'negative_days': negative_days,
                'zero_days': zero_days
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating comprehensive metrics: {e}")
            return {}
    
    def _cumulative_return(self, returns: pd.Series) -> float:
        """Calculate cumulative return."""
        clean = returns.dropna()
        if clean.empty:
            return 0.0
        return (1 + clean).prod() - 1
    
    def _annual_return(self, returns: pd.Series) -> float:
        """Calculate annualized return."""
        clean = returns.dropna()
        total_return = self._cumulative_return(clean)
        years = len(clean) / self.config.periods_per_year
        return (1 + total_return) ** (1 / years) - 1
    
    def _annual_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        clean = returns.dropna()
        if clean.empty:
            return 0.0
        return clean.std() * np.sqrt(self.config.periods_per_year)
    
    def _sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        clean = returns.dropna()
        if clean.empty or clean.std() == 0:
            return 0
        excess_returns = clean - self.config.risk_free_rate / self.config.periods_per_year
        return excess_returns.mean() / clean.std() * np.sqrt(self.config.periods_per_year)
    
    def _sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        clean = returns.dropna()
        if clean.empty:
            return 0
        excess_returns = clean - self.config.risk_free_rate / self.config.periods_per_year
        downside_returns = clean[clean < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        return excess_returns.mean() / downside_returns.std() * np.sqrt(self.config.periods_per_year)
    
    def _calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio."""
        clean = returns.dropna()
        if clean.empty:
            return 0
        annual_return = self._annual_return(clean)
        max_dd = self._max_drawdown(clean)
        if max_dd == 0:
            return 0
        return annual_return / abs(max_dd)
    
    def _max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        clean = returns.dropna()
        if clean.empty:
            return 0
        cumulative = (1 + clean).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _avg_drawdown(self, returns: pd.Series) -> float:
        """Calculate average drawdown."""
        clean = returns.dropna()
        if clean.empty:
            return 0
        cumulative = (1 + clean).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown[drawdown < 0].mean()
    
    def _value_at_risk(self, returns: pd.Series, cutoff: float) -> float:
        """Calculate Value at Risk."""
        clean = returns.dropna()
        if clean.empty:
            return 0
        return np.percentile(clean, cutoff * 100)
    
    def _conditional_value_at_risk(self, returns: pd.Series, cutoff: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        clean = returns.dropna()
        if clean.empty:
            return 0
        var = self._value_at_risk(clean, cutoff)
        return clean[clean <= var].mean()
    
    def _tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio."""
        clean = returns.dropna()
        if clean.empty:
            return 0
        left_tail = np.percentile(clean, 5)
        right_tail = np.percentile(clean, 95)
        return abs(right_tail / left_tail) if left_tail != 0 else 0
    
    def _stability_of_timeseries(self, returns: pd.Series) -> float:
        """Calculate stability of timeseries."""
        clean = returns.dropna()
        if clean.empty or clean.mean() == 0:
            return 0
        return 1 - clean.std() / clean.mean()
    
    def _downside_risk(self, returns: pd.Series) -> float:
        """Calculate downside risk."""
        clean = returns.dropna()
        if clean.empty:
            return 0
        downside_returns = clean[clean < 0]
        if len(downside_returns) == 0:
            return 0
        return downside_returns.std() * np.sqrt(self.config.periods_per_year)
    
    def _upside_risk(self, returns: pd.Series) -> float:
        """Calculate upside risk."""
        clean = returns.dropna()
        if clean.empty:
            return 0
        upside_returns = clean[clean > 0]
        if len(upside_returns) == 0:
            return 0
        return upside_returns.std() * np.sqrt(self.config.periods_per_year)
    
    def _win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate."""
        clean = returns.dropna()
        if clean.empty:
            return 0
        return (clean > 0).mean()
    
    def _profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor."""
        clean = returns.dropna()
        if clean.empty:
            return 0
        gains = clean[clean > 0].sum()
        losses = abs(clean[clean < 0].sum())
        return gains / losses if losses != 0 else 0
    
    def _avg_win(self, returns: pd.Series) -> float:
        """Calculate average win."""
        clean = returns.dropna()
        if clean.empty:
            return 0
        wins = clean[clean > 0]
        return wins.mean() if len(wins) > 0 else 0
    
    def _avg_loss(self, returns: pd.Series) -> float:
        """Calculate average loss."""
        clean = returns.dropna()
        if clean.empty:
            return 0
        losses = clean[clean < 0]
        return losses.mean() if len(losses) > 0 else 0
    
    def _information_ratio(self, returns: pd.Series, benchmark: pd.Series) -> float:
        """Calculate information ratio."""
        active_returns = returns - benchmark
        return active_returns.mean() / active_returns.std() * np.sqrt(self.config.periods_per_year)
    
    def _beta(self, returns: pd.Series, benchmark: pd.Series) -> float:
        """Calculate beta."""
        covariance = np.cov(returns, benchmark)[0, 1]
        variance = np.var(benchmark)
        return covariance / variance if variance != 0 else 0
    
    def _alpha(self, returns: pd.Series, benchmark: pd.Series) -> float:
        """Calculate alpha."""
        beta = self._beta(returns, benchmark)
        return returns.mean() - beta * benchmark.mean()
    
    def _treynor_ratio(self, returns: pd.Series, benchmark: pd.Series) -> float:
        """Calculate Treynor ratio."""
        beta = self._beta(returns, benchmark)
        if beta == 0:
            return 0
        excess_return = returns.mean() - self.config.risk_free_rate / self.config.periods_per_year
        return excess_return / beta * self.config.periods_per_year
    
    def _rolling_sharpe(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        try:
            rolling_mean = returns.rolling(window).mean()
            rolling_std = returns.rolling(window).std()
            
            # Handle division by zero and NaN values
            rolling_sharpe = (rolling_mean - self.config.risk_free_rate / self.config.periods_per_year) / rolling_std * np.sqrt(self.config.periods_per_year)
            
            # Replace infinite values with NaN
            rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan)
            
            return rolling_sharpe
            
        except Exception as e:
            self.logger.error(f"Error calculating rolling Sharpe: {e}")
            return pd.Series(dtype=float)
    
    def _rolling_volatility(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling volatility."""
        try:
            rolling_vol = returns.rolling(window).std() * np.sqrt(self.config.periods_per_year)
            
            # Replace infinite values with NaN
            rolling_vol = rolling_vol.replace([np.inf, -np.inf], np.nan)
            
            return rolling_vol
            
        except Exception as e:
            self.logger.error(f"Error calculating rolling volatility: {e}")
            return pd.Series(dtype=float)
    
    def _calculate_regime_metrics(self, returns: pd.Series) -> Dict:
        """Calculate regime-specific performance metrics."""
        try:
            # Simple regime classification based on volatility
            rolling_vol = returns.rolling(60).std()
            vol_median = rolling_vol.median()
            
            # Define regimes using proper boolean operations - convert to numpy arrays to avoid Series ambiguity
            low_vol_regime = (rolling_vol < vol_median * 0.8).values
            high_vol_regime = (rolling_vol > vol_median * 1.2).values
            normal_regime = ~(low_vol_regime | high_vol_regime)
            
            regime_metrics = {}
            
            for regime_name, regime_mask in [
                ('low_volatility', low_vol_regime),
                ('normal_volatility', normal_regime),
                ('high_volatility', high_vol_regime)
            ]:
                # Convert boolean array back to pandas boolean Series for indexing
                regime_mask_series = pd.Series(regime_mask, index=returns.index)
                regime_returns = returns[regime_mask_series]
                
                if len(regime_returns) > 30:  # Minimum observations
                    regime_metrics[regime_name] = {
                        'return': self._annual_return(regime_returns),
                        'volatility': self._annual_volatility(regime_returns),
                        'sharpe_ratio': self._sharpe_ratio(regime_returns),
                        'max_drawdown': self._max_drawdown(regime_returns),
                        'observations': len(regime_returns)
                    }
            
            return regime_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating regime metrics: {e}")
            return {}
    
    def calculate_pair_specific_metrics(self, pair_returns: pd.Series, 
                                      pair_equity: pd.Series) -> Dict:
        """Calculate pair-specific performance metrics."""
        try:
            # Basic pair metrics
            basic_metrics = self.calculate_comprehensive_metrics(pair_returns, pair_equity)
            
            # Pair-specific additional metrics
            if not pair_returns.empty:
                # Autocorrelation analysis
                try:
                    # Use pandas autocorr method if available, otherwise calculate manually
                    if hasattr(pair_returns, 'autocorr'):
                        autocorr_1 = pair_returns.autocorr(lag=1) if len(pair_returns) > 1 else None
                        autocorr_5 = pair_returns.autocorr(lag=5) if len(pair_returns) > 5 else None
                    else:
                        # Manual autocorrelation calculation
                        autocorr_1 = self._manual_autocorr(pair_returns, 1) if len(pair_returns) > 1 else None
                        autocorr_5 = self._manual_autocorr(pair_returns, 5) if len(pair_returns) > 5 else None
                    
                    # Volatility clustering
                    vol_series = pair_returns.rolling(20).std()
                    if hasattr(vol_series, 'autocorr'):
                        vol_clustering = vol_series.autocorr(lag=1) if len(vol_series) > 20 else None
                    else:
                        vol_clustering = self._manual_autocorr(vol_series, 1) if len(vol_series) > 20 else None
                except Exception as e:
                    self.logger.warning(f"Error calculating autocorrelation: {e}")
                    autocorr_1 = None
                    autocorr_5 = None
                    vol_clustering = None
                
                # Maximum consecutive wins/losses
                consecutive_wins = self._max_consecutive_wins(pair_returns)
                consecutive_losses = self._max_consecutive_losses(pair_returns)
                
                # Recovery time after drawdown
                recovery_metrics = self._calculate_recovery_metrics(pair_returns)
                
                pair_metrics = {
                    **basic_metrics,
                    'autocorr_1': autocorr_1,
                    'autocorr_5': autocorr_5,
                    'volatility_clustering': vol_clustering,
                    'max_consecutive_wins': consecutive_wins,
                    'max_consecutive_losses': consecutive_losses,
                    'recovery_metrics': recovery_metrics
                }
                
                return pair_metrics
            
            return basic_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating pair-specific metrics: {e}")
            return {}
    
    def _max_consecutive_wins(self, returns: pd.Series) -> int:
        """Calculate maximum consecutive winning days."""
        try:
            # Convert to boolean array and handle numpy arrays properly
            wins = (returns > 0).astype(int)  # Convert to 0/1 instead of boolean
            max_consecutive = 0
            current_consecutive = 0
            
            for win in wins:
                # Simple integer comparison
                if win == 1:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
            
            return max_consecutive
            
        except Exception as e:
            self.logger.error(f"Error calculating max consecutive wins: {e}")
            return 0
    
    def _max_consecutive_losses(self, returns: pd.Series) -> int:
        """Calculate maximum consecutive losing days."""
        try:
            # Convert to boolean array and handle numpy arrays properly
            losses = (returns < 0).astype(int)  # Convert to 0/1 instead of boolean
            max_consecutive = 0
            current_consecutive = 0
            
            for loss in losses:
                # Simple integer comparison
                if loss == 1:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
            
            return max_consecutive
            
        except Exception as e:
            self.logger.error(f"Error calculating max consecutive losses: {e}")
            return 0
    
    def _calculate_recovery_metrics(self, returns: pd.Series) -> Dict:
        """Calculate recovery time metrics after drawdowns."""
        try:
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            
            # Find drawdown periods - convert to integer array to avoid boolean issues
            in_drawdown = (drawdown < 0).astype(int)  # Convert to 0/1 instead of boolean
            drawdown_periods = []
            current_period_start = None
            
            for i, is_dd in enumerate(in_drawdown):
                # Simple integer comparison
                if is_dd == 1 and current_period_start is None:
                    current_period_start = i
                elif is_dd == 0 and current_period_start is not None:
                    drawdown_periods.append((current_period_start, i))
                    current_period_start = None
            
            if current_period_start is not None:
                drawdown_periods.append((current_period_start, len(in_drawdown) - 1))
            
            # Calculate recovery metrics
            recovery_times = []
            for start, end in drawdown_periods:
                if end < len(returns) - 1:
                    # Find when we recover to previous peak
                    peak_value = cumulative_returns.iloc[start]
                    recovery_idx = None
                    
                    for i in range(end + 1, len(cumulative_returns)):
                        if cumulative_returns.iloc[i] >= peak_value:
                            recovery_idx = i
                            break
                    
                    if recovery_idx is not None:
                        recovery_times.append(recovery_idx - end)
            
            return {
                'avg_recovery_time': np.mean(recovery_times) if recovery_times else None,
                'max_recovery_time': max(recovery_times) if recovery_times else None,
                'num_drawdown_periods': len(drawdown_periods)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating recovery metrics: {e}")
            return {}
    
    def generate_enhanced_report(self, metrics: Dict, pair_name: str = "Unknown") -> str:
        """Generate comprehensive enhanced metrics report."""
        try:
            if not metrics:
                return "No metrics to report"
            
            # Helper function to safely format values
            def safe_format(value, format_str=".2%", default="N/A"):
                """Safely format a value, handling None and NaN."""
                if value is None or pd.isna(value):
                    return default
                try:
                    if format_str == ".2%":
                        return f"{value:.2%}"
                    elif format_str == ".3f":
                        return f"{value:.3f}"
                    elif format_str == ".0f":
                        return f"{value:.0f}"
                    else:
                        return str(value)
                except (ValueError, TypeError):
                    return default
            
            report = f"""
{'='*80}
ENHANCED PERFORMANCE METRICS REPORT: {pair_name}
{'='*80}

BASIC PERFORMANCE METRICS:
{'-'*40}
Total Return: {safe_format(metrics.get('total_return'), '.2%')}
Annual Return: {safe_format(metrics.get('annual_return'), '.2%')}
Annual Volatility: {safe_format(metrics.get('annual_volatility'), '.2%')}
Observations: {metrics.get('observations', 0)}

RISK-ADJUSTED RETURNS:
{'-'*40}
Sharpe Ratio: {safe_format(metrics.get('sharpe_ratio'), '.3f')}
Sortino Ratio: {safe_format(metrics.get('sortino_ratio'), '.3f')}
Calmar Ratio: {safe_format(metrics.get('calmar_ratio'), '.3f')}
Information Ratio: {safe_format(metrics.get('information_ratio'), '.3f')}
Treynor Ratio: {safe_format(metrics.get('treynor_ratio'), '.3f')}

DRAWDOWN ANALYSIS:
{'-'*40}
Maximum Drawdown: {safe_format(metrics.get('max_drawdown'), '.2%')}
Average Drawdown: {safe_format(metrics.get('avg_drawdown'), '.2%')}
Number of Drawdown Periods: {metrics.get('recovery_metrics', {}).get('num_drawdown_periods', 0)}
Average Recovery Time: {safe_format(metrics.get('recovery_metrics', {}).get('avg_recovery_time'), '.0f')} days

RISK METRICS:
{'-'*40}
Value at Risk (95%): {safe_format(metrics.get('var_95'), '.2%')}
Conditional VaR (95%): {safe_format(metrics.get('cvar_95'), '.2%')}
Value at Risk (99%): {safe_format(metrics.get('var_99'), '.2%')}
Conditional VaR (99%): {safe_format(metrics.get('cvar_99'), '.2%')}
Tail Ratio: {safe_format(metrics.get('tail_ratio'), '.3f')}
Downside Risk: {safe_format(metrics.get('downside_risk'), '.2%')}
Upside Risk: {safe_format(metrics.get('upside_risk'), '.2%')}

TRADE STATISTICS:
{'-'*40}
Win Rate: {safe_format(metrics.get('win_rate'), '.2%')}
Profit Factor: {safe_format(metrics.get('profit_factor'), '.3f')}
Average Win: {safe_format(metrics.get('avg_win'), '.2%')}
Average Loss: {safe_format(metrics.get('avg_loss'), '.2%')}
Max Consecutive Wins: {metrics.get('max_consecutive_wins', 0)}
Max Consecutive Losses: {metrics.get('max_consecutive_losses', 0)}

DISTRIBUTION METRICS:
{'-'*40}
Skewness: {safe_format(metrics.get('skewness'), '.3f')}
Kurtosis: {safe_format(metrics.get('kurtosis'), '.3f')}
Stability: {safe_format(metrics.get('stability'), '.3f')}

BENCHMARK METRICS:
{'-'*40}
Beta: {safe_format(metrics.get('beta'), '.3f')}
Alpha: {safe_format(metrics.get('alpha'), '.2%')}

ROLLING METRICS:
{'-'*40}
Average Rolling Sharpe: {safe_format(metrics.get('rolling_sharpe_mean'), '.3f')}
Rolling Sharpe Std Dev: {safe_format(metrics.get('rolling_sharpe_std'), '.3f')}
Average Rolling Volatility: {safe_format(metrics.get('rolling_volatility_mean'), '.2%')}

PAIR-SPECIFIC METRICS:
{'-'*40}
1-Day Autocorrelation: {safe_format(metrics.get('autocorr_1'), '.3f')}
5-Day Autocorrelation: {safe_format(metrics.get('autocorr_5'), '.3f')}
Volatility Clustering: {safe_format(metrics.get('volatility_clustering'), '.3f')}

{'='*80}
"""
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced report: {e}")
            return f"Error generating report: {e}"
    
    def _manual_autocorr(self, series: pd.Series, lag: int) -> float:
        """Calculate autocorrelation manually."""
        try:
            if len(series) <= lag:
                return None
            
            # Remove NaN values and convert to numpy array to avoid Series ambiguity
            series_clean = series.dropna()
            if len(series_clean) <= lag:
                return None
            
            # Convert to numpy array to avoid Series boolean operations
            series_array = series_clean.values
            
            # Calculate autocorrelation
            mean = np.mean(series_array)
            variance = np.var(series_array)
            
            if variance == 0:
                return 0
            
            # Calculate autocorrelation for given lag
            autocorr_sum = 0
            for i in range(len(series_array) - lag):
                autocorr_sum += (series_array[i] - mean) * (series_array[i + lag] - mean)
            
            autocorr = autocorr_sum / ((len(series_array) - lag) * variance)
            return autocorr
            
        except Exception as e:
            self.logger.error(f"Error calculating manual autocorrelation: {e}")
            return None 