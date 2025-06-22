"""
Enhanced vectorbt-based backtesting module for mean reversion pairs trading.
Provides high-performance vectorized operations, built-in metrics, and optimization capabilities.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
# import empyrical as ep  # Removed for Python 3.13 compatibility
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

@dataclass
class VectorBTConfig:
    """Configuration for vectorbt backtesting."""
    initial_capital: float = 1_000_000
    fees: float = 0.001  # 0.1% per trade
    slippage: float = 0.0002  # 0.02% slippage
    max_concurrent_positions: int = 5
    regime_scaling: bool = True
    regime_volatility_multiplier: float = 1.0
    regime_trend_multiplier: float = 1.0

class VectorBTBacktest:
    """High-performance vectorbt-based backtesting engine."""
    
    def __init__(self, config: VectorBTConfig):
        """Initialize the vectorbt backtest engine."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set vectorbt settings
        vbt.settings.set_theme("dark")
        vbt.settings.portfolio["init_cash"] = config.initial_capital
        vbt.settings.portfolio["fees"] = config.fees
        vbt.settings.portfolio["slippage"] = config.slippage
        
    def calculate_cointegration_stats(self, price1: pd.Series, price2: pd.Series) -> Dict:
        """Calculate cointegration statistics using statsmodels."""
        try:
            # Calculate spread
            spread = price1 - price2
            
            # Augmented Dickey-Fuller test
            adf_result = adfuller(spread.dropna(), regression='ct', autolag='AIC')
            
            # Linear regression for R²
            model = OLS(price1, price2).fit()
            
            return {
                'adf_statistic': adf_result[0],
                'adf_pvalue': adf_result[1],
                'adf_critical_values': adf_result[4],
                'is_cointegrated': adf_result[1] < 0.05,
                'r_squared': model.rsquared,
                'r_squared_adj': model.rsquared_adj,
                'spread_mean': spread.mean(),
                'spread_std': spread.std(),
                'spread_zscore': (spread - spread.mean()) / spread.std()
            }
        except Exception as e:
            self.logger.error(f"Error calculating cointegration stats: {e}")
            return {}
    
    def generate_signals_vectorized(self, price1: pd.Series, price2: pd.Series, 
                                  lookback: int = 20, entry_threshold: float = 2.0,
                                  exit_threshold: float = 0.5) -> Tuple[pd.Series, pd.Series]:
        """Generate vectorized signals using vectorbt."""
        try:
            # Calculate spread and z-score
            spread = price1 - price2
            z_score = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()
            
            # Generate entry signals
            long_entry = z_score < -entry_threshold
            short_entry = z_score > entry_threshold
            
            # Generate exit signals
            long_exit = z_score > -exit_threshold
            short_exit = z_score < exit_threshold
            
            # Combine signals
            entries = pd.Series(0, index=price1.index)
            entries[long_entry] = 1  # Long signal
            entries[short_entry] = -1  # Short signal
            
            exits = pd.Series(0, index=price1.index)
            exits[long_exit] = 1  # Exit long
            exits[short_exit] = -1  # Exit short
            
            return entries, exits
            
        except Exception as e:
            self.logger.error(f"Error generating vectorized signals: {e}")
            return pd.Series(0, index=price1.index), pd.Series(0, index=price1.index)
    
    def apply_regime_scaling(self, signals: pd.Series, regime_series: pd.Series) -> pd.Series:
        """Apply regime-based scaling to signals."""
        if not self.config.regime_scaling:
            return signals
        
        try:
            scaled_signals = signals.copy()
            
            # Scale based on regime
            for regime in regime_series.unique():
                if pd.isna(regime):
                    continue
                    
                regime_mask = regime_series == regime
                
                # Apply regime-specific scaling
                if regime == 0:  # Low volatility regime
                    scaled_signals[regime_mask] *= self.config.regime_volatility_multiplier
                elif regime == 1:  # High volatility regime
                    scaled_signals[regime_mask] *= (2 - self.config.regime_volatility_multiplier)
                elif regime == 2:  # Trend regime
                    scaled_signals[regime_mask] *= self.config.regime_trend_multiplier
            
            return scaled_signals
            
        except Exception as e:
            self.logger.error(f"Error applying regime scaling: {e}")
            return signals
    
    def run_vectorized_backtest(self, price1: pd.Series, price2: pd.Series,
                               regime_series: Optional[pd.Series] = None,
                               **signal_params) -> Dict:
        """Run vectorized backtest using vectorbt."""
        try:
            # Generate signals
            entries, exits = self.generate_signals_vectorized(price1, price2, **signal_params)
            
            # Apply regime scaling if available
            if regime_series is not None:
                entries = self.apply_regime_scaling(entries, regime_series)
            
            # Create price DataFrame for vectorbt
            prices = pd.DataFrame({
                'asset1': price1,
                'asset2': price2
            })
            
            # Run vectorbt portfolio simulation
            portfolio = vbt.Portfolio.from_signals(
                prices,
                entries,
                exits,
                init_cash=self.config.initial_capital,
                fees=self.config.fees,
                slippage=self.config.slippage,
                freq='1D'
            )
            
            # Extract results
            results = {
                'portfolio': portfolio,
                'equity_curve': portfolio.value(),
                'returns': portfolio.returns(),
                'trades': portfolio.trades.records_readable,
                'signals': {
                    'entries': entries,
                    'exits': exits
                },
                'prices': prices
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running vectorized backtest: {e}")
            return {}
    
    def calculate_enhanced_metrics(self, returns: pd.Series, 
                                 equity_curve: pd.Series) -> Dict:
        """Calculate enhanced performance metrics using numpy/pandas/scipy."""
        try:
            # Basic metrics
            total_return = (1 + returns).prod() - 1
            years = len(returns) / 252
            annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            annual_volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
            downside_returns = returns[returns < 0]
            sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252) if downside_returns.std() != 0 else 0
            running_max = (1 + returns).cumprod().expanding().max()
            drawdown = (1 + returns).cumprod() / running_max - 1
            max_drawdown = drawdown.min()
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            # Risk metrics
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
            tail_ratio = abs(np.percentile(returns, 95) / np.percentile(returns, 5)) if np.percentile(returns, 5) != 0 else 0
            # Additional metrics
            stability = 1 - returns.std() / returns.mean() if returns.mean() != 0 else 0
            downside_risk = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            upside_returns = returns[returns > 0]
            upside_risk = upside_returns.std() * np.sqrt(252) if len(upside_returns) > 0 else 0
            win_rate = (returns > 0).mean()
            profit_factor = returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if abs(returns[returns < 0].sum()) > 0 else 0
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'tail_ratio': tail_ratio,
                'stability': stability,
                'downside_risk': downside_risk,
                'upside_risk': upside_risk,
                'win_rate': win_rate,
                'profit_factor': profit_factor
            }
        except Exception as e:
            self.logger.error(f"Error calculating enhanced metrics: {e}")
            return {}
    
    def optimize_parameters(self, price1: pd.Series, price2: pd.Series,
                          regime_series: Optional[pd.Series] = None,
                          param_ranges: Optional[Dict] = None) -> Dict:
        """Optimize strategy parameters using vectorbt's optimization tools."""
        try:
            if param_ranges is None:
                param_ranges = {
                    'lookback': np.arange(10, 31, 5),
                    'entry_threshold': np.arange(1.5, 3.1, 0.5),
                    'exit_threshold': np.arange(0.3, 1.1, 0.2)
                }
            
            # Create parameter combinations
            param_combinations = vbt.utils.combine_params(param_ranges)
            
            # Run optimization
            def run_backtest_wrapper(lookback, entry_threshold, exit_threshold):
                entries, exits = self.generate_signals_vectorized(
                    price1, price2, lookback, entry_threshold, exit_threshold
                )
                
                if regime_series is not None:
                    entries = self.apply_regime_scaling(entries, regime_series)
                
                prices = pd.DataFrame({'asset1': price1, 'asset2': price2})
                
                portfolio = vbt.Portfolio.from_signals(
                    prices, entries, exits,
                    init_cash=self.config.initial_capital,
                    fees=self.config.fees,
                    slippage=self.config.slippage,
                    freq='1D'
                )
                
                return portfolio.sharpe_ratio()
            
            # Run optimization
            results = vbt.run_func(
                run_backtest_wrapper,
                param_combinations,
                engine='ray',  # Use Ray for parallel processing
                show_progress=True
            )
            
            # Find best parameters
            best_idx = results.idxmax()
            best_params = param_combinations.iloc[best_idx]
            best_sharpe = results.iloc[best_idx]
            
            return {
                'best_params': best_params.to_dict(),
                'best_sharpe': best_sharpe,
                'all_results': results,
                'param_combinations': param_combinations
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {e}")
            return {}
    
    def generate_report(self, results: Dict, pair_name: str = "Unknown") -> str:
        """Generate comprehensive backtest report."""
        try:
            if not results:
                return "No results to report"
            
            portfolio = results.get('portfolio')
            metrics = self.calculate_enhanced_metrics(
                portfolio.returns(), portfolio.value()
            )
            
            report = f"""
{'='*80}
VECTORBT BACKTEST REPORT: {pair_name}
{'='*80}

PERFORMANCE METRICS:
{'-'*40}
Total Return: {metrics.get('total_return', 0):.2%}
Annual Return: {metrics.get('annual_return', 0):.2%}
Annual Volatility: {metrics.get('annual_volatility', 0):.2%}
Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}
Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}
Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}
Max Drawdown: {metrics.get('max_drawdown', 0):.2%}

RISK METRICS:
{'-'*40}
Value at Risk (95%): {metrics.get('var_95', 0):.2%}
Conditional VaR (95%): {metrics.get('cvar_95', 0):.2%}
Tail Ratio: {metrics.get('tail_ratio', 0):.3f}
Downside Risk: {metrics.get('downside_risk', 0):.2%}
Upside Risk: {metrics.get('upside_risk', 0):.2%}

TRADE STATISTICS:
{'-'*40}
Win Rate: {metrics.get('win_rate', 0):.2%}
Profit Factor: {metrics.get('profit_factor', 0):.3f}
Stability: {metrics.get('stability', 0):.3f}

PORTFOLIO STATISTICS:
{'-'*40}
Total Trades: {len(portfolio.trades.records_readable)}
Final Value: ${portfolio.value().iloc[-1]:,.2f}
Initial Capital: ${self.config.initial_capital:,.2f}
Total PnL: ${portfolio.value().iloc[-1] - self.config.initial_capital:,.2f}

{'='*80}
"""
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return f"Error generating report: {e}" 