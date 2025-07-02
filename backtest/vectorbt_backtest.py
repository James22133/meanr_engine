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
from core.trade_extraction import filter_long_holding_trades

# Fix for vectorbt API change - use new import
try:
    from vectorbt.utils.params import create_param_combs as combine_params
except ImportError:
    # Fallback for older versions
    try:
        from vectorbt.utils import combine_params
    except ImportError:
        # Final fallback - create our own simple version
        def combine_params(param_dict):
            """Simple parameter combination generator."""
            import itertools
            keys = list(param_dict.keys())
            values = list(param_dict.values())
            combinations = list(itertools.product(*values))
            return pd.DataFrame(combinations, columns=keys)

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
    max_holding_days: int = 30
    atr_stop_loss_mult: Optional[float] = None
    atr_lookback: int = 14

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
            
            # Generate entry signals - vectorbt expects 1 for long, -1 for short, 0 for no position
            entries = pd.Series(0, index=price1.index)
            entries[z_score < -entry_threshold] = 1   # Long when z-score is low (spread is low)
            entries[z_score > entry_threshold] = -1   # Short when z-score is high (spread is high)
            
            # Generate exit signals - vectorbt expects True for exit, False for no exit
            exits = pd.Series(False, index=price1.index)
            exits[(z_score >= -exit_threshold) & (z_score <= exit_threshold)] = True
            
            # Debug logging
            long_signals = (entries == 1).sum()
            short_signals = (entries == -1).sum()
            exit_signals = exits.sum()
            
            self.logger.info(f"VectorBT signals - Long: {long_signals}, Short: {short_signals}, Exit: {exit_signals}")
            self.logger.info(f"Z-score range: {z_score.min():.3f} to {z_score.max():.3f}")
            
            return entries, exits
            
        except Exception as e:
            self.logger.error(f"Error generating vectorized signals: {e}")
            return pd.Series(0, index=price1.index), pd.Series(False, index=price1.index)
    
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

    def _apply_atr_stop_loss(
        self,
        prices: pd.DataFrame,
        entries: pd.DataFrame,
        exits: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create stop-loss exit signals based on ATR of the spread."""
        if self.config.atr_stop_loss_mult is None:
            return exits

        try:
            spread = prices["asset1"] - prices["asset2"]
            tr = spread.diff().abs()
            atr = tr.rolling(self.config.atr_lookback).mean()

            stop_exits = pd.Series(False, index=spread.index)
            position = 0
            entry_spread = 0.0

            for i, date in enumerate(spread.index):
                if position == 0:
                    if entries["asset1"].iloc[i] == 1 or entries["asset1"].iloc[i] == -1:
                        position = int(entries["asset1"].iloc[i])
                        entry_spread = spread.iloc[i]
                else:
                    stop_level = atr.iloc[i] * self.config.atr_stop_loss_mult
                    if position == 1 and spread.iloc[i] - entry_spread < -stop_level:
                        stop_exits.iloc[i] = True
                        position = 0
                    elif position == -1 and spread.iloc[i] - entry_spread > stop_level:
                        stop_exits.iloc[i] = True
                        position = 0

                if exits["asset1"].iloc[i] or exits["asset2"].iloc[i]:
                    position = 0

            exits_combined = exits.copy()
            exits_combined["asset1"] = exits_combined["asset1"] | stop_exits
            exits_combined["asset2"] = exits_combined["asset2"] | stop_exits
            return exits_combined
        except Exception as e:
            self.logger.error(f"Error applying ATR stop loss: {e}")
            return exits
    
    def run_vectorized_backtest(self, price1: pd.Series, price2: pd.Series,
                               regime_series: Optional[pd.Series] = None,
                               **signal_params) -> Dict:
        """Run vectorized backtest using vectorbt with detailed trade logging."""
        try:
            # Generate signals
            entries, exits = self.generate_signals_vectorized(price1, price2, **signal_params)
            
            # Apply regime scaling if available
            if regime_series is not None:
                entries = self.apply_regime_scaling(entries, regime_series)
            
            # Create spread-based portfolio for pairs trading
            # Calculate the spread (price1 - price2)
            spread = price1 - price2
            
            # Validate spread prices - ensure they are finite and positive
            # For pairs trading, we can use the absolute value of the spread
            # or add a constant to ensure positivity
            spread_abs = spread.abs()
            
            # Check if we have valid prices
            if not spread_abs.notna().all() or (spread_abs <= 0).any():
                self.logger.warning(f"Invalid spread values detected. Using absolute values with offset.")
                # Use absolute spread with a small offset to ensure positivity
                spread_tradeable = spread_abs + 0.01
            else:
                spread_tradeable = spread_abs
            
            # Create a single-asset portfolio based on the spread
            # Construct price and signal frames with consistent shapes
            prices = pd.DataFrame({"asset1": price1, "asset2": price2})
            entry_signals = pd.DataFrame({"asset1": entries, "asset2": entries})
            exit_signals = pd.DataFrame({"asset1": exits, "asset2": exits})


            size_df = pd.DataFrame(0.25, index=prices.index, columns=prices.columns)

            portfolio = vbt.Portfolio.from_signals(
                close=prices,
                entries=entry_signals,
                exits=exit_signals,
                size=size_df,
                direction="both",
                init_cash=self.config.initial_capital,
                fees=self.config.fees,
                slippage=self.config.slippage,
                freq="1D",
                accumulate=False,
                upon_long_conflict="ignore",
                upon_short_conflict="ignore"
            )
            
            # Extract detailed trade information
            trade_records = portfolio.trades.records_readable
            pair_label = (getattr(price1, "name", "A"), getattr(price2, "name", "B"))
            detailed_trades = self._extract_detailed_trades(
                trade_records, price1, price2, entries, exits, regime_series, pair_label
            )

            # Filter trades with excessive holding periods
            if detailed_trades and self.config.max_holding_days:
                detailed_trades = filter_long_holding_trades(
                    detailed_trades, self.config.max_holding_days
                )
            
            # Extract results
            results = {
                'portfolio': portfolio,
                'equity_curve': portfolio.value(),
                'returns': portfolio.returns(),
                'trades': trade_records,
                'detailed_trades': detailed_trades,
                'signals': {
                    'entries': entries,
                    'exits': exits
                },
                'prices': pd.DataFrame({
                    'asset1': price1,
                    'asset2': price2,
                    'spread': spread,
                    'spread_tradeable': spread_tradeable
                })
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running vectorized backtest: {e}")
            return {}
    
    def _extract_detailed_trades(self, trade_records: pd.DataFrame,
                                price1: pd.Series, price2: pd.Series,
                                entries: pd.Series, exits: pd.Series,
                                regime_series: Optional[pd.Series] = None,
                                pair_label: Tuple[str, str] = ("A", "B")) -> List[Dict]:
        """Extract detailed trade information for analysis."""
        try:
            detailed_trades = []
            
            # Calculate spread and z-score for the entire period
            spread = price1 - price2
            lookback = 20  # Default lookback
            z_score = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()
            
            # Check if we have any trades
            if trade_records.empty:
                self.logger.warning("No trades found in trade records")
                return []
            
            # Debug: Print column names to understand the structure
            self.logger.info(f"Trade records columns: {list(trade_records.columns)}")
            
            # Determine the correct column names for entry and exit times
            # VectorBT uses 'Entry Timestamp' and 'Exit Timestamp' in newer versions
            entry_time_col = None
            exit_time_col = None
            
            possible_entry_cols = ['Entry Timestamp', 'entry_time', 'EntryTime', 'entry', 'Entry Time']
            possible_exit_cols = ['Exit Timestamp', 'exit_time', 'ExitTime', 'exit', 'Exit Time']
            
            for col in possible_entry_cols:
                if col in trade_records.columns:
                    entry_time_col = col
                    break
                    
            for col in possible_exit_cols:
                if col in trade_records.columns:
                    exit_time_col = col
                    break
            
            if entry_time_col is None or exit_time_col is None:
                self.logger.error(f"Could not find entry/exit time columns. Available: {list(trade_records.columns)}")
                return []
            
            # Also check for PnL column
            pnl_col = None
            possible_pnl_cols = ['PnL', 'pnl', 'Pnl', 'PnL [%]', 'Return [%]', 'Return']
            for col in possible_pnl_cols:
                if col in trade_records.columns:
                    pnl_col = col
                    break
            
            if pnl_col is None:
                self.logger.error(f"Could not find PnL column. Available: {list(trade_records.columns)}")
                return []
            
            for _, trade in trade_records.iterrows():
                try:
                    entry_idx = trade[entry_time_col]
                    exit_idx = trade[exit_time_col]
                    
                    # Get entry and exit information
                    entry_price1 = price1.loc[entry_idx]
                    entry_price2 = price2.loc[entry_idx]
                    exit_price1 = price1.loc[exit_idx]
                    exit_price2 = price2.loc[exit_idx]
                    
                    # Calculate z-scores
                    entry_zscore = z_score.loc[entry_idx] if pd.notna(z_score.loc[entry_idx]) else 0
                    exit_zscore = z_score.loc[exit_idx] if pd.notna(z_score.loc[exit_idx]) else 0
                    
                    # Get regime information
                    entry_regime = regime_series.loc[entry_idx] if regime_series is not None else 0
                    exit_regime = regime_series.loc[exit_idx] if regime_series is not None else 0
                    
                    # Calculate holding period
                    holding_period = (exit_idx - entry_idx).days
                    
                    # Determine trade direction - check multiple possible column names
                    direction = 'unknown'
                    direction_col = None
                    possible_direction_cols = ['Direction', 'direction', 'Side', 'side']
                    for col in possible_direction_cols:
                        if col in trade_records.columns:
                            direction_col = col
                            break
                    
                    if direction_col:
                        if trade[direction_col] == 0 or trade[direction_col] == 'long':
                            direction = 'long'
                        elif trade[direction_col] == 1 or trade[direction_col] == 'short':
                            direction = 'short'
                    
                    # Calculate spread metrics
                    entry_spread = entry_price1 - entry_price2
                    exit_spread = exit_price1 - exit_price2
                    spread_change = exit_spread - entry_spread
                    
                    # Get PnL value with proper handling
                    pnl_value = trade[pnl_col]
                    if pd.isna(pnl_value):
                        pnl_value = 0.0
                    
                    # Get other trade details with safe defaults
                    size = trade.get('Size', 1.0)
                    fees = trade.get('Entry Fees', 0.0) + trade.get('Exit Fees', 0.0)
                    slippage = trade.get('Slippage', 0.0)
                    
                    detailed_trade = {
                        'pair': f"{pair_label[0]}-{pair_label[1]}",
                        'entry_date': entry_idx,
                        'exit_date': exit_idx,
                        'direction': direction,
                        'entry_price1': entry_price1,
                        'entry_price2': entry_price2,
                        'exit_price1': exit_price1,
                        'exit_price2': exit_price2,
                        'entry_spread': entry_spread,
                        'exit_spread': exit_spread,
                        'spread_change': spread_change,
                        'entry_zscore': entry_zscore,
                        'exit_zscore': exit_zscore,
                        'entry_regime': entry_regime,
                        'exit_regime': exit_regime,
                        'holding_period': holding_period,
                        'pnl': pnl_value,
                        'return_pct': trade.get('Return [%]', 0.0),
                        'size': size,
                        'fees': fees,
                        'slippage': slippage
                    }
                    
                    detailed_trades.append(detailed_trade)
                    
                except Exception as e:
                    self.logger.error(f"Error processing individual trade: {e}")
                    continue
            
            self.logger.info(f"Successfully extracted {len(detailed_trades)} detailed trades")
            return detailed_trades
            
        except Exception as e:
            self.logger.error(f"Error extracting detailed trades: {e}")
            return []
    
    def save_trade_logs(self, results: Dict, pair_name: str, output_dir: str = "trade_logs") -> str:
        """Save detailed trade logs to CSV file."""
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            if 'detailed_trades' in results and results['detailed_trades']:
                trades_df = pd.DataFrame(results['detailed_trades'])
                
                # Create filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{output_dir}/{pair_name[0]}_{pair_name[1]}_trades_{timestamp}.csv"
                
                trades_df.to_csv(filename, index=False)
                self.logger.info(f"Saved detailed trade logs to {filename}")
                
                return filename
            else:
                self.logger.warning("No detailed trades found to save")
                return ""
                
        except Exception as e:
            self.logger.error(f"Error saving trade logs: {e}")
            return ""
    
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
            
            # Create parameter combinations using tuple format for vectorbt 0.27.3
            param_tuples = tuple(param_ranges.items())
            param_combinations = combine_params(param_tuples)

            # Define the backtest runner
            def run_backtest_wrapper(lookback, entry_threshold, exit_threshold):
                entries, exits = self.generate_signals_vectorized(
                    price1, price2, lookback, entry_threshold, exit_threshold
                )
                if regime_series is not None:
                    entries = self.apply_regime_scaling(entries, regime_series)

                prices = pd.DataFrame({"asset1": price1, "asset2": price2})
                entry_signals = pd.DataFrame({"asset1": entries, "asset2": entries})
                exit_signals = pd.DataFrame({"asset1": exits, "asset2": exits})
                size_df = pd.DataFrame(0.25, index=prices.index, columns=prices.columns)

                portfolio = vbt.Portfolio.from_signals(
                    close=prices,
                    entries=entry_signals,
                    exits=exit_signals,
                    size=size_df,
                    direction="both",
                    init_cash=self.config.initial_capital,
                    fees=self.config.fees,
                    slippage=self.config.slippage,
                    freq="1D"
                )
                return portfolio.sharpe_ratio()

            # Iterate over parameter combinations as dicts
            results = []
            for params in param_combinations.to_dict(orient="records"):
                try:
                    sharpe = run_backtest_wrapper(**params)
                except Exception as e:
                    self.logger.error(f"Error in optimization for params {params}: {e}")
                    sharpe = float('-inf')
                results.append(sharpe)
            results = pd.Series(results, index=param_combinations.index)

            # Find best parameters
            best_idx = results.idxmax()
            best_params = param_combinations.iloc[best_idx]
            best_sharpe = results.iloc[best_idx]
            self.logger.info(f"Best parameters found: {best_params.to_dict()}")
            self.logger.info(f"Best Sharpe ratio: {best_sharpe:.4f}")
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
            if portfolio is None:
                return f"No portfolio data available for {pair_name}"
            
            try:
                returns = portfolio.returns()
                equity = portfolio.value()
                
                if returns.empty or equity.empty:
                    return f"No valid returns/equity data for {pair_name}"
                
                metrics = self.calculate_enhanced_metrics(returns, equity)
            except Exception as e:
                self.logger.error(f"Error calculating metrics for {pair_name}: {e}")
                return f"Error calculating metrics for {pair_name}: {e}"
            
            # Helper function to safely format values
            def safe_format(value, format_str=".2%", default="N/A"):
                """Safely format a value, handling None, NaN, and Series."""
                if value is None or pd.isna(value):
                    return default
                try:
                    # Convert Series to scalar if needed
                    if hasattr(value, 'item'):
                        value = value.item()
                    elif hasattr(value, 'iloc'):
                        value = value.iloc[-1] if len(value) > 0 else 0
                    
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
VECTORBT BACKTEST REPORT: {pair_name}
{'='*80}

PERFORMANCE METRICS:
{'-'*40}
Total Return: {safe_format(metrics.get('total_return'), '.2%')}
Annual Return: {safe_format(metrics.get('annual_return'), '.2%')}
Annual Volatility: {safe_format(metrics.get('annual_volatility'), '.2%')}
Sharpe Ratio: {safe_format(metrics.get('sharpe_ratio'), '.3f')}
Sortino Ratio: {safe_format(metrics.get('sortino_ratio'), '.3f')}
Calmar Ratio: {safe_format(metrics.get('calmar_ratio'), '.3f')}
Max Drawdown: {safe_format(metrics.get('max_drawdown'), '.2%')}

RISK METRICS:
{'-'*40}
Value at Risk (95%): {safe_format(metrics.get('var_95'), '.2%')}
Conditional VaR (95%): {safe_format(metrics.get('cvar_95'), '.2%')}
Tail Ratio: {safe_format(metrics.get('tail_ratio'), '.3f')}
Downside Risk: {safe_format(metrics.get('downside_risk'), '.2%')}
Upside Risk: {safe_format(metrics.get('upside_risk'), '.2%')}

TRADE STATISTICS:
{'-'*40}
Win Rate: {safe_format(metrics.get('win_rate'), '.2%')}
Profit Factor: {safe_format(metrics.get('profit_factor'), '.3f')}
Stability: {safe_format(metrics.get('stability'), '.3f')}

PORTFOLIO STATISTICS:
{'-'*40}
Total Trades: {len(portfolio.trades.records_readable)}
Final Value: {safe_format(portfolio.value().iloc[-1], '.2f')}
Initial Capital: {safe_format(self.config.initial_capital, '.2f')}
Total PnL: {safe_format(portfolio.value().iloc[-1] - self.config.initial_capital, '.2f')}

{'='*80}
"""
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return f"Error generating report: {e}" 