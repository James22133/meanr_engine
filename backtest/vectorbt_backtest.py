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
    # Slippage per trade side (0.15% results in ~0.3% round trip)
    slippage: float = 0.0015
    max_concurrent_positions: int = 5
    regime_scaling: bool = True
    regime_volatility_multiplier: float = 1.0
    regime_trend_multiplier: float = 1.0
    max_holding_days: int = 30
    atr_stop_loss_mult: Optional[float] = None
    atr_lookback: int = 14
    execution_timing: bool = False
    execution_penalty_factor: float = 1.05

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

            # Rolling ADF test on 30-day window to ensure ongoing cointegration
            adf_window = 30
            rolling_adf = spread.rolling(adf_window).apply(
                lambda x: adfuller(x)[1] if x.notna().sum() == adf_window else np.nan,
                raw=False
            )
            
            # Generate entry signals - vectorbt expects 1 for long, -1 for short, 0 for no position
            entries = pd.Series(0, index=price1.index)
            entries[z_score < -entry_threshold] = 1   # Long when z-score is low (spread is low)
            entries[z_score > entry_threshold] = -1   # Short when z-score is high (spread is high)

            # Cancel entries if recent ADF test fails (p-value > 0.05)
            invalid_coint = rolling_adf > 0.05
            entries[invalid_coint] = 0
            
            # Generate exit signals - vectorbt expects True for exit, False for no exit
            exits = pd.Series(False, index=price1.index)
            exits[(z_score >= -exit_threshold) & (z_score <= exit_threshold)] = True
            
            # Debug logging
            long_signals = (entries == 1).sum()
            short_signals = (entries == -1).sum()
            invalid_days = invalid_coint.sum()
            exit_signals = exits.sum()
            
            self.logger.info(
                f"VectorBT signals - Long: {long_signals}, Short: {short_signals}, Exit: {exit_signals}, Invalid: {invalid_days}"
            )
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

    def _calculate_position_size_df(
        self,
        price1: pd.Series,
        price2: pd.Series,
        scaled_entries: pd.Series,
        vol_lookback: int = 20,
    ) -> pd.DataFrame:
        """Calculate position sizes per trade based on volatility and config."""
        try:
            base_size = 1.0 / max(self.config.max_concurrent_positions, 1)

            spread = price1 - price2
            volatility = spread.pct_change().rolling(vol_lookback).std()
            median_vol = volatility.median()
            if pd.isna(median_vol) or median_vol == 0:
                vol_factor = pd.Series(1.0, index=spread.index)
            else:
                vol_factor = (median_vol / volatility).clip(0.5, 2.0).fillna(1.0)

            size_series = base_size * vol_factor * scaled_entries.abs()
            size_series = size_series.clip(upper=1.0)

            size_df = pd.DataFrame({"asset1": size_series, "asset2": size_series})
            size_df = size_df.reindex(price1.index).fillna(0.0)
            return size_df
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return pd.DataFrame(0.0, index=price1.index, columns=["asset1", "asset2"])

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

#conflict resolved here wkkj7n-codex/modify-backtest-engine-with-slippage-and-filters
    def execute_signals(
        self,
        returns: pd.Series,
        vix: Optional[pd.Series] = None,
        spy_ret_5d: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Apply execution timing penalty only during stress periods."""
        if not self.config.execution_timing:
            return returns

        if vix is None:
            vix = pd.Series(0, index=returns.index)
        if spy_ret_5d is None:
            spy_ret_5d = pd.Series(0, index=returns.index)

        stress_mask = (vix > 25) | (spy_ret_5d < -0.02)
        adjusted = returns.copy()
        adjusted[stress_mask] = adjusted[stress_mask] / self.config.execution_penalty_factor
        return adjusted
#conflict resolved here 
    def execute_signals(self, returns: pd.Series) -> pd.Series:
        """Apply execution timing penalty if enabled."""
        if self.config.execution_timing:
            self.logger.info("Applying execution timing penalty factor")
            return returns / self.config.execution_penalty_factor
        return returns
#conflict resolved here  main
    
    def run_vectorized_backtest(
        self,
        price1: pd.Series,
        price2: pd.Series,
        regime_series: Optional[pd.Series] = None,
        vix_series: Optional[pd.Series] = None,
        spy_series: Optional[pd.Series] = None,
        volume1: Optional[pd.Series] = None,
        volume2: Optional[pd.Series] = None,
        **signal_params,
    ) -> Dict:
        """Run vectorized backtest using vectorbt with detailed trade logging."""
        try:
            # Generate signals
            entries, exits = self.generate_signals_vectorized(price1, price2, **signal_params)
            
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
            # Apply regime scaling to determine position weights while keeping
            # entry direction intact for vectorbt
            if regime_series is not None:
                scaled_entries = self.apply_regime_scaling(entries, regime_series)
            else:
                scaled_entries = entries

            entry_signals = pd.DataFrame({"asset1": np.sign(scaled_entries), "asset2": np.sign(scaled_entries)})
            exit_signals = pd.DataFrame({"asset1": exits, "asset2": exits})

            size_df = self._calculate_position_size_df(price1, price2, scaled_entries)

            # Dynamic slippage adjustment based on dollar volume if provided
            dyn_slippage = self.config.slippage
            if volume1 is not None and volume2 is not None:
                dollar_vol1 = (price1 * volume1).rolling(30).mean().iloc[-1]
                dollar_vol2 = (price2 * volume2).rolling(30).mean().iloc[-1]
                avg_dv = np.nanmean([dollar_vol1, dollar_vol2])
                if not np.isnan(avg_dv) and avg_dv > 0:
                    factor = min(1.5, max(0.5, 5_000_000 / avg_dv))
                    dyn_slippage = self.config.slippage * factor

            portfolio = vbt.Portfolio.from_signals(
                close=prices,
                entries=entry_signals,
                exits=exit_signals,
                size=size_df,
                direction="both",
                init_cash=self.config.initial_capital,
                fees=self.config.fees,
                slippage=dyn_slippage,
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
#conflict resolved here  wkkj7n-codex/modify-backtest-engine-with-slippage-and-filters
            vix = vix_series.reindex(portfolio.returns().index) if vix_series is not None else None
            spy_ret = spy_series.pct_change(5).reindex(portfolio.returns().index) if spy_series is not None else None
            adj_returns = self.execute_signals(portfolio.returns(), vix, spy_ret)
#conflict resolved here 
            adj_returns = self.execute_signals(portfolio.returns())
#conflict resolved here  main
            results = {
                'portfolio': portfolio,
                'equity_curve': self.config.initial_capital * (1 + adj_returns).cumprod(),
                'returns': adj_returns,
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
            if returns.empty or equity_curve.empty:
                return {}
            
            # Ensure we're working with clean data
            returns_clean = returns.dropna()
            equity_clean = equity_curve.dropna()
            
            if returns_clean.empty or equity_clean.empty:
                return {}
            
            # Basic metrics
            total_return = (1 + returns_clean).prod() - 1
            years = len(returns_clean) / 252
            annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            annual_volatility = returns_clean.std() * np.sqrt(252)
            
            # Risk-adjusted returns
            if returns_clean.std() != 0:
                sharpe_ratio = returns_clean.mean() / returns_clean.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0
                
            downside_returns = returns_clean[returns_clean < 0]
            if len(downside_returns) > 0 and downside_returns.std() != 0:
                sortino_ratio = returns_clean.mean() / downside_returns.std() * np.sqrt(252)
            else:
                sortino_ratio = 0
                
            # Drawdown calculation
            running_max = (1 + returns_clean).cumprod().expanding().max()
            drawdown = (1 + returns_clean).cumprod() / running_max - 1
            max_drawdown = drawdown.min()
            
            if max_drawdown != 0:
                calmar_ratio = annual_return / abs(max_drawdown)
            else:
                calmar_ratio = 0
                
            # Risk metrics
            var_95 = np.percentile(returns_clean, 5)
            returns_below_var = returns_clean[returns_clean <= var_95]
            cvar_95 = returns_below_var.mean() if len(returns_below_var) > 0 else 0
            
            percentile_5 = np.percentile(returns_clean, 5)
            percentile_95 = np.percentile(returns_clean, 95)
            tail_ratio = abs(percentile_95 / percentile_5) if percentile_5 != 0 else 0
            
            # Additional metrics
            if returns_clean.mean() != 0:
                stability = 1 - returns_clean.std() / returns_clean.mean()
            else:
                stability = 0
                
            downside_risk = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            upside_returns = returns_clean[returns_clean > 0]
            upside_risk = upside_returns.std() * np.sqrt(252) if len(upside_returns) > 0 else 0
            
            win_rate = (returns_clean > 0).mean()
            
            gains = returns_clean[returns_clean > 0].sum()
            losses = abs(returns_clean[returns_clean < 0].sum())
            profit_factor = gains / losses if losses > 0 else 0
            
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
                    scaled_entries = self.apply_regime_scaling(entries, regime_series)
                else:
                    scaled_entries = entries

                prices = pd.DataFrame({"asset1": price1, "asset2": price2})
                entry_signals = pd.DataFrame({"asset1": np.sign(scaled_entries), "asset2": np.sign(scaled_entries)})
                exit_signals = pd.DataFrame({"asset1": exits, "asset2": exits})
                size_df = self._calculate_position_size_df(price1, price2, scaled_entries)

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
                returns = results.get('returns', portfolio.returns())
                equity = results.get('equity_curve', portfolio.value())
                
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