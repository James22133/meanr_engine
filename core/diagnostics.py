"""
Trade loss diagnostics module for analyzing trading performance and identifying improvement areas.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import os

class TradeDiagnostics:
    """Class for analyzing trade performance and identifying loss patterns."""
    
    def __init__(self, config):
        """Initialize the diagnostics module with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set style
        sns.set_theme()
        
    def analyze_trade_performance(self, all_trades: List[Dict]) -> Dict:
        """Comprehensive analysis of trade performance."""
        try:
            if not all_trades:
                self.logger.warning("No trades found in backtest results")
                return {}
                
            trades_df = pd.DataFrame(all_trades)
            if trades_df.empty:
                self.logger.warning("No trades to analyze")
                return {}
            
            # Check if PnL column exists
            if 'pnl' not in trades_df.columns:
                self.logger.error("PnL column not found in trade data")
                return {}
                
            # Calculate basic statistics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            
            # Calculate PnL statistics with safe handling
            total_pnl = trades_df['pnl'].sum()
            avg_pnl = trades_df['pnl'].mean()
            median_pnl = trades_df['pnl'].median()
            std_pnl = trades_df['pnl'].std()
            
            # Worst trades analysis
            worst_trades = trades_df.nsmallest(max(1, int(total_trades * 0.1)), 'pnl')
            
            # Best trades analysis
            best_trades = trades_df.nlargest(max(1, int(total_trades * 0.1)), 'pnl')
            
            # Loss decomposition
            loss_decomposition = self._decompose_losses(trades_df)
            
            # Pair performance analysis
            pair_performance = self._analyze_pair_performance(trades_df)
            
            # Regime analysis
            regime_analysis = self._analyze_regime_performance(trades_df)
            
            # Holding period analysis
            holding_period_analysis = self._analyze_holding_periods(trades_df)
            
            return {
                'summary': {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                    'total_pnl': total_pnl,
                    'avg_pnl': avg_pnl,
                    'median_pnl': median_pnl,
                    'std_pnl': std_pnl,
                    'max_profit': trades_df['pnl'].max(),
                    'max_loss': trades_df['pnl'].min(),
                    'profit_factor': abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / 
                                       trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else float('inf')
                },
                'worst_trades': worst_trades,
                'best_trades': best_trades,
                'loss_decomposition': loss_decomposition,
                'pair_performance': pair_performance,
                'regime_analysis': regime_analysis,
                'holding_period_analysis': holding_period_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trade performance: {e}")
            return {}
    
    def _decompose_losses(self, trades_df: pd.DataFrame) -> Dict:
        """Decompose losses by various factors."""
        try:
            # Filter losing trades
            losing_trades = trades_df[trades_df['pnl'] < 0].copy()
            
            if losing_trades.empty:
                return {}
            
            # Calculate total loss
            total_loss = losing_trades['pnl'].sum()
            
            # Loss by pair
            loss_by_pair = losing_trades.groupby('pair')['pnl'].agg(['sum', 'count', 'mean']).round(2)
            loss_by_pair['contribution_pct'] = (loss_by_pair['sum'] / total_loss * 100).round(2)
            
            # Loss by regime (if available)
            loss_by_regime = {}
            if 'regime' in losing_trades.columns:
                loss_by_regime = losing_trades.groupby('regime')['pnl'].agg(['sum', 'count', 'mean']).round(2)
                loss_by_regime['contribution_pct'] = (loss_by_regime['sum'] / total_loss * 100).round(2)
            
            # Loss by holding period buckets
            if 'holding_period' in losing_trades.columns:
                losing_trades['holding_bucket'] = pd.cut(
                    losing_trades['holding_period'], 
                    bins=[0, 1, 3, 7, 14, 30, float('inf')],
                    labels=['1d', '2-3d', '4-7d', '8-14d', '15-30d', '30d+']
                )
                loss_by_holding = losing_trades.groupby('holding_bucket')['pnl'].agg(['sum', 'count', 'mean']).round(2)
                loss_by_holding['contribution_pct'] = (loss_by_holding['sum'] / total_loss * 100).round(2)
            else:
                loss_by_holding = pd.DataFrame()
            
            return {
                'total_loss': total_loss,
                'loss_by_pair': loss_by_pair,
                'loss_by_regime': loss_by_regime,
                'loss_by_holding_period': loss_by_holding,
                'avg_loss': losing_trades['pnl'].mean(),
                'median_loss': losing_trades['pnl'].median(),
                'loss_std': losing_trades['pnl'].std()
            }
            
        except Exception as e:
            self.logger.error(f"Error decomposing losses: {e}")
            return {}
    
    def _analyze_pair_performance(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze performance by pair."""
        try:
            pair_stats = trades_df.groupby('pair').agg({
                'pnl': ['sum', 'mean', 'count', 'std'],
                'holding_period': 'mean' if 'holding_period' in trades_df.columns else lambda x: 0
            }).round(2)
            
            # Flatten column names
            pair_stats.columns = ['_'.join(col).strip() for col in pair_stats.columns]
            
            # Calculate win rate per pair
            pair_stats['win_rate'] = trades_df.groupby('pair').apply(
                lambda x: len(x[x['pnl'] > 0]) / len(x) * 100
            ).round(2)
            
            # Calculate profit factor per pair
            pair_stats['profit_factor'] = trades_df.groupby('pair').apply(
                lambda x: abs(x[x['pnl'] > 0]['pnl'].sum() / x[x['pnl'] < 0]['pnl'].sum()) 
                if len(x[x['pnl'] < 0]) > 0 else float('inf')
            ).round(2)
            
            # Sort by total PnL
            pair_stats = pair_stats.sort_values('pnl_sum', ascending=False)
            
            return pair_stats.to_dict('index')
            
        except Exception as e:
            self.logger.error(f"Error analyzing pair performance: {e}")
            return {}
    
    def _analyze_regime_performance(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze performance by market regime."""
        try:
            if 'regime' not in trades_df.columns:
                return {}
                
            regime_stats = trades_df.groupby('regime').agg({
                'pnl': ['sum', 'mean', 'count', 'std'],
                'holding_period': 'mean' if 'holding_period' in trades_df.columns else lambda x: 0
            }).round(2)
            
            # Flatten column names
            regime_stats.columns = ['_'.join(col).strip() for col in regime_stats.columns]
            
            # Calculate win rate per regime
            regime_stats['win_rate'] = trades_df.groupby('regime').apply(
                lambda x: len(x[x['pnl'] > 0]) / len(x) * 100
            ).round(2)
            
            return regime_stats.to_dict('index')
            
        except Exception as e:
            self.logger.error(f"Error analyzing regime performance: {e}")
            return {}
    
    def _analyze_holding_periods(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze performance by holding period."""
        try:
            if 'holding_period' not in trades_df.columns:
                return {}
                
            # Create holding period buckets
            trades_df['holding_bucket'] = pd.cut(
                trades_df['holding_period'], 
                bins=[0, 1, 3, 7, 14, 30, float('inf')],
                labels=['1d', '2-3d', '4-7d', '8-14d', '15-30d', '30d+']
            )
            
            holding_stats = trades_df.groupby('holding_bucket').agg({
                'pnl': ['sum', 'mean', 'count', 'std']
            }).round(2)
            
            # Flatten column names
            holding_stats.columns = ['_'.join(col).strip() for col in holding_stats.columns]
            
            # Calculate win rate per holding period
            holding_stats['win_rate'] = trades_df.groupby('holding_bucket').apply(
                lambda x: len(x[x['pnl'] > 0]) / len(x) * 100
            ).round(2)
            
            return holding_stats.to_dict('index')
            
        except Exception as e:
            self.logger.error(f"Error analyzing holding periods: {e}")
            return {}
    
    def plot_pnl_histogram(self, all_trades: List[Dict], save_path: str = "plots") -> None:
        """Plot histogram of trade PnLs."""
        try:
            if not all_trades:
                self.logger.warning("No trades found for PnL histogram")
                return
                
            trades_df = pd.DataFrame(all_trades)
            if trades_df.empty:
                self.logger.warning("No trades to plot")
                return
            
            # Check if PnL column exists
            if 'pnl' not in trades_df.columns:
                self.logger.error("PnL column not found in trade data for histogram")
                return
            
            # Create plots directory
            os.makedirs(save_path, exist_ok=True)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Trade PnL Analysis', fontsize=16, fontweight='bold')
            
            # 1. PnL Histogram
            axes[0, 0].hist(trades_df['pnl'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].axvline(trades_df['pnl'].mean(), color='red', linestyle='--', 
                              label=f'Mean: ${trades_df["pnl"].mean():.2f}')
            axes[0, 0].axvline(trades_df['pnl'].median(), color='green', linestyle='--', 
                              label=f'Median: ${trades_df["pnl"].median():.2f}')
            axes[0, 0].set_title('Trade PnL Distribution')
            axes[0, 0].set_xlabel('PnL ($)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Cumulative PnL by Trade
            cumulative_pnl = trades_df['pnl'].cumsum()
            axes[0, 1].plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2, color='blue')
            axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            axes[0, 1].set_title('Cumulative PnL by Trade')
            axes[0, 1].set_xlabel('Trade Number')
            axes[0, 1].set_ylabel('Cumulative PnL ($)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. PnL by Pair (if available)
            if 'pair' in trades_df.columns:
                pair_pnl = trades_df.groupby('pair')['pnl'].sum().sort_values()
                axes[1, 0].barh(range(len(pair_pnl)), pair_pnl.values, color=['red' if x < 0 else 'green' for x in pair_pnl.values])
                axes[1, 0].set_yticks(range(len(pair_pnl)))
                axes[1, 0].set_yticklabels(pair_pnl.index)
                axes[1, 0].set_title('Total PnL by Pair')
                axes[1, 0].set_xlabel('PnL ($)')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Win/Loss Distribution
            winning_trades = trades_df[trades_df['pnl'] > 0]['pnl']
            losing_trades = trades_df[trades_df['pnl'] < 0]['pnl']
            
            axes[1, 1].hist(winning_trades, bins=30, alpha=0.7, color='green', label='Winning Trades', density=True)
            axes[1, 1].hist(losing_trades, bins=30, alpha=0.7, color='red', label='Losing Trades', density=True)
            axes[1, 1].set_title('Win/Loss Distribution (Normalized)')
            axes[1, 1].set_xlabel('PnL ($)')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/pnl_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"PnL analysis plots saved to {save_path}/pnl_analysis.png")
            
        except Exception as e:
            self.logger.error(f"Error plotting PnL histogram: {e}")
    
    def print_worst_trades(self, all_trades: List[Dict], top_n: int = 10) -> None:
        """Print the worst performing trades."""
        try:
            if not all_trades:
                self.logger.warning("No trades found for worst trades analysis")
                return
                
            trades_df = pd.DataFrame(all_trades)
            if trades_df.empty:
                self.logger.warning("No trades to analyze")
                return
            
            # Check if PnL column exists
            if 'pnl' not in trades_df.columns:
                self.logger.error("PnL column not found in trade data for worst trades analysis")
                return
            
            # Get worst trades
            worst_trades = trades_df.nsmallest(top_n, 'pnl')
            
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"WORST {top_n} TRADES ANALYSIS")
            self.logger.info(f"{'='*80}")
            
            for idx, trade in worst_trades.iterrows():
                self.logger.info(f"Trade {idx}: {trade.get('pair', 'Unknown')} | PnL: ${trade['pnl']:.2f} | "
                               f"Entry: {trade.get('entry_date', 'Unknown')} | Exit: {trade.get('exit_date', 'Unknown')}")
                if 'holding_period' in trade:
                    self.logger.info(f"  Holding Period: {trade['holding_period']} days")
                if 'regime' in trade:
                    self.logger.info(f"  Market Regime: {trade['regime']}")
            
            # Summary statistics
            total_loss_from_worst = worst_trades['pnl'].sum()
            avg_loss_from_worst = worst_trades['pnl'].mean()
            
            self.logger.info(f"\nWorst {top_n} trades summary:")
            self.logger.info(f"Total loss from worst {top_n} trades: ${total_loss_from_worst:.2f}")
            self.logger.info(f"Average loss from worst {top_n} trades: ${avg_loss_from_worst:.2f}")
            self.logger.info(f"Percentage of total loss: {(total_loss_from_worst / trades_df['pnl'].sum() * 100):.1f}%")
            
        except Exception as e:
            self.logger.error(f"Error printing worst trades: {e}")
    
    def generate_diagnostic_report(self, all_trades: List[Dict], save_path: str = "plots") -> Dict:
        """Generate comprehensive diagnostic report."""
        try:
            # Analyze trade performance
            analysis = self.analyze_trade_performance(all_trades)
            
            # Generate plots
            self.plot_pnl_histogram(all_trades, save_path)
            
            # Print worst trades
            self.print_worst_trades(all_trades)
            
            # Convert trades to DataFrame for regime analysis
            trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
            
            # Analyze performance by regime
            regime_analysis = self.analyze_by_regime(trades_df)
            
            # Get loss decomposition
            loss_decomposition = self.get_loss_decomposition(trades_df)
            
            # Log comprehensive summary
            if analysis and 'summary' in analysis:
                summary = analysis['summary']
                self.logger.info(f"\n{'='*80}")
                self.logger.info("TRADE DIAGNOSTICS SUMMARY")
                self.logger.info(f"{'='*80}")
                self.logger.info(f"Total Trades: {summary['total_trades']}")
                self.logger.info(f"Win Rate: {summary['win_rate']:.2%}")
                self.logger.info(f"Profit Factor: {summary['profit_factor']:.2f}")
                self.logger.info(f"Average PnL: ${summary['avg_pnl']:.2f}")
                self.logger.info(f"PnL Std Dev: ${summary['std_pnl']:.2f}")
                self.logger.info(f"Max Profit: ${summary['max_profit']:.2f}")
                self.logger.info(f"Max Loss: ${summary['max_loss']:.2f}")
            
            # Add regime and loss decomposition to analysis
            analysis['regime_analysis'] = regime_analysis
            analysis['loss_decomposition'] = loss_decomposition
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error generating diagnostic report: {e}")
            return {}

    def calculate_trade_stats(self, trades: List) -> Dict:
        """Calculate comprehensive trade statistics for a list of trades."""
        try:
            if not trades:
                return {}
                
            # Convert trades to DataFrame
            trades_df = pd.DataFrame([trade.__dict__ for trade in trades])
            if trades_df.empty:
                return {}
            
            # Basic statistics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            
            # PnL statistics
            total_pnl = trades_df['pnl'].sum()
            avg_pnl = trades_df['pnl'].mean()
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Profit factor
            total_wins = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            total_losses = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Time-based metrics
            if 'entry_date' in trades_df.columns and 'exit_date' in trades_df.columns:
                trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
                trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
                
                # Calculate holding periods
                trades_df['holding_period'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days
                avg_holding_period = trades_df['holding_period'].mean()
                
                # Calculate time span
                start_date = trades_df['entry_date'].min()
                end_date = trades_df['exit_date'].max()
                total_days = (end_date - start_date).days
            else:
                avg_holding_period = 0
                total_days = 252  # Default to one year
            
            return {
                'pnl_sum': total_pnl,
                'pnl_mean': avg_pnl,
                'pnl_count': total_trades,
                'win_rate': win_rate * 100,  # Convert to percentage
                'profit_factor': profit_factor,
                'avg_holding_period': avg_holding_period,
                'total_days': total_days,
                'start_date': start_date if 'entry_date' in trades_df.columns else None,
                'end_date': end_date if 'exit_date' in trades_df.columns else None
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating trade stats: {e}")
            return {}
    
    def calculate_performance_metrics(self, daily_returns: pd.Series) -> Dict:
        """Calculate performance metrics from daily returns."""
        try:
            if daily_returns.empty:
                return {}
            
            # Ensure we have a Series
            if isinstance(daily_returns, pd.DataFrame):
                daily_returns = daily_returns.iloc[:, 0]
            
            # Basic metrics
            annualized_return = daily_returns.mean() * 252
            annualized_volatility = daily_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
            
            # Calculate max drawdown
            cumulative_returns = (1 + daily_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            return {
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def calculate_pair_equity_curve(self, trades: List, initial_capital: float = 1000000) -> pd.Series:
        """Calculate equity curve for a specific pair from trade data."""
        try:
            if not trades:
                return pd.Series()
            
            # Convert trades to DataFrame
            trades_df = pd.DataFrame([trade.__dict__ for trade in trades])
            if trades_df.empty:
                return pd.Series()
            
            # Ensure dates are datetime
            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
            
            # Create date range
            start_date = trades_df['entry_date'].min()
            end_date = trades_df['exit_date'].max()
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Initialize equity curve
            equity_curve = pd.Series(index=date_range, data=initial_capital)
            
            # Add PnL for each trade on its exit date
            for _, trade in trades_df.iterrows():
                if pd.notna(trade['exit_date']) and pd.notna(trade['pnl']):
                    equity_curve.loc[trade['exit_date']:] += trade['pnl']
            
            return equity_curve
            
        except Exception as e:
            self.logger.error(f"Error calculating pair equity curve: {e}")
            return pd.Series()
    
    def analyze_by_pair(self, backtest_results: Dict) -> Dict:
        """Analyze performance by pair with comprehensive metrics."""
        try:
            pair_performance = {}
            
            for pair, results in backtest_results.items():
                trades = results.get('trades', [])
                if not trades:
                    continue
                
                # Get basic trade statistics
                pair_stats = self.calculate_trade_stats(trades)
                
                # Calculate equity curve for this pair
                initial_capital = getattr(self.config, 'initial_capital', 1000000)
                pair_equity_curve = self.calculate_pair_equity_curve(trades, initial_capital)
                
                # Calculate performance metrics from equity curve
                if not pair_equity_curve.empty:
                    daily_returns = pair_equity_curve.pct_change().fillna(0)
                    performance_metrics = self.calculate_performance_metrics(daily_returns)
                    
                    # Add performance metrics to pair stats
                    pair_stats.update({
                        'annualized_return_pct': performance_metrics.get('annualized_return', 0) * 100,
                        'annualized_volatility_pct': performance_metrics.get('annualized_volatility', 0) * 100,
                        'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
                        'max_drawdown_pct': performance_metrics.get('max_drawdown', 0) * 100
                    })
                else:
                    # Fallback: calculate from trade PnL if no equity curve
                    if pair_stats.get('total_days', 0) > 0:
                        total_return = pair_stats.get('pnl_sum', 0) / initial_capital
                        annualized_return = (1 + total_return) ** (252 / pair_stats['total_days']) - 1
                        
                        # Estimate volatility from trade PnL
                        trades_df = pd.DataFrame([trade.__dict__ for trade in trades])
                        pnl_std = trades_df['pnl'].std() if 'pnl' in trades_df.columns else 0
                        estimated_vol = pnl_std / initial_capital * np.sqrt(252 / pair_stats['total_days'])
                        
                        pair_stats.update({
                            'annualized_return_pct': annualized_return * 100,
                            'annualized_volatility_pct': estimated_vol * 100,
                            'sharpe_ratio': annualized_return / estimated_vol if estimated_vol > 0 else 0,
                            'max_drawdown_pct': 0  # Cannot calculate without equity curve
                        })
                    else:
                        pair_stats.update({
                            'annualized_return_pct': 0,
                            'annualized_volatility_pct': 0,
                            'sharpe_ratio': 0,
                            'max_drawdown_pct': 0
                        })
                
                pair_performance[pair] = pair_stats
            
            return pair_performance
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance by pair: {e}")
            return {}
    
    def analyze_by_regime(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze performance by market regime with comprehensive metrics."""
        try:
            if 'regime' not in trades_df.columns:
                return {}
            
            regime_performance = {}
            
            for regime in trades_df['regime'].unique():
                regime_trades = trades_df[trades_df['regime'] == regime]
                
                if regime_trades.empty:
                    continue
                
                # Basic statistics
                total_trades = len(regime_trades)
                winning_trades = len(regime_trades[regime_trades['pnl'] > 0])
                total_pnl = regime_trades['pnl'].sum()
                avg_pnl = regime_trades['pnl'].mean()
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                
                # Profit factor
                total_wins = regime_trades[regime_trades['pnl'] > 0]['pnl'].sum()
                total_losses = abs(regime_trades[regime_trades['pnl'] < 0]['pnl'].sum())
                profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
                
                # Time-based metrics
                if 'entry_date' in regime_trades.columns and 'exit_date' in regime_trades.columns:
                    regime_trades['entry_date'] = pd.to_datetime(regime_trades['entry_date'])
                    regime_trades['exit_date'] = pd.to_datetime(regime_trades['exit_date'])
                    
                    start_date = regime_trades['entry_date'].min()
                    end_date = regime_trades['exit_date'].max()
                    total_days = (end_date - start_date).days
                    
                    # Calculate holding periods
                    regime_trades['holding_period'] = (regime_trades['exit_date'] - regime_trades['entry_date']).dt.days
                    avg_holding_period = regime_trades['holding_period'].mean()
                else:
                    total_days = 252
                    avg_holding_period = 0
                
                # Calculate performance metrics
                initial_capital = getattr(self.config, 'initial_capital', 1000000)
                if total_days > 0:
                    total_return = total_pnl / initial_capital
                    annualized_return = (1 + total_return) ** (252 / total_days) - 1
                    
                    # Estimate volatility from trade PnL
                    pnl_std = regime_trades['pnl'].std()
                    estimated_vol = pnl_std / initial_capital * np.sqrt(252 / total_days)
                    
                    sharpe_ratio = annualized_return / estimated_vol if estimated_vol > 0 else 0
                else:
                    annualized_return = 0
                    estimated_vol = 0
                    sharpe_ratio = 0
                
                regime_performance[regime] = {
                    'pnl_sum': total_pnl,
                    'pnl_mean': avg_pnl,
                    'pnl_count': total_trades,
                    'win_rate': win_rate * 100,
                    'profit_factor': profit_factor,
                    'avg_holding_period': avg_holding_period,
                    'annualized_return_pct': annualized_return * 100,
                    'annualized_volatility_pct': estimated_vol * 100,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown_pct': 0  # Cannot calculate without equity curve per regime
                }
            
            return regime_performance
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance by regime: {e}")
            return {}

    def get_all_trades_as_df(self, backtest_results: Dict) -> pd.DataFrame:
        """Get all trades as a single DataFrame."""
        try:
            if not backtest_results or 'trades' not in backtest_results:
                self.logger.warning("No trades found in backtest results")
                return pd.DataFrame()
                
            trades_df = pd.DataFrame(backtest_results['trades'])
            if trades_df.empty:
                self.logger.warning("No trades to analyze")
                return pd.DataFrame()
            
            return trades_df
            
        except Exception as e:
            self.logger.error(f"Error getting all trades as DataFrame: {e}")
            return pd.DataFrame()

    def get_loss_decomposition(self, trades_df: pd.DataFrame) -> Dict:
        """Get loss decomposition."""
        try:
            # Check if PnL column exists
            if 'pnl' not in trades_df.columns:
                self.logger.error("PnL column not found in trade data for loss decomposition")
                return {}
            
            # Filter losing trades
            losing_trades = trades_df[trades_df['pnl'] < 0].copy()
            
            if losing_trades.empty:
                return {}
            
            # Calculate total loss
            total_loss = losing_trades['pnl'].sum()
            
            # Loss by pair (if pair column exists)
            loss_by_pair = pd.DataFrame()
            if 'pair' in losing_trades.columns:
                loss_by_pair = losing_trades.groupby('pair')['pnl'].agg(['sum', 'count', 'mean']).round(2)
                loss_by_pair['contribution_pct'] = (loss_by_pair['sum'] / total_loss * 100).round(2)
            
            # Loss by regime (if available)
            loss_by_regime = {}
            if 'regime' in losing_trades.columns:
                loss_by_regime = losing_trades.groupby('regime')['pnl'].agg(['sum', 'count', 'mean']).round(2)
                loss_by_regime['contribution_pct'] = (loss_by_regime['sum'] / total_loss * 100).round(2)
            
            # Loss by holding period buckets
            loss_by_holding = pd.DataFrame()
            if 'holding_period' in losing_trades.columns:
                losing_trades['holding_bucket'] = pd.cut(
                    losing_trades['holding_period'], 
                    bins=[0, 1, 3, 7, 14, 30, float('inf')],
                    labels=['1d', '2-3d', '4-7d', '8-14d', '15-30d', '30d+']
                )
                loss_by_holding = losing_trades.groupby('holding_bucket')['pnl'].agg(['sum', 'count', 'mean']).round(2)
                loss_by_holding['contribution_pct'] = (loss_by_holding['sum'] / total_loss * 100).round(2)
            
            return {
                'total_loss': total_loss,
                'loss_by_pair': loss_by_pair,
                'loss_by_regime': loss_by_regime,
                'loss_by_holding_period': loss_by_holding,
                'avg_loss': losing_trades['pnl'].mean(),
                'median_loss': losing_trades['pnl'].median(),
                'loss_std': losing_trades['pnl'].std()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting loss decomposition: {e}")
            return {} 