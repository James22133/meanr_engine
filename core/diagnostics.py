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

    def analyze_performance_by_regime(self, backtest_results: Dict) -> Dict:
        """Aggregate PnL statistics grouped by market regime."""
        try:
            trades_list = []
            for res in backtest_results.values():
                if 'trades' in res and res['trades']:
                    trades_list.append(pd.DataFrame(res['trades']))

            if not trades_list:
                return {}

            trades_df = pd.concat(trades_list, ignore_index=True)
            if 'regime' not in trades_df.columns and 'entry_regime' in trades_df.columns:
                trades_df['regime'] = trades_df['entry_regime']

            if 'regime' not in trades_df.columns:
                return {}

            grouped = trades_df.groupby('regime')['pnl']
            summary = grouped.agg(['sum', 'mean', 'count']).round(2)
            summary['win_rate'] = grouped.apply(lambda x: (x > 0).mean() * 100).round(2)
            return summary.to_dict('index')

        except Exception as e:
            self.logger.error(f"Error analyzing performance by regime: {e}")
            return {}

    def plot_pnl_by_regime(self, backtest_results: Dict, save_path: str = "plots") -> None:
        """Plot total PnL by regime as a bar chart."""
        try:
            regime_stats = self.analyze_performance_by_regime(backtest_results)
            if not regime_stats:
                return

            os.makedirs(save_path, exist_ok=True)
            regimes = list(regime_stats.keys())
            pnls = [stats['sum'] for stats in regime_stats.values()]
            plt.figure(figsize=(8, 5))
            plt.bar([str(r) for r in regimes], pnls, color='skyblue', edgecolor='black')
            plt.xlabel('Regime')
            plt.ylabel('Total PnL')
            plt.title('PnL by Regime')
            plt.tight_layout()
            plt.savefig(f"{save_path}/pnl_by_regime.png", dpi=300)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error plotting PnL by regime: {e}")

    def summarize_worst_drawdowns(self, equity_curve: pd.Series, regimes: pd.Series, top_n: int = 5) -> List[Dict]:
        """Return worst drawdown days along with the prevailing regime."""
        if equity_curve.empty or regimes is None or regimes.empty:
            return []
        drawdown = (equity_curve - equity_curve.cummax()) / equity_curve.cummax()
        worst = drawdown.nsmallest(top_n)
        summary = []
        for date, dd in worst.items():
            summary.append({'date': date, 'drawdown': float(dd), 'regime': int(regimes.get(date, -1))})
        return summary
        
    def analyze_trade_performance(self, backtest_results: Dict) -> Dict:
        """Comprehensive analysis of trade performance."""
        try:
            if not backtest_results or 'trades' not in backtest_results:
                self.logger.warning("No trades found in backtest results")
                return {}
                
            trades_df = pd.DataFrame(backtest_results['trades'])
            if trades_df.empty:
                self.logger.warning("No trades to analyze")
                return {}
                
            # Calculate basic statistics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            
            # Calculate PnL statistics
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
    
    def plot_pnl_histogram(self, backtest_results: Dict, save_path: str = "plots") -> None:
        """Plot histogram of trade PnLs."""
        try:
            if not backtest_results or 'trades' not in backtest_results:
                self.logger.warning("No trades found for PnL histogram")
                return
                
            trades_df = pd.DataFrame(backtest_results['trades'])
            if trades_df.empty:
                self.logger.warning("No trades to plot")
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
    
    def print_worst_trades(self, backtest_results: Dict, top_n: int = 10) -> None:
        """Print the worst performing trades."""
        try:
            if not backtest_results or 'trades' not in backtest_results:
                self.logger.warning("No trades found for worst trades analysis")
                return
                
            trades_df = pd.DataFrame(backtest_results['trades'])
            if trades_df.empty:
                self.logger.warning("No trades to analyze")
                return
            
            # Get worst trades
            worst_trades = trades_df.nsmallest(top_n, 'pnl')
            
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"WORST {top_n} TRADES ANALYSIS")
            self.logger.info(f"{'='*80}")
            
            for idx, trade in worst_trades.iterrows():
                self.logger.info(f"Trade {idx}: {trade['pair']} | PnL: ${trade['pnl']:.2f} | "
                               f"Entry: {trade['entry_date']} | Exit: {trade['exit_date']}")
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
    
    def generate_diagnostic_report(
        self,
        backtest_results: Dict,
        save_path: str = "plots",
        equity_curve: Optional[pd.Series] = None,
        regimes: Optional[pd.Series] = None,
    ) -> Dict:
        """Generate comprehensive diagnostic report."""
        try:
            # Analyze trade performance
            analysis = self.analyze_trade_performance(backtest_results)
            
            # Generate plots
            self.plot_pnl_histogram(backtest_results, save_path)

            # Optional regime analysis
            if getattr(self.config, 'diagnostics', {}).get('analyze_by_regime', False):
                self.plot_pnl_by_regime(backtest_results, save_path)

            # Print worst trades
            self.print_worst_trades(backtest_results)

            if equity_curve is not None and regimes is not None:
                dd_info = self.summarize_worst_drawdowns(equity_curve, regimes)
                for item in dd_info:
                    self.logger.info(
                        f"Drawdown {item['drawdown']:.2%} on {item['date'].date()} in regime {item['regime']}"
                    )
            
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
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error generating diagnostic report: {e}")
            return {} 