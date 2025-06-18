"""
Plotting module for generating visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import logging
from datetime import datetime
import os

class PlotGenerator:
    """Class for generating plots and visualizations."""
    
    def __init__(self, config):
        """Initialize the plot generator with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set style
        sns.set_theme()

    def plot_pair_analysis(self, backtest_results: Dict, pair_metrics: Dict) -> None:
        """
        Plot pair analysis results.
        
        Args:
            backtest_results: Dictionary of backtest results for each pair
            pair_metrics: Dictionary containing pair metrics
        """
        try:
            # Create directory for plots
            plots_dir = "plots"
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
            
            # Plot each pair
            for pair, results in backtest_results.items():
                self._plot_single_pair_analysis(pair, results, pair_metrics.get(pair, {}), plots_dir)
            
        except Exception as e:
            self.logger.error(f"Error plotting pair analysis: {str(e)}")

    def _plot_single_pair_analysis(self, pair: tuple, results: Dict, metrics: Dict, save_dir: str) -> None:
        """Plot analysis for a single pair."""
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Equity curve
            if 'equity_curve' in results and results['equity_curve'] is not None:
                results['equity_curve'].plot(ax=axes[0, 0])
                axes[0, 0].set_title(f'Equity Curve - {pair}')
                axes[0, 0].set_ylabel('Equity')
                axes[0, 0].grid(True)
            
            # Plot 2: Daily returns
            if 'daily_returns' in results and results['daily_returns'] is not None:
                results['daily_returns'].plot(ax=axes[0, 1])
                axes[0, 1].set_title(f'Daily Returns - {pair}')
                axes[0, 1].set_ylabel('Returns')
                axes[0, 1].grid(True)
            
            # Plot 3: Cumulative returns
            if 'daily_returns' in results and results['daily_returns'] is not None:
                cum_returns = (1 + results['daily_returns']).cumprod()
                cum_returns.plot(ax=axes[1, 0])
                axes[1, 0].set_title(f'Cumulative Returns - {pair}')
                axes[1, 0].set_ylabel('Cumulative Returns')
                axes[1, 0].grid(True)
            
            # Plot 4: Drawdown
            if 'equity_curve' in results and results['equity_curve'] is not None:
                equity = results['equity_curve']
                rolling_max = equity.cummax()
                drawdown = (equity - rolling_max) / rolling_max
                drawdown.plot(ax=axes[1, 1])
                axes[1, 1].set_title(f'Drawdown - {pair}')
                axes[1, 1].set_ylabel('Drawdown')
                axes[1, 1].grid(True)
            
            # Add metrics text
            if metrics:
                metrics_text = (
                    f"Correlation: {metrics.get('correlation', 'N/A'):.3f}\n"
                    f"Cointegration p-value: {metrics.get('coint_pvalue', 'N/A'):.3f}\n"
                    f"Spread Stability: {metrics.get('spread_stability', 'N/A'):.3f}\n"
                    f"Z-score Volatility: {metrics.get('zscore_volatility', 'N/A'):.3f}\n"
                    f"Score: {metrics.get('score', 'N/A'):.3f}"
                )
                fig.suptitle(metrics_text, y=0.95)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'pair_analysis_{pair[0]}_{pair[1]}.png'))
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting single pair analysis for {pair}: {str(e)}")

    def plot_backtest_results(self, backtest_results: Dict) -> None:
        """
        Plot backtest results for all pairs.
        
        Args:
            backtest_results: Dictionary of backtest results for each pair
        """
        try:
            # Create directory for plots
            plots_dir = "plots"
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
            
            # Plot equity curves
            self._plot_equity_curves(backtest_results, plots_dir)
            
            # Plot individual pair results
            for pair, results in backtest_results.items():
                self._plot_pair_backtest(pair, results, plots_dir)
            
        except Exception as e:
            self.logger.error(f"Error plotting backtest results: {str(e)}")

    def _plot_equity_curves(self, backtest_results: Dict, save_dir: str) -> None:
        """Plot equity curves for all pairs."""
        try:
            # Combine equity curves
            equity_curves = {}
            for pair, results in backtest_results.items():
                if 'equity_curve' in results and results['equity_curve'] is not None:
                    equity_curves[pair] = results['equity_curve']
            
            if not equity_curves:
                self.logger.warning("No equity curves found for plotting")
                return
            
            equity_df = pd.DataFrame(equity_curves)
            
            # Calculate portfolio equity
            portfolio_equity = equity_df.sum(axis=1)
            
            # Create plot
            plt.figure(figsize=(15, 8))
            
            # Plot individual equity curves
            for pair in equity_df.columns:
                plt.plot(equity_df[pair], label=f'{pair[0]}-{pair[1]}', alpha=0.7)
            
            # Plot portfolio equity
            plt.plot(portfolio_equity, label='Portfolio', linewidth=3, color='black')
            
            # Add labels and legend
            plt.title('Equity Curves - All Pairs', fontsize=14, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Equity ($)', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'equity_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting equity curves: {str(e)}")

    def _plot_pair_backtest(self, pair: tuple, results: Dict, save_dir: str) -> None:
        """Plot backtest results for a single pair."""
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Equity curve
            if 'equity_curve' in results and results['equity_curve'] is not None:
                results['equity_curve'].plot(ax=axes[0, 0])
                axes[0, 0].set_title(f'Equity Curve - {pair[0]}-{pair[1]}')
                axes[0, 0].set_ylabel('Equity ($)')
                axes[0, 0].grid(True)
            
            # Plot 2: Daily returns
            if 'daily_returns' in results and results['daily_returns'] is not None:
                results['daily_returns'].plot(ax=axes[0, 1])
                axes[0, 1].set_title(f'Daily Returns - {pair[0]}-{pair[1]}')
                axes[0, 1].set_ylabel('Returns')
                axes[0, 1].grid(True)
            
            # Plot 3: Cumulative returns
            if 'daily_returns' in results and results['daily_returns'] is not None:
                cum_returns = (1 + results['daily_returns']).cumprod()
                cum_returns.plot(ax=axes[1, 0])
                axes[1, 0].set_title(f'Cumulative Returns - {pair[0]}-{pair[1]}')
                axes[1, 0].set_ylabel('Cumulative Returns')
                axes[1, 0].grid(True)
            
            # Plot 4: Drawdown
            if 'equity_curve' in results and results['equity_curve'] is not None:
                equity = results['equity_curve']
                rolling_max = equity.cummax()
                drawdown = (equity - rolling_max) / rolling_max
                drawdown.plot(ax=axes[1, 1])
                axes[1, 1].set_title(f'Drawdown - {pair[0]}-{pair[1]}')
                axes[1, 1].set_ylabel('Drawdown')
                axes[1, 1].grid(True)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'backtest_{pair[0]}_{pair[1]}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting pair backtest for {pair}: {str(e)}")

    def plot_performance_metrics(self, portfolio_metrics: Dict) -> None:
        """
        Plot performance metrics.
        
        Args:
            portfolio_metrics: Dictionary containing performance metrics
        """
        try:
            # Create directory for plots
            plots_dir = "plots"
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Key metrics bar chart
            if portfolio_metrics:
                key_metrics = ['annualized_return', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio']
                metric_names = ['Annual Return', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']
                metric_values = [portfolio_metrics.get(metric, 0) for metric in key_metrics]
                
                bars = axes[0, 0].bar(metric_names, metric_values, color=['blue', 'green', 'orange', 'red'])
                axes[0, 0].set_title('Key Performance Metrics')
                axes[0, 0].set_ylabel('Value')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.3f}',
                                   ha='center', va='bottom')
            
            # Plot 2: Risk metrics
            if portfolio_metrics:
                risk_metrics = ['annualized_volatility', 'max_drawdown']
                risk_names = ['Annual Volatility', 'Max Drawdown']
                risk_values = [portfolio_metrics.get(metric, 0) for metric in risk_metrics]
                
                bars = axes[0, 1].bar(risk_names, risk_values, color=['purple', 'brown'])
                axes[0, 1].set_title('Risk Metrics')
                axes[0, 1].set_ylabel('Value')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.3f}',
                                   ha='center', va='bottom')
            
            # Plot 3: Trading statistics
            if portfolio_metrics:
                trade_metrics = ['win_rate', 'total_trades', 'winning_trades']
                trade_names = ['Win Rate', 'Total Trades', 'Winning Trades']
                trade_values = [portfolio_metrics.get(metric, 0) for metric in trade_metrics]
                
                bars = axes[1, 0].bar(trade_names, trade_values, color=['cyan', 'magenta', 'yellow'])
                axes[1, 0].set_title('Trading Statistics')
                axes[1, 0].set_ylabel('Value')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.3f}',
                                   ha='center', va='bottom')
            
            # Plot 4: Summary text
            axes[1, 1].axis('off')
            if portfolio_metrics:
                summary_text = (
                    f"Portfolio Performance Summary\n\n"
                    f"Total Return: {portfolio_metrics.get('total_return', 0):.2%}\n"
                    f"Annualized Return: {portfolio_metrics.get('annualized_return', 0):.2%}\n"
                    f"Sharpe Ratio: {portfolio_metrics.get('sharpe_ratio', 0):.2f}\n"
                    f"Max Drawdown: {portfolio_metrics.get('max_drawdown', 0):.2%}\n"
                    f"Win Rate: {portfolio_metrics.get('win_rate', 0):.2%}\n"
                    f"Total Trades: {portfolio_metrics.get('total_trades', 0)}\n"
                    f"Winning Trades: {portfolio_metrics.get('winning_trades', 0)}\n"
                    f"Avg Holding Period: {portfolio_metrics.get('avg_holding_period', 0):.1f} days"
                )
                axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                               fontsize=12, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting performance metrics: {str(e)}") 