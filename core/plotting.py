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

    def plot_pair_analysis(self, pair_data: pd.DataFrame, pair_metrics: Dict,
                          save_path: Optional[str] = None) -> None:
        """
        Plot pair analysis results.
        
        Args:
            pair_data: DataFrame with price data for the pair
            pair_metrics: Dictionary containing pair metrics
            save_path: Optional path to save the plot
        """
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
            
            # Plot prices
            pair_data.plot(ax=axes[0])
            axes[0].set_title('Price Series')
            axes[0].set_ylabel('Price')
            axes[0].legend()
            
            # Plot spread
            spread = pair_metrics['spread']
            spread.plot(ax=axes[1])
            axes[1].set_title('Spread')
            axes[1].set_ylabel('Spread')
            
            # Plot Z-score
            zscore = pair_metrics['zscore']
            zscore.plot(ax=axes[2])
            axes[2].axhline(y=2, color='r', linestyle='--', alpha=0.5)
            axes[2].axhline(y=-2, color='r', linestyle='--', alpha=0.5)
            axes[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            axes[2].set_title('Z-score')
            axes[2].set_ylabel('Z-score')
            
            # Add metrics to title
            metrics_text = (
                f"Correlation: {pair_metrics['correlation']:.2f}\n"
                f"Cointegration p-value: {pair_metrics['coint_pvalue']:.2f}\n"
                f"Spread Stability: {pair_metrics['spread_stability']:.2f}\n"
                f"Z-score Volatility: {pair_metrics['zscore_volatility']:.2f}"
            )
            fig.suptitle(metrics_text, y=0.95)
            
            # Adjust layout and save
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting pair analysis: {str(e)}")

    def plot_backtest_results(self, backtest_results: Dict[str, Dict],
                            save_dir: Optional[str] = None) -> None:
        """
        Plot backtest results for all pairs.
        
        Args:
            backtest_results: Dictionary of backtest results for each pair
            save_dir: Optional directory to save the plots
        """
        try:
            # Create directory if needed
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Plot equity curves
            self._plot_equity_curves(backtest_results, save_dir)
            
            # Plot individual pair results
            for pair, results in backtest_results.items():
                self._plot_pair_backtest(pair, results, save_dir)
            
        except Exception as e:
            self.logger.error(f"Error plotting backtest results: {str(e)}")

    def _plot_equity_curves(self, backtest_results: Dict[str, Dict],
                           save_dir: Optional[str] = None) -> None:
        """Plot equity curves for all pairs."""
        try:
            # Combine equity curves
            equity_curves = pd.DataFrame({
                pair: results['equity']
                for pair, results in backtest_results.items()
            })
            
            # Calculate portfolio equity
            portfolio_equity = equity_curves.sum(axis=1)
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            # Plot individual equity curves
            for pair in equity_curves.columns:
                plt.plot(equity_curves[pair], label=pair, alpha=0.3)
            
            # Plot portfolio equity
            plt.plot(portfolio_equity, label='Portfolio', linewidth=2)
            
            # Add labels and legend
            plt.title('Equity Curves')
            plt.xlabel('Date')
            plt.ylabel('Equity')
            plt.legend()
            plt.grid(True)
            
            # Save plot
            if save_dir:
                plt.savefig(os.path.join(save_dir, 'equity_curves.png'))
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting equity curves: {str(e)}")

    def _plot_pair_backtest(self, pair: str, results: Dict,
                           save_dir: Optional[str] = None) -> None:
        """Plot backtest results for a single pair."""
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
            
            # Plot equity curve
            results['equity'].plot(ax=axes[0])
            axes[0].set_title(f'Equity Curve - {pair}')
            axes[0].set_ylabel('Equity')
            
            # Plot positions
            results['positions'].plot(ax=axes[1])
            axes[1].set_title('Positions')
            axes[1].set_ylabel('Position')
            
            # Plot PnL
            results['pnl'].plot(ax=axes[2])
            axes[2].set_title('PnL')
            axes[2].set_ylabel('PnL')
            
            # Add metrics to title
            metrics = results['metrics']
            metrics_text = (
                f"Total Return: {metrics['total_return']:.2%}\n"
                f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
                f"Win Rate: {metrics['win_rate']:.2%}"
            )
            fig.suptitle(metrics_text, y=0.95)
            
            # Adjust layout and save
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'backtest_{pair}.png'))
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting pair backtest: {str(e)}")

    def plot_performance_metrics(self, metrics: Dict,
                               save_path: Optional[str] = None) -> None:
        """
        Plot performance metrics.
        
        Args:
            metrics: Dictionary containing performance metrics
            save_path: Optional path to save the plot
        """
        try:
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Plot metrics
            metric_names = ['Total Return', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']
            metric_values = [
                metrics['total_return'],
                metrics['sharpe_ratio'],
                metrics['sortino_ratio'],
                metrics['calmar_ratio']
            ]
            
            # Create bar plot
            bars = plt.bar(metric_names, metric_values)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')
            
            # Add labels and title
            plt.title('Performance Metrics')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            if save_path:
                plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting performance metrics: {str(e)}") 