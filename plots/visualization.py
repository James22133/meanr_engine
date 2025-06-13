import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import os
import logging

def plot_equity_curve(equity_curve: pd.Series, save_path: str = None):
    """Plot equity curve with drawdown overlay."""
    if equity_curve.empty or not isinstance(equity_curve.index, pd.DatetimeIndex):
        logging.warning("Equity curve plot skipped: empty or invalid index.")
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot equity curve
    equity_curve.plot(ax=ax1, label='Equity')
    ax1.set_title('Equity Curve')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True)
    
    # Calculate and plot drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    drawdown.plot(ax=ax2, color='red', label='Drawdown')
    ax2.set_title('Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'equity_curve.png'))
    plt.close()

def plot_monthly_returns_heatmap(returns: pd.Series, save_path: str = None):
    """Plot monthly returns heatmap."""
    if returns.empty or not isinstance(returns.index, pd.DatetimeIndex):
        logging.warning("Monthly returns heatmap skipped: empty or invalid index.")
        return
    # Resample to monthly returns using 'ME' instead of 'M'
    monthly_returns = returns.resample('ME').sum()
    
    # Create year-month matrix
    returns_matrix = monthly_returns.to_frame()
    returns_matrix.index = pd.MultiIndex.from_arrays([
        returns_matrix.index.year,
        returns_matrix.index.month
    ])
    returns_matrix = returns_matrix.unstack()
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(returns_matrix, cmap='RdYlGn', center=0, annot=True, fmt='.2%')
    plt.title('Monthly Returns Heatmap')
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'monthly_returns_heatmap.png'))
    plt.close()

def plot_trade_return_histogram(trades: List, save_path: str = None):
    """Plot histogram of trade returns."""
    returns = [t.pnl for t in trades if t.pnl is not None]
    if not returns:
        logging.warning("Trade return histogram skipped: no trade returns.")
        return
    
    plt.figure(figsize=(10, 6))
    sns.histplot(returns, bins=30, kde=True)
    plt.title('Trade Return Distribution')
    plt.xlabel('Return ($)')
    plt.ylabel('Frequency')
    plt.grid(True)
    if save_path:
        plt.savefig(os.path.join(save_path, 'trade_returns_histogram.png'))
    plt.close()

def plot_sharpe_per_pair(pair_results: List[Dict], save_path: str = None):
    """Plot Sharpe ratio for each pair."""
    if not pair_results:
        logging.warning("Sharpe ratio plot skipped: no pair results.")
        return
    pairs = [f"{r['pair'][0]}-{r['pair'][1]}" for r in pair_results]
    sharpes = [r['metrics']['sharpe_ratio'] for r in pair_results]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=pairs, y=sharpes)
    plt.title('Sharpe Ratio by Pair')
    plt.xlabel('Pair')
    plt.ylabel('Sharpe Ratio')
    plt.xticks(rotation=45)
    plt.grid(True)
    if save_path:
        plt.savefig(os.path.join(save_path, 'sharpe_per_pair.png'))
    plt.close()

def generate_all_plots(backtest, pair_results, save_path="reports/"):
    """Generate all standard plots."""
    os.makedirs(save_path, exist_ok=True)
    
    # Equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(backtest.equity_curve)
    plt.title("Equity Curve")
    plt.savefig(f"{save_path}/equity_curve.png")
    plt.close()
    
    # Drawdown
    plt.figure(figsize=(12, 6))
    drawdown = (backtest.equity_curve - backtest.equity_curve.cummax()) / backtest.equity_curve.cummax()
    plt.plot(drawdown)
    plt.title("Drawdown")
    plt.savefig(f"{save_path}/drawdown.png")
    plt.close()
    
    # Monthly returns heatmap
    monthly_returns = backtest.daily_returns.resample('M').sum()
    monthly_returns.index = pd.to_datetime(monthly_returns.index)
    returns_matrix = monthly_returns.to_frame(name='return')
    returns_matrix['year'] = returns_matrix.index.year
    returns_matrix['month'] = returns_matrix.index.month
    returns_pivot = returns_matrix.pivot(index='year', columns='month', values='return')
    plt.figure(figsize=(12, 8))
    sns.heatmap(returns_pivot, annot=True, fmt='.2%', cmap='RdYlGn')
    plt.title("Monthly Returns Heatmap")
    plt.savefig(f"{save_path}/monthly_returns.png")
    plt.close()

def generate_pair_level_plots(pair_results: List[Dict[str, Any]], save_path: str = "reports/"):
    """Generate pair-specific plots."""
    os.makedirs(save_path, exist_ok=True)
    
    # Pair-level equity curves
    plt.figure(figsize=(15, 8))
    for result in pair_results:
        pair = result['pair']
        equity = result['backtest'].equity_curve
        plt.plot(equity, label=f"{pair[0]}-{pair[1]}")
    plt.title("Pair-Level Equity Curves")
    plt.legend()
    plt.savefig(f"{save_path}/pair_equity_curves.png")
    plt.close()
    
    # Sharpe contribution
    sharpe_contrib = pd.DataFrame([
        {
            'pair': f"{r['pair'][0]}-{r['pair'][1]}",
            'sharpe': r['metrics'].get('sharpe_ratio', 0),
            'return': r['metrics'].get('annualized_return', 0)
        }
        for r in pair_results
    ])
    plt.figure(figsize=(12, 6))
    sns.barplot(data=sharpe_contrib, x='pair', y='sharpe')
    plt.xticks(rotation=45)
    plt.title("Sharpe Ratio by Pair")
    plt.tight_layout()
    plt.savefig(f"{save_path}/sharpe_contribution.png")
    plt.close()
    
    # Rolling Sharpe
    plt.figure(figsize=(15, 8))
    for result in pair_results:
        pair = result['pair']
        returns = result['backtest'].daily_returns
        rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
        plt.plot(rolling_sharpe, label=f"{pair[0]}-{pair[1]}")
    plt.title("Rolling Sharpe Ratio (252-day window)")
    plt.legend()
    plt.savefig(f"{save_path}/rolling_sharpe.png")
    plt.close()

def generate_live_sim_plots(equity_curve: pd.Series, daily_pnl: pd.Series, 
                          active_trades: Dict, save_path: str = "reports/"):
    """Generate plots specific to live simulation."""
    os.makedirs(save_path, exist_ok=True)
    
    # Equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve)
    plt.title("Live Simulation Equity Curve")
    plt.savefig(f"{save_path}/live_equity_curve.png")
    plt.close()
    
    # Daily PnL distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(daily_pnl, bins=50)
    plt.title("Daily PnL Distribution")
    plt.savefig(f"{save_path}/daily_pnl_dist.png")
    plt.close()
    
    # Active positions over time
    position_counts = pd.Series(0, index=equity_curve.index)
    for trade in active_trades.values():
        position_counts[trade.entry_date:trade.exit_date] += 1
    plt.figure(figsize=(12, 6))
    plt.plot(position_counts)
    plt.title("Number of Active Positions Over Time")
    plt.savefig(f"{save_path}/active_positions.png")
    plt.close() 