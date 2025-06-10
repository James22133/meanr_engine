import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import seaborn as sns

def plot_equity_curve(equity_curve: pd.Series, drawdown: pd.Series, title: str = "Equity Curve and Drawdown") -> None:
    """Plot equity curve with drawdown overlay."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot equity curve
    ax1.plot(equity_curve.index, equity_curve, label='Equity', color='blue')
    ax1.set_title(title)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot drawdown
    ax2.fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=0.3)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_monthly_returns(returns: pd.Series, title: str = "Monthly Returns Heatmap") -> None:
    """Plot monthly returns heatmap."""
    # Resample to monthly returns
    monthly_returns = returns.resample('M').sum()
    
    # Create a pivot table for the heatmap
    monthly_returns_pivot = monthly_returns.to_frame()
    monthly_returns_pivot['Year'] = monthly_returns_pivot.index.year
    monthly_returns_pivot['Month'] = monthly_returns_pivot.index.month
    monthly_returns_pivot = monthly_returns_pivot.pivot(index='Year', columns='Month', values=0)
    
    # Plot heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(monthly_returns_pivot * 100, 
                annot=True, 
                fmt='.1f', 
                cmap='RdYlGn', 
                center=0,
                cbar_kws={'label': 'Return (%)'})
    plt.title(title)
    plt.xlabel('Month')
    plt.ylabel('Year')
    plt.tight_layout()
    plt.show()

def plot_trade_distribution(trades: List[Dict], title: str = "Trade Distribution") -> None:
    """Plot distribution of trade P&L and holding periods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # P&L distribution
    pnl_values = [t['pnl'] for t in trades if t['pnl'] is not None]
    sns.histplot(pnl_values, bins=50, ax=ax1)
    ax1.set_title('Trade P&L Distribution')
    ax1.set_xlabel('P&L ($)')
    ax1.set_ylabel('Count')
    
    # Holding period distribution
    holding_periods = [(t['exit_date'] - t['entry_date']).days for t in trades if t['exit_date'] is not None]
    sns.histplot(holding_periods, bins=30, ax=ax2)
    ax2.set_title('Holding Period Distribution')
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()

def plot_regime_performance(returns: pd.Series, regimes: pd.Series, title: str = "Regime-Specific Performance") -> None:
    """Plot performance breakdown by regime."""
    # Calculate returns by regime
    regime_returns = pd.DataFrame({
        'returns': returns,
        'regime': regimes
    })
    
    plt.figure(figsize=(12, 6))

    # Plot cumulative returns for each unique regime separately
    for regime in regime_returns['regime'].unique():
        regime_data = regime_returns[regime_returns['regime'] == regime]
        cumulative = regime_data['returns'].cumsum()
        plt.plot(regime_data.index, cumulative, label=f'Regime {regime}')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_performance_metrics(metrics: Dict[str, float], title: str = "Performance Metrics") -> None:
    """Plot key performance metrics as a bar chart."""
    # Select metrics to plot
    plot_metrics = {
        'Annualized Return': metrics['annualized_return'] * 100,
        'Sharpe Ratio': metrics['sharpe_ratio'],
        'Max Drawdown': metrics['max_drawdown'] * 100,
        'Win Rate': metrics['win_rate'] * 100
    }
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(plot_metrics.keys(), plot_metrics.values())
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%' if 'Rate' in bar.get_x() or 'Drawdown' in bar.get_x() else f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show() 
