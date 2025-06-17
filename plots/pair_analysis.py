import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
import logging

logger = logging.getLogger(__name__)

def calculate_pair_metrics(data, pair, windows):
    """Calculate all metrics for a pair."""
    asset1, asset2 = pair
    prices1 = data[asset1]
    prices2 = data[asset2]
    
    # Calculate spread
    spread = prices1 - prices2
    
    # Calculate z-score
    zscore = (spread - spread.rolling(window=windows['zscore']).mean()) / spread.rolling(window=windows['zscore']).std()
    
    # Calculate correlation
    correlation = prices1.rolling(window=windows['correlation']).corr(prices2)
    
    # Calculate cointegration
    cointegration = pd.Series(index=spread.index)
    for i in range(windows['cointegration'], len(spread)):
        window = spread.iloc[i-windows['cointegration']:i]
        if len(window.dropna()) > 0:
            _, pvalue, _ = coint(prices1.iloc[i-windows['cointegration']:i], 
                               prices2.iloc[i-windows['cointegration']:i])
            cointegration.iloc[i] = pvalue
    
    # Calculate spread stability
    stability = pd.Series(index=spread.index)
    for i in range(windows['stability'], len(spread)):
        window = spread.iloc[i-windows['stability']:i]
        if len(window.dropna()) > 0:
            stability.iloc[i] = 1 / window.std()
    
    return {
        'spread': spread,
        'zscore': zscore,
        'correlation': correlation,
        'cointegration': cointegration,
        'stability': stability
    }

def plot_pair_analysis(
    price1: pd.Series,
    price2: pd.Series,
    pair_name: str,
    save_dir: str,
    windows: Dict[str, int]
) -> None:
    """Generate comprehensive pair analysis plots."""
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(4, 2)
    
    # 1. Price Series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(price1.index, price1, label=price1.name)
    ax1.plot(price2.index, price2, label=price2.name)
    ax1.set_title(f"Price Series: {pair_name}")
    ax1.legend()
    ax1.grid(True)
    
    # 2. Rolling Correlation
    ax2 = fig.add_subplot(gs[1, 0])
    corr = price1.pct_change().rolling(windows['correlation']).corr(price2.pct_change())
    ax2.plot(corr.index, corr)
    ax2.set_title(f"Rolling Correlation ({windows['correlation']}d)")
    ax2.grid(True)
    
    # 3. Spread and Z-Score
    ax3 = fig.add_subplot(gs[1, 1])
    spread = price1 - price2
    zscore = (spread - spread.rolling(windows['zscore']).mean()) / spread.rolling(windows['zscore']).std()
    ax3.plot(spread.index, spread, label='Spread')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(zscore.index, zscore, 'r--', label='Z-Score')
    ax3.set_title("Spread and Z-Score")
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True)
    
    # 4. Cointegration P-Value
    ax4 = fig.add_subplot(gs[2, 0])
    coint_pvals = []
    for i in range(len(price1) - windows['cointegration']):
        window1 = price1.iloc[i:i+windows['cointegration']]
        window2 = price2.iloc[i:i+windows['cointegration']]
        _, pval, _ = coint(window1, window2)
        coint_pvals.append(pval)
    ax4.plot(price1.index[windows['cointegration']:], coint_pvals)
    ax4.axhline(y=0.05, color='r', linestyle='--', label='5% significance')
    ax4.set_title(f"Rolling Cointegration P-Value ({windows['cointegration']}d)")
    ax4.legend()
    ax4.grid(True)
    
    # 5. ADF Test P-Value
    ax5 = fig.add_subplot(gs[2, 1])
    adf_pvals = []
    for i in range(len(spread) - windows['stability']):
        window = spread.iloc[i:i+windows['stability']]
        _, pval, _, _, _, _ = adfuller(window.dropna())
        adf_pvals.append(pval)
    ax5.plot(spread.index[windows['stability']:], adf_pvals)
    ax5.axhline(y=0.05, color='r', linestyle='--', label='5% significance')
    ax5.set_title(f"Rolling ADF P-Value ({windows['stability']}d)")
    ax5.legend()
    ax5.grid(True)
    
    # 6. Stability Metrics
    ax6 = fig.add_subplot(gs[3, :])
    # Calculate rolling volatility
    vol = spread.rolling(windows['stability']).std()
    # Calculate rolling mean reversion speed (negative autocorrelation)
    autocorr = spread.rolling(windows['stability']).apply(
        lambda x: x.autocorr() if len(x.dropna()) > 20 else 0
    )
    ax6.plot(vol.index, vol, label='Spread Volatility')
    ax6_twin = ax6.twinx()
    ax6_twin.plot(autocorr.index, autocorr, 'r--', label='Autocorrelation')
    ax6.set_title("Stability Metrics")
    ax6.legend(loc='upper left')
    ax6_twin.legend(loc='upper right')
    ax6.grid(True)
    
    # Save plot
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{pair_name.replace('-', '_')}_analysis.png"))
    plt.close()

def plot_pair_universe_analysis(data, universe, output_dir, plot_windows):
    """Generate comprehensive analysis plots for each pair in the universe."""
    os.makedirs(output_dir, exist_ok=True)
    
    for pair in universe['pairs']:
        try:
            # Calculate metrics
            metrics = calculate_pair_metrics(data, pair, plot_windows)
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 15))
            gs = fig.add_gridspec(4, 2)
            
            # Price plot
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(data[pair[0]], label=pair[0], alpha=0.7)
            ax1.plot(data[pair[1]], label=pair[1], alpha=0.7)
            ax1.set_title(f'Price Series: {pair[0]} vs {pair[1]}')
            ax1.legend()
            ax1.grid(True)
            
            # Spread plot
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.plot(metrics['spread'], label='Spread')
            ax2.set_title('Price Spread')
            ax2.grid(True)
            
            # Z-score plot
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.plot(metrics['zscore'], label='Z-score')
            ax3.axhline(y=2, color='r', linestyle='--', alpha=0.5)
            ax3.axhline(y=-2, color='r', linestyle='--', alpha=0.5)
            ax3.set_title('Z-score')
            ax3.grid(True)
            
            # Correlation plot
            ax4 = fig.add_subplot(gs[2, 0])
            ax4.plot(metrics['correlation'], label='Correlation')
            ax4.axhline(y=0.8, color='r', linestyle='--', alpha=0.5)
            ax4.set_title('Rolling Correlation')
            ax4.grid(True)
            
            # Cointegration plot
            ax5 = fig.add_subplot(gs[2, 1])
            ax5.plot(metrics['cointegration'], label='Cointegration p-value')
            ax5.axhline(y=0.05, color='r', linestyle='--', alpha=0.5)
            ax5.set_title('Cointegration Test')
            ax5.grid(True)
            
            # Stability plot
            ax6 = fig.add_subplot(gs[3, :])
            ax6.plot(metrics['stability'], label='Spread Stability')
            ax6.set_title('Spread Stability (1/Std)')
            ax6.grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(output_dir, f"{pair[0]}_{pair[1]}_analysis.png"))
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting pair {pair}: {str(e)}")
            continue

def plot_universe_correlation_matrix(data, universe, output_dir):
    """Generate correlation matrix heatmap for the universe."""
    try:
        # Get all unique assets in the universe
        assets = set()
        for pair in universe['pairs']:
            assets.update(pair)
        
        # Calculate correlation matrix
        corr_matrix = data[list(assets)].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='RdYlBu_r',
                   center=0,
                   fmt='.2f',
                   square=True)
        
        plt.title(f'Correlation Matrix - {universe["description"]}')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating correlation matrix: {str(e)}")

def plot_universe_performance(
    metrics_df: pd.DataFrame,
    save_dir: str
) -> None:
    """Generate performance comparison plots for universe pairs."""
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot correlation vs stability
    ax1.scatter(
        metrics_df['correlation'],
        metrics_df['stability'],
        s=metrics_df['score'] * 100,
        alpha=0.6
    )
    for i, row in metrics_df.iterrows():
        ax1.annotate(
            f"{row['pair'][0]}-{row['pair'][1]}",
            (row['correlation'], row['stability'])
        )
    ax1.set_xlabel('Correlation')
    ax1.set_ylabel('Stability')
    ax1.set_title('Correlation vs Stability')
    ax1.grid(True)
    
    # Plot cointegration stability vs z-score volatility
    ax2.scatter(
        metrics_df['coint_stability'],
        metrics_df['zscore_vol'],
        s=metrics_df['score'] * 100,
        alpha=0.6
    )
    for i, row in metrics_df.iterrows():
        ax2.annotate(
            f"{row['pair'][0]}-{row['pair'][1]}",
            (row['coint_stability'], row['zscore_vol'])
        )
    ax2.set_xlabel('Cointegration Stability')
    ax2.set_ylabel('Z-Score Volatility')
    ax2.set_title('Cointegration Stability vs Z-Score Volatility')
    ax2.grid(True)
    
    # Save plot
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "universe_performance.png"))
    plt.close() 