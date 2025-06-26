#!/usr/bin/env python3
"""
Test script to verify the new threshold settings and generate trades.
"""

import sys
import os
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_thresholds():
    """Test the new threshold settings with a simple backtest."""
    
    # Load configuration directly from YAML
    with open('config_optimized.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Configuration loaded successfully")
    
    # Print key threshold settings
    logger.info("=== THRESHOLD SETTINGS ===")
    logger.info(f"Signals entry_threshold: {config['signals']['entry_threshold']}")
    logger.info(f"Signals exit_threshold: {config['signals']['exit_threshold']}")
    logger.info(f"Backtest zscore_entry_threshold: {config['backtest']['zscore_entry_threshold']}")
    logger.info(f"Backtest zscore_exit_threshold: {config['backtest']['zscore_exit_threshold']}")
    
    # Test signal generation with sample data
    logger.info("\n=== TESTING SIGNAL GENERATION ===")
    
    # Create sample price data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Simulate correlated prices
    base_price1 = 100 + np.cumsum(np.random.randn(len(dates)) * 0.01)
    base_price2 = 50 + np.cumsum(np.random.randn(len(dates)) * 0.01)
    
    # Add some mean reversion behavior
    spread = base_price1 - base_price2
    mean_reversion = np.random.randn(len(dates)) * 0.5
    spread = spread + mean_reversion
    
    price1 = base_price1 + spread * 0.1
    price2 = base_price2 - spread * 0.1
    
    pair_data = pd.DataFrame({
        'XOM': price1,
        'CVX': price2
    }, index=dates)
    
    logger.info(f"Sample data shape: {pair_data.shape}")
    
    # Calculate spread and z-score manually
    spread = pair_data['XOM'] - pair_data['CVX']
    z_score = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()
    
    # Get thresholds
    entry_threshold = config['signals']['entry_threshold']
    exit_threshold = config['signals']['exit_threshold']
    
    # Generate signals
    entry_long = z_score < -entry_threshold
    entry_short = z_score > entry_threshold
    exit_signal = (z_score >= -exit_threshold) & (z_score <= exit_threshold)
    
    # Count signals
    long_signals = entry_long.sum()
    short_signals = entry_short.sum()
    exit_signals = exit_signal.sum()
    total_signals = long_signals + short_signals
    
    logger.info(f"=== SIGNAL GENERATION RESULTS ===")
    logger.info(f"Z-score range: {z_score.min():.3f} to {z_score.max():.3f}")
    logger.info(f"Z-score mean: {z_score.mean():.3f}, std: {z_score.std():.3f}")
    logger.info(f"Thresholds - Entry: {entry_threshold}, Exit: {exit_threshold}")
    logger.info(f"Signals generated - Long: {long_signals}, Short: {short_signals}, Exit: {exit_signals}")
    logger.info(f"Total entry signals: {total_signals}")
    
    # Show recent z-scores
    recent_zscore = z_score.tail(10)
    logger.info(f"Recent z-scores: {recent_zscore.values}")
    
    # Check extreme values
    extreme_positive = (z_score > entry_threshold).sum()
    extreme_negative = (z_score < -entry_threshold).sum()
    logger.info(f"Extreme z-scores > {entry_threshold}: {extreme_positive}")
    logger.info(f"Extreme z-scores < -{entry_threshold}: {extreme_negative}")
    
    if total_signals == 0:
        logger.warning("No entry signals generated! Consider lowering entry threshold.")
        # Show z-scores near threshold
        near_threshold = z_score[abs(z_score) >= entry_threshold * 0.8]
        if len(near_threshold) > 0:
            logger.info(f"Z-scores near threshold: {near_threshold.head(5).values}")
    else:
        logger.info(f"✅ SUCCESS: Generated {total_signals} entry signals with threshold {entry_threshold}")
    
    logger.info("\n=== THRESHOLD TEST COMPLETE ===")

if __name__ == "__main__":
    test_thresholds() 