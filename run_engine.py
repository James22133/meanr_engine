"""
Main script for running the mean reversion trading engine.
"""

import argparse
import logging
import yaml
from typing import Dict, List
import os
from datetime import datetime
import pandas as pd

from core.config import Config
from core.data_loader import DataLoader
from core.pair_selection import PairSelector
from backtest.backtest import PairsBacktest
from core.metrics import MetricsCalculator
from core.plotting import PlotGenerator
from core import compute_sector_exposure, max_drawdown

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run mean reversion trading engine')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--save-plots', action='store_true', help='Save plots')
    return parser.parse_args()

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    return Config.from_yaml(config_path)

def main():
    """Main function to run the trading engine."""
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Load configuration
        config = load_config(args.config)
        
        # Setup logging
        logger = setup_logging()
        
        # Initialize components
        data_loader = DataLoader(config)
        pair_selector = PairSelector(config)
        backtest_runner = PairsBacktest(config.backtest)
        metrics_calculator = MetricsCalculator(config)
        plot_generator = PlotGenerator(config)
        
        # Fetch market data
        logger.info("Fetching market data...")
        data = data_loader.fetch_data(
            tickers=config.etf_tickers,
            start_date=config.start_date,
            end_date=config.end_date
        )
        
        if data is None or data.empty:
            logger.error("No data fetched. Exiting.")
            return
        
        # Process each universe efficiently
        selected_pairs = []
        pair_metrics = {}
        
        for universe_name, universe_config in config.pair_universes.items():
            logger.info(f"Processing universe: {universe_name}")
            pairs = universe_config.get('pairs', [])
            
            # Process pairs in batches for better performance
            batch_size = 10
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                
                for pair in batch_pairs:
                    pair_data = data_loader.get_pair_data(pair)
                    if pair_data is not None:
                        metrics = pair_selector.calculate_pair_metrics(pair_data)
                        if metrics is not None:
                            pair_metrics[tuple(pair)] = metrics
        
        # Select pairs based on metrics
        selected_pairs = pair_selector.select_pairs(pair_metrics)
        
        if not selected_pairs:
            logger.warning("No pairs met the filtering criteria across all universes.")
            return
        
        # Run backtests efficiently
        logger.info("Running backtests...")
        backtest_results = {}
        
        # Prepare all signals at once for better performance
        all_signals = {}
        all_regimes = None
        
        for pair in selected_pairs:
            pair_data = data_loader.get_pair_data(pair)
            if pair_data is not None:
                # Generate signals efficiently
                signals = pair_selector.generate_signals(pair_data)
                all_signals[pair] = signals
                
                # Use same regime for all pairs for now (can be optimized later)
                if all_regimes is None:
                    all_regimes = pd.Series(0, index=pair_data.index)
        
        # Run backtests
        for pair in selected_pairs:
            if pair in all_signals:
                backtest_runner.run_backtest(
                    data_loader.get_pair_data(pair), 
                    {pair: all_signals[pair]}, 
                    all_regimes
                )
                backtest_results[pair] = {
                    'equity_curve': backtest_runner.equity_curve,
                    'daily_returns': backtest_runner.daily_returns,
                    'trades': backtest_runner.trades,
                }
        
        # Calculate portfolio metrics
        portfolio_metrics = metrics_calculator.calculate_portfolio_metrics(backtest_results)
        
        # Generate plots
        logger.info("Generating plots...")
        if args.save_plots:
            plot_generator.plot_pair_analysis(backtest_results, pair_metrics)
            plot_generator.plot_backtest_results(backtest_results)
            plot_generator.plot_performance_metrics(portfolio_metrics)
        
        # Log comprehensive results
        logger.info("=" * 80)
        logger.info("BACKTEST RESULTS SUMMARY")
        logger.info("=" * 80)
        
        # Log selected pairs
        logger.info(f"Selected pairs: {selected_pairs}")
        logger.info(f"Total pairs analyzed: {len(pair_metrics)}")
        
        # Log portfolio performance
        if portfolio_metrics:
            logger.info("\n" + "=" * 50)
            logger.info("PORTFOLIO PERFORMANCE METRICS")
            logger.info("=" * 50)
            
            # Basic metrics
            logger.info(f"Initial Capital: ${portfolio_metrics.get('initial_capital', 0):,.2f}")
            logger.info(f"Final Capital: ${portfolio_metrics.get('final_capital', 0):,.2f}")
            logger.info(f"Total PnL: ${portfolio_metrics.get('total_pnl', 0):,.2f}")
            logger.info(f"Total Return: {portfolio_metrics.get('total_return', 0):.2%}")
            logger.info(f"Annualized Return: {portfolio_metrics.get('annualized_return', 0):.2%}")
            logger.info(f"Annualized Volatility: {portfolio_metrics.get('annualized_volatility', 0):.2%}")
            
            # Risk metrics
            logger.info(f"Sharpe Ratio: {portfolio_metrics.get('sharpe_ratio', 0):.3f}")
            logger.info(f"Sortino Ratio: {portfolio_metrics.get('sortino_ratio', 0):.3f}")
            logger.info(f"Calmar Ratio: {portfolio_metrics.get('calmar_ratio', 0):.3f}")
            logger.info(f"Maximum Drawdown: {portfolio_metrics.get('max_drawdown', 0):.2%}")
            logger.info(f"Value at Risk (95%): {portfolio_metrics.get('var_95', 0):.2%}")
            
            # Trading statistics
            logger.info("\n" + "=" * 50)
            logger.info("TRADING STATISTICS")
            logger.info("=" * 50)
            
            total_trades = portfolio_metrics.get('total_trades', 0)
            winning_trades = portfolio_metrics.get('winning_trades', 0)
            losing_trades = portfolio_metrics.get('losing_trades', 0)
            win_rate = portfolio_metrics.get('win_rate', 0)
            
            logger.info(f"Total Trades: {total_trades}")
            logger.info(f"Winning Trades: {winning_trades}")
            logger.info(f"Losing Trades: {losing_trades}")
            logger.info(f"Win Rate: {win_rate:.2%}")
            logger.info(f"Average Win: ${portfolio_metrics.get('avg_win', 0):,.2f}")
            logger.info(f"Average Loss: ${portfolio_metrics.get('avg_loss', 0):,.2f}")
            logger.info(f"Largest Win: ${portfolio_metrics.get('largest_win', 0):,.2f}")
            logger.info(f"Largest Loss: ${portfolio_metrics.get('largest_loss', 0):,.2f}")
            logger.info(f"Profit Factor: {portfolio_metrics.get('profit_factor', 0):.3f}")
            logger.info(f"Average Holding Period: {portfolio_metrics.get('avg_holding_period', 0):.1f} days")
            
            # Advanced metrics
            logger.info("\n" + "=" * 50)
            logger.info("ADVANCED METRICS")
            logger.info("=" * 50)
            
            logger.info(f"Information Ratio: {portfolio_metrics.get('information_ratio', 0):.3f}")
            logger.info(f"Treynor Ratio: {portfolio_metrics.get('treynor_ratio', 0):.3f}")
            logger.info(f"Jensen's Alpha: {portfolio_metrics.get('jensen_alpha', 0):.3f}")
            logger.info(f"Recovery Factor: {portfolio_metrics.get('recovery_factor', 0):.3f}")
            logger.info(f"Sterling Ratio: {portfolio_metrics.get('sterling_ratio', 0):.3f}")
            logger.info(f"Gain to Pain Ratio: {portfolio_metrics.get('gain_to_pain_ratio', 0):.3f}")
            logger.info(f"Best 30-Day Return: {portfolio_metrics.get('best_30d_return', 0):.2%}")
            logger.info(f"Worst 30-Day Return: {portfolio_metrics.get('worst_30d_return', 0):.2%}")
            
            # Risk metrics
            logger.info(f"Skewness: {portfolio_metrics.get('skewness', 0):.3f}")
            logger.info(f"Kurtosis: {portfolio_metrics.get('kurtosis', 0):.3f}")
            logger.info(f"Downside Deviation: {portfolio_metrics.get('downside_deviation', 0):.2%}")
            
            # Individual pair performance
            logger.info("\n" + "=" * 50)
            logger.info("INDIVIDUAL PAIR PERFORMANCE")
            logger.info("=" * 50)
            
            for pair, results in backtest_results.items():
                pair_metrics = metrics_calculator.calculate_pair_metrics(results)
                if pair_metrics:
                    logger.info(f"\nPair: {pair[0]}-{pair[1]}")
                    logger.info(f"  Total Return: {pair_metrics.get('total_return', 0):.2%}")
                    logger.info(f"  Sharpe Ratio: {pair_metrics.get('sharpe_ratio', 0):.3f}")
                    logger.info(f"  Max Drawdown: {pair_metrics.get('max_drawdown', 0):.2%}")
                    logger.info(f"  Total Trades: {pair_metrics.get('total_trades', 0)}")
                    logger.info(f"  Win Rate: {pair_metrics.get('win_rate', 0):.2%}")
                    logger.info(f"  Total PnL: ${pair_metrics.get('total_pnl', 0):,.2f}")
        
        logger.info("\n" + "=" * 80)
        logger.info("Backtest completed successfully")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error running engine: {str(e)}")
        raise

if __name__ == '__main__':
    main()

