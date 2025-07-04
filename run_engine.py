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
from core.diagnostics import TradeDiagnostics
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
    """Load configuration from YAML file into a Config object."""
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
        # Note: We are passing dicts now, not a Config object.
        # The components need to be able to handle this.
        data_loader = DataLoader(config) 
        pair_selector = PairSelector(config)
        backtest_runner = PairsBacktest(config.backtest, data_loader)
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
                # Create a new backtest instance for each pair to avoid trade accumulation
                pair_backtest_runner = PairsBacktest(config.backtest, data_loader)
                pair_backtest_runner.run_backtest(
                    data_loader.get_pair_data(pair), 
                    {pair: all_signals[pair]}, 
                    all_regimes
                )
                backtest_results[pair] = {
                    'equity_curve': pair_backtest_runner.equity_curve,
                    'daily_returns': pair_backtest_runner.daily_returns,
                    'trades': pair_backtest_runner.trades,
                }
        
        # Calculate portfolio metrics
        portfolio_metrics = metrics_calculator.calculate_portfolio_metrics(backtest_results)
        
        # Initialize diagnostics module
        diagnostics = TradeDiagnostics(config)
        
        # Aggregate all trades from each individual backtest result for diagnostics
        all_trades = []
        for result in backtest_results.values():
            all_trades.extend(result.get('trades', []))
        
        # Generate diagnostic report using the aggregated list of all trades
        diagnostic_results = diagnostics.generate_diagnostic_report(all_trades)
        
        # Analyze performance by pair with comprehensive metrics
        pair_performance = diagnostics.analyze_by_pair(backtest_results)
        
        # Add pair performance to diagnostic results
        if diagnostic_results:
            diagnostic_results['pair_performance'] = pair_performance
        
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
        
        # Log portfolio metrics
        logger.info(f"Initial Capital: ${portfolio_metrics['initial_capital']:,.2f}")
        logger.info(f"Final Capital: ${portfolio_metrics['final_capital']:,.2f}")
        logger.info(f"Total PnL: ${portfolio_metrics['total_pnl']:,.2f}")
        logger.info(f"Total Return: {portfolio_metrics['total_return']:.2%}")
        logger.info(f"Annualized Return: {portfolio_metrics['annualized_return']:.2%}")
        logger.info(f"Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.3f}")
        logger.info(f"Maximum Drawdown: {portfolio_metrics['max_drawdown']:.2%}")
        
        # Log trading statistics
        logger.info(f"Total Trades: {portfolio_metrics['total_trades']}")
        logger.info(f"Winning Trades: {portfolio_metrics['winning_trades']}")
        logger.info(f"Losing Trades: {portfolio_metrics['losing_trades']}")
        logger.info(f"Win Rate: {portfolio_metrics['win_rate']:.2%}")
        logger.info(f"Average Win: ${portfolio_metrics['avg_win']:.2f}")
        logger.info(f"Average Loss: ${portfolio_metrics['avg_loss']:.2f}")
        logger.info(f"Profit Factor: {portfolio_metrics['profit_factor']:.3f}")
        logger.info(f"Average Holding Period: {portfolio_metrics['avg_holding_period']:.1f} days")
        
        # Log loss attribution by pair
        if diagnostic_results and 'pair_performance' in diagnostic_results:
            logger.info("\n" + "=" * 80)
            logger.info("PAIR PERFORMANCE SUMMARY")
            logger.info("=" * 80)
            
            pair_summary_data = []
            for pair, stats in diagnostic_results['pair_performance'].items():
                pair_summary_data.append({
                    'Pair': f"{pair[0]}-{pair[1]}",
                    'Total PnL ($)': stats.get('pnl_sum', 0),
                    'Annualized Return (%)': stats.get('annualized_return_pct', 0),
                    'Annualized Volatility (%)': stats.get('annualized_volatility_pct', 0),
                    'Sharpe Ratio': stats.get('sharpe_ratio', 0),
                    'Max Drawdown (%)': stats.get('max_drawdown_pct', 0),
                    'Win Rate (%)': stats.get('win_rate', 0),
                    'Profit Factor': stats.get('profit_factor', 0),
                    'Total Trades': stats.get('pnl_count', 0)
                })
            
            if pair_summary_data:
                summary_df = pd.DataFrame(pair_summary_data)
                logger.info("\n" + summary_df.to_string())

            logger.info("\n" + "=" * 80)
            logger.info("LOSS ATTRIBUTION BY PAIR")
            logger.info("=" * 80)
            
            pair_performance = diagnostic_results['pair_performance']
            for pair, stats in pair_performance.items():
                logger.info(f"{pair}:")
                logger.info(f"  Total PnL: ${stats.get('pnl_sum', 0):.2f}")
                logger.info(f"  Win Rate: {stats.get('win_rate', 0):.1f}%")
                logger.info(f"  Profit Factor: {stats.get('profit_factor', 0):.2f}")
                logger.info(f"  Trade Count: {stats.get('pnl_count', 0)}")
                logger.info(f"  Avg PnL: ${stats.get('pnl_mean', 0):.2f}")
        
        # Log regime performance
        if diagnostic_results and 'regime_analysis' in diagnostic_results:
            logger.info("\n" + "=" * 80)
            logger.info("PERFORMANCE BY MARKET REGIME")
            logger.info("=" * 80)
            
            regime_analysis = diagnostic_results['regime_analysis']
            for regime, stats in regime_analysis.items():
                logger.info(f"Regime {regime}:")
                logger.info(f"  Total PnL: ${stats.get('pnl_sum', 0):.2f}")
                logger.info(f"  Win Rate: {stats.get('win_rate', 0):.1f}%")
                logger.info(f"  Trade Count: {stats.get('pnl_count', 0)}")
                logger.info(f"  Avg PnL: ${stats.get('pnl_mean', 0):.2f}")
        
        # Log loss decomposition
        if diagnostic_results and 'loss_decomposition' in diagnostic_results:
            loss_decomp = diagnostic_results['loss_decomposition']
            if loss_decomp:
                logger.info("\n" + "=" * 80)
                logger.info("LOSS DECOMPOSITION ANALYSIS")
                logger.info("=" * 80)
                
                logger.info(f"Total Loss: ${loss_decomp.get('total_loss', 0):.2f}")
                logger.info(f"Average Loss: ${loss_decomp.get('avg_loss', 0):.2f}")
                logger.info(f"Median Loss: ${loss_decomp.get('median_loss', 0):.2f}")
                
                # Loss by holding period
                if 'loss_by_holding_period' in loss_decomp and not loss_decomp['loss_by_holding_period'].empty:
                    logger.info("\nLoss by Holding Period:")
                    for period, stats in loss_decomp['loss_by_holding_period'].items():
                        logger.info(f"  {period}: ${stats.get('sum', 0):.2f} ({stats.get('contribution_pct', 0):.1f}%)")
        
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error running engine: {str(e)}")
        raise

if __name__ == '__main__':
    main()

