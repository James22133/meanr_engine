"""
Main script for running the mean reversion trading engine.
"""

import argparse
import logging
import yaml
from typing import Dict, List
import os
from datetime import datetime

from core import (
    Config,
    DataLoader,
    PairSelector,
    BacktestRunner,
    MetricsCalculator,
    PlotGenerator
)

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    return Config.from_yaml(config_path)

def process_universe(universe: Dict, data_loader: DataLoader,
                    pair_selector: PairSelector) -> List[str]:
    """Process a universe of pairs."""
    logger = logging.getLogger(__name__)
    logger.info(f"Processing universe: {universe['description']}")
    
    selected_pairs = []
    pair_metrics = {}
    
    # Process each pair in the universe
    for pair in universe['pairs']:
        # Get pair data
        pair_data = data_loader.get_pair_data(pair)
        if pair_data is None:
            continue
        
        # Calculate pair metrics
        metrics = pair_selector.calculate_pair_metrics(pair_data)
        if metrics is None:
            continue
        
        pair_metrics[tuple(pair)] = metrics
    
    # Select pairs based on metrics
    selected_pairs = pair_selector.select_pairs(pair_metrics)
    
    return selected_pairs, pair_metrics

def main():
    """Main function to run the trading engine."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run mean reversion trading engine')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--save-plots', action='store_true', help='Save plots')
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Initialize components
        data_loader = DataLoader(config)
        pair_selector = PairSelector(config)
        backtest_runner = BacktestRunner(config)
        metrics_calculator = MetricsCalculator(config)
        plot_generator = PlotGenerator(config)
        
        # Fetch data
        logger.info("Fetching market data...")
        data_loader.fetch_data()
        
        # Process each universe
        all_selected_pairs = []
        all_pair_metrics = {}
        
        for universe_name, universe in config.pair_universes.items():
            selected_pairs, pair_metrics = process_universe(
                universe, data_loader, pair_selector
            )
            all_selected_pairs.extend(selected_pairs)
            all_pair_metrics.update(pair_metrics)
        
        if not all_selected_pairs:
            logger.warning("No pairs met the filtering criteria across all universes.")
            return
        
        # Run backtests
        logger.info("Running backtests...")
        backtest_results = {}
        
        for pair in all_selected_pairs:
            pair_data = data_loader.get_pair_data(pair)
            if pair_data is not None:
                results = backtest_runner.run_backtest(
                    pair_data, all_pair_metrics[tuple(pair)]
                )
                if results is not None:
                    backtest_results[tuple(pair)] = results
        
        # Calculate portfolio metrics
        portfolio_metrics = metrics_calculator.calculate_portfolio_metrics(backtest_results)
        
        # Generate plots
        if args.save_plots:
            logger.info("Generating plots...")
            plots_dir = os.path.join(config.plots_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot pair analysis
            for pair, metrics in all_pair_metrics.items():
                pair_data = data_loader.get_pair_data(list(pair))
                if pair_data is not None:
                    plot_generator.plot_pair_analysis(
                        pair_data, metrics,
                        save_path=os.path.join(plots_dir, f'analysis_{pair[0]}_{pair[1]}.png')
                    )
            
            # Plot backtest results
            plot_generator.plot_backtest_results(backtest_results, save_dir=plots_dir)
            
            # Plot performance metrics
            plot_generator.plot_performance_metrics(
                portfolio_metrics,
                save_path=os.path.join(plots_dir, 'performance_metrics.png')
            )
        
        # Log results
        logger.info("Backtest completed successfully")
        logger.info(f"Selected pairs: {all_selected_pairs}")
        logger.info("Portfolio metrics:")
        for metric, value in portfolio_metrics.items():
            if isinstance(value, float):
                logger.info(f"{metric}: {value:.2%}")
            else:
                logger.info(f"{metric}: {value}")
        
    except Exception as e:
        logger.error(f"Error running engine: {str(e)}")
        raise

if __name__ == '__main__':
    main()

