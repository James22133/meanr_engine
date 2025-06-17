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
    PlotGenerator,
    WalkForwardValidator,
)
from core import compute_sector_exposure, max_drawdown

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
    pair_list = [tuple(p) for p in universe['pairs']]
    pair_metrics = pair_selector.score_pairs_parallel(pair_list, data_loader)
    
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
        
        # Compile universe pairs
        universe_pairs: List[tuple] = []
        for universe in config.pair_universes.values():
            universe_pairs.extend([tuple(p) for p in universe["pairs"]])

        # Run walk-forward validation
        validator = WalkForwardValidator(
            config,
            data_loader,
            pair_selector,
            backtest_runner,
            metrics_calculator,
        )

        fold_results = validator.run(universe_pairs)
        agg_metrics = validator.aggregate(fold_results)

        for i, fold in enumerate(fold_results, 1):
            for pair, th in fold["thresholds"].items():
                logger.info(
                    f"Fold {i} {pair} entry {th[0]} exit {th[1]}"
                )

        logger.info("Aggregate walk-forward metrics:")
        for k, v in agg_metrics.items():
            if k == "equity_curve":
                continue
            logger.info(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

        return
        
    except Exception as e:
        logger.error(f"Error running engine: {str(e)}")
        raise

if __name__ == '__main__':
    main()

