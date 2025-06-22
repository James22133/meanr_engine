#!/usr/bin/env python3
"""
Enhanced Mean Reversion Pairs Trading Engine
Integrates vectorbt, empyrical, and statsmodels for high-performance, statistically rigorous backtesting.
"""

import argparse
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.data_loader import DataLoader
from core.pair_selection import PairSelector
from core.enhanced_pair_selection import EnhancedPairSelector, StatisticalThresholds
from core.pair_selection import PairSelector as SignalGenerator
from regime.regime_detection import RegimeDetector
from backtest.vectorbt_backtest import VectorBTBacktest, VectorBTConfig
from core.enhanced_metrics import EnhancedMetricsCalculator, EnhancedMetricsConfig
from core.diagnostics import TradeDiagnostics
from core.plotting import PlotGenerator

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('enhanced_engine.log')
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return None

def main():
    print("RUNNING LATEST VERSION of run_engine_enhanced.py")
    """Main function for enhanced mean reversion pairs trading engine."""
    parser = argparse.ArgumentParser(description='Enhanced Mean Reversion Pairs Trading Engine')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file path')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to disk')
    parser.add_argument('--optimize', action='store_true', help='Run parameter optimization')
    parser.add_argument('--vectorbt-only', action='store_true', help='Use only vectorbt backtesting')
    parser.add_argument('--statistical-report', action='store_true', help='Generate detailed statistical report')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Enhanced Mean Reversion Pairs Trading Engine")
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        logger.error("Failed to load configuration")
        return
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        
        # Data loader
        data_loader = DataLoader(config['data'])
        
        # Enhanced pair selector with statistical rigor
        statistical_thresholds = StatisticalThresholds(
            adf_pvalue_max=config.get('statistical', {}).get('adf_pvalue_max', 0.05),
            coint_pvalue_max=config.get('statistical', {}).get('coint_pvalue_max', 0.05),
            r_squared_min=config.get('statistical', {}).get('r_squared_min', 0.7),
            correlation_min=config.get('statistical', {}).get('correlation_min', 0.8),
            hurst_threshold=config.get('statistical', {}).get('hurst_threshold', 0.5),
            min_observations=config.get('statistical', {}).get('min_observations', 252)
        )
        
        enhanced_pair_selector = EnhancedPairSelector(config, statistical_thresholds)
        
        # Signal generator
        signal_generator = SignalGenerator(config['signals'])
        
        # Regime detector
        regime_detector = RegimeDetector(config['regime'])
        
        # VectorBT backtest engine
        vectorbt_config = VectorBTConfig(
            initial_capital=config['backtest'].get('initial_capital', 1_000_000),
            fees=config['backtest'].get('commission_bps', 1.0) / 10000,
            slippage=config['backtest'].get('slippage_bps', 2.0) / 10000,
            max_concurrent_positions=config['backtest'].get('max_concurrent_positions', 5),
            regime_scaling=config['backtest'].get('regime_scaling', True),
            regime_volatility_multiplier=config['backtest'].get('regime_volatility_multiplier', 1.0),
            regime_trend_multiplier=config['backtest'].get('regime_trend_multiplier', 1.0)
        )
        
        vectorbt_backtest = VectorBTBacktest(vectorbt_config)
        
        # Enhanced metrics calculator
        enhanced_metrics_config = EnhancedMetricsConfig(
            risk_free_rate=config.get('metrics', {}).get('risk_free_rate', 0.02),
            confidence_level=config.get('metrics', {}).get('confidence_level', 0.95),
            periods_per_year=config.get('metrics', {}).get('periods_per_year', 252)
        )
        
        enhanced_metrics = EnhancedMetricsCalculator(enhanced_metrics_config)
        
        # Diagnostics
        diagnostics = TradeDiagnostics(config)
        
        # Plot generator
        plot_generator = PlotGenerator(config)
        
        # Load data
        logger.info("Loading market data...")
        data = data_loader.fetch_data(
            config['data']['symbols'],
            config['data']['start_date'],
            config['data']['end_date']
        )
        if data is None or data.empty:
            logger.error("Failed to load data")
            return
        
        # Detect market regimes
        logger.info("Detecting market regimes...")
        all_regimes = regime_detector.detect_regimes(data)
        
        # Enhanced pair selection with statistical rigor
        logger.info("Performing enhanced pair selection...")
        pair_metrics = {}
        
        for pair in data_loader.get_all_pairs():
            pair_data = data_loader.get_pair_data(pair)
            if pair_data is not None:
                metrics = enhanced_pair_selector.calculate_pair_metrics_enhanced(pair_data)
                if metrics:
                    pair_metrics[pair] = metrics
        
        # Generate statistical report if requested
        if args.statistical_report:
            statistical_report = enhanced_pair_selector.generate_statistical_report(pair_metrics)
            logger.info(statistical_report)
            
            # Save report to file
            with open('statistical_analysis_report.txt', 'w') as f:
                f.write(statistical_report)
        
        # Select pairs based on enhanced criteria
        selected_pairs = enhanced_pair_selector.select_pairs_enhanced(pair_metrics)
        logger.info(f"Selected {len(selected_pairs)} pairs for trading")
        
        # Generate signals
        logger.info("Generating trading signals...")
        all_signals = {}
        
        for pair in selected_pairs:
            pair_data = data_loader.get_pair_data(pair)
            if pair_data is not None:
                signals = signal_generator.generate_signals(pair_data)
                if signals is not None:
                    all_signals[pair] = signals
        
        # Run enhanced backtests
        logger.info("Running enhanced backtests...")
        backtest_results = {}
        portfolio_equity = None
        all_trades = []
        
        if args.vectorbt_only:
            # Use only vectorbt for backtesting
            for pair in selected_pairs:
                if pair in all_signals:
                    pair_data = data_loader.get_pair_data(pair)
                    if pair_data is not None:
                        logger.info(f"Running vectorbt backtest for {pair[0]}-{pair[1]}")
                        
                        # Get regime series for this pair
                        regime_series = all_regimes.get(pair, pd.Series(0, index=pair_data.index))
                        
                        # Run vectorbt backtest
                        results = vectorbt_backtest.run_vectorized_backtest(
                            pair_data.iloc[:, 0],  # First asset
                            pair_data.iloc[:, 1],  # Second asset
                            regime_series=regime_series,
                            lookback=config['signals'].get('lookback', 20),
                            entry_threshold=config['signals'].get('entry_threshold', 2.0),
                            exit_threshold=config['signals'].get('exit_threshold', 0.5)
                        )
                        
                        if results:
                            backtest_results[pair] = results
                            
                            # Generate vectorbt report
                            report = vectorbt_backtest.generate_report(results, f"{pair[0]}-{pair[1]}")
                            logger.info(report)
                            
                            # Calculate enhanced metrics
                            if 'returns' in results:
                                enhanced_metrics_result = enhanced_metrics.calculate_pair_specific_metrics(
                                    results['returns'], results['equity_curve']
                                )
                                
                                enhanced_report = enhanced_metrics.generate_enhanced_report(
                                    enhanced_metrics_result, f"{pair[0]}-{pair[1]}"
                                )
                                logger.info(enhanced_report)
                            
                            # Aggregate trades for diagnostics
                            if 'trades' in results and results['trades'] is not None:
                                all_trades.extend(results['trades'].to_dict('records'))
                            
                            # Aggregate equity curves
                            if portfolio_equity is None:
                                portfolio_equity = results['equity_curve']
                            else:
                                portfolio_equity = portfolio_equity.add(results['equity_curve'], fill_value=0)
        else:
            # Use hybrid approach (vectorbt + traditional)
            logger.info("Using hybrid backtesting approach...")
            # This would combine vectorbt with traditional backtesting
            # Implementation depends on specific requirements
        
        # Parameter optimization if requested
        if args.optimize:
            logger.info("Running parameter optimization...")
            for pair in selected_pairs[:3]:  # Optimize top 3 pairs
                pair_data = data_loader.get_pair_data(pair)
                if pair_data is not None:
                    logger.info(f"Optimizing parameters for {pair[0]}-{pair[1]}")
                    
                    optimization_results = vectorbt_backtest.optimize_parameters(
                        pair_data.iloc[:, 0],
                        pair_data.iloc[:, 1],
                        regime_series=all_regimes.get(pair, pd.Series(0, index=pair_data.index))
                    )
                    
                    if optimization_results:
                        logger.info(f"Best parameters for {pair[0]}-{pair[1]}:")
                        logger.info(f"  Parameters: {optimization_results['best_params']}")
                        logger.info(f"  Best Sharpe: {optimization_results['best_sharpe']:.3f}")
        
        # Generate comprehensive diagnostics
        if all_trades:
            logger.info("Generating comprehensive diagnostics...")
            diagnostic_results = diagnostics.generate_diagnostic_report(all_trades)
            
            if diagnostic_results:
                logger.info("Diagnostic analysis completed")
        
        # Generate enhanced portfolio metrics
        if portfolio_equity is not None:
            logger.info("Calculating enhanced portfolio metrics...")
            portfolio_returns = portfolio_equity.pct_change().fillna(0)
            
            portfolio_metrics = enhanced_metrics.calculate_comprehensive_metrics(
                portfolio_returns, portfolio_equity
            )
            
            portfolio_report = enhanced_metrics.generate_enhanced_report(
                portfolio_metrics, "PORTFOLIO"
            )
            logger.info(portfolio_report)
        
        # Generate plots if requested
        if args.save_plots:
            logger.info("Generating plots...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plots_dir = f"plots/{timestamp}"
            os.makedirs(plots_dir, exist_ok=True)
            
            # Generate various plots
            for pair, results in backtest_results.items():
                if 'equity_curve' in results and 'signals' in results:
                    plot_generator.plot_vectorbt_analysis(
                        results['prices'],
                        results['signals']['entries'],
                        results['signals']['exits'],
                        results['equity_curve'],
                        f"{pair[0]}-{pair[1]}",
                        save_path=f"{plots_dir}/analysis_{pair[0]}_{pair[1]}.png"
                    )
            
            # Portfolio plots
            if portfolio_equity is not None:
                plot_generator.plot_equity_curves(
                    portfolio_equity,
                    "Portfolio",
                    save_path=f"{plots_dir}/portfolio_equity.png"
                )
        
        logger.info("Enhanced backtesting completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in enhanced engine: {e}")
        raise

if __name__ == "__main__":
    main() 