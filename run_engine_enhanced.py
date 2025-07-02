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
import json

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
    parser.add_argument('--walkforward', action='store_true', help='Run walk-forward analysis')
    parser.add_argument('--walkforward-windows', type=int, default=5, help='Number of walk-forward windows')
    
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
        logger.info("Performing enhanced pair selection...")
        pair_metrics = {}
        
        # Load statistical thresholds from config
        statistical_config = config.get('statistical', {})
        statistical_thresholds = StatisticalThresholds(
            adf_pvalue_max=statistical_config.get('adf_pvalue_max', 0.05),
            coint_pvalue_max=statistical_config.get('coint_pvalue_max', 0.05),
            r_squared_min=statistical_config.get('r_squared_min', 0.7),
            correlation_min=statistical_config.get('correlation_min', 0.8),
            hurst_threshold=statistical_config.get('hurst_threshold', 0.5),
            min_observations=statistical_config.get('min_observations', 252),
            max_volatility_spread=statistical_config.get('max_volatility_spread', 2.0),
            max_sector_pairs=statistical_config.get('max_sector_pairs', 3),
            min_sector_diversification=statistical_config.get('min_sector_diversification', 3)
        )
        
        # Update enhanced pair selector with new thresholds
        enhanced_pair_selector = EnhancedPairSelector(config, statistical_thresholds)
        
        # Signal generator
        signal_generator = SignalGenerator(config)
        
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
                # Use signal generator for signal generation
                signals = signal_generator.generate_signals(pair_data, pair)
                if signals is not None and not signals.empty:
                    all_signals[pair] = signals
        
        # Run enhanced backtests
        logger.info("Running enhanced backtests...")
        backtest_results = {}
        portfolio_equity = None
        all_trades = []
        
        # By default, use vectorbt for backtesting
        for pair in selected_pairs:
            if pair in all_signals:
                pair_data = data_loader.get_pair_data(pair)
                if pair_data is not None:
                    logger.info(f"Running vectorbt backtest for {pair[0]}-{pair[1]}")
                    
                    regime_series = all_regimes.get(pair, pd.Series(0, index=pair_data.index))
                    
                    results = vectorbt_backtest.run_vectorized_backtest(
                        pair_data.iloc[:, 0],
                        pair_data.iloc[:, 1],
                        regime_series=regime_series,
                        lookback=config['signals'].get('lookback', 20),
                        entry_threshold=config['signals'].get('entry_threshold', 2.0),
                        exit_threshold=config['signals'].get('exit_threshold', 0.5)
                    )
                    
                    if results:
                        backtest_results[pair] = results
                        trade_log_file = vectorbt_backtest.save_trade_logs(results, pair, "trade_logs")
                        if trade_log_file:
                            logger.info(f"Saved trade logs to {trade_log_file}")
                        
                        report = vectorbt_backtest.generate_report(results, f"{pair[0]}-{pair[1]}")
                        logger.info(report)
                        
                        if 'returns' in results:
                            enhanced_metrics_result = enhanced_metrics.calculate_pair_specific_metrics(
                                results['returns'], results['equity_curve']
                            )
                            enhanced_report = enhanced_metrics.generate_enhanced_report(
                                enhanced_metrics_result, f"{pair[0]}-{pair[1]}"
                            )
                            logger.info(enhanced_report)
                        
                        if 'detailed_trades' in results and results['detailed_trades']:
                            all_trades.extend(results['detailed_trades'])
                        
                        # Aggregate portfolio equity across pairs
                        pair_equity = results['equity_curve']
                        if isinstance(pair_equity, pd.DataFrame):
                            pair_equity = pair_equity.sum(axis=1)

                        if portfolio_equity is None:
                            portfolio_equity = pair_equity
                        else:
                            portfolio_equity = portfolio_equity.add(pair_equity, fill_value=0)

        # Aggregated Analytics and Saving
        if all_trades:
            logger.info("Aggregating and saving backtest analytics...")
            
            # Calculate summary metrics
            total_trades = len(all_trades)
            winning_trades = sum(1 for t in all_trades if t.get('pnl', 0) > 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            total_pnl = sum(t.get('pnl', 0) for t in all_trades)
            
            # Calculate portfolio-level metrics from VectorBT results
            portfolio_metrics = {}
            if portfolio_equity is not None and not portfolio_equity.empty:
                try:
                    # Ensure portfolio_equity is a Series
                    if isinstance(portfolio_equity, pd.DataFrame):
                        portfolio_equity = portfolio_equity.sum(axis=1)
                    
                    # Calculate returns with proper handling
                    portfolio_returns = portfolio_equity.pct_change().fillna(0)
                    
                    # Remove any infinite values
                    portfolio_returns = portfolio_returns.replace([np.inf, -np.inf], 0)
                    
                    if not portfolio_returns.empty and portfolio_returns.std() > 0:
                        portfolio_metrics = enhanced_metrics.calculate_comprehensive_metrics(portfolio_returns, portfolio_equity)
                    else:
                        logger.warning("Portfolio returns are empty or have zero volatility")
                except Exception as e:
                    logger.error(f"Error calculating portfolio metrics: {e}")
                    portfolio_metrics = {}
            
            # Prepare analytics dictionary
            backtest_analytics = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'portfolio_metrics': portfolio_metrics,
                'trades': all_trades,
                'timestamp': pd.Timestamp.now().isoformat(),
                'selected_pairs': [f"{pair[0]}-{pair[1]}" for pair in selected_pairs],
                'config_used': {
                    'entry_threshold': config['signals'].get('entry_threshold', 2.0),
                    'exit_threshold': config['signals'].get('exit_threshold', 0.5),
                    'lookback': config['signals'].get('lookback', 20),
                    'initial_capital': config['backtest'].get('initial_capital', 1_000_000)
                }
            }

            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                elif isinstance(obj, pd.Series):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict('records')
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif hasattr(obj, 'item'):  # Handle numpy scalars
                    return obj.item()
                elif hasattr(obj, '__dict__'):  # Handle objects with __dict__
                    return str(obj)
                return obj

            # Save to JSON
            try:
                with open('backtest_analytics.json', 'w') as f:
                    json.dump(backtest_analytics, f, indent=4, default=convert_numpy_types)
                logger.info("Successfully saved backtest analytics to 'backtest_analytics.json'")
                
                # Also save individual pair results
                for pair, results in backtest_results.items():
                    if 'portfolio' in results:
                        portfolio = results['portfolio']
                        pair_analytics = portfolio.stats()
                        pair_filename = f"pair_results_{pair[0]}_{pair[1]}.json"
                        with open(pair_filename, 'w') as f:
                            json.dump(pair_analytics, f, indent=4, default=convert_numpy_types)
                        logger.info(f"Saved {pair[0]}-{pair[1]} results to {pair_filename}")
                        
            except Exception as e:
                logger.error(f"Error saving analytics to JSON: {e}")
        
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
        if portfolio_equity is not None and not portfolio_equity.empty:
            logger.info("Calculating enhanced portfolio metrics...")
            try:
                # Ensure portfolio_equity is a Series
                if isinstance(portfolio_equity, pd.DataFrame):
                    portfolio_equity = portfolio_equity.sum(axis=1)
                
                portfolio_returns = portfolio_equity.pct_change().fillna(0)
                portfolio_returns = portfolio_returns.replace([np.inf, -np.inf], 0)
                
                if not portfolio_returns.empty and portfolio_returns.std() > 0:
                    portfolio_metrics = enhanced_metrics.calculate_comprehensive_metrics(
                        portfolio_returns, portfolio_equity
                    )
                    
                    portfolio_report = enhanced_metrics.generate_enhanced_report(
                        portfolio_metrics, "PORTFOLIO"
                    )
                    logger.info(portfolio_report)
                else:
                    logger.warning("Portfolio returns are empty or have zero volatility - skipping enhanced metrics")
            except Exception as e:
                logger.error(f"Error calculating enhanced portfolio metrics: {e}")
        
        # Generate plots if requested
        if args.save_plots:
            logger.info("Generating plots...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plots_dir = f"plots/{timestamp}"
            os.makedirs(plots_dir, exist_ok=True)
            
            # Generate various plots
            for pair, results in backtest_results.items():
                if 'equity_curve' in results and 'signals' in results:
                    try:
                        plot_generator.plot_vectorbt_analysis(
                            results['prices'],
                            results['signals']['entries'],
                            results['signals']['exits'],
                            results['equity_curve'],
                            f"{pair[0]}-{pair[1]}",
                            save_path=f"{plots_dir}/analysis_{pair[0]}_{pair[1]}.png"
                        )
                        logger.info(f"Generated analysis plot for {pair[0]}-{pair[1]}")
                    except Exception as e:
                        logger.error(f"Error generating plot for {pair[0]}-{pair[1]}: {e}")
            
            # Portfolio plots
            if portfolio_equity is not None:
                try:
                    plot_generator.plot_equity_curves(
                        portfolio_equity,
                        "Portfolio",
                        save_path=f"{plots_dir}/portfolio_equity.png"
                    )
                    logger.info("Generated portfolio equity plot")
                except Exception as e:
                    logger.error(f"Error generating portfolio plot: {e}")
            
            # Save portfolio performance plot using VectorBT
            if portfolio_equity is not None:
                try:
                    # Ensure portfolio_equity is a 1-dimensional Series
                    if isinstance(portfolio_equity, pd.DataFrame):
                        portfolio_equity_series = portfolio_equity.sum(axis=1)
                    else:
                        portfolio_equity_series = portfolio_equity
                    
                    # Create a simple portfolio for plotting
                    portfolio_returns = portfolio_equity_series.pct_change().fillna(0)
                    
                    # Use vectorbt to create portfolio plot - use the correct method
                    import vectorbt as vbt
                    # Create a simple equity curve plot instead of portfolio
                    fig = portfolio_equity_series.plot(title="Portfolio Equity Curve")
                    fig.figure.savefig(f"{plots_dir}/portfolio_performance.png", dpi=300, bbox_inches='tight')
                    logger.info("Generated portfolio performance plot")
                except Exception as e:
                    logger.error(f"Error generating VectorBT portfolio plot: {e}")
            
            logger.info(f"All plots saved to {plots_dir}")
        
        # Walk-forward analysis if requested
        if args.walkforward:
            logger.info("Running walk-forward analysis...")
            walkforward_results = _run_walkforward_analysis(
                data, selected_pairs, config, args.walkforward_windows,
                enhanced_pair_selector, vectorbt_backtest, logger
            )
            
            # Log walk-forward results
            if walkforward_results:
                logger.info("Walk-forward analysis completed successfully!")
                logger.info(f"Average out-of-sample Sharpe: {walkforward_results.get('avg_sharpe', 0):.3f}")
                logger.info(f"Average out-of-sample return: {walkforward_results.get('avg_return', 0):.2%}")
                logger.info(f"Strategy consistency: {walkforward_results.get('consistency', 0):.2%}")
        
        logger.info("Enhanced backtesting completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in enhanced engine: {e}")
        raise

def _run_walkforward_analysis(data, selected_pairs, config, n_windows, 
                            enhanced_pair_selector, vectorbt_backtest, logger):
    """Run walk-forward analysis to validate strategy robustness."""
    try:
        # Calculate window size
        total_days = len(data)
        window_size = total_days // n_windows
        overlap = window_size // 2  # 50% overlap
        
        walkforward_results = {
            'windows': [],
            'avg_sharpe': 0,
            'avg_return': 0,
            'consistency': 0
        }
        
        for i in range(n_windows - 1):  # Leave last window for final testing
            # Define training and testing periods
            train_start = i * (window_size - overlap)
            train_end = train_start + window_size
            test_start = train_end
            test_end = min(test_start + window_size, total_days)
            
            logger.info(f"Walk-forward window {i+1}/{n_windows-1}")
            logger.info(f"Training: {data.index[train_start]} to {data.index[train_end-1]}")
            logger.info(f"Testing: {data.index[test_start]} to {data.index[test_end-1]}")
            
            # Split data
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            # Re-select pairs on training data
            train_pair_metrics = {}
            for pair in selected_pairs:
                pair_data = train_data[[pair[0], pair[1]]].dropna()
                if len(pair_data) > 252:  # Minimum data requirement
                    metrics = enhanced_pair_selector.calculate_pair_metrics_enhanced(pair_data)
                    if metrics and metrics.get('meets_criteria', False):
                        train_pair_metrics[pair] = metrics
            
            # Select pairs for this window
            window_pairs = enhanced_pair_selector.select_pairs_enhanced(train_pair_metrics)
            logger.info(f"Selected {len(window_pairs)} pairs for window {i+1}")
            
            # Test on out-of-sample data
            window_results = []
            for pair in window_pairs:
                if pair[0] in test_data.columns and pair[1] in test_data.columns:
                    pair_data = test_data[[pair[0], pair[1]]].dropna()
                    if len(pair_data) > 50:  # Minimum test data
                        results = vectorbt_backtest.run_vectorized_backtest(
                            pair_data.iloc[:, 0],
                            pair_data.iloc[:, 1],
                            lookback=config['signals'].get('lookback', 20),
                            entry_threshold=config['signals'].get('entry_threshold', 2.0),
                            exit_threshold=config['signals'].get('exit_threshold', 0.5)
                        )
                        
                        if results and 'returns' in results:
                            returns = results['returns']
                            if not returns.empty and returns.std() > 0:
                                sharpe = returns.mean() / returns.std() * np.sqrt(252)
                                total_return = (1 + returns).prod() - 1
                                window_results.append({
                                    'pair': pair,
                                    'sharpe': sharpe,
                                    'return': total_return,
                                    'trades': len(results.get('detailed_trades', []))
                                })
            
            # Aggregate window results
            if window_results:
                avg_sharpe = np.mean([r['sharpe'] for r in window_results])
                avg_return = np.mean([r['return'] for r in window_results])
                total_trades = sum([r['trades'] for r in window_results])
                
                walkforward_results['windows'].append({
                    'window': i+1,
                    'pairs': len(window_pairs),
                    'avg_sharpe': avg_sharpe,
                    'avg_return': avg_return,
                    'total_trades': total_trades,
                    'results': window_results
                })
                
                logger.info(f"Window {i+1} results: Sharpe={avg_sharpe:.3f}, Return={avg_return:.2%}, Trades={total_trades}")
        
        # Calculate overall statistics
        if walkforward_results['windows']:
            walkforward_results['avg_sharpe'] = np.mean([w['avg_sharpe'] for w in walkforward_results['windows']])
            walkforward_results['avg_return'] = np.mean([w['avg_return'] for w in walkforward_results['windows']])
            
            # Calculate consistency (percentage of windows with positive Sharpe)
            positive_sharpe_windows = sum(1 for w in walkforward_results['windows'] if w['avg_sharpe'] > 0)
            walkforward_results['consistency'] = positive_sharpe_windows / len(walkforward_results['windows'])
        
        return walkforward_results
        
    except Exception as e:
        logger.error(f"Error in walk-forward analysis: {e}")
        return None

if __name__ == "__main__":
    main() 