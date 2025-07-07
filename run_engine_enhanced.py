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
from datetime import datetime, time
import os
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.data_loader import DataLoader
from core.enhanced_pair_selection import EnhancedPairSelector, StatisticalThresholds
from core.signal_generation import SignalGenerator
from core.pair_monitor import PairHealthMonitor
from regime.regime_detection import RegimeDetector
from core.regime_filters import RegimeFilter, RegimeFilterConfig
from backtest.vectorbt_backtest import VectorBTBacktest, VectorBTConfig
from core.enhanced_metrics import EnhancedMetricsCalculator, EnhancedMetricsConfig
from core.diagnostics import TradeDiagnostics
from core.plotting import PlotGenerator

from core.behavioral_execution import apply_behavioral_execution_filter
from core.walkforward import walk_forward_backtest
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
    """Main function for enhanced mean reversion pairs trading engine."""
    parser = argparse.ArgumentParser(description='Enhanced Mean Reversion Pairs Trading Engine')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file path')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to disk')
    parser.add_argument('--optimize', action='store_true', help='Run parameter optimization')
    parser.add_argument('--vectorbt-only', action='store_true', help='Use only vectorbt backtesting')
    parser.add_argument('--statistical-report', action='store_true', help='Generate detailed statistical report')
    parser.add_argument('--walkforward', action='store_true', help='Run walk-forward analysis')
    parser.add_argument('--walkforward-windows', type=int, default=5, help='Number of walk-forward windows')
#conflict resolved here  wkkj7n-codex/modify-backtest-engine-with-slippage-and-filters
    parser.add_argument('--walkforward-output', type=str, default='walkforward_stats.csv', help='Output CSV for walk-forward stats')

    parser.add_argument('--behavioral-execution', action='store_true', help='Enable behavioral execution timing')
    parser.add_argument('--override-execution-window', action='store_true', help='Allow trading outside behavioral window')
#conflict resolved here 
    
#conflict resolved here  main
    parser.add_argument("--include-unhealthy", action="store_true", help="Use pairs that fail health criteria")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()
    logger.info("RUNNING LATEST VERSION of run_engine_enhanced.py")
    logger.info("Starting Enhanced Mean Reversion Pairs Trading Engine")
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        logger.error("Failed to load configuration")
        return
    if args.include_unhealthy:
        config.setdefault("backtest", {})["include_unhealthy_pairs"] = True
#conflict resolved here  wkkj7n-codex/modify-backtest-engine-with-slippage-and-filters

    if args.behavioral_execution:
        config.setdefault("backtest", {})["behavioral_execution"] = True
    if args.override_execution_window:
        config.setdefault("backtest", {})["behavioral_execution"] = False
#conflict resolved here 
#conflict resolved here  main
    
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
        pair_monitor = PairHealthMonitor()
        
        # Regime detector
        regime_detector = RegimeDetector(config['regime'])
        
        # Initialize regime filters
        regime_filter_config = RegimeFilterConfig(
            use_vix_filter=config.get('regime_filtering', {}).get('use_vix_filter', True),
            vix_threshold_high=config.get('regime_filtering', {}).get('vix_threshold_high', 25.0),
            vix_threshold_low=config.get('regime_filtering', {}).get('vix_threshold_low', 15.0),
            vix_lookback=config.get('regime_filtering', {}).get('vix_lookback', 5),
            use_trend_filter=config.get('regime_filtering', {}).get('use_trend_filter', True),
            trend_window=config.get('regime_filtering', {}).get('trend_window', 60),
            trend_slope_threshold=config.get('regime_filtering', {}).get('trend_slope_threshold', 0.6),
            trend_ma_window=config.get('regime_filtering', {}).get('trend_ma_window', 20),
            use_rolling_sharpe_filter=config.get('regime_filtering', {}).get('use_rolling_sharpe_filter', True),
            rolling_sharpe_window=config.get('regime_filtering', {}).get('rolling_sharpe_window', 60),
            rolling_sharpe_min=config.get('regime_filtering', {}).get('rolling_sharpe_min', 0.2),
            rolling_sharpe_lookback=config.get('regime_filtering', {}).get('rolling_sharpe_lookback', 252),
            use_market_regime_filter=config.get('regime_filtering', {}).get('use_market_regime_filter', True),
            market_regime_window=config.get('regime_filtering', {}).get('market_regime_window', 20),
            min_regime_stability=config.get('regime_filtering', {}).get('min_regime_stability', 0.7)
        )
        
        regime_filter = RegimeFilter(regime_filter_config)
        logger.info("Initialized regime filters")
        
        # VectorBT backtest engine
        vectorbt_config = VectorBTConfig(
            initial_capital=config['backtest'].get('initial_capital', 1_000_000),
            fees=config['backtest'].get('commission_bps', 1.0) / 10000,
            slippage=config['backtest'].get('slippage_bps', 15.0) / 10000,
            max_concurrent_positions=config['backtest'].get('max_concurrent_positions', 5),
            regime_scaling=config['backtest'].get('regime_scaling', True),
            regime_volatility_multiplier=config['backtest'].get('regime_volatility_multiplier', 1.0),
            regime_trend_multiplier=config['backtest'].get('regime_trend_multiplier', 1.0),
            execution_timing=config['backtest'].get('execution_timing', False),
            execution_penalty_factor=config['backtest'].get('execution_penalty_factor', 1.05)
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
        spy_series = data['SPY'] if 'SPY' in data.columns else None
        
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
        
        include_unhealthy = config.get('backtest', {}).get('include_unhealthy_pairs', False)
        if include_unhealthy:
            selected_pairs = [pair for pair, m in pair_metrics.items() if m and not m.get('meets_criteria', False)]
            logger.info(f"Using {len(selected_pairs)} unhealthy pairs for stress testing")
        else:
            selected_pairs = enhanced_pair_selector.select_pairs_enhanced(pair_metrics)
            logger.info(f"Selected {len(selected_pairs)} pairs for trading")
        
        # Generate signals with regime filtering
        logger.info("Generating trading signals with regime filtering...")
        all_signals = {}
        regime_filtered_signals = {}
        pair_health_status = {}
        
        for pair in selected_pairs:
            pair_data = data_loader.get_pair_data(pair)
            if pair_data is not None:
                # Calculate spread for regime filtering
                spread = pair_data.iloc[:, 0] - pair_data.iloc[:, 1]
                health_df = pair_monitor.evaluate(spread)
                pair_health_status[pair] = health_df
#conflict resolved here wkkj7n-codex/modify-backtest-engine-with-slippage-and-filters
#conflict resolved here 
# conflict markers removed here  f3usdw-codex/modify-backtest-engine-with-slippage-and-filters
#conflict resolved here > main
                if data_loader.ohlc_data is not None and pair[0] in data_loader.ohlc_data['Close'] and pair[1] in data_loader.ohlc_data['Close']:
                    adv1 = (data_loader.ohlc_data['Close'][pair[0]] * data_loader.ohlc_data['Volume'][pair[0]]).rolling(30).mean().iloc[-1]
                    adv2 = (data_loader.ohlc_data['Close'][pair[1]] * data_loader.ohlc_data['Volume'][pair[1]]).rolling(30).mean().iloc[-1]
                    adv = float(np.nanmean([adv1, adv2]))
                else:
                    adv = float('nan')
                if not health_df.dropna().empty:
                    last = health_df.dropna().iloc[-1]
#conflict resolved here  wkkj7n-codex/modify-backtest-engine-with-slippage-and-filters
                    pair_monitor.log_pair_health(
                        f"{pair[0]}-{pair[1]}",
                        last['adf_pvalue'],
                        last['hurst'],
                        adv,
                        bool(last['healthy']),
                        vol_z=last.get('vol_zscore', np.nan),
                        spread_std=last.get('spread_std', np.nan),
                        unstable=bool(last.get('unstable_cointegration', False)),
                        excessive_vol=bool(last.get('excessive_volatility', False)),
                    )
#conflict resolved here 
                    pair_monitor.log_pair_health(f"{pair[0]}-{pair[1]}", last['adf_pvalue'], last['hurst'], adv, bool(last['healthy']))
#conflict resolved here  main

                # Use signal generator for signal generation
                signals = signal_generator.generate_signals(
                    pair_data,
                    pair,
                    regime_filter.vix_data if hasattr(regime_filter, "vix_data") and regime_filter.vix_data is not None else pd.Series(0, index=pair_data.index),
#conflict resolved here  wkkj7n-codex/modify-backtest-engine-with-slippage-and-filters
#conflict resolved here 
# conflict markers removed here  f3usdw-codex/modify-backtest-engine-with-slippage-and-filters
#conflict resolved here main
                    spy_series,
                    health_df,
                )
                if config['backtest'].get('behavioral_execution', False):
                    exec_times = pair_data.index.to_series().apply(lambda d: datetime.combine(d, time(15, 50)))
                    signals['entries'] = apply_behavioral_execution_filter(signals['entries'], exec_times)
#conflict resolved here wkkj7n-codex/modify-backtest-engine-with-slippage-and-filters
#conflict resolved here 
                    health_df,
                )
# conflict markers removed here  main
#conflict resolved here main
                if signals is not None and not signals.empty:
                    all_signals[pair] = signals
                    
                    # Apply regime filters to signals
                    filtered_signals = signals.copy()
                    regime_info_list = []
                    
                    for date in signals.index:
                        if date in spread.index:
                            should_trade, regime_info = regime_filter.should_trade(spread, date)
                            regime_info_list.append(regime_info)
                            
                            if not should_trade:
                                # Zero out signals for unfavorable regimes
                                filtered_signals.loc[date, 'position'] = 0
                                filtered_signals.loc[date, 'entry_signal'] = 0
                                filtered_signals.loc[date, 'exit_signal'] = 0
                            else:
                                # Apply position size multiplier
                                multiplier = regime_info['multiplier']
                                if 'position' in filtered_signals.columns:
                                    filtered_signals.loc[date, 'position'] *= multiplier
                    
                    regime_filtered_signals[pair] = filtered_signals
                    
                    # Log regime filtering summary
                    if regime_info_list:
                        favorable_days = sum(1 for info in regime_info_list if info['should_trade'])
                        total_days = len(regime_info_list)
                        filter_rate = (total_days - favorable_days) / total_days if total_days > 0 else 0
                        logger.info(f"Regime filtering for {pair[0]}-{pair[1]}: {filter_rate:.1%} of signals filtered out")
        
        # Use regime-filtered signals for backtesting
        signals_to_use = regime_filtered_signals if regime_filtered_signals else all_signals
        
        # Run enhanced backtests
        logger.info("Running enhanced backtests...")
        backtest_results = {}
        portfolio_returns = None
        portfolio_equity = None
        all_trades = []
        
        # By default, use vectorbt for backtesting
        for pair in selected_pairs:
            if pair in signals_to_use:
                pair_data = data_loader.get_pair_data(pair)
                if pair_data is not None:
                    logger.info(f"Running vectorbt backtest for {pair[0]}-{pair[1]}")
                    
                    regime_series = all_regimes.get(pair, pd.Series(0, index=pair_data.index))
                    
                    vol1 = None
                    vol2 = None
                    if data_loader.ohlc_data is not None:
                        try:
                            vol1 = data_loader.ohlc_data['Volume'][pair[0]]
                            vol2 = data_loader.ohlc_data['Volume'][pair[1]]
                        except Exception:
                            vol1 = vol2 = None

                    results = vectorbt_backtest.run_vectorized_backtest(
                        pair_data.iloc[:, 0],
                        pair_data.iloc[:, 1],
                        regime_series=regime_series,
                        vix_series=regime_filter.vix_data if hasattr(regime_filter, 'vix_data') else None,
                        spy_series=spy_series,
                        volume1=vol1,
                        volume2=vol2,
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
                        
                        # Aggregate portfolio returns instead of equity
                        if 'returns' in results:
                            pair_returns = results['returns'].fillna(0)
                            if portfolio_returns is None:
                                portfolio_returns = pair_returns.copy()
                            else:
                                aligned_returns = pair_returns.reindex(portfolio_returns.index, fill_value=0)
                                portfolio_returns = portfolio_returns.add(aligned_returns, fill_value=0)

        # Convert aggregated returns to equity curve
        if portfolio_returns is not None:
            portfolio_returns = portfolio_returns.fillna(0)
            portfolio_equity = vectorbt_config.initial_capital * (1 + portfolio_returns).cumprod()

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
            if portfolio_returns is not None and not portfolio_returns.empty:
                try:
                    portfolio_metrics = enhanced_metrics.calculate_comprehensive_metrics(
                        portfolio_returns, portfolio_equity
                    )
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
                'regime_filtering_enabled': config.get('regime_filtering', {}).get('enabled', True),
                'regime_filter_config': config.get('regime_filtering', {}),
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
        if portfolio_returns is not None and not portfolio_returns.empty:
            logger.info("Calculating enhanced portfolio metrics...")
            try:
                portfolio_metrics = enhanced_metrics.calculate_comprehensive_metrics(
                    portfolio_returns, portfolio_equity
                )

                portfolio_report = enhanced_metrics.generate_enhanced_report(
                    portfolio_metrics, "PORTFOLIO"
                )
                logger.info(portfolio_report)
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
            walkforward_results = walk_forward_backtest(
                data,
                selected_pairs,
                config,
                enhanced_pair_selector,
                vectorbt_backtest,
#conflict resolved here wkkj7n-codex/modify-backtest-engine-with-slippage-and-filters
                output_path=args.walkforward_output,
#conflict resolved here  main
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


if __name__ == "__main__":
    main()
