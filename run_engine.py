import pandas as pd
import numpy as np
import logging
import argparse
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import modules
from data.fetch_data import fetch_etf_data, DataFetchError
import sys
from pairs.pair_analysis import apply_kalman_filter, calculate_spread_and_zscore, rolling_cointegration, rolling_hurst, rolling_adf
from regime.regime_detection import calculate_volatility, train_hmm, predict_regimes
from config import load_config

from backtest import (
    PairsBacktest,
    BacktestConfig
)
from optimization import grid_search
from plots.visualization import generate_all_plots, generate_live_sim_plots
from utils.pair_filtering import filter_pairs_by_sharpe, export_top_pairs

# --- Configuration ---
# Parameters are loaded from config.yaml at runtime.

# --- Enhancement: Composite Scoring Function ---
def compute_pair_score(coint_p, hurst, adf_p, zscore_vol):
    """Return ``1 - mean`` of the provided metrics with NaNs treated as ``1.0``.

    Parameters
    ----------
    coint_p : float
        P-value from the cointegration test.
    hurst : float
        Estimated Hurst exponent.
    adf_p : float
        P-value from the augmented Dickey-Fuller test.
    zscore_vol : float
        Volatility of the pair's z-score.

    Returns
    -------
    float
        Composite score where lower metric values yield higher scores. ``NaN``
        values are replaced with ``1.0`` before averaging.
    """

    inputs = [coint_p, hurst, adf_p, zscore_vol]
    normalized = [1.0 if pd.isna(x) else x for x in inputs]
    score = 1 - np.mean(normalized)
    return score


def compute_pair_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute normalized composite scores for all pairs."""
    metrics_cols = ["coint_p", "hurst", "adf_p", "zscore_vol"]
    norm = (df[metrics_cols] - df[metrics_cols].min()) / (
        df[metrics_cols].max() - df[metrics_cols].min() + 1e-8
    )
    df["score"] = 1 - norm.mean(axis=1)
    return df


def calculate_adaptive_thresholds(z_score: pd.Series, window: int, entry_quantile: float, exit_quantile: float) -> tuple:
    """Calculate adaptive entry/exit thresholds as rolling quantiles (Series)."""
    entry_threshold = z_score.rolling(window).quantile(entry_quantile).abs()
    exit_threshold = z_score.rolling(window).quantile(exit_quantile).abs()
    return entry_threshold, exit_threshold


def main(config_path="config.yaml", save_plots=False):
    cfg = load_config(config_path)
    ETF_TICKERS = cfg["ETF_TICKERS"]
    START_DATE = cfg["START_DATE"]
    END_DATE = cfg["END_DATE"]
    COINTEGRATION_WINDOW = cfg["COINTEGRATION_WINDOW"]
    COINTEGRATION_PVALUE_THRESHOLD = cfg["COINTEGRATION_PVALUE_THRESHOLD"]
    ZSCORE_WINDOW = cfg["ZSCORE_WINDOW"]
    HMM_N_COMPONENTS = cfg["HMM_N_COMPONENTS"]
    REGIME_VOLATILITY_WINDOW = cfg["REGIME_VOLATILITY_WINDOW"]
    STABLE_REGIME_INDEX = cfg["STABLE_REGIME_INDEX"]
    TOP_N_PAIRS = cfg["TOP_N_PAIRS"]
    SECTOR_FILTER_ENABLED = cfg["SECTOR_FILTER_ENABLED"]
    SECTOR_MAP = cfg["SECTOR_MAP"]
    MANUAL_PAIR_LIST = [tuple(p) for p in cfg.get("MANUAL_PAIR_LIST", [])]
    PAIR_PARAMS = {tuple(k.split("_")): v for k, v in cfg.get("PAIR_PARAMS", {}).items()}


    # --- Data Fetching ---
    logger.info(f"Fetching data for {ETF_TICKERS}...")
    try:
        data = fetch_etf_data(ETF_TICKERS, START_DATE, END_DATE)
        data = data.dropna(axis=1, thresh=int(0.9 * len(data)))
    except DataFetchError as e:
        logger.error(f"Data fetch failed: {e}")
        sys.exit(1)
    
    if data.empty:
        logger.error("No data returned from data source. Exiting.")
        sys.exit(1)
    
    logger.info("Data fetched successfully.")
    logger.info("Available tickers: %s", data.columns.tolist())
    
    # --- Market Regime Detection ---
    logger.info("Detecting market regimes...")
    # Use volatility, rolling beta, and spread entropy as features
    asset_volatilities = data.apply(calculate_volatility, axis=0, window=REGIME_VOLATILITY_WINDOW)
    market_volatility = asset_volatilities.mean(axis=1).dropna()
    
    # Calculate rolling beta (first two ETFs as example)
    def rolling_beta(y, x, window=20):
        return y.rolling(window).cov(x) / x.rolling(window).var()
    if len(data.columns) >= 2:
        beta = rolling_beta(data[data.columns[0]], data[data.columns[1]], window=20)
    else:
        beta = pd.Series(index=data.index, data=0.0)
    
    # Calculate spread entropy (as rolling std of spread)
    def rolling_entropy(y, x, window=20):
        spread = y - x
        return spread.rolling(window).std()
    if len(data.columns) >= 2:
        entropy = rolling_entropy(data[data.columns[0]], data[data.columns[1]], window=20)
    else:
        entropy = pd.Series(index=data.index, data=0.0)
    
    hmm_features = pd.DataFrame({
        'volatility': market_volatility,
        'beta': beta,
        'entropy': entropy
    }).dropna()
    
    if not hmm_features.empty:
        hmm_model = train_hmm(hmm_features, n_components=HMM_N_COMPONENTS)
        regimes = predict_regimes(hmm_model, hmm_features)
        regime_series = pd.Series(regimes, index=hmm_features.index)
        logger.info("Market regimes detected.")
    else:
        logger.warning("Not enough data to calculate market volatility and train HMM. Skipping regime detection.")
        regime_series = pd.Series(index=data.index, data=STABLE_REGIME_INDEX) # Assume stable if no HMM
    
    # --- Pairs Trading Analysis and Signal Generation ---
    logger.info("Performing pairs analysis and generating signals...")
    pairs_data = {}
    metrics_list = []
    
    # --- Use manual pair list if provided ---
    if MANUAL_PAIR_LIST:
        pair_iter = MANUAL_PAIR_LIST
    else:
        from itertools import combinations
        pair_iter = list(combinations(data.columns, 2))
    
    for pair in pair_iter:
        asset1_ticker, asset2_ticker = pair
        # Sector filter: skip if not both in Energy
        if SECTOR_FILTER_ENABLED and (SECTOR_MAP.get(asset1_ticker) != 'Energy' or SECTOR_MAP.get(asset2_ticker) != 'Energy'):
            continue
        if asset1_ticker not in data.columns or asset2_ticker not in data.columns:
            continue
        logger.info(f"Analyzing pair: {asset1_ticker} and {asset2_ticker}")
        price_y = data[asset1_ticker]
        price_X = data[asset2_ticker]
        aligned_prices = pd.concat([price_y, price_X], axis=1).dropna()
        if len(aligned_prices) < 80:
            logger.info(f"Skipping pair {asset1_ticker}-{asset2_ticker}: Not enough data.")
            continue
        price_y_aligned = aligned_prices.iloc[:, 0]
        price_X_aligned = aligned_prices.iloc[:, 1]
        rolling_coint_pvals = rolling_cointegration(price_y_aligned, price_X_aligned, window=60)
        last_coint_p = rolling_coint_pvals.dropna().iloc[-1] if not rolling_coint_pvals.dropna().empty else 1.0
        kf_states, kf_covs = apply_kalman_filter(price_y_aligned, price_X_aligned)
        spread = price_y_aligned - pd.Series(kf_states[:, 1], index=price_X_aligned.index) * price_X_aligned
        rolling_hurst_vals = rolling_hurst(spread.dropna(), window=60)
        rolling_adf_vals = rolling_adf(spread.dropna(), window=60)
        last_hurst = rolling_hurst_vals.dropna().iloc[-1] if not rolling_hurst_vals.dropna().empty else 1.0
        last_adf_p = rolling_adf_vals.dropna().iloc[-1] if not rolling_adf_vals.dropna().empty else 1.0
        z_score = calculate_spread_and_zscore(price_y_aligned, price_X_aligned, kf_states, rolling_window=ZSCORE_WINDOW)
        zscore_vol = z_score.rolling(60).std().dropna().iloc[-1] if z_score.rolling(60).std().dropna().size > 0 else 1.0
        score = compute_pair_score(last_coint_p, last_hurst, last_adf_p, zscore_vol)
        logger.info(
            f"Pair {asset1_ticker}-{asset2_ticker}: coint_p={last_coint_p:.3f}, hurst={last_hurst:.3f}, adf_p={last_adf_p:.3f}, zscore_vol={zscore_vol:.3f}, score={score:.3f}"
        )
        metrics_list.append({
            'pair': (asset1_ticker, asset2_ticker),
            'coint_p': last_coint_p,
            'hurst': last_hurst,
            'adf_p': last_adf_p,
            'zscore_vol': zscore_vol,
            'z_score': z_score,
            'kalman_states': kf_states,
            'rolling_hurst': rolling_hurst_vals,
            'rolling_adf': rolling_adf_vals,
            'rolling_coint_pvals': rolling_coint_pvals
        })
    
    # --- Enhancement: Top-N Pair Selection with Fallback ---
    # Select top 3-5 Sharpe pairs, fallback to zscore_vol < 1.1 if needed
    min_sharpe = cfg.get('MIN_PAIR_SHARPE', 1.0)
    top_n = max(3, min(cfg.get('TOP_N_PAIRS', 5), 5))
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df[metrics_df['coint_p'] <= COINTEGRATION_PVALUE_THRESHOLD]
    metrics_df = compute_pair_scores(metrics_df)
    metrics_df = metrics_df.sort_values("score", ascending=False)
    sharpe_sorted = metrics_df.sort_values("score", ascending=False)
    top_pairs = sharpe_sorted.head(top_n)
    if top_pairs['score'].lt(min_sharpe).all():
        # Fallback: select up to 2-3 pairs with zscore_vol < 1.1
        fallback = metrics_df[metrics_df['zscore_vol'] < 1.1].head(3)
        top_pairs = pd.concat([top_pairs, fallback]).drop_duplicates(subset=['pair']).head(top_n)
    logger.info(f"Top pairs selected: {top_pairs['pair'].tolist()}")
    pairs_data = {row['pair']: {
        'z_score': row['z_score'],
        'kalman_states': row['kalman_states'],
        'rolling_hurst': row['rolling_hurst'],
        'rolling_adf': row['rolling_adf'],
        'rolling_coint_pvals': row['rolling_coint_pvals']
    } for _, row in top_pairs.iterrows()}
    
    logger.info("Pairs analysis complete.")
    
    # --- Apply Regime Filter and Generate Trading Signals ---
    logger.info("Applying regime filter and generating trading signals...")
    trade_signals = {}
    
    # --- Re-entry Cooldown Logic ---
    last_exit_dates = {pair: None for pair in pairs_data}
    entry_counts = {pair: 0 for pair in pairs_data}
    for pair, pair_data in pairs_data.items():
        asset1_ticker, asset2_ticker = pair
        z_score = pair_data['z_score']
        params = PAIR_PARAMS.get(pair, {})
        entry_threshold = params.get('entry_threshold', 1.5)
        exit_threshold = params.get('exit_threshold', 0.5)
        stop_loss_k = params.get('stop_loss_k', 2.0)
        min_days_between_trades = params.get('min_days_between_trades', 5)
        max_holding_days = params.get('max_holding_days', 20)
        allowed_regimes = set(cfg.get('ALLOWED_ENTRY_REGIMES', [0, 1]))
        aligned_z_score, aligned_regimes = z_score.align(regime_series, join='inner')
        if cfg.get("USE_ADAPTIVE_THRESHOLDS", False):
            entry_threshold, exit_threshold = calculate_adaptive_thresholds(
                aligned_z_score,
                cfg.get("ZSCORE_QUANTILE_WINDOW", 60),
                cfg.get("ENTRY_QUANTILE", 0.9),
                cfg.get("EXIT_QUANTILE", 0.5)
            )
        # Generate trading signals
        entry_short = aligned_z_score > entry_threshold
        entry_long = aligned_z_score < -entry_threshold
        exit_signal = abs(aligned_z_score) < exit_threshold
        # Regime filter: allow entry in allowed_regimes
        can_trade = aligned_regimes.isin(allowed_regimes)
        filtered_entry_short = entry_short & can_trade
        filtered_entry_long = entry_long & can_trade
        filtered_exit = exit_signal
        # Re-entry cooldown: suppress entry if last exit < cooldown
        entries = []
        last_exit = None
        for dt in aligned_z_score.index:
            if filtered_entry_short.loc[dt] or filtered_entry_long.loc[dt]:
                if last_exit is not None and (dt - last_exit).days < min_days_between_trades:
                    filtered_entry_short.loc[dt] = False
                    filtered_entry_long.loc[dt] = False
                else:
                    entries.append(dt)
            if filtered_exit.loc[dt]:
                last_exit = dt
        entry_counts[pair] = len(entries)
        # Debug print
        logger.info(f"Pair {pair}: entry_threshold={entry_threshold}, regimes_allowed={allowed_regimes}, entries={entry_counts[pair]}")
        signals = pd.DataFrame({
            'z_score': aligned_z_score,
            'regime': aligned_regimes,
            'entry_short': filtered_entry_short,
            'entry_long': filtered_entry_long,
            'exit': filtered_exit,
            'stop_loss_k': stop_loss_k,
            'max_holding_days': max_holding_days
        }, index=aligned_z_score.index)
        trade_signals[pair] = signals
    
    logger.info("Regime-filtered signals generated.")
    
    # --- Run Backtest ---
    logger.info("Running backtest across all pairs...")
    config = cfg.get("backtest", BacktestConfig())
    if isinstance(config, dict):
        config = BacktestConfig(**config)
    
    # Codex's improved loop: run backtest per pair and aggregate
    pair_results = []
    
    for pair, signals in trade_signals.items():
        ticker_y, ticker_X = pair
        logger.info(f"Backtesting pair: {ticker_y}-{ticker_X}")
    
        # Use only the dates where both price series are available
        prices = data.loc[:, [ticker_y, ticker_X]].dropna()
        prices.columns = [ticker_y, ticker_X]
    
        # Attach metrics to signals for enhanced trade logging
        metrics_row = metrics_df[metrics_df['pair'] == pair]
        if not metrics_row.empty:
            coint_p = float(metrics_row['coint_p'].iloc[0])
            adf_p = float(metrics_row['adf_p'].iloc[0])
            hurst = float(metrics_row['hurst'].iloc[0])
        else:
            coint_p = adf_p = hurst = None
        # Add these to the signals DataFrame for each date
        signals = signals.copy()
        signals['coint_p'] = coint_p
        signals['adf_p'] = adf_p
        signals['hurst'] = hurst
    
        backtest = PairsBacktest(config)
        backtest.prices = prices
        backtest.signals = {pair: signals}  # For exit z-score logging
        backtest.run_backtest(prices, {pair: signals}, regime_series)
        metrics = backtest.get_performance_metrics()
        pair_results.append({'pair': pair, 'metrics': metrics, 'backtest': backtest})
        logger.info(f"Metrics for {ticker_y}-{ticker_X}: {metrics}")
    
    
    def aggregate_results(results, config):
        if not results:
            backtest = PairsBacktest(config)
            metrics = {}
            return backtest, metrics
    
        combined_returns = pd.Series(dtype=float)
        all_trades = []
        for r in results:
            if combined_returns.empty:
                combined_returns = r['backtest'].daily_returns
            else:
                combined_returns = combined_returns.add(r['backtest'].daily_returns, fill_value=0)
            all_trades.extend(r['backtest'].trades)
    
        # Initialize equity curve with proper float dtype
        equity_curve = pd.Series(index=combined_returns.index, dtype=float)
        equity_curve.iloc[0] = config.initial_capital
        
        # Calculate cumulative equity
        for i in range(1, len(combined_returns)):
            equity_curve.iloc[i] = equity_curve.iloc[i-1] + combined_returns.iloc[i]
    
        # Ensure no NaNs in equity curve using ffill() and bfill()
        equity_curve = equity_curve.ffill().bfill()
    
        backtest = PairsBacktest(config)
        backtest.daily_returns = combined_returns
        backtest.equity_curve = equity_curve
        backtest.trades = all_trades
    
        metrics = backtest.get_performance_metrics()
        return backtest, metrics
    
    # Aggregate backtest results and assign for use later
    backtest, overall_metrics = aggregate_results(pair_results, config)
    combined_equity_curve = backtest.equity_curve
    combined_daily_returns = backtest.daily_returns
    all_trades = backtest.trades
    metrics = overall_metrics
    
    # --- Sharpe Filtering ---
    min_sharpe = cfg.get('MIN_PAIR_SHARPE', 1.0)
    top_n = cfg.get('TOP_N_PAIRS', 5)
    
    # Sort pairs by Sharpe ratio
    pair_results.sort(key=lambda x: x['metrics'].get('sharpe_ratio', -np.inf), reverse=True)
    
    # If no pairs meet minimum Sharpe, take top N instead
    filtered_pair_results = [r for r in pair_results if r['metrics'].get('sharpe_ratio', -np.inf) >= min_sharpe]
    if not filtered_pair_results:
        logger.warning(f"No pairs met Sharpe >= {min_sharpe}. Using top {top_n} pairs instead.")
        filtered_pair_results = pair_results[:top_n]
    else:
        # Still limit to top N even if some meet threshold
        filtered_pair_results = filtered_pair_results[:top_n]

    # --- Export Top Pairs if CLI flag set ---
    if getattr(args, 'export_top_pairs', False):
        export_top_pairs(filtered_pair_results, top_n=cfg.get('TOP_N_PAIRS', 5), out_path='selected_pairs.yaml')
        logger.info("Top pairs exported to selected_pairs.yaml")

    # Use only filtered pairs for analytics/plots
    backtest, overall_metrics = aggregate_results(filtered_pair_results, config)
    combined_equity_curve = backtest.equity_curve
    combined_daily_returns = backtest.daily_returns
    all_trades = backtest.trades
    metrics = overall_metrics
    
    logger.info("Total trades: %s", len(all_trades))
    logger.info(
        "Closed trades: %s",
        len([t for t in all_trades if t.exit_date is not None]),
    )
    logger.info("Nonzero daily returns: %s", (combined_daily_returns != 0).sum())
    logger.info(
        "Equity curve min/max: %s, %s",
        combined_equity_curve.min(),
        combined_equity_curve.max(),
    )
    logger.debug(combined_equity_curve.head(10))
    logger.debug(combined_daily_returns.head(10))
    
    # --- Generate Performance Reports ---
    logger.info("Generating performance reports...")
    
    # Clean equity curve: fill NaN if any
    combined_equity_curve = combined_equity_curve.astype(float)
    if combined_equity_curve.isnull().any():
        combined_equity_curve = combined_equity_curve.ffill().bfill()
    
    # Calculate drawdown on combined equity
    if not combined_equity_curve.empty and combined_equity_curve.notnull().all():
        rolling_max = combined_equity_curve.cummax()
        drawdown = (combined_equity_curve - rolling_max) / rolling_max
    else:
        drawdown = pd.Series(index=combined_equity_curve.index, data=np.nan)
    
    # Calculate performance metrics
    closed_trades = [t for t in all_trades if t.exit_date is not None]
    winning_trades = [t for t in closed_trades if t.pnl and t.pnl > 0]
    annualized_return = combined_daily_returns.mean() * 252
    annualized_vol = combined_daily_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
    max_drawdown = drawdown.min()
    win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
    holding_periods = [(t.exit_date - t.entry_date).days for t in closed_trades]
    avg_holding_period = np.mean(holding_periods) if holding_periods else 0
    metrics = {
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_holding_period': avg_holding_period,
        'total_trades': len(closed_trades),
        'winning_trades': len(winning_trades)
    }
    
    # Save trade history
    trades_df = pd.DataFrame([vars(t) for t in all_trades])
    trades_df.to_csv('trade_history.csv', index=False)
    
    
    logger.info("\nBacktest complete. Performance metrics:")
    for metric, value in metrics.items():
        if 'rate' in metric.lower() or 'drawdown' in metric.lower():
            logger.info(f"{metric}: {value*100:.1f}%")
        else:
            logger.info(f"{metric}: {value:.2f}")
    
    logger.info("\nTrade history saved to trade_history.csv")
    
    logger.info("Initial setup complete. Files created in meanr_engine directory.")
    logger.info("Next steps include refining the backtesting loop, risk management, and performance metrics calculation.")
    
    # --- TODO: Backtesting Loop and Metrics Enhancements (Future Step) ---
    # Expand the existing backtest loop to support more detailed trade simulation,
    # including position management, stop-loss/take-profit logic, and advanced
    # statistics such as drawdown and Sharpe ratio. Iterate through dates while
    # considering the current regime to update portfolio state and metrics.
    
    # --- TODO: Risk Management (Future Step) ---
    # Integrate volatility scaling for position sizing, CVaR control, regime-dependent filters.
    
    # --- TODO: Extensibility (Future Step) ---
    # Refine modularity, add interfaces for different pair selection, filtering, and execution models.
    
    # --- TODO: Output Refinements (Future Step) ---
    # Implement signal heatmap, better performance reporting.

    # Generate plots if requested
    if save_plots:
        reports_dir = "reports"
        generate_all_plots(backtest, pair_results, save_path=reports_dir)
        logger.info(f"Plots saved to {reports_dir}/")

def run_grid_search(
    config_path: str,
    entry_thresholds: list[float] | None = None,
    exit_thresholds: list[float] | None = None,
    stop_loss_ks: list[float] | None = None,
) -> None:
    """Run parameter grid search and save results."""
    cfg = load_config(config_path)
    grid_cfg = cfg.get("grid_search", {})

    if entry_thresholds is None:
        entry_thresholds = grid_cfg.get("entry_thresholds", [1.5, 2.0])
    if exit_thresholds is None:
        exit_thresholds = grid_cfg.get("exit_thresholds", [0.1])
    if stop_loss_ks is None:
        stop_loss_ks = grid_cfg.get("stop_loss_ks", [2.0])

    if cfg.get("MANUAL_PAIR_LIST"):
        asset1, asset2 = cfg["MANUAL_PAIR_LIST"][0]
    else:
        asset1, asset2 = cfg["ETF_TICKERS"][:2]
    logger.info(f"Running grid search on pair: {asset1}-{asset2}")

    try:
        prices = fetch_etf_data([asset1, asset2], cfg["START_DATE"], cfg["END_DATE"])
    except DataFetchError as e:
        logger.error(f"Data fetch failed: {e}")
        return

    prices = prices.dropna()
    regimes = pd.Series(index=prices.index, data=0)

    bt_cfg = cfg.get("backtest", BacktestConfig())
    if isinstance(bt_cfg, dict):
        bt_cfg = BacktestConfig(**bt_cfg)

    df = grid_search.grid_search(
        prices[[asset1, asset2]],
        grid_search.generate_signals,
        regimes,
        entry_thresholds,
        exit_thresholds,
        stop_loss_ks,
        base_config=bt_cfg,
    )

    if df.empty:
        logger.warning("Grid search returned no results")
    else:
        print(df)
        df.to_csv("grid_search_results.csv", index=False)
        logger.info("Results saved to grid_search_results.csv")

def run_live_sim(config_path: str, save_plots: bool = False) -> None:
    """Run live trading simulation with walk-forward updates."""
    cfg = load_config(config_path)
    logger.info("Starting live trading simulation...")
    
    # Initialize data structures
    portfolio = {}
    active_trades = {}
    daily_pnl = pd.Series(dtype=float)
    equity_curve = pd.Series(dtype=float)
    
    # Get initial data
    data = fetch_etf_data(cfg["ETF_TICKERS"], cfg["START_DATE"], cfg["END_DATE"])
    data = data.dropna(axis=1, thresh=int(0.9 * len(data)))
    
    # Load saved pairs if available
    try:
        with open('selected_pairs.yaml', 'r') as f:
            saved_pairs = yaml.safe_load(f)
        logger.info(f"Loaded {len(saved_pairs)} saved pairs")
    except FileNotFoundError:
        saved_pairs = None
        logger.warning("No saved pairs found. Will generate new pairs.")
    
    # Walk forward through dates
    for date in data.index:
        logger.info(f"Processing date: {date}")
        
        # Update market regimes
        if len(data.loc[:date]) >= cfg["REGIME_VOLATILITY_WINDOW"]:
            regime = predict_regimes(train_hmm(calculate_volatility(data.loc[:date])), 
                                  calculate_volatility(data.loc[:date]))[-1]
        else:
            regime = cfg["STABLE_REGIME_INDEX"]
        
        # Check for trade exits
        for pair, trade in list(active_trades.items()):
            if should_exit_trade(trade, data.loc[date], regime):
                pnl = close_trade(trade, data.loc[date])
                daily_pnl.loc[date] = daily_pnl.get(date, 0) + pnl
                del active_trades[pair]
        
        # Check for new entries if below max positions
        if len(active_trades) < cfg.get("max_concurrent_positions", 5):
            for pair in saved_pairs or []:
                if pair not in active_trades and can_enter_trade(pair, data.loc[date], regime):
                    trade = open_trade(pair, data.loc[date], regime)
                    active_trades[pair] = trade
        
        # Update equity curve
        if date == data.index[0]:
            equity_curve.loc[date] = cfg["initial_capital"]
        else:
            equity_curve.loc[date] = equity_curve.iloc[-1] + daily_pnl.get(date, 0)
    
    # Generate final reports
    metrics = calculate_performance_metrics(daily_pnl, equity_curve)
    logger.info("\nLive simulation complete. Performance metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.2f}")
    
    if save_plots:
        generate_live_sim_plots(equity_curve, daily_pnl, active_trades, save_path="reports/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the mean reversion engine")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--mode",
        choices=["run", "grid-search", "live-sim"],
        default="run",
        help="Execution mode",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to reports directory",
    )
    parser.add_argument("--entry-thresholds", help="Comma separated thresholds")
    parser.add_argument("--exit-thresholds", help="Comma separated thresholds")
    parser.add_argument("--stop-loss-ks", help="Comma separated multipliers")
    parser.add_argument('--export-top-pairs', action='store_true', help='Export top Sharpe pairs to YAML')

    args = parser.parse_args()

    def _parse_list(s: str | None) -> list[float] | None:
        if s is None:
            return None
        return [float(x) for x in s.split(",") if x]

    if args.mode == "grid-search":
        run_grid_search(
            args.config,
            entry_thresholds=_parse_list(args.entry_thresholds),
            exit_thresholds=_parse_list(args.exit_thresholds),
            stop_loss_ks=_parse_list(args.stop_loss_ks),
        )
    elif args.mode == "live-sim":
        run_live_sim(args.config, save_plots=args.save_plots)
    else:
        main(args.config, save_plots=args.save_plots)

