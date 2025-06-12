import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import modules
from data.fetch_data import fetch_etf_data, DataFetchError
import sys
from pairs.pair_analysis import apply_kalman_filter, calculate_spread_and_zscore, rolling_cointegration, rolling_hurst, rolling_adf
from regime.regime_detection import calculate_volatility, train_hmm, predict_regimes
from backtest import (
    PairsBacktest,
    BacktestConfig,
    plot_equity_curve,
    plot_monthly_returns,
    plot_trade_distribution,
    plot_regime_performance,
    plot_performance_metrics
)

# --- Configuration ---
ETF_TICKERS = [
    "SPY", "QQQ", "IWM", "EFA", "EMB", "GLD", "SLV", "USO", "TLT", "IEF",
    "XOM", "CVX", "SLB", "HAL", "MPC", "VLO", "COP", "EOG", "DVN", "ET", "EPD", "OXY", "MRO"
]
START_DATE = "2020-01-01"
END_DATE = "2023-01-01"
COINTEGRATION_WINDOW = 90  # days
# Maximum allowable cointegration p-value when selecting pairs.
# Pairs with higher values will be ignored.
COINTEGRATION_PVALUE_THRESHOLD = 0.2
ZSCORE_WINDOW = 20  # days for rolling mean/std of spread
HMM_N_COMPONENTS = 3 # Number of market regimes (increased from 2 to 3)
REGIME_VOLATILITY_WINDOW = 20 # days for calculating volatility feature for HMM
# Define the "stable" regime - this will need to be determined empirically
# after training the HMM. For now, let's assume regime 0 is stable.
STABLE_REGIME_INDEX = 0

# --- Enhancement: Top-N Pair Selection Parameter ---
TOP_N_PAIRS = 5

# --- Optional: Sector Filtering and Manual Pair List ---
SECTOR_FILTER_ENABLED = True
SECTOR_MAP = {
    'XOM': 'Energy', 'CVX': 'Energy', 'SLB': 'Energy', 'HAL': 'Energy',
    'MPC': 'Energy', 'VLO': 'Energy', 'COP': 'Energy', 'EOG': 'Energy',
    'DVN': 'Energy', 'ET': 'Energy', 'EPD': 'Energy', 'OXY': 'Energy', 'MRO': 'Energy'
}
MANUAL_PAIR_LIST = [
    ('XOM', 'CVX'),
    ('SLB', 'HAL'),
    ('MPC', 'VLO'),
    ('COP', 'XOM'),
    ('EOG', 'DVN'),
    ('ET', 'EPD'),
    ('OXY', 'MRO'),
]
PAIR_PARAMS = {
    ('XOM', 'CVX'): {'entry_threshold': 1.8, 'exit_threshold': 0.5, 'stop_loss_k': 2.0},
    ('SLB', 'HAL'): {'entry_threshold': 2.0, 'exit_threshold': 0.7, 'stop_loss_k': 1.5},
    ('MPC', 'VLO'): {'entry_threshold': 1.9, 'exit_threshold': 0.6, 'stop_loss_k': 2.2},
    ('COP', 'XOM'): {'entry_threshold': 2.1, 'exit_threshold': 0.5, 'stop_loss_k': 1.8},
    ('EOG', 'DVN'): {'entry_threshold': 2.0, 'exit_threshold': 0.7, 'stop_loss_k': 2.0},
    ('ET', 'EPD'): {'entry_threshold': 1.7, 'exit_threshold': 0.5, 'stop_loss_k': 1.6},
    ('OXY', 'MRO'): {'entry_threshold': 2.2, 'exit_threshold': 0.8, 'stop_loss_k': 2.1},
}

# --- Enhancement: Composite Scoring Function ---
def compute_pair_score(coint_p, hurst, adf_p, zscore_vol):
    """Return ``1 - np.nanmean`` of the provided metrics.

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
        values are ignored via ``np.nanmean``.
    """

    values = [coint_p, hurst, adf_p, zscore_vol]
    score = 1 - np.nanmean(values)
    return score


def compute_pair_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute normalized composite scores for all pairs."""
    metrics_cols = ["coint_p", "hurst", "adf_p", "zscore_vol"]
    norm = (df[metrics_cols] - df[metrics_cols].min()) / (
        df[metrics_cols].max() - df[metrics_cols].min() + 1e-8
    )
    df["score"] = 1 - norm.mean(axis=1)
    return df

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

# --- Enhancement: Top-N Pair Selection ---
metrics_df = pd.DataFrame(metrics_list)
metrics_df = metrics_df[metrics_df['coint_p'] <= COINTEGRATION_PVALUE_THRESHOLD]
metrics_df = compute_pair_scores(metrics_df)
metrics_df = metrics_df.sort_values("score", ascending=False)
top_pairs = metrics_df.head(TOP_N_PAIRS)
if top_pairs.empty:
    logger.warning("No pairs meet the cointegration p-value threshold.")
logger.info(f"Top {TOP_N_PAIRS} pairs selected: {top_pairs['pair'].tolist()}")

# Only keep top-N pairs for signal generation and backtesting
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

for pair, pair_data in pairs_data.items():
    asset1_ticker, asset2_ticker = pair
    z_score = pair_data['z_score']
    # Per-pair parameter overrides
    params = PAIR_PARAMS.get(pair, {})
    entry_threshold = params.get('entry_threshold', 2.0)
    exit_threshold = params.get('exit_threshold', 0.1)
    stop_loss_k = params.get('stop_loss_k', 2.0)
    # Align Z-score with the regime series index
    aligned_z_score, aligned_regimes = z_score.align(regime_series, join='inner')
    dynamic_threshold = entry_threshold if not isinstance(entry_threshold, dict) else entry_threshold.get('default', 2.0)
    stable_mask = aligned_regimes == STABLE_REGIME_INDEX
    unstable_mask = aligned_regimes != STABLE_REGIME_INDEX
    entry_short = ((stable_mask) & (aligned_z_score > dynamic_threshold)) | \
                  ((unstable_mask) & (aligned_z_score > dynamic_threshold))
    entry_long = ((stable_mask) & (aligned_z_score < -dynamic_threshold)) | \
                 ((unstable_mask) & (aligned_z_score < -dynamic_threshold))
    exit_signal = abs(aligned_z_score) < exit_threshold
    can_trade = (aligned_regimes == STABLE_REGIME_INDEX)
    filtered_entry_short = entry_short & can_trade
    filtered_entry_long = entry_long & can_trade
    filtered_exit = exit_signal
    signals = pd.DataFrame({
        'z_score': aligned_z_score,
        'regime': aligned_regimes,
        'entry_short': filtered_entry_short,
        'entry_long': filtered_entry_long,
        'exit': filtered_exit,
        'stop_loss_k': stop_loss_k
    }, index=aligned_z_score.index)
    trade_signals[pair] = signals

logger.info("Regime-filtered signals generated.")

# --- Run Backtest ---
logger.info("Running backtest across all pairs...")
config = BacktestConfig(
    initial_capital=1_000_000,
    target_volatility=0.10,
    slippage_bps=2.0,
    commission_bps=1.0,
    stop_loss_k=2.0,
    zscore_entry_threshold=2.0,
    zscore_exit_threshold=0.1
)

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

    equity_curve = pd.Series(index=combined_returns.index, dtype=float)
    for i, (date, daily_pnl) in enumerate(combined_returns.items()):
        if i == 0:
            equity_curve[date] = config.initial_capital + daily_pnl
        else:
            prev_date = combined_returns.index[i - 1]
            equity_curve[date] = equity_curve[prev_date] + daily_pnl

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

# Plot equity curve and drawdown
plot_equity_curve(combined_equity_curve, drawdown)

# Plot monthly returns heatmap
plot_monthly_returns(combined_daily_returns)

# Plot trade distribution
plot_trade_distribution([vars(t) for t in all_trades])

# Plot regime-specific performance
plot_regime_performance(combined_daily_returns, regime_series)

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

plot_performance_metrics(metrics)

# Save trade history
trades_df = pd.DataFrame([vars(t) for t in all_trades])
trades_df.to_csv('trade_history.csv', index=False)

# --- Detailed Analytics Output ---
analytics = {}
analytics['total_net_pnl'] = float(trades_df['pnl'].sum())
analytics['average_pnl'] = float(trades_df['pnl'].mean())
analytics['average_pnl_long'] = float(trades_df[trades_df['direction'] == 'long']['pnl'].mean()) if not trades_df[trades_df['direction'] == 'long'].empty else None
analytics['average_pnl_short'] = float(trades_df[trades_df['direction'] == 'short']['pnl'].mean()) if not trades_df[trades_df['direction'] == 'short'].empty else None
analytics['win_rate'] = float((trades_df['pnl'] > 0).mean())
analytics['pnl_std'] = float(trades_df['pnl'].std())
analytics['sharpe_ratio'] = float(sharpe_ratio)
analytics['equity_curve'] = combined_equity_curve.dropna().astype(float).to_dict()
analytics['trade_pnl_histogram'] = np.histogram(trades_df['pnl'].dropna(), bins=20)[0].tolist()
with open('backtest_analytics.json', 'w') as f:
    json.dump(analytics, f, indent=2)
logger.info("\nAnalytics saved to backtest_analytics.json")

logger.info("\nBacktest complete. Performance metrics:")
for metric, value in metrics.items():
    if 'rate' in metric.lower() or 'drawdown' in metric.lower():
        logger.info(f"{metric}: {value*100:.1f}%")
    else:
        logger.info(f"{metric}: {value:.2f}")

logger.info("\nTrade history saved to trade_history.csv")

# --- Visualization/Initial Output ---
logger.info("Generating plots...")
# Plot Z-score with entry/exit signals and regime overlay for a few pairs
num_pairs_to_plot = min(3, len(trade_signals))

for i, (pair, signals) in enumerate(list(trade_signals.items())[:num_pairs_to_plot]):
    asset1_ticker, asset2_ticker = pair
    z_score = signals['z_score']
    regimes = signals['regime']
    entry_short = signals['entry_short']
    entry_long = signals['entry_long']
    exit_signal = signals['exit']

    plt.figure(figsize=(14, 7))
    ax1 = plt.gca()

    # Plot Z-score
    ax1.plot(z_score.index, z_score, label='Z-score', color='blue')
    ax1.axhline(2, color='red', linestyle='--', label='Entry Short (Z=2)')
    ax1.axhline(-2, color='green', linestyle='--', label='Entry Long (Z=-2)')
    ax1.axhline(0, color='gray', linestyle='--', label='Exit (Z=0)')
    ax1.set_ylabel('Z-score', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title(f'Z-score and Regime Overlay for {asset1_ticker}-{asset2_ticker}')

    # Overlay regimes
    ax2 = ax1.twinx()
    # Plot regimes as stepped line or background color
    # Using scatter plot aligned to the date index where regimes and z_scores exist
    regime_colors = ['purple' if r == STABLE_REGIME_INDEX else 'orange' for r in regimes]
    ax2.scatter(regimes.index, regimes, c=regime_colors, alpha=0.3, label='Regime', s=5)
    ax2.set_ylabel('Regime Index', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.set_yticks(np.unique(regimes))

    # Plot trading signals
    ax1.scatter(z_score[entry_short].index, z_score[entry_short], marker='^', color='red', s=100, label='Short Entry (Filtered)')
    ax1.scatter(z_score[entry_long].index, z_score[entry_long], marker='v', color='green', s=100, label='Long Entry (Filtered)')
    # Plot exits - might need position tracking in backtest loop to filter properly
    # For now, just show where the Z-score is near zero for visualization
    # ax1.scatter(z_score[exit_signal].index, z_score[exit_signal], marker='x', color='black', s=50, label='Exit Signal')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.grid(True)
    plt.show()

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
