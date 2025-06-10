import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import modules
from data.fetch_data import fetch_etf_data, DataFetchError
import sys
from pairs.pair_analysis import calculate_cointegration, apply_kalman_filter, calculate_spread_and_zscore
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
ETF_TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EMB", "GLD", "SLV", "USO", "TLT", "IEF"]
START_DATE = "2020-01-01"
END_DATE = "2023-01-01"
COINTEGRATION_WINDOW = 90  # days
COINTEGRATION_PVALUE_THRESHOLD = 0.2  # or even 0.3 for testing
ZSCORE_WINDOW = 20  # days for rolling mean/std of spread
HMM_N_COMPONENTS = 2 # Number of market regimes
REGIME_VOLATILITY_WINDOW = 20 # days for calculating volatility feature for HMM
# Define the "stable" regime - this will need to be determined empirically
# after training the HMM. For now, let's assume regime 0 is stable.
STABLE_REGIME_INDEX = 0

# --- Data Fetching ---
print(f"Fetching data for {ETF_TICKERS}...")
try:
    data = fetch_etf_data(ETF_TICKERS, START_DATE, END_DATE)
    data = data.dropna(axis=1, thresh=int(0.9 * len(data)))
except DataFetchError as e:
    print(f"Data fetch failed: {e}")
    sys.exit(1)

if data.empty:
    print("No data returned from data source. Exiting.")
    sys.exit(1)

print("Data fetched successfully.")
print("Available tickers:", data.columns.tolist())

# --- Market Regime Detection ---
print("Detecting market regimes...")
# Use the average volatility across all assets as a market-wide feature
asset_volatilities = data.apply(calculate_volatility, axis=0, window=REGIME_VOLATILITY_WINDOW)
market_volatility = asset_volatilities.mean(axis=1).dropna()

if not market_volatility.empty:
    hmm_model = train_hmm(market_volatility)
    regimes = predict_regimes(hmm_model, market_volatility)
    # Align regimes with the data index
    regime_series = pd.Series(regimes, index=market_volatility.index)
    print("Market regimes detected.")

    # Optional: Analyze predicted regimes and their characteristics (e.g., mean volatility in each regime)
    # mean_vol_per_regime = market_volatility.groupby(regime_series).mean()
    # print("Mean volatility per regime:", mean_vol_per_regime)

else:
    print("Not enough data to calculate market volatility and train HMM. Skipping regime detection.")
    regime_series = pd.Series(index=data.index, data=STABLE_REGIME_INDEX) # Assume stable if no HMM

# --- Pairs Trading Analysis and Signal Generation ---
print("Performing pairs analysis and generating signals...")
pairs_data = {}
# Iterate through all unique pairs
for i in range(len(data.columns)):
    for j in range(i + 1, len(data.columns)):
        asset1_ticker = data.columns[i]
        asset2_ticker = data.columns[j]
        print(f"Analyzing pair: {asset1_ticker} and {asset2_ticker}")

        price_y = data[asset1_ticker]
        price_X = data[asset2_ticker]

        # Ensure price series are aligned and have enough data
        aligned_prices = pd.concat([price_y, price_X], axis=1).dropna()
        if len(aligned_prices) < COINTEGRATION_WINDOW + ZSCORE_WINDOW:
            print(f"Skipping pair {asset1_ticker}-{asset2_ticker}: Not enough data.")
            continue

        # Use aligned prices for subsequent calculations
        price_y_aligned = aligned_prices.iloc[:, 0]
        price_X_aligned = aligned_prices.iloc[:, 1]

        # Check cointegration on the most recent window
        coint_y = price_y_aligned.tail(COINTEGRATION_WINDOW)
        coint_x = price_X_aligned.tail(COINTEGRATION_WINDOW)
        coint_t, coint_p, _ = calculate_cointegration(coint_y, coint_x)
        if coint_p > COINTEGRATION_PVALUE_THRESHOLD:
            print(f"Skipping pair {asset1_ticker}-{asset2_ticker}: cointegration p-value {coint_p} exceeds threshold.")
            continue

        # Apply Kalman Filter
        # Note: Kalman filter needs input shaped (n_samples, n_features)
        kf_states, kf_covs = apply_kalman_filter(price_y_aligned, price_X_aligned)

        # Calculate Spread and Z-score
        z_score = calculate_spread_and_zscore(
            price_y_aligned, price_X_aligned, kf_states, rolling_window=ZSCORE_WINDOW
        )

        # Store pair data
        pairs_data[(asset1_ticker, asset2_ticker)] = {
            'z_score': z_score,
            'kalman_states': kf_states,
            # 'cointegration_p_value': coint_p # Optional: could add rolling cointegration here
        }

print("Pairs analysis complete.")

# --- Apply Regime Filter and Generate Trading Signals ---
print("Applying regime filter and generating trading signals...")
trade_signals = {}

for pair, pair_data in pairs_data.items():
    asset1_ticker, asset2_ticker = pair
    z_score = pair_data['z_score']

    # Align Z-score with the regime series index
    # This ensures we only consider Z-scores on days for which we have regime data
    aligned_z_score, aligned_regimes = z_score.align(regime_series, join='inner')

    # Generate raw signals based on Z-score (simplified entry/exit)
    # Entry: Z-score > 2 (short spread) or Z-score < -2 (long spread)
    # Exit: Z-score crosses 0
    entry_short = aligned_z_score > 2
    entry_long = aligned_z_score < -2
    exit_signal = abs(aligned_z_score) < 0.1 # Close to zero

    # Apply Regime Filter: Only trade in the stable regime
    can_trade = (aligned_regimes == STABLE_REGIME_INDEX)

    # Filtered signals: only trigger if can_trade is True
    filtered_entry_short = entry_short & can_trade
    filtered_entry_long = entry_long & can_trade
    # Exits should happen regardless of regime to close positions
    filtered_exit = exit_signal # & (position_is_open) - requires backtest state

    # Combine signals for plotting/analysis
    signals = pd.DataFrame({
        'z_score': aligned_z_score,
        'regime': aligned_regimes,
        'entry_short': filtered_entry_short,
        'entry_long': filtered_entry_long,
        'exit': filtered_exit
    }, index=aligned_z_score.index)

    trade_signals[pair] = signals

print("Regime-filtered signals generated.")

# --- Run Backtest ---
print("Running backtest across all pairs...")
config = BacktestConfig(
    initial_capital=1_000_000,
    target_volatility=0.10,
    slippage_bps=2.0,
    commission_bps=1.0,
    stop_loss_std=2.0,
    zscore_entry_threshold=2.0,
    zscore_exit_threshold=0.1
)

# Codex's improved loop: run backtest per pair and aggregate
pair_results = []

for pair, signals in trade_signals.items():
    ticker_y, ticker_X = pair
    print(f"Backtesting pair: {ticker_y}-{ticker_X}")

    # Use only the dates where both price series are available
    prices = data.loc[:, [ticker_y, ticker_X]].dropna()
    prices.columns = [ticker_y, ticker_X]

    backtest = PairsBacktest(config)
    backtest.prices = prices
    backtest.prices = prices
    backtest.run_backtest(prices, {pair: signals}, regime_series)
    metrics = backtest.get_performance_metrics()
    pair_results.append({'pair': pair, 'metrics': metrics, 'backtest': backtest})
    print(f"Metrics for {ticker_y}-{ticker_X}: {metrics}")


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

print(f"Total trades: {len(all_trades)}")
print(f"Nonzero daily returns: {(combined_daily_returns != 0).sum()}")
print(f"Equity curve min/max: {combined_equity_curve.min()}, {combined_equity_curve.max()}")

# --- Generate Performance Reports ---
print("Generating performance reports...")

# Calculate drawdown on combined equity
cummax = combined_equity_curve.cummax()
drawdown = (combined_equity_curve - cummax) / cummax

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

print("\nBacktest complete. Performance metrics:")
for metric, value in metrics.items():
    if 'rate' in metric.lower() or 'drawdown' in metric.lower():
        print(f"{metric}: {value*100:.1f}%")
    else:
        print(f"{metric}: {value:.2f}")

print("\nTrade history saved to trade_history.csv")

# --- Visualization/Initial Output ---
print("Generating plots...")
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

print("Initial setup complete. Files created in meanr_engine directory.")
print("Next steps would be to build the full backtesting loop, risk management, and performance metrics calculation.")

# --- TODO: Backtesting Loop and Metrics (Future Step) ---
# Need to implement the actual backtest loop to simulate trades,
# manage positions, apply stop-loss/take-profit, calculate P&L,
# drawdowns, Sharpe ratio, etc.
# This would iterate through the dates, check for signals on each day,
# consider the current regime, execute trades based on rules,
# and update portfolio state.

# --- TODO: Risk Management (Future Step) ---
# Integrate volatility scaling for position sizing, CVaR control, regime-dependent filters.

# --- TODO: Extensibility (Future Step) ---
# Refine modularity, add interfaces for different pair selection, filtering, and execution models.

# --- TODO: Output Refinements (Future Step) ---
# Implement signal heatmap, better performance reporting. 

def plot_regime_performance(returns: pd.Series, regimes: pd.Series, title: str = "Regime-Specific Performance") -> None:
    regime_returns = pd.DataFrame({'returns': returns, 'regime': regimes})
    plt.figure(figsize=(12, 6))
    for regime in regime_returns['regime'].unique():
        mask = regime_returns['regime'] == regime
        cumulative = (regime_returns.loc[mask, 'returns'] + 1).cumprod() - 1
        plt.plot(regime_returns.loc[mask].index, cumulative, label=f'Regime {regime}')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()
