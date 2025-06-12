import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import argparse

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
    BacktestConfig,
    plot_equity_curve,
    plot_monthly_returns,
    plot_trade_distribution,
    plot_regime_performance,
    plot_performance_metrics,
)

# --- Configuration ---
# Parameters are loaded from config.yaml at runtime.

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


def aggregate_pair_results(results, config):
    """Combine per-pair backtest results."""
    if not results:
        backtest = PairsBacktest(config)
        metrics = {}
        return backtest, metrics

    combined_returns = pd.Series(dtype=float)
    all_trades = []
    for r in results:
        if combined_returns.empty:
            combined_returns = r["backtest"].daily_returns
        else:
            combined_returns = combined_returns.add(
                r["backtest"].daily_returns, fill_value=0
            )
        all_trades.extend(r["backtest"].trades)

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


def run_fold(train_data: pd.DataFrame, test_data: pd.DataFrame, cfg: dict, fold_num: int):
    """Run analysis and backtest for a single walk-forward fold."""
    logger.info(
        f"Running fold {fold_num} | train: {train_data.index[0].date()} - {train_data.index[-1].date()} "
        f"test: {test_data.index[0].date()} - {test_data.index[-1].date()}"
    )

    data = pd.concat([train_data, test_data])

    HMM_N_COMPONENTS = cfg["HMM_N_COMPONENTS"]
    REGIME_VOLATILITY_WINDOW = cfg["REGIME_VOLATILITY_WINDOW"]
    STABLE_REGIME_INDEX = cfg["STABLE_REGIME_INDEX"]
    COINTEGRATION_WINDOW = cfg["COINTEGRATION_WINDOW"]
    COINTEGRATION_PVALUE_THRESHOLD = cfg["COINTEGRATION_PVALUE_THRESHOLD"]
    ZSCORE_WINDOW = cfg["ZSCORE_WINDOW"]
    TOP_N_PAIRS = cfg["TOP_N_PAIRS"]
    SECTOR_FILTER_ENABLED = cfg["SECTOR_FILTER_ENABLED"]
    SECTOR_MAP = cfg["SECTOR_MAP"]
    MANUAL_PAIR_LIST = [tuple(p) for p in cfg.get("MANUAL_PAIR_LIST", [])]
    PAIR_PARAMS = {
        tuple(k.split("_")): v for k, v in cfg.get("PAIR_PARAMS", {}).items()
    }

    # --- Market Regime Detection (train on training slice) ---
    asset_volatilities = data.apply(
        calculate_volatility, axis=0, window=REGIME_VOLATILITY_WINDOW
    )
    market_volatility = asset_volatilities.mean(axis=1).dropna()

    def rolling_beta(y, x, window=20):
        return y.rolling(window).cov(x) / x.rolling(window).var()

    if len(data.columns) >= 2:
        beta = rolling_beta(data[data.columns[0]], data[data.columns[1]], window=20)
    else:
        beta = pd.Series(index=data.index, data=0.0)

    def rolling_entropy(y, x, window=20):
        spread = y - x
        return spread.rolling(window).std()

    if len(data.columns) >= 2:
        entropy = rolling_entropy(data[data.columns[0]], data[data.columns[1]], window=20)
    else:
        entropy = pd.Series(index=data.index, data=0.0)

    hmm_features = (
        pd.DataFrame({"volatility": market_volatility, "beta": beta, "entropy": entropy})
        .dropna()
    )

    if hmm_features.empty:
        regime_series = pd.Series(index=data.index, data=STABLE_REGIME_INDEX)
    else:
        train_feat = hmm_features.loc[train_data.index.intersection(hmm_features.index)]
        hmm_model = train_hmm(train_feat, n_components=HMM_N_COMPONENTS)
        all_regs = predict_regimes(hmm_model, hmm_features)
        regime_series = pd.Series(all_regs, index=hmm_features.index)

    # --- Pair Metrics ---
    pairs_data = {}
    metrics_list = []

    if MANUAL_PAIR_LIST:
        pair_iter = MANUAL_PAIR_LIST
    else:
        from itertools import combinations

        pair_iter = list(combinations(data.columns, 2))

    for pair in pair_iter:
        asset1_ticker, asset2_ticker = pair
        if SECTOR_FILTER_ENABLED and (
            SECTOR_MAP.get(asset1_ticker) != "Energy" or SECTOR_MAP.get(asset2_ticker) != "Energy"
        ):
            continue
        if asset1_ticker not in data.columns or asset2_ticker not in data.columns:
            continue

        price_y = data.loc[: test_data.index[-1], asset1_ticker]
        price_X = data.loc[: test_data.index[-1], asset2_ticker]

        aligned = pd.concat([price_y, price_X], axis=1).dropna()
        if len(aligned) < 80:
            continue

        price_y_train = price_y.loc[train_data.index]
        price_X_train = price_X.loc[train_data.index]

        rolling_coint_pvals = rolling_cointegration(
            price_y_train, price_X_train, window=COINTEGRATION_WINDOW
        )
        last_coint_p = (
            rolling_coint_pvals.dropna().iloc[-1]
            if not rolling_coint_pvals.dropna().empty
            else 1.0
        )

        kf_states, _ = apply_kalman_filter(price_y, price_X)
        spread = price_y - pd.Series(kf_states[:, 1], index=price_X.index) * price_X
        rolling_hurst_vals = rolling_hurst(spread.dropna(), window=COINTEGRATION_WINDOW)
        rolling_adf_vals = rolling_adf(spread.dropna(), window=COINTEGRATION_WINDOW)
        last_hurst = (
            rolling_hurst_vals.dropna().iloc[-1]
            if not rolling_hurst_vals.dropna().empty
            else 1.0
        )
        last_adf_p = (
            rolling_adf_vals.dropna().iloc[-1]
            if not rolling_adf_vals.dropna().empty
            else 1.0
        )

        z_score = calculate_spread_and_zscore(
            price_y, price_X, kf_states, rolling_window=ZSCORE_WINDOW
        )
        zscore_vol = (
            z_score.rolling(60).std().dropna().iloc[-1]
            if z_score.rolling(60).std().dropna().size > 0
            else 1.0
        )
        score = compute_pair_score(last_coint_p, last_hurst, last_adf_p, zscore_vol)

        metrics_list.append(
            {
                "pair": (asset1_ticker, asset2_ticker),
                "coint_p": last_coint_p,
                "hurst": last_hurst,
                "adf_p": last_adf_p,
                "zscore_vol": zscore_vol,
                "z_score": z_score,
                "kalman_states": kf_states,
                "rolling_hurst": rolling_hurst_vals,
                "rolling_adf": rolling_adf_vals,
                "rolling_coint_pvals": rolling_coint_pvals,
            }
        )

    metrics_df = pd.DataFrame(metrics_list)
    if metrics_df.empty:
        top_pairs = pd.DataFrame()
    else:
        metrics_df = metrics_df[metrics_df["coint_p"] <= COINTEGRATION_PVALUE_THRESHOLD]
        metrics_df = compute_pair_scores(metrics_df)
        metrics_df = metrics_df.sort_values("score", ascending=False)
        top_pairs = metrics_df.head(TOP_N_PAIRS)

    pairs_data = {
        row["pair"]: {
            "z_score": row["z_score"],
            "kalman_states": row["kalman_states"],
            "rolling_hurst": row["rolling_hurst"],
            "rolling_adf": row["rolling_adf"],
            "rolling_coint_pvals": row["rolling_coint_pvals"],
        }
        for _, row in top_pairs.iterrows()
    }

    # --- Signals ---
    trade_signals = {}
    regime_series_fold = regime_series.loc[test_data.index]

    for pair, pair_data in pairs_data.items():
        asset1_ticker, asset2_ticker = pair
        z_score = pair_data["z_score"].loc[test_data.index[0] : test_data.index[-1]]
        aligned_z_score, aligned_regimes = z_score.align(regime_series_fold, join="inner")

        params = PAIR_PARAMS.get(pair, {})
        entry_threshold = params.get("entry_threshold", 2.0)
        exit_threshold = params.get("exit_threshold", 0.1)
        stop_loss_k = params.get("stop_loss_k", 2.0)

        dynamic_threshold = (
            entry_threshold if not isinstance(entry_threshold, dict) else entry_threshold.get("default", 2.0)
        )
        stable_mask = aligned_regimes == STABLE_REGIME_INDEX
        unstable_mask = aligned_regimes != STABLE_REGIME_INDEX

        entry_short = (stable_mask & (aligned_z_score > dynamic_threshold)) | (
            unstable_mask & (aligned_z_score > dynamic_threshold)
        )
        entry_long = (stable_mask & (aligned_z_score < -dynamic_threshold)) | (
            unstable_mask & (aligned_z_score < -dynamic_threshold)
        )
        exit_signal = abs(aligned_z_score) < exit_threshold
        can_trade = aligned_regimes == STABLE_REGIME_INDEX
        filtered_entry_short = entry_short & can_trade
        filtered_entry_long = entry_long & can_trade
        filtered_exit = exit_signal

        signals = pd.DataFrame(
            {
                "z_score": aligned_z_score,
                "regime": aligned_regimes,
                "entry_short": filtered_entry_short,
                "entry_long": filtered_entry_long,
                "exit": filtered_exit,
                "stop_loss_k": stop_loss_k,
            },
            index=aligned_z_score.index,
        )
        trade_signals[pair] = signals

    # --- Backtest ---
    config = BacktestConfig(
        initial_capital=1_000_000,
        target_volatility=0.10,
        slippage_bps=2.0,
        commission_bps=1.0,
        stop_loss_k=2.0,
        zscore_entry_threshold=2.0,
        zscore_exit_threshold=0.1,
    )

    pair_results = []
    for pair, signals in trade_signals.items():
        ticker_y, ticker_X = pair
        prices = data.loc[test_data.index, [ticker_y, ticker_X]].dropna()

        metrics_row = metrics_df[metrics_df["pair"] == pair]
        if not metrics_row.empty:
            coint_p = float(metrics_row["coint_p"].iloc[0])
            adf_p = float(metrics_row["adf_p"].iloc[0])
            hurst = float(metrics_row["hurst"].iloc[0])
        else:
            coint_p = adf_p = hurst = None

        signals = signals.copy()
        signals["coint_p"] = coint_p
        signals["adf_p"] = adf_p
        signals["hurst"] = hurst

        backtest = PairsBacktest(config)
        backtest.prices = prices
        backtest.signals = {pair: signals}
        backtest.run_backtest(prices, {pair: signals}, regime_series_fold)
        metrics = backtest.get_performance_metrics()
        pair_results.append({"pair": pair, "metrics": metrics, "backtest": backtest})

    backtest, overall_metrics = aggregate_pair_results(pair_results, config)
    overall_metrics["fold"] = fold_num
    return backtest, overall_metrics


def main(config_path="config.yaml"):
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
    

    CROSSVAL_TRAIN_DAYS = cfg.get("CROSSVAL_TRAIN_DAYS", 365)
    CROSSVAL_TEST_DAYS = cfg.get("CROSSVAL_TEST_DAYS", 90)

    data = data.sort_index()
    fold_results = []
    start_idx = data.index.min()
    end_date = data.index.max()
    fold_num = 1

    while start_idx + pd.Timedelta(days=CROSSVAL_TRAIN_DAYS + CROSSVAL_TEST_DAYS) <= end_date:
        train_end = start_idx + pd.Timedelta(days=CROSSVAL_TRAIN_DAYS - 1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=CROSSVAL_TEST_DAYS - 1)

        train_slice = data.loc[start_idx:train_end]
        test_slice = data.loc[test_start:test_end]
        if train_slice.empty or test_slice.empty:
            break

        _, metrics = run_fold(train_slice, test_slice, cfg, fold_num)
        fold_results.append(metrics)

        start_idx = start_idx + pd.Timedelta(days=CROSSVAL_TEST_DAYS)
        fold_num += 1

    if fold_results:
        df = pd.DataFrame(fold_results)
        avg_metrics = df.drop(columns=["fold"]).mean().to_dict()
        logger.info("Average metrics across folds: %s", avg_metrics)
    else:
        logger.warning("No walk-forward folds executed.")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the mean reversion engine")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live trading mode",
    )
    args = parser.parse_args()
    main(args.config)

