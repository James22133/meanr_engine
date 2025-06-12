import argparse
import logging
import pandas as pd

from config import load_config
from data.fetch_data import fetch_etf_data, DataFetchError
from optimization.grid_search import grid_search, generate_signals

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run parameter grid search")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--entry-range", nargs="+", type=float, default=[1.5, 2.0, 2.5],
        help="Entry threshold values"
    )
    parser.add_argument(
        "--exit-range", nargs="+", type=float, default=[0.1, 0.2],
        help="Exit threshold values"
    )
    parser.add_argument(
        "--stop-range", nargs="+", type=float, default=[2.0],
        help="Stop loss multipliers"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    pair_list = cfg.get("MANUAL_PAIR_LIST")
    if pair_list:
        asset1, asset2 = pair_list[0]
    else:
        asset1, asset2 = cfg["ETF_TICKERS"][:2]

    logger.info("Fetching data for %s and %s", asset1, asset2)
    try:
        prices = fetch_etf_data([asset1, asset2], cfg["START_DATE"], cfg["END_DATE"]).dropna()
    except DataFetchError as e:
        logger.error("Data fetch failed: %s", e)
        return

    regimes = pd.Series(index=prices.index, data=0)

    results = grid_search(
        prices,
        generate_signals,
        regimes,
        entry_thresholds=args.entry_range,
        exit_thresholds=args.exit_range,
        stop_loss_ks=args.stop_range,
        base_config=cfg["backtest"],
    )

    if results.empty:
        logger.warning("Grid search produced no results")
    else:
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
