# Mean Reversion Pairs Trading Engine

A quantitative trading engine that implements a pairs trading strategy using mean reversion principles, Kalman filtering, and regime detection.

## Features

- ETF data fetching and preprocessing
- Cointegration analysis for pair selection
- Kalman filter for dynamic spread modeling
- Composite scoring of pairs using cointegration, ADF, Hurst and Z-score volatility
- Hidden Markov Model for regime detection
- Backtesting engine with:
  - Position sizing based on volatility targeting
  - Stop-loss and take-profit management
  - Transaction cost modeling
  - Performance metrics calculation
- Visualization tools for:
  - Equity curves and drawdowns
  - Monthly returns heatmaps
  - Trade distributions
  - Regime-specific performance
  - Performance metrics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/meanr_engine.git
cd meanr_engine
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

The list includes core packages such as `hmmlearn` and `pykalman` which are
required for regime detection and Kalman filtering.

## Usage

Run the main engine:
```bash
python run_engine.py --config config.yaml
```

Optional flags include:

- `--config PATH` - path to the configuration file (defaults to `config.yaml`)
- `--live` - enable live trading mode (placeholder)

Running the engine will:
1. Fetch ETF data
2. Detect market regimes
3. Generate trading signals
4. Run the backtest
5. Generate performance reports and visualizations

## Project Structure

```
meanr_engine/
├── backtest/
│   ├── __init__.py
│   ├── backtest.py
│   └── visualization.py
├── data/
│   └── fetch_data.py
├── pairs/
│   └── pair_analysis.py
├── regime/
│   └── regime_detection.py
├── run_engine.py
├── requirements.txt
└── README.md
```

## Running Tests

Before running the tests, ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes `pytest>=8.0`, which provides the test
runner used by the `scripts/run_tests.sh` helper.

Running `pytest` or the helper script without these packages (e.g. `hmmlearn`,
`pykalman`) will result in import errors.

Then execute the bundled test runner from the repository root:

```bash
./scripts/run_tests.sh
```

This script adds the project root to `PYTHONPATH` before invoking `pytest`.

## Configuration

All parameters are stored in `config.yaml`, which is read at runtime by
`config.load_config()`. Edit this file to tweak tickers, thresholds and other
settings.

Example snippet:

```yaml
ETF_TICKERS:
  - XOM
  - CVX
PAIR_PARAMS:
  XOM_CVX:
    entry_threshold: 1.8
    exit_threshold: 0.6
```

## Pair Scoring

Pairs are ranked using four metrics:

- Rolling cointegration p-value
- Rolling Hurst exponent
- Rolling ADF test p-value
- Z-score volatility

Each metric is min-max normalized across all candidate pairs. The final score is
`1 - mean(normalized metrics)` so that lower p-values or lower Hurst values
(indicating stronger mean reversion) lead to higher scores. The `TOP_N_PAIRS`
parameter defines how many of the highest scoring pairs are kept for signal
generation and backtesting. Increase this value in `config.yaml` to consider
more pairs or reduce it to focus on fewer.

These relaxed statistical metrics are now used for scoring rather than strict
threshold-based filtering, allowing borderline pairs to be considered while
still prioritizing the strongest candidates.

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
