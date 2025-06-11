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

## Usage

Run the main engine:
```bash
python run_engine.py
```

This will:
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

## Configuration

Key parameters can be adjusted in `run_engine.py`:
- ETF tickers
- Date range
- Cointegration window
- Z-score thresholds
- TOP_N_PAIRS (number of highest scoring pairs to trade)
- Regime detection parameters
- Backtest configuration

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
generation and backtesting. Increase this value in `run_engine.py` to consider
more pairs or reduce it to focus on fewer.

These relaxed statistical metrics are now used for scoring rather than strict
threshold-based filtering, allowing borderline pairs to be considered while
still prioritizing the strongest candidates.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
