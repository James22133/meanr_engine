# Mean Reversion Pairs Trading Engine

A quantitative trading engine that implements a pairs trading strategy using mean reversion principles, Kalman filtering, and regime detection.

## Features

- ETF data fetching and preprocessing
- Cointegration analysis for pair selection
- Kalman filter for dynamic spread modeling
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

## Running Tests

Install the project dependencies if you have not already:

```bash
pip install -r requirements.txt
```

Then execute the bundled test runner from the repository root:

```bash
./scripts/run_tests.sh
```

This script adds the project root to `PYTHONPATH` before invoking `pytest`.

## Configuration

Key parameters can be adjusted in `run_engine.py`:
- ETF tickers
- Date range
- Cointegration window
- Z-score thresholds
- Regime detection parameters
- Backtest configuration

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
