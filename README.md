Mean Reversion Pairs Trading Engine
A quantitative trading engine that implements a pairs trading strategy using mean reversion principles, Kalman filtering, and regime detection.

Features
ETF data fetching and preprocessing

Cointegration analysis for pair selection

Kalman filter for dynamic spread modeling

Composite scoring of pairs using cointegration, ADF, Hurst and Z-score volatility

Hidden Markov Model for regime detection

Backtesting engine with:

Position sizing based on volatility targeting

Stop-loss and take-profit management

Transaction cost modeling

Performance metrics calculation

Visualization tools for:

Equity curves and drawdowns

Monthly returns heatmaps

Trade distributions

Regime-specific performance

Performance metrics

Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/meanr_engine.git
cd meanr_engine
Create and activate a virtual environment:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
The list includes core packages such as hmmlearn and pykalman which are required for regime detection and Kalman filtering.

Usage
Run the main engine:

bash
Copy
Edit
python run_engine.py --config config.yaml
Optional flags include:

--config PATH - path to the configuration file (defaults to config.yaml)

Parameter Grid Search
To evaluate different threshold combinations, run the grid search mode:

bash
Copy
Edit
python run_engine.py --mode grid-search \
    --entry-thresholds 1.5,2.0 \
    --exit-thresholds 0.1 \
    --stop-loss-ks 2.0
Ranges may also be specified in the grid_search section of config.yaml. Results are printed and saved to grid_search_results.csv.

Running the engine will:

Fetch ETF data

Detect market regimes

Generate trading signals

Run the backtest

Generate performance reports and visualizations

Project Structure
kotlin
Copy
Edit
meanr_engine/
├── backtest/
│   ├── __init__.py
│   ├── backtest.py
│   └── visualization.py
├── data/
│   └── fetch_data.py
├── core/
│   └── pair_analysis.py
├── regime/
│   └── regime_detection.py
├── run_engine.py
├── requirements.txt
└── README.md
Live Simulation with Alpaca
alpaca_backtrader_sim.py runs the pairs strategy using Backtrader and Alpaca's paper trading API. Pass your Alpaca credentials and choose between backtest or live modes. Multiple pairs can be traded simultaneously by providing them via the --pairs option:

bash
Copy
Edit
python alpaca_backtrader_sim.py --api_key YOUR_KEY --secret_key YOUR_SECRET \
    --pairs VLO/XLE,COP/CVX,EFA/QQQ --mode backtest
Backtest mode fetches historical data from Alpaca while live mode subscribes to daily bars for simulated execution.

Running Tests
Before running the tests, ensure all dependencies are installed:

bash
Copy
Edit
pip install -r requirements.txt
The requirements.txt file includes pytest>=8.0, which provides the test runner used by the scripts/run_tests.sh helper.

Running pytest or the helper script without these packages (e.g. hmmlearn, pykalman) will result in import errors.

Then execute the bundled test runner from the repository root:

bash
Copy
Edit
./scripts/run_tests.sh
This script adds the project root to PYTHONPATH before invoking pytest.

Configuration
All parameters are stored in config.yaml, which is read at runtime by config.load_config(). Edit this file to tweak tickers, thresholds and other settings.

CROSSVAL_TRAIN_DAYS and CROSSVAL_TEST_DAYS control the size of the rolling training and testing windows used in run_engine.py. The engine trains models on the most recent CROSSVAL_TRAIN_DAYS of data, evaluates on the following CROSSVAL_TEST_DAYS, then advances the window by the test period length to create the next fold.

Example snippet:

yaml
Copy
Edit
ETF_TICKERS:
  - XOM
  - CVX
CROSSVAL_TRAIN_DAYS: 252  # one year of training
CROSSVAL_TEST_DAYS: 63    # three months of testing
PAIR_PARAMS:
  XOM_CVX:
    entry_threshold: 1.8
    exit_threshold: 0.6
Pair Scoring
Pairs are ranked using four metrics:

Rolling cointegration p-value

Rolling Hurst exponent

Rolling ADF test p-value

Z-score volatility

Each metric is min-max normalized across all candidate pairs. The final score is 1 - mean(normalized metrics) so that lower p-values or lower Hurst values (indicating stronger mean reversion) lead to higher scores. The TOP_N_PAIRS parameter defines how many of the highest scoring pairs are kept for signal generation and backtesting. Increase this value in config.yaml to consider more pairs or reduce it to focus on fewer.

These relaxed statistical metrics are now used for scoring rather than strict threshold-based filtering, allowing borderline pairs to be considered while still prioritizing the strongest candidates.

License
This project is licensed under the MIT License.

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.