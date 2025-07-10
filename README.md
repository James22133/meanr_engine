# Enhanced Mean Reversion Pairs Trading Engine

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive, regime-aware pairs trading strategy with advanced performance analysis, visualization, and reporting capabilities.

## 🚀 Features

### Core Strategy
- **Regime-aware trading** with VIX, trend, and Sharpe ratio filters
- **Enhanced pair selection** using cointegration, Hurst exponent, and volatility analysis
- **Behavioral execution** with late-day entry filters
- **Adaptive thresholds** that scale with pair-specific volatility
- **Walk-forward validation** for robust parameter optimization

### Performance Analysis & Reporting
- **Comprehensive metrics** including Sharpe, Sortino, Calmar ratios, and more
- **Benchmark comparisons** against SPY and buy-and-hold portfolios
- **Interactive visualizations** with equity curves, drawdown analysis, and trade distributions
- **Detailed CSV logs** for further analysis
- **Automated report generation** with performance summaries

### Core Trading Engine
- **Advanced Pair Selection**: Multi-factor statistical analysis including cointegration, ADF tests, and Hurst exponent calculations
- **Dynamic Spread Modeling**: Kalman filtering for adaptive hedge ratios and spread estimation
- **Regime Detection**: Hidden Markov Models (HMM) for market regime identification and adaptive strategy parameters
- **Risk Management**: Volatility targeting, position sizing, and comprehensive risk controls
- **Behavioral Execution**: Trades are filtered to the late-day window to mimic retail execution (from 15:30 to market close)
- **Pair Health Logging**: Daily ADF and Hurst metrics are exported for monitoring

- **Stress Filters**: Signal generation halts when VIX spikes or market returns crash

### Backtesting & Analysis
- **VectorBT Integration**: High-performance vectorized backtesting with detailed trade analysis
- **Performance Metrics**: Comprehensive risk-adjusted returns, drawdown analysis, and statistical validation
- **Walk-Forward Analysis**: Out-of-sample validation and strategy robustness testing
- **Walk-Forward CSV Output**: Rolling statistics are saved to `walkforward_stats.csv`
- **Parameter Optimization**: Grid search and optimization for strategy parameters

### Data & Infrastructure
- **Multi-Source Data**: Support for yfinance, Alpaca, and custom data sources
- **Real-time Processing**: Efficient data handling and caching mechanisms
- **Visualization Suite**: Professional-grade charts and analysis plots
- **Modular Architecture**: Clean, maintainable codebase with comprehensive testing

## 📊 Performance Highlights

- **Signal Generation**: Advanced z-score based entry/exit signals with regime adaptation
- **Risk-Adjusted Returns**: Sophisticated performance metrics including Sharpe, Sortino, and Calmar ratios
- **Portfolio Management**: Multi-pair portfolio construction with sector diversification
- **Statistical Rigor**: Institutional-grade statistical testing and validation

## 🛠 Installation

### Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/meanr_engine.git
   cd meanr_engine
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Core Dependencies
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `statsmodels` - Statistical modeling
- `hmmlearn` - Hidden Markov Models
- `vectorbt` - Vectorized backtesting
- `yfinance` - Market data fetching
- `matplotlib` & `seaborn` - Visualization

## 🚀 Usage

### Basic Usage

Run the enhanced trading engine with default configuration:

```bash
python run_engine_enhanced.py --config config_optimized.yaml --vectorbt-only --save-plots
```

### Advanced Usage

**Parameter Optimization Mode:**
```bash
python run_engine_enhanced.py --config config_optimized.yaml --optimize --vectorbt-only
```

**Walk-Forward Analysis:**
```bash
python run_engine_enhanced.py --config config_optimized.yaml --walkforward --walkforward-windows 10
```

**Statistical Analysis Report:**
```bash
python run_engine_enhanced.py --config config_optimized.yaml --statistical-report
```

### Configuration Options

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Configuration file path | `config.yaml` |
| `--vectorbt-only` | Use only VectorBT backtesting | False |
| `--save-plots` | Generate and save analysis plots | False |
| `--optimize` | Run parameter optimization | False |
| `--walkforward` | Enable walk-forward analysis | False |
| `--statistical-report` | Generate statistical analysis report | False |

## 📈 Strategy Overview

### Mean Reversion Pairs Trading

The engine implements a sophisticated pairs trading strategy based on mean reversion principles:

1. **Pair Selection**: Statistical analysis identifies cointegrated asset pairs
2. **Signal Generation**: Z-score based entry/exit signals with adaptive thresholds
3. **Risk Management**: Position sizing, stop-losses, and portfolio-level risk controls
4. **Regime Adaptation**: Strategy parameters adapt to market conditions

### Key Components

#### Pair Selection Algorithm
- **Cointegration Testing**: Engle-Granger test for long-term relationships
- **Stationarity Analysis**: ADF test for mean reversion properties
- **Hurst Exponent**: Measures trend vs. mean reversion characteristics
- **Correlation Analysis**: Ensures sufficient short-term correlation
- **Sector Diversification**: Portfolio-level risk management

#### Signal Generation
- **Z-Score Calculation**: Rolling mean and standard deviation based signals
- **Adaptive Thresholds**: Dynamic entry/exit levels based on market conditions
- **Regime Scaling**: Signal strength adjusted for market volatility regimes
- **Multi-Timeframe**: Combines short and long-term signals

#### Risk Management
- **Volatility Targeting**: Position sizing based on portfolio volatility
- **Stop-Loss Management**: Dynamic stop-loss levels based on spread volatility
- **Sector Limits**: Maximum exposure per sector
- **Correlation Limits**: Maximum correlation between pairs

## 📁 Project Structure

```
meanr_engine/
├── core/                          # Core trading logic
│   ├── pair_selection.py         # Pair selection algorithms
│   ├── enhanced_pair_selection.py # Advanced pair selection
│   ├── backtest_runner.py        # Backtesting engine
│   ├── enhanced_metrics.py       # Performance metrics
│   ├── config.py                 # Configuration management
│   └── plotting.py               # Visualization utilities
├── backtest/                     # Backtesting modules
│   ├── backtest.py              # Traditional backtesting
│   ├── vectorbt_backtest.py     # VectorBT integration
│   └── visualization.py         # Backtest visualization
├── regime/                       # Regime detection
│   └── regime_detection.py      # HMM-based regime detection
├── data/                         # Data handling
│   ├── fetch_data.py            # Data fetching utilities
│   └── data_loader.py           # Data loading and preprocessing
├── optimization/                 # Optimization tools
│   └── grid_search.py           # Parameter optimization
├── tests/                        # Test suite
│   ├── test_pair_selector.py    # Pair selection tests
│   ├── test_metrics.py          # Metrics calculation tests
│   └── test_backtest.py         # Backtesting tests
├── plots/                        # Generated plots and analysis
├── trade_logs/                   # Detailed trade logs
├── config_optimized.yaml         # Optimized configuration
├── run_engine_enhanced.py        # Main execution script
└── requirements.txt              # Python dependencies
```

## ⚙️ Configuration

### Key Configuration Parameters

```yaml
# Signal Generation
signals:
  entry_threshold: 1.4          # Z-score entry threshold
  exit_threshold: 0.5           # Z-score exit threshold
  lookback: 20                  # Rolling window for z-score calculation

# Risk Management
backtest:
  initial_capital: 1000000      # Starting capital
  max_concurrent_positions: 5   # Maximum simultaneous positions
  stop_loss_k: 2.0             # Stop-loss multiplier

# Statistical Thresholds
statistical:
  adf_pvalue_max: 0.10         # Maximum ADF p-value
  coint_pvalue_max: 0.10       # Maximum cointegration p-value
  correlation_min: 0.6         # Minimum correlation
```

### Configuration Files

- `config.yaml` - Default configuration
- `config_optimized.yaml` - Optimized parameters for production
- `config_relaxed.yaml` - Relaxed thresholds for more signals
- `config_enhanced.yaml` - Enhanced features configuration

## 📊 Performance Metrics

The engine calculates and reports the following comprehensive metrics:

### Return Metrics
- **Total Return**: Cumulative strategy performance
- **CAGR**: Compound Annual Growth Rate
- **Annual Volatility**: Standard deviation of returns (annualized)

### Risk-Adjusted Returns
- **Sharpe Ratio**: Excess return per unit of risk
- **Sortino Ratio**: Excess return per unit of downside risk
- **Calmar Ratio**: CAGR divided by maximum drawdown
- **Information Ratio**: Excess return relative to benchmark

### Risk Metrics
- **Maximum Drawdown**: Largest peak-to-trough decline
- **VaR (95%)**: Value at Risk at 95% confidence
- **CVaR (95%)**: Conditional Value at Risk
- **Skewness & Kurtosis**: Distribution shape characteristics

### Trade Statistics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Total profit divided by total loss
- **Average Win/Loss**: Mean profit and loss per trade
- **Average Trade Duration**: Mean holding period in days

### Rolling Metrics
- **30-Day Rolling Sharpe**: Rolling risk-adjusted returns
- **30-Day Rolling Drawdown**: Rolling maximum drawdown
- **Best/Worst 30-Day Returns**: Period performance extremes

## 📈 Output Files

After each backtest, the engine generates comprehensive outputs:

### Performance Summary
- `performance_summary.json` - Complete metrics in JSON format
- `comprehensive_report.txt` - Human-readable performance summary

### CSV Logs
- `daily_equity.csv` - Daily NAV, returns, drawdown, and rolling metrics
- `trade_log.csv` - Individual trade details with entry/exit, PnL, and regime info
- `walkforward_stats.csv` - Walk-forward validation statistics
- `pair_health_log.csv` - Pair health metrics (ADF, Hurst, volatility)

### Visualizations
- `equity_curve_*.png` - Strategy vs benchmark comparison
- `drawdown_analysis_*.png` - Drawdown comparison chart
- `trade_pnl_histogram_*.png` - Trade PnL distribution
- `holding_period_histogram_*.png` - Trade duration distribution
- `rolling_metrics_*.png` - Rolling Sharpe and drawdown
- `signal_chart_*.png` - Entry/exit points on price series
- `performance_summary_*.png` - Key metrics summary chart

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_pair_selector.py
pytest tests/test_metrics.py

# Run with coverage
pytest --cov=core --cov=backtest --cov=regime
```

### Test Coverage

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Backtesting accuracy validation
- **Statistical Tests**: Strategy robustness validation

## 📈 Live Trading

### Alpaca Integration

For live trading with Alpaca's paper trading:

```bash
python alpaca_backtrader_sim.py \
    --api_key YOUR_API_KEY \
    --secret_key YOUR_SECRET_KEY \
    --pairs XOM/CVX,SPY/QQQ \
    --mode paper
```

## 🤝 Contributing

Any Reccommendations for improvement or contributions would be sound.
## Guidlines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
black .
flake8 .

# Run tests
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For issues and questions:
1. Check the test results: `python run_edge_case_tests.py`
2. Review the performance logs in the output directory
3. Examine the comprehensive report for detailed analysis
4. Check the README for configuration guidance

## 🎯 Success Criteria

After running a backtest, you should be able to:
1. ✅ View all major performance metrics
2. ✅ Compare strategy vs. buy & hold performance
3. ✅ Visualize equity curves and risk charts
4. ✅ Export CSVs for further analysis
5. ✅ Generate comprehensive performance reports

## Key Updates (July 2025)

- **Position Sizing Fix:**
  - Backtest engine now uses configurable position sizing (`position_size_pct` in config).
  - Actual dollar PnL is calculated and stored for each trade, not just spread units.
  - Diagnostics print per-trade share size, spread change, and expected PnL.
- **Accurate PnL Reporting:**
  - Analysis script (`analyze_optimized_results.py`) now reports true, scaled PnL per trade and total, reflecting your capital allocation.
- **How to Use:**
  1. Edit `config_optimized_simple.yaml` to set `position_size_pct` (e.g., `0.20` for 20% of capital per trade).
  2. Run the optimized backtest:
     ```bash
     python run_optimized_backtest.py
     ```
  3. Analyze results:
     ```bash
     python analyze_optimized_results.py
     ```
  4. Check the output for true per-trade and total PnL, win rate, and Sharpe ratio.

## Example Output

```
SMH-SPY
  Total Trades: 104
  Total Actual PnL: $6,334.62
  Average Actual PnL per Trade: $60.91
EFA-QQQ
  Total Trades: 96
  Total Actual PnL: -$805.89
  Average Actual PnL per Trade: -$8.39
Overall
  Total Actual PnL: $5,528.72
  Average Actual PnL per Trade: $27.64
```

---

For more details, see the comments in `core/backtest_runner.py` and `analyze_optimized_results.py`.


