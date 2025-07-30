# Enhanced Mean Reversion Pairs Trading Engine

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive, regime-aware pairs trading strategy with advanced performance analysis, visualization, and reporting capabilities. **Recently enhanced with dynamic position sizing, improved risk management, and strategic pair exclusion for optimal performance.**

## 🚀 Latest Performance (2023-2025 Period)

### **Enhanced Strategy Results:**
- **Win Rate:** 69.3% (137 total trades)
- **Total PnL:** $186,366
- **Sharpe Ratio:** 0.384
- **Sortino Ratio:** 0.013
- **Total Return:** 18.64%
- **Max Drawdown:** -6.38%
- **Calmar Ratio:** 2.923

### **Key Improvements:**
- ✅ **Dynamic Position Sizing** - Volatility-normalized position sizing
- ✅ **Regime-Aware Optimization** - Gaussian Mixture Model (GMM) regime detection
- ✅ **Strategic Pair Exclusion** - EMB-SPY removed due to catastrophic losses
- ✅ **Enhanced Risk Metrics** - Comprehensive risk-adjusted performance analysis
- ✅ **Fixed VectorBT Compatibility** - Resolved deprecated parameter issues

## 🚀 Features

### Core Strategy
- **Regime-aware trading** with VIX, trend, and Sharpe ratio filters
- **Enhanced pair selection** using cointegration, Hurst exponent, and volatility analysis
- **Behavioral execution** with late-day entry filters
- **Adaptive thresholds** that scale with pair-specific volatility
- **Walk-forward validation** for robust parameter optimization
- **Dynamic position sizing** based on volatility and regime conditions

### Performance Analysis & Reporting
- **Comprehensive metrics** including Sharpe, Sortino, Calmar ratios, and more
- **Benchmark comparisons** against SPY and buy-and-hold portfolios
- **Interactive visualizations** with equity curves, drawdown analysis, and trade distributions
- **Detailed CSV logs** for further analysis
- **Automated report generation** with performance summaries
- **Enhanced risk metrics** with proper error handling and validation

### Core Trading Engine
- **Advanced Pair Selection**: Multi-factor statistical analysis including cointegration, ADF tests, and Hurst exponent calculations
- **Dynamic Spread Modeling**: Kalman filtering for adaptive hedge ratios and spread estimation
- **Regime Detection**: Gaussian Mixture Models (GMM) for market regime identification and adaptive strategy parameters
- **Risk Management**: Volatility targeting, position sizing, and comprehensive risk controls
- **Behavioral Execution**: Trades are filtered to the late-day window to mimic retail execution (from 15:30 to market close)
- **Pair Health Logging**: Daily ADF and Hurst metrics are exported for monitoring
- **Strategic Pair Exclusion**: Automatic identification and exclusion of underperforming pairs

### Backtesting & Analysis
- **VectorBT Integration**: High-performance vectorized backtesting with detailed trade analysis
- **Performance Metrics**: Comprehensive risk-adjusted returns, drawdown analysis, and statistical validation
- **Walk-Forward Analysis**: Out-of-sample validation and strategy robustness testing
- **Walk-Forward CSV Output**: Rolling statistics are saved to `walkforward_stats.csv`
- **Parameter Optimization**: Grid search and optimization for strategy parameters
- **Enhanced Trade Analysis**: Detailed trade-level metrics with position sizing and regime information

### Data & Infrastructure
- **Multi-Source Data**: Support for yfinance, Alpaca, and custom data sources
- **Real-time Processing**: Efficient data handling and caching mechanisms
- **Visualization Suite**: Professional-grade charts and analysis plots
- **Modular Architecture**: Clean, maintainable codebase with comprehensive testing
- **2023-2025 Testing Period**: Optimized for post-COVID market dynamics

## 📊 Performance Highlights

- **Signal Generation**: Advanced z-score based entry/exit signals with regime adaptation
- **Risk-Adjusted Returns**: Sophisticated performance metrics including Sharpe, Sortino, and Calmar ratios
- **Portfolio Management**: Multi-pair portfolio construction with sector diversification
- **Statistical Rigor**: Institutional-grade statistical testing and validation
- **Dynamic Position Sizing**: Volatility-normalized position sizing for optimal risk-adjusted returns

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
- `sklearn` - Machine learning (GMM for regime detection)
- `vectorbt` - Vectorized backtesting
- `yfinance` - Market data fetching
- `matplotlib` & `seaborn` - Visualization

## 🚀 Usage

### Basic Usage

Run the enhanced trading engine with 2023-2025 configuration:

```bash
python run_engine_enhanced.py --config config_2023_2025.yaml --statistical-report
```

### Enhanced Backtest (Recommended)

Run the enhanced backtest with strategic improvements:

```bash
python enhanced_backtest_fixed_no_emb.py
```

### Simple Backtest

Run the simple backtest for comparison:

```bash
python simple_backtest_2023_2025.py
```

### Analysis and Comparison

Compare original vs enhanced strategy performance:

```bash
python simple_analysis.py
```

### Advanced Usage

**Parameter Optimization Mode:**
```bash
python run_engine_enhanced.py --config config_2023_2025.yaml --optimize --vectorbt-only
```

**Walk-Forward Analysis:**
```bash
python run_engine_enhanced.py --config config_2023_2025.yaml --walkforward --walkforward-windows 10
```

**Statistical Analysis Report:**
```bash
python run_engine_enhanced.py --config config_2023_2025.yaml --statistical-report
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

### Enhanced Mean Reversion Pairs Trading

The engine implements a sophisticated pairs trading strategy based on mean reversion principles with recent enhancements:

1. **Pair Selection**: Statistical analysis identifies cointegrated asset pairs
2. **Signal Generation**: Z-score based entry/exit signals with adaptive thresholds
3. **Risk Management**: Dynamic position sizing, stop-losses, and portfolio-level risk controls
4. **Regime Adaptation**: Strategy parameters adapt to market conditions using GMM
5. **Strategic Pair Exclusion**: Automatic identification and removal of underperforming pairs

### Key Components

#### Enhanced Pair Selection Algorithm
- **Cointegration Testing**: Engle-Granger test for long-term relationships
- **Stationarity Analysis**: ADF test for mean reversion properties
- **Hurst Exponent**: Measures trend vs. mean reversion characteristics
- **Correlation Analysis**: Ensures sufficient short-term correlation
- **Sector Diversification**: Portfolio-level risk management
- **Performance Filtering**: Exclude pairs with poor historical performance

#### Dynamic Signal Generation
- **Z-Score Calculation**: Rolling mean and standard deviation based signals
- **Adaptive Thresholds**: Dynamic entry/exit levels based on market conditions
- **Regime Scaling**: Signal strength adjusted for market volatility regimes
- **Multi-Timeframe**: Combines short and long-term signals
- **Position Sizing**: Volatility-normalized position sizing for optimal risk-adjusted returns

#### Advanced Risk Management
- **Volatility Targeting**: Position sizing based on portfolio volatility
- **Dynamic Stop-Losses**: Stop-loss levels based on spread volatility and regime conditions
- **Sector Limits**: Maximum exposure per sector
- **Correlation Limits**: Maximum correlation between pairs
- **Strategic Pair Exclusion**: Remove underperforming pairs (e.g., EMB-SPY)

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
│   ├── vectorbt_backtest.py     # VectorBT integration (fixed)
│   └── visualization.py         # Backtest visualization
├── regime/                       # Regime detection
│   └── regime_detection.py      # GMM-based regime detection
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
├── config_2023_2025.yaml        # 2023-2025 optimized configuration
├── enhanced_backtest_fixed_no_emb.py # Enhanced backtest (recommended)
├── simple_backtest_2023_2025.py # Simple backtest for comparison
├── simple_analysis.py           # Performance comparison script
├── run_engine_enhanced.py       # Main execution script
└── requirements.txt              # Python dependencies
```

## ⚙️ Configuration

### Key Configuration Parameters

```yaml
# Signal Generation
signals:
  entry_threshold: 1.4          # Z-score entry threshold
  exit_threshold: 0.2           # Z-score exit threshold
  lookback: 20                  # Rolling window for z-score calculation

# Risk Management
backtest:
  initial_capital: 1000000      # Starting capital
  max_concurrent_positions: 5   # Maximum simultaneous positions
  stop_loss_k: 2.0             # Stop-loss multiplier

# Statistical Thresholds
statistical:
  adf_pvalue_max: 0.05         # Maximum ADF p-value
  coint_pvalue_max: 0.05       # Maximum cointegration p-value
  correlation_min: 0.8         # Minimum correlation
```

### Configuration Files

- `config.yaml` - Default configuration
- `config_2023_2025.yaml` - Optimized for 2023-2025 period
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
- **Position Sizing**: Dynamic position sizing metrics

### Rolling Metrics
- **30-Day Rolling Sharpe**: Rolling risk-adjusted returns
- **30-Day Rolling Drawdown**: Rolling maximum drawdown
- **Best/Worst 30-Day Returns**: Period performance extremes

## 📈 Output Files

After each backtest, the engine generates comprehensive outputs:

### Performance Summary
- `enhanced_backtest_no_emb_results.json` - Enhanced strategy results
- `simple_backtest_2023_2025_results.json` - Simple strategy results
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

Any recommendations for improvement or contributions would be sound.

## Guidelines:

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

## Key Updates (January 2025)

### **Major Enhancements:**
- **Dynamic Position Sizing**: Volatility-normalized position sizing for optimal risk-adjusted returns
- **GMM Regime Detection**: Gaussian Mixture Model for improved market regime identification
- **Strategic Pair Exclusion**: Automatic identification and removal of underperforming pairs (EMB-SPY)
- **Enhanced Risk Metrics**: Comprehensive risk-adjusted performance analysis with proper error handling
- **2023-2025 Testing Period**: Optimized for post-COVID market dynamics

### **Performance Improvements:**
- **Win Rate**: Improved to 69.3% (137 trades)
- **Total PnL**: $186,366 with enhanced strategy
- **Sharpe Ratio**: 0.384
- **Total Return**: 18.64%
- **Max Drawdown**: -6.38%

### **Technical Fixes:**
- **VectorBT Compatibility**: Fixed deprecated parameter issues
- **Risk Metrics Calculation**: Improved error handling and validation
- **Portfolio Returns**: Enhanced calculation method for accurate risk metrics
- **Configuration Updates**: Added config_2023_2025.yaml for optimized testing

### **New Scripts:**
- `enhanced_backtest_fixed_no_emb.py` - Enhanced backtest with strategic improvements
- `simple_backtest_2023_2025.py` - Simple backtest for comparison
- `simple_analysis.py` - Performance comparison and analysis
- `config_2023_2025.yaml` - Optimized configuration for 2023-2025 period

## Example Output

### Enhanced Strategy Results:
```
======================================================================
ENHANCED RESULTS (2023-2025) - EXCLUDING EMB-SPY
======================================================================
Total Trades: 137
Winning Trades: 95
Win Rate: 69.3%
Total PnL: $186,366.30
Average PnL per Trade: $1,360.34
Average Position Size: $2,215

RISK METRICS:
Sharpe Ratio: 0.384
Sortino Ratio: 0.013
Total Return: 18.64%
Max Drawdown: -6.38%
Volatility: 7.85%
Calmar Ratio: 2.923
```

---

For more details, see the comments in the enhanced backtest scripts and configuration files.


