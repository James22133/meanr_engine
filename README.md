# Mean Reversion Pairs Trading Engine

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A sophisticated quantitative trading engine implementing advanced pairs trading strategies using mean reversion principles, statistical arbitrage, and machine learning techniques.

## 🚀 Features

### Core Trading Engine
- **Advanced Pair Selection**: Multi-factor statistical analysis including cointegration, ADF tests, and Hurst exponent calculations
- **Dynamic Spread Modeling**: Kalman filtering for adaptive hedge ratios and spread estimation
- **Regime Detection**: Hidden Markov Models (HMM) for market regime identification and adaptive strategy parameters
- **Risk Management**: Volatility targeting, position sizing, and comprehensive risk controls

### Backtesting & Analysis
- **VectorBT Integration**: High-performance vectorized backtesting with detailed trade analysis
- **Performance Metrics**: Comprehensive risk-adjusted returns, drawdown analysis, and statistical validation
- **Walk-Forward Analysis**: Out-of-sample validation and strategy robustness testing
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

The engine calculates comprehensive performance metrics:

### Return Metrics
- **Total Return**: Cumulative strategy performance
- **Annualized Return**: Year-over-year performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Calmar Ratio**: Maximum drawdown-adjusted returns

### Risk Metrics
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Value at Risk (VaR)**: 95% and 99% confidence levels
- **Conditional VaR**: Expected loss beyond VaR
- **Volatility**: Annualized standard deviation
- **Downside Risk**: Negative return volatility

### Trade Statistics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profits to gross losses
- **Average Win/Loss**: Mean profit and loss per trade
- **Maximum Consecutive Wins/Losses**: Streak analysis
- **Recovery Time**: Time to recover from drawdowns

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

### Risk Warnings

⚠️ **Important**: This software is for educational and research purposes. Live trading involves substantial risk of loss. Always:
- Test thoroughly in paper trading mode
- Start with small position sizes
- Monitor performance closely
- Understand all risks involved

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

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

## 📞 Support

- **Issues**: Report bugs and request features via [GitHub Issues](https://github.com/yourusername/meanr_engine/issues)
- **Discussions**: Join community discussions in [GitHub Discussions](https://github.com/yourusername/meanr_engine/discussions)
- **Documentation**: Check the [Wiki](https://github.com/yourusername/meanr_engine/wiki) for detailed guides

## 🙏 Acknowledgments

- **VectorBT**: High-performance backtesting framework
- **Statsmodels**: Statistical modeling and testing
- **Pandas**: Data manipulation and analysis
- **YFinance**: Market data access
- **Alpaca**: Trading API and infrastructure

---

**Disclaimer**: This software is for educational purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results.
