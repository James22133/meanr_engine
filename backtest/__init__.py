from .backtest import PairsBacktest, BacktestConfig, Trade
from .visualization import (
    plot_equity_curve,
    plot_monthly_returns,
    plot_trade_distribution,
    plot_regime_performance,
    plot_performance_metrics
)

__all__ = [
    'PairsBacktest',
    'BacktestConfig',
    'Trade',
    'plot_equity_curve',
    'plot_monthly_returns',
    'plot_trade_distribution',
    'plot_regime_performance',
    'plot_performance_metrics'
] 
