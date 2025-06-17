"""
Core module for the mean reversion trading engine.
"""

from .config import Config, PairSelectionConfig, BacktestConfig
from .data_loader import DataLoader
from .pair_selection import PairSelector
from .backtest_runner import BacktestRunner
from .metrics import MetricsCalculator
from .plotting import PlotGenerator

__all__ = [
    'Config',
    'PairSelectionConfig',
    'BacktestConfig',
    'DataLoader',
    'PairSelector',
    'BacktestRunner',
    'MetricsCalculator',
    'PlotGenerator',
] 