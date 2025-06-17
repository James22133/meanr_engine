"""
Core module for the mean reversion trading engine.
"""

from .config import Config, PairSelectionConfig, BacktestConfig, WalkforwardConfig
from .data_loader import DataLoader
from .pair_selection import PairSelector
from .backtest_runner import BacktestRunner
from .metrics import MetricsCalculator
from .plotting import PlotGenerator
from .walkforward import WalkForwardValidator
from .risk import compute_sector_exposure, max_drawdown
from .cache import save_to_cache, load_from_cache

__all__ = [
    'Config',
    'PairSelectionConfig',
    'BacktestConfig',
    'WalkforwardConfig',
    'DataLoader',
    'PairSelector',
    'BacktestRunner',
    'WalkForwardValidator',
    'MetricsCalculator',
    'PlotGenerator',
    'compute_sector_exposure',
    'max_drawdown',
    'save_to_cache',
    'load_from_cache',
]
