"""
Configuration module for the mean reversion trading engine.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml
import logging

@dataclass
class PairSelectionConfig:
    """Configuration for pair selection parameters."""
    min_correlation: float = 0.5
    min_spread_stability: float = 0.4
    max_zscore_volatility: float = 1.8
    min_cointegration_stability: float = 0.3
    correlation_window: int = 60
    spread_stability_window: int = 60
    stability_lookback: int = 20
    min_data_points: int = 100

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    initial_capital: float = 100000.0
    max_position_size: float = 0.2
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.03
    max_drawdown: float = 0.15
    allowed_entry_regimes: List[str] = None
    regime_lookback: int = 20
    regime_threshold: float = 0.6

    def __post_init__(self):
        if self.allowed_entry_regimes is None:
            self.allowed_entry_regimes = ['mean_reverting', 'trending']

@dataclass
class Config:
    """Main configuration class for the trading engine."""
    etf_tickers: List[str]
    start_date: str
    end_date: str
    pair_selection: PairSelectionConfig
    backtest: BacktestConfig
    pair_universes: Dict[str, Dict]
    data_dir: str = "data"
    reports_dir: str = "reports"
    plots_dir: str = "plots"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from a YAML file."""
        try:
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Extract pair selection config
            pair_selection = PairSelectionConfig(**config_dict.get('PAIR_SELECTION', {}))
            
            # Extract backtest config from lowercase key to match config.yaml
            backtest_cfg = config_dict.get('backtest', {}) or {}
            backtest = BacktestConfig(**backtest_cfg)
            
            # Create main config
            return cls(
                etf_tickers=config_dict.get('ETF_TICKERS', []),
                start_date=config_dict.get('START_DATE', ''),
                end_date=config_dict.get('END_DATE', ''),
                pair_selection=pair_selection,
                backtest=backtest,
                pair_universes=config_dict.get('PAIR_UNIVERSES', {})
            )
        except Exception as e:
            logging.error(f"Error loading config from {yaml_path}: {str(e)}")
            raise 