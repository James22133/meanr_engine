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
class PairScoringConfig:
    """Configuration for pair scoring weights."""
    weights: Dict[str, float] = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                'correlation': 0.25,
                'coint_p': 0.25,
                'hurst': 0.25,
                'zscore_vol': 0.25,
            }

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    initial_capital: float = 1_000_000
    target_volatility: float = 0.10  # 10% annualized
    slippage_bps: float = 15.0  # 0.15% per side (~0.3% round trip)
    commission_bps: float = 1.0
    stop_loss_k: float = 2.0  # Multiplier for volatility-based stop-loss
    zscore_entry_threshold: float = 2.0
    zscore_exit_threshold: float = 0.1
    max_hold_days: Optional[int] = None
    target_profit_pct: Optional[float] = None
    rebalance_freq: int = 21  # Default to weekly rebalancing
    max_concurrent_positions: int = 5  # Default to 5 positions
    risk_control: Optional[dict] = None
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.05  # 5% take profit
    position_size_pct: float = 0.05  # 5% of capital per trade (default)


@dataclass
class WalkforwardConfig:
    """Configuration for walk-forward validation parameters."""
    train_months: int = 24
    test_months: int = 6
    z_entry_values: List[float] = None
    z_exit_values: List[float] = None
    scoring_metric: str = 'sharpe_ratio'

    def __post_init__(self):
        if self.z_entry_values is None:
            self.z_entry_values = [1.0, 1.5, 2.0]
        if self.z_exit_values is None:
            self.z_exit_values = [0.5, 0.0]

@dataclass
class Config:
    """Main configuration class for the trading engine."""
    etf_tickers: List[str]
    start_date: str
    end_date: str
    pair_selection: PairSelectionConfig
    backtest: BacktestConfig
    walkforward: WalkforwardConfig
    pair_universes: Dict[str, Dict]
    pair_scoring: PairScoringConfig
    use_cache: bool = False
    force_refresh: bool = False
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
            backtest_cfg['risk_control'] = config_dict.get('risk_control', {})
            backtest = BacktestConfig(**backtest_cfg)

            # Pair scoring weights
            pair_scoring_cfg = config_dict.get('pair_scoring', {})
            pair_scoring = PairScoringConfig(**pair_scoring_cfg)

            # Walk-forward validation config
            walkforward_cfg = config_dict.get('walkforward', {}) or {}
            walkforward = WalkforwardConfig(**walkforward_cfg)
            
            # Create main config
            return cls(
                etf_tickers=config_dict.get('ETF_TICKERS', []),
                start_date=config_dict.get('START_DATE', ''),
                end_date=config_dict.get('END_DATE', ''),
                pair_selection=pair_selection,
                backtest=backtest,
                walkforward=walkforward,
                pair_universes=config_dict.get('PAIR_UNIVERSES', {}),
                pair_scoring=pair_scoring,
                use_cache=config_dict.get('use_cache', False),
                force_refresh=config_dict.get('force_refresh', False),
            )
        except Exception as e:
            logging.error(f"Error loading config from {yaml_path}: {str(e)}")
            raise 
