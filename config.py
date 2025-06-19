import yaml
from pathlib import Path
from backtest import BacktestConfig


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from a YAML file and parse nested sections."""

    config_path = Path(path)
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)

    backtest_cfg = cfg.get("backtest", {}) or {}
    backtest_cfg["risk_control"] = cfg.get("risk_control", {})
    cfg["backtest"] = BacktestConfig(**backtest_cfg)
    cfg["diagnostics"] = cfg.get("diagnostics", {})

    return cfg
