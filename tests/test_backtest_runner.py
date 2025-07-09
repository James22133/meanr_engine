import pandas as pd
from core.backtest_runner import BacktestRunner

class DummyBacktestCfg:
    def __init__(self):
        self.initial_capital = 100000
        self.position_size_pct = 0.1
        self.slippage_bps = 1.0
        self.commission_bps = 0.5

class DummyConfig:
    def __init__(self):
        self.backtest = DummyBacktestCfg()


def test_calculate_trade_pnl_with_costs():
    runner = BacktestRunner(DummyConfig())
    prices = pd.Series({'asset1': 100.0, 'asset2': 95.0})
    pnl = runner._calculate_trade_pnl(1, 10.0, 12.0, prices, hedge_ratio=1.0)

    shares = 50.0  # 100000 * 0.1 = 10000 -> min(10000/(2*100), (10000)/(2*95))
    cost_pct = (1.0 + 0.5) / 10000
    expected_cost = (10.0 + 12.0) * shares * cost_pct
    expected = 1 * shares * (12.0 - 10.0) - expected_cost

    assert abs(pnl - expected) < 1e-6
