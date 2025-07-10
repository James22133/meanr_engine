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
    gross, cost, net = runner._calculate_trade_pnl(1, 10.0, 12.0, shares=50.0)

    expected_gross = 50.0 * (12.0 - 10.0)
    cost_pct = (1.0 + 0.5) / 10000
    expected_cost = (10.0 + 12.0) * 50.0 * cost_pct
    expected_net = expected_gross - expected_cost

    assert abs(gross - expected_gross) < 1e-6
    assert abs(cost - expected_cost) < 1e-6
    assert abs(net - expected_net) < 1e-6
