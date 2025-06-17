import core.metrics as metrics

class DummyConfig:
    pass


def test_calculate_trade_metrics_empty_trades():
    calculator = metrics.MetricsCalculator(DummyConfig())
    result = calculator._calculate_trade_metrics([])
    assert result == {
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'win_rate': 0.0,
        'avg_return': 0.0,
        'avg_win': 0.0,
        'avg_loss': 0.0,
        'profit_factor': 0.0,
    }
