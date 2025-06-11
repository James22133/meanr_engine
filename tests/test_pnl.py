import datetime
from backtest.backtest import PairsBacktest, BacktestConfig, Trade


def make_backtester():
    config = BacktestConfig(slippage_bps=0.0, commission_bps=0.0)
    return PairsBacktest(config)


def make_trade(**kwargs):
    defaults = dict(
        entry_date=datetime.datetime(2023, 1, 1),
        exit_date=datetime.datetime(2023, 1, 2),
        asset1="A",
        asset2="B",
        direction="long",
        entry_price1=0,
        entry_price2=0,
        exit_price1=0,
        exit_price2=0,
        size=1.0,
        pnl=None,
        exit_reason="signal",
    )
    defaults.update(kwargs)
    return Trade(**defaults)


def test_calculate_trade_pnl_long():
    bt = make_backtester()
    trade = make_trade(
        direction="long",
        entry_price1=100.0,
        entry_price2=95.0,
        exit_price1=105.0,
        exit_price2=90.0,
    )
    pnl = bt.calculate_trade_pnl(trade)
    expected = (105.0 - 100.0) - (90.0 - 95.0)
    assert pnl == expected


def test_calculate_trade_pnl_short():
    bt = make_backtester()
    trade = make_trade(
        direction="short",
        entry_price1=100.0,
        entry_price2=95.0,
        exit_price1=90.0,
        exit_price2=100.0,
    )
    pnl = bt.calculate_trade_pnl(trade)
    expected = (100.0 - 90.0) + (100.0 - 95.0)
    assert pnl == expected
