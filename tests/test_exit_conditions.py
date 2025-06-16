import datetime
import pandas as pd
from backtest.backtest import PairsBacktest, BacktestConfig, Trade


def make_backtester(**kwargs):
    config = BacktestConfig(slippage_bps=0.0, commission_bps=0.0, **kwargs)
    bt = PairsBacktest(config)
    return bt


def make_trade(entry_date, **kwargs):
    defaults = dict(
        entry_date=entry_date,
        exit_date=None,
        asset1="A",
        asset2="B",
        direction="long",
        entry_price1=100.0,
        entry_price2=95.0,
        exit_price1=None,
        exit_price2=None,
        size=100.0,
        pnl=None,
        exit_reason=None,
        stop_loss_k=2.0,
    )
    defaults.update(kwargs)
    return Trade(**defaults)


def test_should_exit_trade_max_hold_days():
    entry_date = datetime.datetime(2023, 1, 1)
    bt = make_backtester(max_hold_days=5)
    prices = pd.DataFrame(
        {"A": [100.0] * 6, "B": [95.0] * 6},
        index=[entry_date + datetime.timedelta(days=i) for i in range(6)],
    )
    bt.prices = prices
    trade = make_trade(entry_date)

    date = entry_date + datetime.timedelta(days=5)
    assert bt.should_exit_trade(trade, date)
    assert trade.exit_reason == "max_hold_days"


def test_should_exit_trade_target_profit():
    entry_date = datetime.datetime(2023, 1, 1)
    bt = make_backtester(target_profit_pct=0.05, max_hold_days=10)
    prices = pd.DataFrame(
        {"A": [100.0, 110.0], "B": [95.0, 92.0]},
        index=[entry_date, entry_date + datetime.timedelta(days=1)],
    )
    bt.prices = prices
    trade = make_trade(entry_date)

    date = entry_date + datetime.timedelta(days=1)
    assert bt.should_exit_trade(trade, date)
    assert trade.exit_reason == "target_profit"
