import datetime
import pandas as pd
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
        stop_loss_k=2.0,
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


def test_calculate_trade_pnl_unrealized():
    bt = make_backtester()
    trade = make_trade(
        exit_date=None,
        exit_price1=None,
        exit_price2=None,
        entry_price1=100.0,
        entry_price2=95.0,
    )
    pnl = bt.calculate_trade_pnl(trade, current_price1=105.0, current_price2=90.0)
    expected = (105.0 - 100.0) - (90.0 - 95.0)
    assert pnl == expected


def test_calculate_daily_pnl_unrealized():
    bt = make_backtester()
    date1 = datetime.datetime(2023, 1, 1)
    date2 = datetime.datetime(2023, 1, 2)
    prices = pd.DataFrame(
        {
            "A": [100.0, 105.0],
            "B": [95.0, 90.0],
        },
        index=[date1, date2],
    )
    bt.prices = prices
    trade = make_trade(
        entry_date=date1,
        exit_date=None,
        exit_price1=None,
        exit_price2=None,
        entry_price1=100.0,
        entry_price2=95.0,
        size=1.0,
    )
    bt.positions[("A", "B")] = trade
    pnl = bt._calculate_daily_pnl(date2)
    expected = (105.0 - 100.0) - (90.0 - 95.0)
    assert pnl == expected


def test_update_positions_uses_trade_stop_loss_k():
    config = BacktestConfig(slippage_bps=0.0, commission_bps=0.0, stop_loss_k=1.0)
    bt = PairsBacktest(config)

    dates = pd.date_range("2023-01-01", periods=21)
    t_values = list(range(19)) + [19, 10]
    prices = pd.DataFrame(
        {
            "A": [10 + t for t in t_values],
            "B": [5] * len(t_values),
        },
        index=dates,
    )
    bt.prices = prices
    bt.realized_pnl = pd.Series(index=prices.index, data=0.0)
    bt.equity_curve = pd.Series(index=prices.index, data=bt.config.initial_capital)

    trade = Trade(
        entry_date=dates[19],
        exit_date=None,
        asset1="A",
        asset2="B",
        direction="long",
        entry_price1=prices.loc[dates[19], "A"],
        entry_price2=prices.loc[dates[19], "B"],
        exit_price1=None,
        exit_price2=None,
        size=1.0,
        pnl=None,
        exit_reason=None,
        stop_loss_k=2.0,
    )
    bt.positions[("A", "B")] = trade

    bt._update_positions(dates[20], prices.loc[dates[20]])
    assert ("A", "B") in bt.positions


def test_rebalance_positions_runs_without_attribute_error():
    bt = make_backtester()
    dates = pd.date_range("2023-01-01", periods=30)
    bt.daily_returns = pd.Series(0.0, index=dates)

    trade = make_trade(
        entry_date=dates[0],
        exit_date=None,
        exit_price1=None,
        exit_price2=None,
    )
    bt.positions[("A", "B")] = trade

    # Should not raise AttributeError even if Trade lacks is_active attribute
    bt.rebalance_positions(dates[-1])
