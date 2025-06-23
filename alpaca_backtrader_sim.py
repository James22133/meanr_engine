import backtrader as bt
import alpaca_trade_api as tradeapi
from datetime import datetime
import numpy as np
import statsmodels.api as sm

class PairsTradingStrategy(bt.Strategy):
    params = dict(
        asset1="VLO",
        asset2="XLE",
        lookback=20,
        entry_z=2.0,
        exit_z=0.5,
        position_size=1000,
        atr_mult=2.0,
    )

    def __init__(self):
        self.data0 = self.datas[0]
        self.data1 = self.datas[1]
        self.atr0 = bt.indicators.ATR(self.data0, period=14)
        self.atr1 = bt.indicators.ATR(self.data1, period=14)
        self.hedge_ratio = 1.0

    def log(self, txt):
        dt = self.datas[0].datetime.datetime(0)
        print(f"{dt.isoformat()} {txt}")

    def next(self):
        if len(self.data0) < self.p.lookback:
            return

        prices1 = np.log(self.data0.close.get(size=self.p.lookback))
        prices2 = np.log(self.data1.close.get(size=self.p.lookback))
        x = sm.add_constant(prices2)
        model = sm.OLS(prices1, x).fit()
        self.hedge_ratio = model.params[1]

        spread = prices1 - self.hedge_ratio * prices2
        zscore = (spread[-1] - spread.mean()) / spread.std()

        if not self.position:
            if zscore > self.p.entry_z:
                self.log(f"SHORT spread z={zscore:.2f}")
                size0 = self.p.position_size / self.data0.close[0]
                size1 = self.p.position_size / self.data1.close[0] * self.hedge_ratio
                self.sell(data=self.data0, size=size0)
                self.buy(data=self.data1, size=size1)
                self.entry_price0 = self.data0.close[0]
                self.entry_price1 = self.data1.close[0]
            elif zscore < -self.p.entry_z:
                self.log(f"LONG spread z={zscore:.2f}")
                size0 = self.p.position_size / self.data0.close[0]
                size1 = self.p.position_size / self.data1.close[0] * self.hedge_ratio
                self.buy(data=self.data0, size=size0)
                self.sell(data=self.data1, size=size1)
                self.entry_price0 = self.data0.close[0]
                self.entry_price1 = self.data1.close[0]
        else:
            if abs(zscore) < self.p.exit_z:
                self.log(f"EXIT z={zscore:.2f}")
                self.close(self.data0)
                self.close(self.data1)
            else:
                if self.position.size > 0:
                    stop = self.p.atr_mult * self.atr0[0]
                    if self.data0.close[0] < self.entry_price0 - stop:
                        self.log("Stop loss asset1")
                        self.close(self.data0)
                    stop = self.p.atr_mult * self.atr1[0]
                    if self.data1.close[0] > self.entry_price1 + stop:
                        self.log("Stop loss asset2")
                        self.close(self.data1)
                else:
                    stop = self.p.atr_mult * self.atr0[0]
                    if self.data0.close[0] > self.entry_price0 + stop:
                        self.log("Stop loss asset1")
                        self.close(self.data0)
                    stop = self.p.atr_mult * self.atr1[0]
                    if self.data1.close[0] < self.entry_price1 - stop:
                        self.log("Stop loss asset2")
                        self.close(self.data1)


def run_backtest(api_key, secret_key, base_url, asset1="VLO", asset2="XLE", start="2023-01-01", end="2024-06-01"):
    store = bt.stores.AlpacaStore(key_id=api_key, secret_key=secret_key, paper=True, usePolygon=False, base_url=base_url)
    cerebro = bt.Cerebro()
    cerebro.addstrategy(PairsTradingStrategy, asset1=asset1, asset2=asset2)
    broker = store.getbroker()
    cerebro.setbroker(broker)

    DataFactory = store.getdata
    data0 = DataFactory(dataname=asset1, historical=True,
                        fromdate=datetime.fromisoformat(start),
                        todate=datetime.fromisoformat(end), timeframe=bt.TimeFrame.Days)
    data1 = DataFactory(dataname=asset2, historical=True,
                        fromdate=datetime.fromisoformat(start),
                        todate=datetime.fromisoformat(end), timeframe=bt.TimeFrame.Days)
    cerebro.adddata(data0, name=asset1)
    cerebro.adddata(data1, name=asset2)
    cerebro.broker.setcash(100000)
    cerebro.run()
    cerebro.plot()


def run_live(api_key, secret_key, base_url, asset1="VLO", asset2="XLE"):
    store = bt.stores.AlpacaStore(key_id=api_key, secret_key=secret_key, paper=True, usePolygon=False, base_url=base_url)
    cerebro = bt.Cerebro()
    cerebro.addstrategy(PairsTradingStrategy, asset1=asset1, asset2=asset2)
    broker = store.getbroker()
    cerebro.setbroker(broker)

    DataFactory = store.getdata
    data0 = DataFactory(dataname=asset1, historical=False, timeframe=bt.TimeFrame.Days)
    data1 = DataFactory(dataname=asset2, historical=False, timeframe=bt.TimeFrame.Days)
    cerebro.adddata(data0, name=asset1)
    cerebro.adddata(data1, name=asset2)
    cerebro.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backtrader Alpaca live simulation")
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--secret_key", required=True)
    parser.add_argument("--base_url", default="https://paper-api.alpaca.markets")
    parser.add_argument("--mode", choices=["backtest", "live"], default="backtest")
    args = parser.parse_args()

    if args.mode == "backtest":
        run_backtest(args.api_key, args.secret_key, args.base_url)
    else:
        run_live(args.api_key, args.secret_key, args.base_url)
