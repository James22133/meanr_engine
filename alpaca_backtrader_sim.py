import backtrader as bt
from datetime import datetime
import numpy as np
import statsmodels.api as sm

class PairsTradingStrategy(bt.Strategy):
    params = dict(
        pairs=[("VLO", "XLE")],
        lookback=20,
        entry_z=2.0,
        exit_z=0.5,
        position_size=1000,
        atr_mult=2.0,
        max_pairs=5,
    )

    def __init__(self):
        self.pair_info = []
        data_index = 0
        for asset1, asset2 in self.p.pairs:
            d0 = self.datas[data_index]
            d1 = self.datas[data_index + 1]
            info = dict(
                asset1=asset1,
                asset2=asset2,
                data0=d0,
                data1=d1,
                atr0=bt.indicators.ATR(d0, period=14),
                atr1=bt.indicators.ATR(d1, period=14),
                hedge_ratio=1.0,
                entry_price0=None,
                entry_price1=None,
            )
            self.pair_info.append(info)
            data_index += 2

    def log(self, txt):
        dt = self.datas[0].datetime.datetime(0)
        print(f"{dt.isoformat()} {txt}")

    def next(self):
        open_trades = sum(
            1
            for info in self.pair_info
            if self.getposition(info["data0"]).size or self.getposition(info["data1"]).size
        )

        for info in self.pair_info:
            d0 = info["data0"]
            d1 = info["data1"]
            if len(d0) < self.p.lookback or len(d1) < self.p.lookback:
                continue

            prices1 = np.log(d0.close.get(size=self.p.lookback))
            prices2 = np.log(d1.close.get(size=self.p.lookback))
            x = sm.add_constant(prices2)
            model = sm.OLS(prices1, x).fit()
            info["hedge_ratio"] = model.params[1]

            spread = prices1 - info["hedge_ratio"] * prices2
            zscore = (spread[-1] - spread.mean()) / spread.std()

            pos0 = self.getposition(d0).size
            pos1 = self.getposition(d1).size
            has_position = pos0 or pos1

            pair_label = f"{info['asset1']}/{info['asset2']}"

            if not has_position:
                if open_trades >= self.p.max_pairs:
                    continue
                if zscore > self.p.entry_z:
                    self.log(f"SHORT {pair_label} z={zscore:.2f}")
                    size0 = self.p.position_size / d0.close[0]
                    size1 = self.p.position_size / d1.close[0] * info["hedge_ratio"]
                    self.sell(data=d0, size=size0)
                    self.buy(data=d1, size=size1)
                    info["entry_price0"] = d0.close[0]
                    info["entry_price1"] = d1.close[0]
                    open_trades += 1
                elif zscore < -self.p.entry_z:
                    self.log(f"LONG {pair_label} z={zscore:.2f}")
                    size0 = self.p.position_size / d0.close[0]
                    size1 = self.p.position_size / d1.close[0] * info["hedge_ratio"]
                    self.buy(data=d0, size=size0)
                    self.sell(data=d1, size=size1)
                    info["entry_price0"] = d0.close[0]
                    info["entry_price1"] = d1.close[0]
                    open_trades += 1
            else:
                if abs(zscore) < self.p.exit_z:
                    self.log(f"EXIT {pair_label} z={zscore:.2f}")
                    self.close(d0)
                    self.close(d1)
                else:
                    if pos0 > 0:
                        stop = self.p.atr_mult * info["atr0"][0]
                        if d0.close[0] < info["entry_price0"] - stop:
                            self.log(f"Stop loss {info['asset1']}")
                            self.close(d0)
                        stop = self.p.atr_mult * info["atr1"][0]
                        if d1.close[0] > info["entry_price1"] + stop:
                            self.log(f"Stop loss {info['asset2']}")
                            self.close(d1)
                    else:
                        stop = self.p.atr_mult * info["atr0"][0]
                        if d0.close[0] > info["entry_price0"] + stop:
                            self.log(f"Stop loss {info['asset1']}")
                            self.close(d0)
                        stop = self.p.atr_mult * info["atr1"][0]
                        if d1.close[0] < info["entry_price1"] - stop:
                            self.log(f"Stop loss {info['asset2']}")
                            self.close(d1)


def run_backtest(api_key, secret_key, base_url, pairs, start="2023-01-01", end="2024-06-01"):
    store = bt.stores.AlpacaStore(key_id=api_key, secret_key=secret_key, paper=True, usePolygon=False, base_url=base_url)
    cerebro = bt.Cerebro()
    cerebro.addstrategy(PairsTradingStrategy, pairs=pairs)
    broker = store.getbroker()
    cerebro.setbroker(broker)

    DataFactory = store.getdata
    for asset1, asset2 in pairs:
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


def run_live(api_key, secret_key, base_url, pairs):
    store = bt.stores.AlpacaStore(key_id=api_key, secret_key=secret_key, paper=True, usePolygon=False, base_url=base_url)
    cerebro = bt.Cerebro()
    cerebro.addstrategy(PairsTradingStrategy, pairs=pairs)
    broker = store.getbroker()
    cerebro.setbroker(broker)

    DataFactory = store.getdata
    for asset1, asset2 in pairs:
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
    parser.add_argument("--pairs", default="VLO/XLE,COP/CVX,EFA/QQQ",
                        help="Comma-separated asset pairs e.g. VLO/XLE,COP/CVX")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-06-01")
    args = parser.parse_args()

    pairs = [tuple(p.split("/")) for p in args.pairs.split(",")]

    if args.mode == "backtest":
        run_backtest(args.api_key, args.secret_key, args.base_url, pairs,
                    start=args.start, end=args.end)
    else:
        run_live(args.api_key, args.secret_key, args.base_url, pairs)
