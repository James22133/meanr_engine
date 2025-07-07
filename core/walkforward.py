"""Walk-forward validation module for adaptive signal tuning."""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import logging
from dateutil.relativedelta import relativedelta
from .pair_monitor import PairHealthMonitor


class WalkForwardValidator:
    """Perform walk-forward validation and adaptive threshold tuning."""

    def __init__(self, config, data_loader, pair_selector, backtest_runner, metrics_calculator):
        self.config = config
        self.data_loader = data_loader
        self.pair_selector = pair_selector
        self.backtest_runner = backtest_runner
        self.metrics_calculator = metrics_calculator
        self.logger = logging.getLogger(__name__)

    def _generate_folds(self) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        start = pd.to_datetime(self.config.start_date)
        end = pd.to_datetime(self.config.end_date)
        train_months = self.config.walkforward.train_months
        test_months = self.config.walkforward.test_months
        folds = []
        cur = start
        while cur + relativedelta(months=train_months + test_months) <= end:
            train_end = cur + relativedelta(months=train_months)
            test_end = train_end + relativedelta(months=test_months)
            folds.append((cur, train_end, test_end))
            cur = cur + relativedelta(months=test_months)
        return folds

    def _tune_thresholds(self, pair: Tuple[str, str], train_data: pd.DataFrame) -> Tuple[float, float]:
        """Grid search thresholds on training data."""
        best_metric = -np.inf
        best_entry = self.config.walkforward.z_entry_values[0]
        best_exit = self.config.walkforward.z_exit_values[0]
        pair_metrics = self.pair_selector.calculate_pair_metrics(train_data)
        for entry in self.config.walkforward.z_entry_values:
            for exit_ in self.config.walkforward.z_exit_values:
                res = self.backtest_runner.run_backtest(
                    train_data,
                    pair_metrics,
                    entry_threshold=entry,
                    exit_threshold=exit_,
                )
                metric = res['metrics'].get(self.config.walkforward.scoring_metric, 0)
                if metric > best_metric:
                    best_metric = metric
                    best_entry = entry
                    best_exit = exit_
        return best_entry, best_exit

    def run(self, pairs: List[Tuple[str, str]]) -> List[Dict]:
        """Run walk-forward validation for the given pairs."""
        data = self.data_loader.data
        folds = self._generate_folds()
        all_fold_results = []
        for i, (train_start, train_end, test_end) in enumerate(folds):
            self.logger.info(f"Fold {i+1}: {train_start.date()} to {test_end.date()}")
            train_slice = data.loc[train_start:train_end]
            test_slice = data.loc[train_end:test_end]
            fold_results = {}
            for pair in pairs:
                pair_train = train_slice[list(pair)].dropna()
                pair_test = test_slice[list(pair)].dropna()
                if pair_train.empty or pair_test.empty:
                    continue
                entry, exit_ = self._tune_thresholds(pair, pair_train)
                pair_metrics = self.pair_selector.calculate_pair_metrics(pd.concat([pair_train, pair_test]))
                result = self.backtest_runner.run_backtest(
                    pair_test,
                    pair_metrics,
                    entry_threshold=entry,
                    exit_threshold=exit_,
                )
                fold_results[pair] = result
            portfolio_metrics = self.metrics_calculator.calculate_portfolio_metrics(fold_results)
            all_fold_results.append({'results': fold_results, 'metrics': portfolio_metrics})
        return all_fold_results

    def aggregate(self, fold_results: List[Dict]) -> Dict:
        """Aggregate equity curves and compute full-period metrics."""
        combined = pd.Series(dtype=float)
        for fold in fold_results:
            if not fold['results']:
                continue
            eq = pd.DataFrame({p: r['equity'] for p, r in fold['results'].items()}).sum(axis=1)
            combined = pd.concat([combined, eq])
        if combined.empty:
            return {}
        combined = combined.sort_index()
        returns = combined.pct_change().dropna()
        total_return = (combined.iloc[-1] / combined.iloc[0]) - 1
        cagr = (1 + total_return) ** (252 / len(returns)) - 1
        sharpe = np.sqrt(252) * returns.mean() / returns.std()
        drawdown = (combined / combined.cummax() - 1).min()
        return {
            'equity_curve': combined,
            'cagr': cagr,
            'sharpe_ratio': sharpe,
            'max_drawdown': drawdown,
        }


#conflict resolved here  wkkj7n-codex/modify-backtest-engine-with-slippage-and-filters
def walk_forward_backtest(
    price_data: pd.DataFrame,
    selected_pairs: List[Tuple[str, str]],
    config: Dict,
    pair_selector,
    backtester,
    output_path: str = "walkforward_stats.csv",
) -> pd.DataFrame:
#conflict resolved here 
def walk_forward_backtest(price_data: pd.DataFrame, selected_pairs: List[Tuple[str, str]], config: Dict, pair_selector, backtester) -> pd.DataFrame:
#conflict resolved here main
    """Simple rolling walk-forward analysis with health checks."""
    results = []
    train_size = 180
    test_size = 60
    monitor = PairHealthMonitor()
    for start in range(0, len(price_data) - train_size - test_size, test_size):
        train = price_data.iloc[start:start + train_size]
        test = price_data.iloc[start + train_size:start + train_size + test_size]
        pair_metrics = {}
        for pair in selected_pairs:
            if pair[0] in train.columns and pair[1] in train.columns:
                pair_df = train[[pair[0], pair[1]]].dropna()
                if len(pair_df) >= 50:
                    m = pair_selector.calculate_pair_metrics_enhanced(pair_df)
                    if m and m.get('meets_criteria', False):
                        pair_metrics[pair] = m
        window_pairs = pair_selector.select_pairs_enhanced(pair_metrics)
        window_stats = []
        for pair in window_pairs:
            if pair[0] in test.columns and pair[1] in test.columns:
                pair_df = test[[pair[0], pair[1]]].dropna()
                if len(pair_df) >= 50:
                    spread = pair_df.iloc[:, 0] - pair_df.iloc[:, 1]
                    health = monitor.evaluate(spread).dropna()
                    if not health.empty:
                        last = health.iloc[-1]
                        if last['healthy']:
                            res = backtester.run_vectorized_backtest(
                                pair_df.iloc[:, 0],
                                pair_df.iloc[:, 1],
                                lookback=config['signals'].get('lookback', 20),
                                entry_threshold=config['signals'].get('entry_threshold', 2.0),
                                exit_threshold=config['signals'].get('exit_threshold', 0.5),
                            )
                            if res and 'returns' in res:
                                ret = res['returns']
                                sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
                                total_ret = (1 + ret).prod() - 1
                                window_stats.append({'pair': pair, 'sharpe': sharpe, 'return': total_ret})
        if window_stats:
            df = pd.DataFrame(window_stats)
#conflict resolved here  wkkj7n-codex/modify-backtest-engine-with-slippage-and-filters
            avg_sharpe = df['sharpe'].mean()
            avg_return = df['return'].mean()

            portfolio_eq = (1 + df['return']).cumprod()
            rolling_sharpe = (
                portfolio_eq.pct_change().rolling(30).mean()
                / portfolio_eq.pct_change().rolling(30).std()
            ).iloc[-1] * np.sqrt(252)
            max_dd = (portfolio_eq / portfolio_eq.cummax() - 1).min()

            results.append({
                'start': train.index[0],
                'end': test.index[-1],
                'avg_sharpe': avg_sharpe,
                'avg_return': avg_return,
                'rolling_sharpe': rolling_sharpe,
                'max_drawdown': max_dd,
                'pairs': len(window_stats)
            })

    if results:
        pd.DataFrame(results).to_csv(output_path, index=False)

#conflict resolved here 
            results.append({
                'start': train.index[0],
                'end': test.index[-1],
                'avg_sharpe': df['sharpe'].mean(),
                'avg_return': df['return'].mean(),
                'pairs': len(window_stats)
            })
    if results:
        pd.DataFrame(results).to_csv('walkforward_stats.csv', index=False)
#conflict resolved here  main
    return pd.DataFrame(results)
