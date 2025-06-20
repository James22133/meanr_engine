"""
Metrics calculation module for analyzing trading performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime

class MetricsCalculator:
    """Class for calculating trading performance metrics."""
    
    def __init__(self, config):
        """Initialize the metrics calculator with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)

    def calculate_portfolio_metrics(self, backtest_results: Dict) -> Dict:
        """Calculate comprehensive portfolio performance metrics."""
        try:
            if not backtest_results:
                self.logger.warning("No backtest results provided for metrics calculation")
                return {}
            
            # Combine all equity curves
            all_equity_curves = []
            all_daily_returns = []
            all_trades = []
            
            for pair, results in backtest_results.items():
                if 'equity_curve' in results and results['equity_curve'] is not None:
                    all_equity_curves.append(results['equity_curve'])
                
                if 'daily_returns' in results and results['daily_returns'] is not None:
                    all_daily_returns.append(results['daily_returns'])
                
                if 'trades' in results and results['trades'] is not None:
                    all_trades.extend(results['trades'])
            
            if not all_equity_curves:
                self.logger.error("No equity curves found in backtest results")
                return {}
            
            # Calculate portfolio equity curve
            portfolio_equity = pd.concat(all_equity_curves, axis=1).sum(axis=1)
            
            # Calculate portfolio daily returns
            if all_daily_returns:
                portfolio_returns = pd.concat(all_daily_returns, axis=1).sum(axis=1)
            else:
                portfolio_returns = portfolio_equity.pct_change().fillna(0)
            
            # Calculate comprehensive metrics
            metrics = self._calculate_basic_metrics(portfolio_equity, portfolio_returns)
            metrics.update(self._calculate_risk_metrics(portfolio_equity, portfolio_returns))
            metrics.update(self._calculate_trade_metrics(all_trades))
            metrics.update(self._calculate_advanced_metrics(portfolio_equity, portfolio_returns))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {}

    def _calculate_basic_metrics(self, equity_curve: pd.Series, daily_returns: pd.Series) -> Dict:
        """Calculate basic performance metrics."""
        try:
            initial_capital = equity_curve.iloc[0]
            final_capital = equity_curve.iloc[-1]
            
            # Total return
            total_return = (final_capital - initial_capital) / initial_capital
            
            # Annualized return
            days = (equity_curve.index[-1] - equity_curve.index[0]).days
            years = days / 365.25
            annualized_return = ((final_capital / initial_capital) ** (1 / years)) - 1 if years > 0 else 0
            
            # Annualized volatility
            annualized_volatility = daily_returns.std() * np.sqrt(252)
            
            # Sharpe ratio
            risk_free_rate = getattr(self.config, 'risk_free_rate', 0.02)  # Default 2%
            excess_return = annualized_return - risk_free_rate
            sharpe_ratio = excess_return / annualized_volatility if annualized_volatility > 0 else 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'initial_capital': initial_capital,
                'final_capital': final_capital,
                'total_pnl': final_capital - initial_capital
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating basic metrics: {str(e)}")
            return {}

    def _calculate_risk_metrics(self, equity_curve: pd.Series, daily_returns: pd.Series) -> Dict:
        """Calculate risk metrics."""
        try:
            # Maximum drawdown
            rolling_max = equity_curve.cummax()
            drawdown = (equity_curve - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Sortino ratio (using downside deviation)
            downside_returns = daily_returns[daily_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252)
            risk_free_rate = getattr(self.config, 'risk_free_rate', 0.02)
            annualized_return = self._calculate_annualized_return(equity_curve)
            excess_return = annualized_return - risk_free_rate
            sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(daily_returns, 5)
            
            # Conditional Value at Risk (Expected Shortfall)
            cvar_95 = daily_returns[daily_returns <= var_95].mean()
            
            # Skewness and Kurtosis
            skewness = daily_returns.skew()
            kurtosis = daily_returns.kurtosis()
            
            return {
                'max_drawdown': max_drawdown,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'downside_deviation': downside_deviation
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {str(e)}")
            return {}

    def _calculate_trade_metrics(self, trades: List) -> Dict:
        """Calculate trade-specific metrics."""
        try:
            if not trades:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'profit_factor': 0.0,
                    'avg_holding_period': 0.0,
                    'largest_win': 0.0,
                    'largest_loss': 0.0,
                    'total_pnl': 0.0,
                    'gross_profit': 0.0,
                    'gross_loss': 0.0
                }
            
            # Filter out trades without PnL and only count closed trades
            valid_trades = [t for t in trades if hasattr(t, 'pnl') and t.pnl is not None and hasattr(t, 'exit_date') and t.exit_date is not None]
            
            if not valid_trades:
                return {
                    'total_trades': len([t for t in trades if hasattr(t, 'exit_date') and t.exit_date is not None]),
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'profit_factor': 0.0,
                    'avg_holding_period': 0.0,
                    'largest_win': 0.0,
                    'largest_loss': 0.0,
                    'total_pnl': 0.0,
                    'gross_profit': 0.0,
                    'gross_loss': 0.0
                }
            
            # Calculate trade statistics
            pnls = [t.pnl for t in valid_trades]
            winning_trades = [p for p in pnls if p > 0]
            losing_trades = [p for p in pnls if p < 0]
            
            total_trades = len(valid_trades)
            winning_count = len(winning_trades)
            losing_count = len(losing_trades)
            
            win_rate = winning_count / total_trades if total_trades > 0 else 0
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            
            gross_profit = sum(winning_trades) if winning_trades else 0
            gross_loss = abs(sum(losing_trades)) if losing_trades else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Calculate holding periods
            holding_periods = []
            for trade in valid_trades:
                if hasattr(trade, 'entry_date') and hasattr(trade, 'exit_date'):
                    if trade.entry_date and trade.exit_date:
                        holding_period = (trade.exit_date - trade.entry_date).days
                        holding_periods.append(holding_period)
            
            avg_holding_period = np.mean(holding_periods) if holding_periods else 0
            
            # Largest win and loss
            largest_win = max(pnls) if pnls else 0
            largest_loss = min(pnls) if pnls else 0
            
            # Total PnL
            total_pnl = sum(pnls)
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_count,
                'losing_trades': losing_count,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'avg_holding_period': avg_holding_period,
                'largest_win': largest_win,
                'largest_loss': largest_loss,
                'total_pnl': total_pnl,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating trade metrics: {str(e)}")
            return {}

    def _calculate_advanced_metrics(self, equity_curve: pd.Series, daily_returns: pd.Series) -> Dict:
        """Calculate advanced performance metrics."""
        try:
            # Information ratio (assuming benchmark return of 0 for simplicity)
            benchmark_return = 0
            excess_returns = daily_returns - benchmark_return
            information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
            
            # Treynor ratio
            beta = 1.0  # Assuming market beta of 1 for simplicity
            risk_free_rate = getattr(self.config, 'risk_free_rate', 0.02)
            annualized_return = self._calculate_annualized_return(equity_curve)
            treynor_ratio = (annualized_return - risk_free_rate) / beta if beta != 0 else 0
            
            # Jensen's alpha
            market_return = risk_free_rate  # Simplified assumption
            jensen_alpha = annualized_return - (risk_free_rate + beta * (market_return - risk_free_rate))
            
            # Recovery factor
            max_drawdown = self._calculate_max_drawdown(equity_curve)
            recovery_factor = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0
            
            # Sterling ratio
            sterling_ratio = (annualized_return - risk_free_rate) / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Gain to pain ratio
            positive_returns = daily_returns[daily_returns > 0]
            negative_returns = daily_returns[daily_returns < 0]
            gain_to_pain = positive_returns.sum() / abs(negative_returns.sum()) if negative_returns.sum() != 0 else float('inf')
            
            # Best and worst periods
            rolling_30d = daily_returns.rolling(30).sum()
            best_30d = rolling_30d.max()
            worst_30d = rolling_30d.min()
            
            return {
                'information_ratio': information_ratio,
                'treynor_ratio': treynor_ratio,
                'jensen_alpha': jensen_alpha,
                'recovery_factor': recovery_factor,
                'sterling_ratio': sterling_ratio,
                'gain_to_pain_ratio': gain_to_pain,
                'best_30d_return': best_30d,
                'worst_30d_return': worst_30d
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating advanced metrics: {str(e)}")
            return {}

    def _calculate_annualized_return(self, equity_curve: pd.Series) -> float:
        """Calculate annualized return from equity curve."""
        try:
            initial_capital = equity_curve.iloc[0]
            final_capital = equity_curve.iloc[-1]
            days = (equity_curve.index[-1] - equity_curve.index[0]).days
            years = days / 365.25
            return ((final_capital / initial_capital) ** (1 / years)) - 1 if years > 0 else 0
        except Exception:
            return 0.0

    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown from equity curve."""
        try:
            rolling_max = equity_curve.cummax()
            drawdown = (equity_curve - rolling_max) / rolling_max
            return drawdown.min()
        except Exception:
            return 0.0

    def calculate_pair_metrics(self, pair_results: Dict) -> Dict:
        """Calculate metrics for a single pair."""
        try:
            if not pair_results or 'equity_curve' not in pair_results:
                return {}
            
            equity_curve = pair_results['equity_curve']
            daily_returns = pair_results.get('daily_returns', equity_curve.pct_change().fillna(0))
            
            metrics = self._calculate_basic_metrics(equity_curve, daily_returns)
            metrics.update(self._calculate_risk_metrics(equity_curve, daily_returns))
            
            if 'trades' in pair_results:
                metrics.update(self._calculate_trade_metrics(pair_results['trades']))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating pair metrics: {str(e)}")
            return {} 