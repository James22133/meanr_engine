"""
Metrics calculator module for computing performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime

class MetricsCalculator:
    """Class for calculating performance metrics."""
    
    def __init__(self, config):
        """Initialize the metrics calculator with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)

    def calculate_portfolio_metrics(self, backtest_results: Dict[str, Dict]) -> Dict:
        """
        Calculate portfolio-level performance metrics.
        
        Args:
            backtest_results: Dictionary of backtest results for each pair
            
        Returns:
            Dict: Portfolio performance metrics
        """
        try:
            # Combine equity curves
            equity_curves = pd.DataFrame({
                pair: results['equity']
                for pair, results in backtest_results.items()
            })
            
            # Calculate portfolio equity
            portfolio_equity = equity_curves.sum(axis=1)
            
            # Calculate returns
            returns = portfolio_equity.pct_change().dropna()
            
            # Calculate metrics
            total_return = (portfolio_equity.iloc[-1] / portfolio_equity.iloc[0]) - 1
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
            sortino_ratio = self._calculate_sortino_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(portfolio_equity)
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Calculate trade metrics
            all_trades = []
            for pair, results in backtest_results.items():
                for trade in results['trades']:
                    trade['pair'] = pair
                    all_trades.append(trade)
            
            trade_metrics = self._calculate_trade_metrics(all_trades)
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'trade_metrics': trade_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {}

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        try:
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0:
                return float('inf')
            
            downside_std = np.sqrt(np.mean(downside_returns ** 2))
            if downside_std == 0:
                return float('inf')
            
            return np.sqrt(252) * returns.mean() / downside_std
            
        except Exception as e:
            self.logger.error(f"Error calculating Sortino ratio: {str(e)}")
            return 0.0

    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            rolling_max = equity.cummax()
            drawdowns = equity / rolling_max - 1
            return drawdowns.min()
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0

    def _calculate_trade_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate trade-level metrics."""
        try:
            if not trades:
                return {}
            
            # Calculate trade returns
            trade_returns = []
            for trade in trades:
                if 'exit_price' in trade and 'entry_price' in trade:
                    returns = (trade['exit_price'] - trade['entry_price']) / abs(trade['entry_price'])
                    trade_returns.append(returns)
            
            if not trade_returns:
                return {}
            
            # Calculate metrics
            trade_returns = np.array(trade_returns)
            winning_trades = trade_returns[trade_returns > 0]
            losing_trades = trade_returns[trade_returns < 0]
            
            return {
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(trades),
                'avg_return': np.mean(trade_returns),
                'avg_win': np.mean(winning_trades) if len(winning_trades) > 0 else 0,
                'avg_loss': np.mean(losing_trades) if len(losing_trades) > 0 else 0,
                'profit_factor': abs(np.sum(winning_trades) / np.sum(losing_trades)) if np.sum(losing_trades) != 0 else float('inf')
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating trade metrics: {str(e)}")
            return {} 