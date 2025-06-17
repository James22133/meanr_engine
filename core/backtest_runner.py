"""
Backtest runner module for executing backtests.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

class BacktestRunner:
    """Class for running backtests on selected pairs."""
    
    def __init__(self, config):
        """Initialize the backtest runner with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)

    def run_backtest(
        self,
        pair_data: pd.DataFrame,
        pair_metrics: Dict,
        entry_threshold: Optional[float] = None,
        exit_threshold: Optional[float] = None,
    ) -> Dict:
        """
        Run backtest for a single pair.
        
        Args:
            pair_data: DataFrame with price data for the pair
            pair_metrics: Dictionary containing pair metrics
            
        Returns:
            Dict: Backtest results
        """
        try:
            # Initialize results
            results = {
                'trades': [],
                'positions': pd.Series(0, index=pair_data.index),
                'pnl': pd.Series(0.0, index=pair_data.index),
                'equity': pd.Series(self.config.backtest.initial_capital, index=pair_data.index)
            }
            
            # Get spread and Z-score
            spread = pair_metrics['spread']
            zscore = pair_metrics['zscore']
            
            # Calculate entry/exit thresholds
            if entry_threshold is None:
                entry_threshold = self._calculate_entry_threshold(zscore)
            if exit_threshold is None:
                exit_threshold = self._calculate_exit_threshold(zscore)
            
            # Run backtest
            position = 0
            entry_price = 0.0
            
            for i in range(1, len(pair_data)):
                # Get current values
                current_zscore = zscore.iloc[i]
                current_spread = spread.iloc[i]
                current_prices = pair_data.iloc[i]
                
                # Check for entry
                if position == 0:
                    if current_zscore <= -entry_threshold:
                        # Long spread
                        position = 1
                        entry_price = current_spread
                        results['trades'].append({
                            'entry_date': pair_data.index[i],
                            'entry_price': entry_price,
                            'position': position,
                            'type': 'long'
                        })
                    elif current_zscore >= entry_threshold:
                        # Short spread
                        position = -1
                        entry_price = current_spread
                        results['trades'].append({
                            'entry_date': pair_data.index[i],
                            'entry_price': entry_price,
                            'position': position,
                            'type': 'short'
                        })
                
                # Check for exit
                elif position != 0:
                    # Check stop loss
                    if self._check_stop_loss(current_spread, entry_price, position):
                        position = 0
                        results['trades'][-1]['exit_date'] = pair_data.index[i]
                        results['trades'][-1]['exit_price'] = current_spread
                        results['trades'][-1]['exit_type'] = 'stop_loss'
                    
                    # Check take profit
                    elif self._check_take_profit(current_spread, entry_price, position):
                        position = 0
                        results['trades'][-1]['exit_date'] = pair_data.index[i]
                        results['trades'][-1]['exit_price'] = current_spread
                        results['trades'][-1]['exit_type'] = 'take_profit'
                    
                    # Check mean reversion exit
                    elif (position == 1 and current_zscore >= -exit_threshold) or \
                         (position == -1 and current_zscore <= exit_threshold):
                        position = 0
                        results['trades'][-1]['exit_date'] = pair_data.index[i]
                        results['trades'][-1]['exit_price'] = current_spread
                        results['trades'][-1]['exit_type'] = 'mean_reversion'
                
                # Update position and PnL
                results['positions'].iloc[i] = position
                if position != 0:
                    pnl = position * (current_spread - entry_price)
                    results['pnl'].iloc[i] = pnl
                    results['equity'].iloc[i] = results['equity'].iloc[i-1] + pnl
                else:
                    results['equity'].iloc[i] = results['equity'].iloc[i-1]
            
            # Calculate performance metrics
            results['metrics'] = self._calculate_performance_metrics(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {str(e)}")
            return None

    def _calculate_entry_threshold(self, zscore: pd.Series) -> float:
        """Calculate dynamic entry threshold based on Z-score volatility."""
        try:
            zscore_vol = zscore.std()
            return max(1.5, min(2.5, zscore_vol * 1.5))
        except Exception as e:
            self.logger.error(f"Error calculating entry threshold: {str(e)}")
            return 2.0

    def _calculate_exit_threshold(self, zscore: pd.Series) -> float:
        """Calculate dynamic exit threshold based on Z-score volatility."""
        try:
            zscore_vol = zscore.std()
            return max(0.5, min(1.0, zscore_vol * 0.5))
        except Exception as e:
            self.logger.error(f"Error calculating exit threshold: {str(e)}")
            return 0.5

    def _check_stop_loss(self, current_spread: float, entry_price: float, position: int) -> bool:
        """Check if stop loss is triggered."""
        try:
            if position == 1:  # Long position
                return (entry_price - current_spread) / abs(entry_price) >= self.config.backtest.stop_loss_pct
            else:  # Short position
                return (current_spread - entry_price) / abs(entry_price) >= self.config.backtest.stop_loss_pct
        except Exception as e:
            self.logger.error(f"Error checking stop loss: {str(e)}")
            return False

    def _check_take_profit(self, current_spread: float, entry_price: float, position: int) -> bool:
        """Check if take profit is triggered."""
        try:
            if position == 1:  # Long position
                return (current_spread - entry_price) / abs(entry_price) >= self.config.backtest.take_profit_pct
            else:  # Short position
                return (entry_price - current_spread) / abs(entry_price) >= self.config.backtest.take_profit_pct
        except Exception as e:
            self.logger.error(f"Error checking take profit: {str(e)}")
            return False

    def _calculate_performance_metrics(self, results: Dict) -> Dict:
        """Calculate performance metrics for the backtest."""
        try:
            # Calculate returns
            returns = results['equity'].pct_change().dropna()
            
            # Calculate metrics
            total_return = (results['equity'].iloc[-1] / results['equity'].iloc[0]) - 1
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
            max_drawdown = (results['equity'] / results['equity'].cummax() - 1).min()
            
            # Calculate trade metrics
            trades = results['trades']
            winning_trades = [t for t in trades if 'exit_price' in t and 
                            (t['position'] * (t['exit_price'] - t['entry_price'])) > 0]
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'win_rate': len(winning_trades) / len(trades) if trades else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {} 