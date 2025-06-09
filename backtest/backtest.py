import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Represents a single pairs trade."""
    entry_date: datetime
    exit_date: Optional[datetime]
    asset1: str
    asset2: str
    direction: str  # 'long' or 'short'
    entry_price1: float
    entry_price2: float
    exit_price1: Optional[float]
    exit_price2: Optional[float]
    size: float  # Position size in dollars
    pnl: Optional[float]
    exit_reason: Optional[str]  # 'take_profit', 'stop_loss', 'signal'

@dataclass
class BacktestConfig:
    """Configuration parameters for backtesting."""
    initial_capital: float = 1_000_000
    target_volatility: float = 0.10  # 10% annualized
    slippage_bps: float = 2.0
    commission_bps: float = 1.0
    stop_loss_std: float = 2.0
    zscore_entry_threshold: float = 2.0
    zscore_exit_threshold: float = 0.1

class PairsBacktest:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trades: List[Trade] = []
        self.positions: Dict[Tuple[str, str], Trade] = {}
        self.equity_curve = pd.Series()
        self.daily_returns = pd.Series()
        
    def calculate_position_size(self, prices: pd.DataFrame, pair: Tuple[str, str]) -> float:
        """Calculate volatility-scaled position size for a pair."""
        asset1, asset2 = pair
        returns1 = prices[asset1].pct_change()
        returns2 = prices[asset2].pct_change()
        
        # Calculate pair volatility (20-day rolling)
        pair_vol = (returns1 - returns2).rolling(20).std() * np.sqrt(252)
        current_vol = pair_vol.iloc[-1]
        
        if current_vol == 0:
            logger.warning(f"Zero volatility detected for pair {pair}")
            return 0
            
        # Scale position size to target volatility
        position_size = (self.config.initial_capital * self.config.target_volatility) / current_vol
        return position_size / 2  # Divide by 2 for dollar-neutral (equal long/short)
    
    def calculate_trade_pnl(self, trade: Trade) -> float:
        """Calculate P&L for a trade including slippage and commission."""
        if trade.exit_price1 is None or trade.exit_price2 is None:
            return 0.0
            
        # Calculate raw P&L
        if trade.direction == 'long':
            pnl = (trade.exit_price1 - trade.entry_price1) - (trade.exit_price2 - trade.entry_price2)
        else:
            pnl = (trade.entry_price1 - trade.exit_price1) - (trade.entry_price2 - trade.exit_price2)
            
        # Apply position size
        pnl *= trade.size
        
        # Apply slippage and commission
        slippage = (self.config.slippage_bps / 10000) * trade.size * 2  # 2 legs
        commission = (self.config.commission_bps / 10000) * trade.size * 2  # 2 legs
        
        return pnl - slippage - commission
    
    def run_backtest(self, 
                    prices: pd.DataFrame,
                    signals: Dict[Tuple[str, str], pd.DataFrame],
                    regimes: pd.Series) -> None:
        """Run the backtest simulation."""
        dates = prices.index
        self.equity_curve = pd.Series(index=dates, data=self.config.initial_capital)
        
        for date in dates:
            # Update existing positions
            self._update_positions(date, prices.loc[date])
            
            # Check for new signals
            for pair, pair_signals in signals.items():
                if date not in pair_signals.index:
                    continue
                    
                signal = pair_signals.loc[date]
                if pair in self.positions:
                    # Check for exit conditions
                    if self._should_exit_position(pair, signal):
                        self._close_position(pair, date, prices.loc[date], 'signal')
                else:
                    # Check for entry conditions
                    if self._should_enter_position(pair, signal, regimes.loc[date]):
                        self._open_position(pair, date, prices.loc[date], signal)
            
            # Update equity curve
            if date > dates[0]:
                self.equity_curve[date] = self.equity_curve[date - 1] + self._calculate_daily_pnl(date)
        
        # Calculate daily returns
        self.daily_returns = self.equity_curve.pct_change()
    
    def _should_enter_position(self, 
                             pair: Tuple[str, str],
                             signal: pd.Series,
                             regime: int) -> bool:
        """Check if we should enter a new position."""
        if signal['entry_long']:
            return True
        if signal['entry_short']:
            return True
        return False
    
    def _should_exit_position(self, pair: Tuple[str, str], signal: pd.Series) -> bool:
        """Check if we should exit an existing position."""
        return signal['exit']
    
    def _open_position(self, 
                      pair: Tuple[str, str],
                      date: datetime,
                      prices: pd.Series,
                      signal: pd.Series) -> None:
        """Open a new position."""
        asset1, asset2 = pair
        size = self.calculate_position_size(prices, pair)
        
        if size == 0:
            logger.warning(f"Skipping trade for {pair} due to zero position size")
            return
            
        direction = 'long' if signal['entry_long'] else 'short'
        
        trade = Trade(
            entry_date=date,
            exit_date=None,
            asset1=asset1,
            asset2=asset2,
            direction=direction,
            entry_price1=prices[asset1],
            entry_price2=prices[asset2],
            exit_price1=None,
            exit_price2=None,
            size=size,
            pnl=None,
            exit_reason=None
        )
        
        self.positions[pair] = trade
        self.trades.append(trade)
        logger.info(f"Opened {direction} position in {pair} at {date}")
    
    def _close_position(self,
                       pair: Tuple[str, str],
                       date: datetime,
                       prices: pd.Series,
                       reason: str) -> None:
        """Close an existing position."""
        trade = self.positions[pair]
        trade.exit_date = date
        trade.exit_price1 = prices[trade.asset1]
        trade.exit_price2 = prices[trade.asset2]
        trade.exit_reason = reason
        trade.pnl = self.calculate_trade_pnl(trade)
        
        del self.positions[pair]
        logger.info(f"Closed position in {pair} at {date} with P&L: ${trade.pnl:,.2f}")
    
    def _calculate_daily_pnl(self, date: datetime) -> float:
        """Calculate P&L for the current day."""
        daily_pnl = 0.0
        for trade in self.positions.values():
            if trade.exit_date is None:  # Position still open
                current_pnl = self.calculate_trade_pnl(trade)
                daily_pnl += current_pnl
        return daily_pnl
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate and return performance metrics."""
        if self.daily_returns.empty:
            return {}
            
        annualized_return = self.daily_returns.mean() * 252
        annualized_vol = self.daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
        
        # Calculate drawdown
        cummax = self.equity_curve.cummax()
        drawdown = (self.equity_curve - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # Calculate win/loss metrics
        closed_trades = [t for t in self.trades if t.exit_date is not None]
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        
        # Calculate average holding period
        holding_periods = [(t.exit_date - t.entry_date).days for t in closed_trades]
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        
        return {
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_holding_period': avg_holding_period,
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades)
        }
    
    def save_trades_to_csv(self, filename: str) -> None:
        """Save trade history to CSV file."""
        trades_df = pd.DataFrame([{
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'asset1': t.asset1,
            'asset2': t.asset2,
            'direction': t.direction,
            'size': t.size,
            'entry_price1': t.entry_price1,
            'entry_price2': t.entry_price2,
            'exit_price1': t.exit_price1,
            'exit_price2': t.exit_price2,
            'pnl': t.pnl,
            'exit_reason': t.exit_reason
        } for t in self.trades])
        
        trades_df.to_csv(filename, index=False)
        logger.info(f"Saved trade history to {filename}")

    def _update_positions(self, date: datetime, prices: pd.Series) -> None:
        """Update existing positions, check stop-losses, and calculate P&L."""
        positions_to_close = []
        
        for pair, trade in self.positions.items():
            if trade.exit_date is not None:  # Skip already closed positions
                continue
            
            # Check stop-loss
            if trade.direction == 'long':
                stop_loss = trade.entry_price1 * (1 - self.config.stop_loss_std * 0.01)
                if prices[trade.asset1] <= stop_loss:
                    positions_to_close.append((pair, 'stop_loss'))
            else:  # short position
                stop_loss = trade.entry_price1 * (1 + self.config.stop_loss_std * 0.01)
                if prices[trade.asset1] >= stop_loss:
                    positions_to_close.append((pair, 'stop_loss'))
        
        # Close positions that hit stop-loss
        for pair, reason in positions_to_close:
            self._close_position(pair, date, prices, reason)
            
        # Update daily P&L
        if date > self.equity_curve.index[0]:
            daily_pnl = self._calculate_daily_pnl(date)
            self.equity_curve[date] = self.equity_curve[date - 1] + daily_pnl 
