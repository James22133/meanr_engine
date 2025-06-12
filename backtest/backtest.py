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
    entry_zscore: Optional[float] = None
    exit_zscore: Optional[float] = None
    entry_spread_vol: Optional[float] = None
    entry_coint_p: Optional[float] = None
    entry_adf_p: Optional[float] = None
    entry_hurst: Optional[float] = None
    entry_regime: Optional[int] = None
    stop_loss_k: float = 2.0

@dataclass
class BacktestConfig:
    """Configuration parameters for backtesting."""
    initial_capital: float = 1_000_000
    target_volatility: float = 0.10  # 10% annualized
    slippage_bps: float = 2.0
    commission_bps: float = 1.0
    stop_loss_k: float = 2.0  # Multiplier for volatility-based stop-loss
    zscore_entry_threshold: float = 2.0
    zscore_exit_threshold: float = 0.1

class PairsBacktest:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trades: List[Trade] = []
        self.positions: Dict[Tuple[str, str], Trade] = {}
        self.equity_curve = pd.Series(dtype=float)
        self.daily_returns = pd.Series(dtype=float)
        self.realized_pnl = pd.Series(dtype=float)
        
    def calculate_position_size(
        self,
        prices: pd.DataFrame,
        pair: Tuple[str, str],
        date: datetime,
        spread_series: Optional[pd.Series] = None
    ) -> float:
        """Calculate volatility-scaled position size for a pair up to a given date using spread volatility."""
        asset1, asset2 = pair
        if spread_series is None:
            spread_series = prices[asset1] - prices[asset2]
        spread_vol = spread_series.rolling(20).std()
        if date not in spread_vol.index:
            logger.warning(f"Date {date} not found for pair {pair} spread volatility")
            return 0
        current_vol = spread_vol.loc[:date].iloc[-1]
        if pd.isna(current_vol) or current_vol == 0:
            logger.warning(f"Spread volatility is NaN/zero for pair {pair} on {date}")
            return 0
        position_size = (self.config.initial_capital * self.config.target_volatility) / float(current_vol)
        return float(position_size) / 2  # Dollar-neutral
    
    def calculate_trade_pnl(
        self,
        trade: Trade,
        current_price1: Optional[float] = None,
        current_price2: Optional[float] = None,
    ) -> float:
        """Calculate P&L for a trade.

        If the trade is still open, current prices can be supplied to mark the
        position. Slippage and commission are only applied when the trade has
        explicit exit prices.
        """

        price1 = trade.exit_price1 if trade.exit_price1 is not None else current_price1
        price2 = trade.exit_price2 if trade.exit_price2 is not None else current_price2

        if price1 is None or price2 is None:
            return 0.0

        # Calculate raw P&L
        if trade.direction == 'long':
            pnl = (price1 - trade.entry_price1) - (price2 - trade.entry_price2)
        else:
            pnl = (trade.entry_price1 - price1) + (price2 - trade.entry_price2)

        # Apply position size
        pnl *= trade.size

        # Apply slippage and commission only when trade is closed
        if trade.exit_price1 is not None and trade.exit_price2 is not None:
            slippage = (self.config.slippage_bps / 10000) * trade.size * 2  # 2 legs
            commission = (
                self.config.commission_bps / 10000
            ) * trade.size * 2  # 2 legs
            pnl -= slippage + commission

        return pnl
    
    def run_backtest(self, 
                    prices: pd.DataFrame,
                    signals: Dict[Tuple[str, str], pd.DataFrame],
                    regimes: pd.Series) -> None:
        """Run the backtest simulation."""
        self.prices = prices  # Store full price history for use in _open_position
        dates = prices.index
        self.equity_curve = pd.Series(index=dates, data=self.config.initial_capital)
        self.realized_pnl = pd.Series(index=dates, data=0.0)

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
                        self._open_position(pair, date, prices, signal)
            
            # Equity curve is updated within _update_positions to avoid double counting

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
    
    def _open_position(
        self,
        pair: Tuple[str, str],
        date: datetime,
        prices: pd.DataFrame,
        signal: pd.Series,
    ) -> None:
        """Open a new position."""
        asset1, asset2 = pair
        # Use z-score and spread volatility from signal if available
        entry_zscore = signal.get('z_score', None)
        entry_regime = signal.get('regime', None)
        # Assume spread is asset1 - asset2
        spread_series = prices[asset1] - prices[asset2]
        entry_spread_vol = spread_series.rolling(20).std().loc[:date].iloc[-1] if date in spread_series.index else None
        # These should be passed in or looked up from metrics if available
        entry_coint_p = signal.get('coint_p', None)
        entry_adf_p = signal.get('adf_p', None)
        entry_hurst = signal.get('hurst', None)
        stop_loss_k = signal.get('stop_loss_k', getattr(self.config, 'stop_loss_k', 2.0))
        size = self.calculate_position_size(prices, pair, date, spread_series)
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
            entry_price1=prices.loc[date, asset1],
            entry_price2=prices.loc[date, asset2],
            exit_price1=None,
            exit_price2=None,
            size=size,
            pnl=None,
            exit_reason=None,
            entry_zscore=entry_zscore,
            entry_spread_vol=entry_spread_vol,
            entry_coint_p=entry_coint_p,
            entry_adf_p=entry_adf_p,
            entry_hurst=entry_hurst,
            entry_regime=entry_regime,
            stop_loss_k=stop_loss_k
        )
        self.positions[pair] = trade
        self.trades.append(trade)
        logger.info(f"Opened {direction} position in {pair} at {date} | z={entry_zscore:.3f} vol={entry_spread_vol:.3f} coint_p={entry_coint_p} adf_p={entry_adf_p} hurst={entry_hurst} regime={entry_regime}")
    
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
        # Log exit z-score if available
        if hasattr(self, 'prices') and hasattr(self, 'signals'):
            try:
                trade.exit_zscore = self.signals[pair].loc[date, 'z_score']
            except Exception:
                trade.exit_zscore = None
        trade.pnl = self.calculate_trade_pnl(trade)

        # Record realized P&L for the day
        if date in self.realized_pnl.index:
            self.realized_pnl.loc[date] += trade.pnl
        else:
            self.realized_pnl.loc[date] = trade.pnl

        # Immediately update equity curve to reflect realized gains
        idx = self.equity_curve.index.get_loc(date) if date in self.equity_curve.index else None
        unrealized_pnl = self._calculate_daily_pnl(date)
        daily_total_pnl = self.realized_pnl.loc[date] + unrealized_pnl

        if idx is not None and idx > 0:
            prev_date = self.equity_curve.index[idx - 1]
            base_equity = float(self.equity_curve.loc[prev_date])
        else:
            base_equity = float(self.config.initial_capital)

        self.equity_curve.loc[date] = float(base_equity + daily_total_pnl)

        del self.positions[pair]
        logger.info(
            f"Closed position in {pair} at {date} with P&L: ${trade.pnl:,.2f}"
        )
    
    def _calculate_daily_pnl(self, date: datetime) -> float:
        """Calculate P&L for the current day."""
        daily_pnl = 0.0
        for trade in self.positions.values():
            if trade.exit_date is None:  # Position still open
                price1 = self.prices.loc[date, trade.asset1]
                price2 = self.prices.loc[date, trade.asset2]
                current_pnl = self.calculate_trade_pnl(
                    trade, current_price1=price1, current_price2=price2
                )
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
        """Save trade history to CSV file with enhanced fields."""
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
            'exit_reason': t.exit_reason,
            'entry_zscore': t.entry_zscore,
            'exit_zscore': t.exit_zscore,
            'entry_spread_vol': t.entry_spread_vol,
            'entry_coint_p': t.entry_coint_p,
            'entry_adf_p': t.entry_adf_p,
            'entry_hurst': t.entry_hurst,
            'entry_regime': t.entry_regime,
            'stop_loss_k': t.stop_loss_k
        } for t in self.trades])
        trades_df.to_csv(filename, index=False)
        logger.info(f"Saved trade history to {filename}")

    def _update_positions(self, date: datetime, prices: pd.Series) -> None:
        """Update existing positions, check stop-losses, and calculate P&L."""
        positions_to_close = []
        for pair, trade in self.positions.items():
            if trade.exit_date is not None:
                continue
            # Volatility-based stop-loss
            spread = self.prices.loc[date, trade.asset1] - self.prices.loc[date, trade.asset2]
            spread_series = self.prices[trade.asset1] - self.prices[trade.asset2]
            spread_vol = spread_series.rolling(20).std().loc[:date].iloc[-1] if date in spread_series.index else None
            if spread_vol is None or pd.isna(spread_vol):
                continue
            k = getattr(trade, 'stop_loss_k', getattr(self.config, 'stop_loss_k', 2.0))
            stop_loss_long = trade.entry_price1 - trade.entry_price2 - k * spread_vol
            stop_loss_short = trade.entry_price1 - trade.entry_price2 + k * spread_vol
            if trade.direction == 'long':
                if spread <= stop_loss_long:
                    positions_to_close.append((pair, 'stop_loss'))
            else:  # short position
                if spread >= stop_loss_short:
                    positions_to_close.append((pair, 'stop_loss'))
        # Close positions that hit stop-loss
        for pair, reason in positions_to_close:
            self._close_position(pair, date, prices, reason)
        # Update daily P&L including realized gains from closed trades
        idx = self.equity_curve.index.get_loc(date)
        daily_pnl = self.realized_pnl.loc[date] + self._calculate_daily_pnl(date)
        if idx > 0:
            prev_date = self.equity_curve.index[idx - 1]
            self.equity_curve.loc[date] = float(self.equity_curve.loc[prev_date]) + float(daily_pnl)
        else:
            self.equity_curve.loc[date] = float(self.config.initial_capital) + float(daily_pnl)
