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
    max_hold_days: Optional[int] = None
    target_profit_pct: Optional[float] = None
    rebalance_freq: int = 21  # Default to weekly rebalancing
    max_concurrent_positions: int = 5  # Default to 5 positions

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 5) -> pd.Series:
    """Calculate Average True Range (ATR) for stop-loss sizing."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=lookback).mean()
    return atr

def estimate_skew_ratio(z_score: pd.Series, window: int = 20) -> Tuple[float, float]:
    """Estimate downside vs upside potential based on historical Z-score behavior."""
    returns = z_score.diff()
    downside = returns[returns < 0].abs().mean()
    upside = returns[returns > 0].abs().mean()
    return downside, upside

class PairsBacktest:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.prices = None
        self.signals = None
        self.trades = []
        self.daily_returns = pd.Series(dtype=float)
        self.equity_curve = pd.Series(dtype=float)
        self.positions = {}  # Track active positions
        self.last_rebalance = None
        self.target_volatility = config.target_volatility
        self.rebalance_freq = config.rebalance_freq
        self.max_concurrent_positions = config.max_concurrent_positions
        self.regime_stats = {0: [], 1: [], 2: []}  # Track returns by regime
        self.initial_capital = config.initial_capital
        # Support either dataclass or dict configs
        if isinstance(config, dict):
            self.risk_config = config.get('risk_management', {})
        else:
            self.risk_config = getattr(config, 'risk_management', {})
        
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
        position_size = (self.config.initial_capital * self.target_volatility) / float(current_vol)
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

        max_hold_days = getattr(self.config, 'max_hold_days', None)
        target_profit_pct = getattr(self.config, 'target_profit_pct', None)

        active_trades = {}
        
        for date in dates:
            try:
                # Check for exits first
                for pair, trade in list(active_trades.items()):
                    if self.should_exit_trade(trade, date):
                        self.close_trade(trade, date)
                        del active_trades[pair]

                # Check for new entries
                for pair, signal_df in signals.items():
                    if pair in active_trades:
                        continue

                    # Skip if signal data is missing for this date
                    if date not in signal_df.index:
                        continue

                    # Skip if price data is missing for either asset
                    if pair[0] not in prices.columns or pair[1] not in prices.columns:
                        continue
                    if pd.isna(prices.loc[date, pair[0]]) or pd.isna(prices.loc[date, pair[1]]):
                        continue

                    if not self.check_skew_filter(pair, date):
                        continue

                    # Get regime, defaulting to 0 if missing
                    regime = regimes.get(date, 0)
                    position_size = self.get_position_size(pair, date, regime)

                    if signal_df.loc[date, 'entry_long']:
                        entry_price = prices.loc[date, pair[0]]
                        stop_loss = self.calculate_stop_loss(pair, date, 'long', entry_price)
                        trade = self.open_trade(pair, date, 'long', entry_price,
                                              stop_loss, position_size)
                        active_trades[pair] = trade

                    elif signal_df.loc[date, 'entry_short']:
                        entry_price = prices.loc[date, pair[1]]
                        stop_loss = self.calculate_stop_loss(pair, date, 'short', entry_price)
                        trade = self.open_trade(pair, date, 'short', entry_price,
                                              stop_loss, position_size)
                        active_trades[pair] = trade

                # Update open positions and equity curve
                self._update_positions(date, prices.loc[date])

            except Exception as e:
                logger.error(f"Error processing date {date}: {str(e)}")
                continue

        # Calculate daily returns
        self.daily_returns = self.equity_curve.pct_change()
        # Fill any NaN values in daily returns with 0
        self.daily_returns = self.daily_returns.fillna(0)
    
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
        logger.info(
            "Opened %s position in %s at %s | z=%s vol=%s coint_p=%s adf_p=%s hurst=%s regime=%s",
            direction,
            pair,
            date,
            f"{entry_zscore:.3f}" if entry_zscore is not None else "NA",
            f"{entry_spread_vol:.3f}" if entry_spread_vol is not None else "NA",
            entry_coint_p,
            entry_adf_p,
            entry_hurst,
            entry_regime,
        )
    
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
    
    def rebalance_positions(self, date):
        """Rebalance positions to maintain target volatility."""
        if self.last_rebalance is None or (date - self.last_rebalance).days >= self.rebalance_freq:
            # Only rebalance open positions. A trade is considered open if it has
            # no recorded exit date yet.
            active_positions = [p for p in self.positions.values() if p.exit_date is None]
            if not active_positions:
                return
            
            # Calculate current portfolio volatility
            recent_returns = self.daily_returns.last(f"{self.rebalance_freq}D")
            current_vol = recent_returns.std() * np.sqrt(252)
            
            if current_vol > self.target_volatility:
                # Reduce position sizes proportionally
                scale_factor = self.target_volatility / current_vol
                for pos in active_positions:
                    pos.size *= scale_factor
            
            self.last_rebalance = date

    def get_performance_metrics(self) -> dict:
        """Calculate enhanced performance metrics including regime-specific stats."""
        if self.daily_returns.empty:
            return {}
        
        # Basic metrics
        annualized_return = self.daily_returns.mean() * 252
        annualized_vol = self.daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
        
        # Sortino Ratio (using downside deviation)
        downside_returns = self.daily_returns[self.daily_returns < 0]
        downside_dev = downside_returns.std() * np.sqrt(252)
        sortino_ratio = annualized_return / downside_dev if downside_dev != 0 else 0
        
        # Calmar Ratio
        max_drawdown = self.calculate_max_drawdown()
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Regime-specific metrics
        regime_returns = {regime: [] for regime in self.regime_stats.keys()}
        for trade in self.trades:
            if trade.exit_date and getattr(trade, 'entry_regime', None) is not None:
                regime_returns[trade.entry_regime].append(trade.pnl)
        
        regime_metrics = {}
        for regime, returns in regime_returns.items():
            if returns:
                regime_metrics[f'regime_{regime}_return'] = np.mean(returns)
                regime_metrics[f'regime_{regime}_sharpe'] = (
                    np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
                )
        
        # Trade statistics
        closed_trades = [t for t in self.trades if t.exit_date is not None]
        winning_trades = [t for t in closed_trades if t.pnl and t.pnl > 0]
        avg_trade_duration = np.mean([(t.exit_date - t.entry_date).days for t in closed_trades]) if closed_trades else 0
        
        return {
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': len(winning_trades) / len(closed_trades) if closed_trades else 0,
            'avg_holding_period': avg_trade_duration,
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            **regime_metrics
        }

    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve."""
        if self.equity_curve.empty:
            return 0.0
        rolling_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - rolling_max) / rolling_max
        return drawdown.min()
    
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
        """Update open positions and equity curve."""
        try:
            # Calculate P&L for all open positions
            daily_pnl = 0.0
            for pair, trade in self.positions.items():
                if pair[0] not in prices.index or pair[1] not in prices.index:
                    continue
                    
                current_price1 = prices[pair[0]]
                current_price2 = prices[pair[1]]
                
                if pd.isna(current_price1) or pd.isna(current_price2):
                    continue
                    
                pnl = self.calculate_trade_pnl(trade, current_price1, current_price2)
                daily_pnl += pnl

            # Update equity curve
            if date in self.equity_curve.index:
                if date == self.equity_curve.index[0]:
                    self.equity_curve.loc[date] = self.initial_capital + daily_pnl
                else:
                    prev_equity = self.equity_curve.loc[:date].iloc[-2]
                    self.equity_curve.loc[date] = prev_equity + daily_pnl

            # Update realized P&L
            self.realized_pnl.loc[date] = daily_pnl

        except Exception as e:
            logger.error(f"Error updating positions for date {date}: {str(e)}")
            # Set default values if update fails
            if date in self.equity_curve.index:
                if date == self.equity_curve.index[0]:
                    self.equity_curve.loc[date] = self.initial_capital
                else:
                    self.equity_curve.loc[date] = self.equity_curve.loc[:date].iloc[-2]
            self.realized_pnl.loc[date] = 0.0

    def calculate_stop_loss(self, pair: Tuple[str, str], date: pd.Timestamp, 
                          position_type: str, entry_price: float) -> float:
        """Calculate stop-loss level based on configured mode."""
        if self.risk_config.get('stop_loss_mode') == 'atr':
            # ATR-based stop-loss
            lookback = self.risk_config.get('atr_lookback', 5)
            prices = self.prices.loc[:date, list(pair)]
            atr = calculate_atr(prices.max(axis=1), prices.min(axis=1), 
                              prices.mean(axis=1), lookback)
            atr_value = atr.loc[date]
            if position_type == 'long':
                return entry_price - (atr_value * self.config.stop_loss_k)
            else:
                return entry_price + (atr_value * self.config.stop_loss_k)
        else:
            # Fixed multiplier stop-loss
            if position_type == 'long':
                return entry_price * (1 - self.config.stop_loss_k / 100)
            else:
                return entry_price * (1 + self.config.stop_loss_k / 100)

    def check_skew_filter(self, pair: Tuple[str, str], date: pd.Timestamp) -> bool:
        """Check if trade passes skew filter criteria."""
        if not self.risk_config.get('skew_filter', False):
            return True
        
        z_score = self.signals[pair]['z_score']
        downside, upside = estimate_skew_ratio(z_score.loc[:date])
        ratio = downside / upside if upside > 0 else float('inf')
        threshold = self.risk_config.get('down_up_ratio_threshold', 1.5)
        return ratio <= threshold

    def get_position_size(self, pair: Tuple[str, str], date: pd.Timestamp, 
                         regime: int) -> float:
        """Calculate position size based on regime and risk settings."""
        base_size = 1.0
        if self.risk_config.get('regime_sizing', False):
            # Reduce size in unstable regimes
            if regime > 0:  # Assuming regime 0 is stable
                base_size = 0.5
        return base_size

    def should_exit_trade(self, trade: Trade, date: pd.Timestamp) -> bool:
        """Determine if a trade should be exited based on the exit condition."""
        if trade.exit_date is not None:
            return True
        if (self.config.max_hold_days is not None and
                date >= trade.entry_date + pd.Timedelta(days=self.config.max_hold_days)):
            trade.exit_date = date
            trade.exit_price1 = self.prices.loc[date, trade.asset1]
            trade.exit_price2 = self.prices.loc[date, trade.asset2]
            trade.exit_reason = 'max_hold_days'
            return True

        if self.config.target_profit_pct is not None:
            price1 = self.prices.loc[date, trade.asset1]
            price2 = self.prices.loc[date, trade.asset2]
            current_pnl = self.calculate_trade_pnl(
                trade, current_price1=price1, current_price2=price2
            )
            if current_pnl / trade.size >= self.config.target_profit_pct:
                trade.exit_date = date
                trade.exit_price1 = price1
                trade.exit_price2 = price2
                trade.exit_reason = 'target_profit'
                return True
        return False

    def close_trade(self, trade: Trade, date: pd.Timestamp) -> None:
        """Close a trade and update equity curve."""
        trade.exit_date = date
        trade.exit_price1 = self.prices.loc[date, trade.asset1]
        trade.exit_price2 = self.prices.loc[date, trade.asset2]
        trade.exit_reason = 'exit'
        trade.pnl = self.calculate_trade_pnl(trade)

        # Record realized P&L for the day
        if date in self.realized_pnl.index:
            self.realized_pnl.loc[date] += trade.pnl
        else:
            self.realized_pnl.loc[date] = trade.pnl

        # Immediately update equity curve to reflect realized gains
        idx = self.equity_curve.index.get_loc(date) if date in self.equity_curve.index else None
        daily_pnl = self.realized_pnl.loc[date] + self._calculate_daily_pnl(date)
        if idx is not None and idx > 0:
            prev_date = self.equity_curve.index[idx - 1]
            self.equity_curve.loc[date] = float(self.equity_curve.loc[prev_date]) + float(daily_pnl)
        else:
            self.equity_curve.loc[date] = float(self.config.initial_capital) + float(daily_pnl)

        # Safely remove position if it exists
        self.positions.pop((trade.asset1, trade.asset2), None)
        logger.info(
            f"Closed position in {(trade.asset1, trade.asset2)} at {date} with P&L: ${trade.pnl:,.2f}"
        )

    def open_trade(self, pair: Tuple[str, str], date: pd.Timestamp, 
                  direction: str, entry_price: float, stop_loss: float, size: float) -> Trade:
        """Open a new trade and update equity curve."""
        asset1, asset2 = pair
        trade = Trade(
            entry_date=date,
            exit_date=None,
            asset1=asset1,
            asset2=asset2,
            direction=direction,
            entry_price1=entry_price,
            entry_price2=entry_price,
            exit_price1=None,
            exit_price2=None,
            size=size,
            pnl=None,
            exit_reason=None,
            entry_zscore=None,
            entry_spread_vol=None,
            entry_coint_p=None,
            entry_adf_p=None,
            entry_hurst=None,
            entry_regime=None,
            stop_loss_k=self.config.stop_loss_k
        )
        
        # Add trade to active trades
        self.positions[pair] = trade
        self.trades.append(trade)
        
        # Update positions
        self._update_positions(trade)
        
        # Update equity curve
        self._update_equity_curve(trade)
        
        # Update risk metrics
        self._update_risk_metrics(trade)
        
        # Log trade details
        self._log_trade_details(trade)
        
        return trade

    def _update_equity_curve(self, trade: Trade):
        """Update equity curve with new trade."""
        # Calculate initial position value
        position_value = trade.size * (trade.entry_price1 + trade.entry_price2)
        
        # Update equity curve
        self.equity_curve[trade.entry_date] = position_value

    def _update_positions(self, trade: Trade):
        """Update position tracking."""
        self.positions[trade.asset1] = trade.size if trade.direction == 'long' else -trade.size
        self.positions[trade.asset2] = -trade.size if trade.direction == 'long' else trade.size

    def _update_risk_metrics(self, trade: Trade):
        """Update risk metrics with new trade."""
        self.current_exposure += trade.size
        self._check_position_limits()

    def _log_trade_details(self, trade: Trade):
        """Log detailed trade information."""
        self.logger.info(f"Trade details: {trade.__dict__}")
