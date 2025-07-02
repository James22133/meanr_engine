import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

from .metrics import calculate_sortino_ratio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Represents a single pairs trade."""
    pair: Tuple[str, str]
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
    holding_period: Optional[int] = None

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
    risk_control: Optional[dict] = None

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
    def __init__(self, config: BacktestConfig, data_loader=None):
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
        self.current_exposure = 0.0  # Add missing attribute
        
        # Store data loader for OHLC access
        self.data_loader = data_loader
        
        # Risk management attributes
        if isinstance(config, dict):
            self.risk_config = config.get('risk_control', {})
        else:
            self.risk_config = getattr(config, 'risk_control', {})
        self.atr_multiplier = self.risk_config.get('atr_multiplier', 2.0)
        self.max_drawdown_per_pair = self.risk_config.get('max_drawdown_per_pair', 0.05)  # 5%
        self.max_drawdown_per_trade = self.risk_config.get('max_drawdown_per_trade', 0.02)  # 2%
        self.max_pair_exposure = self.risk_config.get('max_pair_exposure', 0.1)  # 10%
        self.volatility_scaling = self.risk_config.get('volatility_scaling', True)
        self.atr_period = self.risk_config.get('atr_period', 14)
        self.max_pct_portfolio = self.risk_config.get('max_pct_portfolio', 0.10)
        self.max_leverage = self.risk_config.get('max_leverage', 2.0)
        self.max_total_exposure = self.risk_config.get('max_total_exposure', 1.5)
        
        # Track pair-specific metrics
        self.pair_drawdowns = {}  # Track drawdown per pair
        self.pair_exposures = {}  # Track exposure per pair
        
        # Support either dataclass or dict configs
        if isinstance(config, dict):
            self.config = config
            self.initial_capital = config.get('initial_capital', 1_000_000)
            self.target_volatility = config.get('target_volatility', 0.10)
            self.rebalance_freq = config.get('rebalance_freq', 21)
            self.max_concurrent_positions = config.get('max_concurrent_positions', 5)
            self.stop_loss_k = config.get('stop_loss_k', 2.0)
            self.max_hold_days = config.get('max_hold_days', None)
            self.target_profit_pct = config.get('target_profit_pct', None)
            self.slippage_bps = config.get('slippage_bps', 2.0)
            self.commission_bps = config.get('commission_bps', 1.0)
        else:
            self.config = config.__dict__
            self.initial_capital = config.initial_capital
            self.target_volatility = config.target_volatility
            self.rebalance_freq = config.rebalance_freq
            self.max_concurrent_positions = config.max_concurrent_positions
            self.stop_loss_k = getattr(config, 'stop_loss_k', 2.0)
            self.max_hold_days = getattr(config, 'max_hold_days', None)
            self.target_profit_pct = getattr(config, 'target_profit_pct', None)
            self.slippage_bps = getattr(config, 'slippage_bps', 2.0)
            self.commission_bps = getattr(config, 'commission_bps', 1.0)
        
    def calculate_position_size(
        self, 
        pair: Tuple[str, str], 
        entry_price: float, 
        current_capital: float,
        target_volatility: float = None
    ) -> float:
        """Calculate position size based on risk management rules."""
        try:
            # Use target volatility from config if not provided
            if target_volatility is None:
                target_volatility = self.target_volatility
            
            # Calculate base position size
            position_size = current_capital * target_volatility / 100
            
            # Apply volatility scaling if enabled
            if self.volatility_scaling:
                position_size = self.calculate_volatility_scaled_position(
                    pair, position_size, self.prices[pair].iloc[-1] if self.prices is not None else pd.Series()
                )
            
            # Apply risk limits
            if not self.check_risk_limits(pair, position_size, entry_price, entry_price):
                position_size = 0.0
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def calculate_atr(self, pair_data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range for a pair."""
        try:
            if len(pair_data) < self.atr_period:
                return pd.Series(index=pair_data.index, data=0.0)
            
            # For pairs, we need to calculate ATR on the spread
            # Use the first asset's OHLC data as proxy for spread volatility
            asset1 = pair_data.columns[0]
            
            # Get OHLC data for the first asset
            if hasattr(self, 'data_loader') and self.data_loader is not None:
                ohlc_data = self.data_loader.get_ohlc_data(asset1)
                if not ohlc_data.empty:
                    # Calculate True Range
                    high_low = ohlc_data['high'] - ohlc_data['low']
                    high_close = abs(ohlc_data['high'] - ohlc_data['close'].shift(1))
                    low_close = abs(ohlc_data['low'] - ohlc_data['close'].shift(1))
                    
                    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    
                    # Calculate ATR using simple moving average
                    atr = true_range.rolling(window=self.atr_period).mean()
                    
                    return atr
            
            # Fallback: use price volatility as proxy for ATR
            returns = pair_data.pct_change().dropna()
            volatility = returns.rolling(self.atr_period).std()
            atr = volatility * pair_data.iloc[:, 0]  # Scale by price level
            
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series(index=pair_data.index, data=0.0)
    
    def calculate_volatility_scaled_position(self, pair: Tuple[str, str], 
                                           base_position_size: float,
                                           current_prices: pd.Series) -> float:
        """Calculate position size scaled by volatility."""
        try:
            if not self.volatility_scaling:
                return base_position_size
            
            # Get pair data for volatility calculation
            pair_data = self.prices[pair]
            if len(pair_data) < 20:
                return base_position_size
            
            # Calculate rolling volatility (20-day)
            returns = pair_data.pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1]
            
            # Scale position size inversely with volatility
            if volatility > 0:
                # Higher volatility = smaller position size
                scaled_size = base_position_size / (volatility * 100)  # Scale factor
                # Cap the scaling to reasonable bounds
                scaled_size = max(base_position_size * 0.5, min(base_position_size * 2.0, scaled_size))
                return scaled_size
            
            return base_position_size
            
        except Exception as e:
            logger.error(f"Error calculating volatility scaled position: {e}")
            return base_position_size
    
    def check_risk_limits(self, pair: Tuple[str, str], position_size: float, 
                         entry_price: float, current_price: float) -> bool:
        """Check if position meets risk management criteria."""
        try:
            # Check pair exposure limit
            current_pair_exposure = self.pair_exposures.get(pair, 0.0)
            if current_pair_exposure + position_size > self.max_pair_exposure * self.initial_capital:
                logger.warning(f"Pair {pair} exposure limit exceeded: {current_pair_exposure + position_size:.2f}")
                return False
            
            # Check drawdown per trade
            price_change_pct = abs(current_price - entry_price) / entry_price
            if price_change_pct > self.max_drawdown_per_trade:
                logger.warning(f"Trade drawdown limit exceeded for {pair}: {price_change_pct:.2%}")
                return False
            
            # Check pair drawdown limit
            pair_drawdown = self.pair_drawdowns.get(pair, 0.0)
            if pair_drawdown > self.max_drawdown_per_pair:
                logger.warning(f"Pair {pair} drawdown limit exceeded: {pair_drawdown:.2%}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False
    
    def update_pair_metrics(self, pair: Tuple[str, str], trade_pnl: float, 
                          position_size: float) -> None:
        """Update pair-specific risk metrics."""
        try:
            # Update pair exposure
            self.pair_exposures[pair] = self.pair_exposures.get(pair, 0.0) + position_size
            
            # Update pair drawdown
            if trade_pnl < 0:
                current_drawdown = abs(trade_pnl) / self.initial_capital
                self.pair_drawdowns[pair] = self.pair_drawdowns.get(pair, 0.0) + current_drawdown
            else:
                # Reset drawdown on profitable trades
                self.pair_drawdowns[pair] = max(0.0, self.pair_drawdowns.get(pair, 0.0) - abs(trade_pnl) / self.initial_capital)
                
        except Exception as e:
            logger.error(f"Error updating pair metrics: {e}")
    
    def calculate_atr_stop_loss(self, pair: Tuple[str, str], entry_price: float, 
                              entry_date: datetime, direction: str) -> float:
        """Calculate ATR-based stop loss level."""
        try:
            # Get pair data up to entry date
            pair_data = self.prices[pair].loc[:entry_date]
            if len(pair_data) < self.atr_period:
                return entry_price * (0.95 if direction == 'long' else 1.05)  # Default 5% stop
            
            # Calculate ATR
            atr = self.calculate_atr(pair_data).iloc[-1]
            
            # Set stop loss based on direction and ATR
            if direction == 'long':
                stop_loss = entry_price - (atr * self.atr_multiplier)
            else:  # short
                stop_loss = entry_price + (atr * self.atr_multiplier)
            
            return stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating ATR stop loss: {e}")
            return entry_price * (0.95 if direction == 'long' else 1.05)
    
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
            slippage = (self.slippage_bps / 10000) * trade.size * 2  # 2 legs
            commission = (
                self.commission_bps / 10000
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
        self.equity_curve = pd.Series(index=dates, data=self.initial_capital)
        self.realized_pnl = pd.Series(index=dates, data=0.0)

        max_hold_days = getattr(self.config, 'max_hold_days', None)
        target_profit_pct = getattr(self.config, 'target_profit_pct', None)

        active_trades = {}
        error_count = 0  # Track errors to avoid excessive logging
        
        # Debug: Log signal information
        total_signals = 0
        for pair, signal_df in signals.items():
            long_signals = signal_df['entry_long'].sum() if 'entry_long' in signal_df.columns else 0
            short_signals = signal_df['entry_short'].sum() if 'entry_short' in signal_df.columns else 0
            total_signals += long_signals + short_signals
            logger.info(f"Pair {pair}: {long_signals} long signals, {short_signals} short signals")
        
        logger.info(f"Total signals across all pairs: {total_signals}")
        
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
                self._update_positions(date)

            except Exception as e:
                error_count += 1
                if error_count <= 5:  # Only log first 5 errors
                    logger.error(f"Error processing date {date}: {str(e)}")
                continue

        # Calculate daily returns
        self.daily_returns = self.equity_curve.pct_change()
        # Fill any NaN values in daily returns with 0
        self.daily_returns = self.daily_returns.fillna(0)
        
        # Close all remaining open trades at the end of the backtest
        final_date = dates[-1]
        open_trades_count = len(active_trades)
        if open_trades_count > 0:
            logger.info(f"Closing {open_trades_count} open trades at end of backtest")
            for pair, trade in list(active_trades.items()):
                self.close_trade(trade, final_date)
                del active_trades[pair]
        
        # Debug: Log final results
        logger.info(f"Backtest completed. Total trades: {len(self.trades)}")
        logger.info(f"Final equity: ${self.equity_curve.iloc[-1]:,.2f}")
        
        # Verify all trades are closed
        closed_trades = [t for t in self.trades if t.exit_date is not None]
        open_trades = [t for t in self.trades if t.exit_date is None]
        logger.info(f"Closed trades: {len(closed_trades)}, Open trades: {len(open_trades)}")
    
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
            pair=pair,
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
            base_equity = float(self.initial_capital)

        self.equity_curve.loc[date] = float(base_equity + daily_total_pnl)

        del self.positions[pair]
        self.current_exposure = sum(t.size for t in self.positions.values())
        self._enforce_leverage(date)
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
        losing_trades = [t for t in closed_trades if t.pnl and t.pnl < 0]
        avg_trade_duration = np.mean([(t.exit_date - t.entry_date).days for t in closed_trades]) if closed_trades else 0
        
        # Calculate total PnL from closed trades only
        total_pnl_closed = sum(t.pnl for t in closed_trades if t.pnl is not None)
        
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
            'losing_trades': len(losing_trades),
            'total_pnl_closed': total_pnl_closed,
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
            'stop_loss_k': t.stop_loss_k,
            'holding_period': t.holding_period
        } for t in self.trades])
        trades_df.to_csv(filename, index=False)
        logger.info(f"Saved trade history to {filename}")

    def _update_positions(self, date: datetime) -> None:
        """Update open positions and equity curve."""
        try:
            # Calculate P&L for all open positions (unrealized)
            unrealized_pnl = 0.0
            for pair, trade in self.positions.items():
                if pair[0] not in self.prices.columns or pair[1] not in self.prices.columns:
                    continue
                    
                current_price1 = self.prices.loc[date, pair[0]]
                current_price2 = self.prices.loc[date, pair[1]]
                
                if pd.isna(current_price1) or pd.isna(current_price2):
                    continue
                    
                pnl = self.calculate_trade_pnl(trade, current_price1, current_price2)
                unrealized_pnl += pnl

            # Update equity curve with realized P&L only
            if date in self.equity_curve.index:
                if date == self.equity_curve.index[0]:
                    # First day: start with initial capital plus realized P&L
                    self.equity_curve.loc[date] = self.initial_capital + self.realized_pnl.loc[date]
                else:
                    # Subsequent days: add to previous equity
                    prev_date = self.equity_curve.index[self.equity_curve.index.get_loc(date) - 1]
                    prev_equity = self.equity_curve.loc[prev_date]
                    self.equity_curve.loc[date] = prev_equity + self.realized_pnl.loc[date]

            # Store unrealized P&L separately (don't add to equity curve)
            # This prevents the equity curve from going negative due to unrealized losses

        except Exception as e:
            logger.error(f"Error updating positions for date {date}: {str(e)}")
            # Set default values if update fails
            if date in self.equity_curve.index:
                if date == self.equity_curve.index[0]:
                    self.equity_curve.loc[date] = self.initial_capital + self.realized_pnl.loc[date]
                else:
                    prev_date = self.equity_curve.index[self.equity_curve.index.get_loc(date) - 1]
                    self.equity_curve.loc[date] = self.equity_curve.loc[prev_date] + self.realized_pnl.loc[date]
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
                return entry_price - (atr_value * self.stop_loss_k)
            else:
                return entry_price + (atr_value * self.stop_loss_k)
        else:
            # Fixed multiplier stop-loss
            if position_type == 'long':
                return entry_price * (1 - self.stop_loss_k / 100)
            else:
                return entry_price * (1 + self.stop_loss_k / 100)

    def check_skew_filter(self, pair: Tuple[str, str], date: pd.Timestamp) -> bool:
        """Check if trade passes skew filter criteria."""
        try:
            cfg = {}
            if isinstance(self.config, dict):
                cfg = self.config.get('risk_management', {}).get('skew_filter', {})
            else:
                cfg = getattr(self.config, 'risk_management', {})
                cfg = getattr(cfg, 'skew_filter', {}) if cfg else {}

            if not cfg or not cfg.get('enabled', False):
                return True

            lookback = int(cfg.get('lookback', 60))
            min_ratio = cfg.get('min_down_up_ratio')
            min_sortino = cfg.get('min_sortino')

            if self.prices is None or date not in self.prices.index:
                return True

            end_idx = self.prices.index.get_loc(date)
            start_idx = max(0, end_idx - lookback + 1)
            window = self.prices.iloc[start_idx : end_idx + 1]
            if window.empty or len(window) < 2:
                return True

            spread = window[pair[0]] - window[pair[1]]
            returns = spread.pct_change().dropna()
            if returns.empty:
                return True

            downside = returns[returns < 0].abs().mean()
            upside = returns[returns > 0].mean()
            down_up_ratio = upside / downside if downside and downside > 0 else np.inf

            sortino = calculate_sortino_ratio(returns, 0.0)

            if min_ratio is not None and down_up_ratio < min_ratio:
                return False
            if min_sortino is not None and sortino < min_sortino:
                return False
            return True
        except Exception as e:
            logger.error(f"Error applying skew filter: {e}")
            return True

    def get_position_size(self, pair: Tuple[str, str], date: pd.Timestamp,
                         regime: int) -> float:
        """Calculate position size based on regime and risk settings."""
        # Use a reasonable base position size (1% of capital)
        base_size = self.initial_capital * 0.01

        if self.risk_config.get('regime_sizing', False):
            # Reduce size in unstable regimes
            if regime > 0:  # Assuming regime 0 is stable
                base_size *= 0.5

        # Determine current portfolio value
        if not self.equity_curve.empty and date in self.equity_curve.index:
            current_value = float(self.equity_curve.loc[:date].iloc[-1])
        elif not self.equity_curve.empty:
            current_value = float(self.equity_curve.iloc[-1])
        else:
            current_value = float(self.initial_capital)

        # Enforce per-trade cap
        max_position_value = current_value * self.max_pct_portfolio
        position_size = min(base_size, max_position_value)

        # Exposure limit across portfolio
        current_total = sum(t.size for t in self.positions.values())
        if current_total + position_size > current_value * self.max_total_exposure:
            allowed = current_value * self.max_total_exposure - current_total
            if allowed <= 0:
                logger.warning(
                    f"Total exposure limit reached; skipping trade for {pair}"
                )
                return 0.0
            logger.warning(
                f"Clipping trade size from {position_size} to {allowed} due to exposure limit"
            )
            position_size = allowed

        # Failsafe if trade size exceeds capital
        if position_size > current_value:
            logger.warning(
                f"Clipping trade size from {position_size} to {current_value}"
            )
            position_size = current_value

        return position_size

    def should_exit_trade(self, trade: Trade, date: pd.Timestamp) -> bool:
        """Determine if a trade should be exited based on the exit condition."""
        if trade.exit_date is not None:
            return True
        if (self.max_hold_days is not None and
                date >= trade.entry_date + pd.Timedelta(days=self.max_hold_days)):
            trade.exit_date = date
            trade.exit_price1 = self.prices.loc[date, trade.asset1]
            trade.exit_price2 = self.prices.loc[date, trade.asset2]
            trade.exit_reason = 'max_hold_days'
            return True

        if self.target_profit_pct is not None:
            price1 = self.prices.loc[date, trade.asset1]
            price2 = self.prices.loc[date, trade.asset2]
            current_pnl = self.calculate_trade_pnl(
                trade, current_price1=price1, current_price2=price2
            )
            if current_pnl / trade.size >= self.target_profit_pct:
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
        
        # Calculate holding period
        trade.holding_period = (trade.exit_date - trade.entry_date).days
        
        # Set a default exit reason if not already set by a specific condition
        if trade.exit_reason is None:
            trade.exit_reason = 'End of Backtest'
            
        trade.pnl = self.calculate_trade_pnl(trade)

        # Record realized P&L for the day
        if date in self.realized_pnl.index:
            self.realized_pnl.loc[date] += trade.pnl
        else:
            self.realized_pnl.loc[date] = trade.pnl

        # Update equity curve with realized P&L only
        if date in self.equity_curve.index:
            if date == self.equity_curve.index[0]:
                # First day: start with initial capital plus realized P&L
                self.equity_curve.loc[date] = self.initial_capital + self.realized_pnl.loc[date]
            else:
                # Subsequent days: add to previous equity
                prev_date = self.equity_curve.index[self.equity_curve.index.get_loc(date) - 1]
                prev_equity = self.equity_curve.loc[prev_date]
                self.equity_curve.loc[date] = prev_equity + self.realized_pnl.loc[date]

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
            pair=pair,
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
            stop_loss_k=self.stop_loss_k
        )
        
        # Add trade to active trades
        self.positions[pair] = trade
        self.trades.append(trade)
        
        # Update positions
        self._update_positions(date)
        
        # Update equity curve
        self._update_equity_curve(trade)
        
        # Update risk metrics
        self._update_risk_metrics(trade)

        # Enforce leverage limits
        self._enforce_leverage(date)

        # Log trade details
        self._log_trade_details(trade)
        
        return trade

    def _update_equity_curve(self, trade: Trade):
        """Update equity curve with new trade."""
        # Calculate initial position value
        position_value = trade.size * (trade.entry_price1 + trade.entry_price2)
        
        # Update equity curve
        self.equity_curve[trade.entry_date] = position_value

    def _update_risk_metrics(self, trade: Trade):
        """Update risk metrics with new trade."""
        self.current_exposure = sum(t.size for t in self.positions.values())

    def _enforce_leverage(self, date: pd.Timestamp) -> None:
        """Ensure portfolio leverage stays within limits."""
        if not self.equity_curve.empty and date in self.equity_curve.index:
            current_value = float(self.equity_curve.loc[:date].iloc[-1])
        elif not self.equity_curve.empty:
            current_value = float(self.equity_curve.iloc[-1])
        else:
            current_value = float(self.initial_capital)

        total_notional = sum(t.size for t in self.positions.values())
        exposure = total_notional / current_value if current_value > 0 else 0.0

        if exposure > self.max_leverage:
            scale = self.max_leverage / exposure
            logger.warning(
                f"Leverage {exposure:.2f} exceeds max {self.max_leverage}; scaling positions by {scale:.2f}"
            )
            for t in self.positions.values():
                t.size *= scale
            self.current_exposure = sum(t.size for t in self.positions.values())

        logger.info(
            f"Total exposure: {total_notional:.2f} | Leverage: {exposure:.2f}x"
        )

    def _log_trade_details(self, trade: Trade):
        """Log detailed trade information."""
        if trade.entry_date in self.equity_curve.index:
            current_value = float(self.equity_curve.loc[:trade.entry_date].iloc[-1])
        elif not self.equity_curve.empty:
            current_value = float(self.equity_curve.iloc[-1])
        else:
            current_value = float(self.initial_capital)

        total_notional = sum(t.size for t in self.positions.values())
        leverage = total_notional / current_value if current_value > 0 else 0.0

        logger.info(
            f"Trade details: {trade.__dict__} | Notional: {trade.size:.2f} | Leverage: {leverage:.2f}x"
        )
