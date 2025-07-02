"""
Enhanced Backtesting Module with Advanced Risk Management
Incorporates dynamic position sizing, regime filtering, and adaptive thresholds
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from scipy import stats
from sklearn.linear_model import LinearRegression
import yfinance as yf

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Enhanced position tracking with risk management"""
    pair: Tuple[str, str]
    entry_date: pd.Timestamp
    entry_price: float
    position_size: float
    direction: str  # 'long' or 'short'
    entry_zscore: float
    regime: str
    volatility_at_entry: float
    confidence: float

class EnhancedBacktest:
    """Advanced backtesting with dynamic risk management"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.positions = []
        self.equity_curve = []
        self.trades = []
        self.regime_data = None
        self.vix_data = None
        
    def load_regime_data(self, start_date: str, end_date: str):
        """Load VIX and regime data for filtering"""
        try:
            # Load VIX data for regime detection
            vix = yf.download('^VIX', start=start_date, end=end_date)['Close']
            self.vix_data = vix
            
            # Calculate regime indicators
            vix_ma = vix.rolling(20).mean()
            vix_std = vix.rolling(20).std()
            
            # Define regimes
            high_vol = vix > self.config['regime_detection']['vix_threshold_high']
            low_vol = vix < self.config['regime_detection']['vix_threshold_low']
            normal_vol = ~(high_vol | low_vol)
            
            self.regime_data = pd.DataFrame({
                'vix': vix,
                'vix_ma': vix_ma,
                'vix_std': vix_std,
                'high_vol': high_vol,
                'low_vol': low_vol,
                'normal_vol': normal_vol
            })
            
            logger.info(f"Loaded regime data: {len(self.regime_data)} days")
            
        except Exception as e:
            logger.warning(f"Could not load VIX data: {e}")
            self.regime_data = None
    
    def calculate_dynamic_position_size(self, pair_data: pd.DataFrame, 
                                      current_volatility: float,
                                      regime: str) -> float:
        """Calculate position size based on volatility and regime"""
        
        base_size = self.config['backtest']['max_position_size']
        
        # Volatility scaling
        if self.config['backtest']['volatility_targeting']:
            target_vol = self.config['backtest']['target_pair_volatility']
            vol_ratio = target_vol / current_volatility
            base_size *= min(vol_ratio, 2.0)  # Cap at 2x
        
        # Regime adjustments
        if regime == 'high_vol':
            base_size *= self.config['regime_detection']['high_volatility_multiplier']
        elif regime == 'low_vol':
            base_size *= self.config['regime_detection']['low_volatility_multiplier']
        
        return min(base_size, self.config['backtest']['max_position_size'])
    
    def calculate_adaptive_thresholds(self, zscore_series: pd.Series,
                                    current_regime: str) -> Tuple[float, float]:
        """Calculate adaptive entry/exit thresholds"""
        
        base_entry = self.config['signals']['entry_threshold']
        base_exit = self.config['signals']['exit_threshold']
        
        if not self.config['signals']['adaptive_thresholds']['enabled']:
            return base_entry, base_exit
        
        # Volatility scaling
        if self.config['signals']['adaptive_thresholds']['volatility_scaling']:
            recent_vol = zscore_series.rolling(60).std().iloc[-1]
            vol_multiplier = self.config['signals']['adaptive_thresholds']['volatility_multiplier']
            
            if recent_vol > 1.5:  # High volatility
                base_entry *= vol_multiplier
                base_exit *= vol_multiplier
            elif recent_vol < 0.8:  # Low volatility
                base_entry *= 0.8
                base_exit *= 0.8
        
        # Regime adjustments
        if current_regime == 'high_vol':
            high_vol_mult = self.config['signals']['adaptive_thresholds']['high_vol_multiplier']
            base_entry *= high_vol_mult
            base_exit *= high_vol_mult
        elif current_regime == 'low_vol':
            low_vol_mult = self.config['signals']['adaptive_thresholds']['low_vol_multiplier']
            base_entry *= low_vol_mult
            base_exit *= low_vol_mult
        
        return base_entry, base_exit
    
    def calculate_signal_confidence(self, zscore: float, 
                                  zscore_history: pd.Series) -> float:
        """Calculate signal confidence using Kalman filter approach"""
        
        if not self.config['signals']['adaptive_thresholds']['kalman_filter_enabled']:
            return 0.8  # Default confidence
        
        # Simple confidence based on z-score extremity and consistency
        zscore_abs = abs(zscore)
        recent_vol = zscore_history.rolling(20).std().iloc[-1]
        
        # Higher confidence for more extreme z-scores
        confidence = min(zscore_abs / 3.0, 1.0)
        
        # Adjust for volatility (lower confidence in high vol)
        if recent_vol > 1.5:
            confidence *= 0.8
        
        return max(confidence, 0.3)  # Minimum confidence
    
    def should_enter_trade(self, pair: Tuple[str, str], 
                          zscore: float,
                          current_regime: str,
                          confidence: float) -> bool:
        """Enhanced trade entry logic with regime filtering"""
        
        # Check regime restrictions
        if current_regime == 'high_vol' and not self.config['signals']['allow_all_regimes']:
            return False
        
        # Check signal confidence
        min_confidence = self.config['signals']['adaptive_thresholds']['min_signal_confidence']
        if confidence < min_confidence:
            return False
        
        # Check position limits
        current_positions = len([p for p in self.positions if p.pair == pair])
        max_positions = self.config['backtest']['max_concurrent_positions']
        
        if current_positions >= max_positions:
            return False
        
        return True
    
    def run_enhanced_backtest(self, pair_data: Dict[Tuple[str, str], pd.DataFrame],
                             selected_pairs: List[Tuple[str, str]]) -> Dict:
        """Run enhanced backtest with all advanced features"""
        
        results = {}
        portfolio_equity = pd.Series(0, index=pair_data[selected_pairs[0]].index)
        initial_capital = self.config['backtest']['initial_capital']
        current_capital = initial_capital
        
        for pair in selected_pairs:
            if pair not in pair_data:
                continue
                
            data = pair_data[pair]
            pair_results = self._backtest_single_pair(pair, data, current_capital)
            results[pair] = pair_results
            
            # Aggregate portfolio equity
            if 'equity_curve' in pair_results:
                portfolio_equity = portfolio_equity.add(pair_results['equity_curve'], fill_value=0)
        
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(portfolio_equity, initial_capital)
        
        return {
            'pair_results': results,
            'portfolio_equity': portfolio_equity,
            'portfolio_metrics': portfolio_metrics,
            'trades': self.trades
        }
    
    def _backtest_single_pair(self, pair: Tuple[str, str], 
                             data: pd.DataFrame,
                             initial_capital: float) -> Dict:
        """Backtest a single pair with enhanced features"""
        
        # Calculate spread and z-score
        spread = data.iloc[:, 0] - data.iloc[:, 1]
        zscore = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()
        
        # Initialize tracking
        equity_curve = pd.Series(initial_capital, index=data.index)
        current_capital = initial_capital
        positions = []
        
        for i in range(20, len(data)):
            current_date = data.index[i]
            current_zscore = zscore.iloc[i]
            
            # Get current regime
            current_regime = self._get_current_regime(current_date)
            
            # Calculate adaptive thresholds
            entry_threshold, exit_threshold = self._calculate_adaptive_thresholds(
                zscore.iloc[:i+1], current_regime
            )
            
            # Calculate signal confidence
            confidence = self._calculate_signal_confidence(
                current_zscore, zscore.iloc[:i+1]
            )
            
            # Check for entry signals
            if abs(current_zscore) > entry_threshold:
                if self._should_enter_trade(pair, current_zscore, current_regime, confidence):
                    # Calculate position size
                    current_vol = spread.rolling(60).std().iloc[i]
                    position_size = self._calculate_dynamic_position_size(
                        data.iloc[:i+1], current_vol, current_regime
                    )
                    
                    # Create position
                    direction = 'long' if current_zscore < -entry_threshold else 'short'
                    position = Position(
                        pair=pair,
                        entry_date=current_date,
                        entry_price=data.iloc[i, 0] if direction == 'long' else data.iloc[i, 1],
                        position_size=position_size,
                        direction=direction,
                        entry_zscore=current_zscore,
                        regime=current_regime,
                        volatility_at_entry=current_vol,
                        confidence=confidence
                    )
                    
                    positions.append(position)
            
            # Check for exit signals
            for pos in positions[:]:  # Copy list to avoid modification during iteration
                if abs(current_zscore) < exit_threshold:
                    # Close position
                    exit_price = data.iloc[i, 0] if pos.direction == 'long' else data.iloc[i, 1]
                    pnl = self._calculate_pnl(pos, exit_price, current_date)
                    
                    # Record trade
                    self.trades.append({
                        'pair': pair,
                        'entry_date': pos.entry_date,
                        'exit_date': current_date,
                        'direction': pos.direction,
                        'entry_price': pos.entry_price,
                        'exit_price': exit_price,
                        'position_size': pos.position_size,
                        'pnl': pnl,
                        'entry_zscore': pos.entry_zscore,
                        'exit_zscore': current_zscore,
                        'regime': pos.regime,
                        'confidence': pos.confidence
                    })
                    
                    # Update capital
                    current_capital += pnl
                    positions.remove(pos)
            
            # Update equity curve
            equity_curve.iloc[i] = current_capital
        
        return {
            'equity_curve': equity_curve,
            'trades': [t for t in self.trades if t['pair'] == pair],
            'final_capital': current_capital,
            'total_return': (current_capital - initial_capital) / initial_capital
        }
    
    def _get_current_regime(self, date: pd.Timestamp) -> str:
        """Get current market regime"""
        if self.regime_data is None or date not in self.regime_data.index:
            return 'normal_vol'
        
        row = self.regime_data.loc[date]
        if row['high_vol']:
            return 'high_vol'
        elif row['low_vol']:
            return 'low_vol'
        else:
            return 'normal_vol'
    
    def _calculate_pnl(self, position: Position, exit_price: float, exit_date: pd.Timestamp) -> float:
        """Calculate PnL for a position"""
        if position.direction == 'long':
            price_change = (exit_price - position.entry_price) / position.entry_price
        else:
            price_change = (position.entry_price - exit_price) / position.entry_price
        
        # Apply slippage and commission
        slippage = self.config['backtest']['slippage_bps'] / 10000
        commission = self.config['backtest']['commission_bps'] / 10000
        
        net_return = price_change - slippage - commission
        return position.position_size * net_return
    
    def _calculate_portfolio_metrics(self, equity_curve: pd.Series, initial_capital: float) -> Dict:
        """Calculate comprehensive portfolio metrics"""
        
        returns = equity_curve.pct_change().fillna(0)
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
        annualized_return = total_return * (252 / len(returns))
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Trade metrics
        if self.trades:
            winning_trades = [t for t in self.trades if t['pnl'] > 0]
            losing_trades = [t for t in self.trades if t['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(self.trades)
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        } 