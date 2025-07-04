"""
Regime filtering module for enhanced mean reversion strategies.
Provides VIX-based, trend-based, and rolling Sharpe ratio filters.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from dataclasses import dataclass
import yfinance as yf

@dataclass
class RegimeFilterConfig:
    """Configuration for regime filtering."""
    use_vix_filter: bool = True
    vix_threshold_high: float = 25.0
    vix_threshold_low: float = 15.0
    vix_lookback: int = 5
    
    use_trend_filter: bool = True
    trend_window: int = 60
    trend_slope_threshold: float = 0.6
    trend_ma_window: int = 20
    
    use_rolling_sharpe_filter: bool = True
    rolling_sharpe_window: int = 60
    rolling_sharpe_min: float = 0.2
    rolling_sharpe_lookback: int = 252
    
    use_market_regime_filter: bool = True
    market_regime_window: int = 20
    min_regime_stability: float = 0.7

class RegimeFilter:
    """Regime filtering for mean reversion strategies."""
    
    def __init__(self, config: RegimeFilterConfig):
        """Initialize regime filter with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.vix_data = None
        self._load_vix_data()
    
    def _load_vix_data(self):
        """Load VIX data for volatility regime detection."""
        try:
            if self.config.use_vix_filter:
                self.logger.info("Loading VIX data for regime filtering...")
                vix = yf.download('^VIX', start='2018-01-01', end='2024-12-31', progress=False)
                if not vix.empty:
                    self.vix_data = vix['Close']
                    self.logger.info(f"Loaded VIX data: {len(self.vix_data)} observations")
                else:
                    self.logger.warning("Failed to load VIX data")
        except Exception as e:
            self.logger.error(f"Error loading VIX data: {e}")
            self.vix_data = None
    
    def is_trending(self, spread: pd.Series, window: int = None, threshold: float = None) -> bool:
        """
        Check if spread is trending using linear regression slope.
        
        Args:
            spread: Price spread series
            window: Rolling window for trend calculation
            threshold: Slope threshold for trend detection
            
        Returns:
            bool: True if trending, False if mean-reverting
        """
        if window is None:
            window = self.config.trend_window
        if threshold is None:
            threshold = self.config.trend_slope_threshold
            
        try:
            if len(spread) < window:
                return False
            
            # Calculate moving average
            ma = spread.rolling(window=self.config.trend_ma_window).mean()
            if len(ma.dropna()) < window:
                return False
            
            # Get recent window of data
            recent_ma = ma.tail(window).dropna()
            if len(recent_ma) < window:
                return False
            
            # Calculate linear regression slope
            x = np.arange(len(recent_ma))
            slope, _ = np.polyfit(x, recent_ma.values, 1)
            
            # Normalize slope by spread standard deviation
            spread_std = spread.rolling(window).std().iloc[-1]
            if spread_std > 0:
                normalized_slope = abs(slope) / spread_std
            else:
                normalized_slope = abs(slope)
            
            return normalized_slope > threshold
            
        except Exception as e:
            self.logger.error(f"Error in trend detection: {e}")
            return False
    
    def is_unfavorable_vol_regime(self, date: pd.Timestamp, high_threshold: float = None) -> bool:
        """
        Check if current volatility regime is unfavorable for mean reversion.
        
        Args:
            date: Current date
            high_threshold: VIX threshold for high volatility
            
        Returns:
            bool: True if unfavorable (high volatility), False if favorable
        """
        if not self.config.use_vix_filter or self.vix_data is None:
            return False
            
        if high_threshold is None:
            high_threshold = self.config.vix_threshold_high
            
        try:
            # Get VIX value for the date
            vix_value = self.vix_data.get(date, np.nan)
            if pd.isna(vix_value):
                # Try to get nearest available VIX value
                nearest_vix = self.vix_data.reindex([date], method='nearest')
                if not nearest_vix.empty:
                    vix_value = nearest_vix.iloc[0]
                else:
                    return False
            
            if pd.isna(vix_value):
                return False
            
            # Check if VIX is above threshold - use .item() to get scalar
            if hasattr(vix_value, 'item'):
                vix_scalar = vix_value.item()
            else:
                vix_scalar = float(vix_value)
            
            return vix_scalar > high_threshold
            
        except Exception as e:
            self.logger.error(f"Error in volatility regime check: {e}")
            return False
    
    def calculate_rolling_sharpe(self, spread: pd.Series, window: int = None) -> float:
        """
        Calculate rolling Sharpe ratio for spread mean reversion.
        
        Args:
            spread: Price spread series
            window: Rolling window for calculation
            
        Returns:
            float: Rolling Sharpe ratio
        """
        if window is None:
            window = self.config.rolling_sharpe_window
            
        try:
            if len(spread) < window:
                return 0.0
            
            # Calculate spread returns
            returns = spread.pct_change().dropna()
            if len(returns) < window:
                return 0.0
            
            # Calculate rolling Sharpe ratio
            rolling_mean = returns.rolling(window).mean()
            rolling_std = returns.rolling(window).std()
            
            # Avoid division by zero
            rolling_std = rolling_std.replace(0, np.nan)
            sharpe = rolling_mean / rolling_std * np.sqrt(252)
            
            # Get most recent Sharpe ratio
            current_sharpe = sharpe.iloc[-1]
            return current_sharpe if not pd.isna(current_sharpe) else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating rolling Sharpe: {e}")
            return 0.0
    
    def is_sharpe_favorable(self, spread: pd.Series, min_sharpe: float = None) -> bool:
        """
        Check if rolling Sharpe ratio is favorable for trading.
        
        Args:
            spread: Price spread series
            min_sharpe: Minimum Sharpe ratio threshold
            
        Returns:
            bool: True if Sharpe ratio is favorable
        """
        if not self.config.use_rolling_sharpe_filter:
            return True
            
        if min_sharpe is None:
            min_sharpe = self.config.rolling_sharpe_min
            
        try:
            sharpe = self.calculate_rolling_sharpe(spread)
            return sharpe >= min_sharpe
            
        except Exception as e:
            self.logger.error(f"Error in Sharpe ratio check: {e}")
            return True  # Default to allowing trades if error
    
    def get_regime_multiplier(self, spread: pd.Series, date: pd.Timestamp) -> float:
        """
        Get position size multiplier based on current regime.
        
        Args:
            spread: Price spread series
            date: Current date
            
        Returns:
            float: Position size multiplier (0.0 to 2.0)
        """
        try:
            multiplier = 1.0  # Base multiplier
            
            # Check volatility regime
            if self.is_unfavorable_vol_regime(date):
                multiplier *= 0.5  # Reduce position size in high volatility
            
            # Check trend regime
            if self.is_trending(spread):
                multiplier *= 0.3  # Reduce position size in trending markets
            
            # Check Sharpe ratio regime
            if not self.is_sharpe_favorable(spread):
                multiplier *= 0.7  # Reduce position size if Sharpe is poor
            
            return max(0.1, min(2.0, multiplier))  # Clamp between 0.1 and 2.0
            
        except Exception as e:
            self.logger.error(f"Error calculating regime multiplier: {e}")
            return 1.0
    
    def should_trade(self, spread: pd.Series, date: pd.Timestamp) -> Tuple[bool, Dict]:
        """
        Determine if we should trade based on all regime filters.
        
        Args:
            spread: Price spread series
            date: Current date
            
        Returns:
            Tuple[bool, Dict]: (should_trade, regime_info)
        """
        regime_info = {
            'volatility_regime': 'unknown',
            'trend_regime': 'unknown',
            'sharpe_regime': 'unknown',
            'should_trade': True,
            'multiplier': 1.0
        }
        
        try:
            # Check volatility regime
            if self.config.use_vix_filter:
                if self.is_unfavorable_vol_regime(date):
                    regime_info['volatility_regime'] = 'high'
                    regime_info['should_trade'] = False
                else:
                    regime_info['volatility_regime'] = 'low'
            
            # Check trend regime
            if self.config.use_trend_filter and regime_info['should_trade']:
                if self.is_trending(spread):
                    regime_info['trend_regime'] = 'trending'
                    regime_info['should_trade'] = False
                else:
                    regime_info['trend_regime'] = 'mean_reverting'
            
            # Check Sharpe ratio regime
            if self.config.use_rolling_sharpe_filter and regime_info['should_trade']:
                if not self.is_sharpe_favorable(spread):
                    regime_info['sharpe_regime'] = 'poor'
                    regime_info['should_trade'] = False
                else:
                    regime_info['sharpe_regime'] = 'good'
            
            # Calculate position size multiplier
            regime_info['multiplier'] = self.get_regime_multiplier(spread, date)
            
            return regime_info['should_trade'], regime_info
            
        except Exception as e:
            self.logger.error(f"Error in regime filtering: {e}")
            return True, regime_info
    
    def get_regime_summary(self, spread: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Get regime summary for a time series.
        
        Args:
            spread: Price spread series
            dates: Date index
            
        Returns:
            pd.DataFrame: Regime summary with daily regime information
        """
        try:
            regime_data = []
            
            for date in dates:
                if date in spread.index:
                    should_trade, regime_info = self.should_trade(spread, date)
                    regime_data.append({
                        'date': date,
                        'should_trade': should_trade,
                        'volatility_regime': regime_info['volatility_regime'],
                        'trend_regime': regime_info['trend_regime'],
                        'sharpe_regime': regime_info['sharpe_regime'],
                        'multiplier': regime_info['multiplier']
                    })
            
            return pd.DataFrame(regime_data)
            
        except Exception as e:
            self.logger.error(f"Error generating regime summary: {e}")
            return pd.DataFrame() 