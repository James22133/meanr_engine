#!/usr/bin/env python3
"""
Enhanced Backtest for 2023-2025 Period - Fixed Version (Excluding EMB-SPY)
Implements strategic recommendations with comprehensive risk metrics.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import json
from datetime import datetime
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

def calculate_zscore(spread, lookback=20):
    """Calculate z-score of spread."""
    mean = spread.rolling(lookback).mean()
    std = spread.rolling(lookback).std()
    return (spread - mean) / std

def detect_market_regime(returns, n_regimes=3, lookback=60):
    """Detect market regimes using Gaussian Mixture Model."""
    try:
        # Calculate rolling volatility and returns
        rolling_vol = returns.rolling(lookback).std()
        rolling_ret = returns.rolling(lookback).mean()
        
        # Create features for regime detection
        features = pd.DataFrame({
            'volatility': rolling_vol,
            'returns': rolling_ret,
            'vol_ma_ratio': rolling_vol / rolling_vol.rolling(lookback).mean()
        }).dropna()
        
        if len(features) < n_regimes * 10:  # Need sufficient data
            return pd.Series(0, index=returns.index)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Fit GMM
        gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        regimes = gmm.fit_predict(features_scaled)
        
        # Create regime series aligned with original index
        regime_series = pd.Series(0, index=returns.index)
        regime_series.loc[features.index] = regimes
        
        return regime_series.ffill().fillna(0)
    
    except Exception as e:
        print(f"Regime detection error: {e}")
        return pd.Series(0, index=returns.index)

def calculate_dynamic_position_size(price1, price2, zscore, regime, base_size=1000):
    """Calculate dynamic position size based on volatility and regime."""
    try:
        # Calculate historical volatility
        returns1 = price1.pct_change().dropna()
        returns2 = price2.pct_change().dropna()
        
        vol1 = returns1.rolling(20).std().iloc[-1] if len(returns1) >= 20 else 0.02
        vol2 = returns2.rolling(20).std().iloc[-1] if len(returns2) >= 20 else 0.02
        
        # Volatility normalization factor
        avg_vol = (vol1 + vol2) / 2
        vol_factor = 0.02 / avg_vol if avg_vol > 0 else 1.0  # Normalize to 2% volatility
        
        # Z-score amplitude factor
        zscore_amplitude = abs(zscore)
        amplitude_factor = min(2.0, max(0.5, zscore_amplitude / 2.0))
        
        # Regime factor (higher size in mean-reverting regimes)
        regime_factor = 1.5 if regime == 0 else 1.0  # Assume regime 0 is mean-reverting
        
        # Combine factors
        size_multiplier = vol_factor * amplitude_factor * regime_factor
        
        return float(base_size * size_multiplier)
    except:
        return float(base_size)

def calculate_risk_metrics(returns, risk_free_rate=0.02):
    """Calculate comprehensive risk metrics with proper error handling."""
    # Clean the returns series
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(returns) == 0:
        return {
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'total_return_pct': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0,
            'downside_deviation': 0.0,
            'calmar_ratio': 0.0,
            'win_rate': 0.0
        }
    
    try:
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        total_return_pct = total_return * 100
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
        
        # Sharpe Ratio
        excess_returns = returns - risk_free_rate/252
        sharpe_ratio = (excess_returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (excess_returns.mean() / downside_deviation) * np.sqrt(252) if downside_deviation > 0 else 0.0
        else:
            downside_deviation = 0.0
            sortino_ratio = 0.0
        
        # Maximum Drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0
        
        # Calmar Ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        # Win Rate
        win_rate = (returns > 0).mean() if len(returns) > 0 else 0.0
        
        return {
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'total_return_pct': float(total_return_pct),
            'max_drawdown': float(max_drawdown),
            'volatility': float(volatility),
            'downside_deviation': float(downside_deviation),
            'calmar_ratio': float(calmar_ratio),
            'win_rate': float(win_rate)
        }
    except Exception as e:
        print(f"Error calculating risk metrics: {e}")
        return {
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'total_return_pct': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0,
            'downside_deviation': 0.0,
            'calmar_ratio': 0.0,
            'win_rate': 0.0
        }

def generate_enhanced_signals(price1, price2, spy_returns, lookback=20, entry_threshold=1.4, exit_threshold=0.2):
    """Generate enhanced trading signals with regime awareness."""
    spread = price1 - price2
    zscore = calculate_zscore(spread, lookback)
    
    # Detect market regime
    regime = detect_market_regime(spy_returns)
    
    # Initialize signals
    position = pd.Series(0, index=price1.index)
    entry_signal = pd.Series(0, index=price1.index)
    exit_signal = pd.Series(0, index=price1.index)
    position_size = pd.Series(0.0, index=price1.index)  # Use float dtype
    
    current_position = 0
    current_size = 0.0
    
    for i in range(lookback, len(zscore)):
        z = zscore.iloc[i]
        current_regime = regime.iloc[i] if i < len(regime) else 0
        
        # Adjust thresholds based on regime
        regime_entry_threshold = entry_threshold * (1.2 if current_regime == 0 else 1.0)
        regime_exit_threshold = exit_threshold * (0.8 if current_regime == 0 else 1.0)
        
        if current_position == 0:  # No position
            if z > regime_entry_threshold:
                current_position = -1  # Short spread
                current_size = calculate_dynamic_position_size(
                    price1.iloc[:i+1], price2.iloc[:i+1], z, current_regime
                )
                entry_signal.iloc[i] = -1
                position_size.iloc[i] = current_size
            elif z < -regime_entry_threshold:
                current_position = 1   # Long spread
                current_size = calculate_dynamic_position_size(
                    price1.iloc[:i+1], price2.iloc[:i+1], z, current_regime
                )
                entry_signal.iloc[i] = 1
                position_size.iloc[i] = current_size
        
        elif current_position != 0:  # Have position
            # Check for exit conditions
            exit_condition = abs(z) < regime_exit_threshold
            
            # Add stop-loss (2x entry threshold)
            stop_loss = abs(z) > entry_threshold * 2
            
            if exit_condition or stop_loss:
                exit_signal.iloc[i] = current_position
                current_position = 0
                current_size = 0.0
        
        position.iloc[i] = current_position
    
    return position, entry_signal, exit_signal, position_size, regime

def calculate_enhanced_trade_pnl(entry_price1, entry_price2, exit_price1, exit_price2, position, size):
    """Calculate PnL for a trade with position sizing."""
    if position == 1:  # Long spread
        pnl = size * ((exit_price1 - entry_price1) - (exit_price2 - entry_price2))
    else:  # Short spread
        pnl = size * ((entry_price1 - exit_price1) - (entry_price2 - exit_price2))
    return pnl

def run_enhanced_backtest():
    """Run enhanced backtest with strategic improvements - EXCLUDING EMB-SPY."""
    
    print("=" * 70)
    print("ENHANCED BACKTEST: 2023-2025 PERIOD (EXCLUDING EMB-SPY)")
    print("=" * 70)
    
    # Selected pairs from the analysis - EXCLUDING EMB-SPY
    selected_pairs = [
        ('EPD', 'ET'),
        ('BNO', 'USO'),
        ('QQQ', 'SPY'),
        ('EOG', 'XLE'),
        ('EFA', 'SLV')
    ]
    
    print(f"⚠️  EXCLUDED: EMB-SPY (due to catastrophic losses)")
    print(f"📊 TRADING: {len(selected_pairs)} pairs instead of 6")
    
    # Download data for all symbols
    all_symbols = set()
    for pair in selected_pairs:
        all_symbols.add(pair[0])
        all_symbols.add(pair[1])
    
    print(f"Downloading data for {len(all_symbols)} symbols...")
    raw_data = yf.download(list(all_symbols), start='2023-01-01', end='2025-07-01')
    if isinstance(raw_data.columns, pd.MultiIndex):
        data = raw_data['Close']
    else:
        data = raw_data
    
    # Get SPY returns for regime detection
    spy_returns = data['SPY'].pct_change().dropna()
    
    # Run enhanced backtests for each pair
    all_trades = []
    pair_results = {}
    pair_equity_curves = {}
    
    for i, (symbol1, symbol2) in enumerate(selected_pairs, 1):
        print(f"\nBacktesting pair {i}/{len(selected_pairs)}: {symbol1}-{symbol2}")
        
        try:
            price1 = data[symbol1].dropna()
            price2 = data[symbol2].dropna()
            
            # Align the series
            common_index = price1.index.intersection(price2.index)
            price1 = price1.loc[common_index]
            price2 = price2.loc[common_index]
            
            if len(price1) < 100:
                print(f"  ⚠️  Insufficient data for {symbol1}-{symbol2}")
                continue
            
            # Generate enhanced signals
            position, entry_signal, exit_signal, position_size, regime = generate_enhanced_signals(
                price1, price2, spy_returns, 
                lookback=20, 
                entry_threshold=1.4, 
                exit_threshold=0.2
            )
            
            # Extract trades with position sizing
            trades = []
            current_position = 0
            entry_price1 = entry_price2 = entry_date = None
            current_size = 0.0
            
            # Track equity curve
            equity_curve = pd.Series(1.0, index=price1.index)
            cumulative_pnl = 0.0
            
            for date in price1.index:
                if entry_signal.loc[date] != 0:  # Entry signal
                    if current_position == 0:  # No current position
                        current_position = entry_signal.loc[date]
                        entry_price1 = price1.loc[date]
                        entry_price2 = price2.loc[date]
                        entry_date = date
                        current_size = position_size.loc[date]
                
                elif exit_signal.loc[date] != 0:  # Exit signal
                    if current_position != 0:  # Have position to close
                        exit_price1 = price1.loc[date]
                        exit_price2 = price2.loc[date]
                        exit_date = date
                        
                        # Calculate PnL with position sizing
                        pnl = calculate_enhanced_trade_pnl(
                            entry_price1, entry_price2, 
                            exit_price1, exit_price2, 
                            current_position, current_size
                        )
                        
                        cumulative_pnl += pnl
                        
                        # Determine direction
                        direction = 'long' if current_position == 1 else 'short'
                        
                        # Calculate holding period
                        holding_period = (exit_date - entry_date).days
                        
                        trade = {
                            'pair': f"{symbol1}-{symbol2}",
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'direction': direction,
                            'entry_price1': entry_price1,
                            'entry_price2': entry_price2,
                            'exit_price1': exit_price1,
                            'exit_price2': exit_price2,
                            'pnl': pnl,
                            'position_size': current_size,
                            'holding_period': holding_period,
                            'regime': regime.loc[entry_date] if entry_date in regime.index else 0
                        }
                        
                        trades.append(trade)
                        current_position = 0
                        current_size = 0.0
                
                # Update equity curve
                if date in equity_curve.index:
                    equity_curve.loc[date] = 1 + cumulative_pnl / 100000  # Normalize to $100K initial
            
            # Add any open position at the end
            if current_position != 0:
                exit_price1 = price1.iloc[-1]
                exit_price2 = price2.iloc[-1]
                exit_date = price1.index[-1]
                
                pnl = calculate_enhanced_trade_pnl(
                    entry_price1, entry_price2, 
                    exit_price1, exit_price2, 
                    current_position, current_size
                )
                
                direction = 'long' if current_position == 1 else 'short'
                holding_period = (exit_date - entry_date).days
                
                trade = {
                    'pair': f"{symbol1}-{symbol2}",
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'direction': direction,
                    'entry_price1': entry_price1,
                    'entry_price2': entry_price2,
                    'exit_price1': exit_price1,
                    'exit_price2': exit_price2,
                    'pnl': pnl,
                    'position_size': current_size,
                    'holding_period': holding_period,
                    'regime': regime.loc[entry_date] if entry_date in regime.index else 0
                }
                
                trades.append(trade)
            
            all_trades.extend(trades)
            pair_results[f"{symbol1}-{symbol2}"] = {
                'trades': len(trades),
                'pnl': sum(t['pnl'] for t in trades),
                'winning_trades': sum(1 for t in trades if t['pnl'] > 0),
                'avg_position_size': np.mean([t['position_size'] for t in trades]) if trades else 0
            }
            pair_equity_curves[f"{symbol1}-{symbol2}"] = equity_curve
            
            print(f"  ✅ {len(trades)} trades, PnL: ${sum(t['pnl'] for t in trades):,.2f}")
            print(f"     Avg position size: ${np.mean([t['position_size'] for t in trades]):,.0f}")
                
        except Exception as e:
            print(f"  ❌ Error backtesting {symbol1}-{symbol2}: {e}")
    
    # Calculate overall statistics
    if all_trades:
        total_trades = len(all_trades)
        winning_trades = sum(1 for t in all_trades if t['pnl'] > 0)
        total_pnl = sum(t['pnl'] for t in all_trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate daily returns for portfolio (improved method)
        # Create a proper time series of daily PnL
        trade_dates = []
        trade_pnls = []
        
        for trade in all_trades:
            trade_dates.append(trade['exit_date'])
            trade_pnls.append(trade['pnl'])
        
        # Create daily PnL series
        daily_pnl = pd.Series(trade_pnls, index=trade_dates)
        daily_pnl = daily_pnl.groupby(daily_pnl.index).sum()  # Sum multiple trades on same day
        
        # Create full date range and fill missing days with 0
        full_date_range = pd.date_range(start='2023-01-01', end='2025-07-01', freq='D')
        portfolio_pnl = pd.Series(0.0, index=full_date_range)
        portfolio_pnl.loc[daily_pnl.index] = daily_pnl
        
        # Calculate cumulative PnL and then returns
        cumulative_pnl = portfolio_pnl.cumsum()
        initial_capital = 1000000  # $1M initial capital
        portfolio_equity = initial_capital + cumulative_pnl
        
        # Calculate daily returns
        portfolio_returns = portfolio_equity.pct_change().dropna()
        
        # Calculate comprehensive risk metrics
        risk_metrics = calculate_risk_metrics(portfolio_returns)
        
        print(f"\n" + "=" * 70)
        print("ENHANCED RESULTS (2023-2025) - EXCLUDING EMB-SPY")
        print("=" * 70)
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Total PnL: ${total_pnl:,.2f}")
        print(f"Average PnL per Trade: ${total_pnl/total_trades:,.2f}")
        print(f"Average Position Size: ${np.mean([t['position_size'] for t in all_trades]):,.0f}")
        
        print(f"\nRISK METRICS:")
        print(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio: {risk_metrics['sortino_ratio']:.3f}")
        print(f"Total Return: {risk_metrics['total_return_pct']:.2f}%")
        print(f"Max Drawdown: {risk_metrics['max_drawdown']:.2%}")
        print(f"Volatility: {risk_metrics['volatility']:.2%}")
        print(f"Calmar Ratio: {risk_metrics['calmar_ratio']:.3f}")
        
        # Pair breakdown
        print(f"\nPAIR BREAKDOWN:")
        print("-" * 60)
        for pair, stats in pair_results.items():
            pair_win_rate = stats['winning_trades'] / stats['trades'] if stats['trades'] > 0 else 0
            print(f"{pair:<15} {stats['trades']:>3} trades, {pair_win_rate:.1%} win rate, ${stats['pnl']:>10,.2f}")
            print(f"{'':15} Avg size: ${stats['avg_position_size']:>10,.0f}")
        
        # Save enhanced results
        enhanced_results = {
            'timestamp': datetime.now().isoformat(),
            'period': '2023-2025',
            'strategy': 'Enhanced with Dynamic Position Sizing & Regime Awareness (EMB-SPY Excluded)',
            'excluded_pairs': ['EMB-SPY'],
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': total_pnl / total_trades if total_trades > 0 else 0,
            'avg_position_size': np.mean([t['position_size'] for t in all_trades]),
            'risk_metrics': risk_metrics,
            'pair_results': pair_results,
            'selected_pairs': [f"{p[0]}-{p[1]}" for p in selected_pairs],
            'trades': all_trades
        }
        
        with open('enhanced_backtest_no_emb_results.json', 'w') as f:
            json.dump(enhanced_results, f, indent=4, default=str)
        
        print(f"\nEnhanced results saved to 'enhanced_backtest_no_emb_results.json'")
        
        return enhanced_results
    else:
        print("❌ No trades were generated!")
        return None

if __name__ == "__main__":
    run_enhanced_backtest() 