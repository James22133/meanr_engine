#!/usr/bin/env python3
"""
Simple Backtest for 2023-2025 Period
Manually calculates trades and performance metrics.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import json
from datetime import datetime

def calculate_zscore(spread, lookback=20):
    """Calculate z-score of spread."""
    mean = spread.rolling(lookback).mean()
    std = spread.rolling(lookback).std()
    return (spread - mean) / std

def generate_signals(price1, price2, lookback=20, entry_threshold=1.4, exit_threshold=0.2):
    """Generate trading signals."""
    spread = price1 - price2
    zscore = calculate_zscore(spread, lookback)
    
    # Initialize signals
    position = pd.Series(0, index=price1.index)
    entry_signal = pd.Series(0, index=price1.index)
    exit_signal = pd.Series(0, index=price1.index)
    
    current_position = 0
    
    for i in range(lookback, len(zscore)):
        z = zscore.iloc[i]
        
        if current_position == 0:  # No position
            if z > entry_threshold:
                current_position = -1  # Short spread (short price1, long price2)
                entry_signal.iloc[i] = -1
            elif z < -entry_threshold:
                current_position = 1   # Long spread (long price1, short price2)
                entry_signal.iloc[i] = 1
        
        elif current_position != 0:  # Have position
            if abs(z) < exit_threshold:
                exit_signal.iloc[i] = current_position
                current_position = 0
        
        position.iloc[i] = current_position
    
    return position, entry_signal, exit_signal

def calculate_trade_pnl(entry_price1, entry_price2, exit_price1, exit_price2, position, size=1000):
    """Calculate PnL for a trade."""
    if position == 1:  # Long spread
        pnl = size * ((exit_price1 - entry_price1) - (exit_price2 - entry_price2))
    else:  # Short spread
        pnl = size * ((entry_price1 - exit_price1) - (entry_price2 - exit_price2))
    return pnl

def run_simple_backtest():
    """Run simple backtest for 2023-2025 period."""
    
    print("=" * 60)
    print("SIMPLE BACKTEST: 2023-2025 PERIOD")
    print("=" * 60)
    
    # Selected pairs from the analysis
    selected_pairs = [
        ('EPD', 'ET'),
        ('EMB', 'SPY'), 
        ('BNO', 'USO'),
        ('QQQ', 'SPY'),
        ('EOG', 'XLE'),
        ('EFA', 'SLV')
    ]
    
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
    
    # Run backtests for each pair
    all_trades = []
    pair_results = {}
    
    for i, (symbol1, symbol2) in enumerate(selected_pairs, 1):
        print(f"\nBacktesting pair {i}/6: {symbol1}-{symbol2}")
        
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
            
            # Generate signals
            position, entry_signal, exit_signal = generate_signals(
                price1, price2, 
                lookback=20, 
                entry_threshold=1.4, 
                exit_threshold=0.2
            )
            
            # Extract trades
            trades = []
            current_position = 0
            entry_price1 = entry_price2 = entry_date = None
            
            for date in price1.index:
                if entry_signal.loc[date] != 0:  # Entry signal
                    if current_position == 0:  # No current position
                        current_position = entry_signal.loc[date]
                        entry_price1 = price1.loc[date]
                        entry_price2 = price2.loc[date]
                        entry_date = date
                
                elif exit_signal.loc[date] != 0:  # Exit signal
                    if current_position != 0:  # Have position to close
                        exit_price1 = price1.loc[date]
                        exit_price2 = price2.loc[date]
                        exit_date = date
                        
                        # Calculate PnL
                        pnl = calculate_trade_pnl(
                            entry_price1, entry_price2, 
                            exit_price1, exit_price2, 
                            current_position
                        )
                        
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
                            'holding_period': holding_period
                        }
                        
                        trades.append(trade)
                        current_position = 0
            
            # Add any open position at the end
            if current_position != 0:
                exit_price1 = price1.iloc[-1]
                exit_price2 = price2.iloc[-1]
                exit_date = price1.index[-1]
                
                pnl = calculate_trade_pnl(
                    entry_price1, entry_price2, 
                    exit_price1, exit_price2, 
                    current_position
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
                    'holding_period': holding_period
                }
                
                trades.append(trade)
            
            all_trades.extend(trades)
            pair_results[f"{symbol1}-{symbol2}"] = {
                'trades': len(trades),
                'pnl': sum(t['pnl'] for t in trades),
                'winning_trades': sum(1 for t in trades if t['pnl'] > 0)
            }
            
            print(f"  ✅ {len(trades)} trades, PnL: ${sum(t['pnl'] for t in trades):,.2f}")
                
        except Exception as e:
            print(f"  ❌ Error backtesting {symbol1}-{symbol2}: {e}")
    
    # Calculate overall statistics
    if all_trades:
        total_trades = len(all_trades)
        winning_trades = sum(1 for t in all_trades if t['pnl'] > 0)
        total_pnl = sum(t['pnl'] for t in all_trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        print(f"\n" + "=" * 60)
        print("OVERALL RESULTS (2023-2025)")
        print("=" * 60)
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Total PnL: ${total_pnl:,.2f}")
        print(f"Average PnL per Trade: ${total_pnl/total_trades:,.2f}")
        
        # Pair breakdown
        print(f"\nPAIR BREAKDOWN:")
        print("-" * 40)
        for pair, stats in pair_results.items():
            pair_win_rate = stats['winning_trades'] / stats['trades'] if stats['trades'] > 0 else 0
            print(f"{pair:<15} {stats['trades']:>3} trades, {pair_win_rate:.1%} win rate, ${stats['pnl']:>8,.2f}")
        
        # Save results
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'period': '2023-2025',
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': total_pnl / total_trades if total_trades > 0 else 0,
            'pair_results': pair_results,
            'selected_pairs': [f"{p[0]}-{p[1]}" for p in selected_pairs],
            'trades': all_trades
        }
        
        with open('simple_backtest_2023_2025_results.json', 'w') as f:
            json.dump(results_summary, f, indent=4, default=str)
        
        print(f"\nResults saved to 'simple_backtest_2023_2025_results.json'")
        
        return results_summary
    else:
        print("❌ No trades were generated!")
        return None

if __name__ == "__main__":
    run_simple_backtest() 