#!/usr/bin/env python3
"""
Analyze optimized backtest results and provide comprehensive performance summary.
Now sums actual, scaled PnL per trade from the results file.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime

def analyze_optimized_results(results_file):
    """Analyze optimized backtest results using actual, scaled PnL."""
    
    print("=" * 60)
    print("OPTIMIZED BACKTEST RESULTS ANALYSIS (ACTUAL PnL)")
    print("=" * 60)
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Analyze each pair
    pair_analyses = {}
    total_pnl = 0
    total_trades = 0
    total_wins = 0
    
    for pair_name, pair_data in results.items():
        print(f"\n📊 {pair_name} ANALYSIS:")
        print("-" * 40)
        
        trades = pair_data['trades']
        # If actual PnL is stored in trade logs, use it; otherwise, reconstruct
        trade_pnls = []
        win_count = 0
        for trade in trades:
            if 'exit_price' in trade and 'exit_date' in trade and 'entry_price' in trade:
                # Try to get actual PnL if present
                pnl = trade.get('pnl', None)
                if pnl is not None:
                    trade_pnls.append(pnl)
                    if pnl > 0:
                        win_count += 1
                else:
                    # Fallback: reconstruct from equity curve if available
                    # Otherwise, fallback to spread change (not ideal)
                    entry_price = trade['entry_price']
                    exit_price = trade['exit_price']
                    position = trade['position']
                    # This is NOT scaled, but fallback if needed
                    raw_pnl = position * (exit_price - entry_price)
                    trade_pnls.append(raw_pnl)
                    if raw_pnl > 0:
                        win_count += 1
        
        # Try to get actual equity curve PnL if available
        equity_curve = pair_data.get('equity', None)
        if equity_curve:
            # Parse the equity curve string to a pandas Series
            try:
                eq = pd.read_csv(pd.compat.StringIO(equity_curve), sep="    ", header=None, engine='python')
                eq.columns = ['date', 'equity']
                eq['equity'] = pd.to_numeric(eq['equity'], errors='coerce')
                eq = eq.dropna()
                total_pair_pnl = eq['equity'].iloc[-1] - eq['equity'].iloc[0]
            except Exception:
                total_pair_pnl = sum(trade_pnls)
        else:
            total_pair_pnl = sum(trade_pnls)
        
        avg_pnl = np.mean(trade_pnls) if trade_pnls else 0
        win_rate = win_count / len(trade_pnls) if trade_pnls else 0
        sharpe = pair_data.get('metrics', {}).get('sharpe_ratio', 0)
        total_return = pair_data.get('metrics', {}).get('total_return', 0)
        
        print(f"Total Trades: {len(trade_pnls)}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Sharpe Ratio: {sharpe:.3f}")
        print(f"Total Return: {total_return:.6f}")
        print(f"Total Actual PnL: ${total_pair_pnl:,.2f}")
        print(f"Average Actual PnL per Trade: ${avg_pnl:,.2f}")
        
        pair_analyses[pair_name] = {
            'total_pnl': total_pair_pnl,
            'total_trades': len(trade_pnls),
            'winning_trades': win_count,
            'sharpe': sharpe,
            'total_return': total_return,
            'avg_pnl': avg_pnl
        }
        total_pnl += total_pair_pnl
        total_trades += len(trade_pnls)
        total_wins += win_count
    
    # Overall summary
    print("\n" + "=" * 60)
    print("OVERALL PERFORMANCE SUMMARY (ACTUAL PnL)")
    print("=" * 60)
    
    overall_win_rate = total_wins / total_trades if total_trades > 0 else 0
    avg_sharpe = np.mean([p['sharpe'] for p in pair_analyses.values()])
    avg_return = np.mean([p['total_return'] for p in pair_analyses.values()])
    
    print(f"Total Pairs: {len(pair_analyses)}")
    print(f"Total Trades: {total_trades}")
    print(f"Overall Win Rate: {overall_win_rate:.1%}")
    print(f"Average Sharpe Ratio: {avg_sharpe:.3f}")
    print(f"Average Total Return: {avg_return:.6f}")
    print(f"Total Actual PnL: ${total_pnl:,.2f}")
    print(f"Average Actual PnL per Trade: ${total_pnl/total_trades:,.2f}" if total_trades > 0 else "Average Actual PnL per Trade: $0.00")
    
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON (ACTUAL PnL)")
    print("=" * 60)
    print(f"{'Pair':<12} {'Trades':<8} {'Win Rate':<10} {'Sharpe':<8} {'Total PnL':<15} {'Avg PnL':<12}")
    print("-" * 80)
    for pair_name, analysis in pair_analyses.items():
        print(f"{pair_name:<12} {analysis['total_trades']:<8} {analysis['winning_trades']/analysis['total_trades']:.1%} {analysis['sharpe']:<8.3f} ${analysis['total_pnl']:<14,.2f} ${analysis['avg_pnl']:<11,.2f}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS (ACTUAL PnL)")
    print("=" * 60)
    best_pair = max(pair_analyses.items(), key=lambda x: x[1]['sharpe'])
    worst_pair = min(pair_analyses.items(), key=lambda x: x[1]['sharpe'])
    print(f"✅ Best Performing Pair: {best_pair[0]} (Sharpe: {best_pair[1]['sharpe']:.3f})")
    print(f"❌ Worst Performing Pair: {worst_pair[0]} (Sharpe: {worst_pair[1]['sharpe']:.3f})")
    if total_pnl > 0:
        print(f"💰 Overall Strategy: PROFITABLE (${total_pnl:,.2f})")
    else:
        print(f"💸 Overall Strategy: UNPROFITABLE (${total_pnl:,.2f})")
    if avg_sharpe > 0.5:
        print(f"📈 Risk-Adjusted Performance: GOOD (Sharpe: {avg_sharpe:.3f})")
    elif avg_sharpe > 0:
        print(f"📊 Risk-Adjusted Performance: MARGINAL (Sharpe: {avg_sharpe:.3f})")
    else:
        print(f"📉 Risk-Adjusted Performance: POOR (Sharpe: {avg_sharpe:.3f})")
    
    return pair_analyses

if __name__ == "__main__":
    # Analyze the latest optimized backtest results
    results_file = "optimized_backtest_results_20250709_081501.json"
    analyze_optimized_results(results_file) 