#!/usr/bin/env python3
"""
Simple Analysis: Original vs Enhanced Strategy (2023-2025)
"""

import json

def simple_analysis():
    """Simple analysis of original vs enhanced results."""
    
    print("=" * 70)
    print("ANALYSIS: ORIGINAL vs ENHANCED STRATEGY (2023-2025)")
    print("=" * 70)
    
    # Load results
    try:
        with open('simple_backtest_2023_2025_results.json', 'r') as f:
            original = json.load(f)
        with open('enhanced_backtest_2023_2025_results.json', 'r') as f:
            enhanced = json.load(f)
    except Exception as e:
        print(f"Error loading files: {e}")
        return
    
    print(f"\n📊 PERFORMANCE COMPARISON")
    print("-" * 50)
    print(f"Original Strategy:")
    print(f"  Total Trades: {original['total_trades']}")
    print(f"  Win Rate: {original['win_rate']:.1%}")
    print(f"  Total PnL: ${original['total_pnl']:,.2f}")
    print(f"  Avg PnL/Trade: ${original['avg_pnl_per_trade']:,.2f}")
    
    print(f"\nEnhanced Strategy:")
    print(f"  Total Trades: {enhanced['total_trades']}")
    print(f"  Win Rate: {enhanced['win_rate']:.1%}")
    print(f"  Total PnL: ${enhanced['total_pnl']:,.2f}")
    print(f"  Avg PnL/Trade: ${enhanced['avg_pnl_per_trade']:,.2f}")
    print(f"  Avg Position Size: ${enhanced['avg_position_size']:,.0f}")
    
    # Calculate changes
    trades_change = enhanced['total_trades'] - original['total_trades']
    win_rate_change = enhanced['win_rate'] - original['win_rate']
    pnl_change = enhanced['total_pnl'] - original['total_pnl']
    
    print(f"\n📈 IMPROVEMENTS:")
    print(f"  Trades: {trades_change:+}")
    print(f"  Win Rate: {win_rate_change:+.1%}")
    print(f"  Total PnL: ${pnl_change:+,.2f}")
    
    # Pair analysis
    print(f"\n🔍 PAIR PERFORMANCE:")
    print("-" * 50)
    
    original_pairs = original.get('pair_results', {})
    enhanced_pairs = enhanced.get('pair_results', {})
    
    for pair in enhanced_pairs.keys():
        if pair in original_pairs:
            orig_pnl = original_pairs[pair]['pnl']
            enh_pnl = enhanced_pairs[pair]['pnl']
            change = enh_pnl - orig_pnl
            
            print(f"{pair}:")
            print(f"  Original: ${orig_pnl:,.2f}")
            print(f"  Enhanced: ${enh_pnl:,.2f}")
            print(f"  Change: ${change:+,.2f}")
            
            if change > 0:
                print(f"  ✅ IMPROVED")
            else:
                print(f"  ❌ DECREASED")
            print()
    
    # Find best and worst performers
    best_pair = max(enhanced_pairs.items(), key=lambda x: x[1]['pnl'])
    worst_pair = min(enhanced_pairs.items(), key=lambda x: x[1]['pnl'])
    
    print(f"🎯 KEY INSIGHTS:")
    print(f"  Best Performer: {best_pair[0]} (${best_pair[1]['pnl']:,.2f})")
    print(f"  Worst Performer: {worst_pair[0]} (${worst_pair[1]['pnl']:,.2f})")
    
    print(f"\n🚀 STRATEGIC RECOMMENDATIONS:")
    print(f"  1. Exclude {worst_pair[0]} from portfolio")
    print(f"  2. Scale up {best_pair[0]} position sizes")
    print(f"  3. Implement regime-aware thresholds")
    print(f"  4. Add stop-losses for risk management")

if __name__ == "__main__":
    simple_analysis() 