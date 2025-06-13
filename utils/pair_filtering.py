import yaml
import pandas as pd

def filter_pairs_by_sharpe(pair_results, min_sharpe):
    """Return only pairs with Sharpe >= min_sharpe."""
    return [r for r in pair_results if r['metrics']['sharpe_ratio'] >= min_sharpe]

def export_top_pairs(pair_results, top_n, out_path):
    """Export top N pairs by Sharpe to a YAML file."""
    sorted_pairs = sorted(pair_results, key=lambda r: r['metrics']['sharpe_ratio'], reverse=True)
    top_pairs = [tuple(r['pair']) for r in sorted_pairs[:top_n]]
    with open(out_path, 'w') as f:
        yaml.dump({'selected_pairs': top_pairs}, f) 