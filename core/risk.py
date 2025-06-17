from typing import Dict, List
import pandas as pd


def compute_sector_exposure(trades: List[Dict], sector_map: Dict[str, str] = None) -> Dict[str, float]:
    """Compute normalized exposure by sector based on trade sizes."""
    exposure: Dict[str, float] = {}
    if sector_map is None:
        sector_map = {}
    for trade in trades:
        sector1 = sector_map.get(trade.get('asset1'), 'Unknown')
        sector2 = sector_map.get(trade.get('asset2'), 'Unknown')
        size = abs(trade.get('size', 0))
        exposure[sector1] = exposure.get(sector1, 0.0) + size
        exposure[sector2] = exposure.get(sector2, 0.0) + size
    total = sum(exposure.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in exposure.items()}


def max_drawdown(series: pd.Series) -> float:
    """Calculate maximum drawdown of a cumulative return or spread series."""
    if series.empty:
        return 0.0
    cumulative = series.cumsum() if not series.index.is_monotonic_increasing else series
    running_max = cumulative.cummax()
    drawdowns = cumulative - running_max
    return float(drawdowns.min())
