"""Trade extraction utilities."""

from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd


def filter_long_holding_trades(trades: List[Dict], max_holding_days: int = 30) -> List[Dict]:
    """Filter out trades exceeding the maximum holding period."""
    filtered = []
    for trade in trades:
        holding = trade.get("holding_period")
        if holding is None and "entry_date" in trade and "exit_date" in trade:
            entry = trade["entry_date"]
            exit_ = trade["exit_date"]
            if isinstance(entry, (pd.Timestamp, datetime)) and isinstance(exit_, (pd.Timestamp, datetime)):
                holding = (exit_ - entry).days
        if holding is None or holding <= max_holding_days:
            filtered.append(trade)
    return filtered
