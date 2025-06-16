import pandas as pd
from typing import List, Tuple

def generate_walkforward_windows(
    df: pd.DataFrame,
    train_months: int = 24,
    test_months: int = 6,
    step_months: int = 6
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Generate rolling walk-forward windows for a time series DataFrame.
    Returns a list of (train_start, train_end, test_start, test_end) tuples.
    """
    start = df.index.min()
    end = df.index.max()
    windows = []
    current_start = pd.to_datetime(start)
    while True:
        train_start = current_start
        train_end = train_start + pd.DateOffset(months=train_months) - pd.DateOffset(days=1)
        test_start = train_end + pd.DateOffset(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.DateOffset(days=1)
        if test_end > end:
            break
        windows.append((train_start, train_end, test_start, test_end))
        current_start = current_start + pd.DateOffset(months=step_months)
    return windows
