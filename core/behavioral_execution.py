from datetime import datetime, time
import pandas as pd


def is_valid_execution_window(current_time: datetime) -> bool:
    """Check if a timestamp falls within the behavioral execution window."""
    # Start 30 minutes before the close instead of 15:45 to better
    # capture late-day retail flows.
    return time(15, 30) <= current_time.time() <= time(15, 59)


def apply_behavioral_execution_filter(signal: pd.Series, execution_times: pd.Series) -> pd.Series:
    """Filter signal entries based on behavioral execution times."""
    return signal.where(execution_times.apply(is_valid_execution_window))
