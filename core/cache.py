import os
import pickle
from typing import Any


def save_to_cache(obj: Any, filename: str) -> None:
    """Save object to cache using pickle."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_from_cache(filename: str) -> Any:
    """Load object from cache if file exists."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Cache file {filename} not found")
    with open(filename, 'rb') as f:
        return pickle.load(f)
