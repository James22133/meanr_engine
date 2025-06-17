import pandas as pd
from unittest.mock import patch

from core.data_loader import DataLoader

class DummyConfig:
    etf_tickers = ["A", "B"]
    start_date = "2020-01-01"
    end_date = "2020-01-03"
    pair_selection = type("ps", (), {"min_data_points": 1})()

def make_close_first():
    index = pd.date_range("2020-01-01", periods=2)
    columns = pd.MultiIndex.from_product([["Close"], ["A", "B"]])
    return pd.DataFrame([[1, 2], [3, 4]], index=index, columns=columns)

def make_close_second():
    index = pd.date_range("2020-01-01", periods=2)
    columns = pd.MultiIndex.from_product([["A", "B"], ["Close"]])
    return pd.DataFrame([[1, 2], [3, 4]], index=index, columns=columns)

def test_fetch_data_handles_close_first_level():
    loader = DataLoader(DummyConfig)
    with patch("core.data_loader.yf.download", return_value=make_close_first()):
        df = loader.fetch_data()
    assert list(df.columns) == ["A", "B"]
    assert df.iloc[0, 0] == 1
    assert df.iloc[1, 1] == 4

def test_fetch_data_handles_close_second_level():
    loader = DataLoader(DummyConfig)
    with patch("core.data_loader.yf.download", return_value=make_close_second()):
        df = loader.fetch_data()
    assert list(df.columns) == ["A", "B"]
    assert df.iloc[0, 0] == 1
    assert df.iloc[1, 1] == 4
