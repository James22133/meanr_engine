import pandas as pd
import pytest
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

def make_no_close():
    index = pd.date_range("2020-01-01", periods=2)
    columns = pd.MultiIndex.from_product([["Open"], ["A", "B"]])
    return pd.DataFrame([[1, 2], [3, 4]], index=index, columns=columns)

class DummyConfigInsufficient(DummyConfig):
    pair_selection = type("ps", (), {"min_data_points": 3})()

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


def test_fetch_data_missing_close_raises():
    loader = DataLoader(DummyConfig)
    with patch("core.data_loader.yf.download", return_value=make_no_close()):
        with pytest.raises(ValueError):
            loader.fetch_data()


def test_get_pair_data_missing_column_returns_none():
    loader = DataLoader(DummyConfig)
    loader.data = pd.DataFrame({"A": [1, 2]}, index=pd.date_range("2020-01-01", periods=2))
    result = loader.get_pair_data(["A", "C"])
    assert result is None


def test_get_pair_data_insufficient_after_dropna():
    loader = DataLoader(DummyConfigInsufficient)
    loader.data = pd.DataFrame(
        {"A": [1.0, None], "B": [1.0, 2.0]},
        index=pd.date_range("2020-01-01", periods=2),
    )
    result = loader.get_pair_data(["A", "B"])
    assert result is None
