import pandas as pd
import numpy as np
from unittest.mock import patch

from core.pair_selection import PairSelector


class DummyConfig:
    pair_selection = type(
        "ps",
        (),
        {
            "stability_lookback": 1,
            "max_zscore_volatility": 10.0,
        },
    )()


def test_calculate_pair_metrics_uses_submethods():
    selector = PairSelector(DummyConfig)
    data = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 4.0, 6.0]})

    fake_kalman = {
        "filtered_state_means": np.ones((3, 2)),
        "filtered_state_covariances": np.ones((3, 2, 2)),
        "transition_matrix": np.eye(2),
        "observation_matrix": np.array([[1.0, 0.0]]),
    }
    fake_spread = pd.Series([0.1, 0.2, 0.3])
    fake_zscore = pd.Series([1.0, 0.0, -1.0])

    with patch.object(PairSelector, "_calculate_cointegration", return_value=0.05), \
         patch.object(PairSelector, "_calculate_kalman_params", return_value=fake_kalman), \
         patch.object(PairSelector, "_calculate_spread", return_value=fake_spread), \
         patch.object(PairSelector, "_calculate_stability_metrics", return_value=0.5), \
         patch.object(PairSelector, "_calculate_zscore", return_value=fake_zscore), \
         patch.object(PairSelector, "_calculate_composite_score", return_value=0.8):
        metrics = selector.calculate_pair_metrics(data)

    assert metrics["correlation"] == data.corr().iloc[0, 1]
    assert metrics["coint_pvalue"] == 0.05
    assert metrics["spread_stability"] == 0.5
    assert metrics["zscore_volatility"] == fake_zscore.std()
    assert metrics["score"] == 0.8
    assert metrics["spread"].equals(fake_spread)
    assert metrics["zscore"].equals(fake_zscore)
    assert metrics["kalman_params"] == fake_kalman
