import numpy as np
import pandas as pd

from run_engine import compute_pair_score, compute_pair_scores


def test_compute_pair_score_basic():
    score = compute_pair_score(0.1, 0.2, 0.3, 0.4)
    assert score == 1 - np.nanmean([0.1, 0.2, 0.3, 0.4])


def test_compute_pair_score_handles_nan():
    score = compute_pair_score(np.nan, 0.2, 0.3, 0.4)
    expected = 1 - np.nanmean([np.nan, 0.2, 0.3, 0.4])
    assert np.isclose(score, expected)


def test_compute_pair_scores_normalization():
    df = pd.DataFrame(
        {
            "coint_p": [0.2, 0.1, 0.3],
            "hurst": [0.5, 0.4, 0.6],
            "adf_p": [0.2, 0.3, 0.1],
            "zscore_vol": [0.3, 0.2, 0.4],
        }
    )
    result = compute_pair_scores(df.copy())

    # Manual normalization
    metrics_cols = ["coint_p", "hurst", "adf_p", "zscore_vol"]
    norm = (df[metrics_cols] - df[metrics_cols].min()) / (
        df[metrics_cols].max() - df[metrics_cols].min() + 1e-8
    )
    expected_scores = 1 - norm.mean(axis=1)

    assert np.allclose(result["score"], expected_scores)
