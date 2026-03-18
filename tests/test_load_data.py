import pandas as pd

from src.data.load_and_profile import build_profile


def test_build_profile_returns_expected_keys() -> None:
    df = pd.DataFrame(
        {
            "Time": [0, 1, 2],
            "Amount": [10.0, 20.0, 30.0],
            "Class": [0, 0, 1],
        }
    )

    profile = build_profile(df)

    assert profile["shape"] == [3, 3]
    assert profile["target_column"] == "Class"
    assert profile["class_distribution"]["0"] == 2
    assert profile["class_distribution"]["1"] == 1
    assert "amount_stats" in profile
    assert "time_stats" in profile