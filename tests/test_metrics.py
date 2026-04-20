"""Unit tests for src/analysis/metrics.py — PlayerScorer."""

import math

import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.metrics import PlayerScorer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BATTER_POOL = {
    "player_name": [f"Batter {i}" for i in range(6)],
    "player_id": list(range(100, 106)),
    "year": [2024] * 6,
    "brl_percent":       [10, 30, 50, 60, 80, 95],
    "hard_hit_percent":  [15, 35, 45, 65, 75, 90],
    "xwoba":             [20, 25, 55, 60, 70, 88],
    "sprint_speed":      [40, 50, 60, 50, 70, 85],
    "k_percent":         [25, 40, 55, 65, 75, 90],
}


def _batter_df(n: int = 6) -> pd.DataFrame:
    """Minimal batter percentile-rank DataFrame with all expected columns."""
    return pd.DataFrame({k: v[:n] for k, v in _BATTER_POOL.items()})


def _pitcher_df(n: int = 6) -> pd.DataFrame:
    """Minimal pitcher percentile-rank DataFrame with all expected columns."""
    return pd.DataFrame(
        {
            "player_name": [f"Pitcher {i}" for i in range(n)],
            "player_id": list(range(200, 200 + n)),
            "year": [2024] * n,
            "xera": [10, 25, 45, 60, 80, 95],
            "whiff_percent": [15, 30, 50, 60, 75, 88],
            "chase_percent": [20, 35, 45, 65, 70, 90],
            "brl_percent": [10, 30, 50, 55, 80, 92],
            "hard_hit_percent": [12, 28, 48, 62, 78, 95],
        }
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestPlayerScorerInit:
    def test_default_weights_are_accepted(self):
        PlayerScorer()  # should not raise

    def test_custom_batter_weights_accepted(self):
        custom = {
            "brl_percent": 0.50,
            "hard_hit_percent": 0.20,
            "xwoba": 0.20,
            "sprint_speed": 0.05,
            "k_percent": 0.05,
        }
        PlayerScorer(batter_weights=custom)

    def test_batter_weights_not_summing_to_one_raises(self):
        bad = {"brl_percent": 0.50, "hard_hit_percent": 0.30}  # sums to 0.80
        with pytest.raises(ValueError, match="weights sum to"):
            PlayerScorer(batter_weights=bad)

    def test_pitcher_weights_not_summing_to_one_raises(self):
        bad = {"xera": 0.40, "whiff_percent": 0.40}  # sums to 0.80
        with pytest.raises(ValueError, match="weights sum to"):
            PlayerScorer(pitcher_weights=bad)


# ---------------------------------------------------------------------------
# Batter scoring — output structure
# ---------------------------------------------------------------------------

class TestScoreBattersStructure:
    def setup_method(self):
        self.scorer = PlayerScorer()
        self.result = self.scorer.score_batters(_batter_df())

    def test_returns_dataframe(self):
        assert isinstance(self.result, pd.DataFrame)

    def test_contains_composite_score_column(self):
        assert "composite_score" in self.result.columns

    def test_contains_normalized_metric_columns(self):
        expected_norm = [
            "brl_percent_score",
            "hard_hit_percent_score",
            "xwoba_score",
            "sprint_speed_score",
            "k_percent_score",
        ]
        for col in expected_norm:
            assert col in self.result.columns, f"Missing column: {col}"

    def test_preserves_identity_columns(self):
        for col in ("player_name", "player_id", "year"):
            assert col in self.result.columns

    def test_row_count_unchanged(self):
        assert len(self.result) == 6

    def test_sorted_descending(self):
        scores = self.result["composite_score"].tolist()
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Batter scoring — value correctness
# ---------------------------------------------------------------------------

class TestScoreBattersValues:
    def setup_method(self):
        self.scorer = PlayerScorer()
        self.result = self.scorer.score_batters(_batter_df())

    def test_composite_scores_in_range(self):
        assert self.result["composite_score"].between(0, 100).all()

    def test_normalized_metric_scores_in_range(self):
        norm_cols = [c for c in self.result.columns if c.endswith("_score") and c != "composite_score"]
        for col in norm_cols:
            vals = self.result[col].dropna()
            assert (vals >= 0).all() and (vals <= 100).all(), f"{col} out of [0, 100]"

    def test_top_player_has_highest_composite(self):
        # Last row in fixture has highest values across all metrics.
        top_name = self.result.iloc[0]["player_name"]
        assert top_name == "Batter 5"

    def test_bottom_player_has_lowest_composite(self):
        bottom_name = self.result.iloc[-1]["player_name"]
        assert bottom_name == "Batter 0"

    def test_dominant_player_scores_above_90(self):
        top_score = self.result.iloc[0]["composite_score"]
        assert top_score > 90

    def test_weakest_player_scores_below_25(self):
        # With 6 players, rank(pct=True) minimum is 1/6 ≈ 16.67.
        bottom_score = self.result.iloc[-1]["composite_score"]
        assert bottom_score < 25


# ---------------------------------------------------------------------------
# Pitcher scoring — output structure
# ---------------------------------------------------------------------------

class TestScorePitchersStructure:
    def setup_method(self):
        self.scorer = PlayerScorer()
        self.result = self.scorer.score_pitchers(_pitcher_df())

    def test_returns_dataframe(self):
        assert isinstance(self.result, pd.DataFrame)

    def test_contains_composite_score_column(self):
        assert "composite_score" in self.result.columns

    def test_contains_normalized_metric_columns(self):
        expected_norm = [
            "xera_score",
            "whiff_percent_score",
            "chase_percent_score",
            "brl_percent_score",
            "hard_hit_percent_score",
        ]
        for col in expected_norm:
            assert col in self.result.columns, f"Missing column: {col}"

    def test_sorted_descending(self):
        scores = self.result["composite_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_row_count_unchanged(self):
        assert len(self.result) == 6


# ---------------------------------------------------------------------------
# Pitcher scoring — value correctness
# ---------------------------------------------------------------------------

class TestScorePitchersValues:
    def setup_method(self):
        self.scorer = PlayerScorer()
        self.result = self.scorer.score_pitchers(_pitcher_df())

    def test_composite_scores_in_range(self):
        assert self.result["composite_score"].between(0, 100).all()

    def test_top_pitcher_has_highest_composite(self):
        top_name = self.result.iloc[0]["player_name"]
        assert top_name == "Pitcher 5"

    def test_bottom_pitcher_has_lowest_composite(self):
        bottom_name = self.result.iloc[-1]["player_name"]
        assert bottom_name == "Pitcher 0"


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

class TestNaNHandling:
    def test_missing_metric_values_do_not_propagate_to_composite(self):
        df = _batter_df()
        df.loc[0, "brl_percent"] = float("nan")
        result = PlayerScorer().score_batters(df)
        # Row with NaN should still have a finite composite score.
        nan_row = result[result["player_name"] == "Batter 0"]
        assert math.isfinite(nan_row["composite_score"].iloc[0])

    def test_missing_column_warns_and_redistributes_weight(self, caplog):
        df = _batter_df().drop(columns=["sprint_speed"])
        import logging
        with caplog.at_level(logging.WARNING, logger="analysis.metrics"):
            result = PlayerScorer().score_batters(df)
        assert "sprint_speed" in caplog.text
        # Composite should still be computed over remaining columns.
        assert result["composite_score"].between(0, 100).all()
        assert "sprint_speed_score" not in result.columns


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_player_scores_100(self):
        df = _batter_df(n=1)
        result = PlayerScorer().score_batters(df)
        # Only one player — ranks at 100th percentile on every metric.
        assert result["composite_score"].iloc[0] == pytest.approx(100.0)

    def test_empty_dataframe_raises(self):
        empty = pd.DataFrame(columns=_batter_df().columns)
        with pytest.raises(ValueError, match="empty"):
            PlayerScorer().score_batters(empty)

    def test_no_recognised_columns_raises(self):
        df = pd.DataFrame({"player_name": ["X"], "irrelevant_col": [42]})
        with pytest.raises(ValueError, match="No recognised"):
            PlayerScorer().score_batters(df)

    def test_two_identical_players_score_equally(self):
        df = pd.DataFrame(
            {
                "player_name": ["Alpha", "Beta"],
                "brl_percent": [50, 50],
                "hard_hit_percent": [50, 50],
                "xwoba": [50, 50],
                "sprint_speed": [50, 50],
                "k_percent": [50, 50],
            }
        )
        result = PlayerScorer().score_batters(df)
        assert result["composite_score"].iloc[0] == result["composite_score"].iloc[1]
