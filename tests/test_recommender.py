"""Unit tests for src/recommender.py.

All three dependencies (client, scorer, roster) are replaced with minimal
stub objects that implement the required protocol methods.  No network
calls, no filesystem access, no YAML parsing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recommender import (
    CompareResult,
    Recommender,
    SwapRecommendation,
    _evaluate_swap,
    _my_players_at_position,
    _player_type,
    _select_display_cols,
)


# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

def _batter_scored_df() -> pd.DataFrame:
    """Six batters with realistic scored column names, sorted desc."""
    return pd.DataFrame(
        {
            "player_name": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"],
            "player_id": [100, 101, 102, 103, 104, 105],
            "composite_score": [95.0, 80.0, 65.0, 50.0, 35.0, 10.0],
            "brl_percent_score": [90.0, 75.0, 60.0, 45.0, 30.0, 10.0],
            "hard_hit_percent_score": [92.0, 78.0, 63.0, 48.0, 33.0, 12.0],
            "xwoba_score": [94.0, 80.0, 66.0, 51.0, 36.0, 11.0],
            "sprint_speed_score": [88.0, 70.0, 55.0, 40.0, 25.0, 8.0],
            "k_percent_score": [91.0, 77.0, 62.0, 47.0, 32.0, 9.0],
        }
    )


def _pitcher_scored_df() -> pd.DataFrame:
    """Four pitchers with pitcher-specific score columns."""
    return pd.DataFrame(
        {
            "player_name": ["FlameThrower", "GroundBaller", "FinessePro", "BulletPen"],
            "player_id": [200, 201, 202, 203],
            "composite_score": [88.0, 72.0, 55.0, 30.0],
            "xera_score": [85.0, 70.0, 52.0, 28.0],
            "whiff_percent_score": [90.0, 74.0, 57.0, 32.0],
            "chase_percent_score": [87.0, 71.0, 54.0, 29.0],
            "brl_percent_score": [88.0, 72.0, 55.0, 30.0],
            "hard_hit_percent_score": [89.0, 73.0, 56.0, 31.0],
        }
    )


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _StubClient:
    """Returns pre-built DataFrames; never touches the network."""

    def __init__(self, batter_df: pd.DataFrame, pitcher_df: pd.DataFrame) -> None:
        self._batter = batter_df
        self._pitcher = pitcher_df

    def fetch_batting_data(self, days: int = 14) -> dict[str, pd.DataFrame]:
        return {"percentile_ranks": self._batter.copy(), "play_log": pd.DataFrame()}

    def fetch_pitching_data(self, days: int = 14) -> dict[str, pd.DataFrame]:
        return {"percentile_ranks": self._pitcher.copy(), "pitch_log": pd.DataFrame()}


class _StubScorer:
    """Returns the DataFrame unchanged — scoring already applied in fixture data."""

    def score_batters(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def score_pitchers(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()


class _RosterPlayerStub:
    def __init__(self, player_id: int, positions: list[str]) -> None:
        self.player_id = player_id
        self.positions = positions
        self.name = f"Rostered-{player_id}"


class _StubRoster:
    """Marks player_ids 103, 104, 105 as 'my roster'; 100 as unavailable."""

    # player_id 103 → 3B, player_id 104 → OF, player_id 105 → OF
    my_roster = [
        _RosterPlayerStub(103, ["3B"]),
        _RosterPlayerStub(104, ["OF"]),
        _RosterPlayerStub(105, ["OF"]),
    ]
    _unavailable_ids = {100}  # Alpha is on another team

    def get_available_players(self, df: pd.DataFrame) -> pd.DataFrame:
        all_taken = {p.player_id for p in self.my_roster} | self._unavailable_ids
        return df[~df["player_id"].isin(all_taken)].reset_index(drop=True)


def _make_recommender(
    batter_df: pd.DataFrame | None = None,
    pitcher_df: pd.DataFrame | None = None,
    roster: _StubRoster | None = None,
) -> Recommender:
    bd = batter_df if batter_df is not None else _batter_scored_df()
    pd_ = pitcher_df if pitcher_df is not None else _pitcher_scored_df()
    return Recommender(
        client=_StubClient(bd, pd_),
        scorer=_StubScorer(),
        roster=roster or _StubRoster(),
    )


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------

class TestPlayerType:
    def test_none_returns_batter(self):
        assert _player_type(None) == "batter"

    @pytest.mark.parametrize("pos", ["C", "1B", "2B", "3B", "SS", "OF"])
    def test_batter_positions_return_batter(self, pos):
        assert _player_type(pos) == "batter"

    @pytest.mark.parametrize("pos", ["SP", "RP"])
    def test_pitcher_positions_return_pitcher(self, pos):
        assert _player_type(pos) == "pitcher"


class TestSelectDisplayCols:
    def test_batter_display_cols_returned(self):
        result = _select_display_cols(_batter_scored_df(), "batter")
        assert "composite_score" in result.columns
        assert "xwoba_score" in result.columns
        # pitcher-only columns should be absent
        assert "xera_score" not in result.columns

    def test_pitcher_display_cols_returned(self):
        result = _select_display_cols(_pitcher_scored_df(), "pitcher")
        assert "xera_score" in result.columns
        # batter-only columns absent
        assert "sprint_speed_score" not in result.columns

    def test_missing_cols_silently_dropped(self):
        df = _batter_scored_df().drop(columns=["sprint_speed_score"])
        result = _select_display_cols(df, "batter")
        assert "sprint_speed_score" not in result.columns
        assert "composite_score" in result.columns


class TestMyPlayersAtPosition:
    def test_returns_correct_players(self):
        scored = _batter_scored_df()
        roster = _StubRoster()
        result = _my_players_at_position(scored, roster, "OF")
        assert set(result["player_id"]) == {104, 105}

    def test_position_with_no_roster_players_returns_empty(self):
        scored = _batter_scored_df()
        roster = _StubRoster()
        result = _my_players_at_position(scored, roster, "C")
        assert result.empty

    def test_respects_multi_position_eligibility(self):
        scored = _batter_scored_df()
        roster = _StubRoster()
        result = _my_players_at_position(scored, roster, "3B")
        assert set(result["player_id"]) == {103}

    def test_returns_empty_when_player_id_column_absent(self):
        scored = _batter_scored_df().drop(columns=["player_id"])
        roster = _StubRoster()
        result = _my_players_at_position(scored, roster, "OF")
        assert result.empty


class TestEvaluateSwap:
    def _df(self, name: str, pid: int, score: float) -> pd.DataFrame:
        return pd.DataFrame(
            {"player_name": [name], "player_id": [pid], "composite_score": [score]}
        )

    def test_swap_recommended_when_delta_exceeds_threshold(self):
        available = self._df("WaiverStar", 999, 85.0)
        mine = self._df("MyWeak", 103, 50.0)
        swap = _evaluate_swap(available, mine)
        assert swap is not None
        assert swap.score_delta == pytest.approx(35.0)

    def test_no_swap_when_delta_below_threshold(self):
        available = self._df("WaiverOk", 999, 55.0)
        mine = self._df("MyOk", 103, 50.0)
        assert _evaluate_swap(available, mine) is None

    def test_swap_at_exactly_threshold(self):
        # "at least 10 points higher" → delta == 10.0 qualifies.
        available = self._df("WaiverEdge", 999, 60.0)
        mine = self._df("MyEdge", 103, 50.0)
        assert _evaluate_swap(available, mine) is not None

    def test_no_swap_just_below_threshold(self):
        available = self._df("WaiverClose", 999, 59.9)
        mine = self._df("MyEdge", 103, 50.0)
        assert _evaluate_swap(available, mine) is None

    def test_no_swap_when_available_empty(self):
        empty = pd.DataFrame(columns=["player_name", "player_id", "composite_score"])
        mine = self._df("MyPlayer", 103, 50.0)
        assert _evaluate_swap(empty, mine) is None

    def test_no_swap_when_my_players_empty(self):
        available = self._df("WaiverStar", 999, 90.0)
        empty = pd.DataFrame(columns=["player_name", "player_id", "composite_score"])
        assert _evaluate_swap(available, empty) is None

    def test_swap_fields_populated_correctly(self):
        available = self._df("AddMe", 888, 90.0)
        mine = self._df("DropMe", 103, 40.0)
        swap = _evaluate_swap(available, mine)
        assert isinstance(swap, SwapRecommendation)
        assert swap.add_name == "AddMe"
        assert swap.drop_name == "DropMe"
        assert swap.add_score == pytest.approx(90.0)
        assert swap.drop_score == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# Recommender.recommend_pickups
# ---------------------------------------------------------------------------

class TestRecommendPickups:
    def setup_method(self):
        self.rec = _make_recommender()

    def test_returns_dataframe(self):
        assert isinstance(self.rec.recommend_pickups(), pd.DataFrame)

    def test_default_returns_batters(self):
        result = self.rec.recommend_pickups()
        # Batter display columns present
        assert "xwoba_score" in result.columns
        assert "xera_score" not in result.columns

    def test_pitcher_position_returns_pitcher_cols(self):
        result = self.rec.recommend_pickups(position="SP")
        assert "xera_score" in result.columns
        assert "xwoba_score" not in result.columns

    def test_top_n_respected(self):
        result = self.rec.recommend_pickups(top_n=2)
        assert len(result) <= 2

    def test_my_rostered_players_excluded(self):
        result = self.rec.recommend_pickups()
        my_ids = {p.player_id for p in _StubRoster().my_roster}
        assert set(result["player_id"]).isdisjoint(my_ids)

    def test_unavailable_players_excluded(self):
        result = self.rec.recommend_pickups()
        # player_id 100 (Alpha) is on another team in _StubRoster
        assert 100 not in result["player_id"].values

    def test_results_sorted_descending(self):
        result = self.rec.recommend_pickups()
        scores = result["composite_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_invalid_position_raises(self):
        with pytest.raises(ValueError, match="not a valid position"):
            self.rec.recommend_pickups(position="DH")

    def test_batter_position_returns_batters(self):
        result = self.rec.recommend_pickups(position="OF")
        assert "xwoba_score" in result.columns

    def test_composite_score_column_present(self):
        assert "composite_score" in self.rec.recommend_pickups().columns


# ---------------------------------------------------------------------------
# Recommender.compare_to_roster
# ---------------------------------------------------------------------------

class TestCompareToRoster:
    def setup_method(self):
        self.rec = _make_recommender()

    def test_returns_compare_result(self):
        result = self.rec.compare_to_roster("OF")
        assert isinstance(result, CompareResult)

    def test_position_stored_on_result(self):
        result = self.rec.compare_to_roster("OF")
        assert result.position == "OF"

    def test_player_type_set_correctly_for_batter_position(self):
        result = self.rec.compare_to_roster("OF")
        assert result.player_type == "batter"

    def test_player_type_set_correctly_for_pitcher_position(self):
        result = self.rec.compare_to_roster("SP")
        assert result.player_type == "pitcher"

    def test_my_players_contains_only_eligible(self):
        result = self.rec.compare_to_roster("OF")
        my_of_ids = {104, 105}
        assert set(result.my_players["player_id"]).issubset(my_of_ids)

    def test_available_players_excludes_my_roster(self):
        result = self.rec.compare_to_roster("OF")
        my_ids = {p.player_id for p in _StubRoster().my_roster}
        assert set(result.available_players["player_id"]).isdisjoint(my_ids)

    def test_top_n_limits_available_players(self):
        result = self.rec.compare_to_roster("OF", top_n=2)
        assert len(result.available_players) <= 2

    def test_swap_when_large_score_gap(self):
        # Alpha (id=100) scores 95 — Beta (id=101) scores 80.
        # My OF players are Epsilon (35) and Zeta (10).
        # Best available (Beta, 80) > weakest mine (Zeta, 10) by 70 pts → swap.
        result = self.rec.compare_to_roster("OF")
        assert result.has_swap()

    def test_no_swap_when_my_players_competitive(self):
        # Override: my OF players have high scores — no available beats them by 10.
        class _StrongRoster(_StubRoster):
            my_roster = [
                _RosterPlayerStub(101, ["OF"]),  # Beta: 80.0
                _RosterPlayerStub(102, ["OF"]),  # Gamma: 65.0
            ]

        rec = _make_recommender(roster=_StrongRoster())
        result = rec.compare_to_roster("OF", top_n=3)
        # Best available ≤ 50 (Delta), weakest mine = 65 → no swap
        assert not result.has_swap()

    def test_swap_recommendation_fields(self):
        result = self.rec.compare_to_roster("OF")
        assert result.swap is not None
        assert isinstance(result.swap.drop_name, str)
        assert isinstance(result.swap.add_name, str)
        assert result.swap.score_delta >= 10.0

    def test_invalid_position_raises(self):
        with pytest.raises(ValueError, match="not a valid position"):
            self.rec.compare_to_roster("DH")

    def test_position_with_no_my_players_returns_empty_my_players(self):
        result = self.rec.compare_to_roster("C")  # no C on _StubRoster
        assert result.my_players.empty

    def test_str_output_contains_position(self):
        result = self.rec.compare_to_roster("OF")
        assert "OF" in str(result)


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

class TestProtocolCompliance:
    def test_stub_client_satisfies_protocol(self):
        from recommender import StatcastClientProtocol
        assert isinstance(_StubClient(_batter_scored_df(), _pitcher_scored_df()), StatcastClientProtocol)

    def test_stub_scorer_satisfies_protocol(self):
        from recommender import ScorerProtocol
        assert isinstance(_StubScorer(), ScorerProtocol)

    def test_stub_roster_satisfies_protocol(self):
        from recommender import RosterProtocol
        assert isinstance(_StubRoster(), RosterProtocol)
