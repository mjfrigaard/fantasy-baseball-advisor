"""Top-level recommendation engine.

Ties together the Statcast client, PlayerScorer, and RosterManager to answer
the two core fantasy questions:

1. ``recommend_pickups``  — who is available and worth adding?
2. ``compare_to_roster``  — should I swap anyone on my roster?

All three dependencies are injected at construction time so the class is
fully testable without hitting the network or the filesystem.

Scoring is always performed on the **full player pool** so that composite
scores are percentile-ranked relative to all players in the dataset, not
just the available subset.  The available-player filter is applied after
scoring.

Note on position filtering
--------------------------
pybaseball's percentile-rank endpoints do not include position eligibility
data.  This means the waiver pool can only be split at the batter/pitcher
boundary.  Within-batter position filtering (e.g. "show me only OFs") is
applied to ``my_roster`` using the YAML-declared positions but **not** to
the waiver pool.  Users should verify position eligibility of pickup
targets in their league platform before adding them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import pandas as pd

from roster.manager import BATTER_POSITIONS, PITCHER_POSITIONS, ALL_POSITIONS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Display column sets — trimmed output returned to callers
# ---------------------------------------------------------------------------

_BATTER_DISPLAY_COLS = [
    "player_name",
    "player_id",
    "composite_score",
    "brl_percent_score",
    "hard_hit_percent_score",
    "xwoba_score",
    "sprint_speed_score",
    "k_percent_score",
]

_PITCHER_DISPLAY_COLS = [
    "player_name",
    "player_id",
    "composite_score",
    "xera_score",
    "whiff_percent_score",
    "chase_percent_score",
    "brl_percent_score",
    "hard_hit_percent_score",
]

_SWAP_THRESHOLD = 10.0  # minimum score advantage to trigger a swap recommendation


# ---------------------------------------------------------------------------
# Dependency protocols (structural subtyping — no inheritance required)
# ---------------------------------------------------------------------------

@runtime_checkable
class StatcastClientProtocol(Protocol):
    """Minimal interface required from the Statcast data client."""

    def fetch_batting_data(self, days: int = 14) -> dict[str, pd.DataFrame]:
        ...

    def fetch_pitching_data(self, days: int = 14) -> dict[str, pd.DataFrame]:
        ...


@runtime_checkable
class ScorerProtocol(Protocol):
    """Minimal interface required from the player scorer."""

    def score_batters(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def score_pitchers(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


@runtime_checkable
class RosterProtocol(Protocol):
    """Minimal interface required from the roster manager."""

    my_roster: list  # list of RosterPlayer-like objects with .player_id and .positions

    def get_available_players(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SwapRecommendation:
    """A suggested drop/add pair when a waiver target clearly outperforms a roster player.

    Attributes:
        drop_name:      Display name of the player to drop.
        drop_player_id: MLBAM id of the player to drop.
        drop_score:     Composite score of the player to drop.
        add_name:       Display name of the waiver-wire target.
        add_player_id:  MLBAM id of the waiver-wire target.
        add_score:      Composite score of the waiver-wire target.
        score_delta:    ``add_score - drop_score``; always ≥ :data:`_SWAP_THRESHOLD`.
    """

    drop_name: str
    drop_player_id: int
    drop_score: float
    add_name: str
    add_player_id: int
    add_score: float
    score_delta: float

    def __str__(self) -> str:
        return (
            f"SWAP RECOMMENDED  |  "
            f"DROP {self.drop_name} ({self.drop_score:.1f})  →  "
            f"ADD {self.add_name} ({self.add_score:.1f})  "
            f"[+{self.score_delta:.1f} pts]"
        )


@dataclass
class CompareResult:
    """Output of :meth:`Recommender.compare_to_roster`.

    Attributes:
        position:          The position queried (e.g. ``"OF"`` or ``"SP"``).
        player_type:       ``"batter"`` or ``"pitcher"``.
        available_players: Top available players from the waiver pool, scored.
        my_players:        My rostered players at *position*, scored (ascending
                           so the weakest candidate appears first).
        swap:              A :class:`SwapRecommendation` when the best available
                           player outscores my weakest by at least
                           :data:`_SWAP_THRESHOLD`, otherwise ``None``.
    """

    position: str
    player_type: str
    available_players: pd.DataFrame
    my_players: pd.DataFrame
    swap: Optional[SwapRecommendation]

    def has_swap(self) -> bool:
        """Return ``True`` when a swap is recommended."""
        return self.swap is not None

    def __str__(self) -> str:
        lines = [
            f"Position: {self.position}  ({self.player_type})",
            "",
            f"Top available {self.player_type}s:",
            self.available_players[
                ["player_name", "composite_score"]
            ].to_string(index=False),
            "",
            f"My {self.position} players:",
            self.my_players[
                ["player_name", "composite_score"]
            ].to_string(index=False)
            if not self.my_players.empty
            else "  (none)",
            "",
            str(self.swap) if self.swap else "No swap recommended.",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class Recommender:
    """Orchestrates data fetching, scoring, and roster filtering.

    Args:
        client:  Object satisfying :class:`StatcastClientProtocol`.  In
                 production pass the module-level functions from
                 ``data.statcast_client`` wrapped in a thin adapter, or the
                 module itself if its functions are bound methods.
        scorer:  Object satisfying :class:`ScorerProtocol` — typically a
                 :class:`~analysis.metrics.PlayerScorer` instance.
        roster:  Object satisfying :class:`RosterProtocol` — typically a
                 :class:`~roster.manager.RosterManager` instance.

    Example::

        import data.statcast_client as client
        from analysis.metrics import PlayerScorer
        from roster.manager import RosterManager
        from recommender import Recommender

        rec = Recommender(client, PlayerScorer(), RosterManager())
        print(rec.recommend_pickups(position="SP", top_n=5))
    """

    def __init__(
        self,
        client: StatcastClientProtocol,
        scorer: ScorerProtocol,
        roster: RosterProtocol,
    ) -> None:
        self._client = client
        self._scorer = scorer
        self._roster = roster

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend_pickups(
        self,
        position: Optional[str] = None,
        top_n: int = 10,
        days: int = 14,
    ) -> pd.DataFrame:
        """Return the top available waiver-wire targets.

        Scores the full player pool for the appropriate player type, then
        filters to players not on any roster, and returns the top *top_n*
        results.

        Args:
            position: A position string from ``C 1B 2B 3B SS OF SP RP``.
                      Determines whether batter or pitcher data is used.
                      ``None`` defaults to the batter pool.
            top_n:    Maximum number of players to return (default 10).
            days:     Trailing days of Statcast data to fetch (default 14).

        Returns:
            DataFrame with columns ``player_name``, ``player_id``,
            ``composite_score``, and the normalized per-metric score columns
            for the appropriate player type.  Sorted by ``composite_score``
            descending.

        Raises:
            ValueError: When *position* is not a recognised position string.
        """
        if position is not None and position not in ALL_POSITIONS:
            raise ValueError(
                f"'{position}' is not a valid position. "
                f"Choose from: {sorted(ALL_POSITIONS)}"
            )

        ptype = _player_type(position)
        logger.info("recommend_pickups: position=%s type=%s top_n=%d", position, ptype, top_n)

        scored_all = self._fetch_and_score_all(ptype, days)
        available = self._roster.get_available_players(scored_all)

        logger.info(
            "recommend_pickups: %d scored → %d available", len(scored_all), len(available)
        )

        result = available.head(top_n)
        return _select_display_cols(result, ptype)

    def compare_to_roster(
        self,
        position: str,
        top_n: int = 5,
        days: int = 14,
    ) -> CompareResult:
        """Compare the top available players at *position* to my weakest at that spot.

        Scores the full pool so that all composite scores are on a common
        scale, then splits into "my players at this position" and "available
        players", and checks whether a swap is warranted.

        A :class:`SwapRecommendation` is generated when the best available
        player's score exceeds my weakest player's score by at least
        :data:`_SWAP_THRESHOLD` (10 points).

        Args:
            position: A position string from ``C 1B 2B 3B SS OF SP RP``.
            top_n:    Number of available targets to surface (default 5).
            days:     Trailing days of Statcast data to fetch (default 14).

        Returns:
            A :class:`CompareResult` with ``available_players``,
            ``my_players``, and an optional ``swap``.

        Raises:
            ValueError: When *position* is not a recognised position string.
        """
        if position not in ALL_POSITIONS:
            raise ValueError(
                f"'{position}' is not a valid position. "
                f"Choose from: {sorted(ALL_POSITIONS)}"
            )

        ptype = _player_type(position)
        logger.info("compare_to_roster: position=%s type=%s", position, ptype)

        # Score everyone together so ranks are relative to the full pool.
        scored_all = self._fetch_and_score_all(ptype, days)

        # Split into mine (at this position) and available.
        mine = _my_players_at_position(scored_all, self._roster, position)
        available_all = self._roster.get_available_players(scored_all)

        top_available = _select_display_cols(available_all.head(top_n), ptype)
        my_display = _select_display_cols(
            mine.sort_values("composite_score", ascending=True), ptype
        )

        swap = _evaluate_swap(top_available, my_display)

        logger.info(
            "compare_to_roster: %d available, %d mine at %s, swap=%s",
            len(top_available),
            len(my_display),
            position,
            swap is not None,
        )

        return CompareResult(
            position=position,
            player_type=ptype,
            available_players=top_available.reset_index(drop=True),
            my_players=my_display.reset_index(drop=True),
            swap=swap,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_and_score_all(self, player_type: str, days: int) -> pd.DataFrame:
        """Fetch percentile-rank data and score the full player pool.

        Args:
            player_type: ``"batter"`` or ``"pitcher"``.
            days:        Trailing calendar days to request from the client.

        Returns:
            Scored DataFrame for the full player pool, sorted by
            ``composite_score`` descending.
        """
        if player_type == "batter":
            data = self._client.fetch_batting_data(days=days)
            pct_df = data["percentile_ranks"]
            return self._scorer.score_batters(pct_df)

        data = self._client.fetch_pitching_data(days=days)
        pct_df = data["percentile_ranks"]
        return self._scorer.score_pitchers(pct_df)


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions — easy to unit-test in isolation)
# ---------------------------------------------------------------------------

def _player_type(position: Optional[str]) -> str:
    """Map a position string (or ``None``) to ``"batter"`` or ``"pitcher"``.

    ``None`` and all batter positions → ``"batter"``.
    Pitcher positions → ``"pitcher"``.
    """
    if position is None or position in BATTER_POSITIONS:
        return "batter"
    return "pitcher"


def _select_display_cols(df: pd.DataFrame, player_type: str) -> pd.DataFrame:
    """Trim *df* to the display columns for *player_type*.

    Only includes columns that are actually present in *df* so callers are
    not broken when optional metrics are absent.

    Args:
        df:          Scored player DataFrame.
        player_type: ``"batter"`` or ``"pitcher"``.

    Returns:
        DataFrame containing only the relevant display columns.
    """
    wanted = _BATTER_DISPLAY_COLS if player_type == "batter" else _PITCHER_DISPLAY_COLS
    present = [c for c in wanted if c in df.columns]
    return df[present].copy()


def _my_players_at_position(
    scored_all: pd.DataFrame,
    roster: RosterProtocol,
    position: str,
) -> pd.DataFrame:
    """Return rows in *scored_all* belonging to my roster at *position*.

    Matches by ``player_id`` only — requires a ``player_id`` column in both
    *scored_all* and the roster's player objects.

    Args:
        scored_all: Full scored player pool.
        roster:     Object satisfying :class:`RosterProtocol`.
        position:   Position string to filter by.

    Returns:
        Subset of *scored_all* for my players with eligibility at *position*.
        Empty DataFrame (with same columns) when no matches are found.
    """
    eligible_ids = {
        p.player_id
        for p in roster.my_roster
        if position in getattr(p, "positions", [])
    }

    if not eligible_ids or "player_id" not in scored_all.columns:
        return scored_all.iloc[0:0].copy()  # typed empty

    return scored_all[scored_all["player_id"].isin(eligible_ids)].copy()


def _evaluate_swap(
    available: pd.DataFrame,
    my_players: pd.DataFrame,
) -> Optional[SwapRecommendation]:
    """Check whether the best available player warrants dropping my weakest.

    Args:
        available:  Top available players sorted by ``composite_score`` desc.
        my_players: My roster players sorted by ``composite_score`` asc
                    (weakest first).

    Returns:
        A :class:`SwapRecommendation` when the score delta exceeds
        :data:`_SWAP_THRESHOLD`, otherwise ``None``.
    """
    if available.empty or my_players.empty:
        return None

    best_available = available.iloc[0]
    weakest_mine = my_players.iloc[0]

    delta = float(best_available["composite_score"]) - float(weakest_mine["composite_score"])
    if delta < _SWAP_THRESHOLD:
        return None

    return SwapRecommendation(
        drop_name=str(weakest_mine.get("player_name", "Unknown")),
        drop_player_id=int(weakest_mine.get("player_id", -1)),
        drop_score=float(weakest_mine["composite_score"]),
        add_name=str(best_available.get("player_name", "Unknown")),
        add_player_id=int(best_available.get("player_id", -1)),
        add_score=float(best_available["composite_score"]),
        score_delta=round(delta, 2),
    )
