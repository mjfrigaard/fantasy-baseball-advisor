"""Player scoring engine using Statcast percentile-rank data.

Input DataFrames come from pybaseball's ``statcast_batter_percentile_ranks``
and ``statcast_pitcher_percentile_ranks``.  Those functions already express
every metric as a percentile rank in [1, 100] where **higher always means
better for the player** — including metrics that are nominally "bad" when
high (xERA, barrel rate allowed, k_percent), which Baseball Savant inverts
before ranking.

This module re-normalizes each metric within the supplied player pool so
that composite scores are relative to whoever is being evaluated, not the
full MLB season cohort.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Column names as returned by pybaseball percentile-rank endpoints.
_BATTER_ID_COLS = ["player_name", "player_id", "year"]
_PITCHER_ID_COLS = ["player_name", "player_id", "year"]


class PlayerScorer:
    """Score batters and pitchers using weighted Statcast percentile metrics.

    All input columns are assumed to be pybaseball percentile ranks in
    [1, 100] where higher is better.  Each metric is re-percentile-ranked
    within the supplied pool before weights are applied, producing final
    composite scores in [0, 100].

    Args:
        batter_weights:  Mapping of ``{column_name: weight}`` for batter
            scoring.  Weights must sum to 1.0.  Defaults to the class-level
            ``BATTER_WEIGHTS``.
        pitcher_weights: Same as above for pitchers.  Defaults to
            ``PITCHER_WEIGHTS``.

    Example::

        scorer = PlayerScorer()
        batter_scores = scorer.score_batters(batter_pct_df)
        pitcher_scores = scorer.score_pitchers(pitcher_pct_df)
    """

    BATTER_WEIGHTS: dict[str, float] = {
        "brl_percent": 0.30,       # barrel rate
        "hard_hit_percent": 0.25,  # hard-hit rate (95+ mph)
        "xwoba": 0.25,             # expected wOBA
        "sprint_speed": 0.10,      # sprint speed
        "k_percent": 0.10,         # strikeout rate (already inverted by Savant)
    }

    PITCHER_WEIGHTS: dict[str, float] = {
        "xera": 0.30,              # expected ERA (already inverted by Savant)
        "whiff_percent": 0.25,     # whiff rate
        "chase_percent": 0.15,     # chase rate
        "brl_percent": 0.20,       # barrel rate allowed (already inverted)
        "hard_hit_percent": 0.10,  # hard-hit rate allowed (already inverted)
    }

    def __init__(
        self,
        batter_weights: Optional[dict[str, float]] = None,
        pitcher_weights: Optional[dict[str, float]] = None,
    ) -> None:
        self._bw = batter_weights if batter_weights is not None else self.BATTER_WEIGHTS
        self._pw = pitcher_weights if pitcher_weights is not None else self.PITCHER_WEIGHTS
        self._validate_weights(self._bw, "batter")
        self._validate_weights(self._pw, "pitcher")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def score_batters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute composite batter scores for all players in *df*.

        Missing metric columns are silently skipped and their weight
        redistributed proportionally across the remaining columns.

        Args:
            df: DataFrame from ``statcast_batter_percentile_ranks``.  Must
                contain at least ``player_name`` and one metric column.

        Returns:
            DataFrame with columns ``player_name``, ``player_id``
            (when present), ``year`` (when present), one normalized column
            per metric, and a ``composite_score`` column; sorted by
            ``composite_score`` descending.

        Raises:
            ValueError: When *df* is empty or contains no recognised metric
                columns.
        """
        return self._score(df, self._bw, player_type="batter")

    def score_pitchers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute composite pitcher scores for all players in *df*.

        Args:
            df: DataFrame from ``statcast_pitcher_percentile_ranks``.  Must
                contain at least ``player_name`` and one metric column.

        Returns:
            Same structure as :meth:`score_batters` but using pitcher
            weights; sorted by ``composite_score`` descending.

        Raises:
            ValueError: When *df* is empty or contains no recognised metric
                columns.
        """
        return self._score(df, self._pw, player_type="pitcher")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_weights(weights: dict[str, float], label: str) -> None:
        """Raise ``ValueError`` if weights don't sum to approximately 1.0."""
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"{label} weights sum to {total:.4f}; they must sum to 1.0"
            )

    @staticmethod
    def _normalize(series: pd.Series) -> pd.Series:
        """Re-rank a metric series to [0, 100] within the current player pool.

        Uses fractional rank (average method) so ties receive the mean of
        the ranks they span.  NaN values remain NaN and are excluded from
        the ranking computation.

        Args:
            series: Raw or pre-computed percentile values for one metric.

        Returns:
            Series of float values in [0, 100].
        """
        return series.rank(pct=True, method="average", na_option="keep") * 100

    def _score(
        self,
        df: pd.DataFrame,
        weights: dict[str, float],
        player_type: str,
    ) -> pd.DataFrame:
        """Core scoring routine shared by batters and pitchers.

        Args:
            df:          Input percentile-rank DataFrame.
            weights:     ``{column: weight}`` mapping for this player type.
            player_type: ``"batter"`` or ``"pitcher"`` (used in log messages).

        Returns:
            Scored and sorted DataFrame.
        """
        if df.empty:
            raise ValueError(f"Input DataFrame for {player_type}s is empty.")

        available = {col: w for col, w in weights.items() if col in df.columns}
        missing = set(weights) - set(available)
        if missing:
            logger.warning(
                "%s scoring: columns not found and skipped — %s",
                player_type,
                sorted(missing),
            )
        if not available:
            raise ValueError(
                f"No recognised {player_type} metric columns found in DataFrame. "
                f"Expected one of: {sorted(weights)}"
            )

        # Redistribute weight if any columns were absent.
        weight_total = sum(available.values())
        rescaled = {col: w / weight_total for col, w in available.items()}

        # Build output starting from identity columns that exist in df.
        id_cols = [c for c in ["player_name", "player_id", "year"] if c in df.columns]
        out = df[id_cols].copy()

        composite = pd.Series(0.0, index=df.index)
        for col, weight in rescaled.items():
            norm_col = f"{col}_score"
            normalized = self._normalize(df[col])
            out[norm_col] = normalized.round(2)

            # For rows where the metric is NaN, contribute 0 to composite.
            composite += normalized.fillna(0.0) * weight

        out["composite_score"] = composite.round(2)
        out = out.sort_values("composite_score", ascending=False).reset_index(drop=True)

        logger.info(
            "Scored %d %ss | top: %s (%.1f) | bottom: %s (%.1f)",
            len(out),
            player_type,
            out["player_name"].iloc[0] if "player_name" in out.columns else "?",
            out["composite_score"].iloc[0],
            out["player_name"].iloc[-1] if "player_name" in out.columns else "?",
            out["composite_score"].iloc[-1],
        )
        return out
