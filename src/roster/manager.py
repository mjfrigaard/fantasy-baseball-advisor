"""Fantasy roster manager.

Loads the user's roster and the league-wide unavailable player list from
YAML config files, then exposes filtering helpers that operate on pybaseball
percentile-rank DataFrames.

Player matching uses MLBAM ``player_id`` as the primary key (integer,
unambiguous across name changes and accented characters) with a normalised
case-insensitive name comparison as a fallback for rows that lack an id.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Position constants
# ---------------------------------------------------------------------------

BATTER_POSITIONS: frozenset[str] = frozenset({"C", "1B", "2B", "3B", "SS", "OF"})
PITCHER_POSITIONS: frozenset[str] = frozenset({"SP", "RP"})
ALL_POSITIONS: frozenset[str] = BATTER_POSITIONS | PITCHER_POSITIONS

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RosterPlayer:
    """A single player entry parsed from a YAML roster file.

    Attributes:
        name:        Display name in pybaseball's ``"Last, First"`` format.
        player_id:   MLBAM integer ID — primary match key.
        positions:   List of position eligibility strings (e.g. ``["SS", "OF"]``).
        player_type: ``"batter"`` or ``"pitcher"``.
    """

    name: str
    player_id: int
    positions: list[str] = field(default_factory=list)
    player_type: str = "batter"

    def __post_init__(self) -> None:
        invalid = [p for p in self.positions if p not in ALL_POSITIONS]
        if invalid:
            raise ValueError(
                f"Player '{self.name}' has unrecognised positions: {invalid}. "
                f"Valid positions: {sorted(ALL_POSITIONS)}"
            )
        if self.player_type not in ("batter", "pitcher"):
            raise ValueError(
                f"player_type must be 'batter' or 'pitcher', got '{self.player_type}'"
            )


# ---------------------------------------------------------------------------
# RosterManager
# ---------------------------------------------------------------------------

class RosterManager:
    """Load and query fantasy roster configuration.

    Reads two YAML files:

    * **my_roster.yaml** — the user's active roster, split into ``batters``
      and ``pitchers`` sections.
    * **unavailable_players.yaml** — flat list of players already claimed by
      other teams in the league.

    The union of both lists defines the set of *unavailable* players.
    :meth:`get_available_players` filters any pybaseball percentile-rank
    DataFrame down to the unclaimed waiver pool.

    Args:
        roster_path:      Path to ``my_roster.yaml``.  Defaults to
                          ``config/my_roster.yaml`` relative to the project
                          root.
        unavailable_path: Path to ``unavailable_players.yaml``.  Defaults to
                          ``config/unavailable_players.yaml``.

    Raises:
        FileNotFoundError: When either config file does not exist.
        ValueError:        When a player entry contains an invalid position or
                           is missing required fields.

    Example::

        mgr = RosterManager()
        available = mgr.get_available_players(batter_pct_df)
        sp_targets = mgr.filter_by_position(available_pitcher_df, "SP")
    """

    def __init__(
        self,
        roster_path: Optional[Path] = None,
        unavailable_path: Optional[Path] = None,
    ) -> None:
        self._roster_path = roster_path or _DEFAULT_CONFIG_DIR / "my_roster.yaml"
        self._unavailable_path = (
            unavailable_path or _DEFAULT_CONFIG_DIR / "unavailable_players.yaml"
        )
        self.my_roster: list[RosterPlayer] = self._load_my_roster()
        self._unavailable_raw: list[dict] = self._load_unavailable()
        logger.info(
            "RosterManager loaded: %d my players, %d league-unavailable",
            len(self.my_roster),
            len(self._unavailable_raw),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def my_player_ids(self) -> frozenset[int]:
        """MLBAM IDs of players on my roster."""
        return frozenset(p.player_id for p in self.my_roster)

    @property
    def unavailable_ids(self) -> frozenset[int]:
        """MLBAM IDs of all players on any roster (mine + other teams)."""
        other_ids = {int(p["player_id"]) for p in self._unavailable_raw if p.get("player_id")}
        return self.my_player_ids | other_ids

    @property
    def unavailable_names(self) -> frozenset[str]:
        """Normalised lowercase names of all rostered players (fallback match key)."""
        my_names = {_normalise_name(p.name) for p in self.my_roster}
        other_names = {
            _normalise_name(p["name"])
            for p in self._unavailable_raw
            if p.get("name")
        }
        return my_names | other_names

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_available_players(self, all_players_df: pd.DataFrame) -> pd.DataFrame:
        """Return players in *all_players_df* who are not on any roster.

        Matching strategy:

        1. If the DataFrame has a ``player_id`` column, rows whose id appears
           in :attr:`unavailable_ids` are excluded.
        2. For remaining rows without a numeric ``player_id`` (or when the
           column is absent), a normalised name comparison against
           :attr:`unavailable_names` is used as a fallback.

        Args:
            all_players_df: Any pybaseball percentile-rank DataFrame containing
                            at least a ``player_name`` column.

        Returns:
            Filtered DataFrame with the same columns as the input, index reset.

        Raises:
            ValueError: When *all_players_df* is empty.
        """
        if all_players_df.empty:
            raise ValueError("all_players_df is empty — nothing to filter.")

        df = all_players_df.copy()
        unavail_ids = self.unavailable_ids
        unavail_names = self.unavailable_names

        if "player_id" in df.columns:
            id_mask = df["player_id"].isin(unavail_ids)
        else:
            id_mask = pd.Series(False, index=df.index)

        if "player_name" in df.columns:
            name_mask = df["player_name"].apply(
                lambda n: _normalise_name(str(n)) in unavail_names
            )
        else:
            name_mask = pd.Series(False, index=df.index)

        excluded = id_mask | name_mask
        available = df[~excluded].reset_index(drop=True)

        logger.info(
            "get_available_players: %d total → %d excluded → %d available",
            len(df),
            excluded.sum(),
            len(available),
        )
        return available

    def filter_by_position(
        self, players_df: pd.DataFrame, position: str
    ) -> pd.DataFrame:
        """Filter *players_df* to players eligible at *position*.

        This works by intersecting the requested position against each
        player's declared eligibility in the YAML configs.  Players whose
        id or name does not appear in the loaded configs are **kept** — the
        absence of config data is not a disqualifier.

        Args:
            players_df: DataFrame with a ``player_id`` or ``player_name``
                        column; typically the output of
                        :meth:`get_available_players`.
            position:   One of ``C 1B 2B 3B SS OF SP RP``.

        Returns:
            Filtered DataFrame, index reset.

        Raises:
            ValueError: When *position* is not a recognised position string.
        """
        if position not in ALL_POSITIONS:
            raise ValueError(
                f"'{position}' is not a valid position. "
                f"Choose from: {sorted(ALL_POSITIONS)}"
            )

        eligible_ids = {
            p.player_id for p in self.my_roster if position in p.positions
        }

        if "player_id" in players_df.columns:
            mask = players_df["player_id"].isin(eligible_ids)
            return players_df[mask].reset_index(drop=True)

        # name fallback
        eligible_names = {
            _normalise_name(p.name) for p in self.my_roster if position in p.positions
        }
        if "player_name" in players_df.columns:
            mask = players_df["player_name"].apply(
                lambda n: _normalise_name(str(n)) in eligible_names
            )
            return players_df[mask].reset_index(drop=True)

        logger.warning(
            "filter_by_position: DataFrame has neither player_id nor player_name; "
            "returning full DataFrame unfiltered."
        )
        return players_df.reset_index(drop=True)

    def get_roster_df(self) -> pd.DataFrame:
        """Return my current roster as a tidy DataFrame.

        Returns:
            DataFrame with columns ``player_name``, ``player_id``,
            ``positions``, ``player_type``; one row per player.
        """
        return pd.DataFrame(
            [
                {
                    "player_name": p.name,
                    "player_id": p.player_id,
                    "positions": ", ".join(p.positions),
                    "player_type": p.player_type,
                }
                for p in self.my_roster
            ]
        )

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_my_roster(self) -> list[RosterPlayer]:
        """Parse my_roster.yaml into a list of :class:`RosterPlayer`."""
        raw = _load_yaml(self._roster_path)

        players: list[RosterPlayer] = []
        for player_type, section_key in (("batter", "batters"), ("pitcher", "pitchers")):
            for entry in raw.get(section_key, []):
                players.append(
                    RosterPlayer(
                        name=_require_str(entry, "name", self._roster_path),
                        player_id=_require_int(entry, "player_id", self._roster_path),
                        positions=[str(p) for p in entry.get("positions", [])],
                        player_type=player_type,
                    )
                )

        logger.debug("Loaded %d players from %s", len(players), self._roster_path.name)
        return players

    def _load_unavailable(self) -> list[dict]:
        """Parse unavailable_players.yaml into a raw list of dicts."""
        raw = _load_yaml(self._unavailable_path)
        entries = raw.get("unavailable", [])
        logger.debug(
            "Loaded %d unavailable entries from %s",
            len(entries),
            self._unavailable_path.name,
        )
        return entries


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    """Read and parse a YAML file, raising clear errors on failure."""
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Check that the path is correct relative to the project root."
        )
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _require_str(entry: dict, key: str, source: Path) -> str:
    """Extract a required string field from a YAML entry dict."""
    val = entry.get(key)
    if not isinstance(val, str) or not val.strip():
        raise ValueError(
            f"Missing or invalid '{key}' in entry {entry!r} in {source.name}"
        )
    return val.strip()


def _require_int(entry: dict, key: str, source: Path) -> int:
    """Extract a required integer field from a YAML entry dict."""
    val = entry.get(key)
    if val is None:
        raise ValueError(
            f"Missing '{key}' in entry {entry!r} in {source.name}"
        )
    try:
        return int(val)
    except (TypeError, ValueError):
        raise ValueError(
            f"'{key}' must be an integer in entry {entry!r} in {source.name}"
        )


def _normalise_name(name: str) -> str:
    """Lowercase and strip a player name for fuzzy comparison."""
    return name.strip().lower()
