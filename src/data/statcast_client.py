"""Statcast data client with local parquet caching.

Fetches play-level and percentile-rank data from Baseball Savant via
pybaseball. Results are cached under data/cache/ using a daily timestamp
so that repeated calls within the same day skip the network entirely.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import pybaseball

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIR = _PROJECT_ROOT / "data" / "cache"

# pybaseball prints progress bars by default; silence them.
pybaseball.cache.enable()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(name: str, stamp: str) -> Path:
    """Return the parquet path for a given dataset name and date stamp.

    Args:
        name:  Short identifier, e.g. ``"statcast_raw"`` or ``"batter_pct"``.
        stamp: Date string used as part of the filename (``YYYY-MM-DD`` or
               ``YYYY`` for year-level datasets).

    Returns:
        Absolute ``Path`` inside ``data/cache/``.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{name}_{stamp}.parquet"


def _is_fresh(path: Path, max_age_hours: int = 24) -> bool:
    """Return ``True`` if *path* exists and was modified within *max_age_hours*.

    Args:
        path:          File to check.
        max_age_hours: Staleness threshold (default 24 hours).

    Returns:
        ``True`` when the cache file is present and still fresh.
    """
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=max_age_hours)


def load_cache(path: Path) -> Optional[pd.DataFrame]:
    """Load a parquet cache file if it exists and is less than 24 hours old.

    Args:
        path: Path returned by :func:`_cache_path`.

    Returns:
        A ``DataFrame`` when a fresh cache hit is found, otherwise ``None``.
    """
    if _is_fresh(path):
        logger.info("Cache hit: %s", path.name)
        return pd.read_parquet(path)
    logger.debug("Cache miss: %s", path.name)
    return None


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    """Write *df* to *path* as a parquet file.

    Args:
        df:   DataFrame to persist.
        path: Destination path (parent directory must exist).
    """
    df.to_parquet(path, index=False)
    logger.info("Cached %d rows → %s", len(df), path.name)


# ---------------------------------------------------------------------------
# Raw Statcast (play-level, covers both batters and pitchers)
# ---------------------------------------------------------------------------

def _fetch_statcast_raw(start_dt: str, end_dt: str) -> pd.DataFrame:
    """Pull play-level Statcast data from Baseball Savant.

    This is the shared backing store for both the batting and pitching
    client functions.  Results are cached at a per-day granularity keyed
    on the requested date range.

    Args:
        start_dt: First date in ``YYYY-MM-DD`` format.
        end_dt:   Last date in ``YYYY-MM-DD`` format.

    Returns:
        DataFrame of raw Statcast events.

    Raises:
        RuntimeError: When the Baseball Savant request fails.
    """
    stamp = f"{start_dt}_to_{end_dt}"
    path = _cache_path("statcast_raw", stamp)

    cached = load_cache(path)
    if cached is not None:
        return cached

    logger.info("Fetching raw Statcast data %s → %s", start_dt, end_dt)
    try:
        df = pybaseball.statcast(start_dt=start_dt, end_dt=end_dt, verbose=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch Statcast play data ({start_dt} – {end_dt}): {exc}"
        ) from exc

    if df is None or df.empty:
        logger.warning("Baseball Savant returned no play data for %s – %s", start_dt, end_dt)
        return pd.DataFrame()

    _save_cache(df, path)
    return df


# ---------------------------------------------------------------------------
# Batting
# ---------------------------------------------------------------------------

def fetch_batting_data(days: int = 14) -> dict[str, pd.DataFrame]:
    """Fetch recent Statcast batting data and current-season percentile ranks.

    Two sources are combined:

    * **play-level** — every pitch/event for the last *days* calendar days,
      fetched via :func:`pybaseball.statcast` and filtered to batting events.
    * **percentile ranks** — season-to-date batter percentile ranks from
      :func:`pybaseball.statcast_batter_percentile_ranks`.

    Both are cached to ``data/cache/`` with a daily timestamp.

    Args:
        days: Number of trailing calendar days to include (default 14).

    Returns:
        A dict with two keys:

        ``"play_log"``
            DataFrame of pitch/play events for the window, restricted to
            rows where ``events`` is non-null (i.e. plate appearances).
        ``"percentile_ranks"``
            DataFrame of season batter percentile ranks.

    Raises:
        RuntimeError: On network or API failure.
    """
    today = date.today()
    start_dt = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    end_dt = today.strftime("%Y-%m-%d")
    year = today.year

    # --- play-level data ---
    raw = _fetch_statcast_raw(start_dt, end_dt)
    # Keep only rows that represent completed plate appearances
    play_log = (
        raw[raw["events"].notna()].copy() if not raw.empty else pd.DataFrame()
    )
    logger.info("Batting play log: %d plate appearances", len(play_log))

    # --- percentile ranks ---
    pct_path = _cache_path("batter_percentile", str(year))
    pct_df = load_cache(pct_path)
    if pct_df is None:
        logger.info("Fetching batter percentile ranks for %d", year)
        try:
            pct_df = pybaseball.statcast_batter_percentile_ranks(year)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch batter percentile ranks for {year}: {exc}"
            ) from exc

        if pct_df is None or pct_df.empty:
            logger.warning("No batter percentile rank data returned for %d", year)
            pct_df = pd.DataFrame()
        else:
            _save_cache(pct_df, pct_path)

    return {"play_log": play_log, "percentile_ranks": pct_df}


# ---------------------------------------------------------------------------
# Pitching
# ---------------------------------------------------------------------------

def fetch_pitching_data(days: int = 14) -> dict[str, pd.DataFrame]:
    """Fetch recent Statcast pitching data and current-season percentile ranks.

    Two sources are combined:

    * **pitch-level** — every pitch thrown over the last *days* calendar
      days, fetched via :func:`pybaseball.statcast`.  Unlike the batting
      view, all rows are retained because every row is a pitch event.
    * **percentile ranks** — season-to-date pitcher percentile ranks from
      :func:`pybaseball.statcast_pitcher_percentile_ranks`.

    Both are cached to ``data/cache/`` with a daily timestamp.

    Args:
        days: Number of trailing calendar days to include (default 14).

    Returns:
        A dict with two keys:

        ``"pitch_log"``
            DataFrame of every pitch thrown in the requested window.
        ``"percentile_ranks"``
            DataFrame of season pitcher percentile ranks.

    Raises:
        RuntimeError: On network or API failure.
    """
    today = date.today()
    start_dt = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    end_dt = today.strftime("%Y-%m-%d")
    year = today.year

    # --- pitch-level data (all rows = all pitches) ---
    pitch_log = _fetch_statcast_raw(start_dt, end_dt)
    logger.info("Pitching pitch log: %d pitches", len(pitch_log))

    # --- percentile ranks ---
    pct_path = _cache_path("pitcher_percentile", str(year))
    pct_df = load_cache(pct_path)
    if pct_df is None:
        logger.info("Fetching pitcher percentile ranks for %d", year)
        try:
            pct_df = pybaseball.statcast_pitcher_percentile_ranks(year)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch pitcher percentile ranks for {year}: {exc}"
            ) from exc

        if pct_df is None or pct_df.empty:
            logger.warning("No pitcher percentile rank data returned for %d", year)
            pct_df = pd.DataFrame()
        else:
            _save_cache(pct_df, pct_path)

    return {"pitch_log": pitch_log, "percentile_ranks": pct_df}
