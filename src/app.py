"""Fantasy Baseball Advisor — Streamlit web application.

Run from the project root:
    streamlit run src/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when launched from the project root.
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

# Must be the first Streamlit call.
st.set_page_config(
    page_title="Fantasy Baseball Advisor",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

import logging
from datetime import datetime

import pandas as pd
import yaml

import data.statcast_client as _sc
from analysis.metrics import PlayerScorer
from roster.manager import BATTER_POSITIONS, PITCHER_POSITIONS, RosterManager
from recommender import Recommender

logging.basicConfig(level=logging.WARNING)

# ── Display configuration ─────────────────────────────────────────────────────

_BATTER_LABELS: dict[str, str] = {
    "player_name": "Player",
    "composite_score": "Score",
    "brl_percent_score": "Barrel %",
    "hard_hit_percent_score": "Hard Hit %",
    "xwoba_score": "xwOBA",
    "sprint_speed_score": "Sprint Speed",
    "k_percent_score": "K %",
}
_PITCHER_LABELS: dict[str, str] = {
    "player_name": "Player",
    "composite_score": "Score",
    "xera_score": "xERA",
    "whiff_percent_score": "Whiff %",
    "chase_percent_score": "Chase %",
    "brl_percent_score": "Barrel %",
    "hard_hit_percent_score": "Hard Hit %",
}

_ROSTER_YAML = Path(__file__).parent.parent / "config" / "my_roster.yaml"


# ── Cached data (6-hour TTL) ──────────────────────────────────────────────────

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _load_batter_pct(days: int = 14) -> pd.DataFrame:
    """Fetch batter percentile-rank data; cached for 6 hours."""
    try:
        return _sc.fetch_batting_data(days=days)["percentile_ranks"]
    except Exception as exc:
        raise RuntimeError(f"Could not load batter data from Baseball Savant: {exc}") from exc


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _load_pitcher_pct(days: int = 14) -> pd.DataFrame:
    """Fetch pitcher percentile-rank data; cached for 6 hours."""
    try:
        return _sc.fetch_pitching_data(days=days)["percentile_ranks"]
    except Exception as exc:
        raise RuntimeError(f"Could not load pitcher data from Baseball Savant: {exc}") from exc


# ── Cached resources (singleton per session) ──────────────────────────────────

@st.cache_resource
def _get_scorer() -> PlayerScorer:
    return PlayerScorer()


@st.cache_resource
def _get_roster() -> RosterManager:
    return RosterManager()


@st.cache_resource
def _get_recommender() -> Recommender:
    """Build the Recommender backed by the Streamlit-cached data functions."""

    class _AppClient:
        def fetch_batting_data(self, days: int = 14) -> dict:
            return {"percentile_ranks": _load_batter_pct(days), "play_log": pd.DataFrame()}

        def fetch_pitching_data(self, days: int = 14) -> dict:
            return {"percentile_ranks": _load_pitcher_pct(days), "pitch_log": pd.DataFrame()}

    return Recommender(client=_AppClient(), scorer=_get_scorer(), roster=_get_roster())


# ── Display helpers ───────────────────────────────────────────────────────────

def _for_display(df: pd.DataFrame, ptype: str) -> pd.DataFrame:
    """Trim to relevant columns and apply human-readable labels."""
    labels = _BATTER_LABELS if ptype == "batter" else _PITCHER_LABELS
    keep = {k: v for k, v in labels.items() if k in df.columns}
    return df[list(keep.keys())].rename(columns=keep)


def _add_rank(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "#", range(1, len(out) + 1))
    return out


def _style(
    df: pd.DataFrame,
    highlight_above: float | None = None,
) -> pd.io.formats.style.Styler:
    """Green gradient on Score column; optionally highlight swap-candidate rows."""
    styler = df.style.format({"Score": "{:.1f}"}, na_rep="—")

    if "Score" in df.columns:
        styler = styler.background_gradient(
            subset=["Score"], cmap="YlGn", vmin=0, vmax=100
        )

    if highlight_above is not None and "Score" in df.columns:
        def _row_style(row: pd.Series) -> list[str]:
            try:
                if float(row["Score"]) >= highlight_above:
                    return ["background-color:#c3e6cb; font-weight:bold"] * len(row)
            except (ValueError, TypeError):
                pass
            return [""] * len(row)

        styler = styler.apply(_row_style, axis=1)

    return styler


def _effective_pitcher_pos(pos_filter: str | None) -> str:
    """Return a pitcher-pool position when the user has selected All or a specific pitcher slot."""
    return pos_filter if pos_filter in PITCHER_POSITIONS else "SP"


# ── Tab 1: Top Pickups ────────────────────────────────────────────────────────

def _tab_top_pickups(
    recommender: Recommender,
    player_type: str,
    pos_filter: str | None,
    top_n: int,
) -> None:
    st.subheader("Top Waiver Wire Pickups")
    st.caption(
        "Composite scores are re-percentile-ranked within the full MLB pool for the current season. "
        "Position eligibility in your league platform should be verified before adding any player."
    )

    def _show(ptype: str, position: str | None) -> None:
        label = "Batters" if ptype == "batter" else "Pitchers"
        try:
            with st.spinner(f"Scoring available {label.lower()}…"):
                recs = recommender.recommend_pickups(position=position, top_n=top_n)
        except RuntimeError as exc:
            st.error(f"Failed to load {label.lower()}: {exc}")
            return

        if recs.empty:
            st.info(f"No available {label.lower()} found.")
            return

        display = _add_rank(_for_display(recs, ptype))
        st.dataframe(_style(display), use_container_width=True, hide_index=True)

        st.download_button(
            label="📥 Download as CSV",
            data=recs.to_csv(index=False),
            file_name=f"pickups_{label.lower()}_{datetime.now():%Y%m%d}.csv",
            mime="text/csv",
            key=f"dl_{ptype}",
        )

    if player_type == "Both":
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Batters")
            _show("batter", None)
        with c2:
            st.markdown("#### Pitchers")
            _show("pitcher", "SP")
    elif player_type == "Pitchers":
        _show("pitcher", _effective_pitcher_pos(pos_filter))
    else:
        _show("batter", pos_filter)


# ── Tab 2: Roster Comparison ──────────────────────────────────────────────────

def _tab_roster_comparison(recommender: Recommender) -> None:
    st.subheader("Roster Comparison")

    all_positions = sorted(BATTER_POSITIONS | PITCHER_POSITIONS)
    default = all_positions.index("OF") if "OF" in all_positions else 0
    selected_pos = st.selectbox("Compare at position:", all_positions, index=default)

    try:
        with st.spinner("Comparing players…"):
            result = recommender.compare_to_roster(selected_pos, top_n=3)
    except RuntimeError as exc:
        st.error(str(exc))
        return

    ptype = result.player_type

    # Swap badge
    if result.has_swap():
        swap = result.swap
        st.success(
            f"✅ **Recommended Swap** — "
            f"ADD **{swap.add_name}** ({swap.add_score:.1f}) &nbsp;·&nbsp; "
            f"DROP **{swap.drop_name}** ({swap.drop_score:.1f})"
        )
    else:
        st.info(f"No swap recommended — your {selected_pos} situation is solid.")

    # Threshold for row-level green highlighting in the available table
    weakest_score: float | None = None
    if not result.my_players.empty and "composite_score" in result.my_players.columns:
        weakest_score = float(result.my_players["composite_score"].min())
    swap_threshold = weakest_score + 10.0 if weakest_score is not None else None

    col_avail, col_mine = st.columns(2)

    with col_avail:
        st.markdown(f"##### Top Available ({selected_pos} pool)")
        if result.available_players.empty:
            st.caption("No available players found.")
        else:
            disp = _for_display(result.available_players, ptype)
            st.dataframe(
                _style(disp, highlight_above=swap_threshold),
                use_container_width=True,
                hide_index=True,
            )

    with col_mine:
        st.markdown(f"##### My Roster at {selected_pos}")
        if result.my_players.empty:
            st.caption(
                f"No players declared at **{selected_pos}** in your roster YAML. "
                "Add them under the My Roster tab."
            )
        else:
            disp = _for_display(result.my_players, ptype)
            st.dataframe(_style(disp), use_container_width=True, hide_index=True)

    # st.metric score breakdown
    if result.has_swap():
        swap = result.swap
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Best Available", f"{swap.add_score:.1f}")
        m2.metric("Weakest Rostered", f"{swap.drop_score:.1f}")
        m3.metric(
            "Score Improvement",
            f"+{swap.score_delta:.1f}",
            delta=f"{swap.score_delta:.1f}",
            delta_color="normal",
        )


# ── Tab 3: My Roster ──────────────────────────────────────────────────────────

def _tab_my_roster(roster: RosterManager) -> None:
    st.subheader("My Current Roster")

    scorer = _get_scorer()
    scored_b = scored_p = pd.DataFrame()

    try:
        bpct = _load_batter_pct()
        if not bpct.empty:
            scored_b = scorer.score_batters(bpct)
    except RuntimeError:
        pass

    try:
        ppct = _load_pitcher_pct()
        if not ppct.empty:
            scored_p = scorer.score_pitchers(ppct)
    except RuntimeError:
        pass

    roster_df = roster.get_roster_df()

    def _with_score(sub: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
        if scored.empty or "player_id" not in scored.columns:
            return sub.assign(Score="N/A")
        merged = sub.merge(
            scored[["player_id", "composite_score"]], on="player_id", how="left"
        )
        merged["Score"] = merged["composite_score"].apply(
            lambda v: f"{v:.1f}" if pd.notna(v) else "N/A"
        )
        return merged.drop(columns=["composite_score"], errors="ignore")

    for ptype, heading, scored in [
        ("batter", "Batters", scored_b),
        ("pitcher", "Pitchers", scored_p),
    ]:
        section = roster_df[roster_df["player_type"] == ptype].copy()
        if section.empty:
            continue
        st.markdown(f"### {heading}")
        enriched = _with_score(section, scored)
        show = [c for c in ["player_name", "positions", "Score"] if c in enriched.columns]
        st.dataframe(
            enriched[show].rename(columns={"player_name": "Player", "positions": "Positions"}),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    with st.expander("✏️ Edit Roster YAML"):
        st.caption(
            "Player names must be in **Last, First** format to match pybaseball data. "
            "Changes are saved immediately to `config/my_roster.yaml`."
        )
        try:
            current_yaml = _ROSTER_YAML.read_text(encoding="utf-8")
        except FileNotFoundError:
            current_yaml = "# config/my_roster.yaml not found\n"

        new_yaml = st.text_area(
            "Roster YAML", value=current_yaml, height=440, key="yaml_editor"
        )

        if st.button("💾 Save Roster", type="primary"):
            try:
                parsed = yaml.safe_load(new_yaml)
                if not isinstance(parsed, dict):
                    st.error("Invalid YAML: the root element must be a mapping.")
                    return
                _ROSTER_YAML.write_text(new_yaml, encoding="utf-8")
                _get_roster.clear()
                _get_recommender.clear()
                st.success("Roster saved — reloading…")
                st.rerun()
            except yaml.YAMLError as exc:
                st.error(f"YAML parse error: {exc}")


# ── Tab 4: Methodology ────────────────────────────────────────────────────────

def _tab_methodology() -> None:
    st.subheader("How Scores Are Calculated")

    st.markdown("""
Data is sourced from **Baseball Savant** via the
[pybaseball](https://github.com/jldbc/pybaseball) library, which wraps the
public Statcast API. Percentile ranks are pre-computed by Baseball Savant for
every player who meets minimum playing-time thresholds.

> **Sample-size caveat** — percentile ranks are season-to-date averages.
> A player just returning from injury, or on a hot/cold streak, may have a
> misleading cumulative rank. Always cross-check against recent game logs and
> beat reporter notes before making a move.

### Scoring Method

1. Raw percentile ranks (1–100) are fetched from Baseball Savant. A **higher rank
   always means better for the player**, because Baseball Savant inverts
   "bad" statistics (ERA, strikeout rate, barrel rate allowed) before ranking.
2. Each metric is **re-ranked within the displayed player pool** using fractional
   rank (`Series.rank(pct=True) × 100`), producing a 0–100 score relative to
   whoever is currently being evaluated — not the full MLB cohort.
3. Normalized scores are combined with the weights below into a **composite score**.
""")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Batter Weights")
        bw = PlayerScorer.BATTER_WEIGHTS
        st.dataframe(
            pd.DataFrame({
                "Metric": [
                    "Barrel Rate",
                    "Hard-Hit Rate (95+ mph)",
                    "Expected wOBA (xwOBA)",
                    "Sprint Speed",
                    "Strikeout Rate (inverted)",
                ],
                "Source Column": list(bw.keys()),
                "Weight": [f"{v:.0%}" for v in bw.values()],
            }),
            hide_index=True,
            use_container_width=True,
        )

    with col2:
        st.markdown("#### Pitcher Weights")
        pw = PlayerScorer.PITCHER_WEIGHTS
        st.dataframe(
            pd.DataFrame({
                "Metric": [
                    "Expected ERA / xERA (inverted)",
                    "Whiff Rate",
                    "Chase Rate",
                    "Barrel Rate Allowed (inverted)",
                    "Hard-Hit Rate Allowed (inverted)",
                ],
                "Source Column": list(pw.keys()),
                "Weight": [f"{v:.0%}" for v in pw.values()],
            }),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("""
### Minimum Playing-Time Thresholds

Players below these thresholds are excluded from Baseball Savant's percentile
ranks and will not appear in recommendations.

| Player Type | Threshold |
|---|---|
| Batters | ≥ 2.1 PA per team game played |
| Pitchers | ≥ 1.25 IP per team game played |

Early in the season (April) percentile ranks are based on small samples — treat
scores with extra caution until mid-May.

### Caching

Statcast data is cached in two layers:

1. **Parquet files** under `data/cache/` — deduplicate network calls across
   app restarts; refreshed once per day.
2. **Streamlit in-memory cache** (`@st.cache_data`, TTL 6 h) — eliminates
   re-fetching within the same session.

Use the **Refresh Statcast Data** button in the sidebar to force both layers
to update immediately.

---
*Data source: [Baseball Savant](https://baseballsavant.mlb.com) via
[pybaseball](https://github.com/jldbc/pybaseball)*
""")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = datetime.now()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚾ Fantasy Baseball\nAdvisor")
        st.markdown("---")

        if st.button("🔄 Refresh Statcast Data", use_container_width=True):
            _load_batter_pct.clear()
            _load_pitcher_pct.clear()
            with st.spinner("Fetching fresh data from Baseball Savant…"):
                try:
                    _load_batter_pct()
                    _load_pitcher_pct()
                    st.session_state["last_refresh"] = datetime.now()
                    st.success("Data refreshed!")
                except RuntimeError as exc:
                    st.error(str(exc))

        last = st.session_state.get("last_refresh")
        st.caption(f"Last refresh: {last:%Y-%m-%d %H:%M}" if last else "Last refresh: —")

        st.markdown("---")

        player_type: str = st.radio(
            "Player Type",
            options=["Batters", "Pitchers", "Both"],
            horizontal=True,
        )

        if player_type == "Batters":
            pos_options = ["All"] + sorted(BATTER_POSITIONS)
        elif player_type == "Pitchers":
            pos_options = ["All", "SP", "RP"]
        else:
            pos_options = ["All"]

        pos_label: str = st.selectbox("Position", pos_options)
        pos_filter: str | None = None if pos_label == "All" else pos_label

        top_n: int = st.slider("Recommendations", min_value=5, max_value=25, value=10, step=1)

        st.markdown("---")
        st.caption("Data: [Baseball Savant](https://baseballsavant.mlb.com) via pybaseball")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔎 Top Pickups",
        "🔀 Roster Comparison",
        "📋 My Roster",
        "📐 Methodology",
    ])

    recommender = _get_recommender()
    roster = _get_roster()

    with tab1:
        _tab_top_pickups(recommender, player_type, pos_filter, top_n)
    with tab2:
        _tab_roster_comparison(recommender)
    with tab3:
        _tab_my_roster(roster)
    with tab4:
        _tab_methodology()


if __name__ == "__main__":
    main()
