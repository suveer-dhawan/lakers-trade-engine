"""
Ingestion layer for stats.nba.com via the nba_api Python package.

All public functions follow the same contract:
  - Accept a season string in nba_api format ("2024-25")
  - Check the local cache before hitting the network
  - Sleep NBA_API_DELAY_SECONDS before each live API call
  - Return a pandas DataFrame (empty on unrecoverable errors, with a logged warning)

Endpoint notes
--------------
LeagueDashPlayerStats  -- base and advanced stats; uses per_mode_detailed (not per_mode_simple)
LeagueDashPtStats      -- point-type shooting splits; pt_measure_type = "CatchShoot" | "PullUpShot"
LeagueDashPtDefend     -- close-defender matchup data; primary key is CLOSE_DEF_PERSON_ID
CommonAllPlayers       -- full player registry for a season (includes inactive/historical players)
"""
import logging
import time

import pandas as pd
from nba_api.stats.endpoints import (
    commonallplayers,
    leaguedashplayerstats,
    leaguedashptdefend,
    leaguedashptstats,
)

from .cache import get_or_fetch
from .config import DEFAULT_REQUEST_TIMEOUT, NBA_API_DELAY_SECONDS

logger = logging.getLogger(__name__)

def _sleep() -> None:
    time.sleep(NBA_API_DELAY_SECONDS)


def _safe_fetch(label: str, fetch_fn):
    """Wrap a fetch callable with error handling; returns empty DataFrame on failure."""
    try:
        return fetch_fn()
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", label, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_all_players(season: str) -> pd.DataFrame:
    """
    All players with basic roster info for *season* via CommonAllPlayers.

    Columns: PERSON_ID, DISPLAY_FIRST_LAST, ROSTERSTATUS, FROM_YEAR, TO_YEAR,
    TEAM_ID, TEAM_CITY, TEAM_NAME, TEAM_ABBREVIATION, GAMES_PLAYED_FLAG, ...

    Returns every player in the NBA registry who was active at any point, with
    team info reflecting *season*. Filter on ROSTERSTATUS / FROM_YEAR / TO_YEAR
    downstream to restrict to players active that year.
    """
    def _fetch() -> pd.DataFrame:
        _sleep()
        endpoint = commonallplayers.CommonAllPlayers(
            is_only_current_season=0,
            league_id="00",
            season=season,
            timeout=DEFAULT_REQUEST_TIMEOUT,
        )
        return endpoint.get_data_frames()[0]

    return _safe_fetch(
        f"all_players_{season}",
        lambda: get_or_fetch(f"nba_all_players_{season}", _fetch),
    )


def get_player_stats(season: str, per_mode: str = "PerGame") -> pd.DataFrame:
    """
    League-wide base stats via LeagueDashPlayerStats (MeasureType=Base).

    Parameters
    ----------
    season   : "2024-25" format
    per_mode : "PerGame" | "Totals" | "Per36" | "Per100Possessions" etc.
               Must match nba_api's PerModeDetailed values.

    Columns include: PLAYER_ID, PLAYER_NAME, TEAM_ABBREVIATION, AGE, GP, MIN,
    FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT, FTM, FTA, FT_PCT,
    OREB, DREB, REB, AST, TOV, STL, BLK, PF, PTS, PLUS_MINUS.
    """
    def _fetch() -> pd.DataFrame:
        _sleep()
        endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed=per_mode,
            measure_type_detailed_defense="Base",
            timeout=DEFAULT_REQUEST_TIMEOUT,
        )
        return endpoint.get_data_frames()[0]

    cache_key = f"nba_player_stats_{season}_{per_mode}"
    return _safe_fetch(cache_key, lambda: get_or_fetch(cache_key, _fetch))


def get_player_advanced(season: str) -> pd.DataFrame:
    """
    Advanced stats via LeagueDashPlayerStats (MeasureType=Advanced).

    Available from nba.com: OFF_RATING, DEF_RATING, NET_RATING, AST_PCT,
    AST_TO, AST_RATIO, OREB_PCT, DREB_PCT, REB_PCT, TO_RATIO, EFG_PCT,
    TS_PCT, USG_PCT, PACE, PIE (Player Impact Estimate).

    NOT available from nba_api (Basketball Reference only): PER, WS, WS/48,
    BPM, VORP. Phase 0.2 will add BBRef ingestion to fill these gaps.
    """
    def _fetch() -> pd.DataFrame:
        _sleep()
        endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Advanced",
            timeout=DEFAULT_REQUEST_TIMEOUT,
        )
        return endpoint.get_data_frames()[0]

    cache_key = f"nba_player_advanced_{season}"
    return _safe_fetch(cache_key, lambda: get_or_fetch(cache_key, _fetch))


def get_player_shooting(season: str) -> pd.DataFrame:
    """
    Catch-and-shoot + pull-up splits via LeagueDashPtStats.

    Makes two API calls (PtMeasureType = CatchShoot, PullUpShot) and merges
    on PLAYER_ID. Stat columns are prefixed CS_ and PU_ respectively.

    Key columns after merge:
      CS_FGM, CS_FGA, CS_FG_PCT, CS_FG3M, CS_FG3A, CS_FG3_PCT  (catch & shoot)
      PU_FGM, PU_FGA, PU_FG_PCT, PU_FG3M, PU_FG3A, PU_FG3_PCT  (pull-up)
    """
    _ID_COLS = {"PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "AGE", "GP"}

    def _fetch_type(pt_type: str, prefix: str) -> pd.DataFrame:
        _sleep()
        endpoint = leaguedashptstats.LeagueDashPtStats(
            season=season,
            per_mode_simple="PerGame",
            pt_measure_type=pt_type,
            player_or_team="Player",
            timeout=DEFAULT_REQUEST_TIMEOUT,
        )
        df = endpoint.get_data_frames()[0]
        rename = {c: f"{prefix}{c}" for c in df.columns if c not in _ID_COLS}
        return df.rename(columns=rename)

    def _fetch_catch_shoot() -> pd.DataFrame:
        return _fetch_type("CatchShoot", "CS_")

    def _fetch_pullup() -> pd.DataFrame:
        return _fetch_type("PullUpShot", "PU_")

    cs = _safe_fetch(
        f"shooting_catchshoot_{season}",
        lambda: get_or_fetch(f"nba_shooting_catchshoot_{season}", _fetch_catch_shoot),
    )
    pu = _safe_fetch(
        f"shooting_pullup_{season}",
        lambda: get_or_fetch(f"nba_shooting_pullup_{season}", _fetch_pullup),
    )

    if cs.empty and pu.empty:
        return pd.DataFrame()
    if cs.empty:
        return pu
    if pu.empty:
        return cs

    merge_on = [c for c in ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"] if c in cs.columns and c in pu.columns]
    return cs.merge(pu, on=merge_on, how="outer", suffixes=("", "_pu_dup"))


def get_player_defense(season: str) -> pd.DataFrame:
    """
    Defensive matchup data via LeagueDashPtDefend (defense_category=Overall).

    Captures opponent FG% when this player is the closest defender:
    D_FGM, D_FGA, D_FG_PCT, NORMAL_FG_PCT, PCT_PLUSMINUS (negative = good defender).

    Note: the raw endpoint uses CLOSE_DEF_PERSON_ID as the player key.
    This function renames it to PLAYER_ID for consistency with other tables.
    """
    def _fetch() -> pd.DataFrame:
        _sleep()
        endpoint = leaguedashptdefend.LeagueDashPtDefend(
            defense_category="Overall",
            season=season,
            per_mode_simple="PerGame",
            timeout=DEFAULT_REQUEST_TIMEOUT,
        )
        df = endpoint.get_data_frames()[0]
        # Normalize primary key to PLAYER_ID for merge consistency
        if "CLOSE_DEF_PERSON_ID" in df.columns:
            df = df.rename(columns={"CLOSE_DEF_PERSON_ID": "PLAYER_ID"})
        return df

    cache_key = f"nba_player_defense_{season}"
    return _safe_fetch(cache_key, lambda: get_or_fetch(cache_key, _fetch))


def get_multi_season(fetch_fn, seasons: list) -> pd.DataFrame:
    """
    Call fetch_fn(season) for each season in *seasons*, concatenate results.

    Prepends a SEASON column to each frame. Failed seasons are skipped with
    a warning rather than raising -- the pipeline should be resilient to
    one bad season not blocking the rest.

    Example
    -------
    df = get_multi_season(get_player_advanced, ["2022-23", "2023-24", "2024-25"])
    """
    frames = []
    for season in seasons:
        logger.info("Fetching season %s ...", season)
        df = fetch_fn(season)
        if df.empty:
            logger.warning("No data returned for season %s -- skipping", season)
            continue
        df = df.copy()
        df.insert(0, "SEASON", season)
        frames.append(df)

    if not frames:
        logger.warning("get_multi_season: no data fetched for any season")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
