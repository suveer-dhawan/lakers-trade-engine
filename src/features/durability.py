"""
Durability feature engineering.

Measures how consistently a player has stayed healthy over recent seasons.
Recency-weighted so a healthy current season outweighs an old injury year.

Output columns added to a player DataFrame:
  DURABILITY_SCORE   -- float in [0, 1]; 1.0 = played every game across all lookback seasons
  MAJOR_INJURY_FLAGS -- int; count of seasons where the player played <50% of games
"""
from __future__ import annotations

import logging

import pandas as pd

from src.data.nba_stats import get_player_stats

logger = logging.getLogger(__name__)

# Recency weights applied to [current, prev1, prev2] seasons
_WEIGHTS = [2.0, 1.5, 1.0]


def _season_before(season: str) -> str:
    """Return the season immediately prior. "2025-26" -> "2024-25"."""
    start = int(season[:4]) - 1
    return f"{start}-{str(start + 1)[-2:]}"


def compute_durability_score(
    player_id: int,
    seasons: list[str],
    games_per_season: int = 82,
) -> dict:
    """
    Compute a recency-weighted durability score for a single player.

    Parameters
    ----------
    player_id : int
        nba_api numeric player ID.
    seasons : list[str]
        Seasons to analyze, ordered most-recent first, e.g.
        ["2025-26", "2024-25", "2023-24"]. At most 3 are used.
    games_per_season : int
        Full season game count (default 82). Used as the denominator.

    Returns
    -------
    dict with keys:
      durability_score   -- float in [0, 1]
      major_injury_flags -- int (seasons with GP < 50% of games_per_season)
      seasons_analyzed   -- int
      raw_gp             -- list[int] of games played per season (same order as input)
    """
    active_seasons = seasons[:3]
    gp_list: list[int] = []

    for season in active_seasons:
        stats = get_player_stats(season, per_mode="PerGame")
        if not stats.empty and "PLAYER_ID" in stats.columns and "GP" in stats.columns:
            row = stats[stats["PLAYER_ID"] == player_id]
            gp = int(row["GP"].values[0]) if len(row) > 0 else 0
        else:
            gp = 0
        gp_list.append(gp)

    if not gp_list:
        return {"durability_score": 0.0, "major_injury_flags": 0, "seasons_analyzed": 0, "raw_gp": []}

    weights = _WEIGHTS[: len(gp_list)]
    total_weight = sum(weights)
    weighted_gp = sum(gp * w for gp, w in zip(gp_list, weights))
    score = min(weighted_gp / (games_per_season * total_weight), 1.0)

    threshold = games_per_season * 0.5
    major_injuries = sum(1 for gp in gp_list if gp < threshold)

    return {
        "durability_score": round(score, 4),
        "major_injury_flags": major_injuries,
        "seasons_analyzed": len(gp_list),
        "raw_gp": gp_list,
    }


def add_durability_to_dataset(
    df: pd.DataFrame,
    current_season: str,
    lookback: int = 3,
) -> pd.DataFrame:
    """
    Add DURABILITY_SCORE and MAJOR_INJURY_FLAGS columns to a player DataFrame.

    Builds the lookback season list from current_season, then calls
    compute_durability_score for each PLAYER_ID. Uses the cached nba_api
    data so no live API calls are made if the cache is warm.

    Parameters
    ----------
    df : DataFrame
        Output of build_player_dataset() -- must have a PLAYER_ID column.
    current_season : str
        Most recent season, e.g. "2025-26".
    lookback : int
        Number of seasons to consider (1-3). Capped at 3 by the weight formula.

    Returns
    -------
    df with two new columns appended: DURABILITY_SCORE, MAJOR_INJURY_FLAGS.
    """
    if "PLAYER_ID" not in df.columns:
        logger.warning("add_durability_to_dataset: PLAYER_ID column missing, skipping")
        return df

    # Build season list most-recent first
    seasons: list[str] = [current_season]
    s = current_season
    for _ in range(min(lookback, 3) - 1):
        s = _season_before(s)
        seasons.append(s)

    logger.info("Computing durability for %d players over seasons %s", len(df), seasons)

    scores: list[float] = []
    flags: list[int] = []

    for pid in df["PLAYER_ID"]:
        result = compute_durability_score(int(pid), seasons)
        scores.append(result["durability_score"])
        flags.append(result["major_injury_flags"])

    df = df.copy()
    df["DURABILITY_SCORE"] = scores
    df["MAJOR_INJURY_FLAGS"] = flags
    return df
