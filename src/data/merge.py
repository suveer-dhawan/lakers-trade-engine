"""
Unified player dataset builder.

Merges nba_api (base stats, advanced, shooting, defense) with BBRef advanced
(PER, WS, BPM, VORP) and salary data into a single player-level DataFrame.

Join strategy:
  - nba_api sources share PLAYER_ID (numeric) - joined exactly.
  - BBRef and salary sources use display names - joined on normalized_name
    after stripping accents, lowercasing, and removing suffixes.
  - NAME_OVERRIDES handles the ~10 known mismatches that survive normalization.

Entry point: build_player_dataset(season) -> DataFrame
Diagnostic:  data_completeness_report(df) -> dict
"""
from __future__ import annotations

import logging
import re
import unicodedata

import pandas as pd

# EPM paywalled - using BPM as substitute. Uncomment if EPM CSV becomes available.
# from .advanced_metrics import get_epm
from .bbref_stats import get_advanced_stats
from .nba_stats import get_player_advanced, get_player_defense, get_player_shooting, get_player_stats
from .salaries import load_bbref_salaries

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

# Manual overrides: normalized_source_name -> normalized_target_name
# Keys are normalized BBRef/salary display names; values are normalized nba_api names.
# Add entries here for players that still don't match after normalization.
NAME_OVERRIDES: dict[str, str] = {
    # BBRef uses initials with periods; nba_api drops them
    "p.j. washington": "pj washington",
    "p.j. tucker": "pj tucker",
    "a.j. green": "aj green",
    "a.j. lawson": "aj lawson",
    "a.j. johnson": "aj johnson",
    "t.j. warren": "tj warren",
    "t.j. mcconnell": "tj mcconnell",
    "t.j. leaf": "tj leaf",
    "c.j. mccollum": "cj mccollum",
    "c.j. miles": "cj miles",
    "o.g. anunoby": "og anunoby",
    "r.j. hampton": "rj hampton",
    "d.j. augustin": "dj augustin",
    "j.j. redick": "jj redick",
    "k.j. martin": "kj martin",
    # BBRef uses full first name; nba_api uses nickname
    "nicolas claxton": "nic claxton",
    "nicolas batum": "nicolas batum",  # same - no override needed but documented
    "robert williams": "robert williams iii",
    # Spelling variants
    "moe wagner": "moritz wagner",
    "nah'shon hyland": "bones hyland",
    "sviatoslav mykhailiuk": "svi mykhailiuk",
    "isaiah hartenstein": "isaiah hartenstein",
    "aleksej pokusevski": "aleksej pokusevski",
}


def normalize_name(name: str) -> str:
    """
    Normalize a player display name for fuzzy joining across sources.

    Steps:
      1. Decompose Unicode (strip accents: Doncic not Doncic special chars)
      2. Encode to ASCII, ignoring non-ASCII characters
      3. Lowercase and strip whitespace
      4. Remove common name suffixes (Jr., Sr., II, III, IV)
      5. Collapse internal whitespace

    The result is safe for dict-key comparison across sources.
    """
    if not isinstance(name, str) or not name.strip():
        return ""

    # Strip accents
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_bytes = decomposed.encode("ascii", "ignore")
    result = ascii_bytes.decode("ascii").lower().strip()

    # Remove suffixes
    for suffix in (" jr.", " jr", " sr.", " sr", " iv", " iii", " ii"):
        if result.endswith(suffix):
            result = result[: -len(suffix)].strip()

    # Collapse whitespace
    result = " ".join(result.split())

    # Apply manual override if present
    return NAME_OVERRIDES.get(result, result)


def _add_normalized_name(df: pd.DataFrame, source_col: str) -> pd.DataFrame:
    """Add a _norm_name column derived from source_col."""
    df = df.copy()
    df["_norm_name"] = df[source_col].apply(normalize_name)
    return df


# ---------------------------------------------------------------------------
# nba_api merge helpers
# ---------------------------------------------------------------------------

# Columns to drop from advanced before merging (overlap with base stats or noise)
_ADV_DROP = {
    "PLAYER_NAME", "NICKNAME", "TEAM_ID", "TEAM_ABBREVIATION",
    "AGE", "GP", "W", "L", "W_PCT",
    # _RANK columns are noisy - keep them if the caller wants, but drop from core merge
}

_SHOOTING_DROP = {
    "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GP",
    "CS_W", "CS_L", "PU_W", "PU_L", "GP_pu_dup",
}

_DEFENSE_DROP = {
    "PLAYER_NAME", "PLAYER_LAST_TEAM_ID", "PLAYER_POSITION",
    "AGE", "GP", "G",
}


def _merge_nba_sources(season: str) -> pd.DataFrame:
    """
    Merge all four nba_api sources on PLAYER_ID.

    Returns a spine DataFrame with base stats plus advanced, shooting, and defense
    columns appended. The spine is base stats (PerGame); all joins are left joins.
    """
    base = get_player_stats(season, per_mode="PerGame")
    if base.empty:
        logger.warning("No base stats for %s - cannot build dataset", season)
        return pd.DataFrame()

    adv = get_player_advanced(season)
    shooting = get_player_shooting(season)
    defense = get_player_defense(season)

    df = base.copy()

    if not adv.empty:
        adv_cols = [c for c in adv.columns if c not in _ADV_DROP or c == "PLAYER_ID"]
        df = df.merge(adv[adv_cols], on="PLAYER_ID", how="left", suffixes=("", "_adv_dup"))
        df = df.loc[:, ~df.columns.str.endswith("_adv_dup")]

    if not shooting.empty:
        shoot_cols = [c for c in shooting.columns if c not in _SHOOTING_DROP or c == "PLAYER_ID"]
        df = df.merge(shooting[shoot_cols], on="PLAYER_ID", how="left", suffixes=("", "_sh_dup"))
        df = df.loc[:, ~df.columns.str.endswith("_sh_dup")]

    if not defense.empty:
        def_cols = [c for c in defense.columns if c not in _DEFENSE_DROP or c == "PLAYER_ID"]
        df = df.merge(defense[def_cols], on="PLAYER_ID", how="left", suffixes=("", "_def_dup"))
        df = df.loc[:, ~df.columns.str.endswith("_def_dup")]

    return df


# ---------------------------------------------------------------------------
# BBRef merge
# ---------------------------------------------------------------------------

_BBREF_ADV_KEEP = [
    "PLAYER_NAME_BBREF", "PLAYER_ID_BBREF", "TEAM_BBREF",
    "Pos", "G", "MP",
    "PER", "TS%", "three_PAr", "FTr",
    "ORB%", "DRB%", "TRB%", "AST%", "STL%", "BLK%", "TOV%", "USG%",
    "OWS", "DWS", "WS", "WS_48",
    "OBPM", "DBPM", "BPM", "VORP",
    "_norm_name",
]


def _merge_bbref(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Left-join BBRef advanced stats onto df on normalized player name."""
    bbref = get_advanced_stats(season)
    if bbref.empty:
        logger.warning("BBRef advanced empty for season %d - skipping BBRef join", season)
        return df

    bbref = _add_normalized_name(bbref, "PLAYER_NAME_BBREF")
    keep = [c for c in _BBREF_ADV_KEEP if c in bbref.columns]
    bbref_slim = bbref[keep].copy()

    # Drop duplicate normalized names (shouldn't happen after TOT dedup in bbref_stats.py)
    bbref_slim = bbref_slim.drop_duplicates(subset=["_norm_name"], keep="first")

    df = _add_normalized_name(df, "PLAYER_NAME")
    df = df.merge(bbref_slim, on="_norm_name", how="left", suffixes=("", "_bbref_dup"))
    df = df.loc[:, ~df.columns.str.endswith("_bbref_dup")]
    return df


# ---------------------------------------------------------------------------
# Salary merge (BBRef multi-year contract CSV)
# ---------------------------------------------------------------------------

_SALARY_KEEP_STATIC = [
    "PLAYER_NAME_SALARY", "SALARY", "YEARS_REMAINING", "TOTAL_GUARANTEED",
    "IS_EXPIRING", "AVG_ANNUAL_VALUE", "_norm_name",
]


def _merge_salaries(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Left-join BBRef salary data onto df on normalized player name."""
    sal = load_bbref_salaries(season)
    if sal.empty:
        logger.warning("BBRef salary data empty for %s - skipping salary join", season)
        return df

    sal = _add_normalized_name(sal, "PLAYER_NAME_SALARY")

    # Include dynamic year columns (e.g. "2025-26", "2026-27") alongside static ones
    year_cols = [c for c in sal.columns if re.match(r"^\d{4}-\d{2}$", str(c))]
    keep_cols = [c for c in _SALARY_KEEP_STATIC if c in sal.columns] + year_cols
    keep_cols = list(dict.fromkeys(keep_cols))  # preserve order, dedup

    sal_slim = sal[keep_cols].drop_duplicates(subset=["_norm_name"], keep="first").copy()

    if "_norm_name" not in df.columns:
        df = _add_normalized_name(df, "PLAYER_NAME")

    df = df.merge(
        sal_slim.drop(columns=["PLAYER_NAME_SALARY"], errors="ignore"),
        on="_norm_name",
        how="left",
        suffixes=("", "_sal_dup"),
    )
    df = df.loc[:, ~df.columns.str.endswith("_sal_dup")]
    return df


# ---------------------------------------------------------------------------
# EPM merge (paywalled - commented out; BPM used as substitute)
# ---------------------------------------------------------------------------

# _EPM_KEEP = ["PLAYER_NAME_EPM", "EPM", "EPMO", "EPMD", "GP_EPM", "MIN_EPM", "_norm_name"]
#
# def _merge_epm(df: pd.DataFrame, season: str) -> pd.DataFrame:
#     """Left-join EPM data onto df on normalized player name."""
#     epm = get_epm(season)
#     if epm.empty:
#         logger.info("EPM data not available for %s - skipping EPM join", season)
#         return df
#     epm = _add_normalized_name(epm, "PLAYER_NAME_EPM")
#     keep = [c for c in _EPM_KEEP if c in epm.columns]
#     epm_slim = epm[keep].drop_duplicates(subset=["_norm_name"], keep="first")
#     if "_norm_name" not in df.columns:
#         df = _add_normalized_name(df, "PLAYER_NAME")
#     df = df.merge(epm_slim, on="_norm_name", how="left", suffixes=("", "_epm_dup"))
#     df = df.loc[:, ~df.columns.str.endswith("_epm_dup")]
#     return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_player_dataset(season: str) -> pd.DataFrame:
    """
    Build the unified player dataset for a given season.

    Merges in this order (left-join from nba_api outward):
      1. nba_api base stats (spine)
      2. nba_api advanced (PLAYER_ID join)
      3. nba_api shooting splits (PLAYER_ID join)
      4. nba_api defense (PLAYER_ID join)
      5. BBRef advanced / PER / WS / BPM / VORP (normalized name join)
      6. Salary data from HoopsHype (normalized name join)
      7. EPM from Dunks & Threes (normalized name join)

    Parameters
    ----------
    season : str
        Season in nba_api format, e.g. "2024-25".

    Returns
    -------
    DataFrame with one row per player-season. Players who appear on multiple
    teams in nba_api (TEAM_COUNT > 1) represent their season totals; the
    TEAM_ABBREVIATION column reflects their last team.

    Use data_completeness_report(df) to see which sources successfully joined.
    """
    logger.info("Building player dataset for %s ...", season)

    # nba_api sources (exact PLAYER_ID join)
    df = _merge_nba_sources(season)
    if df.empty:
        return df

    # BBRef advanced (int season year = ending year of season string)
    bbref_season = int(season[:4]) + 1  # "2024-25" -> 2025
    df = _merge_bbref(df, bbref_season)

    # Salary (BBRef multi-year CSV)
    df = _merge_salaries(df, season)

    # EPM paywalled - using BPM as substitute. Uncomment if EPM CSV becomes available.
    # df = _merge_epm(df, season)

    # Drop internal join key
    df = df.drop(columns=["_norm_name"], errors="ignore")

    logger.info(
        "Dataset built: %d players, %d columns for %s", len(df), len(df.columns), season
    )
    return df


def data_completeness_report(df: pd.DataFrame) -> dict:
    """
    Report what percentage of players have data from each source.

    Checks for sentinel columns that are only present if that source joined
    successfully, then counts non-null rows.

    Parameters
    ----------
    df : DataFrame
        Output of build_player_dataset().

    Returns
    -------
    dict mapping source_name -> {"col": str, "pct": float, "n_with": int, "n_total": int}
    """
    n_total = len(df)
    if n_total == 0:
        return {}

    source_sentinels = {
        "nba_api_base": "PTS",
        "nba_api_advanced": "NET_RATING",
        "nba_api_shooting": "CS_CATCH_SHOOT_FG3_PCT",
        "nba_api_defense": "PCT_PLUSMINUS",
        "bbref_advanced": "BPM",
        "bbref_ws": "WS",
        "bbref_per": "PER",
        "bbref_vorp": "VORP",
        "salary_current": "SALARY",
        "salary_years": "YEARS_REMAINING",
        "salary_guaranteed": "TOTAL_GUARANTEED",
        # EPM paywalled - using BPM as substitute. Uncomment if EPM CSV becomes available.
        # "epm": "EPM",
    }

    report = {}
    for source, col in source_sentinels.items():
        if col in df.columns:
            n_with = int(df[col].notna().sum())
            report[source] = {
                "col": col,
                "pct": round(100.0 * n_with / n_total, 1),
                "n_with": n_with,
                "n_total": n_total,
            }
        else:
            report[source] = {
                "col": col,
                "pct": 0.0,
                "n_with": 0,
                "n_total": n_total,
            }

    return report


def completeness_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pretty-print version of data_completeness_report.

    Returns a DataFrame sorted by coverage percentage, suitable for display()
    in a notebook.
    """
    report = data_completeness_report(df)
    rows = [
        {"source": src, "sentinel_col": v["col"], "coverage_pct": v["pct"],
         "n_players": v["n_with"], "n_total": v["n_total"]}
        for src, v in report.items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("coverage_pct", ascending=False)
        .reset_index(drop=True)
    )
