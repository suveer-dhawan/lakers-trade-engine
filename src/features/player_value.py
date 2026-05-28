"""
Composite player valuation model.

Phase 1.5b -- Player Valuation (percentile-rank surplus).

Builds ON_COURT_VALUE as a weighted blend of z-scored advanced metrics,
then computes contract-relative SURPLUS_VALUE via percentile-rank comparison.
Integrates durability discounting and an age-curve multiplier.

Output columns added to the input DataFrame:
  Z-scores (qualifying population mean/std applied to all players):
    Z_BPM, Z_VORP, Z_WS, Z_WS_48, Z_NET_RATING, Z_TS_PCT
  Valuation:
    ON_COURT_VALUE         -- weighted z-score composite (qualifying players only)
    SALARY_CURRENT         -- 2025-26 salary (historical reference)
    SALARY_FORWARD         -- 2026-27 salary (forward-looking cost)
    IS_FREE_AGENT          -- True if no 2026-27 salary on record
    CONTRACT_VALUE         -- SALARY_FORWARD / SALARY_CAP_2026_27 (kept for display)
    CONTRACT_VALUE_Z       -- z-score of CONTRACT_VALUE among non-FAs
    VALUE_PERCENTILE       -- production rank within population [0, 1]
    SALARY_PERCENTILE      -- salary rank within non-FA population [0, 1] (non-FAs only)
    SURPLUS_VALUE          -- VALUE_PERCENTILE - SALARY_PERCENTILE (positive = underpaid)
                             For free agents: = VALUE_PERCENTILE (zero assumed cost)
    SURPLUS_RANK           -- rank by SURPLUS_VALUE (1 = most undervalued)
    SALARY_RANK_PCT        -- salary percentile among players with 26-27 salary
  Age-adjusted:
    AGE_FACTOR             -- step-function multiplier based on age curve
    AGE_ADJUSTED_SURPLUS   -- SURPLUS_VALUE * AGE_FACTOR
  Durability-adjusted:
    DURABILITY_SCORE       -- recency-weighted availability score [0, 1]
    MAJOR_INJURY_FLAGS     -- count of seasons where GP < 50% of games played
    ADJUSTED_VALUE         -- ON_COURT_VALUE * DURABILITY_SCORE

Weights rationale (Phase 1.5):
  BPM leads (0.35) -- best single box-score catch-all; independent of team context.
  VORP dropped     -- derived directly from BPM x minutes; double-counting with BPM.
  WS/48 (0.20)    -- different estimation methodology from BPM; genuine additional signal.
  NET_RATING (0.25)-- on/off team-level impact; captures defensive contributions BPM misses.
  TS% (0.20)      -- pure scoring efficiency; independent axis from the above.
  Z_VORP is still computed (for exploration) but excluded from the composite.
  Z_WS is computed (used by downstream modules) but excluded from the composite.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.features.durability import add_durability_to_dataset
from src.utils.constants import SALARY_CAP_2026_27

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weight constants -- tune these, don't bury them
# ---------------------------------------------------------------------------

ON_COURT_WEIGHTS: dict[str, float] = {
    "Z_BPM": 0.35,
    "Z_WS_48": 0.20,
    "Z_NET_RATING": 0.25,
    "Z_TS_PCT": 0.20,
    # Z_VORP excluded: derived from BPM x minutes, double-counts with Z_BPM.
    # Z_WS excluded: cumulative volume stat; Z_WS_48 (rate) is the better complement.
}

# Minimum thresholds for a player to qualify for z-score normalization.
# Below these thresholds the player's sample is too small to be reliable.
MIN_GP = 20
MIN_MIN = 15.0

# Source column mapping: our z-score name -> column name in the merged DataFrame.
# If BBRef and nba_api provide the same concept, use the BBRef version (WS_48, BPM, VORP)
# because it is the more widely cited and comparably scaled metric.
_METRIC_COLS: dict[str, str] = {
    "Z_BPM": "BPM",
    "Z_VORP": "VORP",
    "Z_WS": "WS",
    "Z_WS_48": "WS_48",
    "Z_NET_RATING": "NET_RATING",
    "Z_TS_PCT": "TS_PCT",
}


# ---------------------------------------------------------------------------
# Age curve
# ---------------------------------------------------------------------------


def age_factor(age: float) -> float:
    """
    Step-function multiplier for expected performance trajectory.

    Peak window is 25-28. Before peak = upside bonus. After peak = decline penalty.
    Applied to SURPLUS_VALUE to compute AGE_ADJUSTED_SURPLUS.
    A continuous fitted curve (from historical aging data) can replace this in Phase 2.
    """
    if age <= 22:
        return 1.25
    elif age <= 24:
        return 1.15
    elif age <= 28:
        return 1.00
    elif age <= 30:
        return 0.90
    elif age <= 32:
        return 0.80
    elif age <= 34:
        return 0.65
    else:
        return 0.50


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _zscore_within_qualifying(
    series: pd.Series, qualifying_mask: pd.Series
) -> pd.Series:
    """
    Compute z-scores using only the qualifying player population as the reference.

    Mean and std are derived from qualifying players only, so 0 = average
    rotation-quality player rather than average warm body. The z-score is then
    applied to ALL players (qualifying and non-qualifying alike) -- non-qualifying
    players still receive a score for exploration purposes, but ON_COURT_VALUE
    gates them out separately. Players with NaN source values receive NaN.
    """
    qualified_vals = series[qualifying_mask]
    mu = qualified_vals.mean()
    sigma = qualified_vals.std(ddof=1)
    if sigma == 0 or pd.isna(sigma):
        return pd.Series(np.nan, index=series.index)
    return (series - mu) / sigma


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_player_value(
    df: pd.DataFrame,
    current_season: str = "2025-26",
    add_durability: bool = True,
) -> pd.DataFrame:
    """
    Add valuation columns to the merged player dataset.

    Parameters
    ----------
    df : DataFrame
        Output of build_player_dataset(). Must contain PLAYER_ID, GP, MIN,
        BPM, VORP, WS, WS_48, NET_RATING, TS_PCT, SALARY, and year-salary
        columns (e.g. "2026-27").
    current_season : str
        The season the dataset represents, e.g. "2025-26". Used to derive
        the forward salary column name and durability lookback.
    add_durability : bool
        If True, call add_durability_to_dataset() to append DURABILITY_SCORE
        and MAJOR_INJURY_FLAGS, then compute ADJUSTED_VALUE.
        Set False to skip the API calls during testing.

    Returns
    -------
    df with valuation columns appended. Original rows and columns are
    preserved; the function never drops rows.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # Step 1: Z-scores within qualifying population
    # ------------------------------------------------------------------
    qualifying = (df["GP"] >= MIN_GP) & (df["MIN"] >= MIN_MIN)
    logger.info(
        "Qualifying players (GP>=%d, MIN>=%.0f): %d / %d",
        MIN_GP, MIN_MIN, qualifying.sum(), len(df),
    )

    for z_col, src_col in _METRIC_COLS.items():
        if src_col not in df.columns:
            logger.warning("Column %s not found -- %s will be NaN", src_col, z_col)
            df[z_col] = np.nan
            continue
        df[z_col] = _zscore_within_qualifying(df[src_col], qualifying)

    # ------------------------------------------------------------------
    # Step 2: Weighted composite on-court value
    # ------------------------------------------------------------------
    weights = ON_COURT_WEIGHTS
    assert abs(sum(weights.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

    composite = pd.Series(0.0, index=df.index)
    weight_coverage = pd.Series(0.0, index=df.index)  # sum of weights for non-null terms

    for z_col, w in weights.items():
        has_val = df[z_col].notna()
        composite += df[z_col].fillna(0.0) * w
        weight_coverage += has_val * w

    # Scale composite by actual weight coverage so partial-data players are
    # not unfairly penalized by missing metrics; players with zero coverage get NaN.
    with np.errstate(invalid="ignore"):
        composite = np.where(weight_coverage > 0, composite / weight_coverage, np.nan)
    df["ON_COURT_VALUE"] = np.where(qualifying, composite, np.nan)

    logger.info(
        "ON_COURT_VALUE computed for %d qualifying players",
        df["ON_COURT_VALUE"].notna().sum(),
    )

    # ------------------------------------------------------------------
    # Step 3: Contract value (forward-looking to 2026-27)
    # ------------------------------------------------------------------
    forward_col = _forward_salary_col(current_season)

    df["SALARY_CURRENT"] = df["SALARY"]  # 2025-26 reference

    if forward_col in df.columns:
        df["SALARY_FORWARD"] = pd.to_numeric(df[forward_col], errors="coerce")
    else:
        logger.warning(
            "Forward salary column '%s' not found. SALARY_FORWARD will be NaN.", forward_col
        )
        df["SALARY_FORWARD"] = np.nan

    df["IS_FREE_AGENT"] = df["SALARY_FORWARD"].isna() | (df["SALARY_FORWARD"] == 0)
    df["CONTRACT_VALUE"] = np.where(
        ~df["IS_FREE_AGENT"],
        df["SALARY_FORWARD"] / SALARY_CAP_2026_27,
        np.nan,
    )

    # Salary rank percentile among players with a 26-27 salary
    has_fwd = ~df["IS_FREE_AGENT"]
    df["SALARY_RANK_PCT"] = np.nan
    if has_fwd.any():
        df.loc[has_fwd, "SALARY_RANK_PCT"] = df.loc[has_fwd, "SALARY_FORWARD"].rank(
            pct=True, ascending=True
        )

    # ------------------------------------------------------------------
    # Step 4: Surplus value (percentile-rank method)
    #
    # Surplus = VALUE_PERCENTILE - SALARY_PERCENTILE.
    # Comparing ranks rather than raw values handles the skewed salary
    # distribution cleanly: a star in the 90th percentile of production
    # but only the 90th percentile of salary has surplus ~= 0 (fairly paid).
    # OLS can't do this because salary is right-skewed while production is normal.
    # ------------------------------------------------------------------
    cv_z = _zscore_within_qualifying(df["CONTRACT_VALUE"], has_fwd)
    df["CONTRACT_VALUE_Z"] = cv_z

    non_fa_mask = (
        has_fwd
        & df["ON_COURT_VALUE"].notna()
        & df["SALARY_FORWARD"].notna()
    )

    df["VALUE_PERCENTILE"] = np.nan
    df["SALARY_PERCENTILE"] = np.nan
    df["SURPLUS_VALUE"] = np.nan

    if non_fa_mask.sum() >= 10:
        subset = df.loc[non_fa_mask].copy()
        subset["VALUE_PERCENTILE"] = subset["ON_COURT_VALUE"].rank(pct=True)
        subset["SALARY_PERCENTILE"] = subset["SALARY_FORWARD"].rank(pct=True)
        subset["SURPLUS_VALUE"] = subset["VALUE_PERCENTILE"] - subset["SALARY_PERCENTILE"]
        df.loc[non_fa_mask, "VALUE_PERCENTILE"] = subset["VALUE_PERCENTILE"]
        df.loc[non_fa_mask, "SALARY_PERCENTILE"] = subset["SALARY_PERCENTILE"]
        df.loc[non_fa_mask, "SURPLUS_VALUE"] = subset["SURPLUS_VALUE"]
        logger.info(
            "Percentile-rank surplus computed for %d non-FA players", non_fa_mask.sum()
        )
    else:
        logger.warning(
            "Too few qualifying non-FAs (%d) for surplus computation", non_fa_mask.sum()
        )

    # Free agents: surplus = VALUE_PERCENTILE (full production value, zero assumed cost).
    # Ranked within the full qualified population so FAs and non-FAs are comparable.
    all_qual_mask = df["ON_COURT_VALUE"].notna()
    if all_qual_mask.any():
        all_val_pctile = df.loc[all_qual_mask, "ON_COURT_VALUE"].rank(pct=True)
        fa_mask = df["IS_FREE_AGENT"] & all_qual_mask
        df.loc[fa_mask, "VALUE_PERCENTILE"] = all_val_pctile.loc[fa_mask]
        df.loc[fa_mask, "SURPLUS_VALUE"] = df.loc[fa_mask, "VALUE_PERCENTILE"]
        logger.info("FA surplus (VALUE_PERCENTILE) set for %d free agents", fa_mask.sum())

    # Surplus rank: 1 = most undervalued
    has_surplus = df["SURPLUS_VALUE"].notna()
    df["SURPLUS_RANK"] = np.nan
    if has_surplus.any():
        df.loc[has_surplus, "SURPLUS_RANK"] = df.loc[has_surplus, "SURPLUS_VALUE"].rank(
            ascending=False, method="min"
        )

    # ------------------------------------------------------------------
    # Step 4.5: Age-adjusted surplus
    # ------------------------------------------------------------------
    if "AGE" in df.columns:
        df["AGE_FACTOR"] = df["AGE"].apply(
            lambda a: age_factor(float(a)) if pd.notna(a) else np.nan
        )
        df["AGE_ADJUSTED_SURPLUS"] = np.where(
            df["SURPLUS_VALUE"].notna() & df["AGE_FACTOR"].notna(),
            df["SURPLUS_VALUE"] * df["AGE_FACTOR"],
            np.nan,
        )
    else:
        logger.warning("AGE column not found; AGE_ADJUSTED_SURPLUS skipped")
        df["AGE_FACTOR"] = np.nan
        df["AGE_ADJUSTED_SURPLUS"] = np.nan

    # ------------------------------------------------------------------
    # Step 5: Durability adjustment
    # ------------------------------------------------------------------
    if add_durability:
        df = add_durability_to_dataset(df, current_season=current_season)
        df["ADJUSTED_VALUE"] = np.where(
            df["ON_COURT_VALUE"].notna(),
            df["ON_COURT_VALUE"] * df["DURABILITY_SCORE"],
            np.nan,
        )
        logger.info("Durability scores added and ADJUSTED_VALUE computed.")
    else:
        logger.info("Durability step skipped (add_durability=False).")

    return df


# ---------------------------------------------------------------------------
# Convenience reporters
# ---------------------------------------------------------------------------


def surplus_leaderboard(
    df: pd.DataFrame,
    n: int = 30,
    exclude_teams: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return top-n undervalued players sorted by SURPLUS_VALUE descending.

    Parameters
    ----------
    exclude_teams : list of team abbreviations to filter out (e.g. ["LAL"]).
    """
    cols = [
        "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE", "ON_COURT_VALUE",
        "VALUE_PERCENTILE", "SALARY_PERCENTILE", "SALARY_FORWARD",
        "CONTRACT_VALUE", "SURPLUS_VALUE", "SURPLUS_RANK",
        "DURABILITY_SCORE", "ADJUSTED_VALUE",
    ]
    available = [c for c in cols if c in df.columns]
    mask = df["SURPLUS_VALUE"].notna()
    if exclude_teams:
        mask &= ~df["TEAM_ABBREVIATION"].isin(exclude_teams)
    return (
        df[mask][available]
        .sort_values("SURPLUS_VALUE", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def overpaid_leaderboard(df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """Return top-n overpaid players sorted by SURPLUS_VALUE ascending."""
    cols = [
        "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE", "ON_COURT_VALUE",
        "VALUE_PERCENTILE", "SALARY_PERCENTILE", "SALARY_FORWARD",
        "CONTRACT_VALUE", "SURPLUS_VALUE",
    ]
    available = [c for c in cols if c in df.columns]
    mask = df["SURPLUS_VALUE"].notna()
    return (
        df[mask][available]
        .sort_values("SURPLUS_VALUE", ascending=True)
        .head(n)
        .reset_index(drop=True)
    )


def fa_leaderboard(
    df: pd.DataFrame,
    n: int = 30,
    complement_col: str | None = "ARCHETYPE_DISTANCE",
) -> pd.DataFrame:
    """
    Return top-n free agents ranked by ON_COURT_VALUE (production, unconstrained by cost).

    If complement_col exists in df, include it in output for context.
    """
    cols = [
        "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE", "ON_COURT_VALUE",
        "VALUE_PERCENTILE", "AGE_FACTOR", "DURABILITY_SCORE", "ADJUSTED_VALUE",
    ]
    if complement_col and complement_col in df.columns:
        cols.append(complement_col)
    available = [c for c in cols if c in df.columns]
    mask = df["IS_FREE_AGENT"] & df["ON_COURT_VALUE"].notna()
    return (
        df[mask][available]
        .sort_values("ON_COURT_VALUE", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------


def _forward_salary_col(current_season: str) -> str:
    """Derive the next-season salary column from the current season string."""
    start_year = int(current_season[:4])
    next_year = start_year + 1
    short = str(next_year + 1)[-2:]
    return f"{next_year}-{short}"
