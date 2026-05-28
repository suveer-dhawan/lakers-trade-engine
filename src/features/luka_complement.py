"""
Luka Complement Score -- role-fit metric for star-complementary players.

Phase 1.5b: Historical archetype profiling replaces K-means clustering.

Approach A: Historical archetype profiling.
  Rather than clustering 2025-26 players and checking which clusters
  historical Luka teammates land in, we go the other direction:
    1. Pull each Dallas Luka-teammate's stats from their actual Luka season.
    2. Compute the mean feature vector per archetype (the centroid).
    3. Scale current 2025-26 players using a StandardScaler fitted on
       the current league population.
    4. Classify each current player by distance to the nearest archetype centroid.

  This eliminates the key failure mode of the K-means approach: Brunson's
  2025-26 Knicks profile (primary creator) was landing in a ball-dominant
  cluster and contaminating the "complement" definition.

  Output columns:
    BEST_ARCHETYPE        -- closest archetype (rim_runner/three_and_d/
                             secondary_creator/shooter), or None if too far
    ARCHETYPE_DISTANCE    -- Euclidean distance to closest centroid (scaled space)
    DIST_RIM_RUNNER       -- distance to rim_runner centroid
    DIST_THREE_AND_D      -- distance to three_and_d centroid
    DIST_SECONDARY_CREATOR-- distance to secondary_creator centroid
    DIST_SHOOTER          -- distance to shooter centroid

Approach B: Big-man fit scores (three separate dimensions).
  ROLL_GRAVITY_SCORE = z(NORMAL_FG_PCT) + z(OREB_PCT) - z(USG_PCT)
  RIM_PROTECTION_SCORE = z(BLK%) + z(DREB_PCT)
  BIG_MAN_FIT_SCORE = ROLL_GRAVITY_SCORE + RIM_PROTECTION_SCORE

Combined output:
    ARCHETYPE_DISTANCE is used as a FILTER gate, not a continuous input.
    Players within ARCHETYPE_DISTANCE_THRESHOLD are ranked by AGE_ADJUSTED_SURPLUS.
    Players outside the filter receive a 0.5x discount on their combined score.
    COMBINED_TARGET_SCORE   -- filter-then-rank combined score
    COMBINED_TARGET_RANK    -- rank (1 = top trade/FA target)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Features available in nba_api advanced + per-game stats caches for ALL seasons
# (2019-20 through 2025-26). Using this minimal common set ensures historical
# teammates from 2021-22 and 2022-23 can be loaded without imputation.
ARCHETYPE_FEATURES: list[str] = [
    "USG_PCT",    # usage rate -- how ball-dominant (advanced)
    "AST_PCT",    # assist percentage (advanced)
    "OREB_PCT",   # offensive rebound rate (advanced)
    "DREB_PCT",   # defensive rebound rate (advanced)
    "TS_PCT",     # true shooting % (advanced)
    "NET_RATING", # net rating (advanced)
    "FG3_PCT",    # 3-point percentage -- shooting threat proxy (stats)
    "FG_PCT",     # FG% -- finishing proxy (stats)
]

# Distance threshold: players within this distance are "in the filter".
# Players outside get ARCHETYPE_OUTSIDE_DISCOUNT applied to their combined score.
# Set at 2.5 to ensure rim runners (Gafford ~2.14 from centroid) are captured.
ARCHETYPE_DISTANCE_THRESHOLD = 2.5
ARCHETYPE_OUTSIDE_DISCOUNT = 0.5

# Minimum player sample for historical stats to count as valid
HIST_MIN_GP = 15
HIST_MIN_MIN = 10.0

# Minimum thresholds for 2025-26 player inclusion in scoring
CLUSTER_MIN_GP = 20
CLUSTER_MIN_MIN = 15.0

# ---------------------------------------------------------------------------
# Historical Dallas teammate archetype map
#
# Each entry: player_name -> {id, season (when they played with Luka), archetype}
# Seasons corrected to when each player was actually on the Dallas roster.
# ---------------------------------------------------------------------------

DALLAS_ARCHETYPE_MAP: dict[str, dict] = {
    # 2023-24 Finals team -- primary validation season
    "Daniel Gafford":      {"id": 1629655, "season": "2023-24", "archetype": "rim_runner"},
    "Dereck Lively II":    {"id": 1641726, "season": "2023-24", "archetype": "rim_runner"},
    "Dwight Powell":       {"id": 203939,  "season": "2023-24", "archetype": "rim_runner"},
    "P.J. Washington":     {"id": 1629023, "season": "2023-24", "archetype": "three_and_d"},
    "Derrick Jones Jr.":   {"id": 1627884, "season": "2023-24", "archetype": "three_and_d"},
    "Josh Green":          {"id": 1630182, "season": "2023-24", "archetype": "three_and_d"},
    # Kyrie excluded: his 2023-24 stats (~32% USG, 27 PPG) reflect a co-primary role,
    # not a secondary creator role. Including him would pull the secondary_creator
    # centroid toward ball-dominant guards, causing false positives (Fox, Harden).
    # 2022-23 -- some role players from this team
    "Tim Hardaway Jr.":    {"id": 203501,  "season": "2022-23", "archetype": "shooter"},
    "Maxi Kleber":         {"id": 1628467, "season": "2022-23", "archetype": "shooter"},
    # 2021-22 -- DFS and Bullock on DAL; Dinwiddie/Brunson as true secondary creators
    "Dorian Finney-Smith": {"id": 1627827, "season": "2021-22", "archetype": "three_and_d"},
    "Reggie Bullock":      {"id": 203493,  "season": "2021-22", "archetype": "three_and_d"},
    "Spencer Dinwiddie":   {"id": 203915,  "season": "2021-22", "archetype": "secondary_creator"},
    "Jalen Brunson":       {"id": 1628973, "season": "2021-22", "archetype": "secondary_creator"},
}

# Roll gravity features: lob threat / finishing ability (signs: +1 additive, -1 subtracted)
ROLL_GRAVITY_FEATURES: dict[str, int] = {
    "NORMAL_FG_PCT": +1,
    "OREB_PCT": +1,
    "USG_PCT": -1,
}

# Rim protection features: defensive anchor value
RIM_PROTECTION_FEATURES: dict[str, int] = {
    "BLK%": +1,
    "DREB_PCT": +1,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _zscore_series(series: pd.Series) -> pd.Series:
    """Z-score a series; returns NaN for all-NaN input."""
    mu = series.mean()
    sigma = series.std(ddof=1)
    if sigma == 0 or pd.isna(sigma):
        return pd.Series(np.nan, index=series.index)
    result = (series - mu) / sigma
    result.attrs = {}  # prevent DataFrame-valued attrs from breaking pd.concat comparisons
    return result


def _qualifying_mask(df: pd.DataFrame) -> pd.Series:
    return (df["GP"] >= CLUSTER_MIN_GP) & (df["MIN"] >= CLUSTER_MIN_MIN)


def _load_player_historical_stats(
    player_id: int,
    season: str,
    features: list[str],
) -> dict | None:
    """
    Load one player's feature values from nba_api advanced + per-game stats caches.

    Returns {feature: value} dict with NaN for features not found in cache,
    or None if the player is not found at all for that season.

    Logs every attempt so retrieval failures are visible (no silent skips).
    """
    adv_file = _CACHE_DIR / f"nba_player_advanced_{season}.parquet"
    stats_file = _CACHE_DIR / f"nba_player_stats_{season}_PerGame.parquet"

    result: dict[str, float] = {f: np.nan for f in features}
    found = False

    # Load advanced stats (USG_PCT, AST_PCT, OREB_PCT, DREB_PCT, TS_PCT, NET_RATING)
    if adv_file.exists():
        adv = pd.read_parquet(adv_file)
        row = adv[adv["PLAYER_ID"] == player_id]
        if not row.empty:
            r = row.iloc[0]
            # Validate sample size
            gp = r.get("GP", 0)
            mn = r.get("MIN", 0.0)
            if gp < HIST_MIN_GP or mn < HIST_MIN_MIN:
                logger.warning(
                    "Player %d (%s): GP=%s MIN=%s below thresholds -- skipping",
                    player_id, season, gp, mn,
                )
                return None
            for feat in features:
                if feat in r.index and pd.notna(r[feat]):
                    result[feat] = float(r[feat])
            found = True
            logger.info(
                "  Player %d (%s): found in advanced cache (GP=%s, MIN=%s)",
                player_id, season, gp, mn,
            )
        else:
            logger.warning("  Player %d (%s): not in advanced cache", player_id, season)
    else:
        logger.warning("  Advanced cache missing: %s", adv_file.name)

    if not found:
        return None

    # Load per-game stats (FG3_PCT, FG_PCT) -- merge into result
    if stats_file.exists():
        stats = pd.read_parquet(stats_file)
        srow = stats[stats["PLAYER_ID"] == player_id]
        if not srow.empty:
            sr = srow.iloc[0]
            for feat in features:
                if feat in sr.index and pd.notna(sr[feat]) and np.isnan(result[feat]):
                    result[feat] = float(sr[feat])
    else:
        logger.warning("  Stats cache missing: %s", stats_file.name)

    # Report which features were available vs missing
    missing = [f for f in features if np.isnan(result[f])]
    if missing:
        logger.info("  Player %d (%s): features missing (will be imputed): %s", player_id, season, missing)

    return result


# ---------------------------------------------------------------------------
# Archetype centroid computation
# ---------------------------------------------------------------------------


def build_archetype_centroids(
    scaler: StandardScaler,
    features: list[str],
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """
    Build archetype centroids from DALLAS_ARCHETYPE_MAP historical player stats.

    The scaler should be fitted on the current 2025-26 qualifying player population
    so that distances are in units of current-league standard deviations.

    Returns
    -------
    centroids : dict {archetype_name: centroid_vector (in scaled space)}
    validation_rows : list of dicts summarizing retrieval results for notebook display
    """
    archetype_vectors: dict[str, list[np.ndarray]] = {}
    validation_rows = []

    logger.info("Building archetype centroids from %d historical players:", len(DALLAS_ARCHETYPE_MAP))

    for player_name, info in DALLAS_ARCHETYPE_MAP.items():
        pid = info["id"]
        season = info["season"]
        archetype = info["archetype"]
        logger.info("Loading %s (%s, %s):", player_name, season, archetype)

        stats = _load_player_historical_stats(pid, season, features)
        if stats is None:
            validation_rows.append({
                "Player": player_name, "Season": season, "Archetype": archetype,
                "Found": False, "Features Available": 0, "Features Imputed": 0,
                "Note": "Not found or below sample threshold",
            })
            continue

        feat_vec = np.array([stats[f] for f in features])
        n_missing = int(np.isnan(feat_vec).sum())
        n_avail = len(features) - n_missing

        if n_missing > len(features) // 2:
            logger.warning(
                "  %s: %d of %d features missing -- too many missing, skipping centroid contribution",
                player_name, n_missing, len(features),
            )
            validation_rows.append({
                "Player": player_name, "Season": season, "Archetype": archetype,
                "Found": True, "Features Available": n_avail, "Features Imputed": n_missing,
                "Note": f"Skipped: {n_missing}/{len(features)} features missing",
            })
            continue

        # Impute missing features with the scaler mean (population average in raw space)
        for i, f in enumerate(features):
            if np.isnan(feat_vec[i]):
                feat_vec[i] = scaler.mean_[i]

        # Scale using the current-season scaler
        feat_scaled = scaler.transform(feat_vec.reshape(1, -1))[0]

        if archetype not in archetype_vectors:
            archetype_vectors[archetype] = []
        archetype_vectors[archetype].append(feat_scaled)

        validation_rows.append({
            "Player": player_name, "Season": season, "Archetype": archetype,
            "Found": True, "Features Available": n_avail, "Features Imputed": n_missing,
            "Note": "OK",
        })

    # Compute centroid as mean of all scaled vectors per archetype
    centroids: dict[str, np.ndarray] = {}
    for archetype, vecs in archetype_vectors.items():
        centroids[archetype] = np.mean(vecs, axis=0)
        logger.info(
            "Archetype '%s': centroid from %d players", archetype, len(vecs)
        )

    return centroids, validation_rows


def _fit_scaler_on_current(df: pd.DataFrame, qualifying: pd.Series) -> tuple[StandardScaler, np.ndarray, pd.Index]:
    """
    Fit a StandardScaler on current-season qualifying players with complete feature data.

    Returns (scaler, X_scaled, complete_index).
    """
    feat_data = df.loc[qualifying, ARCHETYPE_FEATURES].copy()
    # Fill FG3_PCT=0 for players with no 3PA (not missing, genuinely 0)
    if "FG3_PCT" in feat_data.columns:
        feat_data["FG3_PCT"] = feat_data["FG3_PCT"].fillna(0.0)

    complete_mask = feat_data.notna().all(axis=1)
    complete_idx = feat_data[complete_mask].index

    scaler = StandardScaler()
    X = scaler.fit_transform(feat_data.loc[complete_idx])
    logger.info(
        "Scaler fitted on %d qualifying players with complete archetype features",
        len(complete_idx),
    )
    return scaler, X, complete_idx


def _classify_by_archetype(
    df: pd.DataFrame,
    qualifying: pd.Series,
    centroids: dict[str, np.ndarray],
    scaler: StandardScaler,
    X_scaled: np.ndarray,
    complete_idx: pd.Index,
) -> pd.DataFrame:
    """
    Assign BEST_ARCHETYPE and distance columns to each qualifying current player.

    Distance columns: DIST_{ARCHETYPE_NAME.upper()} for each archetype.
    BEST_ARCHETYPE is None for players farther than ARCHETYPE_DISTANCE_THRESHOLD
    from all centroids.
    """
    df = df.copy()
    archetype_names = list(centroids.keys())
    dist_cols = {a: f"DIST_{a.upper()}" for a in archetype_names}

    # Initialize distance columns
    for col in dist_cols.values():
        df[col] = np.nan
    df["ARCHETYPE_DISTANCE"] = np.nan
    df["BEST_ARCHETYPE"] = None

    if not centroids:
        logger.warning("No archetype centroids available -- archetype classification skipped")
        return df

    # USG_PCT gate for secondary_creator archetype.
    # A player using > 26% of team possessions cannot realistically play a secondary
    # creator role alongside Luka (who uses ~35%). High-USG players physically cannot
    # occupy this role regardless of stylistic similarity to Brunson/Dinwiddie.
    usg_col = "USG_PCT"
    usg_vals = df[usg_col] if usg_col in df.columns else pd.Series(0.0, index=df.index)
    SECONDARY_CREATOR_MAX_USG = 0.26

    # Compute distance for each complete qualifying player
    for i, global_idx in enumerate(complete_idx):
        player_vec = X_scaled[i]
        player_usg = float(usg_vals.at[global_idx]) if pd.notna(usg_vals.at[global_idx]) else 0.0
        min_dist = float("inf")
        best_arch = None
        for arch, centroid in centroids.items():
            dist = float(np.linalg.norm(player_vec - centroid))
            df.at[global_idx, dist_cols[arch]] = dist
            # Skip secondary_creator for high-usage players
            if arch == "secondary_creator" and player_usg > SECONDARY_CREATOR_MAX_USG:
                continue
            if dist < min_dist:
                min_dist = dist
                best_arch = arch

        df.at[global_idx, "ARCHETYPE_DISTANCE"] = min_dist
        if min_dist <= ARCHETYPE_DISTANCE_THRESHOLD:
            df.at[global_idx, "BEST_ARCHETYPE"] = best_arch

    in_filter = df["BEST_ARCHETYPE"].notna()
    logger.info(
        "Archetype classification: %d players within threshold (%.1f), %d outside",
        in_filter.sum(), ARCHETYPE_DISTANCE_THRESHOLD,
        qualifying.sum() - in_filter.sum(),
    )
    return df


# ---------------------------------------------------------------------------
# Approach B: Roll Gravity Score
# ---------------------------------------------------------------------------


def _compute_roll_gravity(df: pd.DataFrame, qualifying: pd.Series) -> pd.Series:
    """
    Compute ROLL_GRAVITY_SCORE for qualifying players.

    Each feature is z-scored within the qualifying population.
    Missing features are excluded from the sum (partial scores are still computed).
    """
    scores = pd.Series(np.nan, index=df.index)
    qual_idx = df.index[qualifying]

    component_series = []
    available_signs = {}
    for feat, sign in ROLL_GRAVITY_FEATURES.items():
        if feat not in df.columns:
            logger.warning("Roll gravity feature '%s' not in dataset -- skipping", feat)
            continue
        z = _zscore_series(df.loc[qualifying, feat])
        component_series.append(z * sign)
        available_signs[feat] = sign

    if not component_series:
        logger.warning("No roll gravity features available")
        return scores

    logger.info("Roll gravity using features: %s", list(available_signs.keys()))

    combined = pd.concat(component_series, axis=1).sum(axis=1, min_count=1)
    scores.loc[qual_idx] = combined.values
    return scores


def _compute_rim_protection(df: pd.DataFrame, qualifying: pd.Series) -> pd.Series:
    """
    Compute RIM_PROTECTION_SCORE for qualifying players.

    Each feature is z-scored within the qualifying population.
    Missing features are excluded from the sum (partial scores still computed).
    """
    scores = pd.Series(np.nan, index=df.index)
    qual_idx = df.index[qualifying]

    component_series = []
    for feat, sign in RIM_PROTECTION_FEATURES.items():
        if feat not in df.columns:
            logger.warning("Rim protection feature '%s' not in dataset -- skipping", feat)
            continue
        z = _zscore_series(df.loc[qualifying, feat])
        component_series.append(z * sign)

    if not component_series:
        logger.warning("No rim protection features available")
        return scores

    combined = pd.concat(component_series, axis=1).sum(axis=1, min_count=1)
    scores.loc[qual_idx] = combined.values
    return scores


# ---------------------------------------------------------------------------
# Combined target score
# ---------------------------------------------------------------------------

COMPLEMENT_DISTANCE_THRESHOLD = ARCHETYPE_DISTANCE_THRESHOLD  # alias for compatibility


def _compute_combined_target(df: pd.DataFrame) -> pd.Series:
    """
    Filter-then-rank combined target score.

    Archetype fit is a GATE, not a continuous input:
      - Step 1: Flag players with BEST_ARCHETYPE not None (within threshold) OR
                ARCHETYPE_DISTANCE < ARCHETYPE_DISTANCE_THRESHOLD.
      - Step 2: Base score = AGE_ADJUSTED_SURPLUS (or SURPLUS_VALUE for FAs).
      - Step 3: Players outside the filter receive ARCHETYPE_OUTSIDE_DISCOUNT penalty.
    """
    # Base value: prefer age-adjusted surplus
    if "AGE_ADJUSTED_SURPLUS" in df.columns:
        value_series = df["AGE_ADJUSTED_SURPLUS"].copy()
    elif "SURPLUS_VALUE" in df.columns:
        value_series = df["SURPLUS_VALUE"].copy()
    else:
        value_series = df.get("ON_COURT_VALUE", pd.Series(np.nan, index=df.index)).copy()

    value_z = _zscore_series(value_series.dropna()).reindex(df.index)
    combined = value_z.copy()

    # Apply archetype filter
    if "ARCHETYPE_DISTANCE" in df.columns:
        in_filter = df["BEST_ARCHETYPE"].notna() if "BEST_ARCHETYPE" in df.columns else (
            df["ARCHETYPE_DISTANCE"] < ARCHETYPE_DISTANCE_THRESHOLD
        )
        outside_filter = ~in_filter & combined.notna()
        combined.loc[outside_filter] *= ARCHETYPE_OUTSIDE_DISCOUNT
        logger.info(
            "Complement filter: %d players in-filter, %d outside (%.1fx discount)",
            in_filter.sum(), outside_filter.sum(), ARCHETYPE_OUTSIDE_DISCOUNT,
        )
    else:
        logger.warning("ARCHETYPE_DISTANCE not present; no complement filter applied")

    return combined


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_luka_complement(
    df: pd.DataFrame,
    exclude_lakers: bool = False,
) -> pd.DataFrame:
    """
    Add Luka complement scoring columns to the valued player DataFrame.

    Expects compute_player_value() has already been called (needs IS_FREE_AGENT,
    SURPLUS_VALUE, ON_COURT_VALUE, AGE_ADJUSTED_SURPLUS).

    Parameters
    ----------
    df : DataFrame
        Output of compute_player_value().
    exclude_lakers : bool
        If True, Lakers players are excluded from the target ranking but their
        archetype scores are still computed.

    Returns
    -------
    df with BEST_ARCHETYPE, ARCHETYPE_DISTANCE, DIST_* columns,
    ROLL_GRAVITY_SCORE, RIM_PROTECTION_SCORE, BIG_MAN_FIT_SCORE,
    COMBINED_TARGET_SCORE, COMBINED_TARGET_RANK appended.
    """
    df = df.copy()
    qualifying = _qualifying_mask(df)

    # ------------------------------------------------------------------
    # Approach A: historical archetype profiling
    # ------------------------------------------------------------------

    # Step 1: Fit scaler on current 2025-26 qualifying players
    available_features = [f for f in ARCHETYPE_FEATURES if f in df.columns]
    missing_features = [f for f in ARCHETYPE_FEATURES if f not in df.columns]
    if missing_features:
        logger.warning("Archetype features missing from dataset: %s", missing_features)

    scaler, X_scaled, complete_idx = _fit_scaler_on_current(df, qualifying)

    # Step 2: Build archetype centroids from historical Dallas teammates
    centroids, validation_rows = build_archetype_centroids(scaler, available_features)
    # Store as plain Python objects (not DataFrames) to avoid pd.concat attr comparison bugs
    df.attrs["archetype_validation_rows"] = validation_rows
    df.attrs["archetype_centroid_names"] = list(centroids.keys())
    df.attrs["archetype_features"] = available_features

    # Step 3: Classify current players by archetype distance
    df = _classify_by_archetype(df, qualifying, centroids, scaler, X_scaled, complete_idx)

    # ------------------------------------------------------------------
    # Approach B: roll gravity + rim protection + big-man fit
    # ------------------------------------------------------------------
    df["ROLL_GRAVITY_SCORE"] = _compute_roll_gravity(df, qualifying)
    has_rg = df["ROLL_GRAVITY_SCORE"].notna()
    df["ROLL_GRAVITY_RANK"] = np.nan
    if has_rg.any():
        df.loc[has_rg, "ROLL_GRAVITY_RANK"] = df.loc[has_rg, "ROLL_GRAVITY_SCORE"].rank(
            ascending=False, method="min"
        )

    df["RIM_PROTECTION_SCORE"] = _compute_rim_protection(df, qualifying)
    has_rp = df["RIM_PROTECTION_SCORE"].notna()
    df["RIM_PROTECTION_RANK"] = np.nan
    if has_rp.any():
        df.loc[has_rp, "RIM_PROTECTION_RANK"] = df.loc[has_rp, "RIM_PROTECTION_SCORE"].rank(
            ascending=False, method="min"
        )

    both_available = df["ROLL_GRAVITY_SCORE"].notna() & df["RIM_PROTECTION_SCORE"].notna()
    df["BIG_MAN_FIT_SCORE"] = np.nan
    df.loc[both_available, "BIG_MAN_FIT_SCORE"] = (
        df.loc[both_available, "ROLL_GRAVITY_SCORE"]
        + df.loc[both_available, "RIM_PROTECTION_SCORE"]
    )
    has_bmf = df["BIG_MAN_FIT_SCORE"].notna()
    df["BIG_MAN_FIT_RANK"] = np.nan
    if has_bmf.any():
        df.loc[has_bmf, "BIG_MAN_FIT_RANK"] = df.loc[has_bmf, "BIG_MAN_FIT_SCORE"].rank(
            ascending=False, method="min"
        )

    # ------------------------------------------------------------------
    # Combined target score
    # ------------------------------------------------------------------
    df["COMBINED_TARGET_SCORE"] = _compute_combined_target(df)

    target_mask = df["COMBINED_TARGET_SCORE"].notna()
    if exclude_lakers and "TEAM_ABBREVIATION" in df.columns:
        target_mask &= df["TEAM_ABBREVIATION"] != "LAL"

    df["COMBINED_TARGET_RANK"] = np.nan
    if target_mask.any():
        df.loc[target_mask, "COMBINED_TARGET_RANK"] = df.loc[
            target_mask, "COMBINED_TARGET_SCORE"
        ].rank(ascending=False, method="min")

    logger.info(
        "Archetype profiling complete. Archetypes: %s. Players classified: %d.",
        list(centroids.keys()),
        df["BEST_ARCHETYPE"].notna().sum(),
    )

    return df


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def archetype_top_players(
    df: pd.DataFrame,
    archetype: str,
    n: int = 10,
) -> pd.DataFrame:
    """
    Return top-n current players closest to the given archetype centroid,
    sorted by ARCHETYPE_DISTANCE ascending.
    """
    dist_col = f"DIST_{archetype.upper()}"
    cols = [
        "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE",
        dist_col, "ARCHETYPE_DISTANCE", "BEST_ARCHETYPE",
        "ON_COURT_VALUE", "SURPLUS_VALUE", "AGE_ADJUSTED_SURPLUS",
    ]
    available = [c for c in cols if c in df.columns]
    mask = df[dist_col].notna() if dist_col in df.columns else pd.Series(False, index=df.index)
    return (
        df[mask][available]
        .sort_values(dist_col, ascending=True)
        .head(n)
        .reset_index(drop=True)
    )


def archetype_centroid_profile(
    centroids: dict[str, np.ndarray],
    features: list[str],
    validation_rows: list[dict] | None = None,
) -> pd.DataFrame:
    """
    Return a DataFrame showing each archetype's centroid feature values (scaled space).

    Parameters mirror what compute_luka_complement() builds internally.
    Pass df.attrs['archetype_centroid_names'] to get the centroid dict keys.
    """
    if not centroids:
        return pd.DataFrame()

    rows = []
    for arch, centroid in centroids.items():
        n_players = 0
        if validation_rows:
            n_players = sum(
                1 for r in validation_rows
                if r.get("Archetype") == arch and r.get("Note") == "OK"
            )
        row = {"archetype": arch, "n_players": n_players}
        for i, feat in enumerate(features):
            row[feat] = round(float(centroid[i]), 3)
        rows.append(row)

    return pd.DataFrame(rows).set_index("archetype")


def get_archetype_members(
    df: pd.DataFrame,
    n_per_archetype: int = 10,
) -> pd.DataFrame:
    """
    Return the top-n players per archetype sorted by ON_COURT_VALUE descending.
    """
    if "BEST_ARCHETYPE" not in df.columns:
        return pd.DataFrame()

    cols = [
        "PLAYER_NAME", "TEAM_ABBREVIATION", "BEST_ARCHETYPE",
        "ARCHETYPE_DISTANCE", "ON_COURT_VALUE", "SURPLUS_VALUE",
    ]
    available = [c for c in cols if c in df.columns]
    mask = df["BEST_ARCHETYPE"].notna() & df["ON_COURT_VALUE"].notna()
    return (
        df[mask][available]
        .sort_values(["BEST_ARCHETYPE", "ON_COURT_VALUE"], ascending=[True, False])
        .groupby("BEST_ARCHETYPE")
        .head(n_per_archetype)
        .reset_index(drop=True)
    )
