"""
Luka Complement Score -- role-fit metric for star-complementary players.

Two complementary approaches per design principles (empirical first):

Approach A: Empirical K-means clustering on style features.
  Players are clustered on stylometric features without any pre-specified
  notion of what a "Luka complement" looks like. We then observe which
  cluster(s) known successful Luka teammates fall into and measure every
  other player's distance to those cluster centroids.

  Output columns:
    COMPLEMENT_CLUSTER         -- KMeans cluster assignment (int)
    COMPLEMENT_DISTANCE        -- Euclidean distance to nearest Luka complement centroid
    COMPLEMENT_FIT_RANK        -- rank among all players (1 = closest fit)
    IS_LUKA_COMPLEMENT_CLUSTER -- True if player is in a qualifying Luka complement cluster

Approach B: Big-man fit scores (three separate dimensions).
  ROLL_GRAVITY_SCORE = z(NORMAL_FG_PCT) + z(OREB_PCT) - z(USG_PCT)
    Measures lob threat / finishing ability: restricted-area FG%, offensive glass
    presence, and low usage (doesn't compete with Luka for creation reps).
    BLK% removed from this score -- rim protection is a separate dimension.

  RIM_PROTECTION_SCORE = z(BLK%) + z(DREB_PCT)
    Measures defensive anchor value: shot-blocking rate and defensive rebounding.
    Separately tracked because some bigs (Lively) excel as lob threats but not
    rim protectors, while others (Williams III) excel at both.

  BIG_MAN_FIT_SCORE = ROLL_GRAVITY_SCORE + RIM_PROTECTION_SCORE
    Combined score for identifying bigs who fill both roles Luka needs.

  Validation: Gafford, Lively, Hayes should score highly on roll gravity.
  Output columns: ROLL_GRAVITY_SCORE, ROLL_GRAVITY_RANK,
                  RIM_PROTECTION_SCORE, RIM_PROTECTION_RANK,
                  BIG_MAN_FIT_SCORE, BIG_MAN_FIT_RANK.

Combined output:
    Complement distance is used as a FILTER, not a continuous input.
    Players in complement clusters (or within COMPLEMENT_DISTANCE_THRESHOLD)
    are ranked by AGE_ADJUSTED_SURPLUS * durability penalty.
    Players outside the filter receive a 0.5 discount on their combined score.
    COMBINED_TARGET_SCORE   -- filter-then-rank combined score
    COMBINED_TARGET_RANK    -- rank (1 = top trade/FA target)

Constants at the top of this module control k, minimum sample thresholds,
and validation player IDs. Change those, not the logic.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"

# ---------------------------------------------------------------------------
# Constants -- tune these, not the logic
# ---------------------------------------------------------------------------

KMEANS_K = 8
KMEANS_RANDOM_STATE = 42
SILHOUETTE_K_RANGE = range(5, 13)  # evaluate k=5 through k=12 for cluster quality
COMPLEMENT_CLUSTER_MIN = 2         # clusters with >= this many validation players qualify

# Expanded validation set: historical Luka teammates, organized by archetype.
# Each entry: {player_id, season (when they played with Luka), archetype}
# For players confirmed in current 2025-26 dataset, season can be "2025-26".
# Archetype groupings: rim_runner, 3d_wing, secondary_creator, shooter.
EXPANDED_VALIDATION: dict[str, dict] = {
    # Rim Runners / Lob Threats
    "Daniel Gafford":    {"id": 1629655, "season": "2023-24", "archetype": "rim_runner"},
    "Dereck Lively II":  {"id": 1641726, "season": "2023-24", "archetype": "rim_runner"},
    "Jaxson Hayes":      {"id": 1629637, "season": "2025-26", "archetype": "rim_runner"},
    # 3&D Wings
    "PJ Washington":     {"id": 1629023, "season": "2023-24", "archetype": "3d_wing"},
    "Dorian Finney-Smith":{"id": 1627827, "season": "2022-23", "archetype": "3d_wing"},
    "Derrick Jones Jr":  {"id": 1628407, "season": "2023-24", "archetype": "3d_wing"},
    # Secondary Creators
    "Kyrie Irving":      {"id": 202681,  "season": "2023-24", "archetype": "secondary_creator"},
    "Jalen Brunson":     {"id": 1628973, "season": "2021-22", "archetype": "secondary_creator"},
    "Spencer Dinwiddie": {"id": 203915,  "season": "2021-22", "archetype": "secondary_creator"},
    # Pure Shooters
    "Tim Hardaway Jr":   {"id": 1627758, "season": "2022-23", "archetype": "shooter"},
    "Maxi Kleber":       {"id": 1628467, "season": "2022-23", "archetype": "shooter"},
}

# Style features used for clustering. Mix of nba_api and BBRef columns.
# Using nba_api columns where possible for better coverage across all players.
CLUSTER_FEATURES: list[str] = [
    "USG_PCT",               # nba_api: usage rate
    "CS_CATCH_SHOOT_FG3_PCT",# nba_api: catch-and-shoot 3PT%
    "CS_CATCH_SHOOT_FG3A",   # nba_api: catch-and-shoot 3PT volume
    "PCT_PLUSMINUS",         # nba_api: defensive matchup plus/minus
    "three_PAr",             # BBRef: 3-point attempt rate (3PA / FGA)
    "AST_PCT",               # nba_api: assist percentage
    "STL%",                  # BBRef: steal percentage
    "BLK%",                  # BBRef: block percentage
    "TOV%",                  # BBRef: turnover percentage
    "OREB_PCT",              # nba_api: offensive rebound rate
    "DREB_PCT",              # nba_api: defensive rebound rate
]

# Roll gravity features: lob threat / finishing ability (signs: +1 additive, -1 subtracted)
ROLL_GRAVITY_FEATURES: dict[str, int] = {
    "NORMAL_FG_PCT": +1,  # proxy for restricted-area finishing
    "OREB_PCT": +1,       # offensive rebounding / roll gravity
    "USG_PCT": -1,        # lower usage = less ball-dominant = better complement
    # BLK% removed -- rim protection is tracked separately in RIM_PROTECTION_FEATURES
}

# Rim protection features: defensive anchor value
RIM_PROTECTION_FEATURES: dict[str, int] = {
    "BLK%": +1,      # shot-blocking rate
    "DREB_PCT": +1,  # defensive rebounding
}

# Minimum thresholds for inclusion in clustering / roll gravity
CLUSTER_MIN_GP = 20
CLUSTER_MIN_MIN = 15.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _zscore_series(series: pd.Series) -> pd.Series:
    """Z-score a series; returns NaN for all-NaN input."""
    mu = series.mean()
    sigma = series.std(ddof=1)
    if sigma == 0 or pd.isna(sigma):
        return pd.Series(np.nan, index=series.index)
    return (series - mu) / sigma


def _qualifying_mask(df: pd.DataFrame) -> pd.Series:
    return (df["GP"] >= CLUSTER_MIN_GP) & (df["MIN"] >= CLUSTER_MIN_MIN)


# ---------------------------------------------------------------------------
# Approach A: K-means clustering
# ---------------------------------------------------------------------------


def _impute_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute zero-attempt catch-and-shoot stats before clustering.

    CS_CATCH_SHOOT_FG3_PCT and CS_CATCH_SHOOT_FG3A are NaN for players who
    never attempt catch-and-shoot 3s (e.g. Gafford). The NaN is an artifact of
    0/0 -- the player's 3PT% and volume are genuinely 0, not unknown.
    Fill with 0 so rim-running bigs aren't excluded from clustering.
    """
    df = df.copy()
    cs_fga = "CS_CATCH_SHOOT_FGA"
    cs_3a = "CS_CATCH_SHOOT_FG3A"
    cs_3pct = "CS_CATCH_SHOOT_FG3_PCT"

    # Players who appear in the shooting data (have some CS attempts) but no 3PA: fill 0
    has_cs_data = df[cs_fga].notna() if cs_fga in df.columns else pd.Series(False, index=df.index)
    if cs_3a in df.columns:
        df.loc[has_cs_data & df[cs_3a].isna(), cs_3a] = 0.0
    if cs_3pct in df.columns:
        df.loc[has_cs_data & df[cs_3pct].isna(), cs_3pct] = 0.0

    return df


def _load_historical_features(player_id: int, season: str, feature_cols: list[str]) -> dict | None:
    """
    Load a validation player's cluster features from the nba_api advanced stats cache.

    Only features available in the advanced stats parquet (USG_PCT, AST_PCT,
    OREB_PCT, DREB_PCT, NET_RATING, TS_PCT) can be retrieved this way.
    Features absent from the cache return NaN for that player.

    Returns a dict {feature: value} or None if the player isn't found for that season.
    """
    cache_file = _CACHE_DIR / f"nba_player_advanced_{season}.parquet"
    if not cache_file.exists():
        logger.warning("No advanced stats cache for %s (%s)", season, cache_file.name)
        return None

    adv = pd.read_parquet(cache_file)
    row = adv[adv["PLAYER_ID"] == player_id]
    if row.empty:
        return None

    row = row.iloc[0]
    return {col: row.get(col, np.nan) for col in feature_cols}


def _compute_silhouette_scores(X: np.ndarray, k_range: range) -> dict[int, float]:
    """Compute silhouette score for each k in k_range. Returns {k: score}."""
    scores: dict[int, float] = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=KMEANS_RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        try:
            score = silhouette_score(X, labels)
        except ValueError:
            score = float("nan")
        scores[k] = round(float(score), 4)
        logger.info("Silhouette score k=%d: %.4f", k, score)
    return scores


def _fit_complement_clusters(
    df: pd.DataFrame,
    qualifying: pd.Series,
) -> tuple:
    """
    Fit K-means on qualifying players. Returns:
      - cluster labels Series (NaN for non-qualifying / missing features)
      - cluster centroids array (scaled space)
      - list of Luka complement cluster IDs
      - fitted scaler
      - scaled feature matrix X
      - index of complete-feature qualifying players
      - cluster input DataFrame for complete players
      - list of available feature names
      - dict of silhouette scores {k: score}

    Validation uses EXPANDED_VALIDATION: historical Luka teammates are looked up
    from their specified season's nba_api advanced cache, then their cluster
    assignment is predicted using the fitted scaler + KMeans model.
    """
    from collections import Counter

    df = _impute_cluster_features(df)

    available_features = [f for f in CLUSTER_FEATURES if f in df.columns]
    missing = [f for f in CLUSTER_FEATURES if f not in df.columns]
    if missing:
        logger.warning("Clustering features not in dataset: %s", missing)

    cluster_input = df.loc[qualifying, available_features].copy()
    complete_mask = cluster_input.notna().all(axis=1)

    if complete_mask.sum() < KMEANS_K:
        logger.warning(
            "Only %d players have complete cluster features (need >= %d). "
            "Cannot cluster.", complete_mask.sum(), KMEANS_K
        )
        labels = pd.Series(np.nan, index=df.index)
        empty = np.array([])
        return labels, empty, [], None, empty, cluster_input.index[:0], cluster_input.head(0), available_features, {}

    scaler = StandardScaler()
    X = scaler.fit_transform(cluster_input[complete_mask])
    complete_global_idx = cluster_input[complete_mask].index

    # Compute silhouette scores across k range to justify choice of k
    silhouette_scores = _compute_silhouette_scores(X, SILHOUETTE_K_RANGE)
    best_k = max(silhouette_scores, key=silhouette_scores.get)
    logger.info(
        "Silhouette scores: %s | Best k=%d (%.4f) | Using k=%d",
        silhouette_scores, best_k, silhouette_scores[best_k], KMEANS_K,
    )

    km = KMeans(n_clusters=KMEANS_K, random_state=KMEANS_RANDOM_STATE, n_init=20)
    km.fit(X)

    # Assign labels only to complete-feature qualifying players
    labels = pd.Series(np.nan, index=df.index, dtype="float64")
    labels.loc[complete_global_idx] = km.labels_.astype(float)

    # ----------------------------------------------------------------
    # Identify Luka complement clusters using multi-season validation.
    # For players in the current dataset: use their cluster assignment.
    # For historical players: load their advanced stats from the cache,
    #   transform with the fitted scaler, and predict their cluster.
    # ----------------------------------------------------------------
    archetype_cluster_hits: Counter = Counter()  # (archetype, cluster_id) -> count
    cluster_counts: Counter = Counter()           # cluster_id -> total hits

    for player_name, info in EXPANDED_VALIDATION.items():
        pid = info["id"]
        season = info["season"]
        archetype = info["archetype"]

        # Try current dataset first (player may be in 2025-26 data)
        pid_mask = df["PLAYER_ID"] == pid
        if pid_mask.any():
            cluster_id = labels.loc[pid_mask].values[0]
            if not np.isnan(cluster_id):
                cid = int(cluster_id)
                cluster_counts[cid] += 1
                archetype_cluster_hits[(archetype, cid)] += 1
                logger.info("Validation %s (current) -> cluster %d [%s]", player_name, cid, archetype)
                continue
            else:
                logger.warning("Validation %s in dataset but has NaN cluster (missing features)", player_name)

        # Player not in current dataset (or missing features): load historical stats
        hist = _load_historical_features(pid, season, available_features)
        if hist is None:
            logger.warning("Validation %s: no data for %s -- skipping", player_name, season)
            continue

        # Build a feature vector; impute missing features with scaler mean (-> ~0 in scaled space)
        feat_vec = np.array([
            hist.get(col, np.nan) if not np.isnan(hist.get(col, np.nan)) else scaler.mean_[i]
            for i, col in enumerate(available_features)
        ]).reshape(1, -1)
        feat_vec_scaled = scaler.transform(feat_vec)
        cid = int(km.predict(feat_vec_scaled)[0])
        cluster_counts[cid] += 1
        archetype_cluster_hits[(archetype, cid)] += 1
        logger.info("Validation %s (%s) -> cluster %d [%s]", player_name, season, cid, archetype)

    # A cluster qualifies as a complement cluster if it contains >= COMPLEMENT_CLUSTER_MIN
    # players from the same archetype, OR >= COMPLEMENT_CLUSTER_MIN total validation players.
    archetype_qualified = {
        cid for (arch, cid), n in archetype_cluster_hits.items() if n >= COMPLEMENT_CLUSTER_MIN
    }
    total_qualified = {cid for cid, n in cluster_counts.items() if n >= COMPLEMENT_CLUSTER_MIN}
    complement_clusters = list(archetype_qualified | total_qualified)

    if not complement_clusters and cluster_counts:
        logger.warning(
            "No clusters met threshold=%d. Falling back to each validation player's cluster.",
            COMPLEMENT_CLUSTER_MIN,
        )
        complement_clusters = list(cluster_counts.keys())

    logger.info(
        "Luka complement clusters: %s (%d archetypes, %d total validation hits)",
        complement_clusters, len(archetype_qualified), sum(cluster_counts.values()),
    )

    centroids_scaled = km.cluster_centers_
    return (
        labels, centroids_scaled, complement_clusters, scaler,
        X, complete_global_idx, cluster_input[complete_mask],
        available_features, silhouette_scores,
    )


def _compute_complement_distance(
    df: pd.DataFrame,
    labels: pd.Series,
    complement_clusters: list[int],
    centroids_scaled: np.ndarray,
    scaler: object,
    X_scaled: np.ndarray,
    complete_global_idx: pd.Index,
    available_features: list[str],
) -> pd.Series:
    """
    Compute Euclidean distance from each qualifying player to the nearest
    Luka complement cluster centroid (in scaled feature space).
    """
    distances = pd.Series(np.nan, index=df.index)

    if not complement_clusters:
        return distances

    complement_centroids = centroids_scaled[complement_clusters]

    for i, global_idx in enumerate(complete_global_idx):
        player_vec = X_scaled[i].reshape(1, -1)
        dists = np.linalg.norm(complement_centroids - player_vec, axis=1)
        distances.loc[global_idx] = float(dists.min())

    return distances


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


COMPLEMENT_DISTANCE_THRESHOLD = 2.0  # players within this distance are in the complement filter
COMPLEMENT_OUTSIDE_DISCOUNT = 0.5   # multiplier applied to players outside the filter


def _compute_combined_target(df: pd.DataFrame) -> pd.Series:
    """
    Filter-then-rank combined target score.

    Complement fit is a GATE, not a continuous input:
      - Step 1: Flag players in complement clusters OR within COMPLEMENT_DISTANCE_THRESHOLD.
      - Step 2: Base score = AGE_ADJUSTED_SURPLUS (or SURPLUS_VALUE, or FA_VALUE).
                Applied as a z-score across all players.
      - Step 3: Players outside the filter receive a COMPLEMENT_OUTSIDE_DISCOUNT penalty.

    This prevents low-value complement fits from outranking high-value non-complements.
    """
    # Determine base value: prefer age-adjusted surplus, fall back to surplus, then FA value
    if "AGE_ADJUSTED_SURPLUS" in df.columns:
        value_series = df["AGE_ADJUSTED_SURPLUS"].where(
            ~df["IS_FREE_AGENT"].fillna(True),
            df.get("FA_VALUE", pd.Series(np.nan, index=df.index)),
        )
    else:
        value_col = np.where(
            ~df["IS_FREE_AGENT"].fillna(True),
            df.get("SURPLUS_VALUE", pd.Series(np.nan, index=df.index)),
            df.get("FA_VALUE", pd.Series(np.nan, index=df.index)),
        )
        value_series = pd.Series(value_col, index=df.index)

    value_z = _zscore_series(value_series.dropna()).reindex(df.index)
    combined = value_z.copy()

    # Apply complement filter
    if "COMPLEMENT_DISTANCE" in df.columns and "IS_LUKA_COMPLEMENT_CLUSTER" in df.columns:
        in_filter = (
            df["IS_LUKA_COMPLEMENT_CLUSTER"]
            | (df["COMPLEMENT_DISTANCE"] < COMPLEMENT_DISTANCE_THRESHOLD)
        )
        outside_filter = ~in_filter & combined.notna()
        combined.loc[outside_filter] *= COMPLEMENT_OUTSIDE_DISCOUNT
        logger.info(
            "Complement filter: %d players in-filter, %d outside (%.1fx discount)",
            in_filter.sum(), outside_filter.sum(), COMPLEMENT_OUTSIDE_DISCOUNT,
        )
    else:
        logger.warning("COMPLEMENT_DISTANCE not present; no complement filter applied")

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

    This function expects that compute_player_value() has already been called
    (needs IS_FREE_AGENT, SURPLUS_VALUE, FA_VALUE, ON_COURT_VALUE).

    Parameters
    ----------
    df : DataFrame
        Output of compute_player_value() -- merged + valued player dataset.
    exclude_lakers : bool
        If True, Lakers players are excluded from the target ranking but their
        cluster/distance scores are still computed.

    Returns
    -------
    df with COMPLEMENT_CLUSTER, COMPLEMENT_DISTANCE, COMPLEMENT_FIT_RANK,
    IS_LUKA_COMPLEMENT_CLUSTER, ROLL_GRAVITY_SCORE, ROLL_GRAVITY_RANK,
    COMBINED_TARGET_SCORE, COMBINED_TARGET_RANK columns appended.
    """
    df = df.copy()
    qualifying = _qualifying_mask(df)

    # ------------------------------------------------------------------
    # Approach A: clustering
    # ------------------------------------------------------------------
    cluster_result = _fit_complement_clusters(df, qualifying)
    (
        labels, centroids_scaled, complement_clusters, scaler,
        X_scaled, complete_global_idx, _, available_features, silhouette_scores,
    ) = cluster_result
    df.attrs["silhouette_scores"] = silhouette_scores  # expose for notebook display

    df["COMPLEMENT_CLUSTER"] = labels.where(labels.notna(), other=np.nan)

    complement_distances = _compute_complement_distance(
        df, labels, complement_clusters, centroids_scaled,
        scaler, X_scaled, complete_global_idx, available_features,
    )
    df["COMPLEMENT_DISTANCE"] = complement_distances

    df["IS_LUKA_COMPLEMENT_CLUSTER"] = (
        df["COMPLEMENT_CLUSTER"].isin([float(c) for c in complement_clusters])
    )

    # Rank by distance ascending (lower distance = better fit)
    has_dist = df["COMPLEMENT_DISTANCE"].notna()
    df["COMPLEMENT_FIT_RANK"] = np.nan
    if has_dist.any():
        df.loc[has_dist, "COMPLEMENT_FIT_RANK"] = df.loc[has_dist, "COMPLEMENT_DISTANCE"].rank(
            ascending=True, method="min"
        )

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

    # BIG_MAN_FIT_SCORE: sum of both scores where both are available
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
        "Complement clustering complete. Complement clusters: %s. "
        "Players with distance score: %d.",
        complement_clusters,
        has_dist.sum(),
    )

    return df


def cluster_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return mean z-scores of cluster features per cluster.

    Useful for labeling clusters (e.g. 'low-usage shooters', 'rim protectors').
    """
    if "COMPLEMENT_CLUSTER" not in df.columns:
        return pd.DataFrame()

    available = [f for f in CLUSTER_FEATURES if f in df.columns]
    cluster_data = df[df["COMPLEMENT_CLUSTER"].notna()].copy()
    if cluster_data.empty:
        return pd.DataFrame()

    for feat in available:
        cluster_data[f"_z_{feat}"] = _zscore_series(cluster_data[feat])

    z_cols = [f"_z_{feat}" for feat in available]
    profile = cluster_data.groupby("COMPLEMENT_CLUSTER")[z_cols].mean()
    profile.columns = available
    profile.index.name = "cluster"
    return profile.round(2)


def get_complement_cluster_members(
    df: pd.DataFrame,
    n_per_cluster: int = 10,
) -> pd.DataFrame:
    """
    Return the top-n players per Luka complement cluster, sorted by ON_COURT_VALUE.
    """
    if "IS_LUKA_COMPLEMENT_CLUSTER" not in df.columns:
        return pd.DataFrame()

    cols = [
        "PLAYER_NAME", "TEAM_ABBREVIATION", "COMPLEMENT_CLUSTER",
        "COMPLEMENT_DISTANCE", "ON_COURT_VALUE", "SURPLUS_VALUE",
    ]
    available = [c for c in cols if c in df.columns]
    mask = df["IS_LUKA_COMPLEMENT_CLUSTER"] & df["ON_COURT_VALUE"].notna()
    return (
        df[mask][available]
        .sort_values(["COMPLEMENT_CLUSTER", "ON_COURT_VALUE"], ascending=[True, False])
        .groupby("COMPLEMENT_CLUSTER")
        .head(n_per_cluster)
        .reset_index(drop=True)
    )
