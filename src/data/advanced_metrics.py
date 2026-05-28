"""
Ingestion of public advanced metrics not available through nba_api.

Sources implemented:
  - EPM (Estimated Plus-Minus): dunksandthrees.com/epm
    Try scraping first; if the site blocks or changes structure, fall back to
    load_epm() which reads a manually downloaded CSV/JSON export.
  - RAPTOR (archived): FiveThirtyEight GitHub CSV - historical seasons only.

All functions return DataFrames with a standardized column set so they can be
joined to the main player dataset in merge.py without per-source renaming logic.

Manual download instructions (EPM):
  1. Go to https://dunksandthrees.com/epm
  2. Filter to the season you want
  3. Use browser DevTools -> Network tab to find the JSON data endpoint, or
     download their CSV export if available
  4. Save to data/raw/epm_{season}.csv and call load_epm()
"""
from __future__ import annotations

import logging
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from .cache import get_or_fetch
from .config import BBALL_REF_DELAY_SECONDS, DEFAULT_REQUEST_TIMEOUT, DUNKS_AND_THREES_URL

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://dunksandthrees.com/",
}

# Expected output columns after standardization
EPM_COLUMNS = [
    "PLAYER_NAME_EPM",
    "TEAM_EPM",
    "SEASON_EPM",
    "EPM",
    "EPMO",   # offensive EPM
    "EPMD",   # defensive EPM
    "GP_EPM",
    "MIN_EPM",
]

RAPTOR_COLUMNS = [
    "PLAYER_NAME_RAPTOR",
    "TEAM_RAPTOR",
    "SEASON_RAPTOR",
    "RAPTOR_TOTAL",
    "RAPTOR_OFF",
    "RAPTOR_DEF",
    "WAR_TOTAL",
]


# ---------------------------------------------------------------------------
# EPM - Dunks & Threes
# ---------------------------------------------------------------------------


def get_epm(season: str) -> pd.DataFrame:
    """
    Fetch EPM (Estimated Plus-Minus) data for a given season.

    Tries the Dunks & Threes JSON API first, then falls back to HTML scraping.
    Returns empty DataFrame with a warning if both fail - use load_epm() with a
    manually downloaded file as the fallback.

    Parameters
    ----------
    season : str
        Season in nba_api format, e.g. "2024-25".

    Returns
    -------
    DataFrame with columns: PLAYER_NAME_EPM, TEAM_EPM, SEASON_EPM, EPM, EPMO, EPMD,
    GP_EPM, MIN_EPM. Returns empty DataFrame on failure.
    """
    cache_key = f"epm_{season}"
    try:
        return get_or_fetch(cache_key, lambda: _fetch_epm(season))
    except Exception as exc:
        logger.warning("get_epm(%s) cache layer failed: %s", season, exc)
        return pd.DataFrame(columns=EPM_COLUMNS)


def _fetch_epm(season: str) -> pd.DataFrame:
    """Internal: attempt to fetch EPM from Dunks & Threes."""
    # Try JSON API endpoint first (season format varies - try both "2024-25" and "2025")
    season_year = int(season[:4]) + 1  # "2024-25" -> 2025

    api_candidates = [
        f"https://dunksandthrees.com/api/epm?season={season_year}",
        f"https://dunksandthrees.com/api/players?season={season_year}&metric=epm",
        f"https://dunksandthrees.com/api/epm?year={season_year}",
    ]

    for api_url in api_candidates:
        df = _try_json_endpoint(api_url, season)
        if not df.empty:
            return df

    # Fall back to HTML scraping
    time.sleep(BBALL_REF_DELAY_SECONDS)
    df = _try_html_scrape(DUNKS_AND_THREES_URL, season)
    if not df.empty:
        return df

    logger.warning(
        "EPM scraping failed for %s. "
        "Download manually from https://dunksandthrees.com/epm and use load_epm().",
        season,
    )
    return pd.DataFrame(columns=EPM_COLUMNS)


def _try_json_endpoint(url: str, season: str) -> pd.DataFrame:
    """Try fetching EPM data from a JSON API endpoint."""
    try:
        time.sleep(1.0)
        resp = requests.get(url, headers=_HEADERS, timeout=DEFAULT_REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return pd.DataFrame()

        content_type = resp.headers.get("Content-Type", "")
        if "json" not in content_type:
            return pd.DataFrame()

        data = resp.json()
        # Handle both list-of-records and {"players": [...]} shapes
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = data.get("players", data.get("data", data.get("results", [])))
        else:
            return pd.DataFrame()

        if not records:
            return pd.DataFrame()

        df = pd.json_normalize(records)
        return _standardize_epm_columns(df, season)

    except Exception as exc:
        logger.debug("JSON endpoint %s failed: %s", url, exc)
        return pd.DataFrame()


def _try_html_scrape(url: str, season: str) -> pd.DataFrame:
    """Try scraping EPM from the Dunks & Threes HTML page."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=DEFAULT_REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        logger.debug("HTML scrape failed for %s: %s", url, exc)
        return pd.DataFrame()

    try:
        tables = pd.read_html(StringIO(resp.text))
        if not tables:
            return pd.DataFrame()
        # Find the table most likely to contain EPM data
        for table in tables:
            cols_lower = [str(c).lower() for c in table.columns]
            if any("epm" in c for c in cols_lower):
                return _standardize_epm_columns(table, season)
        return pd.DataFrame()
    except Exception as exc:
        logger.debug("read_html failed for EPM page: %s", exc)
        return pd.DataFrame()


def _standardize_epm_columns(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Map raw EPM column names (vary by API version) to our canonical set."""
    col_map = _build_epm_column_map(df.columns.tolist())

    out = pd.DataFrame()
    out["PLAYER_NAME_EPM"] = df[col_map["name"]].astype(str).str.strip() if col_map.get("name") else ""
    out["TEAM_EPM"] = df[col_map["team"]].astype(str).str.strip() if col_map.get("team") else ""
    out["SEASON_EPM"] = season
    out["EPM"] = pd.to_numeric(df[col_map["epm"]], errors="coerce") if col_map.get("epm") else float("nan")
    out["EPMO"] = pd.to_numeric(df[col_map["epmo"]], errors="coerce") if col_map.get("epmo") else float("nan")
    out["EPMD"] = pd.to_numeric(df[col_map["epmd"]], errors="coerce") if col_map.get("epmd") else float("nan")
    out["GP_EPM"] = pd.to_numeric(df[col_map["gp"]], errors="coerce") if col_map.get("gp") else float("nan")
    out["MIN_EPM"] = pd.to_numeric(df[col_map["min"]], errors="coerce") if col_map.get("min") else float("nan")

    out = out[out["PLAYER_NAME_EPM"].str.strip() != ""].reset_index(drop=True)
    logger.info("EPM: standardized %d rows for %s", len(out), season)
    return out


def _build_epm_column_map(columns: list[str]) -> dict[str, str]:
    """Heuristically map raw column names to canonical slots."""
    cols_lower = {c.lower(): c for c in columns}
    result: dict[str, str] = {}

    for slot, keywords in [
        ("name", ["player", "name", "player_name"]),
        ("team", ["team", "tm", "franchise"]),
        ("epm", ["epm", "total_epm", "estimated_plus_minus"]),
        ("epmo", ["o_epm", "off_epm", "epmo", "offensive_epm", "epm_offense"]),
        ("epmd", ["d_epm", "def_epm", "epmd", "defensive_epm", "epm_defense"]),
        ("gp", ["gp", "games", "games_played", "g"]),
        ("min", ["min", "minutes", "mpg", "mp"]),
    ]:
        for kw in keywords:
            if kw in cols_lower:
                result[slot] = cols_lower[kw]
                break

    return result


def load_epm(filepath: str | Path | None = None, season: str = "2024-25") -> pd.DataFrame:
    """
    Load EPM data from a manually downloaded file (CSV or JSON).

    Primary path: data/raw/epm_{season}.csv (see data/raw/MANUAL_DOWNLOAD_GUIDE.md).
    Pass an explicit filepath to override.

    Parameters
    ----------
    filepath : str, Path, or None
        Path to a CSV or JSON file. If None, defaults to data/raw/epm_{season}.csv.
    season : str
        Season string used to build the default filename, e.g. "2024-25".

    Returns
    -------
    DataFrame with standardized EPM columns.

    Raises
    ------
    FileNotFoundError if the resolved file does not exist.
    """
    from .config import RAW_DIR
    path = Path(filepath) if filepath is not None else RAW_DIR / f"epm_{season}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"EPM file not found: {path}\n"
            "See data/raw/MANUAL_DOWNLOAD_GUIDE.md for download instructions."
        )

    if path.suffix.lower() == ".json":
        with open(path) as f:
            import json
            data = json.load(f)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.json_normalize(data)
    else:
        df = pd.read_csv(path)

    # Infer season from filename if present (e.g., "epm_2024-25.csv")
    season_guess = ""
    for part in path.stem.split("_"):
        if "-" in part and len(part) == 7:
            season_guess = part
            break

    df = _standardize_epm_columns(df, season_guess)
    logger.info("Loaded %d EPM rows from %s", len(df), path.name)
    return df


# ---------------------------------------------------------------------------
# RAPTOR - FiveThirtyEight archive
# ---------------------------------------------------------------------------


def get_raptor(refresh: bool = False) -> pd.DataFrame:
    """
    Load archived RAPTOR data from the FiveThirtyEight GitHub CSV.

    This is historical data (seasons through 2022-23); it is not updated.
    Covers: RAPTOR_OFF, RAPTOR_DEF, RAPTOR_TOTAL, WAR, pace_impact.

    Returns
    -------
    DataFrame with columns: PLAYER_NAME_RAPTOR, TEAM_RAPTOR, SEASON_RAPTOR,
    RAPTOR_TOTAL, RAPTOR_OFF, RAPTOR_DEF, WAR_TOTAL.
    Returns empty DataFrame with a warning on failure.
    """
    from .config import RAPTOR_ARCHIVE_URL

    cache_key = "raptor_archive"
    try:
        return get_or_fetch(cache_key, lambda: _fetch_raptor(RAPTOR_ARCHIVE_URL), refresh=refresh)
    except Exception as exc:
        logger.warning("get_raptor() failed: %s", exc)
        return pd.DataFrame(columns=RAPTOR_COLUMNS)


def _fetch_raptor(url: str) -> pd.DataFrame:
    """Fetch and standardize the FiveThirtyEight RAPTOR CSV."""
    try:
        df = pd.read_csv(url)
    except Exception as exc:
        logger.warning("RAPTOR CSV fetch failed: %s", exc)
        return pd.DataFrame(columns=RAPTOR_COLUMNS)

    out = pd.DataFrame()
    out["PLAYER_NAME_RAPTOR"] = df.get("player_name", df.get("name", "")).astype(str).str.strip()
    out["TEAM_RAPTOR"] = df.get("team", "").astype(str).str.strip()
    out["SEASON_RAPTOR"] = df.get("season", "").astype(str)

    out["RAPTOR_OFF"] = pd.to_numeric(df.get("raptor_offense", df.get("raptor_off")), errors="coerce")
    out["RAPTOR_DEF"] = pd.to_numeric(df.get("raptor_defense", df.get("raptor_def")), errors="coerce")
    out["RAPTOR_TOTAL"] = pd.to_numeric(df.get("raptor_total"), errors="coerce")
    out["WAR_TOTAL"] = pd.to_numeric(df.get("war_total", df.get("war")), errors="coerce")

    # FiveThirtyEight uses season end year as an integer
    # Convert to nba_api string format: 2023 -> "2022-23"
    if "season" in df.columns:
        def _season_str(yr):
            try:
                y = int(yr)
                return f"{y - 1}-{str(y)[-2:]}"
            except (ValueError, TypeError):
                return str(yr)
        out["SEASON_RAPTOR"] = df["season"].apply(_season_str)

    out = out[out["PLAYER_NAME_RAPTOR"].str.strip() != ""].reset_index(drop=True)
    logger.info("RAPTOR archive: loaded %d rows", len(out))
    return out
