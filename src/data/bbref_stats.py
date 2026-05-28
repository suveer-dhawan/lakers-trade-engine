"""
Basketball Reference scraping for advanced stats and per-100 possessions data.

Rate limit: 20 requests/minute -> use BBALL_REF_DELAY_SECONDS from config (3.5s).
Cache-first via get_or_fetch: never scrapes if parquet already exists.

BBRef quirks handled here:
  - Some tables are wrapped in HTML comments (JS rendering workaround); we unwrap them.
  - Repeated header rows every 20 players (tr class="thead") land as data rows
    with Rk="Rk" -- filtered out after parsing.
  - Mid-season trades: player appears as TOT + per-team rows. We keep TOT, drop
    team-specific duplicates for that player.
  - BBRef player IDs extracted from <a href="/players/x/PLAYERID.html"> anchors.
  - Spacer columns (Unnamed: N) are dropped.
  - WS/48 arrives as "WS/48" -- renamed to WS_48 for parquet safety.
"""
from __future__ import annotations

import io
import logging
import re
import time
from typing import Callable

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

from .cache import get_or_fetch
from .config import BBALL_REF_DELAY_SECONDS, DEFAULT_REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.basketball-reference.com"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _sleep() -> None:
    time.sleep(BBALL_REF_DELAY_SECONDS)


def _get_html(url: str) -> str | None:
    """Fetch raw HTML from url, returning None on failure."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=DEFAULT_REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        logger.warning("BBRef request failed for %s: %s", url, exc)
        return None


def _find_table(html: str, table_id: str) -> "BeautifulSoup | None":
    """
    Find a BBRef table by ID. Checks both direct HTML and HTML-comment wrappers
    (BBRef wraps some tables in <!-- ... --> to limit easy scraping).
    """
    soup = BeautifulSoup(html, "html.parser")

    table = soup.find("table", {"id": table_id})
    if table:
        return table

    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if table_id in comment:
            inner = BeautifulSoup(comment, "html.parser")
            table = inner.find("table", {"id": table_id})
            if table:
                return table

    return None


def _extract_bbref_ids(table) -> dict[str, str]:
    """
    Return {player_display_name: bbref_player_id} from <a> hrefs in the name column.
    BBRef hrefs look like /players/j/jamesle01.html.
    """
    ids: dict[str, str] = {}
    for row in table.find_all("tr"):
        td = row.find("td", {"data-stat": "player"})
        if td:
            a = td.find("a")
            if a and a.get("href"):
                player_id = a["href"].rstrip("/").split("/")[-1].replace(".html", "")
                ids[a.get_text(strip=True)] = player_id
    return ids


def _parse_table(table, bbref_ids: dict[str, str]) -> pd.DataFrame:
    """
    Convert a BBRef BeautifulSoup table to a cleaned DataFrame.

    Steps:
      1. pandas.read_html to parse (handles colspan headers gracefully)
      2. Flatten MultiIndex columns if present
      3. Drop spacer columns (Unnamed: N)
      4. Drop repeated header rows (Rk == "Rk")
      5. Drop rows with null player name
      6. Keep TOT for traded players, drop per-team duplicates
      7. Rename columns to safe identifiers (WS/48 → WS_48)
      8. Attach PLAYER_ID_BBREF
      9. Coerce numeric columns
    """
    try:
        dfs = pd.read_html(io.StringIO(str(table)))
    except Exception as exc:
        logger.warning("read_html failed: %s", exc)
        return pd.DataFrame()

    df = dfs[0]

    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(c) for c in col if not str(c).startswith("Unnamed")).strip("_")
            for col in df.columns.to_flat_index()
        ]

    # Drop spacer columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Determine the player name column
    name_col = next((c for c in ["Player", "PLAYER_NAME_BBREF"] if c in df.columns), None)
    if name_col is None:
        logger.warning("Could not identify player name column; columns: %s", list(df.columns))
        return pd.DataFrame()

    # Drop repeated header rows
    if "Rk" in df.columns:
        df = df[df["Rk"] != "Rk"].copy()

    # Drop empty name rows
    df = df[df[name_col].notna() & (df[name_col].astype(str).str.strip() != "")].copy()

    # Keep TOT rows for traded players, drop per-team duplicates
    if "Tm" in df.columns:
        traded = df[df["Tm"] == "TOT"][name_col].unique()
        df = df[~((df[name_col].isin(traded)) & (df["Tm"] != "TOT"))].copy()

    # Rename to canonical columns
    df = df.rename(columns={name_col: "PLAYER_NAME_BBREF", "Tm": "TEAM_BBREF"})

    # Safe column names: WS/48 → WS_48, 3PAr → three_PAr, etc.
    df = df.rename(columns={"WS/48": "WS_48", "3PAr": "three_PAr"})

    # Attach BBRef player IDs
    df["PLAYER_ID_BBREF"] = df["PLAYER_NAME_BBREF"].map(bbref_ids).fillna("")

    # Coerce numeric columns
    non_numeric = {"PLAYER_NAME_BBREF", "PLAYER_ID_BBREF", "TEAM_BBREF", "Pos"}
    for col in df.columns:
        if col not in non_numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.reset_index(drop=True)
    logger.info("Parsed %d player rows from BBRef table", len(df))
    return df


def _fetch_bbref_table(url: str, table_id: str) -> pd.DataFrame:
    _sleep()
    html = _get_html(url)
    if html is None:
        return pd.DataFrame()

    table = _find_table(html, table_id)
    if table is None:
        logger.warning("Table '%s' not found at %s", table_id, url)
        return pd.DataFrame()

    bbref_ids = _extract_bbref_ids(table)
    return _parse_table(table, bbref_ids)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_bbref_advanced_csv(season: str) -> pd.DataFrame:
    """
    Load BBRef advanced stats from a manually downloaded CSV.

    Parameters
    ----------
    season : str
        Season in nba_api format, e.g. "2024-25". Used to locate the file
        data/raw/bbref_advanced_{season}.csv.

    Returns
    -------
    DataFrame with the same column schema as get_advanced_stats() output:
    PLAYER_NAME_BBREF, PLAYER_ID_BBREF, TEAM_BBREF, Pos, Age, G, MP,
    PER, TS%, three_PAr, FTr, ORB%, DRB%, TRB%, AST%, STL%, BLK%,
    TOV%, USG%, OWS, DWS, WS, WS_48, OBPM, DBPM, BPM, VORP.
    Returns empty DataFrame with a warning if the file is not found.
    """
    from .config import RAW_DIR

    path = RAW_DIR / f"bbref_advanced_{season}.csv"
    if not path.exists():
        logger.warning("BBRef advanced CSV not found: %s", path)
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Strip any non-ASCII sort-indicator characters BBRef appends to column headers (e.g. sort arrows)
    df.columns = [re.sub(r"[^\x00-\x7F]+", "", c).strip() for c in df.columns]

    # Drop Rk and Awards columns
    df = df.drop(columns=["Rk", "Awards"], errors="ignore")

    # Drop repeated header rows (BBRef re-emits headers every 20 rows)
    if "Player" in df.columns:
        df = df[df["Player"] != "Player"].copy()

    # Strip asterisks BBRef appends to HOF-eligible player names
    if "Player" in df.columns:
        df["Player"] = df["Player"].astype(str).str.replace("*", "", regex=False).str.strip()

    # For traded players keep the TOT row, drop the per-team rows
    if "Team" in df.columns:
        traded = df[df["Team"] == "TOT"]["Player"].unique()
        df = df[~((df["Player"].isin(traded)) & (df["Team"] != "TOT"))].copy()

    # Rename to canonical column names matching the scraper's output
    df = df.rename(
        columns={
            "Player": "PLAYER_NAME_BBREF",
            "Player-additional": "PLAYER_ID_BBREF",
            "Team": "TEAM_BBREF",
            "WS/48": "WS_48",
            "3PAr": "three_PAr",
        }
    )

    # Coerce all non-identity columns to numeric
    non_numeric = {"PLAYER_NAME_BBREF", "PLAYER_ID_BBREF", "TEAM_BBREF", "Pos"}
    for col in df.columns:
        if col not in non_numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.reset_index(drop=True)
    logger.info("Loaded %d BBRef advanced rows from CSV (%s)", len(df), season)
    return df


def get_advanced_stats(season: int) -> pd.DataFrame:
    """
    Return BBRef advanced stats for a given season.

    Tries a manually downloaded CSV first (data/raw/bbref_advanced_{season}.csv),
    then falls back to live scraping with the cache layer.

    Parameters
    ----------
    season : int
        Ending year of the season (e.g., 2025 for 2024-25).

    Returns
    -------
    DataFrame with columns including: PLAYER_NAME_BBREF, PLAYER_ID_BBREF, TEAM_BBREF,
    Pos, Age, G, MP, PER, TS%, three_PAr, FTr, ORB%, DRB%, TRB%, AST%, STL%, BLK%,
    TOV%, USG%, OWS, DWS, WS, WS_48, OBPM, DBPM, BPM, VORP.

    Returns empty DataFrame with a warning on any failure.
    """
    # e.g. season=2025 -> "2024-25"
    season_str = f"{season - 1}-{str(season)[-2:]}"
    csv_df = load_bbref_advanced_csv(season_str)
    if not csv_df.empty:
        return csv_df

    # Fall back to scraping (cached)
    url = f"{_BASE_URL}/leagues/NBA_{season}_advanced.html"
    cache_key = f"bbref_advanced_{season}"
    try:
        return get_or_fetch(cache_key, lambda: _fetch_bbref_table(url, "advanced"))
    except Exception as exc:
        logger.warning("get_advanced_stats(%d) failed: %s", season, exc)
        return pd.DataFrame()


def get_per_100(season: int) -> pd.DataFrame:
    """
    Scrape the BBRef Per 100 Possessions table for a given season.

    Parameters
    ----------
    season : int
        Ending year of the season (e.g., 2025 for 2024-25).

    Returns
    -------
    DataFrame with per-100-possession rate stats for usage-normalized comparisons.
    Returns empty DataFrame with a warning on any failure.
    """
    url = f"{_BASE_URL}/leagues/NBA_{season}_per_poss.html"
    cache_key = f"bbref_per100_{season}"
    try:
        return get_or_fetch(cache_key, lambda: _fetch_bbref_table(url, "per_poss"))
    except Exception as exc:
        logger.warning("get_per_100(%d) failed: %s", season, exc)
        return pd.DataFrame()


def get_multi_season_bbref(fetch_fn: Callable[[int], pd.DataFrame], seasons: list[int]) -> pd.DataFrame:
    """
    Call fetch_fn(season) for each season (int ending year), concatenate results.

    The delay between calls is already embedded in each fetch_fn via _sleep().
    Adds a SEASON column (int) to each frame. Failed seasons are skipped with a warning.

    Parameters
    ----------
    fetch_fn : callable
        A function like get_advanced_stats or get_per_100 that takes an int season.
    seasons : list[int]
        Ending years, e.g. [2023, 2024, 2025].

    Returns
    -------
    Concatenated DataFrame with a leading SEASON column, or empty DataFrame if all fail.

    Example
    -------
    df = get_multi_season_bbref(get_advanced_stats, [2023, 2024, 2025])
    """
    frames = []
    for season in seasons:
        logger.info("Fetching BBRef season %d ...", season)
        df = fetch_fn(season)
        if df.empty:
            logger.warning("No BBRef data for season %d -- skipping", season)
            continue
        df = df.copy()
        df.insert(0, "SEASON", season)
        frames.append(df)

    if not frames:
        logger.warning("get_multi_season_bbref: no data fetched for any season")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
