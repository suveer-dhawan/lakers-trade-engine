"""
Salary and contract data ingestion.

Three modes of operation:
  1. load_salaries(filepath)          -- load from a manually downloaded CSV (always works)
  2. scrape_hoopshype_salaries(season) -- scrape HoopsHype salary table (simpler, try first)
  3. scrape_spotrac_salaries(season)   -- scrape Spotrac team pages (JS-heavy, may fail)

If scraping fails, call generate_salary_template() to get an empty DataFrame with the
right schema for manual data entry, save it to data/raw/salaries_{season}.csv, fill it in,
then use load_salaries() to load it.

Expected schema (SALARY_SCHEMA):
  player_name, team, season, salary, years_remaining, guaranteed,
  option_type, cap_hit, two_way
"""
from __future__ import annotations

import logging
import re
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.utils.constants import KNOWN_OPTIONS_2026_27, PLAYER_OPTION_OVERRIDES

from .cache import get_or_fetch
from .config import BBALL_REF_DELAY_SECONDS, DEFAULT_REQUEST_TIMEOUT, HOOPSHYPE_SALARIES_URL

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------

SALARY_SCHEMA: dict = {
    "player_name": "str  -- display name, e.g. 'LeBron James'",
    "team": "str  -- three-letter abbreviation, e.g. 'LAL'",
    "season": "str  -- nba_api format, e.g. '2025-26'",
    "salary": "int  -- base salary in USD (no commas)",
    "years_remaining": "int  -- contract years left including current season",
    "guaranteed": "int  -- guaranteed dollars remaining on contract",
}

SALARY_SCHEMA_OPTIONAL: dict = {
    "option_type": "str  -- 'player', 'team', 'UFA', 'RFA', or '' for fully guaranteed",
    "cap_hit": "int  -- cap hit value (often = salary, differs for converted two-ways)",
    "two_way": "bool -- True if on a two-way contract",
}

_EMPTY_SCHEMA_COLS = list(SALARY_SCHEMA.keys()) + list(SALARY_SCHEMA_OPTIONAL.keys())


# ---------------------------------------------------------------------------
# Manual loader
# ---------------------------------------------------------------------------


def load_salaries(filepath: str | Path | None = None, season: str = "2024-25") -> pd.DataFrame:
    """
    Load salary data from a manually downloaded CSV.

    Primary path: data/raw/salaries_{season}.csv (see data/raw/MANUAL_DOWNLOAD_GUIDE.md).
    Pass an explicit filepath to override.

    The CSV must contain at minimum the columns in SALARY_SCHEMA.
    Extra columns are preserved. salary/guaranteed columns are coerced to int.

    Parameters
    ----------
    filepath : str, Path, or None
        Path to the CSV file. If None, defaults to data/raw/salaries_{season}.csv.
    season : str
        Season string used to build the default filename, e.g. "2024-25".

    Raises FileNotFoundError if the resolved file does not exist.
    Raises ValueError if required columns are missing.
    """
    from .config import RAW_DIR
    path = Path(filepath) if filepath is not None else RAW_DIR / f"salaries_{season}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Salary CSV not found: {path}\n"
            "See data/raw/MANUAL_DOWNLOAD_GUIDE.md for download instructions, or "
            "generate a template with generate_salary_template()."
        )

    df = pd.read_csv(path)

    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    missing = [col for col in SALARY_SCHEMA if col not in df.columns]
    if missing:
        raise ValueError(
            f"Salary CSV is missing required columns: {missing}\n"
            f"Expected: {list(SALARY_SCHEMA.keys())}\n"
            f"Found:    {list(df.columns)}"
        )

    for col in ("salary", "guaranteed"):
        if col in df.columns and df[col].dtype == object:
            df[col] = (
                df[col].str.replace(r"[\$,]", "", regex=True).str.strip().astype(int)
            )

    logger.info("Loaded %d salary rows from %s", len(df), path.name)
    return df


# ---------------------------------------------------------------------------
# Template generator
# ---------------------------------------------------------------------------


def generate_salary_template() -> pd.DataFrame:
    """
    Return an empty DataFrame with the full salary schema.

    Use this when scraping is unavailable:
      template = generate_salary_template()
      template.to_csv("data/raw/salaries_2025-26.csv", index=False)
      # Fill in the CSV manually, then:
      df = load_salaries("data/raw/salaries_2025-26.csv")
    """
    return pd.DataFrame(columns=_EMPTY_SCHEMA_COLS)


# ---------------------------------------------------------------------------
# BBRef salary CSV loader (primary path for Phase 0.3+)
# ---------------------------------------------------------------------------


def load_bbref_salaries(season: str) -> pd.DataFrame:
    """
    Load multi-year contract data from a manually downloaded BBRef salary CSV.

    BBRef salary CSV headers (example for 2025-26 download):
      Rk, Player, Tm, 2025-26, 2026-27, ..., Guaranteed, -9999

    Parameters
    ----------
    season : str
        Season in nba_api format, e.g. "2025-26". Used to locate the file
        data/raw/salaries_{season}.csv and to identify the current-year column.

    Returns
    -------
    DataFrame with columns: PLAYER_NAME_SALARY, TEAM_SALARY, SALARY,
    YEARS_REMAINING, TOTAL_GUARANTEED, IS_EXPIRING, AVG_ANNUAL_VALUE,
    plus the raw year-by-year salary columns (e.g. "2025-26", "2026-27").

    Returns empty DataFrame with a warning if the file is not found.
    """
    from .config import RAW_DIR

    path = RAW_DIR / f"salaries_{season}.csv"
    if not path.exists():
        logger.warning("BBRef salary CSV not found: %s", path)
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Drop BBRef rank column and the player-ID column (header "-9999")
    df = df.drop(columns=["Rk", "-9999"], errors="ignore")

    # Drop repeated header rows
    if "Player" in df.columns:
        df = df[df["Player"] != "Player"].copy()

    # Strip asterisks from player names (HOF indicator)
    if "Player" in df.columns:
        df["Player"] = df["Player"].astype(str).str.replace("*", "", regex=False).str.strip()

    # For multi-team players keep the TOT row, drop per-team rows
    if "Tm" in df.columns:
        traded = df[df["Tm"] == "TOT"]["Player"].unique()
        df = df[~((df["Player"].isin(traded)) & (df["Tm"] != "TOT"))].copy()

    # Rename standard identifier columns
    df = df.rename(columns={"Player": "PLAYER_NAME_SALARY", "Tm": "TEAM_SALARY"})

    # Identify year columns (format: "YYYY-YY", e.g. "2025-26")
    year_cols = [c for c in df.columns if re.match(r"^\d{4}-\d{2}$", str(c))]

    # Parse salary year columns and Guaranteed: strip "$" and commas, convert to float
    money_cols = year_cols + (["Guaranteed"] if "Guaranteed" in df.columns else [])
    for col in money_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .str.strip()
            .replace({"": None, "nan": None})
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if year_cols:
        current_col = year_cols[0]
        # forward_cols: only seasons AFTER the current season (2026-27 onward).
        # The current season (2025-26) is over for trade analysis purposes.
        forward_cols = year_cols[1:]

        df["SALARY"] = df[current_col]

        # Apply player option overrides before computing derived columns.
        # Null out the forward salary for players confirmed to be declining their option.
        for player_name, (option_season, action, _notes) in PLAYER_OPTION_OVERRIDES.items():
            if action == "decline" and option_season in df.columns:
                name_mask = df["PLAYER_NAME_SALARY"] == player_name
                if name_mask.any():
                    df.loc[name_mask, option_season] = np.nan
                    logger.info(
                        "Option override: %s declining %s PO, treating as free agent",
                        player_name, option_season,
                    )

        # YEARS_REMAINING: count only forward seasons (2026-27 onward) with a salary.
        if forward_cols:
            df["YEARS_REMAINING"] = df[forward_cols].apply(
                lambda row: int(sum(1 for v in row if pd.notna(v) and v > 0)), axis=1
            )
            df["IS_EXPIRING"] = df[forward_cols].apply(
                lambda row: all(pd.isna(v) or v == 0 for v in row), axis=1
            )
        else:
            df["YEARS_REMAINING"] = 0
            df["IS_EXPIRING"] = True

        df["TOTAL_GUARANTEED"] = df["Guaranteed"] if "Guaranteed" in df.columns else float("nan")

        def _avg_annual(row) -> float:
            total = sum(v for c in forward_cols for v in [row[c]] if pd.notna(v) and v > 0)
            years = row["YEARS_REMAINING"]
            return total / years if years > 0 else float("nan")

        df["AVG_ANNUAL_VALUE"] = df.apply(_avg_annual, axis=1)

        # Annotate known option years.
        df["HAS_OPTION"] = False
        df["OPTION_TYPE"] = ""
        for player_name, option_info in KNOWN_OPTIONS_2026_27.items():
            name_mask = df["PLAYER_NAME_SALARY"] == player_name
            if name_mask.any():
                df.loc[name_mask, "HAS_OPTION"] = True
                df.loc[name_mask, "OPTION_TYPE"] = option_info["type"]

    df = df.drop(columns=["Guaranteed"], errors="ignore")
    df = df.reset_index(drop=True)
    logger.info("Loaded %d BBRef salary rows from %s", len(df), path.name)
    return df


# ---------------------------------------------------------------------------
# HoopsHype scraper
# ---------------------------------------------------------------------------


def scrape_hoopshype_salaries(season: str) -> pd.DataFrame:
    """
    Scrape current-season player salaries from HoopsHype.

    Parameters
    ----------
    season : str
        Season in nba_api format, e.g. "2025-26". Used only for the cache key and
        the 'season' column in the output - HoopsHype always shows the current season.

    Returns
    -------
    DataFrame with columns: player_name, team, season, salary.
    years_remaining, guaranteed, option_type are not available from HoopsHype;
    supplement with load_salaries() or scrape_spotrac_salaries() for full contract detail.

    Returns empty DataFrame with a warning if scraping fails.
    """
    cache_key = f"salaries_hoopshype_{season}"
    try:
        return get_or_fetch(cache_key, lambda: _fetch_hoopshype(season))
    except Exception as exc:
        logger.warning("scrape_hoopshype_salaries(%s) cache layer failed: %s", season, exc)
        return pd.DataFrame(columns=["player_name", "team", "season", "salary"])


def _fetch_hoopshype(season: str) -> pd.DataFrame:
    """Internal: actually scrape HoopsHype and return a DataFrame."""
    time.sleep(BBALL_REF_DELAY_SECONDS)
    try:
        resp = requests.get(
            HOOPSHYPE_SALARIES_URL, headers=_HEADERS, timeout=DEFAULT_REQUEST_TIMEOUT
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("HoopsHype request failed: %s", exc)
        return pd.DataFrame(columns=["player_name", "team", "season", "salary"])

    soup = BeautifulSoup(resp.text, "html.parser")

    # HoopsHype renders salaries in a table with class "hh-salaries-ranking-table"
    table = soup.find("table", class_=lambda c: c and "hh-salaries" in c)
    if table is None:
        # Fall back: try any table on the page
        tables = soup.find_all("table")
        if not tables:
            logger.warning("No salary table found on HoopsHype page")
            return pd.DataFrame(columns=["player_name", "team", "season", "salary"])
        table = tables[0]

    try:
        dfs = pd.read_html(StringIO(str(table)))
        df = dfs[0]
    except Exception as exc:
        logger.warning("Failed to parse HoopsHype table: %s", exc)
        return pd.DataFrame(columns=["player_name", "team", "season", "salary"])

    # HoopsHype columns vary - normalize what we can find
    df = _normalize_hoopshype_columns(df, season)
    logger.info("HoopsHype: scraped %d salary rows for %s", len(df), season)
    return df


def _normalize_hoopshype_columns(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Map HoopsHype raw column names to our schema."""
    # Drop index/rank columns
    df = df.loc[:, ~df.columns.str.match(r"^\d+$|^#$|Rank")]

    # Try to identify player name and salary columns by content heuristics
    name_candidates = [c for c in df.columns if any(
        kw in str(c).lower() for kw in ["player", "name"]
    )]
    salary_candidates = [c for c in df.columns if any(
        kw in str(c).lower() for kw in ["salary", season[:4], str(int(season[:4]) + 1)]
    )]

    if not name_candidates or not salary_candidates:
        # Last resort: assume first col is name, last col is salary
        name_col = df.columns[0]
        salary_col = df.columns[-1]
    else:
        name_col = name_candidates[0]
        salary_col = salary_candidates[0]

    out = pd.DataFrame()
    out["player_name"] = df[name_col].astype(str).str.strip()
    out["team"] = ""  # HoopsHype doesn't always include team on this page
    out["season"] = season

    # Coerce salary: strip $, commas
    salary_raw = df[salary_col].astype(str)
    out["salary"] = (
        salary_raw.str.replace(r"[\$,]", "", regex=True)
        .str.strip()
        .replace("", "0")
        .apply(lambda x: int(float(x)) if x.replace(".", "").isdigit() else 0)
    )

    # Drop rows with no name or zero salary
    out = out[
        (out["player_name"] != "") &
        (out["player_name"] != "nan") &
        (out["salary"] > 0)
    ].reset_index(drop=True)

    return out


# ---------------------------------------------------------------------------
# Spotrac scraper
# ---------------------------------------------------------------------------

_SPOTRAC_TEAM_SLUGS = {
    "ATL": "atlanta-hawks", "BOS": "boston-celtics", "BKN": "brooklyn-nets",
    "CHA": "charlotte-hornets", "CHI": "chicago-bulls", "CLE": "cleveland-cavaliers",
    "DAL": "dallas-mavericks", "DEN": "denver-nuggets", "DET": "detroit-pistons",
    "GSW": "golden-state-warriors", "HOU": "houston-rockets", "IND": "indiana-pacers",
    "LAC": "los-angeles-clippers", "LAL": "los-angeles-lakers", "MEM": "memphis-grizzlies",
    "MIA": "miami-heat", "MIL": "milwaukee-bucks", "MIN": "minnesota-timberwolves",
    "NOP": "new-orleans-pelicans", "NYK": "new-york-knicks", "OKC": "oklahoma-city-thunder",
    "ORL": "orlando-magic", "PHI": "philadelphia-76ers", "PHX": "phoenix-suns",
    "POR": "portland-trail-blazers", "SAC": "sacramento-kings", "SAS": "san-antonio-spurs",
    "TOR": "toronto-raptors", "UTA": "utah-jazz", "WAS": "washington-wizards",
}


def scrape_spotrac_salaries(season: str) -> pd.DataFrame:
    """
    Scrape player salaries from Spotrac team cap pages.

    Attempts to scrape all 30 team pages and concatenate. Spotrac is JavaScript-heavy
    and may block or return incomplete data - fall back to HoopsHype or manual CSV
    if this returns an empty DataFrame.

    Parameters
    ----------
    season : str
        Season string, e.g. "2025-26". Used for cache key and output column.

    Returns
    -------
    DataFrame with columns: player_name, team, season, salary, cap_hit,
    years_remaining, option_type. Returns empty DataFrame with warning on failure.
    """
    cache_key = f"salaries_spotrac_{season}"
    try:
        return get_or_fetch(cache_key, lambda: _fetch_spotrac_all(season))
    except Exception as exc:
        logger.warning("scrape_spotrac_salaries(%s) failed: %s", season, exc)
        return pd.DataFrame(columns=["player_name", "team", "season", "salary"])


def _fetch_spotrac_all(season: str) -> pd.DataFrame:
    """Internal: scrape all Spotrac team pages for the season."""
    season_year = season[:4]  # "2025" from "2025-26"
    frames = []

    for abbr, slug in _SPOTRAC_TEAM_SLUGS.items():
        url = f"https://www.spotrac.com/nba/{slug}/cap/{season_year}/"
        time.sleep(BBALL_REF_DELAY_SECONDS)
        df = _fetch_spotrac_team(url, abbr, season)
        if not df.empty:
            frames.append(df)
        else:
            logger.warning("Spotrac: no data for %s (%s)", abbr, url)

    if not frames:
        logger.warning("Spotrac: no data fetched for any team in %s", season)
        return pd.DataFrame(columns=["player_name", "team", "season", "salary"])

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Spotrac: scraped %d total rows for %s", len(combined), season)
    return combined


def _fetch_spotrac_team(url: str, team_abbr: str, season: str) -> pd.DataFrame:
    """Scrape a single Spotrac team cap page."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=DEFAULT_REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Spotrac request failed for %s: %s", url, exc)
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Spotrac's player rows are in a <table> or <div> with salary data
    tables = soup.find_all("table")
    if not tables:
        logger.warning("No tables found on Spotrac page: %s", url)
        return pd.DataFrame()

    rows = []
    for table in tables:
        try:
            parsed = pd.read_html(StringIO(str(table)))[0]
            if "Player" in parsed.columns or any("Name" in c for c in parsed.columns):
                rows.append(parsed)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = rows[0]

    # Identify player name column
    name_col = next(
        (c for c in df.columns if any(kw in str(c).lower() for kw in ["player", "name"])),
        df.columns[0],
    )

    # Identify salary column
    salary_col = next(
        (c for c in df.columns if any(kw in str(c).lower() for kw in ["salary", "base", "cap hit"])),
        None,
    )

    if salary_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["player_name"] = df[name_col].astype(str).str.strip()
    out["team"] = team_abbr
    out["season"] = season
    out["salary"] = (
        df[salary_col].astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.strip()
        .apply(lambda x: int(float(x)) if x.replace(".", "").isdigit() else 0)
    )

    # Optional columns
    cap_col = next((c for c in df.columns if "cap" in str(c).lower()), None)
    out["cap_hit"] = (
        df[cap_col].astype(str).str.replace(r"[\$,]", "", regex=True)
        .apply(lambda x: int(float(x)) if x.replace(".", "").isdigit() else 0)
        if cap_col else 0
    )

    years_col = next((c for c in df.columns if "year" in str(c).lower()), None)
    out["years_remaining"] = (
        pd.to_numeric(df[years_col], errors="coerce").fillna(0).astype(int)
        if years_col else 0
    )

    option_col = next(
        (c for c in df.columns if any(kw in str(c).lower() for kw in ["option", "type", "status"])),
        None,
    )
    out["option_type"] = df[option_col].astype(str).str.strip() if option_col else ""

    out = out[
        (out["player_name"] != "") & (out["player_name"] != "nan") & (out["salary"] > 0)
    ].reset_index(drop=True)

    return out
