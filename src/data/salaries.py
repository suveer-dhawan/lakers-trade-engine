"""
Salary and contract data ingestion.

Phase 0.1 (current): load from a manually downloaded CSV.
Phase 0.2 (planned): add Spotrac scraping via requests + BeautifulSoup.

To bootstrap without scraping, download salary data manually from HoopsHype
(https://hoopshype.com/salaries/players/) or Basketball Reference, save as a
CSV matching SALARY_SCHEMA below, and pass the filepath to load_salaries().

Expected CSV format (one row per player-season):
  player_name, team, season, salary, years_remaining, guaranteed
"""
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

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

# Optional columns that are useful but not always present
SALARY_SCHEMA_OPTIONAL: dict = {
    "option_type": "str  -- 'player', 'team', 'UFA', 'RFA', or '' for fully guaranteed",
    "cap_hold": "int  -- cap hold value for planning scenarios (0 if on roster)",
    "two_way": "bool -- True if on a two-way contract",
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_salaries(filepath: str | Path) -> pd.DataFrame:
    """
    Load salary data from a manually downloaded CSV.

    The CSV must contain at minimum the columns defined in SALARY_SCHEMA.
    Extra columns are preserved. String columns are stripped of whitespace;
    the 'salary' and 'guaranteed' columns are coerced to int (dollars).

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame with columns matching SALARY_SCHEMA (plus any extras in the file).

    Raises
    ------
    FileNotFoundError  if the path does not exist.
    ValueError         if required columns are missing from the CSV.

    Notes
    -----
    Phase 0.2 will replace / supplement this with a Spotrac scraper that
    populates option_type, cap_hold, and partial guarantee details automatically.
    For now, manually export from HoopsHype or Basketball Reference and
    add any missing columns by hand.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Salary CSV not found: {path}\n"
            "Download salary data from https://hoopshype.com/salaries/players/ "
            "and save as a CSV with columns: " + ", ".join(SALARY_SCHEMA.keys())
        )

    df = pd.read_csv(path)

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Validate required columns
    missing = [col for col in SALARY_SCHEMA if col not in df.columns]
    if missing:
        raise ValueError(
            f"Salary CSV is missing required columns: {missing}\n"
            f"Expected: {list(SALARY_SCHEMA.keys())}\n"
            f"Found:    {list(df.columns)}"
        )

    # Coerce numeric columns -- strip commas/dollar signs if present
    for col in ("salary", "guaranteed"):
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .str.replace(r"[\$,]", "", regex=True)
                .str.strip()
                .astype(int)
            )

    logger.info("Loaded %d salary rows from %s", len(df), path.name)
    return df
