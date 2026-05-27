"""
Central configuration for the Lakers Trade Engine data pipeline.

Defines filesystem paths, season ranges, API rate limits, and source URLs
consumed by src/data/ ingestion modules. All other modules should import
constants from here rather than hardcoding paths or dates.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/data/config.py -> src/ -> project root
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RAW_DIR = DATA_DIR / "raw"

# Ensure subdirectories exist at import time
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Season range
# ---------------------------------------------------------------------------

FIRST_SEASON = 2019  # e.g. "2019-20"
LAST_SEASON = 2026   # e.g. "2025-26" (current season)
SEASONS = list(range(FIRST_SEASON, LAST_SEASON + 1))

# Canonical season string format used by nba_api: "2024-25"
def season_str(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"

SEASON_STRINGS = [season_str(y) for y in SEASONS]
CURRENT_SEASON = season_str(LAST_SEASON - 1)  # "2025-26"

# ---------------------------------------------------------------------------
# API rate limits
# ---------------------------------------------------------------------------

NBA_API_DELAY_SECONDS = 0.6        # conservative floor to avoid 429s on stats.nba.com
BBALL_REF_DELAY_SECONDS = 3.5      # Basketball Reference: ~20 req/min max → 3s floor
DEFAULT_REQUEST_TIMEOUT = 30       # seconds

# ---------------------------------------------------------------------------
# Source URLs (reference only — actual fetching lives in src/data/ modules)
# ---------------------------------------------------------------------------

RAPTOR_ARCHIVE_URL = (
    "https://raw.githubusercontent.com/fivethirtyeight/nba-player-advanced-metrics"
    "/master/nba-raptor.csv"
)
HOOPSHYPE_SALARIES_URL = "https://hoopshype.com/salaries/players/"
SPOTRAC_BASE_URL = "https://www.spotrac.com/nba/"
DUNKS_AND_THREES_URL = "https://dunksandthrees.com/epm"
