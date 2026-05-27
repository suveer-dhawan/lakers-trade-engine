"""
Local caching layer for API responses and scraped data.

Stores DataFrames as Parquet files under CACHE_DIR (config.py).
Cache keys map 1-to-1 with filenames; unsafe characters are sanitized.
Build the cache first, then analyse offline -- rate limits are real.
"""
import fnmatch
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import CACHE_DIR

logger = logging.getLogger(__name__)


def _cache_path(cache_key: str) -> Path:
    safe = cache_key.replace("/", "_").replace("\\", "_").replace(":", "_")
    return CACHE_DIR / f"{safe}.parquet"


def get_or_fetch(
    cache_key: str,
    fetch_fn: Callable[[], pd.DataFrame],
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a cached DataFrame if one exists, otherwise call fetch_fn to get it.

    Parameters
    ----------
    cache_key : str
        Unique string identifying this dataset (becomes the parquet filename).
    fetch_fn : callable
        Zero-argument function that fetches and returns a DataFrame.
    refresh : bool
        If True, ignore any cached file and re-fetch.
    """
    path = _cache_path(cache_key)
    if path.exists() and not refresh:
        logger.info("cache hit: %s", cache_key)
        return pd.read_parquet(path)
    logger.info("cache miss -- fetching: %s", cache_key)
    df = fetch_fn()
    df.to_parquet(path, index=False)
    logger.info("cached %d rows to %s", len(df), path.name)
    return df


def cache_info() -> pd.DataFrame:
    """List all cached files with size (KB) and last-modified timestamp."""
    rows = []
    for p in sorted(CACHE_DIR.glob("*.parquet")):
        stat = p.stat()
        rows.append(
            {
                "file": p.name,
                "key": p.stem,
                "size_kb": round(stat.st_size / 1024, 1),
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime(
                    "%Y-%m-%d %H:%M"
                ),
            }
        )
    cols = ["file", "key", "size_kb", "modified"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def clear_cache(pattern: str = "*") -> int:
    """
    Delete cached parquet files whose stem matches *pattern* (fnmatch syntax).

    Returns the number of files deleted.

    Examples
    --------
    clear_cache()                   # delete everything
    clear_cache("nba_stats_*")      # clear all nba_stats files
    clear_cache("*2019*")           # clear 2019-20 season entries
    """
    deleted = 0
    for p in CACHE_DIR.glob("*.parquet"):
        if fnmatch.fnmatch(p.stem, pattern):
            p.unlink()
            logger.info("deleted cache file: %s", p.name)
            deleted += 1
    return deleted
