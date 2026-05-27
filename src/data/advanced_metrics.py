"""
Ingestion of public advanced metrics not available through nba_api.

Phase 0 — Data Pipeline (see README).

Planned sources and metrics:
  - EPM (Estimated Plus-Minus): Dunks & Threes — dunksandthrees.com/epm
  - RAPTOR (archived): FiveThirtyEight GitHub CSV — historical seasons only
  - LEBRON / RAPM: nbarapm.com — scraping or manual CSV download
  - BPM / VORP / WS / PER: Basketball Reference via basketball-reference-scraper

All metrics land in a unified player-season DataFrame keyed on (player_id, season).
Schema alignment handles name mismatches between sources (accents, suffixes, etc.).
"""
