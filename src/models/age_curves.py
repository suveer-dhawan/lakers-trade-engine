"""
Age-based value projection using positional decline curves.

Phase 1 — Player Valuation (see README).

Planned responsibilities:
  - Fit historical aging curves by position/archetype using regression on
    multi-season player data (value metric vs. age delta from peak)
  - Identify peak age by position (guards tend to peak earlier than bigs)
  - Project a player's surplus value over the remaining years of their contract
  - Discount future seasons by uncertainty (further out = wider confidence band)
  - Feed projected values into src/features/player_value.py and src/models/trade_eval.py

Technique: polynomial regression or LOESS curve per position bucket.
Historical data window: FIRST_SEASON through LAST_SEASON (config.py).
"""
