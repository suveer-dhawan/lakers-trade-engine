"""
NBA-specific constants for the Lakers Trade Engine.

Covers team identifiers, 2025-26 CBA financial thresholds, and position
mappings. CBA dollar figures are encoded here because they are rules, not
data — they don't come from an API, they come from the collective bargaining
agreement documentation and must be manually updated each season.

Sources:
  - Team IDs: nba_api.stats.static.teams
  - 2025-26 salary cap / tax lines: NBA CBA (2023) + annual adjustment
"""

# ---------------------------------------------------------------------------
# Team IDs  (nba_api integer identifiers)
# ---------------------------------------------------------------------------

LAKERS_TEAM_ID = 1610612747
MAVERICKS_TEAM_ID = 1610612742

# Handy lookup for any team referenced in trade analysis
TEAM_IDS: dict[str, int] = {
    "LAL": LAKERS_TEAM_ID,
    "DAL": MAVERICKS_TEAM_ID,
}

# ---------------------------------------------------------------------------
# Key player IDs
# ---------------------------------------------------------------------------

LUKA_DONCIC_PLAYER_ID = 1629029  # nba_api static player record

# ---------------------------------------------------------------------------
# 2025-26 CBA Financial Thresholds (USD)
# ---------------------------------------------------------------------------

SALARY_CAP = 154_600_000          # Soft cap
LUXURY_TAX_LINE = 187_900_000     # First-dollar luxury tax threshold
FIRST_APRON = 194_900_000         # First apron (restricts certain moves)
SECOND_APRON = 207_900_000        # Second apron (hard cap, additional restrictions)

# Minimum salary (veteran minimum, 1 year of service, 2025-26)
VETERAN_MINIMUM_SALARY = 1_164_164

# Mid-level exception amounts (approximate — varies by tax status)
NON_TAX_MLE = 13_700_000
TAX_MLE = 5_200_000
ROOM_MLE = 5_200_000

# Bi-annual exception
BAE = 4_600_000

# ---------------------------------------------------------------------------
# Trade salary matching rules (2023 CBA)
# ---------------------------------------------------------------------------

# Incoming salary can exceed outgoing by at most this multiplier + flat add
# when the team is under the first apron.
TRADE_SALARY_MULTIPLIER = 1.75
TRADE_SALARY_FLAT_ADD = 250_000   # $250k flat adder on top of multiplier

# ---------------------------------------------------------------------------
# Position mappings
# ---------------------------------------------------------------------------

POSITION_GROUPS: dict[str, list[str]] = {
    "guard": ["PG", "SG", "G"],
    "wing": ["SF", "SG/SF", "G/F"],
    "big": ["PF", "C", "F/C"],
}

# Canonical 5-position scheme used across feature engineering
POSITIONS = ["PG", "SG", "SF", "PF", "C"]
