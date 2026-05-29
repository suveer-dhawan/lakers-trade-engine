"""
NBA-specific constants for the Lakers Trade Engine.

Covers team identifiers, 2025-26 CBA financial thresholds, and position
mappings. CBA dollar figures are encoded here because they are rules, not
data - they don't come from an API, they come from the collective bargaining
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

# ---------------------------------------------------------------------------
# 2026-27 CBA Thresholds (USD)
# Verified from NBA/NBPA projections, May 2026.
# ---------------------------------------------------------------------------

SALARY_CAP_2026_27 = 165_000_000          # Soft cap
LUXURY_TAX_2026_27 = 201_000_000          # First-dollar luxury tax threshold
FIRST_APRON_2026_27 = 209_000_000         # First apron
SECOND_APRON_2026_27 = 222_000_000        # Second apron (hard cap)

# Minimum salary (veteran minimum, 1 year of service, 2025-26)
VETERAN_MINIMUM_SALARY = 1_164_164

# Mid-level and bi-annual exception amounts (2026-27)
NON_TAXPAYER_MLE_2026_27 = 15_048_000     # Non-taxpayer mid-level exception
TAXPAYER_MLE_2026_27 = 6_065_000          # Taxpayer mid-level exception
TEAM_ROOM_MLE_2026_27 = 9_369_000         # Room mid-level exception
BAE_2026_27 = 5_478_000                   # Bi-annual exception

# Legacy names kept for any callers not yet updated
NON_TAX_MLE = NON_TAXPAYER_MLE_2026_27
TAX_MLE = TAXPAYER_MLE_2026_27
ROOM_MLE = TEAM_ROOM_MLE_2026_27
BAE = BAE_2026_27

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

# ---------------------------------------------------------------------------
# Player option overrides for 2026-27 offseason analysis
# Players here have their forward salary nullified so they appear as free agents.
# ---------------------------------------------------------------------------

PLAYER_OPTION_OVERRIDES: dict[str, tuple] = {
    # player_name: (season, action, notes)
    "Austin Reaves": (
        "2026-27",
        "decline",
        "Will decline $14.9M PO to sign max extension",
    ),
}

# All known option years for 2026-27. Used to add HAS_OPTION / OPTION_TYPE columns.
KNOWN_OPTIONS_2026_27: dict[str, dict] = {
    "Austin Reaves": {
        "type": "player",
        "salary": 14_900_000,
        "likely_action": "decline",
    },
    "Marcus Smart": {
        "type": "player",
        "salary": 5_390_000,
        "likely_action": "exercise",
    },
    "Deandre Ayton": {
        "type": "player",
        "salary": 8_100_000,
        "likely_action": "uncertain",
    },
}

# ---------------------------------------------------------------------------
# Lakers 2026-27 tradeable contracts
# Salaries are 2026-27 figures for players under contract.
# Player option holders only tradeable if they opt in.
# ---------------------------------------------------------------------------

LAKERS_TRADEABLE: dict[str, int] = {
    "Jarred Vanderbilt": 12_428_571,   # Under contract; injury history but useful matching salary
    "Jake LaRavia":       6_000_000,
    "Dalton Knecht":      4_200_000,   # Rookie deal, real potential - high cost to trade
    "Bronny James":       2_300_000,   # Symbolic/brand value beyond basketball production
    "Adou Thiero":        2_150_000,
}

# Player options pending -- re-add to LAKERS_TRADEABLE if they opt in (deadline: June 29):
LAKERS_PO_PENDING: dict[str, dict] = {
    "Deandre Ayton": {"salary": 8_100_000, "option_type": "player", "deadline": "June 29"},
    "Marcus Smart":  {"salary": 5_390_000, "option_type": "player", "deadline": "June 29"},
}

# Untradeable: Luka Dončić ($49.8M, franchise cornerstone)
# Austin Reaves: free agent; can be part of a sign-and-trade IF Lakers are below second apron
# (sign-and-trade is banned above the second apron)

# ---------------------------------------------------------------------------
# Extension-eligible players (new salary would affect trade math if extended)
# Deadlines are for the 2026 offseason.
# ---------------------------------------------------------------------------

LAKERS_EXTENSION_ELIGIBLE: dict[str, dict] = {
    "Austin Reaves":      {"max_extension": "4yr/$87.4M", "deadline": "June 30"},
    "Rui Hachimura":      {"max_extension": "4yr/$114.5M", "deadline": "June 30"},
    "Maxi Kleber":        {"max_extension": "4yr/$87M",   "deadline": "June 30"},
    "Jarred Vanderbilt":  {"max_extension": "4yr/$92.8M", "deadline": "Sept 18"},
    "Bronny James":       {"max_extension": "4yr/$92.8M", "deadline": "Day after Finals"},
}

# ---------------------------------------------------------------------------
# Lakers draft pick inventory (verified from ESPN, May 2026)
# ---------------------------------------------------------------------------

LAKERS_DRAFT_PICKS: dict[str, dict] = {
    "2026_1st_25": {
        "round": 1, "year": 2026, "pick_number": 25,
        "status": "own", "tradeable": True,
        "expected_range": (25, 25),
        "note": "Known position - most liquid asset.",
    },
    "2027_1st": {
        "round": 1, "year": 2027,
        "status": "owed_to_memphis_top4_protected",
        "tradeable": False,
        "expected_range": (5, 30),
        "note": "Top-4 protected to Memphis. Lakers keep only if pick lands 1-4.",
    },
    "2029_1st": {
        "round": 1, "year": 2029,
        "status": "owed_to_dallas_unprotected",
        "tradeable": False,
        "expected_range": (15, 30),
        "note": "Unprotected to Dallas. Not available to trade.",
    },
    "2031_1st": {
        "round": 1, "year": 2031,
        "status": "own", "tradeable": True,
        "expected_range": (8, 20),
        "note": "Luka will be 32. Window likely closing. Projects as mid-lottery.",
    },
    "2032_1st": {
        "round": 1, "year": 2032,
        "status": "own", "tradeable": True,
        "expected_range": (5, 18),
        "note": "Cannot trade in combination with 2031 or 2033 (Stepien Rule). Luka 33.",
        "stepien_conflicts": ["2031_1st", "2033_1st"],
    },
    "2033_1st": {
        "round": 1, "year": 2033,
        "status": "own", "tradeable": True,
        "expected_range": (3, 16),
        "note": "Luka 34. Possible rebuild territory. Most valuable distant pick.",
    },
    "2032_2nd": {
        "round": 2, "year": 2032,
        "status": "own", "tradeable": True,
        "expected_range": (31, 60),
        "note": "Second-round pick.",
    },
}

# Swap rights held by the Lakers (can swap firsts in these years if theirs is better)
LAKERS_SWAP_RIGHTS: list[int] = [2028, 2030, 2031, 2032, 2033]

# Tradeable firsts are limited to 2 max in any rolling 7-year window (Stepien Rule).
# In practice: can trade 2031 + 2033 together, but cannot include 2032.
TRADEABLE_FIRST_LIMIT = 2

# ---------------------------------------------------------------------------
# Restricted free agents among top trade/FA targets (current team can match)
# ---------------------------------------------------------------------------

RESTRICTED_FREE_AGENTS: list[str] = [
    "Jalen Duren",     # DET
    "Walker Kessler",  # UTA
    "Peyton Watson",   # DEN
    "Tari Eason",      # HOU
]
