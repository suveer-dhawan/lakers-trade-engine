"""
Draft pick valuation model.

Phase 2A.2 -- Trade Evaluator.

Estimates the surplus value of a draft pick based on its expected draft position.
Uses a standard pick-value curve (similar in spirit to Pelton/Hollinger) representing
the expected surplus value over the 4-year rookie contract, in $M.

Key insight for Lakers picks specifically:
  The standard "distant pick = discount" logic is INVERTED for a contending team.
  A team with Luka + a strong core will be good NOW (2026-28), meaning near-term picks
  project as late firsts (low value). By 2031-33, Luka is 32-34 and the team may be
  declining or rebuilding -- making those picks project as mid-lottery or better.

  For trade partners, Lakers 2031 and 2033 firsts are MORE attractive than a
  typical distant pick because they're betting on the Lakers being worse by then.
  This is the same dynamic as the Nets-Celtics 2013 trade (Brooklyn's distant picks
  became Jaylen Brown and Jayson Tatum).

  estimate_pick_value() uses the expected_range for each specific Lakers pick
  rather than applying a generic time discount.
"""
from __future__ import annotations

import numpy as np

from src.utils.constants import LAKERS_DRAFT_PICKS

# ---------------------------------------------------------------------------
# Pick value curve
# Estimated surplus value in $M over the 4-year rookie contract.
# Source: public model inspired by Pelton, Hollinger, and Massey-Thaler.
# Pick 1 = most valuable; pick 30 = least valuable first-rounder.
# Second-round picks: $0.3-0.8M surplus.
# ---------------------------------------------------------------------------

PICK_VALUE_CURVE: dict[int, float] = {
    1: 30.0, 2: 25.0, 3: 22.0, 4: 19.0, 5: 17.0,
    6: 15.0, 7: 13.5, 8: 12.0, 9: 11.0, 10: 10.0,
    11: 9.0,  12: 8.5,  13: 8.0,  14: 7.5,  15: 7.0,
    16: 6.5,  17: 6.0,  18: 5.5,  19: 5.0,  20: 4.5,
    21: 4.0,  22: 3.5,  23: 3.0,  24: 2.8,  25: 2.5,
    26: 2.2,  27: 2.0,  28: 1.8,  29: 1.5,  30: 1.2,
}

SECOND_ROUND_VALUE = 0.5  # midpoint of $0.3-0.8M range


def estimate_pick_value(
    pick_round: int,
    expected_position_range: tuple[int, int] = (20, 30),
) -> float:
    """
    Estimate trade value of a draft pick in $M surplus over 4-year rookie deal.

    For future picks where position is uncertain, averages value across the
    expected range. Uses the PICK_VALUE_CURVE for first-rounders.

    Parameters
    ----------
    pick_round : int
        1 for first round, 2 for second round.
    expected_position_range : tuple[int, int]
        (low, high) pick position range. Inclusive. For a known pick (e.g. #25),
        pass (25, 25).

    Returns
    -------
    float : estimated surplus value in $M
    """
    if pick_round == 2:
        return SECOND_ROUND_VALUE

    low, high = expected_position_range
    low = max(1, min(30, low))
    high = max(1, min(30, high))
    if low > high:
        low, high = high, low

    positions = range(low, high + 1)
    values = [PICK_VALUE_CURVE.get(pos, 1.0) for pos in positions]
    return float(np.mean(values))


def value_pick_by_key(pick_key: str) -> float | None:
    """
    Look up a Lakers draft pick by its key in LAKERS_DRAFT_PICKS and return
    its estimated value in $M. Returns None if the pick is not tradeable.

    Parameters
    ----------
    pick_key : str
        Key from LAKERS_DRAFT_PICKS, e.g. '2026_1st_25', '2031_1st'.
    """
    info = LAKERS_DRAFT_PICKS.get(pick_key)
    if info is None:
        return None
    if not info.get("tradeable", False):
        return None  # pick owed to another team or not tradeable

    pick_round = info["round"]
    expected_range = info.get("expected_range", (20, 30))
    return estimate_pick_value(pick_round, expected_position_range=expected_range)


def lakers_tradeable_picks_summary() -> list[dict]:
    """
    Return a summary of all tradeable Lakers picks with estimated values.

    Returns list of dicts with keys: pick_key, year, round, expected_range,
    estimated_value_M, note.
    """
    rows = []
    for key, info in LAKERS_DRAFT_PICKS.items():
        if not info.get("tradeable", False):
            continue
        val = value_pick_by_key(key)
        rows.append({
            "pick_key": key,
            "year": info["year"],
            "round": info["round"],
            "expected_range": info.get("expected_range", (20, 30)),
            "estimated_value_M": val,
            "note": info.get("note", ""),
        })
    rows.sort(key=lambda r: (r["year"], r["round"]))
    return rows


def pick_combination_value(pick_keys: list[str]) -> float:
    """
    Total value of a list of picks in $M.

    Ignores non-tradeable picks silently (their value contributes 0).
    Caller should validate Stepien Rule compliance separately.
    """
    return sum(
        v for k in pick_keys if (v := value_pick_by_key(k)) is not None
    )


def check_stepien_rule(pick_keys: list[str]) -> tuple[bool, str]:
    """
    Check whether a combination of Lakers picks violates the Stepien Rule.

    The Stepien Rule prevents trading first-round picks in consecutive years.
    For Lakers specifically: 2031, 2032, and 2033 firsts are owned, but
    2032 cannot be traded in combination with 2031 or 2033.

    Parameters
    ----------
    pick_keys : list of pick keys from LAKERS_DRAFT_PICKS to be included in trade.

    Returns
    -------
    (is_valid, reason)
    """
    first_round_picks = [
        k for k in pick_keys
        if LAKERS_DRAFT_PICKS.get(k, {}).get("round") == 1
        and LAKERS_DRAFT_PICKS.get(k, {}).get("tradeable", False)
    ]

    # Check each pick's Stepien conflicts
    all_keys_set = set(pick_keys)
    for key in first_round_picks:
        conflicts = LAKERS_DRAFT_PICKS.get(key, {}).get("stepien_conflicts", [])
        for conflict in conflicts:
            if conflict in all_keys_set:
                return (
                    False,
                    f"Stepien Rule violation: cannot trade '{key}' and '{conflict}' in the "
                    f"same deal (consecutive first-round picks).",
                )

    if len(first_round_picks) > 2:
        return (
            False,
            f"Cannot trade more than 2 first-round picks within a 7-year window. "
            f"Attempting to trade {len(first_round_picks)}: {first_round_picks}.",
        )

    return True, "Pick combination is Stepien Rule compliant."
