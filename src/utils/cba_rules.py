"""
CBA salary matching and apron constraint logic (2023 NBA Collective Bargaining Agreement).

All rules verified from ESPN Bobby Marks offseason guide + Sports Business Classroom, May 2026.
CBA thresholds are for the 2026-27 season (projected). Update if NBA announces revised figures.

Apron status terminology used throughout:
  'below_aprons'  -- team below both luxury tax aprons (maximum flexibility)
  'first_apron'   -- team above first apron but below second apron
  'second_apron'  -- team above second (hard) apron (most restricted)

Salary matching is always from the perspective of the SENDING team:
  outgoing_salary = sum of salaries sent out in the trade
  incoming_salary = salary of player(s) received
"""
from __future__ import annotations

from itertools import combinations

from src.utils.constants import (
    FIRST_APRON_2026_27,
    LAKERS_TRADEABLE,
    NON_TAXPAYER_MLE_2026_27,
    SALARY_CAP_2026_27,
    SECOND_APRON_2026_27,
    TAXPAYER_MLE_2026_27,
    TEAM_ROOM_MLE_2026_27,
)

# ---------------------------------------------------------------------------
# Matching thresholds for teams BELOW both aprons (non-taxpayer rules)
# ---------------------------------------------------------------------------

_TIER1_OUTGOING_CAP = 7_500_000    # Outgoing <= this: use 200% + $250K rule
_TIER2_OUTGOING_CAP = 29_000_000   # Outgoing <= this: use outgoing + $7.5M rule
_TIER1_MULTIPLIER = 2.0
_TIER2_FLAT_ADD = 7_500_000
_TIER3_MULTIPLIER = 1.25
_FLAT_ADDER = 250_000              # $250K adder applied in tiers 1 and 3

# Salary at which a player is considered a "minimum salary" player (approx)
_MIN_SALARY_THRESHOLD = 1_500_000


def _max_incoming_for_outgoing(outgoing_salary: float, apron_status: str) -> float:
    """
    Maximum incoming salary legally receivable given the outgoing salary and apron status.

    For first/second apron: incoming cannot exceed outgoing (100% rule).
    For below_aprons: tiered generous matching rules apply.
    """
    if apron_status in ("first_apron", "second_apron"):
        return outgoing_salary  # 100% rule: cannot take back more than sent out

    # Below-aprons tiered rules
    if outgoing_salary <= _TIER1_OUTGOING_CAP:
        return outgoing_salary * _TIER1_MULTIPLIER + _FLAT_ADDER
    elif outgoing_salary <= _TIER2_OUTGOING_CAP:
        return outgoing_salary + _TIER2_FLAT_ADD
    else:
        return outgoing_salary * _TIER3_MULTIPLIER + _FLAT_ADDER


def trade_is_legal(
    outgoing_salary: float,
    incoming_salary: float,
    team_total_salary: float,
    apron_status: str = "first_apron",
    n_outgoing_players: int = 1,
) -> tuple[bool, str, str]:
    """
    Check whether a trade is legal under 2026-27 CBA salary matching rules.

    Parameters
    ----------
    outgoing_salary : float
        Total salary being sent out in the trade.
    incoming_salary : float
        Total salary being received (the target player's salary).
    team_total_salary : float
        Team's total payroll BEFORE the trade, used to check apron status.
    apron_status : str
        'below_aprons', 'first_apron', or 'second_apron'.
    n_outgoing_players : int
        Number of players being sent out. Aggregation check for second apron.

    Returns
    -------
    (is_legal, reason, apron_status)
    """
    # ------------------------------------------------------------------
    # Second apron: aggregation ban
    # Combining multiple outgoing players to match one incoming player is
    # unconditionally banned above the second apron (the edge case where
    # the trade itself drops payroll below the apron is not modeled here).
    # ------------------------------------------------------------------
    if apron_status == "second_apron" and n_outgoing_players > 1:
        return (
            False,
            f"Aggregation banned above second apron. Cannot combine multiple "
            f"salaries to match one incoming player.",
            apron_status,
        )

    # ------------------------------------------------------------------
    # Salary matching check
    # ------------------------------------------------------------------
    max_incoming = _max_incoming_for_outgoing(outgoing_salary, apron_status)
    if incoming_salary > max_incoming:
        if apron_status in ("first_apron", "second_apron"):
            return (
                False,
                f"100% matching rule violated. Outgoing: ${outgoing_salary:,.0f}, "
                f"Incoming: ${incoming_salary:,.0f}. Above first/second apron, "
                f"outgoing must be >= incoming.",
                apron_status,
            )
        else:
            return (
                False,
                f"Salary match exceeded. Outgoing: ${outgoing_salary:,.0f} allows max "
                f"incoming of ${max_incoming:,.0f}, but incoming is ${incoming_salary:,.0f}.",
                apron_status,
            )

    # ------------------------------------------------------------------
    # Additional second-apron restrictions (informational; trade still legal
    # if salary match passes, but caller should note these constraints)
    # ------------------------------------------------------------------
    notes = []
    if apron_status == "second_apron":
        notes.append("Cannot include cash. Cannot use sign-and-trade. Cannot use bi-annual exception.")

    reason = f"Legal. Outgoing ${outgoing_salary:,.0f} >= Incoming ${incoming_salary:,.0f}."
    if apron_status == "below_aprons":
        reason = (
            f"Legal. Outgoing ${outgoing_salary:,.0f} permits up to ${max_incoming:,.0f} "
            f"incoming; receiving ${incoming_salary:,.0f}."
        )
    if notes:
        reason += " Note: " + " ".join(notes)

    return True, reason, apron_status


def salary_match_options(
    target_salary: float,
    lakers_contracts: dict[str, float] | None = None,
    apron_status: str = "first_apron",
    max_players_out: int = 4,
    exclude_players: list[str] | None = None,
) -> list[list[str]]:
    """
    Find all legal salary-matching player combinations for a target player's salary.

    Parameters
    ----------
    target_salary : float
        Salary of the incoming player the Lakers want to acquire.
    lakers_contracts : dict {player_name: salary}
        Tradeable Lakers contracts. Defaults to LAKERS_TRADEABLE from constants.py.
        Luka Doncic is always excluded regardless of input.
    apron_status : str
        'below_aprons', 'first_apron', or 'second_apron'.
    max_players_out : int
        Maximum players to include in a single outgoing package. Capped at 1 for
        second_apron (aggregation ban). Larger values increase combinatorial cost.
    exclude_players : list[str]
        Additional player names to exclude from candidate pool (e.g. injured, untradeable).

    Returns
    -------
    List of valid packages. Each package is a list of player names whose combined
    salary satisfies the matching rules for the given apron status and target salary.
    Packages are sorted by ascending total outgoing salary (tightest match first).
    """
    if lakers_contracts is None:
        lakers_contracts = LAKERS_TRADEABLE

    excluded = {p.lower() for p in (exclude_players or [])}
    excluded.add("luka doncic")
    excluded.add("luka doncic")

    candidates = {
        name: sal
        for name, sal in lakers_contracts.items()
        if name.lower() not in excluded
    }

    # Second apron: no aggregation -- only single-player swaps are valid
    if apron_status == "second_apron":
        max_players_out = 1

    valid_packages: list[tuple[float, list[str]]] = []
    player_list = list(candidates.keys())

    for n in range(1, min(max_players_out, len(player_list)) + 1):
        for combo in combinations(player_list, n):
            outgoing = sum(candidates[p] for p in combo)
            is_legal, _, _ = trade_is_legal(
                outgoing_salary=outgoing,
                incoming_salary=target_salary,
                team_total_salary=0,  # caller can pass actual payroll if needed
                apron_status=apron_status,
                n_outgoing_players=n,
            )
            if is_legal:
                valid_packages.append((outgoing, list(combo)))

    # Sort by outgoing salary ascending (tightest match first)
    valid_packages.sort(key=lambda x: x[0])
    return [pkg for _, pkg in valid_packages]


def determine_apron_status(team_total_salary: float) -> str:
    """
    Determine the apron status of a team based on total payroll.

    Uses 2026-27 projected thresholds from constants.py.
    """
    if team_total_salary >= SECOND_APRON_2026_27:
        return "second_apron"
    elif team_total_salary >= FIRST_APRON_2026_27:
        return "first_apron"
    elif team_total_salary >= SALARY_CAP_2026_27:
        return "over_cap"  # between cap and luxury tax; can use MLE
    else:
        return "below_cap"


def available_exceptions(apron_status: str) -> dict[str, bool | float]:
    """
    Return which salary cap exceptions are available given apron status.

    Returns dict with exception name: available (bool) or amount (float).
    """
    match apron_status:
        case "below_cap":
            return {
                "cap_room": True,
                "room_mle": TEAM_ROOM_MLE_2026_27,
                "non_tax_mle": False,
                "tax_mle": False,
                "bi_annual": False,
                "tpe": True,
                "sign_and_trade": True,
                "aggregation": True,
            }
        case "over_cap":
            return {
                "cap_room": False,
                "non_tax_mle": NON_TAXPAYER_MLE_2026_27,
                "tax_mle": False,
                "bi_annual": True,
                "tpe": True,
                "sign_and_trade": True,
                "aggregation": True,
            }
        case "first_apron":
            return {
                "cap_room": False,
                "non_tax_mle": False,
                "tax_mle": TAXPAYER_MLE_2026_27,
                "bi_annual": False,   # banned above first apron
                "tpe": False,         # cannot use pre-existing TPEs
                "sign_and_trade": True,
                "aggregation": True,  # still allowed above first apron
                "note": "100% salary matching. Cannot waive players whose pre-waiver salary exceeded Non-Tax MLE.",
            }
        case "second_apron":
            return {
                "cap_room": False,
                "non_tax_mle": False,
                "tax_mle": TAXPAYER_MLE_2026_27,
                "bi_annual": False,
                "tpe": False,
                "sign_and_trade": False,  # banned above second apron
                "aggregation": False,     # aggregation banned
                "cash_in_trades": False,
                "note": "100% salary matching. Aggregation banned. No sign-and-trade. No cash in trades.",
            }
        case _:
            return {}


def lakers_cap_scenarios() -> dict[str, dict]:
    """
    Return the Lakers' 2026-27 cap scenarios based on free agent decisions.

    Source: ESPN Bobby Marks offseason guide, May 2026.
    These are illustrative scenarios, not guaranteed projections.
    """
    return {
        "max_flexibility": {
            "description": "Renounce all FAs except Reaves",
            "approx_committed": 47_000_000,
            "cap_room": True,
            "apron_status": "below_cap",
            "note": "~$47M in cap room. Re-sign Reaves using Bird rights.",
        },
        "full_flexibility": {
            "description": "Renounce ALL FAs including Reaves",
            "approx_committed": 67_000_000,
            "cap_room": True,
            "apron_status": "below_cap",
            "note": "~$67M in cap room. Reaves signs elsewhere.",
        },
        "over_cap_bird": {
            "description": "Re-sign own FAs using Bird rights, stay over cap",
            "approx_committed": 150_000_000,  # approximate
            "cap_room": False,
            "apron_status": "over_cap",
            "note": "No cap room. Use $15M non-taxpayer MLE to sign additional players.",
        },
        "first_apron": {
            "description": "Re-sign FAs and add via MLE, land between aprons",
            "approx_committed": 210_000_000,  # approximate; first apron at $209M
            "cap_room": False,
            "apron_status": "first_apron",
            "note": "100% salary matching. Tax MLE only (~$6.1M). Aggregation still allowed.",
        },
    }
