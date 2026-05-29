"""
Trade package evaluation engine.

Phase 2A.3 -- Trade Evaluator.

Given a target player, generates and ranks all legal trade packages the Lakers
could construct under the 2026-27 CBA, using their available contracts and picks.

Design:
  1. Salary matching via cba_rules.salary_match_options()
  2. Pick value from pick_value.estimate_pick_value()
  3. Player value from the ON_COURT_VALUE / SURPLUS_VALUE in the Phase 1 DataFrame
  4. Trade fairness heuristic: offering team needs ~110-130% of target value to get a deal

Outputs TradePackage dataclasses, sorted by descending feasibility score.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from src.models.pick_value import check_stepien_rule, estimate_pick_value, pick_combination_value
from src.utils.cba_rules import salary_match_options, trade_is_legal
from src.utils.constants import LAKERS_DRAFT_PICKS, LAKERS_TRADEABLE, RESTRICTED_FREE_AGENTS

logger = logging.getLogger(__name__)

# Minimum premium the offering team must provide for a trade to be realistic.
# Stars command higher premiums; role players closer to 1.10.
_FAIR_VALUE_MULTIPLIER_LOW = 1.10
_FAIR_VALUE_MULTIPLIER_HIGH = 1.30


@dataclass
class TradePackage:
    """
    A single proposed trade package from the Lakers' perspective.

    All value figures use ON_COURT_VALUE units (z-score based; 0 = avg rotation player).
    Pick values are in $M surplus (from pick_value.py).
    """
    target_player: str
    target_salary: float
    target_value: float                   # ON_COURT_VALUE from Phase 1
    outgoing_players: list[str]
    outgoing_salary: float
    outgoing_value: float                 # sum of outgoing players' ON_COURT_VALUE
    picks_included: list[str] = field(default_factory=list)
    picks_value_M: float = 0.0           # total pick value in $M surplus
    is_legal: bool = True
    legality_reason: str = ""
    lakers_net_value: float = 0.0        # positive = Lakers gain court value
    other_team_value_premium: float = 0.0  # ratio: what other team gets / target value
    feasibility_score: float = 0.0       # 0-1, higher = more likely deal gets done
    is_rfa: bool = False                  # target is restricted free agent
    apron_status: str = "first_apron"

    def summary(self) -> str:
        status = "LEGAL" if self.is_legal else "ILLEGAL"
        picks_str = ", ".join(self.picks_included) if self.picks_included else "none"
        return (
            f"[{status}] {', '.join(self.outgoing_players)} + picks [{picks_str}] "
            f"-> {self.target_player} | "
            f"Out: ${self.outgoing_salary:,.0f} | In: ${self.target_salary:,.0f} | "
            f"Feasibility: {self.feasibility_score:.2f}"
        )


def _lookup_player_value(player_name: str, df: pd.DataFrame) -> float:
    """
    Look up a player's ON_COURT_VALUE in the valued DataFrame.
    Returns 0.0 if not found (treats unknown players as replacement level).
    """
    matches = df[df["PLAYER_NAME"].str.lower() == player_name.lower()]
    if matches.empty:
        # Fuzzy: try contains
        matches = df[df["PLAYER_NAME"].str.lower().str.contains(player_name.lower(), na=False)]
    if matches.empty:
        logger.warning("Player not found in dataset: %s", player_name)
        return 0.0
    val = matches.iloc[0]["ON_COURT_VALUE"]
    return float(val) if pd.notna(val) else 0.0


def _lookup_player_salary(player_name: str, df: pd.DataFrame) -> float | None:
    """
    Look up a player's SALARY_FORWARD in the valued DataFrame.
    Returns None if not found.
    """
    matches = df[df["PLAYER_NAME"].str.lower() == player_name.lower()]
    if matches.empty:
        matches = df[df["PLAYER_NAME"].str.lower().str.contains(player_name.lower(), na=False)]
    if matches.empty:
        return None
    sal = matches.iloc[0].get("SALARY_FORWARD")
    return float(sal) if pd.notna(sal) else None


def evaluate_trade_fairness(
    package: TradePackage,
    value_premium_required: float = _FAIR_VALUE_MULTIPLIER_LOW,
) -> float:
    """
    Heuristic feasibility score for a trade package (0 to 1).

    A trade needs to make sense for BOTH sides:
      - Lakers get the target (already established by Phase 1 archetype fit)
      - Other team gets assets exceeding the target's value by a premium

    Rule of thumb: offering team needs to send ~110-130% of the target's value.
    Stars command higher premiums; role players closer to 110%.

    Returns a score from 0 (non-starter) to 1 (very attractive offer).
    """
    if not package.is_legal:
        return 0.0

    # Value the other team receives: outgoing player value + pick value premium
    # Pick value ($M) is converted to an approximate on-court value units using
    # a rough calibration: $1M surplus ~ 0.05 ON_COURT_VALUE units
    PICK_VALUE_SCALE = 0.05
    other_team_receives = package.outgoing_value + package.picks_value_M * PICK_VALUE_SCALE

    if package.target_value <= 0:
        # Target has minimal court value; any legal match is feasible
        return 0.7 if package.is_legal else 0.0

    premium_ratio = other_team_receives / max(package.target_value, 0.01)
    package.other_team_value_premium = premium_ratio

    # Score peaks at 1.0 when ratio >= value_premium_required
    # Drops steeply below that threshold
    if premium_ratio >= _FAIR_VALUE_MULTIPLIER_HIGH:
        score = 1.0
    elif premium_ratio >= value_premium_required:
        # Linear interpolation between required and high premium
        score = 0.70 + 0.30 * (premium_ratio - value_premium_required) / (
            _FAIR_VALUE_MULTIPLIER_HIGH - value_premium_required
        )
    elif premium_ratio >= 0.90:
        # Below threshold but close -- deal possible with negotiation
        score = 0.30 + 0.40 * (premium_ratio - 0.90) / (value_premium_required - 0.90)
    else:
        score = max(0.0, premium_ratio * 0.30 / 0.90)

    # Penalize RFA targets (current team can match any offer)
    if package.is_rfa:
        score *= 0.60

    return round(min(1.0, score), 3)


def build_trade_packages(
    target_player_name: str,
    df: pd.DataFrame,
    lakers_contracts: dict[str, float] | None = None,
    lakers_picks: dict[str, dict] | None = None,
    apron_status: str = "first_apron",
    max_packages: int = 10,
    include_picks: list[str] | None = None,
    max_players_out: int = 4,
) -> list[TradePackage]:
    """
    Generate and rank legal trade packages for a target player.

    Parameters
    ----------
    target_player_name : str
        Name of the player the Lakers want to acquire.
    df : pd.DataFrame
        Valued player dataset from Phase 1 (output of compute_player_value +
        compute_luka_complement). Must have PLAYER_NAME, ON_COURT_VALUE, SALARY_FORWARD.
    lakers_contracts : dict {player_name: salary}, optional
        Tradeable Lakers contracts. Defaults to LAKERS_TRADEABLE from constants.py.
    lakers_picks : dict, optional
        Draft pick inventory. Defaults to LAKERS_DRAFT_PICKS from constants.py.
    apron_status : str
        'below_aprons', 'first_apron', or 'second_apron'. Determines matching rules.
    max_packages : int
        Maximum number of packages to return.
    include_picks : list[str], optional
        List of pick keys to include in ALL packages (e.g. always attach '2026_1st_25').
        Pass [] or None for no mandatory picks.
    max_players_out : int
        Maximum players to send in a single package.

    Returns
    -------
    List of TradePackage objects, sorted by feasibility_score descending.
    """
    if lakers_contracts is None:
        lakers_contracts = LAKERS_TRADEABLE
    if lakers_picks is None:
        lakers_picks = LAKERS_DRAFT_PICKS
    if include_picks is None:
        include_picks = []

    # ------------------------------------------------------------------
    # Step 1: Get target's salary and value
    # ------------------------------------------------------------------
    target_salary = _lookup_player_salary(target_player_name, df)
    if target_salary is None:
        # Try lakers_contracts (target might be on Lakers)
        logger.warning(
            "Target player '%s' salary not found in dataset. Cannot build packages.",
            target_player_name,
        )
        return []

    target_value = _lookup_player_value(target_player_name, df)
    is_rfa = target_player_name in RESTRICTED_FREE_AGENTS

    logger.info(
        "Building trade packages for %s | salary: $%,.0f | value: %.3f | RFA: %s",
        target_player_name, target_salary, target_value, is_rfa,
    )

    # ------------------------------------------------------------------
    # Step 2: Compute pick value being included
    # ------------------------------------------------------------------
    picks_val_M = 0.0
    picks_stepien_ok = True
    picks_stepien_reason = ""
    if include_picks:
        valid_stepien, picks_stepien_reason = check_stepien_rule(include_picks)
        picks_stepien_ok = valid_stepien
        if valid_stepien:
            picks_val_M = pick_combination_value(include_picks)
        else:
            logger.warning("Stepien Rule violation in include_picks: %s", picks_stepien_reason)

    # ------------------------------------------------------------------
    # Step 3: Find all salary-matching player combinations
    # ------------------------------------------------------------------
    player_combos = salary_match_options(
        target_salary=target_salary,
        lakers_contracts=lakers_contracts,
        apron_status=apron_status,
        max_players_out=max_players_out,
    )

    if not player_combos:
        logger.info(
            "No salary-matching combinations found for %s ($%,.0f) under %s rules.",
            target_player_name, target_salary, apron_status,
        )

    # ------------------------------------------------------------------
    # Step 4: Build and score each package
    # ------------------------------------------------------------------
    packages: list[TradePackage] = []
    seen_combos: set[frozenset] = set()

    for combo in player_combos:
        combo_key = frozenset(combo)
        if combo_key in seen_combos:
            continue
        seen_combos.add(combo_key)

        outgoing_salary = sum(lakers_contracts.get(p, 0) for p in combo)
        outgoing_value = sum(_lookup_player_value(p, df) for p in combo)

        is_legal, reason, _ = trade_is_legal(
            outgoing_salary=outgoing_salary,
            incoming_salary=target_salary,
            team_total_salary=0,
            apron_status=apron_status,
            n_outgoing_players=len(combo),
        )

        # Check Stepien compliance if picks included
        final_picks = include_picks if picks_stepien_ok else []
        final_picks_val = picks_val_M if picks_stepien_ok else 0.0
        if not picks_stepien_ok:
            reason += f" Picks excluded: {picks_stepien_reason}"

        pkg = TradePackage(
            target_player=target_player_name,
            target_salary=target_salary,
            target_value=target_value,
            outgoing_players=list(combo),
            outgoing_salary=outgoing_salary,
            outgoing_value=outgoing_value,
            picks_included=final_picks,
            picks_value_M=final_picks_val,
            is_legal=is_legal,
            legality_reason=reason,
            lakers_net_value=target_value - outgoing_value,
            is_rfa=is_rfa,
            apron_status=apron_status,
        )
        pkg.feasibility_score = evaluate_trade_fairness(pkg)
        packages.append(pkg)

    # ------------------------------------------------------------------
    # Step 5: Sort and return top packages
    # ------------------------------------------------------------------
    packages.sort(key=lambda p: (-p.feasibility_score, -p.lakers_net_value))

    if not packages:
        logger.info("No valid packages found for %s.", target_player_name)
    else:
        logger.info(
            "Generated %d packages for %s. Top feasibility: %.2f",
            len(packages), target_player_name, packages[0].feasibility_score,
        )

    return packages[:max_packages]


def packages_to_dataframe(packages: list[TradePackage]) -> pd.DataFrame:
    """Convert a list of TradePackage objects to a display DataFrame."""
    if not packages:
        return pd.DataFrame()

    rows = []
    for pkg in packages:
        rows.append({
            "Target": pkg.target_player,
            "Target Salary": f"${pkg.target_salary:,.0f}",
            "Outgoing Players": " + ".join(pkg.outgoing_players),
            "Outgoing Salary": f"${pkg.outgoing_salary:,.0f}",
            "Picks": ", ".join(pkg.picks_included) if pkg.picks_included else "-",
            "Pick Value ($M)": f"{pkg.picks_value_M:.1f}",
            "Legal": "Yes" if pkg.is_legal else "No",
            "Lakers Net Value": f"{pkg.lakers_net_value:+.3f}",
            "Other Team Premium": f"{pkg.other_team_value_premium:.2f}x",
            "Feasibility": f"{pkg.feasibility_score:.2f}",
            "RFA Risk": "Yes" if pkg.is_rfa else "No",
            "Legality Note": pkg.legality_reason,
        })
    return pd.DataFrame(rows)


def evaluate_single_trade(
    outgoing_players: list[str],
    incoming_player: str,
    df: pd.DataFrame,
    picks_included: list[str] | None = None,
    apron_status: str = "first_apron",
    lakers_contracts: dict[str, float] | None = None,
) -> TradePackage:
    """
    Evaluate a single user-defined trade scenario.

    Useful for the dashboard Trade Simulator page where the user selects
    specific assets rather than letting the engine enumerate all combinations.

    Parameters
    ----------
    outgoing_players : list of Lakers player names being sent out.
    incoming_player : name of the player being received.
    df : valued player DataFrame.
    picks_included : list of pick keys from LAKERS_DRAFT_PICKS to include.
    apron_status : team's apron situation.
    lakers_contracts : tradeable contract dict (defaults to LAKERS_TRADEABLE).

    Returns
    -------
    TradePackage with legality, value, and feasibility assessed.
    """
    if lakers_contracts is None:
        lakers_contracts = LAKERS_TRADEABLE
    if picks_included is None:
        picks_included = []

    outgoing_salary = sum(lakers_contracts.get(p, 0) for p in outgoing_players)
    target_salary = _lookup_player_salary(incoming_player, df) or 0.0
    target_value = _lookup_player_value(incoming_player, df)
    outgoing_value = sum(_lookup_player_value(p, df) for p in outgoing_players)

    is_legal, reason, _ = trade_is_legal(
        outgoing_salary=outgoing_salary,
        incoming_salary=target_salary,
        team_total_salary=0,
        apron_status=apron_status,
        n_outgoing_players=len(outgoing_players),
    )

    picks_val_M = pick_combination_value(picks_included) if picks_included else 0.0
    stepien_ok, stepien_reason = check_stepien_rule(picks_included) if picks_included else (True, "")
    if not stepien_ok:
        is_legal = False
        reason += f" Stepien Rule: {stepien_reason}"
        picks_val_M = 0.0

    pkg = TradePackage(
        target_player=incoming_player,
        target_salary=target_salary,
        target_value=target_value,
        outgoing_players=outgoing_players,
        outgoing_salary=outgoing_salary,
        outgoing_value=outgoing_value,
        picks_included=picks_included,
        picks_value_M=picks_val_M,
        is_legal=is_legal,
        legality_reason=reason,
        lakers_net_value=target_value - outgoing_value,
        is_rfa=incoming_player in RESTRICTED_FREE_AGENTS,
        apron_status=apron_status,
    )
    pkg.feasibility_score = evaluate_trade_fairness(pkg)
    return pkg
