"""
Composite player valuation model.

Phase 1 — Player Valuation (see README).

Planned responsibilities:
  - Aggregate advanced metrics (EPM, BPM, WS/48, RAPTOR) into a single
    on-court value score via a weighted blend (weights determined empirically)
  - Compute contract cost as salary / salary_cap percentage
  - Surplus value = on-court value score − contract cost (the inefficiency signal)
  - Apply age-curve adjustment from src/models/age_curves.py to project
    surplus value over the remaining contract duration
  - Output a player-season DataFrame with columns:
    [player_id, player_name, season, on_court_value, contract_cost,
     surplus_value, age_adjusted_surplus]

The backtest anchor is the 2023-24 Mavericks deadline: the model should
rank PJ Washington and Daniel Gafford as high-surplus acquisitions
prospectively to establish face validity.
"""
