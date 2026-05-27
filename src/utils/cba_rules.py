"""
CBA salary matching and apron constraint logic (2023 NBA Collective Bargaining Agreement).

Phase 3 — Trade Evaluator (see README).

Planned responsibilities:
  - Validate whether a trade is legal under salary matching rules:
      * Teams above first apron: incoming salary ≤ 110% of outgoing
      * Teams below first apron: incoming ≤ outgoing × TRADE_SALARY_MULTIPLIER + flat add
  - Compute available trade exceptions (TPE) and whether they can be used
  - Flag second-apron restrictions: teams above SECOND_APRON (constants.py) cannot
    aggregate salaries in trades, cannot use BAE, cannot sign-and-trade
  - Mid-level exception availability check based on team tax status
  - Minimum salary exception rules

CBA rules are encoded as logic, not pulled from data — they come from the
2023 CBA documentation and must be manually updated if the CBA changes.
Reference constants: src/utils/constants.py (thresholds), src/data/config.py (paths).
"""
