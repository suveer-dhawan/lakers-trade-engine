"""
Trade package evaluation engine.

Phase 3 — Trade Evaluator (see README).

Planned responsibilities:
  - Accept a trade definition: outgoing assets (players + picks), incoming assets
  - Validate salary matching against 2023 CBA rules via src/utils/cba_rules.py
  - Compute net value exchanged:
      net = sum(surplus_value of incoming assets) − sum(surplus_value of outgoing)
  - Include draft pick probabilistic value from src/models/pick_value.py
  - Flag apron implications: does the trade push the team above FIRST_APRON or
    SECOND_APRON (constants.py), and what restrictions does that trigger?
  - Output a structured trade report with value breakdown by asset

Primary use case (Phase 5): interactive Lakers offseason scenario explorer
powered by the Streamlit dashboard (dashboard/app.py).
"""
