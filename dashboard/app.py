"""
Streamlit dashboard — Lakers Offseason Scenario Explorer.

Phase 5 — Lakers Offseason App (see README).

Planned responsibilities:
  - Interactive trade builder: select outgoing/incoming assets, see real-time
    valuation and CBA legality check powered by src/models/trade_eval.py
  - Player comparison view: surplus value, age curve, Luka complement score
    side-by-side for any two players in the dataset
  - Free agency explorer: filter available players by position, salary range,
    complement score; sort by surplus value
  - Mavs backtest panel: replay the 2024 trade deadline, show what the model
    would have recommended vs. what actually happened
  - Deployment target: Streamlit Cloud (free tier, public URL)

Run locally: streamlit run dashboard/app.py  (or: make dashboard)
"""
