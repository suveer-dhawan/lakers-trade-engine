"""
Probabilistic draft pick value model.

Phase 3 — Trade Evaluator (see README).

Planned responsibilities:
  - Historical production by draft slot: regress career value metrics on pick number
    to build a base pick-value curve (similar in spirit to Massey-Thaler)
  - Team finish projection: estimate the pick's likely lottery position given
    current roster quality signals
  - Protection modeling: Monte Carlo simulation over projected finish distributions
    to compute expected value of a protected pick
      e.g. "top-4 protected" → P(conveys) × E[value | conveys] + P(slides) × E[next year]
  - Output: scalar expected value for any pick specification, with confidence interval

This feeds directly into src/models/trade_eval.py for asset-level valuation.
"""
