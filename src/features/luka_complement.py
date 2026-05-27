"""
Luka Complement Score — role-fit metric for star-complementary players.

Phase 2 — Luka Complement Score (see README).

Planned responsibilities:
  - Quantify how well any player complements a ball-dominant creator like Luka
  - Candidate input dimensions:
      * Catch-and-shoot 3PT% × volume (gravity and spacing)
      * Defensive versatility: positions guarded effectively, DRPM by matchup
      * Usage rate (low = doesn't compete for creation reps)
      * Off-ball movement metrics (cuts, screens drawn)
      * Switchability: ability to guard multiple positions defensively
  - Formula shaped by clustering output from src/models/ — archetypes first,
    then score each archetype's fit, rather than hand-coding weights
  - Output: luka_complement_score per player-season, plus archetype label

The 2023-24 Mavericks (Washington, Gafford, Jones, Hardy) serve as the
positive validation set. High scores for that cohort = model has face validity.
"""
