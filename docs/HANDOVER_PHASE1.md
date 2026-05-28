# Lakers Trade Engine — Chat Handover Document
**Date:** May 28, 2026
**Purpose:** Full context transfer to continue this project in a new chat session.

---

## What This Project Is

A data science research project building an NBA player valuation and trade analysis engine. The primary lens is the Los Angeles Lakers and roster optimization around Luka Dončić, but the underlying model is league-wide by design. The end deliverable is a Streamlit dashboard (interactive, shareable) backed by rigorous analysis in Jupyter notebooks, with a blog post series as secondary output.

**GitHub repo:** https://github.com/suveer-dhawan/lakers-trade-engine.git

---

## Current State (End of Phase 1.0)

### What's Built and Working

**Data Pipeline (Phase 0 — Complete):**
- `src/data/nba_stats.py` — nba_api ingestion (base stats, advanced, shooting splits, defense). Cached as parquet. Covers 2019-2026 seasons.
- `src/data/bbref_stats.py` — Basketball Reference scraper + CSV loader. CSV is the primary path (scraping was flaky due to lxml issues). Manual CSVs in `data/raw/`.
- `src/data/salaries.py` — BBRef salary CSV loader. Derives SALARY_FORWARD, YEARS_REMAINING, IS_EXPIRING, TOTAL_GUARANTEED, AVG_ANNUAL_VALUE.
- `src/data/advanced_metrics.py` — EPM ingestion exists but EPM is paywalled ($50). Skipped; BPM from BBRef substitutes.
- `src/data/merge.py` — Merges all sources into one player-level DataFrame. 582 players × 185 columns for 2025-26.
- `src/data/cache.py` — Parquet-based caching layer. All API calls go through this.

**Manual CSV Data Files (in `data/raw/`):**
- `bbref_advanced_2023-24.csv`, `bbref_advanced_2024-25.csv`, `bbref_advanced_2025-26.csv` — BBRef advanced stats (PER, WS, BPM, VORP)
- `salaries_2025-26.csv` — BBRef salary table with multi-year contract columns through 2030-31
- No historical salary data (couldn't find 2023-24 salaries on BBRef)
- No EPM data (paywalled at dunksandthrees.com)

**Player Valuation (Phase 1.0 — Complete but needs refinement):**
- `src/features/player_value.py` — Full valuation pipeline:
  - Z-scores for 6 metrics (BPM, VORP, WS, WS/48, NET_RATING, TS%)
  - ON_COURT_VALUE = weighted composite (BPM 35%, VORP 25%, WS/48 15%, NET_RATING 15%, TS% 10%)
  - SALARY_FORWARD = 2026-27 salary (forward-looking, not current season)
  - SURPLUS_VALUE = on-court value minus contract cost (z-score space)
  - IS_FREE_AGENT flag for players with no 26-27 salary
  - FA_VALUE for free agents (pure on-court value, no contract denominator)
- `src/features/luka_complement.py` — Two approaches:
  - K-means clustering (k=8) on 11 style features → identifies complement archetypes
  - Roll Gravity Score = z(FG_PCT) + z(BLK%) + z(OREB_PCT) - z(USG_PCT)
  - Combined target score blending surplus + complement fit
- `src/features/durability.py` — Recency-weighted games played % + major injury flags. Module exists but was skipped in Phase 1 notebook run (add_durability=False) to avoid API re-queries.
- Valued dataset saved: `data/cache/player_valued_2025-26.parquet` (582 × 210 columns)

**Notebooks (all execute cleanly):**
- `00_setup_test.ipynb` — Environment validation ✅
- `01_data_exploration.ipynb` — nba_api data discovery ✅
- `02_data_completeness.ipynb` — Source integration validation ✅
- `03_phase1_ready.ipynb` — BBRef + salary integration confirmed ✅
- `04_player_valuation.ipynb` — Full Phase 1 results (38 cells, visualizations in Lakers colors) ✅

**Other files:**
- `CLAUDE.md` — Project context for AI assistants (update after each phase)
- `README.md` — Living PRD with design principles, data sources, architecture
- `data/raw/MANUAL_DOWNLOAD_GUIDE.md` — Instructions for manual CSV downloads
- `Makefile` — setup, test, notebook, dashboard, lint commands
- `pyproject.toml` — Python project config with all dependencies

### Key Results from Phase 1.0

**Valuation model passes smell test:**
- Top 4 ON_COURT_VALUE: Jokic (4.16) → SGA (3.73) → Wembanyama (3.07) → Luka (2.56)
- Most undervalued: Wembanyama (supermax production, rookie salary), Neemias Queta, SGA
- Most overpaid: Zach LaVine, Anthony Davis (now on Wizards at 33), Jaren Jackson Jr.

**Lakers 26-27 cap situation:**
- ~$105M committed: Luka ($49.8M), Reaves ($14.9M), Vanderbilt ($12.4M), Ayton ($8.1M), LaRavia ($6M), others
- Expiring off books: LeBron ($52.6M), Rui ($18.3M), Kennard ($11M), Kleber ($11M), Bufkin ($5.5M), Hayes ($3.5M)
- 8 expiring contracts = significant flexibility
- Projected cap $157M, first apron $199M, second apron $212M

**Top combined targets identified:**
- Trade targets: Santi Aldama (MEM), PJ Washington (DAL), Jaylen Wells (MEM), Tristan da Silva (ORL), Cameron Johnson (DEN)
- Free agent targets: Jalen Duren (DET), Robert Williams III (POR), Mitchell Robinson (NYK), Jaxson Hayes (LAL — re-sign?)

**Clustering found 3 Luka complement archetypes:**
- Cluster 0 (87 players): High catch-and-shoot volume/%, low usage — the "3&D wing" archetype
- Cluster 6 (26 players): Rim protectors / lob threats — the "rim runner" archetype  
- Cluster 7 (58 players): Mixed archetype (PJ Washington landed here)

---

## Known Issues and What Needs Fixing

### Critical: Clustering Validation Is Fragile
The clustering used 6 validation players but only 3 were found in the 2025-26 dataset:
- PJ Washington (cluster 7), Daniel Gafford (cluster 6), Dorian Finney-Smith (cluster 0)
- Derrick Jones Jr — NOT in 2025-26 dataset
- Tim Hardaway Jr — NOT in 2025-26 dataset  
- Maxi Kleber — excluded due to low minutes

**Three data points across three clusters is too fragile.** The clustering needs to be re-run with:
1. An expanded validation list (see below)
2. Multi-season data — pull validation players from the seasons they actually played with Luka
3. A "secondary creator" archetype is completely missing (Kyrie Irving type)

### Expanded Validation Player List (agreed upon, not yet implemented)

**Rim Runners / Lob Threats:**
- Daniel Gafford (1629655) — 2023-24 Mavs
- Dereck Lively II (look up ID) — 2023-24 Mavs
- Jaxson Hayes (1629637) — current Laker, lob threat

**3&D Wings:**
- PJ Washington (1629023) — 2023-24 Mavs
- Dorian Finney-Smith (1627827) — 2022-23 Mavs (traded to BKN Feb 2023, was NOT on 23-24 Finals team)
- Derrick Jones Jr (1628407) — 2023-24 Mavs

**Secondary Creators:**
- Kyrie Irving (202681) — 2023-24 Mavs (proven Luka complement)
- Jalen Brunson (1628973) — 2020-22 Mavs (left for Knicks in FA summer 2022)
- Spencer Dinwiddie (203915) — 2021-22 Mavs stint

**Pure Shooters:**
- Tim Hardaway Jr (203501) — early Luka-era Mavs designated shooter (he is a 6'5" SG, NOT a stretch big — this was misclassified in an earlier conversation)
- Maxi Kleber (1628467) — stretch big / floor spacer across multiple Luka seasons

**IMPORTANT: These players need to be pulled from the SPECIFIC SEASONS they played with Luka, not current 2025-26 data.** DFS data comes from 2022-23, Brunson from 2021-22, etc. The nba_api cache covers 2019-2026 so the data exists.

**Also consider:** Austin Reaves as a current secondary creator alongside Luka (per analyst consensus). His contract extension valuation is relevant to the Lakers specifically.

### Combined Score Formula Needs Rebalancing
The current combined target score may overweight complement distance relative to actual value. Santi Aldama at #1 trade target with surplus of 0.082 suggests the weighting is off. Review the formula in `luka_complement.py` and consider whether complement distance should be a filter (must be below threshold) rather than a continuous input to the score.

### Durability Not Yet Integrated in Results
The durability module exists but Phase 1 notebook ran with `add_durability=False`. Needs to be enabled — this is important for trade targets where injury history matters (Robert Williams III is a great example: elite rim runner, chronic injury problems).

### Season Context Matters
- **2025-26 stats** = current production (primary analysis season)
- **2026-27 salary** = forward-looking contract cost (the offseason we're building for)
- **2023-24 data** = Mavs backtest season (did model identify Gafford/PJW as undervalued pre-deadline?)
- The mismatch between stats year (25-26) and salary year (26-27) is a valid analytical choice: we're using best available performance data to evaluate future contract value, same as any front office. Should be documented in methodology.

### Contract Nuances Missing
- Player options vs team options (lost in CSV export from BBRef — color coded on their site)
- RFA vs UFA distinction for free agents
- These can be manually enriched for the ~30-50 players the model identifies as interesting targets
- Not needed for all 582 players

---

## Agreed Design Principles (from earlier discussion)

1. **Generalize first.** League-wide valuation before team-specific filters. Lakers lens is a query layer.
2. **Empirically derived archetypes.** Cluster on all features, observe which clusters successful Luka teammates fall into. Don't pre-select features.
3. **Contractual utility matters.** Expiring deals and multi-year deals have different trade utility under second-apron rules.
4. **Acknowledge system effects.** Player efficiency is partly a product of teammates. We measure output, not isolated quality. State this limitation.
5. **Document as you code.** Notebook markdown cells explain the why before the code.
6. **The "lob threat" / roll gravity variable** is a heavily weighted feature based on evidence from Luka's career (DJJ, Lively, Gafford all thrived catching lobs).

---

## Workflow: How This Project Uses AI

**This chat (Claude.ai):** Research advisor / senior engineer. Discusses methodology, evaluates results, makes architecture decisions, writes Claude Code task briefs.

**Claude Code (VS Code):** Executor. Takes scoped task briefs and implements them. One session per task.

**The human (Suveer):** PM / domain expert. Makes modeling decisions with basketball knowledge, evaluates whether results make sense, orchestrates between the two AI tools.

---

## Phase Plan (Updated Priority)

| Phase | Status | Description |
|-------|--------|-------------|
| 0 — Data Pipeline | ✅ Complete | nba_api + BBRef CSV + salary data |
| 1.0 — Player Valuation | ✅ Complete | On-court value, surplus value, clustering, roll gravity |
| **1.5 — Clustering Fix** | **Next** | **Re-run with expanded validation list + multi-season data** |
| 2 — Trade Evaluator | Planned | Asset packaging, pick value, salary matching, CBA rules |
| 3 — Streamlit Dashboard | Planned (pulled forward) | Interactive exploration of valuations and trade scenarios |
| 4 — Mavs Backtest | Planned | Validate model against 2023-24 trade deadline |
| 5 — Blog / Write-up | Planned | Generated from dashboard methodology pages |

### Immediate Next Steps (Phase 1.5)

1. **Fix clustering with expanded multi-season validation list** (details above)
2. **Enable durability scoring** in the valuation pipeline
3. **Re-run Phase 1 notebook** with both fixes and evaluate updated target list
4. **Rebalance combined target score** formula based on results
5. **Add "secondary creator" archetype** to the complement model (Kyrie/Brunson type — not just 3&D)

### After Phase 1.5

6. Start Phase 2 (trade evaluator) — this needs CBA salary matching rules, pick value modeling
7. Begin Streamlit dashboard scaffolding in parallel — even a basic "browse player valuations" page is valuable early

---

## Technical Notes

- **Python 3.10+**, dependencies in pyproject.toml
- All API calls go through `src/data/cache.py` — never hit an API without caching
- Rate limits: nba_api = 0.6s delay, Basketball Reference = 3s delay
- Data joins use PLAYER_ID (nba_api numeric) as primary key; BBRef/salary join on normalized player name
- Parquet for all cached data, CSVs only for manual data entry in `data/raw/`
- Constants in `src/utils/constants.py`: team IDs, salary cap figures, CBA thresholds
- 2026-27 projected salary cap: $157M, first apron: $199M, second apron: $212M
- Valued dataset: `data/cache/player_valued_2025-26.parquet` (582 × 210 columns)

---

## Files the New Chat Needs Access To

If starting a new chat, provide:
1. This handover document (HANDOVER.md)
2. The Phase 1 notebook (04_player_valuation.ipynb) — or at minimum, Claude Code's summary of results
3. The CLAUDE.md from the repo (for project conventions)

The new chat can pick up from Phase 1.5 with full context.
