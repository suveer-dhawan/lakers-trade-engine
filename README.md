# Lakers Trade Engine
 
A data science research project exploring NBA player valuation, trade analysis, and roster optimization through the lens of the Los Angeles Lakers and Luka Dončić.
 
**Core thesis:** One player valuation model, three decision surfaces — trades, free agency, and extensions. Built to identify undervalued complementary players, evaluate trade packages, and quantify roster fit around a star player.
 
---
 
## Motivation
 
The 2023-24 Mavericks proved that smart acquisitions at the margins — PJ Washington, Daniel Gafford — can transform a roster from good to Finals-bound. This project attempts to build a systematic framework for identifying those moves *prospectively*.
 
This isn't a prediction engine ("will the Celtics accept this trade?"). It's a decision-support tool ("is this trade rational on our end, given what we value?"). The distinction matters — we're modeling our own willingness to pay, not the other side's.
 
---
 
## Project Status
 
> 🚧 **Active development** — this README evolves with the project.
 
| Phase | Status | Description |
|-------|--------|-------------|
| 0 — Data Pipeline | Not started | Ingest stats, salaries, advanced metrics |
| 1 — Player Valuation | Not started | Surplus value model + age curves |
| 2 — Luka Complement Score | Not started | Role-fit metric for star-complementary players |
| 3 — Trade Evaluator | Not started | Asset packaging, pick value, salary matching |
| 4 — Mavs Backtest | Not started | Validate model against 2024 trade deadline |
| 5 — Lakers Offseason App | Not started | Interactive dashboard for 2026 scenarios |
| 6 — Write-up | Not started | Blog series / methodology documentation |
 
---
 
## Data Sources
 
The project relies on publicly available data. Sources may evolve as we discover what's accessible and reliable.
 
### Confirmed — Free & Accessible
 
| Source | What it provides | Access method |
|--------|-----------------|---------------|
| [nba_api](https://github.com/swar/nba_api) (Python) | Per-game stats, advanced stats, shooting splits, play-by-play, on/off data | Python package, pulls from stats.nba.com |
| [Basketball Reference](https://www.basketball-reference.com/) | BPM, VORP, WS, PER, historical data | Scraping (rate limited: 20 req/min) |
| [basketball-reference-scraper](https://github.com/vishaalagartha/basketball_reference_scraper) | Structured scraper for BBRef | Python package |
| [Dunks & Threes](https://dunksandthrees.com/) | EPM (Estimated Plus-Minus) leaderboards | Manual download / scraping |
| [nbarapm.com](https://www.nbarapm.com/) | RAPM, DARKO, LEBRON, RAPTOR cross-metric data | Manual download / scraping |
| [Crafted NBA](https://craftednba.com/) | Player dashboards, meta-metric comparisons | Manual download / scraping |
| [Spotrac](https://www.spotrac.com/) | Salary, contract details, cap holds, trade exceptions | Scraping (paid API exists but not required) |
| [HoopsHype](https://hoopshype.com/salaries/) | Salary data (alternative/supplement to Spotrac) | Scraping |
| [FiveThirtyEight RAPTOR (archived)](https://github.com/fivethirtyeight/nba-player-advanced-metrics) | Historical RAPTOR data | GitHub CSV download |
 
### To Evaluate
 
| Source | Potential use | Notes |
|--------|--------------|-------|
| [balldontlie.io](https://www.balldontlie.io/) | Alternative stats API, free tier | Need to test coverage and rate limits |
| [Highlightly NBA API](https://highlightly.net/nba-api/) | Free tier with 100 req/day | May supplement nba_api for specific endpoints |
| [PBP Stats](https://www.pbpstats.com/) | Possession-based stats, lineup data | Public API for subscribers — evaluate if free tier suffices |
| [Cleaning the Glass](https://cleaningtheglass.com/) | Filtered stats (garbage time removed) | Paid, but methodology is gold standard. May reference without ingesting |
 
### Known Limitations
 
- **No single API covers everything.** The pipeline merges 3-4+ sources. Schema alignment is a real engineering task.
- **Advanced metrics sites don't have formal APIs.** Expect scraping or manual CSV downloads for EPM, RAPTOR, LEBRON.
- **Rate limits are real.** Build a local cache early. Don't hit APIs live for analysis.
- **Salary data has edge cases.** Incentives, trade bonuses, partial guarantees — Spotrac is the most complete but still requires manual verification for complex contracts.
- **CBA rules are encoded as logic, not pulled from data.** The apron constraints, salary matching rules, and trade exceptions are implemented in code based on the 2023 CBA documentation.
---
 
## Architecture
 
```
lakers-trade-engine/
├── README.md
├── notebooks/                # Exploration & analysis (numbered sequentially)
│   ├── 01_data_exploration.ipynb
│   ├── 02_player_valuation.ipynb
│   ├── 03_luka_complement_score.ipynb
│   ├── 04_mavs_backtest.ipynb
│   └── 05_lakers_offseason.ipynb
├── src/                      # Refactored production code
│   ├── data/                 # Ingestion, caching, schema alignment
│   ├── features/             # Feature engineering, composite metrics
│   ├── models/               # Valuation, age curves, clustering, contract prediction
│   └── utils/                # Helpers, CBA rules, salary matching logic
├── dashboard/                # Streamlit app (Phase 5)
├── data/                     # Local cache — parquet files, SQLite (gitignored)
├── blog/                     # Write-up drafts, figures
├── tests/
└── pyproject.toml
```
 
**Stack:**
- **Language:** Python
- **Data:** pandas, parquet files (SQLite if needed)
- **Modeling:** scikit-learn, scipy
- **Visualization:** plotly, matplotlib/seaborn (notebooks), Streamlit (dashboard)
- **Deployment:** Streamlit Cloud (free) for the dashboard
---
 
## Methodology (will evolve)
 
### Player Valuation
 
Composite value score combining:
- **On-court value:** Weighted blend of public advanced metrics (EPM, BPM, WS/48 — specific weights TBD through exploration)
- **Contract cost:** Salary as percentage of cap, years remaining
- **Surplus value:** On-court value minus contract cost — the inefficiency signal
- **Age-curve adjustment:** Positional decline curves fit from historical data to project remaining value over contract duration

### Luka Complement Score
 
Quantified role-fit metric rather than subjective "3&D" labels. Candidate inputs:
- Catch-and-shoot 3PT% × volume
- Defensive versatility (positions guarded effectively)
- Low usage rate (doesn't need the ball)
- Off-ball movement / gravity metrics
- Switchability on defense
*The specific formula will be shaped by what the clustering reveals — let the data define archetypes rather than hand-coding them.*
 
### Trade Evaluator
 
- Player surplus value (from above)
- Draft pick probabilistic value (historical production by pick + team projected finish)
- Pick protection modeling (Monte Carlo simulation of outcomes)
- Salary matching under CBA rules (trade exceptions, apron constraints)
- Net value exchanged = what you send minus what you receive
### Where ML/AI Fits
 
| Technique | Application | Complexity |
|-----------|-------------|------------|
| Regression | Age-curve projections by position/archetype | Core |
| Clustering (k-means or similar) | Player archetype identification | Core |
| Monte Carlo simulation | Draft pick value under protections | Core |
| Contract prediction model | Predict market value, find overpays/underpays | Stretch |
| Similarity scoring | "Players like PJ Washington pre-trade" | Stretch |
 
What we're **not** doing: deep learning (dataset too small), LLM chatbot wrappers (adds flash, not substance), real-time prediction systems.
 
---
 
## Development Notes
 
This project is built iteratively. Findings from each phase inform the next. The methodology section above represents starting hypotheses — expect them to change as we explore the data.
 
**Backtest anchor:** The 2023-24 Mavericks trade deadline (PJ Washington, Daniel Gafford acquisitions) serves as our primary validation case. If the model can identify those moves as high-value prospectively, it has face validity.
 
---
 
## License
 
TBD
 
---
 
## Acknowledgments
 
Data sourced from NBA.com (via nba_api), Basketball Reference, Dunks & Threes, nbarapm.com, Crafted NBA, Spotrac, and the FiveThirtyEight RAPTOR archive. This project is for research and portfolio purposes — not affiliated with the NBA, the Los Angeles Lakers, or any NBA franchise.