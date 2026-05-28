# Manual Download Guide

The scrapers for EPM and salary data are unreliable (JS-heavy sites, anti-scraping). The
BBRef scraper works but can break if BBRef changes their HTML structure. This guide explains
how to get clean CSVs into `data/raw/` as the reliable fallback path.

Once downloaded, the pipeline uses these files via:
- `load_epm(season="2024-25")` -- reads `data/raw/epm_2024-25.csv`
- `load_salaries(season="2024-25")` -- reads `data/raw/salaries_2024-25.csv`
- BBRef CSV: pass directly to `pd.read_csv()` then clean manually (or wait for the scraper)

---

## 1. BBRef Advanced Stats (backup if scraper is flaky)

**URL:** https://www.basketball-reference.com/leagues/NBA_{YEAR}_advanced.html
  (e.g., https://www.basketball-reference.com/leagues/NBA_2025_advanced.html for 2024-25)

**Steps:**
1. Open the page and scroll to the "Advanced" table.
2. Click the **"Share & Export"** dropdown above the table.
3. Select **"Get table as CSV (for Excel)"**.
4. Copy-paste the output into a file named `data/raw/bbref_advanced_{YEAR}.csv`
   (e.g., `bbref_advanced_2025.csv` for the 2024-25 season).

**Key columns:** Player, Tm, Pos, Age, G, MP, PER, TS%, WS, WS/48, BPM, VORP, OBPM, DBPM

**Note:** The BBRef CSV has a repeated header row every 20 players. Drop rows where
`Rk == "Rk"` after loading.

---

## 2. EPM (Estimated Plus-Minus) -- Dunks & Threes

**URL:** https://dunksandthrees.com/epm

**Steps:**
1. Go to the URL above and select the season you want from the dropdown.
2. Look for an **Export** or **Download CSV** button (usually top-right of the table).
   If unavailable, use browser DevTools:
   - Open DevTools -> Network tab -> reload the page
   - Filter by XHR/Fetch requests, find one returning JSON with player stats
   - Copy the response and save as `data/raw/epm_{season}.json`
3. Save the file as `data/raw/epm_{season}.csv` (e.g., `epm_2024-25.csv`).

**Required columns (flexible naming, `load_epm()` normalizes):**
- Player name: `player`, `name`, or `player_name`
- Team: `team` or `tm`
- EPM total: `epm` or `total_epm`
- Optional: offensive EPM (`o_epm`, `epmo`), defensive EPM (`d_epm`, `epmd`), games, minutes

**Load:**
```python
from src.data.advanced_metrics import load_epm
df = load_epm()                      # defaults to data/raw/epm_2024-25.csv
df = load_epm(season="2023-24")      # loads data/raw/epm_2023-24.csv
```

---

## 3. Salary Data

### Option A: HoopsHype (easier, current-season salaries)

**URL:** https://hoopshype.com/salaries/players/

**Steps:**
1. Open the page. The table shows all current-season player salaries with team.
2. Use your browser's "Select All" -> copy, or right-click -> Save as (HTML).
   Alternatively, click **Export** if available, or use the table copy-paste approach.
3. Clean into a CSV with at least these columns:
   `player_name, team, season, salary`
4. Save as `data/raw/salaries_{season}.csv` (e.g., `salaries_2024-25.csv`).

### Option B: Spotrac (full contract detail)

**URL:** https://www.spotrac.com/nba/rankings/

**Steps:**
1. Navigate to Contracts -> Player Rankings for the current year.
2. Export or copy the table. Spotrac has per-player contract pages too
   (e.g., https://www.spotrac.com/nba/los-angeles-lakers/luka-doncic/) for detail.
3. For the full roster build, the team cap pages at
   `https://www.spotrac.com/nba/{team-slug}/cap/{year}/` are most complete.
4. Save as `data/raw/salaries_{season}.csv`.

**Required columns:**
```
player_name, team, season, salary, years_remaining, guaranteed
```
**Optional:**
```
option_type, cap_hit, two_way
```

**Load:**
```python
from src.data.salaries import load_salaries
df = load_salaries()                  # defaults to data/raw/salaries_2024-25.csv
df = load_salaries(season="2023-24")  # loads data/raw/salaries_2023-24.csv
```

To generate an empty template CSV with the correct column headers:
```python
from src.data.salaries import generate_salary_template
template = generate_salary_template()
template.to_csv("data/raw/salaries_2024-25.csv", index=False)
# Fill in the CSV, then load with load_salaries()
```

---

## File Naming Convention

| Source         | Filename pattern                     | Example                      |
|----------------|--------------------------------------|------------------------------|
| BBRef Advanced | `bbref_advanced_{end_year}.csv`      | `bbref_advanced_2025.csv`    |
| EPM            | `epm_{season}.csv` or `.json`        | `epm_2024-25.csv`            |
| Salaries       | `salaries_{season}.csv`              | `salaries_2024-25.csv`       |

`{end_year}` = 4-digit year the season ends (e.g., 2025 for 2024-25).
`{season}` = nba_api format string (e.g., `2024-25`).
