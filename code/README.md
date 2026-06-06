# Analysis code

Reproducible pipeline for the LK household energy-behaviour survey EDA.

## Run

```sh
uv sync                    # Python 3.12 env (see pyproject.toml)
uv run python run_all.py   # runs 00_clean.py then 01..09 in order
```

Artifacts: `../data/processed/` (cleaned data, codebook, ECI scores) and
`outputs/{figures,tables}/` (PNG+PDF figures, CSV+LaTeX tables).

## Scripts

| Script | Produces |
| ------ | -------- |
| `00_clean.py` | `cleaned.csv`, `codebook.csv`, `eci_scores.csv`, `clean_summary.json` |
| `01_sample_quality.py` | Sample composition, representativeness, missingness |
| `02_dwelling.py` | Housing fabric & dwelling characteristics |
| `03_demographics_ses.py` | Expenditure / socioeconomic profile |
| `04_energy_systems.py` | Generation, solar, wiring, billing |
| `05_behaviours.py` | The five behaviours + ECI reliability / multidimensionality |
| `06_associations.py` | Cramér's V & Spearman association maps (FDR-controlled) |
| `07_typologies.py` | MCA/FAMD + clustering household typologies |
| `08_inferential.py` | Undirected screen: 17 predictors × 5 behaviours (effect sizes, bootstrap CIs, BH-FDR) |
| `09_targeted.py` | Directed follow-up: two theory-motivated correlates |

## Library (`lib/`)

- `config.py` — paths, raw→code column map, variable roles, **the ECI scoring rubric**,
  multi-select parsing specs, district→province map.
- `io.py` — `load_raw()`, `load_clean()`, `load_eci()`.
- `stats.py` — `cramers_v`, `assoc_categorical`, `kruskal_effect`, `cronbach_alpha`,
  `bootstrap_ci`, `bootstrap_ci_pair`, `spearman_with_ci`, `bh_fdr`.
- `plotting.py` — publication style + `save_fig`, `hbar_counts`.

All scripts import `lib`; edit shared behaviour there, not in each script.
