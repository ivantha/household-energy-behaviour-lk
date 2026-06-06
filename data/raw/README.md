# `data/raw/` — survey source data

## `survey.csv`

The de-identified source dataset for this study: the **N = 70** analysis sample,
one row per household, with the **original survey-question text as column headers**
(self-documenting). The pipeline (`code/lib/io.load_raw`) reads this file and maps
the columns to short codes; `code/00_clean.py` then produces the cleaned table and
the data dictionary at `data/processed/codebook.csv`.

| | |
| ---- | ---- |
| Rows | 70 households |
| Columns | 26 survey items |
| Encoding | UTF-8 |

## Provenance

The 70 responses were collected via the survey's online and in-person channels in
**August 2023**. The data was originally captured in two exports with
byte-identical headers (a 69-row export plus one genuine later response, the 70th
household); these have been concatenated here into a single file.

## De-identification

No directly identifying information (names, contact numbers, addresses, GPS
coordinates, or age) was ever collected. To further reduce re-identification risk
in such a small sample, **one column from the original instrument has been removed
from this public release:**

- **Divisional secretariat** ("Which Secretarial division your house belonged?")
  — the finest-grained geographic field (hundreds of divisions nationally). It is
  **never used** in any analysis script, so its removal does not change any
  published figure, table, or statistic.

The coarser **district** and (derived) **province** fields are retained, because the
representativeness analysis in the paper reports geographic coverage at those
levels.

This dataset is licensed under CC BY 4.0; see `../LICENSE`.
