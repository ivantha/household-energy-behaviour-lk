"""00 -- Clean the raw survey export and build the Energy-Consciousness Index.

Outputs (data/processed/):
    cleaned.csv        master analysis table (short codes, derived features, ECI)
    eci_scores.csv     per-respondent ECI item scores + composite + band
    codebook.csv       variable dictionary (code, role, label, original question)
    clean_summary.json machine-readable cleaning report (alpha, missingness, ...)

Run:  uv run python 00_clean.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from lib import config as C
from lib import stats as S
from lib import io as IO


def expand_multiselect(df, colcode, spec, sentinel=None, count_col=None):
    """Split a comma-separated multi-select column into 0/1 indicator columns.

    Returns (created_columns, unmatched_tokens). Unmatched tokens are surfaced so
    a future response option cannot be silently dropped.
    """
    created = sorted(set(spec.values()))
    for ind in created:
        df[ind] = 0
    unmatched = set()
    for idx, val in df[colcode].items():
        if pd.isna(val):
            continue  # treated as 'none reported' (all indicators stay 0)
        for tok in C.split_tokens(val):
            t = C._norm(tok)
            if not t or (sentinel and sentinel in t):
                continue
            matched = False
            for kw, ind in spec.items():
                if kw in t:
                    df.at[idx, ind] = 1
                    matched = True
            if not matched:
                unmatched.add(tok.strip())
    if count_col:
        df[count_col] = df[created].sum(axis=1)
    return created, unmatched


def main():
    raw = IO.load_raw(rename=False)  # original export + later responses, concatenated
    raw_cols = list(raw.columns)
    assert len(raw) == C.N_EXPECTED, f"expected {C.N_EXPECTED} rows, got {len(raw)}"

    df = raw.copy()
    df.columns = [C.COLCODE.get(i, c) for i, c in enumerate(raw_cols)]

    # --- numeric cleaning: impossible zeros -> NaN -------------------------- #
    for col in ["area_sqft", "stories", "food_exp", "nonfood_exp", "red_notices"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    invalid_zero = {}
    for col in ["area_sqft", "food_exp", "nonfood_exp", "stories"]:
        invalid_zero[col] = int((df[col] == 0).sum())
        df.loc[df[col] == 0, col] = np.nan  # a dwelling cannot have 0 area/stories/spend

    # --- derived numerics --------------------------------------------------- #
    df["total_exp"] = df["food_exp"] + df["nonfood_exp"]
    df["area_per_story"] = df["area_sqft"] / df["stories"]
    df["any_red_notice"] = (df["red_notices"] > 0).astype(int)
    df["province"] = df["district"].map(C.DISTRICT_PROVINCE)
    missing_prov = sorted(set(df.loc[df["province"].isna(), "district"].dropna()))

    # --- multi-select expansion -------------------------------------------- #
    vent_cols, vent_un = expand_multiselect(df, "vent_raw", C.VENT_SPEC, count_col="vent_count")
    df["vent_reported"] = df["vent_raw"].notna().astype(int)
    gen_cols, gen_un = expand_multiselect(df, "gen_raw", C.GEN_SPEC,
                                          sentinel=C.GEN_SENTINEL, count_col="gen_count")
    df["gen_none"] = df["gen_raw"].apply(
        lambda v: int(C.GEN_SENTINEL in C._norm(v)) if pd.notna(v) else 0)
    df["uses_renewable"] = (
        (df["gen_solar"] + df["gen_hydro"] + df["gen_othergreen"]) > 0).astype(int)
    solar_cols, solar_un = expand_multiselect(df, "solar_raw", C.SOLAR_SPEC,
                                              sentinel=C.SOLAR_SENTINEL,
                                              count_col="solar_purpose_count")
    df["solar_user"] = df["solar_raw"].apply(
        lambda v: int(C.SOLAR_SENTINEL not in C._norm(v)) if pd.notna(v) else 0)
    for nm, un in [("vent", vent_un), ("gen", gen_un), ("solar", solar_un)]:
        assert not un, f"{nm}: unmatched multi-select tokens {un}"

    # --- ECI item scoring + composite -------------------------------------- #
    item_cols, norm_cols = [], []
    for item, (rawcode, scorer, mx) in C.ECI_ITEMS.items():
        df[item] = df[rawcode].apply(scorer)
        df[item + "_norm"] = df[item] / mx
        item_cols.append(item)
        norm_cols.append(item + "_norm")
    df["eci_raw_sum"] = df[item_cols].sum(axis=1)
    df["eci"] = df[norm_cols].mean(axis=1) * 100.0
    # tertile bands via ranks (robust to ties)
    df["eci_band"] = pd.qcut(df["eci"].rank(method="first"), 3,
                             labels=["Low", "Medium", "High"])

    # reliability
    alpha = S.cronbach_alpha(df[norm_cols])
    item_total = {}
    for c in norm_cols:
        rest = df[[x for x in norm_cols if x != c]].sum(axis=1)
        item_total[c] = float(np.corrcoef(df[c], rest)[0, 1])

    # --- write cleaned + ECI ----------------------------------------------- #
    df.to_csv(C.CLEAN_CSV, index=False)
    df[["eci", "eci_band", "eci_raw_sum"] + item_cols + norm_cols].to_csv(C.ECI_CSV, index=False)

    # --- codebook ----------------------------------------------------------- #
    rows = []
    for i, orig in enumerate(raw_cols):
        code = C.COLCODE.get(i, orig)
        rows.append({
            "index": i, "code": code, "role": C.ROLE.get(code, ""),
            "label": C.LABEL.get(code, ""), "dtype": str(df[code].dtype),
            "n_nonnull": int(df[code].notna().sum()),
            "n_unique": int(df[code].nunique()),
            "original_question": orig,
        })
    seen = {r["code"] for r in rows}
    derived = (["total_exp", "area_per_story", "any_red_notice", "province",
                "vent_count", "vent_reported", "gen_count", "gen_none",
                "uses_renewable", "solar_user", "solar_purpose_count",
                "eci", "eci_band", "eci_raw_sum"]
               + item_cols + norm_cols + vent_cols + gen_cols + solar_cols)
    for code in derived:
        if code in seen or code not in df:
            continue
        seen.add(code)
        rows.append({
            "index": "", "code": code, "role": C.ROLE.get(code, "derived"),
            "label": C.LABEL.get(code, ""), "dtype": str(df[code].dtype),
            "n_nonnull": int(df[code].notna().sum()),
            "n_unique": int(df[code].nunique()), "original_question": "[derived]",
        })
    pd.DataFrame(rows).to_csv(C.CODEBOOK_CSV, index=False)

    # --- summary ------------------------------------------------------------ #
    summary = {
        "n": int(len(df)),
        "n_columns_clean": int(df.shape[1]),
        "invalid_zeros_set_missing": invalid_zero,
        "districts_unmapped_to_province": missing_prov,
        "eci_cronbach_alpha": float(alpha),
        "eci_item_total_corr": item_total,
        "eci_describe": {k: float(v) for k, v in df["eci"].describe().items()},
        "eci_band_counts": {str(k): int(v) for k, v in df["eci_band"].value_counts().items()},
        "missing_by_col": {c: int(df[c].isna().sum())
                           for c in df.columns if df[c].isna().any()},
    }
    C.SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print("\nWROTE:")
    for p in (C.CLEAN_CSV, C.ECI_CSV, C.CODEBOOK_CSV, C.SUMMARY_JSON):
        print("  ", p)


if __name__ == "__main__":
    main()
