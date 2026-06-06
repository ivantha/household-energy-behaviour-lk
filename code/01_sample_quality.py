"""01 -- Sample composition, representativeness & data quality.

Theme 1 of the exploratory EDA. This script is descriptive by design: it
quantifies WHO is in the convenience sample (and therefore who the findings can
and cannot generalise to), maps geographic coverage onto Sri Lanka's nine
provinces, and audits data quality (missingness + the impossible-zero values
that 00_clean.py recoded to NaN).

For the headline demographic skews we attach Wilson score 95% confidence
intervals for a single proportion (the small-n appropriate interval; it does not
collapse to a point even near 0/1 and never leaves [0,1]). These CIs describe
sampling uncertainty for *this* recruitment process; they are not a license to
generalise beyond the convenience frame.

Outputs:
    outputs/figures/01_*.{png,pdf}
    outputs/tables/01_*.{csv,tex}
    stdout: one compact JSON object of key findings (last line).

Run:  uv run python 01_sample_quality.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as ss

from lib import io, config as C, stats as S, plotting as P


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def wilson_ci(k: int, n: int, conf: float = 0.95):
    """Wilson score interval for a binomial proportion (small-n appropriate)."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    z = ss.norm.ppf(1 - (1 - conf) / 2)
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return float(phat), float(max(0.0, centre - half)), float(min(1.0, centre + half))


def shannon_evenness(counts) -> float:
    """Pielou's evenness J = H/ln(S) in [0,1]; 1 == perfectly uniform categories."""
    c = np.asarray([v for v in counts if v > 0], float)
    if len(c) < 2:
        return 0.0
    p = c / c.sum()
    H = -(p * np.log(p)).sum()
    return float(H / np.log(len(c)))


# All nine provinces of Sri Lanka (the sampling-frame denominator for coverage).
ALL_PROVINCES = [
    "Western", "Central", "Southern", "Northern", "Eastern",
    "North Western", "North Central", "Uva", "Sabaragamuwa",
]
# All 25 administrative districts (denominator for district coverage).
ALL_DISTRICTS = sorted(set(C.DISTRICT_PROVINCE))


def main():
    df = io.load_clean()
    N = len(df)
    rng_seed = 0

    # ===================================================================== #
    # 1. Sample-composition frequencies (the five demographic / meta vars)
    # ===================================================================== #
    comp_vars = ["src_mode", "gender", "edu_attendance", "occupation", "rel_head"]
    comp_rows = []
    for var in comp_vars:
        vc = df[var].value_counts(dropna=False)
        n_missing = int(df[var].isna().sum())
        for cat, n in vc.items():
            cat_label = "(missing)" if (isinstance(cat, float) and pd.isna(cat)) else str(cat)
            comp_rows.append({
                "variable": C.LABEL.get(var, var),
                "code": var,
                "category": cat_label,
                "n": int(n),
                "pct": round(100.0 * n / N, 1),
            })
    comp_tbl = pd.DataFrame(comp_rows)

    # ===================================================================== #
    # 2. Headline sampling skews + Wilson 95% CIs
    # ===================================================================== #
    skews = {
        "male": (int((df.gender == "Male").sum()), "gender == Male"),
        "university": (int((df.edu_attendance == "University").sum()),
                       "edu_attendance == University"),
        "professional": (int((df.occupation == "Professional").sum()),
                         "occupation == Professional"),
        "son_daughter": (int((df.rel_head == "Son/daughter").sum()),
                         "rel_head == Son/daughter"),
        "online_mode": (int((df.src_mode == "Online").sum()), "src_mode == Online"),
        "western_province": (int((df.province == "Western").sum()),
                             "province == Western"),
    }
    skew_rows = []
    for key, (k, defn) in skews.items():
        phat, lo, hi = wilson_ci(k, N)
        skew_rows.append({
            "indicator": key, "definition": defn, "k": k, "n": N,
            "pct": round(100 * phat, 1),
            "wilson_lo_pct": round(100 * lo, 1),
            "wilson_hi_pct": round(100 * hi, 1),
        })
    skew_tbl = pd.DataFrame(skew_rows)

    # ===================================================================== #
    # 3. Geographic coverage (districts + provinces)
    # ===================================================================== #
    dist_counts = df["district"].value_counts()
    prov_counts = df["province"].value_counts()
    districts_present = sorted(dist_counts.index.tolist())
    districts_missing = [d for d in ALL_DISTRICTS if d not in set(districts_present)]
    provinces_present = sorted(prov_counts.index.tolist())
    provinces_missing = [p for p in ALL_PROVINCES if p not in set(provinces_present)]

    # province coverage table over ALL nine provinces (0 where uncovered)
    prov_rows = []
    for prov in ALL_PROVINCES:
        n = int(prov_counts.get(prov, 0))
        prov_rows.append({
            "province": prov,
            "n": n,
            "pct": round(100.0 * n / N, 1),
            "covered": "yes" if n > 0 else "no",
        })
    prov_tbl = pd.DataFrame(prov_rows).sort_values("n", ascending=False).reset_index(drop=True)

    # district coverage table over ALL 25 districts (0 where uncovered)
    dist_rows = []
    for dist in ALL_DISTRICTS:
        n = int(dist_counts.get(dist, 0))
        dist_rows.append({
            "district": dist,
            "province": C.DISTRICT_PROVINCE[dist],
            "n": n,
            "pct": round(100.0 * n / N, 1),
            "covered": "yes" if n > 0 else "no",
        })
    dist_tbl = pd.DataFrame(dist_rows).sort_values(
        ["n", "district"], ascending=[False, True]).reset_index(drop=True)

    prov_evenness = shannon_evenness(prov_counts.values)

    # ===================================================================== #
    # 4. Data quality: missingness + impossible-zeros recoded to NaN
    # ===================================================================== #
    # original survey variables (codes 0..26) -- exclude derived columns
    orig_codes = [C.COLCODE[i] for i in sorted(C.COLCODE)]
    miss_rows = []
    for c in orig_codes:
        n_miss = int(df[c].isna().sum())
        miss_rows.append({
            "code": c,
            "label": C.LABEL.get(c, ""),
            "n_missing": n_miss,
            "pct_missing": round(100.0 * n_miss / N, 1),
        })
    miss_tbl = pd.DataFrame(miss_rows).sort_values(
        ["n_missing", "code"], ascending=[False, True]).reset_index(drop=True)
    miss_any = miss_tbl[miss_tbl.n_missing > 0].reset_index(drop=True)

    # impossible zeros recoded to NaN by 00_clean.py (re-derive from raw)
    raw = io.load_raw(rename=True)
    invalid_zero = {}
    for col in ["area_sqft", "food_exp", "nonfood_exp", "stories"]:
        invalid_zero[col] = int((pd.to_numeric(raw[col], errors="coerce") == 0).sum())
    total_invalid_zero = int(sum(invalid_zero.values()))

    vent_missing = int(df["vent_raw"].isna().sum())  # the 16 missing ventilation responses
    # overall cell completeness across the 26 original survey items
    total_cells = N * len(orig_codes)
    total_missing_cells = int(sum(r["n_missing"] for r in miss_rows))
    completeness = round(100.0 * (1 - total_missing_cells / total_cells), 2)

    # =========================================================================
    # FIGURES
    # =========================================================================
    figpaths = {}

    # --- Fig 01_mode: data-collection mode --------------------------------- #
    fig, ax = plt.subplots(figsize=P.figsize("FULL", 2.6))
    mode_vc = df["src_mode"].value_counts()
    mode_lbl = {"Online": "Online", "Phone": "Phone interview", "F2F": "Face-to-face"}
    order = mode_vc.sort_values().index
    ax.barh([mode_lbl.get(m, m) for m in order], mode_vc[order].values, color=P.ACCENT)
    for i, m in enumerate(order):
        v = int(mode_vc[m])
        ax.text(v + 0.4, i, f"{v}  ({100*v/N:.0f}%)", va="center", fontsize=P.ANNOT)
    ax.set_xlabel("Respondents")
    ax.set_xlim(0, mode_vc.max() * 1.25)
    figpaths["mode"] = P.save_fig(fig, "01_mode")

    # --- Fig 01_province_coverage: 9 provinces, covered vs not ------------- #
    fig, ax = plt.subplots(figsize=P.figsize("FULL", 3.4))
    pv = prov_tbl.sort_values("n")
    colors = [P.ACCENT if c == "yes" else P.MUTED for c in pv["covered"]]
    ax.barh(pv["province"], pv["n"], color=colors)
    for i, (_, r) in enumerate(pv.iterrows()):
        if r["n"] > 0:
            ax.text(r["n"] + 0.5, i, f"{int(r['n'])} ({r['pct']:.0f}%)",
                    va="center", fontsize=P.ANNOT)
        else:
            ax.text(0.3, i, "0  (not covered)", va="center", fontsize=P.ANNOT,
                    color=P.RULE, style="italic")
    ax.set_xlabel("Respondents")
    ax.set_xlim(0, prov_counts.max() * 1.28)
    figpaths["province_coverage"] = P.save_fig(fig, "01_province_coverage")

    # --- Fig 01_skew_panel: demographic-skew small multiples --------------- #
    panel_specs = [
        ("gender", "Gender", "Male"),
        ("edu_attendance", "Educational attendance", "University"),
        ("occupation", "Main occupation", "Professional"),
        ("rel_head", "Relationship to head", "Son/daughter"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=P.figsize("FULL", 4.4))
    for ax, (var, title, dom) in zip(axes.ravel(), panel_specs):
        vc = df[var].value_counts().sort_values()
        # wrap long category labels
        labels = [(s[:26] + "…") if len(str(s)) > 27 else str(s) for s in vc.index]
        bar_colors = [P.ACCENT2 if cat == dom else P.ACCENT for cat in vc.index]
        ax.barh(labels, vc.values, color=bar_colors)
        for i, v in enumerate(vc.values):
            ax.text(v + N * 0.012, i, f"{int(v)} ({100*v/N:.0f}%)",
                    va="center", fontsize=P.ANNOT_SMALL)
        ax.set_xlim(0, vc.max() * 1.30)
        ax.set_title(title)
        ax.set_xlabel("n")
    fig.tight_layout()
    figpaths["skew_panel"] = P.save_fig(fig, "01_skew_panel")

    # --- Fig 01_missingness: per-variable missingness --------------------- #
    fig, ax = plt.subplots(figsize=P.figsize("FULL", 3.0))
    mm = miss_any.sort_values("n_missing")
    # human labels for codes not in the global LABEL map (e.g. the raw
    # multi-select ventilation column, which would otherwise leak "vent_raw")
    fig_labels = {"vent_raw": "Ventilation features"}
    labels = [fig_labels.get(c) or C.LABEL.get(c, c) or c for c in mm["code"]]
    ax.barh(labels, mm["n_missing"].values, color=P.ACCENT)
    for i, (_, r) in enumerate(mm.iterrows()):
        ax.text(r["n_missing"] + 0.15, i, f"{int(r['n_missing'])} ({r['pct_missing']:.1f}%)",
                va="center", fontsize=P.ANNOT)
    ax.set_xlabel("Missing responses")
    ax.set_xlim(0, mm["n_missing"].max() * 1.35)
    figpaths["missingness"] = P.save_fig(fig, "01_missingness")

    # =========================================================================
    # TABLES (csv + LaTeX)
    # =========================================================================
    tblpaths = {}

    def save_table(dfobj, name, caption, label):
        csv_p = C.TBL_DIR / f"{name}.csv"
        tex_p = C.TBL_DIR / f"{name}.tex"
        dfobj.to_csv(csv_p, index=False)
        # render pct-like float columns to 1 decimal so the LaTeX is paper-clean
        fmt_cols = {c: "{:.1f}".format for c in dfobj.columns
                    if dfobj[c].dtype.kind == "f"}
        dfobj.to_latex(tex_p, index=False, escape=True, caption=caption,
                       label=label, longtable=False, formatters=fmt_cols)
        tblpaths[name] = {"csv": str(csv_p), "tex": str(tex_p)}

    save_table(comp_tbl, "01_sample_composition",
               "Sample composition: frequency and percentage of each category for "
               "the five demographic and meta variables (N=70).",
               "tab:sample_composition")
    save_table(skew_tbl, "01_sampling_skews",
               "Headline sampling skews with Wilson score 95\\% confidence "
               "intervals (N=70).", "tab:sampling_skews")
    save_table(prov_tbl, "01_province_coverage",
               "Provincial coverage over all nine provinces of Sri Lanka (N=70).",
               "tab:province_coverage")
    save_table(dist_tbl, "01_district_coverage",
               "District coverage over all 25 administrative districts (N=70).",
               "tab:district_coverage")
    save_table(miss_tbl, "01_missingness",
               "Per-variable missingness across the 26 original survey items (N=70).",
               "tab:missingness")

    # =========================================================================
    # KEY FINDINGS (compact JSON -- MUST be last stdout line)
    # =========================================================================
    def fmt(key):
        r = skew_tbl.set_index("indicator").loc[key]
        return {"k": int(r.k), "pct": float(r.pct),
                "ci": [float(r.wilson_lo_pct), float(r.wilson_hi_pct)]}

    findings = {
        "N": N,
        "n_columns": int(df.shape[1]),
        "src_mode": {k: int(v) for k, v in df.src_mode.value_counts().items()},
        "skews": {
            "male": fmt("male"),
            "university": fmt("university"),
            "professional": fmt("professional"),
            "son_daughter": fmt("son_daughter"),
            "online_mode": fmt("online_mode"),
            "western_province": fmt("western_province"),
        },
        "geography": {
            "n_districts_present": len(districts_present),
            "n_districts_total": len(ALL_DISTRICTS),
            "districts_present": districts_present,
            "n_provinces_present": len(provinces_present),
            "n_provinces_total": len(ALL_PROVINCES),
            "provinces_present": provinces_present,
            "provinces_missing": provinces_missing,
            "province_top": {"name": prov_tbl.iloc[0]["province"],
                             "n": int(prov_tbl.iloc[0]["n"]),
                             "pct": float(prov_tbl.iloc[0]["pct"])},
            "province_evenness_pielou": round(prov_evenness, 3),
        },
        "data_quality": {
            "invalid_zeros_recoded": invalid_zero,
            "total_invalid_zeros": total_invalid_zero,
            "vent_missing": vent_missing,
            "missing_by_var": {r["code"]: r["n_missing"]
                               for r in miss_rows if r["n_missing"] > 0},
            "cell_completeness_pct": completeness,
            "n_complete_vars": int((miss_tbl.n_missing == 0).sum()),
            "n_original_vars": len(orig_codes),
        },
        "figures": {k: v for k, v in figpaths.items()},
        "tables": {k: v["csv"] for k, v in tblpaths.items()},
    }
    print(json.dumps(findings, default=str))


if __name__ == "__main__":
    main()
