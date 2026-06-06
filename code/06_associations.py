"""06 -- Bivariate association map (FDR-controlled).

THEME 6. Two complementary association structures over the N=70 convenience
sample, treated as STRICTLY EXPLORATORY (effect-size-led, FDR-controlled):

  (A) Cramer's V matrix (Bergsma-Wicher bias-corrected) across the
      categorical / ordinal variables -- dwelling, demographic, energy-system,
      context (province) and the FIVE independent energy-conserving behaviours.
  (B) Spearman rho matrix among the numerics -- area_sqft, stories, food_exp,
      nonfood_exp, total_exp, red_notices, the five behaviour item scores, eci.

Every categorical pair is re-tested with S.assoc_categorical (Fisher 2x2 /
chi-square) and Benjamini-Hochberg FDR is applied ACROSS the FULL family. The
numeric pairs get Spearman p-values + bootstrap CIs and a SEPARATE full-family
BH-FDR pass. (FDR is deliberately NOT applied to a post-hoc top-K slice: ranking
pairs by effect size and correcting only the strongest is a selection-induced
multiplicity error that under-counts the family and deflates q-values.) We report
effect sizes and q-values throughout and state explicitly whether anything
survives FDR.

Outputs:
  figures: 06_cramersv_heatmap, 06_spearman_heatmap
  tables : 06_top_associations(.csv/.tex), 06_cramersv_matrix(.csv/.tex),
           06_spearman_matrix(.csv/.tex)
Run:  uv run python 06_associations.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lib import io, config as C, stats as S, plotting as P

SEED = 0
N_BOOT = 10000

# --------------------------------------------------------------------------- #
# Variable sets
# --------------------------------------------------------------------------- #
# Categorical / ordinal variables for the Cramer's V map. We deliberately keep
# the substantively meaningful blocks (dwelling, demographic, energy-system,
# context) plus the five behaviours, and exclude the data-collection mode (meta)
# and gen_none (a near-degenerate 90/10 split that duplicates uses_renewable).
CAT_VARS = [
    # dwelling
    "build_period", "arch_design", "house_type", "wall_material", "roof_type",
    # demographic
    "rel_head", "gender", "edu_attendance", "occupation",
    # energy-system
    "pay_practice", "wiring_pro", "any_red_notice", "uses_renewable", "solar_user",
    # context
    "province",
    # five independent energy-conserving behaviours
    "eci_billknow", "eci_iron", "eci_nightlight", "eci_metercheck", "eci_enrating",
]

# Numeric variables for the Spearman map.
NUM_VARS = [
    "area_sqft", "stories", "food_exp", "nonfood_exp", "total_exp", "red_notices",
    "eci_billknow", "eci_iron", "eci_nightlight", "eci_metercheck", "eci_enrating",
    "eci",
]

# Short axis labels (heatmaps get crowded otherwise).
SHORT = {
    "build_period": "Build period", "arch_design": "Arch. design",
    "house_type": "House type", "wall_material": "Wall material",
    "roof_type": "Roof type", "rel_head": "Rel. to head", "gender": "Gender",
    "edu_attendance": "Education", "occupation": "Occupation",
    "pay_practice": "Pay practice", "wiring_pro": "Pro wiring",
    "any_red_notice": "Any red notice", "uses_renewable": "Uses renewable",
    "solar_user": "Solar user", "province": "Province",
    "eci_billknow": "Bill literacy", "eci_iron": "Ironing eff.",
    "eci_nightlight": "Night-light", "eci_metercheck": "Meter check",
    "eci_enrating": "Energy rating",
    "area_sqft": "Floor area", "stories": "Stories", "food_exp": "Food exp.",
    "nonfood_exp": "Nonfood exp.", "total_exp": "Total exp.",
    "red_notices": "Red notices", "eci": "ECI",
}

BLOCK = {  # for the interpretation column
    "build_period": "dwelling", "arch_design": "dwelling", "house_type": "dwelling",
    "wall_material": "dwelling", "roof_type": "dwelling",
    "rel_head": "demographic", "gender": "demographic",
    "edu_attendance": "demographic", "occupation": "demographic",
    "pay_practice": "energy", "wiring_pro": "energy", "any_red_notice": "energy",
    "uses_renewable": "energy", "solar_user": "energy", "province": "context",
    "eci_billknow": "behaviour", "eci_iron": "behaviour",
    "eci_nightlight": "behaviour", "eci_metercheck": "behaviour",
    "eci_enrating": "behaviour",
}


def interp(v: float) -> str:
    """Cohen-style verbal label for Cramer's V / |rho| magnitude."""
    a = abs(v)
    if np.isnan(a):
        return "undefined"
    if a < 0.10:
        return "negligible"
    if a < 0.30:
        return "small"
    if a < 0.50:
        return "moderate"
    return "large"


def pair_block(a: str, b: str) -> str:
    ba, bb = BLOCK.get(a, "?"), BLOCK.get(b, "?")
    return f"{ba}~{bb}" if ba != bb else ba


# Structurally dependent (tautological / part-whole) numeric pairs that MUST NOT
# be read as substantive findings: total_exp = food_exp + nonfood_exp, and
# eci = mean of the five normalised behaviour items (so each item is part of eci).
_BEH = {"eci_billknow", "eci_iron", "eci_nightlight", "eci_metercheck", "eci_enrating"}


def is_tautological(a: str, b: str) -> bool:
    s = {a, b}
    if "total_exp" in s and ({"food_exp", "nonfood_exp"} & s):
        return True
    if "eci" in s and (s & _BEH):
        return True
    return False


# --------------------------------------------------------------------------- #
def main():
    df = io.load_clean()
    n = len(df)

    # ===================================================================== #
    # (A) Cramer's V matrix (bias-corrected) over categorical/ordinal vars
    # ===================================================================== #
    k = len(CAT_VARS)
    V = pd.DataFrame(np.nan, index=CAT_VARS, columns=CAT_VARS, dtype=float)
    for i in range(k):
        V.iloc[i, i] = 1.0
        for j in range(i + 1, k):
            v = S.cramers_v(df[CAT_VARS[i]], df[CAT_VARS[j]])
            V.iloc[i, j] = v
            V.iloc[j, i] = v
    V_lab = V.rename(index=SHORT, columns=SHORT)

    # ----- heatmap (lower triangle) -------------------------------------- #
    # ~20x20 dense grid: drop per-cell numbers (rely on the colorbar; survivors
    # are tabulated in the paper). Sequential 0..~0.6 field -> P.SEQ_CMAP.
    mask = np.triu(np.ones_like(V_lab, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=P.figsize("FULL", 7.8))
    sns.heatmap(V_lab, mask=mask, cmap=P.SEQ_CMAP, vmin=0, vmax=0.6,
                square=True, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Cramer's V (bias-corrected)", "shrink": 0.7},
                annot=False, ax=ax)
    # Sanctioned dense-grid exception: 20 long category labels need an explicit
    # small tick size to fit; this is the only figure that keeps a labelsize.
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelrotation=0, labelsize=7)
    cramers_paths = P.save_fig(fig, "06_cramersv_heatmap")

    # ===================================================================== #
    # (B) Spearman matrix over numerics
    # ===================================================================== #
    R = df[NUM_VARS].corr(method="spearman")
    R_lab = R.rename(index=SHORT, columns=SHORT)
    mask2 = np.triu(np.ones_like(R_lab, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=P.figsize("FULL", 7.2))
    sns.heatmap(R_lab, mask=mask2, cmap=P.DIVERGING, vmin=-1, vmax=1, center=0,
                square=True, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Spearman rho", "shrink": 0.7},
                annot=True, fmt=".2f", annot_kws={"size": P.ANNOT_SMALL}, ax=ax)
    ax.tick_params(axis="x", labelrotation=90)
    ax.tick_params(axis="y", labelrotation=0)
    spearman_paths = P.save_fig(fig, "06_spearman_heatmap")

    # ===================================================================== #
    # (A2) Re-test strongest categorical pairs + BH-FDR across the family
    # ===================================================================== #
    cat_pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = CAT_VARS[i], CAT_VARS[j]
            cat_pairs.append((a, b, V.iloc[i, j]))
    # rank by |V| (NaN last) for reporting order only
    cat_pairs.sort(key=lambda t: (-1 if np.isnan(t[2]) else t[2]), reverse=True)
    # BH-FDR is applied over the FULL family of categorical pairs, not a post-hoc
    # top-K slice: selecting the strongest pairs before correction under-counts
    # the family and deflates q-values (a selection-induced multiplicity error).
    cat_family = cat_pairs

    cat_rows = []
    for a, b, v in cat_family:
        res = S.assoc_categorical(df[a], df[b])
        cat_rows.append({
            "var_a": a, "var_b": b, "block": pair_block(a, b),
            "test": res["test"], "n": res["n"],
            "rows": res["rows"], "cols": res["cols"],
            "min_expected": res["min_expected"],
            "effect": "cramers_v", "effect_size": res["cramers_v"],
            "p": res["p"],
        })
    cat_tbl = pd.DataFrame(cat_rows)
    cat_tbl["q"] = S.bh_fdr(cat_tbl["p"].values)
    cat_tbl["interpretation"] = cat_tbl["effect_size"].map(interp)
    cat_tbl["family"] = "categorical"
    cat_tbl["relation"] = "substantive"  # no part-whole pairs among categoricals
    cat_tbl = cat_tbl.sort_values("p").reset_index(drop=True)

    # ===================================================================== #
    # (B2) Strongest numeric pairs: Spearman + bootstrap CI + separate BH-FDR
    # ===================================================================== #
    num_pairs = []
    for i in range(len(NUM_VARS)):
        for j in range(i + 1, len(NUM_VARS)):
            a, b = NUM_VARS[i], NUM_VARS[j]
            num_pairs.append((a, b, R.iloc[i, j]))
    num_pairs.sort(key=lambda t: (-1 if np.isnan(t[2]) else abs(t[2])), reverse=True)
    # Full numeric family for BH-FDR (every pair in the Spearman map), not a
    # post-hoc top-K slice -- see the categorical note above.
    num_family = num_pairs

    num_rows = []
    for a, b, _ in num_family:
        sp = S.spearman_with_ci(df[a].values, df[b].values, n_boot=N_BOOT, seed=SEED)
        num_rows.append({
            "var_a": a, "var_b": b, "block": pair_block(a, b),
            "test": "spearman", "n": sp["n"], "rows": np.nan, "cols": np.nan,
            "min_expected": np.nan,
            "effect": "spearman_rho", "effect_size": sp["rho"],
            "ci_lo": sp["lo"], "ci_hi": sp["hi"], "p": sp["p"],
            "relation": "tautological" if is_tautological(a, b) else "substantive",
        })
    num_tbl = pd.DataFrame(num_rows)
    num_tbl["q"] = S.bh_fdr(num_tbl["p"].values)
    num_tbl["interpretation"] = num_tbl["effect_size"].map(interp)
    num_tbl["family"] = "numeric"
    num_tbl = num_tbl.sort_values("p").reset_index(drop=True)

    # ===================================================================== #
    # Combined "top associations" table (both families)
    # ===================================================================== #
    cols = ["family", "var_a", "var_b", "block", "relation", "test", "effect",
            "effect_size", "ci_lo", "ci_hi", "p", "q", "n", "rows", "cols",
            "min_expected", "interpretation"]
    for c in ["ci_lo", "ci_hi"]:
        if c not in cat_tbl:
            cat_tbl[c] = np.nan
    top = pd.concat([cat_tbl[cols], num_tbl[cols]], ignore_index=True)

    # pretty labels for the LaTeX/CSV table
    def lab(c):
        return C.LABEL.get(c, SHORT.get(c, c))
    top_disp = top.copy()
    top_disp["var_a"] = top_disp["var_a"].map(lab)
    top_disp["var_b"] = top_disp["var_b"].map(lab)
    for c in ["effect_size", "ci_lo", "ci_hi", "p", "q", "min_expected"]:
        top_disp[c] = top_disp[c].astype(float).round(4)

    # ----- write tables (csv + latex) ------------------------------------ #
    def write_table(frame, name, caption, label):
        csv_p = C.TBL_DIR / f"{name}.csv"
        tex_p = C.TBL_DIR / f"{name}.tex"
        frame.to_csv(csv_p, index=False)
        frame.to_latex(tex_p, index=False, escape=True, longtable=False,
                       caption=caption, label=label, na_rep="--",
                       float_format="%.3f")
        return str(csv_p), str(tex_p)

    top_csv, top_tex = write_table(
        top_disp, "06_top_associations",
        "Strongest bivariate associations across two test families "
        "(categorical: Fisher/chi-square + bias-corrected Cramer's V; numeric: "
        "Spearman with bootstrap 95\\% CI). q = Benjamini-Hochberg FDR-adjusted "
        "p within each family. N=70, exploratory.",
        "tab:top_assoc")

    Vmat_csv, Vmat_tex = write_table(
        V_lab.round(3).reset_index().rename(columns={"index": "variable"}),
        "06_cramersv_matrix",
        "Bias-corrected Cramer's V matrix across categorical/ordinal variables (N=70).",
        "tab:cramersv")

    Rmat_csv, Rmat_tex = write_table(
        R_lab.round(3).reset_index().rename(columns={"index": "variable"}),
        "06_spearman_matrix",
        "Spearman rho matrix across numeric variables (N=70).",
        "tab:spearman")

    # ===================================================================== #
    # Survival accounting + headline numbers
    # ===================================================================== #
    alpha = 0.05
    cat_sig_raw = int((cat_tbl["p"] < alpha).sum())
    cat_sig_fdr = int((cat_tbl["q"] < alpha).sum())
    num_sig_raw = int((num_tbl["p"] < alpha).sum())
    num_sig_fdr = int((num_tbl["q"] < alpha).sum())
    # honest counts that exclude part-whole tautologies (total_exp=food+nonfood,
    # eci=mean of behaviour items) -- these are the substantive survivors.
    num_subst = num_tbl[num_tbl["relation"] == "substantive"]
    num_taut = num_tbl[num_tbl["relation"] == "tautological"]
    num_subst_sig_fdr = int((num_subst["q"] < alpha).sum())
    num_taut_in_family = int(len(num_taut))
    # strongest SUBSTANTIVE numeric pair (drop tautologies before picking)
    num_subst_sorted = num_subst.sort_values("p").reset_index(drop=True)

    def top_row(frame):
        r = frame.iloc[0]
        return {
            "pair": f'{r["var_a"]} x {r["var_b"]}',
            "test": r["test"], "effect": r["effect"],
            "effect_size": round(float(r["effect_size"]), 3),
            "p": round(float(r["p"]), 4), "q": round(float(r["q"]), 4),
            "n": int(r["n"]),
        }

    # behaviour-vs-behaviour strongest (the "do the 5 behaviours cohere?" check)
    beh = ["eci_billknow", "eci_iron", "eci_nightlight", "eci_metercheck", "eci_enrating"]
    beh_v = []
    for i in range(len(beh)):
        for j in range(i + 1, len(beh)):
            beh_v.append((beh[i], beh[j], V.loc[beh[i], beh[j]]))
    beh_v.sort(key=lambda t: (-1 if np.isnan(t[2]) else t[2]), reverse=True)
    max_beh_v = float(beh_v[0][2])
    # also the max |rho| among the 5 behaviour scores from the Spearman matrix
    beh_rho = []
    for i in range(len(beh)):
        for j in range(i + 1, len(beh)):
            beh_rho.append(abs(R.loc[beh[i], beh[j]]))
    max_beh_rho = float(np.nanmax(beh_rho))

    # strongest behaviour-vs-NONbehaviour categorical link (any external driver?)
    beh_set = set(beh)
    cross = [(a, b, v) for (a, b, v) in cat_pairs
             if (a in beh_set) ^ (b in beh_set) and not np.isnan(v)]
    cross.sort(key=lambda t: t[2], reverse=True)
    top_cross = cross[0] if cross else (None, None, np.nan)

    findings = {
        "n": n,
        "n_cat_vars": k,
        "n_num_vars": len(NUM_VARS),
        "cat_family_size": len(cat_family),
        "num_family_size": len(num_family),
        "cramersv_max_offdiag": round(float(np.nanmax(V.values[~np.eye(k, dtype=bool)])), 3),
        "cramersv_median_offdiag": round(float(np.nanmedian(V.values[~np.eye(k, dtype=bool)])), 3),
        "cramersv_n_nan_pairs": int(np.isnan(V.values[~np.eye(k, dtype=bool)]).sum()) // 2,
        "spearman_max_abs_offdiag": round(float(np.nanmax(np.abs(R.values[~np.eye(len(NUM_VARS), dtype=bool)]))), 3),
        "cat_strongest_pair": top_row(cat_tbl),
        "num_strongest_pair": top_row(num_tbl),
        "num_strongest_substantive_pair": top_row(num_subst_sorted),
        "cat_sig_raw_p05": cat_sig_raw,
        "cat_sig_fdr_q05": cat_sig_fdr,
        "num_sig_raw_p05": num_sig_raw,
        "num_sig_fdr_q05": num_sig_fdr,
        "num_tautological_in_family": num_taut_in_family,
        "num_substantive_sig_fdr_q05": num_subst_sig_fdr,
        "any_survives_fdr": bool((cat_sig_fdr + num_sig_fdr) > 0),
        "any_substantive_survives_fdr": bool((cat_sig_fdr + num_subst_sig_fdr) > 0),
        "max_behaviour_behaviour_cramersv": round(max_beh_v, 3),
        "max_behaviour_behaviour_abs_rho": round(max_beh_rho, 3),
        "strongest_behaviour_external_pair": {
            "pair": f"{SHORT.get(top_cross[0], top_cross[0])} x {SHORT.get(top_cross[1], top_cross[1])}",
            "cramers_v": round(float(top_cross[2]), 3),
        },
        "figures": {"cramersv_heatmap": cramers_paths, "spearman_heatmap": spearman_paths},
        "tables": {
            "top_associations": [top_csv, top_tex],
            "cramersv_matrix": [Vmat_csv, Vmat_tex],
            "spearman_matrix": [Rmat_csv, Rmat_tex],
        },
    }

    print("\n=== Categorical family (top {}), sorted by p ===".format(len(cat_family)))
    print(cat_tbl[["var_a", "var_b", "test", "effect_size", "p", "q",
                   "n", "min_expected", "interpretation"]].to_string(index=False))
    print("\n=== Numeric family (top {}), sorted by p ===".format(len(num_family)))
    print(num_tbl[["var_a", "var_b", "relation", "effect_size", "ci_lo", "ci_hi",
                   "p", "q", "n", "interpretation"]].to_string(index=False))
    print("\nFILES:")
    for p in (cramers_paths + spearman_paths + [top_csv, top_tex, Vmat_csv,
              Vmat_tex, Rmat_csv, Rmat_tex]):
        print("  ", p)
    print(json.dumps(findings, default=str))


if __name__ == "__main__":
    main()
