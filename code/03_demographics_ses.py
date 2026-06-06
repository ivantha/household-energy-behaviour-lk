"""03 -- Socioeconomic status (household expenditure as the SES proxy).

THEME 3. The survey carries no kWh or bill amount; monthly household
expenditure (food, non-food, and their sum) is the only continuous SES proxy
available. This script (a) describes the three expenditure variables with
robust, tail-aware summaries, (b) relates expenditure to dwelling floor area
(Spearman rho + bootstrap CI) and to occupation (Kruskal-Wallis + epsilon^2),
and (c) draws three publication figures. All inference is EXPLORATORY: effect
sizes lead, bootstrap 95% CIs are reported, and a Benjamini-Hochberg FDR is
applied across the association family. The sample is a skewed convenience
sample (occupation is 57/69 'Professional'), so occupation contrasts are
descriptive only.

Run:  cd code && uv run python 03_demographics_ses.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib import io, config as C, stats as S, plotting as P

PREFIX = "03_"
EXP_VARS = ["food_exp", "nonfood_exp", "total_exp"]
RNG_SEED = 0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def describe_exp(s: pd.Series) -> dict:
    """Tail-aware numeric descriptives for one (positive, right-skewed) variable."""
    x = s.dropna().astype(float)
    q1, q2, q3 = x.quantile([0.25, 0.50, 0.75])
    # bootstrap CI for both the mean and the (more robust) median
    mean, mean_lo, mean_hi = S.bootstrap_ci(x, np.mean, seed=RNG_SEED)
    med, med_lo, med_hi = S.bootstrap_ci(x, np.median, seed=RNG_SEED)
    return {
        "n": int(x.size),
        "mean": float(mean), "mean_lo": float(mean_lo), "mean_hi": float(mean_hi),
        "sd": float(x.std(ddof=1)),
        "median": float(med), "median_lo": float(med_lo), "median_hi": float(med_hi),
        "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
        "min": float(x.min()), "max": float(x.max()),
        "skew": float(x.skew()), "kurtosis": float(x.kurtosis()),
        "cv": float(x.std(ddof=1) / x.mean()),  # coefficient of variation
    }


def main():
    df = io.load_clean()
    rng_label = f"seed={RNG_SEED}, n_boot=10000"

    # ----------------------------------------------------------------- #
    # 1. Numeric descriptives (Table 03_ses_descriptives)
    # ----------------------------------------------------------------- #
    desc_rows = {v: describe_exp(df[v]) for v in EXP_VARS}
    # add dwelling area for reference (it is the SES correlate, not an SES proxy)
    desc_rows["area_sqft"] = describe_exp(df["area_sqft"])

    desc = pd.DataFrame(desc_rows).T
    desc.index = [C.LABEL.get(i, i) for i in desc.index]
    desc.index.name = "Variable"
    # order/round columns for the paper
    col_order = ["n", "mean", "mean_lo", "mean_hi", "sd", "median",
                 "median_lo", "median_hi", "q1", "q3", "iqr", "min", "max",
                 "skew", "kurtosis", "cv"]
    desc = desc[col_order]
    desc_round = desc.copy()
    for c in ["mean", "mean_lo", "mean_hi", "sd", "median", "median_lo",
              "median_hi", "q1", "q3", "iqr", "min", "max"]:
        desc_round[c] = desc_round[c].round(0).astype(int)
    desc_round["n"] = desc_round["n"].astype(int)
    for c in ["skew", "kurtosis", "cv"]:
        desc_round[c] = desc_round[c].round(2)

    desc_csv = C.TBL_DIR / f"{PREFIX}ses_descriptives.csv"
    desc_tex = C.TBL_DIR / f"{PREFIX}ses_descriptives.tex"
    desc_round.to_csv(desc_csv)
    desc_round.to_latex(
        desc_tex, index=True, escape=True, float_format="%.2f",
        caption=("Household monthly expenditure (SES proxy) and dwelling floor "
                 "area: tail-aware descriptives. Bootstrap 95\\% CIs "
                 f"({rng_label}). All monetary values in Sri Lankan Rupees/month."),
        label="tab:ses_descriptives",
    )

    # ----------------------------------------------------------------- #
    # 2a. Expenditure vs dwelling area  (Spearman + bootstrap CI)
    # ----------------------------------------------------------------- #
    area_assoc = {}
    for v in EXP_VARS:
        area_assoc[v] = S.spearman_with_ci(df["area_sqft"], df[v], seed=RNG_SEED)

    # internal coherence check: do the three expenditure measures co-move?
    coexp = {
        "food_vs_nonfood": S.spearman_with_ci(df["food_exp"], df["nonfood_exp"], seed=RNG_SEED),
    }

    # ----------------------------------------------------------------- #
    # 2b. Expenditure vs occupation (Kruskal-Wallis + epsilon^2)
    # Report the full (small-cell) breakdown AND a 2-group collapse
    # (Professional vs Other) that has a defensible cell size.
    # ----------------------------------------------------------------- #
    occ_full_groups = [g["total_exp"].dropna().values
                       for _, g in df.groupby("occupation", observed=True)]
    kw_full = S.kruskal_effect(*occ_full_groups)

    df["_occ2"] = np.where(df["occupation"] == "Professional",
                           "Professional", "Other")
    g_prof = df.loc[df["_occ2"] == "Professional", "total_exp"].dropna().values
    g_other = df.loc[df["_occ2"] == "Other", "total_exp"].dropna().values
    kw_2 = S.kruskal_effect(g_prof, g_other)
    # Difference of group medians + a two-sample (stratified) bootstrap CI.
    # bootstrap_ci_pair assumes PAIRED equal-length arrays, so for two
    # independent groups of unequal size we resample each group separately.
    diff_med = float(np.median(g_prof) - np.median(g_other))
    _rng = np.random.default_rng(RNG_SEED)
    _boot = np.array([
        np.median(g_prof[_rng.integers(0, len(g_prof), len(g_prof))])
        - np.median(g_other[_rng.integers(0, len(g_other), len(g_other))])
        for _ in range(10000)
    ])
    diff_lo, diff_hi = (float(x) for x in np.nanpercentile(_boot, [2.5, 97.5]))

    occ_table = (df.groupby("occupation", observed=True)["total_exp"]
                 .agg(n="count", median="median", mean="mean",
                      q1=lambda s: s.quantile(.25),
                      q3=lambda s: s.quantile(.75))
                 .sort_values("n", ascending=False))
    occ_table_round = occ_table.copy()
    for c in ["median", "mean", "q1", "q3"]:
        occ_table_round[c] = occ_table_round[c].round(0).astype(int)
    occ_csv = C.TBL_DIR / f"{PREFIX}exp_by_occupation.csv"
    occ_tex = C.TBL_DIR / f"{PREFIX}exp_by_occupation.tex"
    occ_table_round.to_csv(occ_csv)
    occ_table_round.to_latex(
        occ_tex, index=True, escape=True,
        caption=("Total monthly expenditure (Rs) by main occupation. Cells are "
                 "highly unbalanced (57/69 'Professional'); contrasts are "
                 "descriptive only."),
        label="tab:exp_by_occupation",
    )

    # ----------------------------------------------------------------- #
    # 3. Multiple-comparison control across the association family
    #    Family = {3 area-vs-expenditure Spearman, full-occ KW, 2-grp KW}
    # ----------------------------------------------------------------- #
    fam_names = [f"area~{v}" for v in EXP_VARS] + ["occ(full)~total", "occ(Prof/Other)~total"]
    fam_p = [area_assoc[v]["p"] for v in EXP_VARS] + [kw_full["p"], kw_2["p"]]
    fam_q = S.bh_fdr(fam_p)
    fdr = {n: {"p": float(p), "q": float(q)}
           for n, p, q in zip(fam_names, fam_p, fam_q)}

    # ----------------------------------------------------------------- #
    # FIGURES
    # ----------------------------------------------------------------- #
    fig_paths = {}

    # Fig 1: expenditure distributions -- histogram (top) + box (bottom),
    # log10 x-axis to tame the heavy right tail.
    fig, axes = plt.subplots(2, 3, figsize=P.figsize("FULL", 4.1),
                             gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    colors = P.CATEGORICAL[:3]
    for j, v in enumerate(EXP_VARS):
        x = df[v].dropna().astype(float)
        ax_h, ax_b = axes[0, j], axes[1, j]
        bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 14)
        counts, _, _ = ax_h.hist(x, bins=bins, color=colors[j],
                                  edgecolor="white", alpha=0.9)
        ax_h.axvline(x.median(), color=P.RULE, lw=1.4, ls="-",
                     label=f"median {x.median():,.0f}")
        ax_h.axvline(x.mean(), color=P.RULE, lw=1.2, ls="--",
                     label=f"mean {x.mean():,.0f}")
        ax_h.set_xscale("log")
        ax_h.set_title(C.LABEL[v])
        # headroom so the median/mean legend never sits on top of the bars
        ax_h.set_ylim(0, counts.max() * 1.28)
        ax_h.legend(loc="upper right", fontsize=P.ANNOT, frameon=True,
                    framealpha=0.9, edgecolor=P.MUTED)
        if j == 0:
            ax_h.set_ylabel("Households")
        ax_b.boxplot(x, vert=False, widths=0.6,
                     patch_artist=True,
                     boxprops=dict(facecolor=colors[j], alpha=0.85),
                     medianprops=dict(color=P.RULE, lw=1.4),
                     flierprops=dict(marker="o", markersize=3,
                                     markerfacecolor=P.RULE, alpha=0.6))
        ax_b.set_xscale("log")
        ax_b.set_yticks([])
        ax_b.set_xlabel("Rs / month (log scale)")
        ax_b.annotate(f"skew={x.skew():.2f}", xy=(0.02, 0.18),
                      xycoords="axes fraction", fontsize=P.ANNOT, color=P.RULE)
    fig.tight_layout()
    fig_paths["dist"] = P.save_fig(fig, f"{PREFIX}exp_distributions")
    plt.close(fig)

    # Fig 2: total expenditure by occupation (Professional vs Other emphasised).
    fig, ax = plt.subplots(figsize=P.figsize("FULL", 4.4))
    order = list(occ_table.index)  # already sorted by n desc
    data = [df.loc[df["occupation"] == o, "total_exp"].dropna().values for o in order]
    bp = ax.boxplot(data, vert=True, widths=0.55, patch_artist=True,
                    medianprops=dict(color=P.RULE, lw=1.5),
                    flierprops=dict(marker="o", markersize=3,
                                    markerfacecolor=P.RULE, alpha=0.5))
    for patch, o in zip(bp["boxes"], order):
        patch.set_facecolor(P.ACCENT if o == "Professional" else P.MUTED)
        patch.set_alpha(0.85)
    # jittered raw points
    rng = np.random.default_rng(RNG_SEED)
    for i, d in enumerate(data, start=1):
        jit = rng.normal(0, 0.06, size=len(d))
        ax.scatter(np.full(len(d), i) + jit, d, s=14, color=P.INK,
                   alpha=0.55, zorder=3, linewidths=0)
    ax.set_yscale("log")
    ax.set_ylabel("Total expenditure (Rs/mo, log scale)")
    wrapped = [o.replace(", ", ",\n").replace(" and ", " &\n") + f"\n(n={len(d)})"
               for o, d in zip(order, data)]
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(wrapped)
    fig.tight_layout()
    fig_paths["occ"] = P.save_fig(fig, f"{PREFIX}exp_by_occupation")
    plt.close(fig)

    # Fig 3: area vs total expenditure scatter with Spearman annotation.
    fig, ax = plt.subplots(figsize=P.figsize("FULL", 5.0))
    sub = df[["area_sqft", "total_exp", "_occ2"]].dropna(subset=["area_sqft", "total_exp"])
    for grp, mk, col in [("Professional", "o", P.ACCENT), ("Other", "s", P.RULE)]:
        g = sub[sub["_occ2"] == grp]
        ax.scatter(g["area_sqft"], g["total_exp"], s=42, marker=mk,
                   color=col, alpha=0.75, edgecolor="white", linewidth=0.6,
                   label=f"{grp} (n={len(g)})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(C.LABEL["area_sqft"] + " (log scale)")
    ax.set_ylabel(C.LABEL["total_exp"] + " (log scale)")
    a = area_assoc["total_exp"]
    ax.annotate(
        f"Spearman $\\rho$={a['rho']:.3f}\n"
        f"95% CI [{a['lo']:.2f}, {a['hi']:.2f}]\n"
        f"p={a['p']:.2f},  n={a['n']}",
        xy=(0.03, 0.97), xycoords="axes fraction", va="top", ha="left",
        fontsize=P.ANNOT, bbox=dict(boxstyle="round,pad=0.4", fc="white",
                              ec=P.MUTED, alpha=0.9))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig_paths["scatter"] = P.save_fig(fig, f"{PREFIX}area_vs_total_exp")
    plt.close(fig)

    # ----------------------------------------------------------------- #
    # Compact JSON of key findings (LAST stdout line)
    # ----------------------------------------------------------------- #
    findings = {
        "n_total": int(len(df)),
        "n_expenditure_valid": int(df["total_exp"].notna().sum()),
        "n_area_valid": int(df["area_sqft"].notna().sum()),
        "descriptives": {
            v: {
                "mean": round(desc_rows[v]["mean"], 0),
                "median": round(desc_rows[v]["median"], 0),
                "median_ci": [round(desc_rows[v]["median_lo"], 0),
                              round(desc_rows[v]["median_hi"], 0)],
                "iqr": round(desc_rows[v]["iqr"], 0),
                "min": round(desc_rows[v]["min"], 0),
                "max": round(desc_rows[v]["max"], 0),
                "skew": round(desc_rows[v]["skew"], 2),
                "cv": round(desc_rows[v]["cv"], 2),
            } for v in EXP_VARS
        },
        "area_sqft_median": round(desc_rows["area_sqft"]["median"], 0),
        "area_sqft_iqr": round(desc_rows["area_sqft"]["iqr"], 0),
        "spearman_area_vs": {
            v: {"rho": round(area_assoc[v]["rho"], 3),
                "ci": [round(area_assoc[v]["lo"], 3), round(area_assoc[v]["hi"], 3)],
                "p": round(area_assoc[v]["p"], 4), "n": area_assoc[v]["n"]}
            for v in EXP_VARS
        },
        "spearman_food_vs_nonfood": {
            "rho": round(coexp["food_vs_nonfood"]["rho"], 3),
            "ci": [round(coexp["food_vs_nonfood"]["lo"], 3),
                   round(coexp["food_vs_nonfood"]["hi"], 3)],
            "p": round(coexp["food_vs_nonfood"]["p"], 4),
        },
        "kruskal_occupation_full": {
            "H": round(kw_full["H"], 3), "p": round(kw_full["p"], 4),
            "eps2": round(kw_full["eps2"], 4), "k": kw_full["k"], "n": kw_full["n"],
        },
        "kruskal_prof_vs_other": {
            "H": round(kw_2["H"], 3), "p": round(kw_2["p"], 4),
            "eps2": round(kw_2["eps2"], 4),
            "n_prof": int(len(g_prof)), "n_other": int(len(g_other)),
            "median_prof": round(float(np.median(g_prof)), 0),
            "median_other": round(float(np.median(g_other)), 0),
            "median_diff": round(diff_med, 0),
            "median_diff_ci": [round(diff_lo, 0), round(diff_hi, 0)],
        },
        "bh_fdr_family": {n: {"p": round(d["p"], 4), "q": round(d["q"], 4)}
                          for n, d in fdr.items()},
        "n_survive_fdr_q05": int(np.nansum(np.asarray(fam_q) < 0.05)),
        "figures": {k: [str(p) for p in v] for k, v in fig_paths.items()},
        "tables": {
            "ses_descriptives": [str(desc_csv), str(desc_tex)],
            "exp_by_occupation": [str(occ_csv), str(occ_tex)],
        },
    }
    print("\n=== THEME 3 (SES / expenditure) key findings ===")
    print(json.dumps(findings, indent=2, default=str))
    print("\nWROTE FIGURES:")
    for v in fig_paths.values():
        for p in v:
            print("  ", p)
    print("WROTE TABLES:")
    for p in (desc_csv, desc_tex, occ_csv, occ_tex):
        print("  ", p)
    # LAST line: one compact JSON object
    print(json.dumps(findings, default=str))


if __name__ == "__main__":
    main()
