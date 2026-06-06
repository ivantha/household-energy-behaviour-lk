"""02 -- Theme 2: Dwelling & housing fabric (descriptive EDA).

Convenience sample, N=70 Sri Lankan households. This survey did NOT measure
electricity consumption; we describe the physical housing stock only. All output
is descriptive/exploratory: counts and shares for categorical fabric variables
(with bootstrap CIs on the leading-category share), a full robust numeric summary
of floor area and area-per-story (median/IQR/range/skew with bootstrap median CIs),
and the ventilation/lighting indicator profile.

Outputs:
  outputs/figures/02_*.{png,pdf}   categorical panel, area distribution, ventilation
  outputs/tables/02_*.{csv,tex}    dwelling descriptives
  stdout last line                 one compact JSON of key findings
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats as ss
import matplotlib.pyplot as plt

from lib import io, config as C, stats as S, plotting as P

PREFIX = "02_"
N_BOOT = 10000
SEED = 0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def cat_share_table(df, col, order=None):
    """Count / percent (of non-missing) / Wilson-style bootstrap CI for each level."""
    s = df[col]
    n_total = len(s)
    n_miss = int(s.isna().sum())
    valid = s.dropna()
    n_valid = len(valid)
    vc = valid.value_counts()
    if order is not None:
        vc = vc.reindex([o for o in order if o in vc.index]).dropna()
    else:
        vc = vc.sort_values(ascending=False)
    rows = []
    for level, cnt in vc.items():
        cnt = int(cnt)
        ind = (valid == level).astype(float).values
        # bootstrap CI on the proportion
        est, lo, hi = S.bootstrap_ci(ind, np.mean, n_boot=N_BOOT, ci=95, seed=SEED)
        rows.append({
            "variable": C.LABEL.get(col, col),
            "level": str(level),
            "count": cnt,
            "pct_of_valid": round(100 * cnt / n_valid, 1),
            "share": round(est, 4),
            "share_lo": round(lo, 4),
            "share_hi": round(hi, 4),
            "n_valid": n_valid,
            "n_missing": n_miss,
            "n_total": n_total,
        })
    return pd.DataFrame(rows)


def numeric_summary(series, name):
    x = pd.to_numeric(series, errors="coerce").dropna().values
    n = len(x)
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    # bootstrap CI for the median
    med, med_lo, med_hi = S.bootstrap_ci(x, np.median, n_boot=N_BOOT, ci=95, seed=SEED)
    # Shapiro normality (exploratory; flags departure from normality)
    sh_W, sh_p = ss.shapiro(x) if n >= 3 else (np.nan, np.nan)
    return {
        "variable": name,
        "n": n,
        "n_missing": int(series.isna().sum()),
        "mean": round(float(np.mean(x)), 1),
        "sd": round(float(np.std(x, ddof=1)), 1),
        "min": round(float(np.min(x)), 1),
        "q1": round(float(q1), 1),
        "median": round(float(q2), 1),
        "median_lo": round(float(med_lo), 1),
        "median_hi": round(float(med_hi), 1),
        "q3": round(float(q3), 1),
        "max": round(float(np.max(x)), 1),
        "iqr": round(float(q3 - q1), 1),
        "skew": round(float(ss.skew(x, bias=False)), 3),
        "cv": round(float(np.std(x, ddof=1) / np.mean(x)), 3),
        "shapiro_W": round(float(sh_W), 3),
        "shapiro_p": float(sh_p),
    }


def save_table(df, name):
    csv_p = C.TBL_DIR / f"{name}.csv"
    tex_p = C.TBL_DIR / f"{name}.tex"
    df.to_csv(csv_p, index=False)
    df.to_latex(tex_p, index=False, escape=True, na_rep="",
                caption=name.replace("_", " "), label=f"tab:{name}")
    return str(csv_p), str(tex_p)


# --------------------------------------------------------------------------- #
def main():
    df = io.load_clean()
    n = len(df)

    # Sensible display orders (chronology / floor count) where natural ----------
    ORDERS = {
        "build_period": ["Before 1980", "1980 - 1989", "1990 - 1999",
                         "2000 - 2009", "2010 - 2019", "After 2019"],
        "stories": [1.0, 2.0, 3.0],
        "house_type": ["Single House - Single Floor", "Single House -Double Floor",
                       "Single House - Above double floor", "Flat", "Slum / Shanty"],
        "wall_material": ["Brick", "Cement Block", "Cabook"],
        "roof_type": ["Tile", "Asbestos", "Concrete"],
    }
    CAT_VARS = ["build_period", "arch_design", "house_type", "stories",
                "wall_material", "roof_type"]

    # === Categorical descriptives table ===================================== #
    cat_tabs = []
    cat_summary = {}
    for c in CAT_VARS:
        t = cat_share_table(df, c, order=ORDERS.get(c))
        cat_tabs.append(t)
        top = t.iloc[0]
        cat_summary[c] = {
            "n_levels": int(df[c].nunique(dropna=True)),
            "top_level": top["level"],
            "top_count": int(top["count"]),
            "top_pct": float(top["pct_of_valid"]),
            "top_share_ci": [float(top["share_lo"]), float(top["share_hi"])],
            "n_missing": int(df[c].isna().sum()),
        }
    cat_table = pd.concat(cat_tabs, ignore_index=True)
    cat_csv, cat_tex = save_table(cat_table, f"{PREFIX}dwelling_categorical")

    # === Numeric descriptives table ========================================= #
    num_rows = [
        numeric_summary(df["area_sqft"], C.LABEL["area_sqft"]),
        numeric_summary(df["area_per_story"], "Floor area per story (sq ft)"),
    ]
    num_table = pd.DataFrame(num_rows)
    num_csv, num_tex = save_table(num_table, f"{PREFIX}dwelling_numeric")

    # === Ventilation indicator table ======================================== #
    vent_inds = ["vent_window", "vent_glasswall", "vent_glassroof",
                 "vent_pergola", "vent_other"]
    vent_labels = {
        "vent_window": "Window wall", "vent_glasswall": "Transparent wall",
        "vent_glassroof": "Transparent roof", "vent_pergola": "Pergola",
        "vent_other": "Other",
    }
    n_reported = int(df["vent_reported"].sum())          # answered the item
    n_unreported = n - n_reported                        # left blank (no extra vent)
    vent_rows = []
    for v in vent_inds:
        cnt = int(df[v].sum())
        # share among ALL households (blank counts as not-using that feature)
        ind_all = df[v].astype(float).values
        est, lo, hi = S.bootstrap_ci(ind_all, np.mean, n_boot=N_BOOT, ci=95, seed=SEED)
        vent_rows.append({
            "indicator": vent_labels[v],
            "count": cnt,
            "pct_of_all": round(100 * cnt / n, 1),
            "pct_of_reporters": round(100 * cnt / n_reported, 1),
            "share_all": round(est, 4),
            "share_all_lo": round(lo, 4),
            "share_all_hi": round(hi, 4),
        })
    vent_table = pd.DataFrame(vent_rows).sort_values("count", ascending=False).reset_index(drop=True)
    vent_csv, vent_tex = save_table(vent_table, f"{PREFIX}ventilation_indicators")

    # vent_count summary
    vc_counts = df["vent_count"].value_counts().sort_index()
    vent_count_dist = {int(k): int(v) for k, v in vc_counts.items()}
    vent_count_med = float(df["vent_count"].median())
    vent_count_mean = float(df["vent_count"].mean())

    # ====================================================================== #
    # FIGURES
    # ====================================================================== #
    # --- Fig 1: categorical dwelling panel (2x3) --------------------------- #
    fig1, axes = plt.subplots(2, 3, figsize=P.figsize("FULL", 4.6))
    panel_specs = [
        ("build_period", "Construction period"),
        ("house_type", "House type"),
        ("stories", "Number of stories"),
        ("arch_design", "Architectural design level"),
        ("wall_material", "Outer-wall material"),
        ("roof_type", "Roof type"),
    ]
    # short labels for the long arch_design categories
    arch_short = {
        "The house was designed by a certified architect": "Certified architect",
        "House was not specifically designed by an architect, but the plan was inspected by a certified architect/engineer": "Plan inspected by architect/eng.",
        "No such professional designing was done; the house was designed to suit our needs and wants": "Self-designed (needs/wants)",
        "I am not aware of it": "Not aware",
        "House was designed barely to pass the statutory requirements of the local authorities": "Bare statutory compliance",
        "Other": "Other",
    }
    house_short = {
        "Single House - Single Floor": "Single, single floor",
        "Single House -Double Floor": "Single, double floor",
        "Single House - Above double floor": "Single, >2 floors",
        "Flat": "Flat", "Slum / Shanty": "Slum / shanty",
    }
    for ax, (col, title) in zip(axes.ravel(), panel_specs):
        order = ORDERS.get(col)
        vc = df[col].value_counts()
        if order is not None:
            vc = vc.reindex([o for o in order if o in vc.index]).dropna()
        else:
            vc = vc.sort_values(ascending=False)
        idx = vc.index.tolist()
        if col == "arch_design":
            labels = [arch_short.get(str(i), str(i)) for i in idx]
        elif col == "house_type":
            labels = [house_short.get(str(i), str(i)) for i in idx]
        elif col == "stories":
            labels = [str(int(i)) for i in idx]
        else:
            labels = [str(i) for i in idx]
        ypos = np.arange(len(vc))[::-1]  # first item on top
        ax.barh(ypos, vc.values, color=P.ACCENT)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels)
        ax.set_title(title)
        ax.set_xlabel("Households")
        for yp, val in zip(ypos, vc.values):
            ax.text(val + 0.4, yp, str(int(val)), va="center", fontsize=P.ANNOT)
        ax.margins(x=0.12)
    fig1.tight_layout()
    f1 = P.save_fig(fig1, f"{PREFIX}categorical_panel")

    # --- Fig 2: area_sqft distribution (hist + box, raw and log) ----------- #
    area = df["area_sqft"].dropna().values
    fig2, (axh, axb) = plt.subplots(2, 1, figsize=P.figsize("FULL", 4.0),
                                    gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    bins = np.histogram_bin_edges(area, bins="auto")
    axh.hist(area, bins=bins, color=P.ACCENT, alpha=0.85, edgecolor="white")
    med = np.median(area)
    mean = np.mean(area)
    axh.axvline(med, color=P.ACCENT2, lw=2, label=f"median = {med:.0f}")
    axh.axvline(mean, color=P.RULE, lw=1.6, ls="--", label=f"mean = {mean:.0f}")
    axh.set_ylabel("Households")
    axh.legend()
    axb.boxplot(area, vert=False, widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor=P.ACCENT, alpha=0.6),
                medianprops=dict(color=P.ACCENT2, lw=2),
                flierprops=dict(marker="o", markersize=4, markerfacecolor=P.RULE,
                                markeredgecolor="none", alpha=0.6))
    axb.set_yticks([])
    axb.set_xlabel("Floor area (sq ft)")
    fig2.tight_layout()
    f2 = P.save_fig(fig2, f"{PREFIX}area_distribution")

    # --- Fig 3: ventilation indicators ------------------------------------- #
    fig3, (axv1, axv2) = plt.subplots(1, 2, figsize=P.figsize("FULL", 4.2),
                                      gridspec_kw={"width_ratios": [3, 2]})
    # left: indicator prevalence (share of all households) w/ bootstrap CI
    vt = vent_table.sort_values("count")
    ypos = np.arange(len(vt))
    pct = vt["pct_of_all"].values
    lo = vt["share_all_lo"].values * 100
    hi = vt["share_all_hi"].values * 100
    xerr = np.vstack([pct - lo, hi - pct])
    axv1.barh(ypos, pct, color=P.ACCENT, alpha=0.9)
    axv1.errorbar(pct, ypos, xerr=xerr, fmt="none", ecolor=P.RULE, capsize=3, lw=1.2)
    axv1.set_yticks(ypos)
    axv1.set_yticklabels(vt["indicator"].tolist())
    axv1.set_xlabel("% of all households (95% bootstrap CI)")
    axv1.set_title("Ventilation/lighting features")
    for yp, c in zip(ypos, vt["count"].values):
        axv1.text(1, yp, f"n={int(c)}", va="center", fontsize=P.ANNOT, color="white")
    # right: number of extra features per household
    order_k = sorted(vent_count_dist)
    vals = [vent_count_dist[k] for k in order_k]
    colors = [P.MUTED] + [P.ACCENT] * (len(order_k) - 1)
    axv2.bar([str(k) for k in order_k], vals, color=colors)
    for i, v in enumerate(vals):
        axv2.text(i, v + 0.4, str(v), ha="center", fontsize=P.ANNOT)
    axv2.set_xlabel("No. of extra ventilation features")
    axv2.set_ylabel("Households")
    axv2.set_title("Feature count per household")
    fig3.tight_layout()
    f3 = P.save_fig(fig3, f"{PREFIX}ventilation_indicators")

    # ====================================================================== #
    # KEY FINDINGS JSON
    # ====================================================================== #
    findings = {
        "theme": "dwelling_housing_fabric",
        "n": n,
        "categorical": cat_summary,
        "area_sqft": {k: num_rows[0][k] for k in
                      ["n", "n_missing", "median", "median_lo", "median_hi",
                       "iqr", "min", "max", "mean", "sd", "skew", "shapiro_p"]},
        "area_per_story": {k: num_rows[1][k] for k in
                           ["n", "n_missing", "median", "median_lo", "median_hi",
                            "iqr", "min", "max", "mean", "sd", "skew", "shapiro_p"]},
        "ventilation": {
            "n_reporting_any_field": n_reported,
            "n_no_extra_features": n_unreported,
            "pct_no_extra_features": round(100 * n_unreported / n, 1),
            "indicator_pct_of_all": {r["indicator"]: r["pct_of_all"] for r in vent_rows},
            "vent_count_median": vent_count_med,
            "vent_count_mean": round(vent_count_mean, 2),
            "vent_count_dist": vent_count_dist,
        },
        "figures": {
            "categorical_panel": f1,
            "area_distribution": f2,
            "ventilation": f3,
        },
        "tables": {
            "categorical": [cat_csv, cat_tex],
            "numeric": [num_csv, num_tex],
            "ventilation": [vent_csv, vent_tex],
        },
    }
    print(json.dumps(findings, default=str))


if __name__ == "__main__":
    main()
