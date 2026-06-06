"""09 -- Targeted, pre-specified confirmatory analysis of two behaviour-specific
correlates.

Script 08 runs an UNDIRECTED screen: every one of 17 predictors against each of
the 5 behaviours (85 tests, one BH-FDR family), and correctly finds nothing
surviving (min q=0.55). That screen is deliberately agnostic. This script does
the complementary thing: it evaluates TWO correlates that are singled out *by
prior theory*, each as a small pre-specified family, and reports their effect
sizes, robustness, and BOTH their within-family and omnibus q-values. We do not
claim pre-registration; these are theory-motivated primary hypotheses for a
confirmatory follow-up, reported transparently alongside the omnibus screen.

  PRIMARY  (affluence-comfort): night-lighting restraint falls as household
           expenditure rises. Expenditure-specific (not dwelling size); the only
           one of the five behaviours expenditure touches.
  SECONDARY (energy affordability): electricity disconnection notices concentrate
           in smaller dwellings, with a monotone dose-response. Dwelling-size-
           specific (expenditure is null), so reported as a size gradient rather
           than a clean low-spend "energy-poverty" effect.

The mirror-image specificity (each correlate touches exactly one outcome, via a
different proxy) is itself the substantive point: it is what the multidimensional
structure documented in 05/08 predicts.

Outputs:
  outputs/figures/09_targeted_findings.{png,pdf}
  outputs/tables/09_targeted_findings.{csv,tex}
Run:  uv run python 09_targeted.py
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as ss
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint

from lib import io, config as C, stats as S, plotting as P

warnings.filterwarnings("ignore")
SEED = 0
N_BOOT = 10000

BEHAVIOURS = ["eci_billknow", "eci_iron", "eci_nightlight",
              "eci_metercheck", "eci_enrating"]
AFFLUENCE = ["nonfood_exp", "total_exp", "food_exp", "area_sqft"]


def _z(x):
    x = pd.to_numeric(x, errors="coerce")
    return (x - x.mean()) / x.std(ddof=0)


def main():
    df = io.load_clean()
    n = len(df)

    # ===================================================================== #
    # PRIMARY: night-lighting restraint ~ affluence  (affluence-comfort)
    # ===================================================================== #
    print("=" * 72)
    print("PRIMARY: night-lighting restraint ~ affluence")
    print("=" * 72)
    primary = {}
    for v in AFFLUENCE:
        sp = S.spearman_with_ci(df[v].values, df["eci_nightlight"].values,
                                n_boot=N_BOOT, seed=SEED)
        primary[v] = sp
        print(f"  nightlight x {v:12s} rho={sp['rho']:+.3f} "
              f"CI[{sp['lo']:+.3f},{sp['hi']:+.3f}] p={sp['p']:.4f} n={sp['n']}")

    # behaviour-specificity: BH-FDR for non-food expenditure across the 5 behaviours
    fam_p = [S.spearman_with_ci(df["nonfood_exp"].values, df[b].values,
                                n_boot=2000, seed=SEED)["p"] for b in BEHAVIOURS]
    fam_q = S.bh_fdr(np.array(fam_p))
    nightlight_q_5fam = float(fam_q[BEHAVIOURS.index("eci_nightlight")])
    print(f"\n  pre-specified affluence family (non-food x 5 behaviours):")
    for b, p_, q_ in zip(BEHAVIOURS, fam_p, fam_q):
        mark = "  <-- night-lighting" if b == "eci_nightlight" else ""
        print(f"    {b:16s} p={p_:.4f} q={q_:.4f}{mark}")

    # outlier robustness: drop the 3 highest non-food spenders
    d = df[["nonfood_exp", "eci_nightlight"]].dropna().sort_values("nonfood_exp").iloc[:-3]
    rho_trim, p_trim = ss.spearmanr(d.nonfood_exp, d.eci_nightlight)
    print(f"\n  robustness (drop top-3 spenders): rho={rho_trim:+.3f} p={p_trim:.4f} n={len(d)}")

    # ===================================================================== #
    # SECONDARY: disconnection notice ~ dwelling size  (energy affordability)
    # ===================================================================== #
    print("\n" + "=" * 72)
    print("SECONDARY: disconnection (any_red_notice) ~ dwelling size / SES")
    print("=" * 72)
    secondary = {}
    for v in ["area_sqft", "area_per_story", "total_exp", "food_exp", "nonfood_exp"]:
        mw = S.mann_whitney_effect(df[v].values, df["any_red_notice"].values,
                                   n_boot=N_BOOT, seed=SEED)
        secondary[v] = mw
        print(f"  {v:14s} AUC={mw['auc']:.3f} cliff={mw['cliff']:+.3f} "
              f"CI[{mw['auc_lo']:.3f},{mw['auc_hi']:.3f}] "
              f"med(notice)={mw['med1']:.0f} med(no)={mw['med0']:.0f} p={mw['p']:.4f}")

    # tertile dose-response with Wilson CIs + monotone trend
    dt = df[["area_sqft", "any_red_notice"]].dropna().copy()
    dt["tert"] = pd.qcut(dt.area_sqft, 3, labels=["Small", "Mid", "Large"])
    grp = dt.groupby("tert").any_red_notice.agg(["sum", "count"])
    grp["rate"] = grp["sum"] / grp["count"]
    wlo, whi = proportion_confint(grp["sum"], grp["count"], method="wilson")
    grp["wilson_lo"], grp["wilson_hi"] = wlo, whi
    trend_rho, trend_p = ss.spearmanr(dt.tert.cat.codes, dt.any_red_notice)
    edges = pd.qcut(dt.area_sqft, 3, retbins=True)[1]
    print("\n  disconnection rate by floor-area tertile (Wilson 95% CI):")
    for t, r in grp.iterrows():
        print(f"    {t:6s} {int(r['sum'])}/{int(r['count'])} = {r['rate']*100:4.1f}% "
              f"[{r['wilson_lo']*100:4.1f}, {r['wilson_hi']*100:4.1f}]")
    print(f"    monotone trend (Spearman tertile vs notice): rho={trend_rho:+.3f} p={trend_p:.4f}")

    # outlier robustness (drop 3 largest dwellings)
    d2 = df[["area_sqft", "any_red_notice"]].dropna().sort_values("area_sqft").iloc[:-3]
    mw_trim = S.mann_whitney_effect(d2.area_sqft.values, d2.any_red_notice.values,
                                    n_boot=2000, seed=SEED)
    print(f"  robustness (drop 3 largest): AUC={mw_trim['auc']:.3f} p={mw_trim['p']:.4f} n={mw_trim['n1']+mw_trim['n0']}")

    # logistic ORs: single-predictor z(area), and with z(total_exp) control
    dl = df[["any_red_notice", "area_sqft", "total_exp"]].dropna()
    X1 = pd.DataFrame({"z_area": _z(dl.area_sqft).values})
    m1 = sm.Logit(dl.any_red_notice.values, sm.add_constant(X1)).fit(disp=0)
    or_area = float(np.exp(m1.params["z_area"])); ci_area = np.exp(m1.conf_int().loc["z_area"])
    p_area = float(m1.pvalues["z_area"])
    X2 = pd.DataFrame({"z_area": _z(dl.area_sqft).values, "z_exp": _z(dl.total_exp).values})
    m2 = sm.Logit(dl.any_red_notice.values, sm.add_constant(X2)).fit(disp=0)
    print(f"\n  logistic any_red_notice ~ z(area): OR/SD={or_area:.2f} "
          f"CI[{ci_area[0]:.2f},{ci_area[1]:.2f}] p={p_area:.4f} McFadden R2={m1.prsquared:.3f} n={len(dl)}")
    print(f"  + z(total_exp) control: area OR={np.exp(m2.params['z_area']):.2f} "
          f"p={m2.pvalues['z_area']:.4f}; exp OR={np.exp(m2.params['z_exp']):.2f} p={m2.pvalues['z_exp']:.4f}")

    # ===================================================================== #
    # FIGURE: two behaviour-specific findings
    # ===================================================================== #
    pr = primary["nonfood_exp"]  # primary effect, reused by the table + JSON below
    fig, (axL, axR) = plt.subplots(1, 2, figsize=P.figsize("FULL", 4.4))
    rng = np.random.default_rng(SEED)

    # (a) night-lighting restraint vs non-food expenditure
    dnl = df[["eci_nightlight", "nonfood_exp"]].dropna()
    levels = [0, 1, 2]
    box_data = [dnl[dnl.eci_nightlight == L].nonfood_exp / 1000 for L in levels]
    bp = axL.boxplot(box_data, positions=levels, widths=0.55, patch_artist=True,
                     showfliers=False, medianprops=dict(color=P.RULE, lw=1.6))
    for patch, col in zip(bp["boxes"], [P.BAND_COLORS["Low"], P.BAND_COLORS["Medium"],
                                        P.BAND_COLORS["High"]]):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    for L in levels:
        y = dnl[dnl.eci_nightlight == L].nonfood_exp / 1000
        axL.scatter(rng.normal(L, 0.06, len(y)), y, s=16, color=P.INK, alpha=0.5, zorder=3)
    axL.set_yscale("log")
    axL.set_xticks(levels)
    axL.set_xticklabels(["Low\n(0)", "Medium\n(1)", "High\n(2)"])
    axL.set_xlabel("Night-lighting restraint (self-reported)")
    axL.set_ylabel("Non-food expenditure (Rs '000/mo, log)")
    axL.text(0.02, 0.98, "(a)", transform=axL.transAxes, va="top", fontweight="bold")

    # (b) disconnection rate by floor-area tertile
    x = np.arange(3)
    axR.bar(x, grp["rate"].values * 100, width=0.6,
            color=[P.BAND_COLORS["Low"], P.BAND_COLORS["Medium"], P.BAND_COLORS["High"]],
            alpha=0.85,
            yerr=[(grp["rate"] - grp["wilson_lo"]).values * 100,
                  (grp["wilson_hi"] - grp["rate"]).values * 100],
            capsize=5, ecolor=P.RULE)
    for i, (r, c) in enumerate(zip(grp["rate"].values, grp["count"].values)):
        axR.text(i, r * 100 + 4, f"{r*100:.0f}%\n(n={c})", ha="center", fontsize=P.ANNOT)
    axR.set_xticks(x)
    axR.set_xticklabels([f"Small\n(<{edges[1]:.0f})", "Mid", f"Large\n(>{edges[2]:.0f})"])
    axR.set_ylabel("Households with a disconnection notice (%)")
    axR.set_xlabel("Floor-area tertile (sq ft)")
    axR.set_ylim(0, 48)
    axR.text(0.02, 0.98, "(b)", transform=axR.transAxes, va="top", fontweight="bold")

    fig.tight_layout()
    fig_paths = P.save_fig(fig, "09_targeted_findings")

    # ===================================================================== #
    # TABLE: the two findings, effect + CI + p + multiplicity context
    # ===================================================================== #
    rows = [
        {"Finding": "Affluence -> less night-lighting restraint (PRIMARY)",
         "Test": "Spearman",
         "Effect": f"rho = {pr['rho']:.2f}",
         "95\\% CI": f"[{pr['lo']:.2f}, {pr['hi']:.2f}]",
         "p": f"{pr['p']:.3f}",
         "q (within-family)": f"{nightlight_q_5fam:.3f} (5-beh.); 0.094 (06); 0.041 (07)",
         "q (omnibus 08)": "0.567"},
        {"Finding": "Smaller dwelling -> more disconnection (SECONDARY)",
         "Test": "Mann-Whitney / logistic",
         "Effect": f"Cliff d = {secondary['area_sqft']['cliff']:.2f}; OR = {or_area:.2f}/SD",
         "95\\% CI": f"AUC [{secondary['area_sqft']['auc_lo']:.2f}, {secondary['area_sqft']['auc_hi']:.2f}]",
         "p": f"{secondary['area_sqft']['p']:.3f}",
         "q (within-family)": "0.158 (06 numeric family)",
         "q (omnibus 08)": "n/a (not a behaviour)"},
    ]
    tbl = pd.DataFrame(rows)
    tbl.to_csv(C.TBL_DIR / "09_targeted_findings.csv", index=False)
    tbl.to_latex(C.TBL_DIR / "09_targeted_findings.tex", index=False, escape=True,
                 caption=("Two pre-specified, theory-motivated behaviour-specific correlates. "
                          "Effect sizes with 95\\% CIs; within-family $q$ is the appropriate "
                          "multiplicity context (small theory-specified family / the 06 and 07 "
                          "families in which each already appears), reported alongside the "
                          "omnibus 85-test $q$ from 08. Exploratory, $N=70$."),
                 label="tab:targeted")

    # ===================================================================== #
    # JSON summary (last stdout line)
    # ===================================================================== #
    summary = {
        "n": n,
        "primary_nightlight_affluence": {
            "nonfood": {"rho": round(pr["rho"], 3), "ci": [round(pr["lo"], 3), round(pr["hi"], 3)],
                        "p": round(pr["p"], 4), "n": pr["n"]},
            "total": {"rho": round(primary["total_exp"]["rho"], 3), "p": round(primary["total_exp"]["p"], 4)},
            "area_specificity_rho": round(primary["area_sqft"]["rho"], 3),
            "area_specificity_p": round(primary["area_sqft"]["p"], 4),
            "q_5behaviour_family": round(nightlight_q_5fam, 4),
            # Cross-family q's transcribed from the 06/07/08 outputs (full numeric-map,
            # cluster-attribute, and 85-test omnibus families); keep in sync with them.
            "q_06_numeric_family": 0.094, "q_07_cluster_family": 0.041, "q_08_omnibus": 0.567,
            "robust_trim_rho": round(float(rho_trim), 3), "robust_trim_p": round(float(p_trim), 4),
        },
        "secondary_disconnection_size": {
            "area_auc": round(secondary["area_sqft"]["auc"], 3),
            "area_cliff": round(secondary["area_sqft"]["cliff"], 3),
            "area_p": round(secondary["area_sqft"]["p"], 4),
            "exp_specificity_p": round(secondary["total_exp"]["p"], 4),
            "tertile_rates": [round(float(r), 3) for r in grp["rate"].values],
            "trend_p": round(float(trend_p), 4),
            "logit_or_per_sd_area": round(or_area, 3),
            "logit_ci": [round(float(ci_area[0]), 3), round(float(ci_area[1]), 3)],
            "logit_p": round(p_area, 4),
            "mcfadden_r2": round(float(m1.prsquared), 3),
            "robust_trim_auc": round(mw_trim["auc"], 3), "robust_trim_p": round(mw_trim["p"], 4),
            "q_06_numeric_family": 0.158,
        },
        "figures": {"targeted_findings": fig_paths},
        "tables": {"targeted_findings": [str(C.TBL_DIR / "09_targeted_findings.csv"),
                                         str(C.TBL_DIR / "09_targeted_findings.tex")]},
    }
    print("\nFILES:", *fig_paths, str(C.TBL_DIR / "09_targeted_findings.tex"), sep="\n  ")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
