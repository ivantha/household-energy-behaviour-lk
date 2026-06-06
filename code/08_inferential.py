"""08 -- Inferential: correlates of EACH of the five energy-conserving behaviours.

The five behaviours are treated as INDEPENDENT ordinal outcomes (they do not form
a reliable scale: Cronbach alpha ~ 0.04). For every (predictor x behaviour) pair
we estimate an EFFECT SIZE with a bootstrap CI, then control the false-discovery
rate across the WHOLE family of tests with Benjamini-Hochberg. Given n=69 and a
heavily skewed convenience sample, this is exploratory; we lead with effect sizes
and are explicit about whether anything survives FDR.

Methods (all small-n, all from lib.stats):
  * numeric predictors  -> Spearman rho + bootstrap 95% CI (spearman_with_ci)
  * categorical preds    -> Kruskal-Wallis with epsilon-squared (kruskal_effect)
  * one combined predictors x behaviours effect-size matrix (variance-explained
    footing: eps^2 for KW, rho^2 for Spearman), with BH q-values across the family
  * exploratory ordinal logistic regression (statsmodels OrderedModel, logit) for
    the two best-behaved outcomes (metercheck, iron), clearly flagged.

Outputs:
  outputs/figures/08_effectsize_heatmap.{png,pdf}
  outputs/figures/08_top_categorical.{png,pdf}
  outputs/figures/08_top_numeric.{png,pdf}
  outputs/tables/08_notable_correlates.{csv,tex}
  outputs/tables/08_effectsize_matrix.{csv,tex}
  outputs/tables/08_ordinal_logit.{csv,tex}

Run:  uv run python 08_inferential.py
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as ss

from lib import io, config as C, stats as S, plotting as P

warnings.filterwarnings("ignore")
RNG_SEED = 0
N_BOOT = 10000

# --------------------------------------------------------------------------- #
# Outcomes and predictors
# --------------------------------------------------------------------------- #
BEHAVIOURS = {
    "eci_billknow":   "Bill-calc literacy",
    "eci_iron":       "Ironing efficiency",
    "eci_nightlight": "Night-light restraint",
    "eci_metercheck": "Meter-checking",
    "eci_enrating":   "Energy-rating use",
}
NUMERIC = {
    "area_sqft":   "Floor area (sq ft)",
    "stories":     "No. of stories",
    "food_exp":    "Food exp (Rs/mo)",
    "nonfood_exp": "Non-food exp (Rs/mo)",
    "total_exp":   "Total exp (Rs/mo)",
}
# Short labels for the 12 categorical predictors (collapsed below).
CATEG = {
    "build_period":   "Construction period",
    "arch_design":    "Architectural design",
    "house_type":     "House type",
    "wall_material":  "Wall material",
    "roof_type":      "Roof type",
    "province":       "Province",
    "gender":         "Gender",
    "edu_attendance": "University-educated",
    "occupation":     "Professional occup.",
    "wiring_pro":     "Pro. wiring",
    "uses_renewable": "Uses renewable",
    "pay_practice":   "Bill-payment style",
}
PRED_LABEL = {**NUMERIC, **CATEG}
BEH_LABEL = BEHAVIOURS


# --------------------------------------------------------------------------- #
# Collapse rare categorical levels (n=69: keep groups large enough to compare)
# --------------------------------------------------------------------------- #
def collapse_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    g = pd.DataFrame(index=df.index)
    # construction period: 6 ordered bins, all cells >= 6 -> keep as-is
    g["build_period"] = df["build_period"]
    ad = df["arch_design"]
    g["arch_design"] = np.select(
        [ad.str.contains("designed by a certified architect", na=False),
         ad.str.contains("inspected by a certified", na=False)],
        ["Architect-designed", "Architect-inspected"],
        default="No/unknown design")
    g["house_type"] = df["house_type"].map({
        "Single House - Single Floor": "Single floor",
        "Single House -Double Floor": "Double floor",
        "Single House - Above double floor": "Above double floor",
    })  # Slum (1) and Flat (1) -> NaN (too few to be a group)
    g["wall_material"] = np.where(df["wall_material"].eq("Brick"), "Brick", "Block/Cabook")
    g["roof_type"] = df["roof_type"]                       # Asbestos/Tile/Concrete (>=7)
    g["province"] = df["province"].where(
        df["province"].isin(["Western", "Southern", "North Western"]), "Other")
    g["gender"] = df["gender"]
    g["edu_attendance"] = np.where(df["edu_attendance"].eq("University"),
                                   "University", "Not university")
    g["occupation"] = np.where(df["occupation"].eq("Professional"),
                               "Professional", "Non-professional")
    g["wiring_pro"] = np.where(df["wiring_pro"].eq("Yes"),
                               "Yes (professional)", "No/unaware")
    g["uses_renewable"] = df["uses_renewable"].map({0: "No", 1: "Yes"})
    pp = df["pay_practice"]
    g["pay_practice"] = np.select(
        [pp.str.contains("cover the whole", na=False),
         pp.str.contains("exact amount", na=False)],
        ["Rounds up (buffer)", "Pays exact"],
        default="Rounds down/partial")
    return g


def cliffs_delta(a, b) -> float:
    """Cliff's delta effect size for two groups (direction-aware, in [-1,1])."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return float((gt - lt) / (len(a) * len(b)))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    df = io.load_clean()
    n = len(df)
    cats = collapse_categoricals(df)

    long_rows = []          # one row per (predictor, behaviour) test
    eff_matrix = {}         # behaviour -> {predictor: variance-explained effect}
    for b in BEHAVIOURS:
        eff_matrix[b] = {}

    # ---- numeric predictors: Spearman ------------------------------------- #
    for pcode in NUMERIC:
        for b in BEHAVIOURS:
            res = S.spearman_with_ci(df[pcode], df[b], n_boot=N_BOOT, seed=RNG_SEED)
            rho = res["rho"]
            eff_matrix[b][pcode] = float(rho ** 2) if not np.isnan(rho) else np.nan
            long_rows.append({
                "predictor": pcode, "predictor_label": PRED_LABEL[pcode],
                "predictor_kind": "numeric", "behaviour": b,
                "behaviour_label": BEH_LABEL[b], "test": "spearman",
                "n": res["n"], "effect_name": "rho", "effect": rho,
                "effect_lo": res["lo"], "effect_hi": res["hi"],
                "var_explained": float(rho ** 2) if not np.isnan(rho) else np.nan,
                "stat": rho, "p": res["p"],
            })

    # ---- categorical predictors: Kruskal-Wallis + eps^2 ------------------- #
    for pcode in CATEG:
        gvar = cats[pcode]
        for b in BEHAVIOURS:
            sub = pd.DataFrame({"g": gvar, "y": df[b]}).dropna()
            groups = [grp["y"].values for _, grp in sub.groupby("g") if len(grp) > 0]
            kw = S.kruskal_effect(*groups)
            eps2 = kw["eps2"]
            eff_matrix[b][pcode] = float(eps2) if not np.isnan(eps2) else np.nan
            # bootstrap CI for eps^2 (percentile, resample within the merged sample)
            lo, hi = _boot_eps2(sub, n_boot=2000, seed=RNG_SEED)
            # for 2-group predictors also record Cliff's delta (direction)
            delta = np.nan
            levs = sub["g"].unique()
            if len(levs) == 2:
                a = sub.loc[sub["g"] == levs[0], "y"].values
                bb = sub.loc[sub["g"] == levs[1], "y"].values
                delta = cliffs_delta(a, bb)
            long_rows.append({
                "predictor": pcode, "predictor_label": PRED_LABEL[pcode],
                "predictor_kind": "categorical", "behaviour": b,
                "behaviour_label": BEH_LABEL[b], "test": "kruskal",
                "n": int(kw.get("n", len(sub))), "effect_name": "eps2",
                "effect": eps2, "effect_lo": lo, "effect_hi": hi,
                "var_explained": float(eps2) if not np.isnan(eps2) else np.nan,
                "stat": kw["H"], "p": kw["p"], "k_groups": kw.get("k", np.nan),
                "cliffs_delta": delta,
            })

    long = pd.DataFrame(long_rows)

    # ---- BH-FDR across the WHOLE family ----------------------------------- #
    long["q_value"] = S.bh_fdr(long["p"].values)
    long["sig_raw"] = long["p"] < 0.05
    long["survives_fdr"] = long["q_value"] < 0.05
    long = long.sort_values("p", na_position="last").reset_index(drop=True)

    n_tests = int(long["p"].notna().sum())
    n_raw = int(long["sig_raw"].sum())
    n_fdr = int(long["survives_fdr"].sum())

    # ---- effect-size matrix (predictors x behaviours), variance explained -- #
    pred_order = list(NUMERIC) + list(CATEG)
    mat = pd.DataFrame(
        {b: [eff_matrix[b].get(p, np.nan) for p in pred_order] for b in BEHAVIOURS},
        index=[PRED_LABEL[p] for p in pred_order],
    )
    mat.columns = [BEH_LABEL[b] for b in BEHAVIOURS]

    # save effect-size matrix table
    mat_out = mat.copy().round(3)
    mat_out.to_csv(C.TBL_DIR / "08_effectsize_matrix.csv")
    mat_out.to_latex(C.TBL_DIR / "08_effectsize_matrix.tex",
                     float_format="%.3f", na_rep="--",
                     caption=("Variance-explained effect size for every predictor "
                              "x behaviour pair (rho^2 for numeric predictors via "
                              "Spearman; epsilon^2 for categorical predictors via "
                              "Kruskal-Wallis). n=%d." % n),
                     label="tab:eff_matrix")

    # ---- notable correlates table (top by effect size) -------------------- #
    notable_cols = ["predictor_label", "behaviour_label", "predictor_kind", "test",
                    "n", "effect_name", "effect", "effect_lo", "effect_hi",
                    "var_explained", "p", "q_value", "survives_fdr"]
    notable = long.sort_values("var_explained", ascending=False).head(15)[notable_cols].copy()
    notable_round = notable.copy()
    for c in ["effect", "effect_lo", "effect_hi", "var_explained"]:
        notable_round[c] = notable_round[c].astype(float).round(3)
    for c in ["p", "q_value"]:
        notable_round[c] = notable_round[c].astype(float).round(4)
    notable_round = notable_round.rename(columns={
        "predictor_label": "Predictor", "behaviour_label": "Behaviour",
        "predictor_kind": "Kind", "test": "Test", "n": "n",
        "effect_name": "Effect", "effect": "Value", "effect_lo": "CI lo",
        "effect_hi": "CI hi", "var_explained": "Var.expl.", "p": "p",
        "q_value": "q (BH)", "survives_fdr": "FDR<.05",
    })
    notable_round.to_csv(C.TBL_DIR / "08_notable_correlates.csv", index=False)
    notable_round.to_latex(C.TBL_DIR / "08_notable_correlates.tex", index=False,
                           float_format="%.3f", na_rep="--",
                           caption=("Fifteen largest predictor x behaviour effect "
                                    "sizes (of %d tests). 'Effect' is Spearman rho "
                                    "(numeric) or epsilon^2 (categorical) with "
                                    "bootstrap 95%% CI; q is Benjamini-Hochberg "
                                    "across the whole family." % n_tests),
                           label="tab:notable")

    # ====================================================================== #
    # Figures
    # ====================================================================== #
    # (1) effect-size heatmap (predictors x behaviours)
    fig, ax = plt.subplots(figsize=P.figsize("FULL", 4.9))
    import seaborn as sns
    hm_vmax = max(0.12, float(np.nanmax(mat.values)))
    hm = sns.heatmap(mat, annot=True, fmt=".2f", cmap=P.SEQ_CMAP,
                     vmin=0, vmax=hm_vmax,
                     linewidths=0.5, linecolor="white",
                     annot_kws={"size": 7.5},
                     cbar_kws={"label": "Variance explained (rho$^2$ / $\\epsilon^2$)"},
                     ax=ax)
    # contrast-aware annotations: rocket_r is dark at high values, so flip the
    # text to white once the cell crosses ~60% of the colour range.
    mat_vals = mat.values.ravel(order="C")
    for txt, val in zip(ax.texts, mat_vals):
        if np.isfinite(val) and val > 0.6 * hm_vmax:
            txt.set_color("white")
        else:
            txt.set_color(P.INK)
    ax.set_xlabel("Energy-conserving behaviour (independent outcomes)")
    ax.set_ylabel("Predictor")
    ax.tick_params(axis="x", rotation=35)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
    fig.tight_layout()
    heatmap_paths = P.save_fig(fig, "08_effectsize_heatmap")

    # identify the single largest categorical and largest numeric effect
    cat_rows = long[long["predictor_kind"] == "categorical"].dropna(subset=["var_explained"])
    num_rows = long[long["predictor_kind"] == "numeric"].dropna(subset=["var_explained"])
    top_cat = cat_rows.sort_values("var_explained", ascending=False).iloc[0]
    top_num = num_rows.sort_values("var_explained", ascending=False).iloc[0]

    # (2) top categorical relationship: box/strip of behaviour across groups
    fig2, ax2 = plt.subplots(figsize=P.figsize("SMALL", 3.1))
    pcode = top_cat["predictor"]; bcode = top_cat["behaviour"]
    plot_df = pd.DataFrame({"grp": cats[pcode], "y": df[bcode]}).dropna()
    order = plot_df.groupby("grp")["y"].median().sort_values().index.tolist()
    sns.boxplot(data=plot_df, x="grp", y="y", order=order, color=P.ACCENT,
                width=0.55, fliersize=0, ax=ax2, boxprops=dict(alpha=0.55))
    sns.stripplot(data=plot_df, x="grp", y="y", order=order, color=P.INK,
                  size=4, alpha=0.6, jitter=0.18, ax=ax2)
    ax2.set_xlabel(PRED_LABEL[pcode])
    ax2.set_ylabel(BEH_LABEL[bcode] + " (ordinal score)")
    ax2.tick_params(axis="x", rotation=20)
    for lab in ax2.get_xticklabels():
        lab.set_ha("right")
    fig2.tight_layout()
    topcat_paths = P.save_fig(fig2, "08_top_categorical")

    # (3) top numeric relationship: scatter on ranks with Spearman annotation
    fig3, ax3 = plt.subplots(figsize=P.figsize("SMALL", 3.0))
    pcode = top_num["predictor"]; bcode = top_num["behaviour"]
    sd = pd.DataFrame({"x": df[pcode], "y": df[bcode]}).dropna()
    jitter = (np.random.default_rng(RNG_SEED).uniform(-0.12, 0.12, len(sd)))
    ax3.scatter(sd["x"], sd["y"] + jitter, color=P.ACCENT, alpha=0.7, s=28,
                edgecolor="white", linewidth=0.5)
    # LOWESS-free trend: median behaviour within area tertiles for visual guidance
    try:
        tert = pd.qcut(sd["x"], 3, duplicates="drop")
        med = sd.groupby(tert)["y"].median()
        centers = sd.groupby(tert)["x"].median()
        ax3.plot(centers.values, med.values, color=P.ACCENT2, marker="o",
                 lw=2, label="tertile median")
        ax3.legend(loc="best")
    except Exception:
        pass
    ax3.set_xlabel(PRED_LABEL[pcode])
    ax3.set_ylabel(BEH_LABEL[bcode] + " (ordinal score, jittered)")
    fig3.tight_layout()
    topnum_paths = P.save_fig(fig3, "08_top_numeric")

    # ====================================================================== #
    # Exploratory ordinal logistic regression (statsmodels OrderedModel)
    # Two best-behaved (most ordinal levels, well-spread) outcomes.
    # Predictors kept few and standardised to avoid separation with n=69.
    # ====================================================================== #
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    ord_rows = []
    ord_models = {
        "eci_metercheck": ["z_total_exp", "z_area_sqft", "university", "renewable"],
        "eci_iron":       ["z_total_exp", "z_area_sqft", "university", "renewable"],
    }
    model_df = pd.DataFrame(index=df.index)
    model_df["z_total_exp"] = _z(df["total_exp"])
    model_df["z_area_sqft"] = _z(df["area_sqft"])
    model_df["university"] = (df["edu_attendance"].eq("University")).astype(float)
    model_df["renewable"] = df["uses_renewable"].astype(float)

    ord_fit_info = {}
    for outcome, preds in ord_models.items():
        d = pd.concat([df[outcome].rename("y"), model_df[preds]], axis=1).dropna()
        y = d["y"].astype(int)
        X = d[preds].astype(float)
        try:
            mod = OrderedModel(y, X, distr="logit")
            res = mod.fit(method="bfgs", disp=False, maxiter=200)
            ll = float(res.llf); ll0 = float(res.llnull)
            mcfadden = 1.0 - ll / ll0 if ll0 != 0 else np.nan
            ord_fit_info[outcome] = {
                "n": int(len(d)), "llf": ll, "llnull": ll0,
                "mcfadden_r2": float(mcfadden), "converged": bool(res.mle_retvals.get("converged", False)),
            }
            params = res.params
            bse = res.bse
            pvals = res.pvalues
            for name in preds:
                if name in params.index:
                    coef = float(params[name])
                    ord_rows.append({
                        "outcome": outcome, "outcome_label": BEH_LABEL[outcome],
                        "term": name, "coef": coef, "odds_ratio": float(np.exp(coef)),
                        "se": float(bse[name]), "z": float(coef / bse[name]) if bse[name] else np.nan,
                        "p": float(pvals[name]), "n": int(len(d)),
                        "mcfadden_r2": float(mcfadden),
                    })
        except Exception as e:
            ord_fit_info[outcome] = {"error": str(e)}

    ord_tab = pd.DataFrame(ord_rows)
    if not ord_tab.empty:
        # BH-FDR within the ordinal-regression family (separate, small family)
        ord_tab["q_value"] = S.bh_fdr(ord_tab["p"].values)
        ord_round = ord_tab.copy()
        for c in ["coef", "odds_ratio", "se", "z", "mcfadden_r2"]:
            ord_round[c] = ord_round[c].astype(float).round(3)
        for c in ["p", "q_value"]:
            ord_round[c] = ord_round[c].astype(float).round(4)
        term_label = {"z_total_exp": "Total exp (z)", "z_area_sqft": "Floor area (z)",
                      "university": "University-educated", "renewable": "Uses renewable"}
        ord_round["term"] = ord_round["term"].map(term_label).fillna(ord_round["term"])
        ord_round["outcome_label"] = ord_round["outcome_label"]
        ord_disp = ord_round[["outcome_label", "term", "coef", "odds_ratio", "se",
                              "z", "p", "q_value", "mcfadden_r2", "n"]].rename(columns={
            "outcome_label": "Outcome", "term": "Term", "coef": "beta",
            "odds_ratio": "OR", "se": "SE", "z": "z", "p": "p",
            "q_value": "q (BH)", "mcfadden_r2": "McFadden R2", "n": "n"})
        ord_disp.to_csv(C.TBL_DIR / "08_ordinal_logit.csv", index=False)
        ord_disp.to_latex(C.TBL_DIR / "08_ordinal_logit.tex", index=False,
                          float_format="%.3f", na_rep="--",
                          caption=("EXPLORATORY proportional-odds ordinal logistic "
                                   "regression for the two best-behaved behaviours. "
                                   "Continuous predictors are z-scored; OR is per 1 SD "
                                   "(continuous) or vs reference (binary). No term is "
                                   "significant. Interpret as hypothesis-generating only."),
                          label="tab:ordlogit")
    else:
        ord_disp = pd.DataFrame()
        pd.DataFrame().to_csv(C.TBL_DIR / "08_ordinal_logit.csv", index=False)
        Path(C.TBL_DIR / "08_ordinal_logit.tex").write_text("% ordinal model did not fit\n")

    # ====================================================================== #
    # Compact JSON summary (LAST stdout line)
    # ====================================================================== #
    # max effect overall + per-behaviour
    per_beh_top = {}
    for b in BEHAVIOURS:
        sub = long[long["behaviour"] == b].dropna(subset=["var_explained"])
        r = sub.sort_values("var_explained", ascending=False).iloc[0]
        per_beh_top[b] = {
            "top_predictor": r["predictor"], "effect_name": r["effect_name"],
            "effect": round(float(r["effect"]), 3),
            "var_explained": round(float(r["var_explained"]), 3),
            "p": round(float(r["p"]), 4), "q": round(float(r["q_value"]), 4),
        }

    summary = {
        "n": n,
        "n_behaviours": len(BEHAVIOURS),
        "n_predictors": len(PRED_LABEL),
        "n_tests": n_tests,
        "n_sig_raw_p<.05": n_raw,
        "n_survive_BH_FDR<.05": n_fdr,
        "min_q_value": round(float(long["q_value"].min()), 4),
        "min_p_value": round(float(long["p"].min()), 4),
        "max_var_explained": round(float(np.nanmax(mat.values)), 3),
        "median_var_explained": round(float(np.nanmedian(long["var_explained"])), 4),
        "top_categorical": {
            "predictor": top_cat["predictor"], "behaviour": top_cat["behaviour"],
            "eps2": round(float(top_cat["effect"]), 3),
            "p": round(float(top_cat["p"]), 4), "q": round(float(top_cat["q_value"]), 4),
        },
        "top_numeric": {
            "predictor": top_num["predictor"], "behaviour": top_num["behaviour"],
            "rho": round(float(top_num["effect"]), 3),
            "ci": [round(float(top_num["effect_lo"]), 3), round(float(top_num["effect_hi"]), 3)],
            "p": round(float(top_num["p"]), 4), "q": round(float(top_num["q_value"]), 4),
        },
        "per_behaviour_top": per_beh_top,
        "ordinal_logit": {k: ({kk: (round(vv, 3) if isinstance(vv, float) else vv)
                               for kk, vv in v.items()}) for k, v in ord_fit_info.items()},
        "figures": {
            "heatmap": heatmap_paths,
            "top_categorical": topcat_paths,
            "top_numeric": topnum_paths,
        },
        "tables": {
            "notable_correlates": [str(C.TBL_DIR / "08_notable_correlates.csv"),
                                   str(C.TBL_DIR / "08_notable_correlates.tex")],
            "effectsize_matrix": [str(C.TBL_DIR / "08_effectsize_matrix.csv"),
                                  str(C.TBL_DIR / "08_effectsize_matrix.tex")],
            "ordinal_logit": [str(C.TBL_DIR / "08_ordinal_logit.csv"),
                              str(C.TBL_DIR / "08_ordinal_logit.tex")],
        },
    }
    print("\n=== effect-size matrix (variance explained) ===")
    print(mat.round(3).to_string())
    print("\n=== top correlates ===")
    print(notable_round.head(10).to_string(index=False))
    if not ord_disp.empty:
        print("\n=== exploratory ordinal logit ===")
        print(ord_disp.to_string(index=False))
    print("\nFDR family: %d tests, %d raw p<.05, %d survive BH-FDR (min q=%.3f)"
          % (n_tests, n_raw, n_fdr, float(long["q_value"].min())))
    print(json.dumps(summary))


def _z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.mean()) / s.std(ddof=0)


def _boot_eps2(sub: pd.DataFrame, n_boot=2000, seed=0):
    """Percentile bootstrap CI for Kruskal-Wallis epsilon-squared.

    Resamples respondents (rows of the merged group/value frame) with replacement,
    preserving each respondent's (group, value) pair.
    """
    rng = np.random.default_rng(seed)
    vals = sub["y"].values.astype(float)
    grp = sub["g"].values
    idx_all = np.arange(len(sub))
    boots = []
    for _ in range(n_boot):
        take = rng.integers(0, len(sub), len(sub))
        g2 = grp[take]; v2 = vals[take]
        groups = [v2[g2 == lev] for lev in np.unique(g2)]
        groups = [x for x in groups if len(x) > 0]
        if len(groups) < 2:
            continue
        kw = S.kruskal_effect(*groups)
        if not np.isnan(kw["eps2"]):
            boots.append(kw["eps2"])
    if len(boots) < 10:
        return (np.nan, np.nan)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return (float(lo), float(hi))


if __name__ == "__main__":
    main()
