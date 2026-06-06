"""05 -- The five energy-conserving behaviours individually, and the
reliability / multidimensionality story (KEY SECTION).

This script deliberately interrogates whether the five behavioural survey items
form a single "energy-consciousness" trait. They do NOT: Cronbach alpha is near
zero, the inter-item Spearman matrix is essentially flat, and the PCA spectrum
of the standardised items shows no dominant first component. We therefore frame
the composite Energy-Consciousness Index (ECI) as a descriptive summary only and
argue that the five behaviours must be analysed as INDEPENDENT outcomes.

The survey measured NO electricity consumption (no kWh, no bill amount). These
are five self-reported conserving behaviours; "ECI" is a convenience composite,
not a validated psychometric scale.

Figures (prefix 05_):
    05_behaviour_distributions  five-behaviour distribution panel (raw + ordinal)
    05_interitem_heatmap        inter-item Spearman correlation heatmap
    05_pca_scree                PCA scree on standardised items (flat spectrum)
    05_eci_histogram            ECI histogram coloured by tertile band

Tables (prefix 05_, each as .csv AND .tex):
    05_eci_rubric               scoring rubric (item, category, score, max)
    05_reliability              alpha, item-rest r, alpha-if-deleted per item
    05_interitem_spearman       inter-item Spearman matrix
    05_behaviour_summary        per-behaviour ordinal score summary + bootstrap CI

Run:  cd .../code && uv run python 05_behaviours.py
"""
from __future__ import annotations
import sys, json, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from lib import io, config as C, stats as S, plotting as P

RNG_SEED = 12345
N_BOOT = 10000

# Short, paper-friendly labels for each behaviour item.
ITEM_SHORT = {
    "eci_billknow": "Bill-calc literacy",
    "eci_iron": "Ironing efficiency",
    "eci_nightlight": "Night-light restraint",
    "eci_metercheck": "Meter-checking",
    "eci_enrating": "Energy-rating check",
}
ITEMS = list(C.ECI_ITEMS.keys())            # ordinal score columns
NORM = [i + "_norm" for i in ITEMS]         # [0,1]-normalised columns

# Human-readable text for each ordinal level, used in the rubric table.
LEVEL_TEXT = {
    "eci_billknow": {0: "Not aware how bill is calculated",
                     1: "Learnt via media / other means",
                     2: "Learnt from bill / family / utility"},
    "eci_iron": {0: "Iron daily", 1: "Iron when need arises",
                 2: "Iron twice a week", 3: "Iron weekly", 4: "Do not iron"},
    "eci_nightlight": {0: "More than two night-lights on",
                       1: "Fewer than two night-lights on",
                       2: "No lights on while sleeping"},
    "eci_metercheck": {0: "Never check meter", 1: "Rarely (once a year)",
                       2: "Some months", 3: "Every month"},
    "eci_enrating": {0: "Does not check energy rating",
                     1: "Checks energy rating when buying"},
}


def build_rubric_table(df: pd.DataFrame) -> pd.DataFrame:
    """Scoring rubric: one row per (item, ordinal level) with score, max, count."""
    rows = []
    for item, (rawcode, scorer, mx) in C.ECI_ITEMS.items():
        score_counts = df[item].value_counts()
        for level in sorted(LEVEL_TEXT[item]):
            rows.append({
                "Behaviour": ITEM_SHORT[item],
                "Category (scored meaning)": LEVEL_TEXT[item][level],
                "Score": int(level),
                "Max": int(mx),
                "n": int(score_counts.get(level, 0)),
            })
    return pd.DataFrame(rows)


def build_reliability_table(eci: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Cronbach alpha overall + per-item item-rest r and alpha-if-deleted.

    Uses the [0,1]-normalised items so each behaviour contributes equally,
    matching how the composite ECI is built (mean of normalised items).
    """
    alpha_overall = S.cronbach_alpha(eci[NORM])
    rows = []
    for item, c in zip(ITEMS, NORM):
        rest_cols = [x for x in NORM if x != c]
        rest = eci[rest_cols].sum(axis=1)
        # corrected item-rest (item-total) correlation
        r_rest = float(np.corrcoef(eci[c], rest)[0, 1])
        a_del = S.cronbach_alpha(eci[rest_cols])
        rows.append({
            "Behaviour": ITEM_SHORT[item],
            "Item-rest r": round(r_rest, 3),
            "Alpha if deleted": round(a_del, 3),
        })
    tbl = pd.DataFrame(rows)
    return tbl, float(alpha_overall)


def main():
    df = io.load_clean()
    eci = io.load_eci()
    n = len(df)

    # ----------------------------------------------------------------- #
    # 1. Reliability + multidimensionality statistics
    # ----------------------------------------------------------------- #
    rel_tbl, alpha_overall = build_reliability_table(eci)

    # Inter-item Spearman matrix on the ordinal scores.
    spear = eci[ITEMS].corr(method="spearman")
    spear_lbl = spear.rename(index=ITEM_SHORT, columns=ITEM_SHORT)

    # Off-diagonal summary + which pair is strongest, with bootstrap CI + FDR.
    pairs, pair_r, pair_p = [], [], []
    for a, b in itertools.combinations(ITEMS, 2):
        res = S.spearman_with_ci(eci[a], eci[b], n_boot=N_BOOT, seed=RNG_SEED)
        pairs.append((ITEM_SHORT[a], ITEM_SHORT[b]))
        pair_r.append(res["rho"])
        pair_p.append(res["p"])
    pair_q = S.bh_fdr(pair_p)
    pair_df = pd.DataFrame({
        "Pair A": [p[0] for p in pairs],
        "Pair B": [p[1] for p in pairs],
        "Spearman r": np.round(pair_r, 3),
        "p": np.round(pair_p, 4),
        "q (BH-FDR)": np.round(pair_q, 4),
    }).sort_values("Spearman r", key=lambda s: s.abs(), ascending=False)

    abs_off = np.abs(pair_r)
    max_idx = int(np.argmax(abs_off))
    max_pair = pairs[max_idx]
    max_r = float(pair_r[max_idx])
    # bootstrap CI for the strongest pair's rho
    a_code = ITEMS[[ITEM_SHORT[i] for i in ITEMS].index(max_pair[0])]
    b_code = ITEMS[[ITEM_SHORT[i] for i in ITEMS].index(max_pair[1])]
    max_ci = S.spearman_with_ci(eci[a_code], eci[b_code], n_boot=N_BOOT, seed=RNG_SEED)
    mean_abs_off = float(np.mean(abs_off))
    n_sig_fdr = int(np.nansum(pair_q < 0.05))

    # PCA scree on standardised ordinal items (correlation-matrix eigenvalues).
    X = eci[ITEMS].astype(float).values
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    pca = PCA().fit(X)
    eig = pca.explained_variance_              # eigenvalues
    evr = pca.explained_variance_ratio_
    n_kaiser = int((eig > 1.0).sum())
    pc1_pct = float(evr[0] * 100)

    # ----------------------------------------------------------------- #
    # 2. Per-behaviour ordinal summary + bootstrap CI on normalised mean
    # ----------------------------------------------------------------- #
    summ_rows = []
    for item, c in zip(ITEMS, NORM):
        mx = C.ECI_ITEMS[item][2]
        mean_raw, lo_raw, hi_raw = S.bootstrap_ci(
            eci[item], np.mean, n_boot=N_BOOT, seed=RNG_SEED)
        mean_norm = float(eci[c].mean())
        summ_rows.append({
            "Behaviour": ITEM_SHORT[item],
            "Levels": mx + 1,
            "Max score": mx,
            "Mean score": round(mean_raw, 3),
            "Mean score 95% CI": f"[{lo_raw:.2f}, {hi_raw:.2f}]",
            "Mean (norm 0-1)": round(mean_norm, 3),
            "% at ceiling": round(100 * float((eci[item] == mx).mean()), 1),
            "% at floor": round(100 * float((eci[item] == 0).mean()), 1),
        })
    behav_summary = pd.DataFrame(summ_rows)

    # ECI composite description + band counts + bootstrap CI on mean.
    eci_mean, eci_lo, eci_hi = S.bootstrap_ci(eci["eci"], np.mean,
                                              n_boot=N_BOOT, seed=RNG_SEED)
    band_counts = (eci["eci_band"].value_counts()
                   .reindex(["Low", "Medium", "High"]).fillna(0).astype(int))

    # ================================================================= #
    # FIGURES
    # ================================================================= #
    figpaths = {}

    # --- Fig 1: five-behaviour distribution panel (ordinal scores) ---------- #
    fig, axes = plt.subplots(2, 3, figsize=P.figsize("FULL", 4.4))
    axes = axes.ravel()
    for ax, item in zip(axes, ITEMS):
        mx = C.ECI_ITEMS[item][2]
        counts = (eci[item].value_counts()
                  .reindex(range(mx + 1)).fillna(0).astype(int))
        bars = ax.bar(counts.index, counts.values, color=P.ACCENT,
                      edgecolor="white", linewidth=0.6)
        ax.set_title(ITEM_SHORT[item])
        ax.set_xlabel("Ordinal score")
        ax.set_ylabel("Households")
        ax.set_xticks(range(mx + 1))
        ax.set_ylim(0, max(counts.values) * 1.18 + 1)
        for b, v in zip(bars, counts.values):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.4, str(int(v)),
                        ha="center", va="bottom", fontsize=P.ANNOT)
    # 6th panel: composite ECI per-item normalised means with bootstrap CIs
    ax = axes[5]
    means = [eci[c].mean() for c in NORM]
    los, his = [], []
    for c in NORM:
        _, lo, hi = S.bootstrap_ci(eci[c], np.mean, n_boot=N_BOOT, seed=RNG_SEED)
        los.append(lo); his.append(hi)
    ypos = np.arange(len(ITEMS))
    means = np.array(means)
    err = np.vstack([means - np.array(los), np.array(his) - means])
    ax.errorbar(means, ypos, xerr=err, fmt="o", color=P.ACCENT2,
                capsize=3, markersize=6, lw=1.4)
    ax.set_yticks(ypos)
    ax.set_yticklabels([ITEM_SHORT[i] for i in ITEMS])
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("Normalised mean (0-1)")
    ax.set_title("Per-item conservation level")
    fig.tight_layout()
    figpaths["behaviour_distributions"] = P.save_fig(fig, "05_behaviour_distributions")

    # --- Fig 2: inter-item Spearman correlation heatmap --------------------- #
    fig, ax = plt.subplots(figsize=P.figsize("FULL", 5.4))
    M = spear_lbl.values
    im = ax.imshow(M, cmap=P.DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(ITEMS)))
    ax.set_yticks(range(len(ITEMS)))
    ax.set_xticklabels([ITEM_SHORT[i] for i in ITEMS], rotation=35, ha="right")
    ax.set_yticklabels([ITEM_SHORT[i] for i in ITEMS])
    for i in range(len(ITEMS)):
        for j in range(len(ITEMS)):
            val = M[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=P.ANNOT_SMALL,
                    color="white" if abs(val) > 0.6 else P.INK,
                    fontweight="bold" if i != j else "normal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman rho")
    fig.tight_layout()
    figpaths["interitem_heatmap"] = P.save_fig(fig, "05_interitem_heatmap")

    # --- Fig 3: PCA scree (flat spectrum) ----------------------------------- #
    fig, ax = plt.subplots(figsize=P.figsize("SMALL", 3.3))
    comps = np.arange(1, len(eig) + 1)
    ax.plot(comps, eig, "o-", color=P.ACCENT, lw=1.8, markersize=8,
            label="Observed eigenvalue")
    ax.axhline(1.0, color=P.RULE, ls="--", lw=1.1, label="Kaiser criterion (=1)")
    ax.axhline(1.0, color=P.ACCENT2, ls=":", lw=1.4,
               label="Sphericity (uniform = 1.0)")
    for x, y in zip(comps, eig):
        ax.text(x, y + 0.04, f"{y:.2f}\n({evr[list(comps).index(x)]*100:.0f}%)",
                ha="center", va="bottom", fontsize=P.ANNOT)
    ax.set_xticks(comps)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Eigenvalue (of correlation matrix)")
    ax.set_ylim(0, max(eig) * 1.22)
    ax.legend(loc="upper right")
    fig.tight_layout()
    figpaths["pca_scree"] = P.save_fig(fig, "05_pca_scree")

    # --- Fig 4: ECI histogram coloured by band ------------------------------ #
    fig, ax = plt.subplots(figsize=P.figsize("FULL", 4.6))
    bins = np.linspace(eci["eci"].min(), eci["eci"].max(), 13)
    for band in ["Low", "Medium", "High"]:
        sub = eci.loc[eci["eci_band"] == band, "eci"]
        ax.hist(sub, bins=bins, color=P.BAND_COLORS[band], alpha=0.85,
                edgecolor="white", linewidth=0.6,
                label=f"{band} (n={int((eci['eci_band']==band).sum())})")
    ax.axvline(eci_mean, color=P.RULE, ls="--", lw=1.3,
               label=f"Mean = {eci_mean:.1f}")
    ax.set_xlabel(C.LABEL["eci"])
    ax.set_ylabel("Households")
    ax.legend(title="Tertile band")
    fig.tight_layout()
    figpaths["eci_histogram"] = P.save_fig(fig, "05_eci_histogram")

    # ================================================================= #
    # TABLES (csv + LaTeX)
    # ================================================================= #
    tblpaths = {}

    def save_table(dfp: pd.DataFrame, name: str, caption: str, index=False):
        csv_p = C.TBL_DIR / f"{name}.csv"
        tex_p = C.TBL_DIR / f"{name}.tex"
        dfp.to_csv(csv_p, index=index)
        dfp.to_latex(tex_p, index=index, escape=True, caption=caption,
                     label=f"tab:{name}", longtable=False)
        tblpaths[name] = [str(csv_p), str(tex_p)]

    rubric_tbl = build_rubric_table(df)
    save_table(rubric_tbl, "05_eci_rubric",
               "ECI scoring rubric: the five energy-conserving behaviours, the "
               "ordinal score assigned to each response category (higher = more "
               "conserving), the item ceiling, and the observed household count "
               f"(N={n}).")

    # Reliability table: append an overall-alpha summary row.
    rel_out = rel_tbl.copy()
    rel_out = pd.concat([
        rel_out,
        pd.DataFrame([{"Behaviour": "OVERALL (5 items)",
                       "Item-rest r": np.nan,
                       "Alpha if deleted": round(alpha_overall, 3)}]),
    ], ignore_index=True)
    save_table(rel_out, "05_reliability",
               "Internal-consistency diagnostics for the five-behaviour ECI "
               f"(N={n}). Cronbach alpha = {alpha_overall:.3f}. Item-rest "
               "correlations are near zero or negative and deleting items does "
               "not recover reliability, indicating the behaviours do not form a "
               "single scale.")

    spear_out = spear_lbl.round(3).reset_index().rename(columns={"index": ""})
    save_table(spear_out, "05_interitem_spearman",
               "Inter-item Spearman rank correlations among the five "
               f"behaviours (N={n}). The strongest association is "
               f"{max_pair[0]}-{max_pair[1]} (rho = {max_r:.2f}); the mean "
               f"absolute off-diagonal correlation is {mean_abs_off:.2f}.")

    save_table(behav_summary, "05_behaviour_summary",
               "Per-behaviour ordinal score summary with bias-corrected "
               f"bootstrap 95% CIs ({N_BOOT:,} resamples) for the mean score "
               f"(N={n}).")

    # Pairwise correlation + FDR table (extra, supports the heatmap).
    save_table(pair_df, "05_interitem_pairs_fdr",
               "Pairwise Spearman correlations among the five behaviours with "
               f"Benjamini-Hochberg FDR-adjusted q-values across all 10 pairs "
               f"(N={n}).")

    # ================================================================= #
    # COMPACT JSON FINDINGS (last stdout line)
    # ================================================================= #
    findings = {
        "n": n,
        "n_behaviours": len(ITEMS),
        "measured_consumption": False,
        "cronbach_alpha_overall": round(alpha_overall, 4),
        "alpha_if_deleted": {ITEM_SHORT[i]: round(
            S.cronbach_alpha(eci[[x for x in NORM if x != c]]), 3)
            for i, c in zip(ITEMS, NORM)},
        "item_rest_r": {r["Behaviour"]: r["Item-rest r"]
                        for _, r in rel_tbl.iterrows()},
        "interitem_spearman_max": round(max_r, 3),
        "interitem_spearman_max_pair": list(max_pair),
        "interitem_spearman_max_ci": [round(max_ci["lo"], 3), round(max_ci["hi"], 3)],
        "interitem_mean_abs_off_diag": round(mean_abs_off, 3),
        "n_pairs_sig_after_fdr": n_sig_fdr,
        "pca_eigenvalues": [round(float(e), 3) for e in eig],
        "pca_explained_var_ratio": [round(float(e), 3) for e in evr],
        "pca_pc1_pct": round(pc1_pct, 1),
        "pca_n_eigen_gt1": n_kaiser,
        "eci_mean": round(float(eci_mean), 2),
        "eci_mean_ci": [round(eci_lo, 2), round(eci_hi, 2)],
        "eci_sd": round(float(eci["eci"].std()), 2),
        "eci_min": round(float(eci["eci"].min()), 2),
        "eci_max": round(float(eci["eci"].max()), 2),
        "eci_band_counts": {k: int(v) for k, v in band_counts.items()},
        "behaviour_norm_means": {ITEM_SHORT[i]: round(float(eci[c].mean()), 3)
                                 for i, c in zip(ITEMS, NORM)},
        "conclusion": ("No single energy-consciousness trait exists; alpha approx 0, "
                       "flat inter-item correlations, no dominant PCA component. "
                       "The five behaviours are analysed as independent outcomes."),
        "figures": {k: v for k, v in figpaths.items()},
        "tables": tblpaths,
    }
    print("\n" + json.dumps(findings, default=str))


if __name__ == "__main__":
    main()
