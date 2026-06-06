"""07 -- Multivariate household typologies.

Exploratory, small-n (N=70), convenience sample. We do NOT measure electricity
consumption; the five energy-conserving behaviours are independent (alpha=0.04),
so the ECI here is used only as a coarse descriptive summary, NOT as a reliable
scale.

Pipeline
--------
1. Assemble a compact set of structural/contextual categorical features
   (dwelling fabric, energy system, demographics, geography). Rare levels are
   collapsed to "Other" so a single respondent cannot define a cluster.
2. Dimension reduction: MCA (prince) on the categoricals for a 2-D map; FAMD
   (prince) on the mixed set (categoricals + a few numerics) for a sensitivity
   view.
3. Cluster households two ways and compare:
     (a) Gower distance (gower.gower_matrix) on the mixed feature set +
         AgglomerativeClustering (average linkage, precomputed).
     (b) k-modes (kmodes) on the categoricals.
   k is chosen by silhouette (on the Gower distance) over k=2..5, balanced
   against interpretability.
4. Profile clusters: size, dominant category per feature, mean behaviour-item
   scores and mean ECI, with bootstrap 95% CIs for ECI, plus exploratory
   Kruskal/Cramer effect sizes (BH-FDR corrected) for cluster vs. attributes.

Figures (07_): combined MCA + FAMD scatter coloured by cluster (one shared
legend); silhouette vs k; cluster-profile heatmap (standardised attribute
prevalences).
Tables  (07_): cluster profiles; attribute-vs-cluster effect sizes.

Run:  cd code && uv run python 07_typologies.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from lib import io, config as C, stats as S, plotting as P

import prince
import gower
from kmodes.kmodes import KModes
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

SEED = 0
np.random.seed(SEED)


# --------------------------------------------------------------------------- #
# 1. Feature assembly
# --------------------------------------------------------------------------- #
def collapse(series: pd.Series, keep: list[str], other: str = "Other") -> pd.Series:
    """Keep listed levels, fold everything else into a single 'Other' level."""
    return series.where(series.isin(keep), other).astype(str)


def build_features(df: pd.DataFrame):
    """Return (cat_df, num_df, mixed_df) of analysis features with tidy labels.

    Rare categories are collapsed so no cluster can be defined by one household.
    Numerics retained for FAMD/Gower are floor area and total expenditure.
    """
    f = pd.DataFrame(index=df.index)

    # --- dwelling fabric --------------------------------------------------- #
    f["Build period"] = collapse(
        df["build_period"],
        ["1990 - 1999", "2000 - 2009", "2010 - 2019"],
    ).replace({"Other": "<=1989 / >=2020"})
    f["House type"] = collapse(
        df["house_type"],
        ["Single House -Double Floor", "Single House - Single Floor",
         "Single House - Above double floor"],
    ).replace({
        "Single House -Double Floor": "Double floor",
        "Single House - Single Floor": "Single floor",
        "Single House - Above double floor": "3+ floors",
        "Other": "Flat/Shanty",
    })
    f["Wall material"] = df["wall_material"].astype(str)            # Brick/Cement Block/Cabook
    f["Roof type"] = df["roof_type"].astype(str)                   # Asbestos/Tile/Concrete
    f["Architect-designed"] = np.where(
        df["arch_design"].str.contains("certified architect", case=False, na=False),
        "Yes", "No",
    )

    # --- energy system ----------------------------------------------------- #
    f["Renewable use"] = np.where(df["uses_renewable"] == 1, "Yes", "No")
    f["Pro wiring"] = collapse(df["wiring_pro"], ["Yes", "No"]).replace(
        {"Other": "Unaware"})
    f["Red notice (yr)"] = np.where(df["any_red_notice"] == 1, "Yes", "No")
    f["Bill rounding"] = np.where(
        df["pay_practice"].str.contains("rounded-off amount which will cover",
                                        case=False, na=False),
        "Rounds up (covers bill)", "Exact / under / partial",
    )

    # --- demographics / geography ----------------------------------------- #
    f["Gender"] = df["gender"].astype(str)
    f["University-educated"] = np.where(
        df["edu_attendance"].str.contains("University", case=False, na=False),
        "Yes", "No",
    )
    f["Relationship to head"] = np.where(
        df["rel_head"].str.contains("Son/daughter", case=False, na=False),
        "Son/daughter", "Head / spouse / other",
    )
    f["Province"] = collapse(df["province"], ["Western", "Southern", "North Western"]
                             ).replace({"Other": "Other province"})

    cat_cols = list(f.columns)

    # --- numerics for FAMD / Gower (median-imputed so distances are defined) #
    num = pd.DataFrame(index=df.index)
    num["Floor area (sqft)"] = pd.to_numeric(df["area_sqft"], errors="coerce")
    num["Total exp (Rs/mo)"] = pd.to_numeric(df["total_exp"], errors="coerce")
    num = num.fillna(num.median())
    num_cols = list(num.columns)

    mixed = pd.concat([f, num], axis=1)
    return f, num, mixed, cat_cols, num_cols


# --------------------------------------------------------------------------- #
# 2. Cluster-selection diagnostics
# --------------------------------------------------------------------------- #
def gower_distance(mixed: pd.DataFrame, cat_cols) -> np.ndarray:
    cat_mask = np.array([c in cat_cols for c in mixed.columns])
    D = gower.gower_matrix(mixed, cat_features=cat_mask)
    D = np.asarray(D, dtype=float)
    D = (D + D.T) / 2.0          # enforce exact symmetry
    np.fill_diagonal(D, 0.0)
    return D


def choose_k(D: np.ndarray, ks=(2, 3, 4, 5), linkage="complete"):
    """Silhouette (precomputed Gower) for agglomerative clustering.

    Default linkage is COMPLETE: average/single linkage on this Gower matrix
    merely isolate 1-2 outliers (chaining), giving uninterpretable 67-vs-2
    splits; complete linkage yields balanced, profileable groups. We log all
    linkages for transparency in `linkage_scan`.
    """
    rows = []
    labels_by_k = {}
    for k in ks:
        model = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage=linkage)
        labels = model.fit_predict(D)
        sil = float(silhouette_score(D, labels, metric="precomputed"))
        sizes = pd.Series(labels).value_counts().sort_index().tolist()
        rows.append({"k": k, "silhouette": sil,
                     "min_cluster_size": int(min(sizes)), "sizes": sizes})
        labels_by_k[k] = labels
    return pd.DataFrame(rows), labels_by_k


def linkage_scan(D: np.ndarray, ks=(2, 3, 4, 5)):
    """Transparency: silhouette + min cluster size for every linkage x k.

    Documents WHY complete linkage is chosen over average/single (which produce
    degenerate outlier-isolating partitions on this Gower matrix)."""
    rows = []
    for link in ("complete", "average", "single"):
        for k in ks:
            lab = AgglomerativeClustering(
                n_clusters=k, metric="precomputed", linkage=link).fit_predict(D)
            sizes = pd.Series(lab).value_counts().sort_index().tolist()
            sil = (float(silhouette_score(D, lab, metric="precomputed"))
                   if len(set(lab)) > 1 else np.nan)
            rows.append({"linkage": link, "k": k, "silhouette": round(sil, 3),
                         "min_cluster_size": int(min(sizes)), "sizes": sizes})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. Profiling
# --------------------------------------------------------------------------- #
def dominant(series: pd.Series):
    vc = series.value_counts()
    lvl = vc.index[0]
    return f"{lvl} ({vc.iloc[0]}/{vc.sum()}, {100*vc.iloc[0]/vc.sum():.0f}%)"


def profile_clusters(df, f, num, labels, cat_cols, num_cols, label_names):
    """Build a tidy cluster-profile table."""
    rows = []
    eci = df["eci"].values
    items = ["eci_billknow", "eci_iron", "eci_nightlight",
             "eci_metercheck", "eci_enrating"]
    for cl in sorted(np.unique(labels)):
        m = labels == cl
        rec = {"Cluster": label_names[cl], "n": int(m.sum())}
        for c in cat_cols:
            rec[c] = dominant(f.loc[m, c])
        for c in num_cols:
            rec[c] = f"{num.loc[m, c].median():.0f}"
        est, lo, hi = S.bootstrap_ci(eci[m], np.mean, n_boot=10000, seed=SEED)
        rec["ECI mean"] = round(float(est), 1)
        rec["ECI 95% CI"] = f"[{lo:.1f}, {hi:.1f}]"
        for it in items:
            rec[it] = round(float(df.loc[m, it].mean()), 2)
        rows.append(rec)
    return pd.DataFrame(rows)


def attribute_effects(df, f, labels, cat_cols):
    """Exploratory cluster-vs-attribute association (Cramer's V) + cluster-vs-ECI
    and cluster-vs-items (Kruskal eps^2). BH-FDR across the whole family."""
    lab = pd.Series(labels, index=df.index)
    rows = []
    for c in cat_cols:
        a = S.assoc_categorical(f[c], lab)
        rows.append({"attribute": c, "kind": "categorical",
                     "effect": "Cramer's V", "value": a["cramers_v"],
                     "test": a["test"], "p": a["p"]})
    # cluster vs ECI + each behaviour item
    targets = {"ECI (0-100)": df["eci"]}
    for it, lbl in [("eci_billknow", "Bill literacy"), ("eci_iron", "Ironing"),
                    ("eci_nightlight", "Night-light"),
                    ("eci_metercheck", "Meter-check"),
                    ("eci_enrating", "Energy-rating")]:
        targets[lbl] = df[it]
    for name, y in targets.items():
        groups = [y[lab == cl].values for cl in sorted(lab.unique())]
        ke = S.kruskal_effect(*groups)
        rows.append({"attribute": name, "kind": "ordinal",
                     "effect": "epsilon^2", "value": ke["eps2"],
                     "test": "kruskal", "p": ke["p"]})
    out = pd.DataFrame(rows)
    out["q_bh"] = S.bh_fdr(out["p"].values)
    return out.sort_values("p", na_position="last").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4. Figures
# --------------------------------------------------------------------------- #
def fig_silhouette(diag_g, diag_modes, saved):
    fig, ax = plt.subplots(figsize=P.figsize("SMALL", 2.5))
    ax.plot(diag_g["k"], diag_g["silhouette"], "-o", color=P.ACCENT,
            label="Agglomerative complete (Gower)")
    if diag_modes is not None:
        ax.plot(diag_modes["k"], diag_modes["silhouette"], "-s",
                color=P.ACCENT2, label="k-modes (Gower-eval)")
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Mean silhouette (Gower)")
    ax.set_xticks(list(diag_g["k"]))
    ax.legend()
    for _, r in diag_g.iterrows():
        ax.annotate(f"{r['silhouette']:.2f}", (r["k"], r["silhouette"]),
                    textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=P.ANNOT, color=P.ACCENT)
    saved["07_silhouette_vs_k"] = P.save_fig(fig, "07_silhouette_vs_k")


def fig_cluster_maps(panels, labels, label_names, name, saved):
    """Combined MCA + FAMD cluster maps in one figure with a shared legend.

    Authored at the FULL canonical width (not the old HALF-width pair). The
    descriptive cluster names are long, and as per-panel legends they blew the
    saved width past 5 in, so LaTeX over-shrank each half-width panel (and every
    font with it). Drawing both projections as subplots of a single full-width
    figure, with one shared legend spanning the bottom, keeps the on-page text
    at the paper's normal size and gives the long labels the full measure.

    ``panels`` is a list of ``(title, coords, inertia)``; a panel whose
    ``coords`` is ``None`` renders an "unavailable" placeholder so the paper
    figure stays intact even if a projection could not be computed.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=P.figsize("FULL", 3.5))
    axes = np.atleast_1d(axes)
    handles = handle_labels = None
    for ax, (title, coords, inertia) in zip(axes, panels):
        if coords is None:
            ax.text(0.5, 0.5, f"{title} unavailable", ha="center",
                    va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        for cl in sorted(np.unique(labels)):
            m = labels == cl
            ax.scatter(coords[m, 0], coords[m, 1], s=42,
                       color=P.CLUST_PAL[cl % len(P.CLUST_PAL)],
                       edgecolor="white", linewidth=0.5,
                       label=f"{label_names[cl]} (n={int(m.sum())})", alpha=0.9)
            # centroid
            ax.scatter(coords[m, 0].mean(), coords[m, 1].mean(), marker="X",
                       s=150, color=P.CLUST_PAL[cl % len(P.CLUST_PAL)],
                       edgecolor=P.INK, linewidth=1.0, zorder=5)
        ax.axhline(0, color=P.MUTED, lw=0.8, zorder=0)
        ax.axvline(0, color=P.MUTED, lw=0.8, zorder=0)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(f"Dim 1 ({inertia[0]:.1f}% inertia)", fontsize=11)
        ax.set_ylabel(f"Dim 2 ({inertia[1]:.1f}% inertia)", fontsize=11)
        ax.tick_params(labelsize=10)
        if handles is None:
            handles, handle_labels = ax.get_legend_handles_labels()
    fig.tight_layout()
    # single shared legend below both panels: the full measure fits the long
    # descriptive names that the old half-width panels could not.
    fig.legend(handles, handle_labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=1, frameon=False,
               handletextpad=0.4, fontsize=10)
    saved[name] = P.save_fig(fig, name)


def fig_profile_heatmap(df, f, labels, cat_cols, label_names, saved):
    """Heatmap of within-cluster prevalence of each (attribute=level) indicator,
    standardised against the overall prevalence (prevalence - overall)."""
    dummies = pd.get_dummies(f[cat_cols])
    overall = dummies.mean(axis=0)
    # keep informative levels (drive the contrast); drop near-constant ones
    keep = overall[(overall > 0.05) & (overall < 0.95)].index.tolist()
    dummies = dummies[keep]
    overall = overall[keep]
    rows, idx = [], []
    for cl in sorted(np.unique(labels)):
        m = labels == cl
        rows.append(dummies.loc[m].mean(axis=0) - overall)
        # short y-label: the full descriptive cluster name (up to ~75 chars)
        # would crush the matrix width, so key the rows by their ID + size and
        # leave the full names to the profile table / MCA-FAMD legends.
        cid = label_names[cl].split(":", 1)[0]          # "C1", "C2", ...
        idx.append(f"{cid} (n={int(m.sum())})")
    M = pd.DataFrame(rows, index=idx)
    # order columns by spread across clusters for readability
    M = M[M.abs().max(axis=0).sort_values(ascending=False).index]
    # Cap to the most informative columns (largest absolute spread across
    # clusters) so the ~40 dummy labels do not smear into illegibility; the
    # dropped attributes have low cross-cluster deviation and add little signal.
    n_show = min(16, M.shape[1])
    if M.shape[1] > n_show:
        print(f"  [heatmap] showing top {n_show} of {M.shape[1]} attribute=level "
              f"columns by cross-cluster spread (lower-spread ones dropped for legibility)")
    M = M.iloc[:, :n_show]

    vmax = 0.6
    fig, ax = plt.subplots(figsize=P.figsize("FULL", 4.6))
    im = ax.imshow(M.values, aspect="auto", cmap=P.DIVERGING,
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(M.shape[1]))
    # "attribute_level" -> "attribute: level" (get_dummies joins with "_")
    ax.set_xticklabels([c.replace("_", ": ", 1) for c in M.columns],
                       rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(M.index)
    ax.tick_params(axis="y", length=0)
    # contrast-aware cell annotations (white on saturated cells, black on light)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M.values[i, j]
            txt_color = "white" if abs(val) > 0.6 * vmax else P.INK
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=6.5, color=txt_color)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("prevalence - overall")
    fig.tight_layout()
    saved["07_cluster_profile_heatmap"] = P.save_fig(fig, "07_cluster_profile_heatmap")
    return M


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    df = io.load_clean()
    f, num, mixed, cat_cols, num_cols = build_features(df)

    # ---- Gower distance + k selection ------------------------------------ #
    D = gower_distance(mixed, cat_cols)
    scan = linkage_scan(D, ks=(2, 3, 4, 5))   # transparency over linkages
    diag_g, labels_by_k = choose_k(D, ks=(2, 3, 4, 5), linkage="complete")

    # ---- k-modes (categoricals) for comparison; evaluate on Gower -------- #
    Xc = f[cat_cols].astype(str).values
    modes_rows, modes_labels = [], {}
    for k in (2, 3, 4, 5):
        km = KModes(n_clusters=k, init="Huang", n_init=20, random_state=SEED,
                    verbose=0)
        lk = km.fit_predict(Xc)
        modes_labels[k] = lk
        sil = float(silhouette_score(D, lk, metric="precomputed")) \
            if len(np.unique(lk)) > 1 else np.nan
        modes_rows.append({"k": k, "silhouette": sil,
                           "cost": float(km.cost_),
                           "sizes": pd.Series(lk).value_counts().sort_index().tolist()})
    diag_modes = pd.DataFrame(modes_rows)

    # ---- choose k --------------------------------------------------------- #
    # Require interpretable, balanced clusters (each >= 8 households, ~12% of N)
    # then take the best silhouette. Silhouettes are uniformly low (~0.1), so we
    # also apply a parsimony tie-break: prefer the smaller k when a larger k beats
    # it by < 0.02 silhouette (avoids over-segmenting 69 weakly-separated cases).
    elig = diag_g[diag_g["min_cluster_size"] >= 8].copy()
    pool = elig if len(elig) else diag_g.copy()
    pool = pool.sort_values(["silhouette", "k"], ascending=[False, True])
    best = pool.iloc[0]
    parsimonious = pool[pool["silhouette"] >= best["silhouette"] - 0.02]
    pick = parsimonious.sort_values("k").iloc[0]
    k_star = int(pick["k"])
    labels = labels_by_k[k_star]

    # order clusters by mean ECI (Low->High) for stable, readable naming
    order = (pd.Series(df["eci"].values).groupby(labels).mean()
             .sort_values().index.tolist())
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in labels])
    sizes_sorted = pd.Series(labels).value_counts().sort_index()

    # human-readable cluster names assigned AFTER profiling (below); placeholder
    label_names = {cl: f"C{cl+1}" for cl in range(k_star)}

    # ---- MCA (categoricals) ---------------------------------------------- #
    mca = prince.MCA(n_components=2, random_state=SEED)
    mca = mca.fit(f[cat_cols].astype(str))
    mca_coords = mca.transform(f[cat_cols].astype(str)).values
    try:
        mca_inertia = list(np.asarray(mca.percentage_of_variance_)[:2])
    except Exception:
        ev = np.asarray(mca.eigenvalues_, float)
        mca_inertia = list(100 * ev[:2] / ev.sum())

    # ---- FAMD (mixed) ----------------------------------------------------- #
    famd_coords, famd_inertia = None, [np.nan, np.nan]
    try:
        famd = prince.FAMD(n_components=2, random_state=SEED)
        famd = famd.fit(mixed)
        famd_coords = famd.transform(mixed).values
        try:
            famd_inertia = list(np.asarray(famd.percentage_of_variance_)[:2])
        except Exception:
            ev = np.asarray(famd.eigenvalues_, float)
            famd_inertia = list(100 * ev[:2] / ev.sum())
    except Exception as e:
        print("FAMD failed (continuing without it):", repr(e))

    # ---- profiles + naming ----------------------------------------------- #
    prof = profile_clusters(df, f, num, labels, cat_cols, num_cols, label_names)

    # Derive interpretable names from the attributes that genuinely separate the
    # clusters (prevalence within cluster vs. overall), plus dwelling size, the
    # SES (expenditure) tier and the ECI rank. Names are descriptive labels for a
    # convenience sample, NOT population types.
    area_med_all = float(num["Floor area (sqft)"].median())
    exp_terc = num["Total exp (Rs/mo)"].quantile([1 / 3, 2 / 3]).values

    def share(sub, col, lvl):
        return float((sub[col] == lvl).mean())

    def name_cluster(cl):
        m = labels == cl
        sub = f.loc[m]
        eci_m = df.loc[m, "eci"].mean()
        area_m = num.loc[m, "Floor area (sqft)"].median()
        exp_m = num.loc[m, "Total exp (Rs/mo)"].median()
        tags = []
        # dwelling size
        if area_m <= 0.6 * area_med_all:
            tags.append("compact single-floor")
        elif share(sub, "House type", "Double floor") >= 0.6:
            tags.append("double-floor")
        elif share(sub, "House type", "3+ floors") >= 0.3:
            tags.append("large multi-floor")
        else:
            tags.append("mixed-fabric")
        # professionalisation / education axis
        if share(sub, "University-educated", "Yes") >= 0.7:
            tags.append("university-educated")
        elif share(sub, "Architect-designed", "No") >= 0.7 and \
                share(sub, "University-educated", "No") >= 0.6:
            tags.append("non-professionalised")
        # SES tier by expenditure tercile
        if exp_m >= exp_terc[1]:
            tags.append("high-spend")
        elif exp_m <= exp_terc[0]:
            tags.append("low-spend")
        # billing behaviour
        if share(sub, "Bill rounding", "Rounds up (covers bill)") >= 0.7:
            tags.append("bill-rounding")
        elif share(sub, "Bill rounding", "Exact / under / partial") >= 0.7:
            tags.append("exact-paying")
        # conservation rank (descriptive, not a reliable scale)
        tags.append("higher-conservation" if eci_m >= df["eci"].mean() + 3
                    else "lower-conservation" if eci_m <= df["eci"].mean() - 3
                    else "mid-conservation")
        return ", ".join(tags)

    label_names = {cl: f"C{cl+1}: " + name_cluster(cl) for cl in range(k_star)}
    prof["Cluster"] = [label_names[cl] for cl in sorted(np.unique(labels))]

    # ---- effect-size table (BH-FDR) -------------------------------------- #
    eff = attribute_effects(df, f, labels, cat_cols)

    # ---- figures ---------------------------------------------------------- #
    saved = {}
    fig_silhouette(diag_g, diag_modes, saved)
    # MCA (left) + FAMD (right) as one full-width figure with a shared legend;
    # a None FAMD panel renders a placeholder so the paper figure stays intact.
    fig_cluster_maps(
        [("MCA", mca_coords, mca_inertia),
         ("FAMD", famd_coords, famd_inertia)],
        labels, label_names, "07_cluster_maps", saved)
    heat = fig_profile_heatmap(df, f, labels, cat_cols, label_names, saved)

    # ---- agreement between the two clustering methods (at k*) ------------ #
    from sklearn.metrics import adjusted_rand_score
    ari_modes = float(adjusted_rand_score(labels, modes_labels[k_star]))

    # ---- write tables ----------------------------------------------------- #
    prof_path_csv = C.TBL_DIR / "07_cluster_profiles.csv"
    prof_path_tex = C.TBL_DIR / "07_cluster_profiles.tex"
    prof.to_csv(prof_path_csv, index=False)
    prof.to_latex(prof_path_tex, index=False, escape=True,
                  caption="Household typology profiles (k=%d). Dominant category "
                          "shown with within-cluster count and share; ECI and "
                          "behaviour-item means are descriptive only (the five "
                          "items do not form a reliable scale)." % k_star,
                  label="tab:cluster_profiles", longtable=False)

    eff_path_csv = C.TBL_DIR / "07_cluster_attribute_effects.csv"
    eff_path_tex = C.TBL_DIR / "07_cluster_attribute_effects.tex"
    eff_round = eff.copy()
    for c in ["value", "p", "q_bh"]:
        eff_round[c] = eff_round[c].astype(float).round(4)
    eff_round.to_csv(eff_path_csv, index=False)
    eff_round.to_latex(eff_path_tex, index=False, escape=True,
                       float_format="%.3f",
                       caption="Exploratory association of the derived typology "
                               "with each household attribute (Cramer's V) and "
                               "with the ECI / behaviour items (epsilon^2), "
                               "Benjamini-Hochberg FDR corrected. Convenience "
                               "sample, N=70; interpret as effect sizes, not "
                               "confirmatory tests.",
                       label="tab:cluster_effects", longtable=False)

    sil_path_csv = C.TBL_DIR / "07_silhouette_by_k.csv"
    diag_all = diag_g.merge(diag_modes, on="k", suffixes=("_agglom", "_kmodes"))
    diag_all.to_csv(sil_path_csv, index=False)

    scan_path_csv = C.TBL_DIR / "07_linkage_scan.csv"
    scan.to_csv(scan_path_csv, index=False)

    # ---- compact JSON summary -------------------------------------------- #
    n_surv = int((eff["q_bh"] < 0.05).sum())
    cluster_eci = {label_names[cl]: round(float(df.loc[labels == cl, "eci"].mean()), 1)
                   for cl in sorted(np.unique(labels))}
    cluster_sizes = {label_names[cl]: int((labels == cl).sum())
                     for cl in sorted(np.unique(labels))}
    top_eff = eff.dropna(subset=["value"]).sort_values(
        "value", ascending=False).head(3)
    summary = {
        "n": int(len(df)),
        "n_features_cat": len(cat_cols),
        "n_features_num": len(num_cols),
        "k_selected": k_star,
        "primary_linkage": "complete",
        "silhouette_at_k": round(float(pick["silhouette"]), 3),
        "silhouette_by_k_agglom": {int(r.k): round(float(r.silhouette), 3)
                                   for r in diag_g.itertuples()},
        "silhouette_by_k_kmodes": {int(r.k): (round(float(r.silhouette), 3)
                                              if not np.isnan(r.silhouette) else None)
                                   for r in diag_modes.itertuples()},
        "method_agreement_ARI_agglom_vs_kmodes": round(ari_modes, 3),
        "mca_inertia_dim12_pct": [round(float(x), 1) for x in mca_inertia],
        "famd_inertia_dim12_pct": [round(float(x), 1) if x == x else None
                                   for x in famd_inertia],
        "cluster_sizes": cluster_sizes,
        "cluster_eci_mean": cluster_eci,
        "overall_eci_mean": round(float(df["eci"].mean()), 1),
        "n_effects_surviving_fdr_q05": n_surv,
        "top_effects": [{"attribute": r.attribute, "effect": r.effect,
                         "value": round(float(r.value), 3),
                         "p": round(float(r.p), 4),
                         "q_bh": round(float(r.q_bh), 4)}
                        for r in top_eff.itertuples()],
        "cluster_names": list(label_names.values()),
        "figures": [p for ps in saved.values() for p in ps],
        "tables": [str(prof_path_csv), str(prof_path_tex),
                   str(eff_path_csv), str(eff_path_tex),
                   str(sil_path_csv), str(scan_path_csv)],
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
