"""Small-sample statistics utilities.

Chosen for n=69: bias-corrected Cramer's V for categorical association, Fisher's
exact for 2x2 tables, Kruskal-Wallis (+ epsilon-squared) for category-vs-ordinal
comparisons, bootstrap confidence intervals, and Benjamini-Hochberg FDR control
for multiple comparisons. We report effect sizes alongside every p-value.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as ss


# --------------------------------------------------------------------------- #
# Association / effect size
# --------------------------------------------------------------------------- #
def cramers_v(x, y, bias_correction: bool = True) -> float:
    """Cramer's V with the Bergsma-Wicher bias correction (recommended for small n)."""
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan
    chi2 = ss.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    phi2 = chi2 / n
    r, k = ct.shape
    if bias_correction:
        phi2 = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
        r = r - (r - 1) ** 2 / (n - 1)
        k = k - (k - 1) ** 2 / (n - 1)
    denom = min(k - 1, r - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else np.nan


def assoc_categorical(x, y) -> dict:
    """Association between two categoricals: Fisher (2x2) or chi-square, + Cramer's V."""
    ct = pd.crosstab(x, y)
    out = {"n": int(ct.values.sum()), "rows": ct.shape[0], "cols": ct.shape[1]}
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        out.update(test=None, p=np.nan, cramers_v=np.nan, min_expected=np.nan)
        return out
    if ct.shape == (2, 2):
        out["test"] = "fisher"
        out["p"] = float(ss.fisher_exact(ct)[1])
        out["min_expected"] = float(ss.chi2_contingency(ct, correction=False)[3].min())
    else:
        chi2, p, dof, exp = ss.chi2_contingency(ct, correction=False)
        out["test"] = "chi2"
        out["p"] = float(p)
        out["min_expected"] = float(exp.min())
    out["cramers_v"] = cramers_v(x, y)
    return out


def kruskal_effect(*groups) -> dict:
    """Kruskal-Wallis H test with epsilon-squared effect size."""
    groups = [np.asarray(g, float) for g in groups]
    groups = [g[~np.isnan(g)] for g in groups]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return {"H": np.nan, "p": np.nan, "eps2": np.nan, "k": len(groups)}
    H, p = ss.kruskal(*groups)
    n = sum(len(g) for g in groups)
    k = len(groups)
    eps2 = (H - k + 1) / (n - k) if n > k else np.nan
    return {"H": float(H), "p": float(p), "eps2": float(eps2), "k": k, "n": n}


# --------------------------------------------------------------------------- #
# Reliability
# --------------------------------------------------------------------------- #
def cronbach_alpha(items_df: pd.DataFrame) -> float:
    items = items_df.dropna()
    k = items.shape[1]
    if k < 2:
        return np.nan
    item_var = items.var(axis=0, ddof=1).sum()
    total_var = items.sum(axis=1).var(ddof=1)
    return float((k / (k - 1)) * (1 - item_var / total_var)) if total_var else np.nan


# --------------------------------------------------------------------------- #
# Bootstrap CIs
# --------------------------------------------------------------------------- #
def bootstrap_ci(data, stat_func, n_boot: int = 10000, ci: float = 95, seed: int = 0):
    rng = np.random.default_rng(seed)
    data = np.asarray(data, float)
    data = data[~np.isnan(data)]
    n = len(data)
    if n < 2:
        return (np.nan, np.nan, np.nan)
    boots = np.array([stat_func(data[rng.integers(0, n, n)]) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(stat_func(data)), float(lo), float(hi)


def bootstrap_ci_pair(x, y, stat_func, n_boot: int = 10000, ci: float = 95, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return (np.nan, np.nan, np.nan)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(stat_func(x[idx], y[idx]))
    lo, hi = np.nanpercentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(stat_func(x, y)), float(lo), float(hi)


def spearman_with_ci(x, y, n_boot: int = 10000, seed: int = 0):
    """Spearman rho, p-value, and bootstrap 95% CI."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"rho": np.nan, "p": np.nan, "lo": np.nan, "hi": np.nan, "n": len(x)}
    rho, p = ss.spearmanr(x, y)
    _, lo, hi = bootstrap_ci_pair(x, y, lambda a, b: ss.spearmanr(a, b)[0], n_boot, seed=seed)
    return {"rho": float(rho), "p": float(p), "lo": lo, "hi": hi, "n": int(len(x))}


# --------------------------------------------------------------------------- #
# Multiple comparisons
# --------------------------------------------------------------------------- #
def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    out = np.full(p.shape, np.nan)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


# --------------------------------------------------------------------------- #
# Group-difference effect sizes (continuous outcome ~ binary group)
# --------------------------------------------------------------------------- #
def cliffs_delta(a, b) -> float:
    """Cliff's delta for two samples (direction-aware, in [-1, 1]).

    delta = P(a > b) - P(a < b); 0 is stochastic equality. Robust at small n.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return float((gt - lt) / (len(a) * len(b)))


def mann_whitney_effect(values, group, n_boot: int = 10000, seed: int = 0) -> dict:
    """Mann-Whitney U for `values` where group==1 vs group==0, reported with the
    common-language effect size.

    Returns the AUC = P(X1 > X0) (probability of superiority / common-language
    effect size of McGraw & Wong), Cliff's delta (= 2*AUC - 1), the two group
    medians, the two-sided p-value, and a percentile bootstrap 95% CI on the AUC
    (each group resampled independently with a fixed seed). Chosen over a t-test
    for the small, skewed n=69 sample and an unequal binary split.
    """
    v = np.asarray(values, float)
    g = np.asarray(group, float)
    m = ~(np.isnan(v) | np.isnan(g))
    v, g = v[m], g[m]
    x1, x0 = v[g == 1], v[g == 0]
    n1, n0 = len(x1), len(x0)
    if n1 < 3 or n0 < 3:
        return {"n1": int(n1), "n0": int(n0), "med1": np.nan, "med0": np.nan,
                "auc": np.nan, "auc_lo": np.nan, "auc_hi": np.nan,
                "cliff": np.nan, "p": np.nan}
    U1, p = ss.mannwhitneyu(x1, x0, alternative="two-sided")
    auc = U1 / (n1 * n0)                       # P(adopter value > non-adopter value)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        bx1 = x1[rng.integers(0, n1, n1)]
        bx0 = x0[rng.integers(0, n0, n0)]
        boots[i] = ss.mannwhitneyu(bx1, bx0, alternative="two-sided").statistic / (n1 * n0)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"n1": int(n1), "n0": int(n0), "med1": float(np.median(x1)),
            "med0": float(np.median(x0)), "auc": float(auc),
            "auc_lo": float(lo), "auc_hi": float(hi),
            "cliff": float(2 * auc - 1), "p": float(p)}
