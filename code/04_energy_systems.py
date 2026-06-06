"""04 -- Energy systems & billing (Theme 4).

Descriptive, effect-size-led profile of how the N=70 surveyed households
generate electricity, use solar, get their wiring done, pay their bills, and
whether they receive disconnection ("red") notices. Everything is EXPLORATORY
and descriptive: this is a skewed convenience sample, so we lead with
proportions accompanied by both Wilson score 95% intervals (the appropriate
small-n binomial interval) and bootstrap 95% percentile intervals from the
shared library, and we never imply causation.

Outputs (prefix 04_):
    figures/  04_generation_sources, 04_solar_purposes, 04_pay_practice,
              04_red_notice_counts
    tables/   04_energy_systems_summary{.csv,.tex},
              04_generation_sources{.csv,.tex},
              04_solar_purposes{.csv,.tex}

The last stdout line is one compact JSON object of key findings.

Run:  uv run python 04_energy_systems.py
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
# Helpers
# --------------------------------------------------------------------------- #
def wilson_ci(k: int, n: int, z: float = 1.959963984540054):
    """Wilson score interval for a binomial proportion (good for small n / extreme p)."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (float(phat), float(max(0.0, center - half)), float(min(1.0, center + half)))


def prop_row(label: str, k: int, n: int, seed: int = 0) -> dict:
    """One proportion with count, Wilson CI, and bootstrap CI (shared lib)."""
    phat, wlo, whi = wilson_ci(k, n)
    # bootstrap CI on the 0/1 indicator (mean == proportion) for cross-checking
    indicator = np.concatenate([np.ones(k), np.zeros(n - k)]) if n else np.array([])
    if n >= 2:
        bmean, blo, bhi = S.bootstrap_ci(indicator, np.mean, n_boot=10000, seed=seed)
    else:
        bmean, blo, bhi = (phat, np.nan, np.nan)
    return {
        "metric": label, "k": int(k), "n": int(n),
        "proportion": round(phat, 4),
        "wilson_lo": round(wlo, 4), "wilson_hi": round(whi, 4),
        "boot_lo": round(blo, 4) if blo == blo else np.nan,
        "boot_hi": round(bhi, 4) if bhi == bhi else np.nan,
    }


def fmt_pct(p):
    return f"{100 * p:.1f}%"


def main():
    df = io.load_clean()
    N = len(df)
    rng_seed = 20240529

    # --------------------------------------------------------------------- #
    # 1. Generation sources
    # --------------------------------------------------------------------- #
    gen_cols = ["gen_grid", "gen_solar", "gen_hydro", "gen_othergreen", "gen_none"]
    gen_labels = {
        "gen_grid": "Grid power", "gen_solar": "Solar", "gen_hydro": "Mini-hydro",
        "gen_othergreen": "Other green", "gen_none": "None of the above",
    }
    gen_rows = [prop_row(gen_labels[c], int(df[c].sum()), N, seed=rng_seed + i)
                for i, c in enumerate(gen_cols)]
    gen_rows.append(prop_row("Any renewable (solar/hydro/other-green)",
                             int(df["uses_renewable"].sum()), N, seed=rng_seed + 50))
    gen_tbl = pd.DataFrame(gen_rows)

    # generation source combination (mutually exclusive profile)
    def gen_profile(r):
        if r["gen_none"] == 1 and r["gen_count"] == 0:
            return "None reported"
        labs = [gen_labels[c].replace(" power", "") for c in
                ["gen_grid", "gen_solar", "gen_hydro", "gen_othergreen"] if r[c] == 1]
        return " + ".join(labs) if labs else "None reported"
    gen_combo = df.apply(gen_profile, axis=1).value_counts()

    # --------------------------------------------------------------------- #
    # 2. Solar: user rate + purposes among users
    # --------------------------------------------------------------------- #
    n_solar_users = int(df["solar_user"].sum())
    solar_user_row = prop_row("Solar user (any solar purpose reported)",
                              n_solar_users, N, seed=rng_seed + 60)
    # discrepancy: generation-source 'solar' vs purpose-based 'solar_user'
    n_gen_solar = int(df["gen_solar"].sum())

    purpose_cols = ["solar_water", "solar_outdoor", "solar_cooking",
                    "solar_car", "solar_agri", "solar_other"]
    purpose_labels = {
        "solar_water": "Water heating", "solar_outdoor": "Outdoor lighting",
        "solar_cooking": "Cooking", "solar_car": "Car charging",
        "solar_agri": "Agriculture", "solar_other": "Other",
    }
    users = df[df["solar_user"] == 1]
    solar_rows = [prop_row(purpose_labels[c], int(users[c].sum()), n_solar_users,
                           seed=rng_seed + 70 + i)
                  for i, c in enumerate(purpose_cols)]
    solar_tbl = pd.DataFrame(solar_rows)
    # number of distinct purposes per user
    purpose_count_dist = (users["solar_purpose_count"].value_counts()
                          .sort_index().astype(int))

    # --------------------------------------------------------------------- #
    # 3. Professional wiring
    # --------------------------------------------------------------------- #
    wiring_counts = df["wiring_pro"].value_counts()
    n_wiring_yes = int((df["wiring_pro"] == "Yes").sum())
    wiring_yes_row = prop_row("Wiring done by a professional (Yes)",
                              n_wiring_yes, N, seed=rng_seed + 90)
    n_wiring_unaware = int((df["wiring_pro"] == "I am not aware of it").sum())

    # --------------------------------------------------------------------- #
    # 4. Red notices (sparse -> treat as binary any/none + raw distribution)
    # --------------------------------------------------------------------- #
    n_any_red = int(df["any_red_notice"].sum())
    red_row = prop_row("Received >=1 red notice last year", n_any_red, N,
                       seed=rng_seed + 100)
    red_counts = df["red_notices"].value_counts().sort_index().astype(int)
    among = df.loc[df["red_notices"] > 0, "red_notices"]
    red_among_desc = among.describe()

    # --------------------------------------------------------------------- #
    # 5. Bill-payment practice
    # --------------------------------------------------------------------- #
    pay_short = {
        "I try to pay the exact amount in the bill (ex: 4566)": "Exact amount",
        "I try to pay a rounded-off amount which will cover the whole bill amount (ex: 4600": "Round up (covers bill)",
        "I try to pay a rounded-off amount which is mostly lower than the bill (ex: 4500)": "Round down (under bill)",
        "I try to pay a portion of my bill amount that is possible for me (ex: 3500)": "Partial payment",
    }
    pay = df["pay_practice"].map(lambda v: pay_short.get(v, v))
    pay_counts = pay.value_counts()
    pay_order = ["Exact amount", "Round up (covers bill)",
                 "Round down (under bill)", "Partial payment"]
    pay_counts = pay_counts.reindex([o for o in pay_order if o in pay_counts.index])
    # full settlement = exact OR round-up (both cover the whole bill)
    n_full_settle = int(pay.isin(["Exact amount", "Round up (covers bill)"]).sum())
    full_settle_row = prop_row("Pays at least the full bill (exact or round-up)",
                               n_full_settle, N, seed=rng_seed + 110)
    n_under = int(pay.isin(["Round down (under bill)", "Partial payment"]).sum())
    under_row = prop_row("Pays less than the full bill (round-down or partial)",
                         n_under, N, seed=rng_seed + 120)

    # --------------------------------------------------------------------- #
    # 6. Exploratory association: any_red_notice x under-payment (2x2 Fisher)
    #    Honest, hypothesis-generating only; single test, report effect size.
    # --------------------------------------------------------------------- #
    underpay = pay.isin(["Round down (under bill)", "Partial payment"]).astype(int)
    assoc = S.assoc_categorical(df["any_red_notice"], underpay)
    ct_red_pay = pd.crosstab(df["any_red_notice"], underpay)

    # --------------------------------------------------------------------- #
    # Master energy-systems summary table
    # --------------------------------------------------------------------- #
    summary_rows = (
        gen_rows
        + [solar_user_row]
        + [wiring_yes_row, prop_row("Wiring: not aware", n_wiring_unaware, N, seed=rng_seed + 91)]
        + [full_settle_row, under_row]
        + [red_row]
    )
    summary_tbl = pd.DataFrame(summary_rows)
    summary_tbl_disp = summary_tbl.copy()
    # pretty 95% CI strings for the paper table
    summary_tbl_disp["Wilson 95% CI"] = summary_tbl_disp.apply(
        lambda r: f"[{fmt_pct(r['wilson_lo'])}, {fmt_pct(r['wilson_hi'])}]", axis=1)
    summary_tbl_disp["Bootstrap 95% CI"] = summary_tbl_disp.apply(
        lambda r: (f"[{fmt_pct(r['boot_lo'])}, {fmt_pct(r['boot_hi'])}]"
                   if pd.notna(r["boot_lo"]) else "--"), axis=1)
    summary_tbl_disp["%"] = summary_tbl_disp["proportion"].map(fmt_pct)
    summary_tbl_disp = summary_tbl_disp[
        ["metric", "k", "n", "%", "Wilson 95% CI", "Bootstrap 95% CI"]
    ].rename(columns={"metric": "Indicator", "k": "Count", "n": "Base"})

    # ------------------------------------------------------------------ #
    # Save tables (csv + latex)
    # ------------------------------------------------------------------ #
    def save_table(df_csv, df_tex, name, caption, label):
        cpath = C.TBL_DIR / f"{name}.csv"
        tpath = C.TBL_DIR / f"{name}.tex"
        df_csv.to_csv(cpath, index=False)
        df_tex.to_latex(tpath, index=False, escape=True, caption=caption,
                        label=label, longtable=False)
        return str(cpath), str(tpath)

    t_sum = save_table(
        summary_tbl, summary_tbl_disp, "04_energy_systems_summary",
        "Energy-systems and billing profile of the surveyed households (N=70). "
        "Proportions with Wilson score and bootstrap (10000-resample) 95\\% "
        "confidence intervals. Sample is a skewed convenience sample; estimates "
        "are exploratory.", "tab:energy_systems_summary")

    # generation table for paper (counts + Wilson CI strings)
    gen_disp = gen_tbl.copy()
    gen_disp["%"] = gen_disp["proportion"].map(fmt_pct)
    gen_disp["Wilson 95% CI"] = gen_disp.apply(
        lambda r: f"[{fmt_pct(r['wilson_lo'])}, {fmt_pct(r['wilson_hi'])}]", axis=1)
    gen_disp = gen_disp[["metric", "k", "n", "%", "Wilson 95% CI"]].rename(
        columns={"metric": "Generation source", "k": "Households", "n": "Base"})
    t_gen = save_table(
        gen_tbl, gen_disp, "04_generation_sources",
        "Electricity generation sources (multi-select), N=70. Shares sum to more "
        "than 100\\% because households may report more than one source.",
        "tab:generation_sources")

    solar_disp = solar_tbl.copy()
    solar_disp["%"] = solar_disp["proportion"].map(fmt_pct)
    solar_disp["Wilson 95% CI"] = solar_disp.apply(
        lambda r: f"[{fmt_pct(r['wilson_lo'])}, {fmt_pct(r['wilson_hi'])}]", axis=1)
    solar_disp = solar_disp[["metric", "k", "n", "%", "Wilson 95% CI"]].rename(
        columns={"metric": "Solar purpose", "k": "Users", "n": "Base"})
    t_solar = save_table(
        solar_tbl, solar_disp, "04_solar_purposes",
        f"Purposes of solar-energy use among the {n_solar_users} solar-using "
        "households (multi-select). Shares are of solar users, not the full sample.",
        "tab:solar_purposes")

    # ------------------------------------------------------------------ #
    # Figures
    # ------------------------------------------------------------------ #
    figs = {}

    # Fig 1: generation sources (horizontal bars, share of all households)
    g = gen_tbl[gen_tbl["metric"].isin(list(gen_labels.values()))].copy()
    g = g.sort_values("k")
    fig, ax = plt.subplots(figsize=P.figsize("FULL", 3.4))
    ypos = np.arange(len(g))
    ax.barh(ypos, g["k"].values, color=P.ACCENT)
    # Wilson CI as % of households -> convert to counts for error bars on count axis
    lo_cnt = g["wilson_lo"].values * g["n"].values
    hi_cnt = g["wilson_hi"].values * g["n"].values
    ax.errorbar(g["k"].values, ypos,
                xerr=[g["k"].values - lo_cnt, hi_cnt - g["k"].values],
                fmt="none", ecolor=P.RULE, elinewidth=1.1, capsize=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(g["metric"].values)
    ax.set_xlabel("Households (of 69)")
    for i, (k, p) in enumerate(zip(g["k"].values, g["proportion"].values)):
        ax.text(hi_cnt[i] + 0.8, i, f"{int(k)} ({fmt_pct(p)})", va="center", fontsize=P.ANNOT)
    ax.set_xlim(0, max(hi_cnt) + 12)
    figs["04_generation_sources"] = P.save_fig(fig, "04_generation_sources")

    # Fig 2: solar purposes among users (share of users)
    s = solar_tbl.sort_values("k").copy()
    fig, ax = plt.subplots(figsize=P.figsize("FULL", 3.4))
    ypos = np.arange(len(s))
    ax.barh(ypos, s["proportion"].values * 100, color=P.ACCENT2)
    lo = s["wilson_lo"].values * 100
    hi = s["wilson_hi"].values * 100
    ax.errorbar(s["proportion"].values * 100, ypos,
                xerr=[s["proportion"].values * 100 - lo, hi - s["proportion"].values * 100],
                fmt="none", ecolor=P.RULE, elinewidth=1.1, capsize=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(s["metric"].values)
    ax.set_xlabel(f"Share of solar users (n={n_solar_users})  [%]")
    for i, (k, p) in enumerate(zip(s["k"].values, s["proportion"].values)):
        ax.text(hi[i] + 1.5, i, f"{int(k)}/{n_solar_users}", va="center", fontsize=P.ANNOT)
    ax.set_xlim(0, 105)
    figs["04_solar_purposes"] = P.save_fig(fig, "04_solar_purposes")

    # Fig 3: bill-payment practice
    fig, ax = plt.subplots(figsize=P.figsize("FULL", 3.2))
    pc = pay_counts.sort_values()
    colors = [P.GOOD if lbl in ("Exact amount", "Round up (covers bill)")
              else P.BAD for lbl in pc.index]
    ypos = np.arange(len(pc))
    ax.barh(ypos, pc.values, color=colors)
    ax.set_yticks(ypos)
    ax.set_yticklabels(pc.index)
    ax.set_xlabel("Households (of 69)")
    for i, v in enumerate(pc.values):
        ax.text(v + 0.4, i, f"{int(v)} ({fmt_pct(v / N)})", va="center", fontsize=P.ANNOT)
    ax.set_xlim(0, max(pc.values) + 9)
    # legend proxy
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=P.GOOD, label="Covers full bill"),
                       Patch(color=P.BAD, label="Under-pays bill")],
              loc="lower right")
    figs["04_pay_practice"] = P.save_fig(fig, "04_pay_practice")

    # Fig 4: red-notice counts (raw distribution, sparse)
    fig, ax = plt.subplots(figsize=P.figsize("FULL", 3.2))
    rc = red_counts
    bars = ax.bar(rc.index.astype(int).astype(str), rc.values, color=P.ACCENT)
    bars[0].set_color(P.MUTED)  # the "0" bar is the no-notice majority
    ax.set_xlabel("Red notices received last year")
    ax.set_ylabel("Households")
    for x, v in zip(range(len(rc)), rc.values):
        ax.text(x, v + 0.6, str(int(v)), ha="center", fontsize=P.ANNOT)
    ax.set_ylim(0, max(rc.values) + 6)
    figs["04_red_notice_counts"] = P.save_fig(fig, "04_red_notice_counts")

    # ------------------------------------------------------------------ #
    # Verify all files exist
    # ------------------------------------------------------------------ #
    all_paths = []
    for v in figs.values():
        all_paths.extend(v)
    all_paths += [t_sum[0], t_sum[1], t_gen[0], t_gen[1], t_solar[0], t_solar[1]]
    missing = [p for p in all_paths if not Path(p).exists()]
    assert not missing, f"MISSING OUTPUT FILES: {missing}"

    # ------------------------------------------------------------------ #
    # Key findings JSON (last stdout line)
    # ------------------------------------------------------------------ #
    findings = {
        "N": N,
        "generation": {
            "grid": prop_row("grid", int(df["gen_grid"].sum()), N),
            "solar_gen_source": prop_row("solar_gen", n_gen_solar, N),
            "hydro": int(df["gen_hydro"].sum()),
            "othergreen": int(df["gen_othergreen"].sum()),
            "none": prop_row("none", int(df["gen_none"].sum()), N),
            "any_renewable": prop_row("renewable", int(df["uses_renewable"].sum()), N),
        },
        "solar": {
            "solar_user_rate": solar_user_row,
            "n_solar_users": n_solar_users,
            "n_gen_solar_source": n_gen_solar,
            "purpose_gen_vs_user_gap": n_solar_users - n_gen_solar,
            "top_purpose": {
                "name": "Outdoor lighting",
                "k": int(users["solar_outdoor"].sum()),
                "n": n_solar_users,
                "share": round(users["solar_outdoor"].mean(), 4),
            },
            "multi_purpose_users": int((users["solar_purpose_count"] >= 2).sum()),
        },
        "wiring": {
            "professional_yes": wiring_yes_row,
            "not_aware": n_wiring_unaware,
            "no": int((df["wiring_pro"] == "No").sum()),
        },
        "payment": {
            "full_settle": full_settle_row,
            "underpay": under_row,
            "counts": {k: int(v) for k, v in pay_counts.items()},
        },
        "red_notices": {
            "any_red_notice": red_row,
            "n_zero": int((df["red_notices"] == 0).sum()),
            "max": int(df["red_notices"].max()),
            "mean_among_recipients": round(float(red_among_desc["mean"]), 2),
            "median_among_recipients": round(float(among.median()), 2),
        },
        "explore_redxunderpay": {
            "test": assoc["test"], "p": round(float(assoc["p"]), 4),
            "cramers_v": round(float(assoc["cramers_v"]), 4),
            "min_expected": round(float(assoc["min_expected"]), 2),
            "ct": ct_red_pay.values.tolist(),
            "note": "single exploratory 2x2 test; hypothesis-generating only",
        },
        "figures": [p for v in figs.values() for p in v],
        "tables": [t_sum[0], t_sum[1], t_gen[0], t_gen[1], t_solar[0], t_solar[1]],
    }

    print("\n# Generation source combinations:")
    print(gen_combo.to_string())
    print("\n# Solar purpose-count distribution (users):")
    print(purpose_count_dist.to_string())
    print("\n# Red-notice 2x2 (rows=any_red_notice, cols=underpay):")
    print(ct_red_pay.to_string())
    print("\nWROTE FIGURES:")
    for v in figs.values():
        for p in v:
            print("  ", p)
    print("WROTE TABLES:")
    for p in [t_sum[0], t_sum[1], t_gen[0], t_gen[1], t_solar[0], t_solar[1]]:
        print("  ", p)
    print("\n" + json.dumps(findings, default=str))


if __name__ == "__main__":
    main()
