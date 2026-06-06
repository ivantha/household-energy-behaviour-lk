"""Publication plotting style and figure-saving helpers.

All figures are saved as both PNG (LaTeX build / quick view) and PDF (vector,
print-quality). Fonts are embedded (Type 42) for portability. The whole figure
theme -- font family, size hierarchy, palette, canonical widths -- lives in this
module so the analysis scripts carry no per-figure style code: import it for its
side effect of setting the global style.

Design notes
------------
* **Consistent, readable on-page text.** Figures are authored at a small set of
  canonical *widths* (``FIG_W``) chosen to match the LaTeX include widths, so a
  point size set here lands at roughly the same size on the page rather than being
  downscaled unevenly. Use ``figsize(preset, height)`` instead of a literal width.
* **Font matches the paper (Libertinus serif).** Libertinus is registered from the
  local TeX install when present; STIX Two Text is the portable fallback. Math uses
  the matching ``stix`` set. Flip ``FONT_MODE`` to ``"sans"`` to switch the whole
  set in one place.
* **Semantic colours are load-bearing** and must not be genericised: the ECI
  Low/Medium/High band colours (red/amber/green) and the bill-payment
  covers/under-pays green/red.
"""
from __future__ import annotations
import glob
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from . import config as C

# --------------------------------------------------------------------------- #
# Palette (cohesive; the semantic colours below are referenced by meaning)
# --------------------------------------------------------------------------- #
ACCENT = "#1b7a78"          # primary teal
ACCENT2 = "#c8553d"         # secondary terracotta
SEQ_CMAP = "rocket_r"       # sequential heatmaps (0 .. max): effect sizes, Cramer's V
DIVERGING = "vlag"          # diverging heatmaps (-1 .. 1): correlations, deviations

# ECI tertile bands -- red=Low, amber=Medium, green=High. Reused for any ordinal
# Low/Medium/High encoding (e.g. the area tertiles in 09) and as the single source
# of the bill-payment semantic green/red. Semantic -- never re-themed.
BAND_COLORS = {"Low": "#d9534f", "Medium": "#f0ad4e", "High": "#5cb85c"}
GOOD = BAND_COLORS["High"]  # bill-payment: "covers the bill" (green)
BAD = BAND_COLORS["Low"]    # bill-payment: "under-pays" (red)

# Qualitative cycle (teal, terracotta, olive, plum, straw) for generic categorical
# series and the k=5 household typologies. Installed as the default prop_cycle in
# set_style(), so a bare plot draws from this theme rather than matplotlib's tab10.
CATEGORICAL = [ACCENT, ACCENT2, "#8aa399", "#7d5ba6", "#e3b23c"]
CLUST_PAL = CATEGORICAL

# Neutral scale -- the de-emphasis / annotation greys, centralised so the same role
# uses the same value across every figure (replaces ad-hoc per-script greys).
MUTED = "#c9ccd1"           # light: de-emphasised / secondary bar fills, guide lines
RULE = "#5a5f66"            # mid: reference lines, error bars, outlier markers, 2nd series
INK = "#222222"             # near-black: text, dark data markers (jitter / strip points)

# Data-annotation font sizes -- the only sizes set outside rcParams (bar-value
# labels, heatmap cell numbers). ANNOT_SMALL is for dense grids.
ANNOT = 8
ANNOT_SMALL = 7

# --------------------------------------------------------------------------- #
# Canonical figure widths (author inches) keyed to the LaTeX include width.
#   FULL  -> \includegraphics[width=\textwidth]      (14 cm measure = 5.51 in page)
#   SMALL -> \includegraphics[width=0.7\textwidth]   (simple single-panel plots)
#   HALF  -> \includegraphics[width=0.49\textwidth]  (the 07 MCA/FAMD pair)
# A 6.0 in author width shown at 5.51 in is a ~0.92x scale, so a 10 pt label prints
# at ~9.2 pt and every figure matches. SMALL/HALF keep that exact scale by embedding
# at the matching width fraction (4.2/6 ~ 0.7, 3.0/6 ~ 0.49): the figure shrinks but
# the fonts print at the same size. Height stays per-figure via the `h` arg.
# --------------------------------------------------------------------------- #
FIG_W = {"FULL": 6.0, "SMALL": 4.2, "HALF": 3.0}


def figsize(preset: str, h: float) -> tuple[float, float]:
    """(width, height) in inches for a canonical width preset."""
    return (FIG_W[preset], h)


# --------------------------------------------------------------------------- #
# Font family: match the paper (Libertinus serif) with a portable fallback.
# --------------------------------------------------------------------------- #
FONT_MODE = "serif"  # "serif" matches the Libertinus paper; "sans" flips the set

# Register Libertinus from the local TeX install if available (matplotlib cannot
# see TeX-only fonts otherwise); a harmless no-op when absent, in which case the
# serif fallback chain resolves to STIX Two Text.
for _f in glob.glob(
    "/usr/local/texlive/*/texmf-dist/fonts/opentype/public/"
    "libertinus-fonts/LibertinusSerif-*.otf"
):
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass


def _font_rc() -> dict:
    if FONT_MODE == "serif":
        return {
            "font.family": "serif",
            "font.serif": ["Libertinus Serif", "STIX Two Text", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        }
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
    }


def set_style():
    """Apply the unified figure theme. Called once at import (idempotent)."""
    # No font_scale: explicit point sizes below are authoritative.
    sns.set_theme(context="paper", style="whitegrid")
    plt.rcParams.update({
        # resolution / export
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # font-size hierarchy (on-page ~0.92x scale => ~9 pt primary text, uniform)
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.title_fontsize": 8.5,
        "figure.titlesize": 11,
        # look
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        # theme as the default: a bare plot draws the custom palette / sequential
        # cmap, never matplotlib's tab10 / viridis. Set after sns.set_theme (which
        # resets rcParams) so these win.
        "axes.prop_cycle": matplotlib.cycler(color=CATEGORICAL),
        "image.cmap": SEQ_CMAP,
        **_font_rc(),
    })
    sns.set_palette(CATEGORICAL)  # seaborn's own default palette tracks the theme too


set_style()


def save_fig(fig, name: str, formats=("png", "pdf")) -> list[str]:
    """Save a figure under outputs/figures/<name>.<fmt>; returns the paths."""
    paths = []
    for fmt in formats:
        p = C.FIG_DIR / f"{name}.{fmt}"
        fig.savefig(p)
        paths.append(str(p))
    plt.close(fig)
    return paths


def hbar_counts(series, title=None, xlabel="Count", top=None, ax=None, annot=True):
    """Horizontal bar chart of value counts, sorted descending.

    ``title`` is optional: in-figure titles are omitted in the paper (the LaTeX
    caption is the title), so callers pass ``title=None``. A truthy title still
    renders, for ad-hoc/exploratory use.
    """
    vc = series.value_counts()
    if top:
        vc = vc.head(top)
    vc = vc.sort_values()
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=figsize("FULL", max(2.2, 0.45 * len(vc) + 1)))
    ax.barh(vc.index.astype(str), vc.values, color=ACCENT)
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    if annot:
        for i, v in enumerate(vc.values):
            ax.text(v + max(vc.values) * 0.01, i, str(int(v)), va="center", fontsize=ANNOT)
    return (ax.figure, ax) if own else ax
