"""Single source of truth: paths, column mapping, variable roles, and the
Energy-Consciousness Index (ECI) scoring rubric.

The ECI is the central analytic construct of this study. Each of the five
behavioural survey items is mapped onto an ordinal "energy-consciousness" scale
where a HIGHER score means MORE energy-conserving behaviour. Mappings use robust
keyword matching (the survey text contains curly apostrophes and typos), and
every observed category MUST map or cleaning raises -- there is no silent
miscoding. The rubric is deliberately explicit and documented so it can be
audited and sensitivity-tested.
"""
from __future__ import annotations
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
CODE_DIR = Path(__file__).resolve().parent.parent          # .../code
ROOT = CODE_DIR.parent                                      # project root
RAW_DIR = ROOT / "data" / "raw"
RAW_CSV = RAW_DIR / "survey.csv"                           # merged, de-identified export
# The two original survey exports (byte-identical headers) were concatenated into
# this single file, and the finest-grained geographic field (divisional secretariat)
# was dropped to protect respondent confidentiality. That field is never used in the
# analysis, so reproducibility is unaffected. See data/raw/README.md for provenance.
RAW_CSVS = [RAW_CSV]                                       # load_raw() iterates this list
PROC_DIR = ROOT / "data" / "processed"
CLEAN_CSV = PROC_DIR / "cleaned.csv"
ECI_CSV = PROC_DIR / "eci_scores.csv"
CODEBOOK_CSV = PROC_DIR / "codebook.csv"
SUMMARY_JSON = PROC_DIR / "clean_summary.json"
FIG_DIR = CODE_DIR / "outputs" / "figures"
TBL_DIR = CODE_DIR / "outputs" / "tables"
for _d in (PROC_DIR, FIG_DIR, TBL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

N_EXPECTED = 70

# --------------------------------------------------------------------------- #
# Raw column index -> short code  (order matches the survey CSV export)
# --------------------------------------------------------------------------- #
COLCODE = {
    0: "src_mode", 1: "build_period", 2: "arch_design", 3: "house_type",
    4: "stories", 5: "area_sqft", 6: "rel_head", 7: "gender",
    8: "edu_attendance", 9: "occupation", 10: "food_exp", 11: "nonfood_exp",
    12: "wall_material", 13: "vent_raw", 14: "roof_type", 15: "red_notices",
    16: "pay_practice", 17: "wiring_pro", 18: "gen_raw", 19: "solar_raw",
    20: "billknow_raw", 21: "iron_raw", 22: "nightlight_raw",
    23: "metercheck_raw", 24: "enrating_raw", 25: "district",
}

# Short human-readable titles (used for figure axes / table headers)
LABEL = {
    "src_mode": "Data collection mode",
    "build_period": "Construction period",
    "arch_design": "Architectural design level",
    "house_type": "House type",
    "stories": "Number of stories",
    "area_sqft": "Floor area (sq ft)",
    "rel_head": "Relationship to head",
    "gender": "Gender",
    "edu_attendance": "Educational attendance",
    "occupation": "Main occupation",
    "food_exp": "Food expenditure (Rs/mo)",
    "nonfood_exp": "Non-food expenditure (Rs/mo)",
    "total_exp": "Total expenditure (Rs/mo)",
    "wall_material": "Outer-wall material",
    "roof_type": "Roof type",
    "red_notices": "Red notices (last year)",
    "pay_practice": "Bill-payment practice",
    "wiring_pro": "Professional wiring",
    "any_red_notice": "Any disconnection notice",
    "solar_user": "Solar user",
    "uses_renewable": "Uses renewable source",
    "area_per_story": "Floor area per story (sq ft)",
    "district": "District",
    "province": "Province",
    "eci": "Energy-Consciousness Index (0-100)",
    "eci_band": "Energy-consciousness band",
    "eci_billknow": "Bill-calculation literacy",
    "eci_iron": "Ironing efficiency",
    "eci_nightlight": "Night-lighting restraint",
    "eci_metercheck": "Meter-checking frequency",
    "eci_enrating": "Energy-rating awareness",
}

# Roles -- groups variables for thematic analysis.
ROLE = {
    "src_mode": "meta",
    "district": "context", "province": "context",
    "build_period": "dwelling", "arch_design": "dwelling", "house_type": "dwelling",
    "stories": "dwelling", "area_sqft": "dwelling", "wall_material": "dwelling",
    "roof_type": "dwelling", "area_per_story": "dwelling",
    "rel_head": "demographic", "gender": "demographic",
    "edu_attendance": "demographic", "occupation": "demographic",
    "food_exp": "ses", "nonfood_exp": "ses", "total_exp": "ses",
    "red_notices": "energy_system", "pay_practice": "energy_system",
    "wiring_pro": "energy_system",
    "eci_billknow": "behaviour_item", "eci_iron": "behaviour_item",
    "eci_nightlight": "behaviour_item", "eci_metercheck": "behaviour_item",
    "eci_enrating": "behaviour_item",
    "eci": "outcome", "eci_band": "outcome",
}

# --------------------------------------------------------------------------- #
# District -> Province (for geographic aggregation; n is too small per-district)
# --------------------------------------------------------------------------- #
DISTRICT_PROVINCE = {
    "Colombo": "Western", "Gampaha": "Western", "Kalutara": "Western",
    "Kandy": "Central", "Matale": "Central", "Nuwara Eliya": "Central",
    "Galle": "Southern", "Matara": "Southern", "Hambantota": "Southern",
    "Jaffna": "Northern", "Kilinochchi": "Northern", "Mannar": "Northern",
    "Vavuniya": "Northern", "Mullaitivu": "Northern",
    "Batticaloa": "Eastern", "Ampara": "Eastern", "Trincomalee": "Eastern",
    "Kurunegala": "North Western", "Puttalam": "North Western",
    "Anuradhapura": "North Central", "Polonnaruwa": "North Central",
    "Badulla": "Uva", "Monaragala": "Uva",
    "Ratnapura": "Sabaragamuwa", "Kegalle": "Sabaragamuwa",
}

# --------------------------------------------------------------------------- #
# ECI rubric: ordinal scorers (higher == more energy-conserving)
# Each scorer maps a raw category string -> int, via keyword matching.
# MAX defines the item ceiling used for [0,1] normalisation.
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return str(s).strip().lower().replace("’", "'")  # curly -> straight


def split_tokens(s: str) -> list[str]:
    """Split a multi-select string on top-level commas only.

    Some response options contain commas inside parentheses, e.g.
    "Agriculture equipment and systems (irrigation systems, etc)" -- a naive
    comma split would shatter these into junk tokens.
    """
    toks, depth, cur = [], 0, []
    for ch in str(s):
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            cur.append(ch)
        elif ch == "," and depth == 0:
            toks.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        toks.append("".join(cur))
    return [t for t in toks if t.strip()]


def score_billknow(v: str) -> int:
    """Bill-calculation literacy: 0 unaware, 1 passive source, 2 active learning."""
    t = _norm(v)
    if "not aware" in t:
        return 0
    if "media" in t or "another mean" in t:
        return 1
    if "reading the monthly bill" in t or "family member" in t \
            or "leco" in t or "ceb" in t:
        return 2
    raise ValueError(f"billknow: unmapped category {v!r}")


def score_iron(v: str) -> int:
    """Ironing efficiency by energy intensity (batching is more efficient)."""
    t = _norm(v)
    if "daily" in t:
        return 0
    if "twice a week" in t:
        return 2
    if "weekly" in t:
        return 3
    if "need arises" in t:
        return 1
    if "don't iron" in t or ("don" in t and "iron" in t):
        return 4
    raise ValueError(f"iron: unmapped category {v!r}")


def score_nightlight(v: str) -> int:
    """Night-lighting restraint (fewer lights left on == more conserving)."""
    t = _norm(v)
    if "more than two" in t:
        return 0
    if "less than two" in t:
        return 1
    if "don't keep any" in t or "any of the lights" in t:
        return 2
    raise ValueError(f"nightlight: unmapped category {v!r}")


def score_metercheck(v: str) -> int:
    """Meter-checking frequency for over-consumption."""
    t = _norm(v)
    if "never" in t:
        return 0
    if "rarely" in t:
        return 1
    if "some of the months" in t:
        return 2
    if "every month" in t:
        return 3
    raise ValueError(f"metercheck: unmapped category {v!r}")


def score_enrating(v: str) -> int:
    """Checks appliance energy rating when buying."""
    t = _norm(v)
    if t.startswith("yes"):
        return 1
    if t.startswith("no"):
        return 0
    raise ValueError(f"enrating: unmapped category {v!r}")


# item code -> (raw column code, scorer, max-score)
ECI_ITEMS = {
    "eci_billknow":  ("billknow_raw",   score_billknow,   2),
    "eci_iron":      ("iron_raw",       score_iron,       4),
    "eci_nightlight":("nightlight_raw", score_nightlight, 2),
    "eci_metercheck":("metercheck_raw", score_metercheck, 3),
    "eci_enrating":  ("enrating_raw",   score_enrating,   1),
}

# --------------------------------------------------------------------------- #
# Multi-select parsing: keyword -> indicator-column suffix
# Tokens are comma-separated; a sentinel ("none"/"not using") yields no flags.
# --------------------------------------------------------------------------- #
VENT_SPEC = {
    "window wall": "vent_window",
    "transparent wall": "vent_glasswall",
    "transparent roof": "vent_glassroof",
    "pergola": "vent_pergola",
    "other": "vent_other",
}
GEN_SPEC = {
    "grid": "gen_grid",
    "solar": "gen_solar",
    "hydro": "gen_hydro",
    "other green": "gen_othergreen",
}
GEN_SENTINEL = "none of the above"
SOLAR_SPEC = {
    "water heating": "solar_water",
    "outdoor lighting": "solar_outdoor",
    "cooking": "solar_cooking",
    "car charging": "solar_car",
    "agriculture": "solar_agri",
    "other": "solar_other",
}
SOLAR_SENTINEL = "not using solar"
