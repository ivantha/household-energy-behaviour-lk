"""Data loading helpers. All analysis scripts load through here so the column
codes and dtypes stay consistent."""
from __future__ import annotations
import pandas as pd
from . import config as C


def load_raw(rename: bool = True) -> pd.DataFrame:
    """Raw survey export(s), concatenated into the full analysis sample.

    Reads every file in ``C.RAW_CSVS`` -- the original export plus any responses
    collected afterwards -- in order. Their headers are byte-identical, so the
    rows stack directly; the original export is never modified. With rename=True,
    columns are renamed to short codes.
    """
    frames = [pd.read_csv(p) for p in C.RAW_CSVS]
    df = pd.concat(frames, ignore_index=True)
    if rename:
        df.columns = [C.COLCODE.get(i, c) for i, c in enumerate(df.columns)]
    return df


def load_clean() -> pd.DataFrame:
    """Cleaned analysis table produced by 00_clean.py."""
    df = pd.read_csv(C.CLEAN_CSV)
    if "eci_band" in df:
        df["eci_band"] = pd.Categorical(
            df["eci_band"], categories=["Low", "Medium", "High"], ordered=True
        )
    return df


def load_eci() -> pd.DataFrame:
    return pd.read_csv(C.ECI_CSV)
