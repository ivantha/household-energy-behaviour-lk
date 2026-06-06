"""Reproduce the entire analysis: runs 00_clean.py then every NN_*.py theme
script in order, regenerating all processed data, figures, and tables.

Usage:  uv run python run_all.py
"""
from __future__ import annotations
import sys, subprocess, glob, os, time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def discover():
    scripts = ["00_clean.py"]
    scripts += sorted(p.name for p in HERE.glob("[0-9][0-9]_*.py")
                      if p.name != "00_clean.py")
    # de-dup while preserving order
    seen, ordered = set(), []
    for s in scripts:
        if s not in seen and (HERE / s).exists():
            seen.add(s)
            ordered.append(s)
    return ordered


def main():
    scripts = discover()
    print("Running analysis pipeline:", ", ".join(scripts))
    failures = []
    for s in scripts:
        print(f"\n{'='*70}\n>>> {s}\n{'='*70}")
        t0 = time.time()
        r = subprocess.run([sys.executable, s], cwd=HERE)
        dt = time.time() - t0
        status = "ok" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
        print(f"<<< {s}: {status} in {dt:.1f}s")
        if r.returncode != 0:
            failures.append(s)
    print("\n" + "=" * 70)
    if failures:
        print("PIPELINE FAILURES:", failures)
        sys.exit(1)
    print(f"Pipeline complete: {len(scripts)} scripts OK.")
    figs = glob.glob(str(HERE / "outputs" / "figures" / "*.png"))
    tbls = glob.glob(str(HERE / "outputs" / "tables" / "*.csv"))
    print(f"Figures: {len(figs)} | Tables: {len(tbls)}")


if __name__ == "__main__":
    main()
