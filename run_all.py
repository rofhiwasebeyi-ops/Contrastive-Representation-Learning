"""
Reproduces every result and figure.

Usage: python run_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"
ARUBA = ROOT / "data" / "raw" / "aruba.txt"
MILAN = ROOT / "data" / "raw" / "milan.txt"


def check_data_present():
    missing = [f for f in (ARUBA, MILAN) if not f.exists()]
    if missing:
        print("ERROR: missing real data file(s):")
        for f in missing:
            print(f"  {f}")
        print("\nDownload new_labeled_data.zip from https://zenodo.org/records/17180309,")
        print("unzip it, and place aruba.txt / milan.txt in data/raw/ before running this script.")
        sys.exit(1)


def run(step: str, args: list[str]):
    print(f"\n=== {step} ===")
    result = subprocess.run([sys.executable, *args], cwd=SRC)
    if result.returncode != 0:
        print(f"\nERROR: step failed ({step}). Stopping.")
        sys.exit(result.returncode)


def main():
    check_data_present()

    run("[1/5] Main results: Aruba, with injected deviations", [
        "train.py", "--home", "../data/raw/aruba.txt", "--reference-days", "14",
        "--inject-real-deviations", "15", "--injection-seed", "1",
        "--save-json", "../outputs/results/aruba.json",
    ])

    run("[2/5] Main results: Milan, with injected deviations", [
        "train.py", "--home", "../data/raw/milan.txt", "--reference-days", "14",
        "--inject-real-deviations", "8", "--injection-seed", "1",
        "--save-json", "../outputs/results/milan.json",
    ])

    run("[3/5] Config comparison: Aruba (5 hand-picked configs)", [
        "tune.py", "--home", "../data/raw/aruba.txt", "--reference-days", "14",
        "--n-deviations", "15", "--seeds", "1,2,3",
        "--save-json", "../outputs/results/tune_aruba.json",
    ])
    run("[4/5] Config comparison: Milan (5 hand-picked configs)", [
        "tune.py", "--home", "../data/raw/milan.txt", "--reference-days", "14",
        "--n-deviations", "8", "--seeds", "1,2,3",
        "--save-json", "../outputs/results/tune_milan.json",
    ])

    run("[5/5] Generating figures (report + generated-output examples)", [
        "plots.py", "--results-dir", "../outputs/results", "--outdir", "../outputs/figures",
    ])
    run("[5b/5] Generated-output examples", [
        "show_generated_samples.py", "--home", "../data/raw/aruba.txt",
        "--outdir", "../outputs/figures",
    ])

    print("\nDone. Results in outputs/results/, figures in outputs/figures/.")


if __name__ == "__main__":
    main()
