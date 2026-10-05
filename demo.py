"""
Demo script for a presentation walkthrough. Runs ONLY the fast
pieces -- the full evidence (20-trial tuning search, both homes) takes
~15-20 minutes and should already be pre-generated before the demo.

Usage: python demo.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"
ARUBA = ROOT / "data" / "raw" / "aruba.txt"


def banner(text: str):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70 + "\n")


def check_data_present():
    if not ARUBA.exists():
        print(f"ERROR: {ARUBA} not found.")
        print("Download from https://zenodo.org/records/17180309 and place")
        print("aruba.txt in data/raw/ before running the demo.")
        sys.exit(1)


def run(args: list[str]):
    result = subprocess.run([sys.executable, *args], cwd=SRC)
    if result.returncode != 0:
        print("\nDemo step failed -- stopping.")
        sys.exit(result.returncode)


def main():
    check_data_present()

    banner(
        "PART 1: Live run on real Aruba data (1.7M raw sensor events, 220 days)\n"
        "Parsing -> daily feature engineering -> 3 continual-learning strategies\n"
        "-> injected-deviation AUROC -> storage comparison."
    )
    run([
        "train.py", "--home", "../data/raw/aruba.txt", "--reference-days", "14",
        "--inject-real-deviations", "15", "--injection-seed", "1",
    ])

    banner(
        "PART 2: Proving the config-comparison harness itself runs (~30s)"
    )
    run([
        "tune.py", "--home", "../data/raw/aruba.txt", "--reference-days", "14",
        "--n-deviations", "15", "--seeds", "1",
    ])

    banner(
        "Demo complete."
    )


if __name__ == "__main__":
    main()
