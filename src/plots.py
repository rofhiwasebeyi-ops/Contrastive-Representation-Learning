"""
Generates the figures.

Usage (after generating the JSON files per README):
    python plots.py --outdir ../outputs/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STRATEGY_LABELS = {
    "naive_finetuning": "Naive\nfine-tuning",
    "real_buffer_replay": "Real-buffer\nreplay",
    "generative_replay": "Generative\nreplay",
}
COLORS = {
    "naive_finetuning": "#94a3b8",
    "real_buffer_replay": "#60a5fa",
    "generative_replay": "#f97316",  # highlighted -- the proposed method
}


def plot_auroc_comparison(results: dict[str, dict], outpath: Path):
    """Grouped bar chart: AUROC per strategy, one group per home."""
    homes = list(results.keys())
    strategies = list(STRATEGY_LABELS.keys())

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(homes))
    width = 0.22

    for i, strat in enumerate(strategies):
        vals = [results[home]["auroc"].get(strat, np.nan) for home in homes]
        offset = (i - len(strategies) / 2) * width + width / 2
        ax.bar(x + offset, vals, width, label=STRATEGY_LABELS[strat],
               color=COLORS[strat], edgecolor="white")

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="Chance (0.5)")
    ax.set_ylabel("AUROC (injected-deviation detection)")
    ax.set_title("Anomaly-detection AUROC by strategy and home")
    ax.set_xticks(x)
    ax.set_xticklabels([h.capitalize() for h in homes])
    ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  wrote {outpath}")


def plot_storage_comparison(results: dict[str, dict], outpath: Path):
    """Bar chart: capped buffer vs unbounded buffer vs VAE, per home."""
    homes = [h for h in results if results[h].get("storage") is not None]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(homes))
    width = 0.25
    labels = ["Real buffer\n(capped, 30 days)", "Real buffer\n(unbounded)", "Generative replay\n(VAE, fixed)"]
    keys = ["capped_buffer_kb", "unbounded_buffer_kb_at_current_length", "vae_kb"]
    colors = ["#60a5fa", "#93c5fd", "#f97316"]

    for i, (key, label, color) in enumerate(zip(keys, labels, colors)):
        vals = [results[h]["storage"][key] for h in homes]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color, edgecolor="white")
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)

    ax.set_ylabel("Storage (KB)")
    ax.set_title("Storage/privacy comparison: what's retained, per strategy")
    ax.set_xticks(x)
    ax.set_xticklabels([h.capitalize() for h in homes])
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  wrote {outpath}")


def plot_config_comparison(tune_results: dict[str, dict], outpath: Path):
    """Grouped bar chart: generative_replay vs real_buffer_replay AUROC
    across the 5 hand-picked configs, one panel per home."""
    homes = list(tune_results.keys())
    fig, axes = plt.subplots(1, len(homes), figsize=(7 * len(homes), 5), squeeze=False)
    axes = axes[0]

    for ax, home in zip(axes, homes):
        configs = tune_results[home]["configs"]
        names = [c["config_name"] for c in configs]
        gen = [c["auroc"]["generative_replay"] for c in configs]
        real = [c["auroc"]["real_buffer_replay"] for c in configs]
        wins = sum(1 for g, r in zip(gen, real) if g > r)

        x = np.arange(len(names))
        width = 0.35
        ax.bar(x - width / 2, real, width, label="real_buffer_replay", color="#60a5fa")
        ax.bar(x + width / 2, gen, width, label="generative_replay", color="#f97316")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("AUROC")
        ax.set_title(f"{home.capitalize()}: generative_replay wins {wins}/{len(configs)} configs")
        ax.legend(fontsize=8)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)

    fig.suptitle("Hand-picked config comparison: generative vs. real-buffer replay")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  wrote {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="../outputs/results")
    ap.add_argument("--outdir", default="../outputs/figures")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    home_results = {}
    for home_file in ["aruba.json", "milan.json"]:
        path = results_dir / home_file
        if not path.exists():
            print(f"  (skipping {home_file} -- not found; run train.py --save-json first)")
            continue
        with open(path) as f:
            data = json.load(f)
        home_name = Path(data["home"]).stem
        auroc = {row["strategy"]: row["auroc"] for row in data["summary"]}
        home_results[home_name] = {"auroc": auroc, "storage": data.get("storage_comparison")}

    if home_results:
        print("Generating AUROC comparison figure...")
        plot_auroc_comparison(home_results, outdir / "auroc_comparison.png")
        print("Generating storage comparison figure...")
        plot_storage_comparison(home_results, outdir / "storage_comparison.png")
    else:
        print("No train.py --save-json results found -- skipping AUROC/storage figures.")

    tune_results = {}
    for tune_file in ["tune_aruba.json", "tune_milan.json"]:
        path = results_dir / tune_file
        if not path.exists():
            print(f"  (skipping {tune_file} -- not found; run tune.py --save-json first)")
            continue
        with open(path) as f:
            data = json.load(f)
        home_name = Path(data["home"]).stem
        tune_results[home_name] = data

    if tune_results:
        print("Generating config comparison figure...")
        plot_config_comparison(tune_results, outdir / "config_comparison.png")
    else:
        print("No tune.py --save-json results found -- skipping config comparison figure.")

    print(f"\nAll figures written to {outdir}/")


if __name__ == "__main__":
    main()
