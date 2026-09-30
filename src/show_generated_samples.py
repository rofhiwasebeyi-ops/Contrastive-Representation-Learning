"""
Produces the "examples of generated outputs" figure:
real daily routine feature vectors vs. synthetic ones sampled from the
trained generative-replay VAE, side by side. Uses the SAME GenerativeReplay
strategy class as the main pipeline (train.py)

Usage: python show_generated_samples.py --home ../data/raw/aruba.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import parse_casas_file
from features import build_schema, build_daily_features, N_HOUR_BINS
from strategies import GenerativeReplay


def train_generator_on_stream(X: torch.Tensor, reference_days: int, seed: int = 1) -> GenerativeReplay:
    """Runs the exact same online training loop as train.py's main pipeline
    for the generative_replay strategy, so the VAE we sample from here is
    the real one, not a separately-trained stand-in."""
    torch.manual_seed(seed)
    strat = GenerativeReplay(X.shape[1], resident_id=0, n_residents=1)
    for i in range(X.shape[0]):
        strat.observe_day(X[i:i+1], i, is_deviation=False)
    return strat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default="../data/raw/aruba.txt")
    ap.add_argument("--reference-days", type=int, default=14)
    ap.add_argument("--n-examples", type=int, default=4)
    ap.add_argument("--outdir", default="../outputs/figures")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    print(f"Parsing {args.home} ...")
    events, _ = parse_casas_file(args.home)
    schema = build_schema(events)
    daily, schema = build_daily_features(events, schema, reference_days=args.reference_days)
    X_np = daily[schema.feature_names].to_numpy(dtype=np.float32)
    X = torch.tensor(X_np, dtype=torch.float32)

    print("Training the generative-replay VAE online over the full stream "
          "(same procedure as train.py's main pipeline)...")
    strat = train_generator_on_stream(X, args.reference_days, seed=args.seed)

    # Sample synthetic days from the trained generator.
    torch.manual_seed(args.seed)
    synthetic = strat.vae.sample(args.n_examples, resident_id=0).detach().numpy()

    # Pick real days spread across the deployment (not just the first few,
    # so the comparison isn't cherry-picked from the easiest/most-trained part).
    real_idx = np.linspace(args.reference_days, X_np.shape[0] - 1, args.n_examples, dtype=int)
    real_days = X_np[real_idx]

    hour_start = len(schema.motion_sensors) + len(schema.door_sensors)
    hour_end = hour_start + N_HOUR_BINS

    # --- Figure 1: hour-of-day routine shape, real vs generated ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for i in range(args.n_examples):
        axes[0].plot(range(N_HOUR_BINS), real_days[i, hour_start:hour_end],
                     alpha=0.7, label=f"Day {real_idx[i]}")
        axes[1].plot(range(N_HOUR_BINS), synthetic[i, hour_start:hour_end],
                     alpha=0.7, label=f"Sample {i+1}")
    axes[0].set_title("Real days (hour-of-day activity)")
    axes[1].set_title("VAE-generated synthetic days (hour-of-day activity)")
    for ax in axes:
        ax.set_xlabel("Hour of day")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Activity level (z-scored)")
    fig.suptitle("Generated output example: real vs. synthetic daily routine shape")
    fig.tight_layout()
    outpath1 = Path(args.outdir) / "generated_samples_routine_shape.png"
    fig.savefig(outpath1, dpi=150)
    plt.close(fig)
    print(f"  wrote {outpath1}")

    # --- Figure 2: full feature-vector heatmap, real vs generated ---
    # Percentile-clipped color scale, not full min/max -- a couple of extreme
    # values from the heavy-tailed sensor would
    # otherwise wash out everything else on a linear 0-to-max scale.
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    vmin, vmax = np.percentile(X_np, [2, 90])
    im0 = axes[0].imshow(real_days, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Real days (full 72-dim feature vector)")
    axes[0].set_yticks(range(args.n_examples))
    axes[0].set_yticklabels([f"Day {i}" for i in real_idx])
    im1 = axes[1].imshow(synthetic, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("VAE-generated synthetic days (full 72-dim feature vector)")
    axes[1].set_yticks(range(args.n_examples))
    axes[1].set_yticklabels([f"Sample {i+1}" for i in range(args.n_examples)])
    axes[1].set_xlabel("Feature index (motion | door | hour-of-day | temperature)")
    fig.colorbar(im0, ax=axes, shrink=0.8, label="z-scored feature value")
    fig.suptitle("Generated output example: full feature vectors, real vs. synthetic")
    outpath2 = Path(args.outdir) / "generated_samples_heatmap.png"
    fig.savefig(outpath2, dpi=150)
    plt.close(fig)
    print(f"  wrote {outpath2}")


if __name__ == "__main__":
    main()
