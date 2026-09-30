"""
Hyperparameter comparison: a small, hand-picked set of configurations evaluated 
the honest way -- real AUROC against synthetic deviations injected into real data, 
averaged across a few seeds so a single lucky/unlucky seed can't decide a "best" configuration.

Each config specifies SHARED encoder params plus STRATEGY-SPECIFIC params for
real_buffer_replay and generative_replay.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import parse_casas_file
from features import build_schema, build_daily_features
from inject_deviations import inject_synthetic_deviations
from strategies import STRATEGY_REGISTRY, NaiveFineTuning, RealBufferReplay, GenerativeReplay
from train import evaluate_detection

# Five hand-picked configurations spanning small/large capacity and
# conservative/aggressive learning rates.
CONFIGS = [
    {
        "name": "default",
        "shared": {"embed_dim": 32, "hidden_dim": 64, "lr": 1e-3, "temperature": 0.5,
                   "jitter_std": 0.05, "dropout_p": 0.1, "prototype_momentum": 0.95},
        "real_buffer_replay": {"buffer_size": 30, "replay_batch": 8},
        "generative_replay": {"vae_lr": 1e-3, "vae_hidden_dim": 64, "vae_latent_dim": 16,
                               "kl_weight": 0.1, "replay_batch": 8},
    },
    {
        "name": "larger_capacity",
        "shared": {"embed_dim": 64, "hidden_dim": 128, "lr": 1e-3, "temperature": 0.5,
                   "jitter_std": 0.05, "dropout_p": 0.1, "prototype_momentum": 0.95},
        "real_buffer_replay": {"buffer_size": 30, "replay_batch": 8},
        "generative_replay": {"vae_lr": 1e-3, "vae_hidden_dim": 128, "vae_latent_dim": 32,
                               "kl_weight": 0.1, "replay_batch": 8},
    },
    {
        "name": "conservative_lr",
        "shared": {"embed_dim": 32, "hidden_dim": 64, "lr": 1e-4, "temperature": 0.5,
                   "jitter_std": 0.05, "dropout_p": 0.1, "prototype_momentum": 0.95},
        "real_buffer_replay": {"buffer_size": 30, "replay_batch": 8},
        "generative_replay": {"vae_lr": 1e-4, "vae_hidden_dim": 64, "vae_latent_dim": 16,
                               "kl_weight": 0.1, "replay_batch": 8},
    },
    {
        "name": "more_augmentation",
        "shared": {"embed_dim": 32, "hidden_dim": 64, "lr": 1e-3, "temperature": 0.7,
                   "jitter_std": 0.2, "dropout_p": 0.3, "prototype_momentum": 0.95},
        "real_buffer_replay": {"buffer_size": 30, "replay_batch": 8},
        "generative_replay": {"vae_lr": 1e-3, "vae_hidden_dim": 64, "vae_latent_dim": 16,
                               "kl_weight": 0.1, "replay_batch": 8},
    },
    {
        "name": "slower_prototype_adaptation",
        "shared": {"embed_dim": 32, "hidden_dim": 64, "lr": 1e-3, "temperature": 0.5,
                   "jitter_std": 0.05, "dropout_p": 0.1, "prototype_momentum": 0.8},
        "real_buffer_replay": {"buffer_size": 30, "replay_batch": 8},
        "generative_replay": {"vae_lr": 1e-3, "vae_hidden_dim": 64, "vae_latent_dim": 16,
                               "kl_weight": 0.1, "replay_batch": 8},
    },
]


def build_strategy(name: str, input_dim: int, config: dict, device: str = "cpu"):
    """Explicitly constructs each strategy with the right kwargs -- no
    introspection/kwarg-filtering magic, since we only have 3 known
    strategies and can just say directly what each one takes."""
    shared = config["shared"]
    if name == "naive_finetuning":
        return NaiveFineTuning(input_dim, device=device, **shared)
    if name == "real_buffer_replay":
        return RealBufferReplay(input_dim, device=device, **shared, **config["real_buffer_replay"])
    if name == "generative_replay":
        return GenerativeReplay(input_dim, device=device, resident_id=0, n_residents=1,
                                 **shared, **config["generative_replay"])
    raise ValueError(f"Unknown strategy: {name}")


def run_config(config: dict, injected_by_seed: dict, reference_days: int,
               device: str = "cpu") -> dict:
    """Runs all 3 strategies under `config`, once per seed in
    `injected_by_seed`. Returns {strategy_name: mean_auroc_across_seeds}."""
    per_strategy_aurocs = {name: [] for name in STRATEGY_REGISTRY}

    for seed, (X, deviation_flags) in injected_by_seed.items():
        for name in STRATEGY_REGISTRY:
            torch.manual_seed(seed)  # reset PER STRATEGY for independent initialisation
            strat = build_strategy(name, X.shape[1], config, device=device)

            results = []
            for i in range(X.shape[0]):
                r = strat.observe_day(X[i:i+1], i, deviation_flags[i])
                results.append(r)
                if i == reference_days - 1:
                    ref_scores = [res.deviation_score for res in results if not res.is_injected_deviation]
                    if len(ref_scores) > 1:
                        strat.deviation_threshold = float(np.mean(ref_scores) + 3 * np.std(ref_scores))
                elif i > reference_days and (i - reference_days) % reference_days == 0:
                    trailing = results[max(0, i - reference_days):i]
                    trailing_scores = [res.deviation_score for res in trailing if not res.is_injected_deviation]
                    if len(trailing_scores) > 1:
                        strat.deviation_threshold = float(
                            np.mean(trailing_scores) + 3 * np.std(trailing_scores))

            metrics = evaluate_detection(results, reference_days)
            per_strategy_aurocs[name].append(metrics.get("auroc", float("nan")))

    return {name: float(np.nanmean(v)) for name, v in per_strategy_aurocs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--reference-days", type=int, default=14)
    ap.add_argument("--n-deviations", type=int, default=15)
    ap.add_argument("--seeds", type=str, default="1,2,3")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save-json", type=str, default="")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"Loading {args.home} ...")
    events, _ = parse_casas_file(args.home)
    schema = build_schema(events)
    daily, schema = build_daily_features(events, schema, reference_days=args.reference_days)
    X_base = daily[schema.feature_names].to_numpy(dtype=np.float32)

    # Injection is done ONCE per seed here so
    # every config is evaluated against the exact same set of injected days.
    injected_by_seed = {}
    for seed in seeds:
        X_inj, flags = inject_synthetic_deviations(
            X_base, schema, reference_days=args.reference_days,
            n_deviations=args.n_deviations, seed=seed,
        )
        injected_by_seed[seed] = (torch.tensor(X_inj, dtype=torch.float32, device=args.device), flags)

    print(f"\nRunning {len(CONFIGS)} hand-picked configs x {len(seeds)} seeds each...\n")

    all_results = []
    for config in CONFIGS:
        t0 = time.time()
        mean_per_strategy = run_config(config, injected_by_seed, args.reference_days, device=args.device)
        elapsed = time.time() - t0
        all_results.append({"config_name": config["name"], "auroc": mean_per_strategy, "seconds": elapsed})
        summary = "  ".join(f"{name.split('_')[0]}={auroc:.3f}" for name, auroc in mean_per_strategy.items())
        print(f"{config['name']:<28} ({elapsed:5.1f}s): {summary}")

    print("\n" + "=" * 60)
    gen_vals = [r["auroc"]["generative_replay"] for r in all_results]
    real_vals = [r["auroc"]["real_buffer_replay"] for r in all_results]
    wins = sum(1 for g, r in zip(gen_vals, real_vals) if g > r)
    print(f"generative_replay beat real_buffer_replay in {wins}/{len(CONFIGS)} configs")
    print(f"generative_replay: mean={np.mean(gen_vals):.3f}")
    print(f"real_buffer_replay: mean={np.mean(real_vals):.3f}")

    if args.save_json:
        import json
        payload = {"home": args.home, "reference_days": args.reference_days,
                    "n_deviations": args.n_deviations, "seeds": seeds,
                    "configs": all_results}
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved results to {args.save_json}")

    return all_results


if __name__ == "__main__":
    main()
