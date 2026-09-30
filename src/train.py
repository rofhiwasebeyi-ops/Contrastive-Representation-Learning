"""
End-to-end pipeline runner: parse -> feature-engineer -> run all continual
learning strategies + baseline over the chronological day stream -> evaluate
-> print a comparison table.

Usage:
    python train.py --home ../data/raw/aruba.txt --reference-days 14 \
        --inject-real-deviations 15 --injection-seed 1

Real CASAS data has no true anomaly ground truth, so --inject-real-deviations
perturbs a chosen number of real post-reference days to give a ground-truth AUROC.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import parse_casas_file
from features import build_schema, build_daily_features
from strategies import STRATEGY_REGISTRY
from baselines import PCAPersonalBaseline
from inject_deviations import inject_synthetic_deviations


def compute_forgetting_measure(scores_per_strategy_over_time: list[float]) -> float:
    """Simplified forgetting measure: the largest drop from a running best
    (lower deviation-score volatility on genuinely normal days = more stable
    representation). Here we proxy it with the peak-to-trough range of the
    prototype-distance score on days NOT flagged as injected deviations --
    high volatility on days that should be unremarkable indicates instability."""
    arr = np.array(scores_per_strategy_over_time)
    if len(arr) < 2:
        return 0.0
    running_min = np.minimum.accumulate(arr)
    return float(np.max(arr - running_min))


def storage_comparison(n_days: int, feature_dim: int, capped_buffer_size: int,
                        vae_bytes: int) -> dict:
    """Reports storage under two different real-buffer-replay policies, since
    'real-buffer replay' is capped (e.g. 30 days, standard
    continual-learning practice).
    """
    bytes_per_day = feature_dim * 4  # float32
    capped_bytes = min(capped_buffer_size, n_days) * bytes_per_day
    unbounded_bytes_now = n_days * bytes_per_day
    crossover_day = vae_bytes / bytes_per_day if bytes_per_day > 0 else float("inf")

    return {
        "capped_buffer_kb": capped_bytes / 1024,
        "unbounded_buffer_kb_at_current_length": unbounded_bytes_now / 1024,
        "vae_kb": vae_bytes / 1024,
        "unbounded_crossover_day": crossover_day,
        "dataset_exceeds_crossover": n_days > crossover_day,
    }


def run_strategy(name: str, X: torch.Tensor, deviation_flags: list[bool],
                  reference_days: int, device: str = "cpu", seed: int = 0):
    torch.manual_seed(seed)
    input_dim = X.shape[1]

    kwargs = {}
    if name == "generative_replay":
        kwargs = dict(resident_id=0, n_residents=1)
    strat = STRATEGY_REGISTRY[name](input_dim, device=device, **kwargs)

    results = []
    for i in range(X.shape[0]):
        x_day = X[i : i + 1]
        r = strat.observe_day(x_day, i, deviation_flags[i])
        results.append(r)

        # Calibrate the deviation threshold from the reference window, then
        # RECALIBRATE periodically using a trailing window of recent scores
        # on non-flagged days.
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

    return strat, results


def evaluate_detection(results, reference_days: int) -> dict:
    post_ref = results[reference_days:]
    normal_scores = [r.deviation_score for r in post_ref if not r.is_injected_deviation]
    forgetting = compute_forgetting_measure(normal_scores)

    if not post_ref or len(set(r.is_injected_deviation for r in post_ref)) < 2:
        return {"auroc": float("nan"), "forgetting": forgetting,
                "n_flagged": sum(1 for r in post_ref if r.is_injected_deviation)}

    y_true = [int(r.is_injected_deviation) for r in post_ref]
    y_score = [r.deviation_score for r in post_ref]
    auroc = roc_auc_score(y_true, y_score)
    return {"auroc": auroc, "forgetting": forgetting}


def run_baseline(X_np: np.ndarray, deviation_flags: list[bool], reference_days: int) -> dict:
    X_ref = X_np[:reference_days]
    X_post = X_np[reference_days:]
    y_post = np.array(deviation_flags[reference_days:])

    if len(set(y_post.tolist())) < 2:
        return {}

    pca_base = PCAPersonalBaseline()
    pca_base.fit(X_ref)
    return {"pca_personal_baseline": roc_auc_score(y_post, pca_base.score(X_post))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True, help="Path to CASAS home .txt file")
    ap.add_argument("--reference-days", type=int, default=14)
    ap.add_argument("--inject-real-deviations", type=int, default=0,
                     help="Number of REAL days to synthetically perturb (Section 5.6b) "
                          "for ground-truth AUROC evaluation. Real data has no true "
                          "anomaly labels of its own, so this is the only source of "
                          "ground truth for AUROC.")
    ap.add_argument("--injection-seed", type=int, default=0)
    ap.add_argument("--save-json", type=str, default="",
                     help="If set, write the summary + storage comparison to this JSON path "
                          "(consumed by plots.py to generate report figures).")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print(f"Parsing {args.home} ...")
    events, stats = parse_casas_file(args.home)
    print(f"  {stats.parsed}/{stats.total_lines} lines parsed ({stats.skipped} skipped)")

    schema = build_schema(events)
    daily, schema = build_daily_features(events, schema, reference_days=args.reference_days)
    print(f"  Daily feature matrix: {daily.shape} (dim={schema.dim})")

    feature_cols = schema.feature_names
    X_np = daily[feature_cols].to_numpy(dtype=np.float32)

    if args.inject_real_deviations > 0:
        X_np, deviation_flags = inject_synthetic_deviations(
            X_np, schema, reference_days=args.reference_days,
            n_deviations=args.inject_real_deviations, seed=args.injection_seed,
        )
        print(f"  Injected {sum(deviation_flags)} synthetic deviations into real data "
              f"(seed={args.injection_seed})")
    else:
        deviation_flags = [False] * X_np.shape[0]  # no ground truth -- AUROC will be nan

    X = torch.tensor(X_np, dtype=torch.float32, device=args.device)

    print(f"\nRunning {len(STRATEGY_REGISTRY)} continual-learning strategies over "
          f"{X.shape[0]} days (reference window = first {args.reference_days} days)...\n")

    summary = []
    for name in STRATEGY_REGISTRY:
        t0 = time.time()
        strat, results = run_strategy(name, X, deviation_flags, args.reference_days, device=args.device)
        metrics = evaluate_detection(results, args.reference_days)
        elapsed = time.time() - t0
        summary.append({
            "strategy": name,
            "auroc": metrics.get("auroc", float("nan")),
            "forgetting": metrics.get("forgetting", float("nan")),
            "storage_bytes": strat.storage_bytes(),
            "seconds": round(elapsed, 2),
        })

    print("Running baseline comparison method (sub-RQ4)...\n")
    baseline_results = run_baseline(X_np, deviation_flags, args.reference_days)

    print(f"{'Strategy':<24}{'AUROC':>8}{'Forgetting':>12}{'Storage(KB)':>13}{'Time(s)':>9}")
    print("-" * 66)
    for row in summary:
        storage_kb = row["storage_bytes"] / 1024 if row["storage_bytes"] >= 0 else float("nan")
        print(f"{row['strategy']:<24}{row['auroc']:>8.3f}{row['forgetting']:>12.4f}"
              f"{storage_kb:>13.2f}{row['seconds']:>9.2f}")

    print(f"\n{'Baseline (sub-RQ4)':<24}{'AUROC':>8}")
    print("-" * 32)
    for name, auroc in baseline_results.items():
        print(f"{name:<24}{auroc:>8.3f}")

    # Storage/privacy comparison.
    vae_row = next((r for r in summary if r["strategy"] == "generative_replay"), None)
    storage = None
    if vae_row is not None:
        storage = storage_comparison(
            n_days=X.shape[0], feature_dim=schema.dim,
            capped_buffer_size=30, vae_bytes=vae_row["storage_bytes"],
        )
        print(f"\n{'Storage comparison':<40}{'KB':>10}")
        print("-" * 50)
        print(f"{'Real buffer (capped, 30 days)':<40}{storage['capped_buffer_kb']:>10.2f}")
        print(f"{'Real buffer (unbounded, all days so far)':<40}{storage['unbounded_buffer_kb_at_current_length']:>10.2f}")
        print(f"{'Generative replay (VAE, fixed)':<40}{storage['vae_kb']:>10.2f}")
        print(f"\nUnbounded buffer would exceed the VAE's fixed size after "
              f"~{storage['unbounded_crossover_day']:.0f} days of deployment "
              f"({'already exceeded' if storage['dataset_exceeds_crossover'] else 'not yet reached in this dataset'} "
              f"-- this dataset spans {X.shape[0]} days).")

    if args.save_json:
        import json
        payload = {
            "home": args.home,
            "n_days": int(X.shape[0]),
            "reference_days": args.reference_days,
            "inject_real_deviations": args.inject_real_deviations,
            "injection_seed": args.injection_seed,
            "summary": summary,
            "baseline_results": baseline_results,
            "storage_comparison": storage,
        }
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved results to {args.save_json}")


if __name__ == "__main__":
    main()
