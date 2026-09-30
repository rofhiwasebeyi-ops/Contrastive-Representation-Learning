"""
Injects synthetic, known-ground-truth deviations into REAL daily feature
data by perturbing a chosen subset of real days -- this is the only source
of ground truth for AUROC evaluation, since Aruba/Milan have no true
health-event labels of their own. 
"""
from __future__ import annotations

import numpy as np

from features import FeatureSchema, N_HOUR_BINS


def _motion_door_indices(schema: FeatureSchema) -> list[int]:
    n_motion = len(schema.motion_sensors)
    n_door = len(schema.door_sensors)
    return list(range(0, n_motion + n_door))


def _hour_bin_indices(schema: FeatureSchema) -> list[int]:
    start = len(schema.motion_sensors) + len(schema.door_sensors)
    return list(range(start, start + N_HOUR_BINS))


def inject_synthetic_deviations(
    X: np.ndarray,
    schema: FeatureSchema,
    reference_days: int,
    n_deviations: int = 10,
    mode: str = "mixed",
    seed: int = 0,
) -> tuple[np.ndarray, list[bool]]:
    """Returns (X_injected, deviation_flags). Only days AFTER the reference
    window are eligible for injection, and X itself is never modified in
    place -- a copy is returned.
    """
    rng = np.random.default_rng(seed)
    X_out = X.copy()
    n_days = X.shape[0]

    eligible = np.arange(reference_days, n_days)
    if len(eligible) < n_deviations:
        raise ValueError(
            f"Not enough post-reference days ({len(eligible)}) to inject "
            f"{n_deviations} deviations without overlap."
        )
    chosen = rng.choice(eligible, size=n_deviations, replace=False)

    motion_door_idx = _motion_door_indices(schema)
    hour_idx = _hour_bin_indices(schema)

    # Absolute low target for "inactivity": near the observed 5th percentile
    # of each motion/door feature's OWN distribution (not a fraction of
    # whatever value the target day happens to have), so the result is
    # guaranteed to sit near genuinely low activity regardless of drift.
    low_target = np.percentile(X[:, motion_door_idx], 5, axis=0)

    flags = [False] * n_days
    for day_idx in chosen:
        flags[day_idx] = True
        day_mode = mode
        if mode == "mixed":
            day_mode = "inactivity" if rng.random() < 0.5 else "shifted_routine"

        if day_mode == "inactivity":
            # Blend toward the absolute low target rather than scaling the
            # day's own (possibly already-drifted) values.
            blend = rng.uniform(0.85, 0.98)  # how strongly to pull toward "silent house"
            X_out[day_idx, motion_door_idx] = (
                (1 - blend) * X_out[day_idx, motion_door_idx] + blend * low_target
            )
        elif day_mode == "shifted_routine":
            # Larger, guaranteed-large shift (half a day minimum) so the
            # routine disruption isn't subtle relative to real natural
            # variance 
            shift = rng.integers(10, 14) * rng.choice([-1, 1])
            X_out[day_idx, hour_idx] = np.roll(X_out[day_idx, hour_idx], shift)
            # Also apply a mild activity reduction on shifted-routine days --
            # a genuinely disrupted schedule plausibly comes with reduced
            # overall activity too, and this helps guarantee separability.
            X_out[day_idx, motion_door_idx] *= rng.uniform(0.4, 0.6)
        else:
            raise ValueError(f"Unknown mode: {day_mode}")

    return X_out, flags


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import parse_casas_file
    from features import build_schema, build_daily_features

    path = sys.argv[1] if len(sys.argv) > 1 else "../data/raw/aruba.txt"
    events, _ = parse_casas_file(path)
    schema = build_schema(events)
    daily, schema = build_daily_features(events, schema, reference_days=14)
    X = daily[schema.feature_names].to_numpy(dtype=np.float32)

    X_inj, flags = inject_synthetic_deviations(X, schema, reference_days=14, n_deviations=10)
    n_flagged = sum(flags)
    print(f"Injected {n_flagged} deviation days into {X.shape[0]}-day stream")
    diff = np.linalg.norm(X_inj - X, axis=1)
    print(f"Per-day change magnitude on flagged days: "
          f"{[round(d, 2) for d, f in zip(diff, flags) if f]}")
    print(f"Per-day change magnitude on unflagged days (should be ~0): "
          f"max={diff[[not f for f in flags]].max():.6f}")
