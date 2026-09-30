"""
Aggregates parsed CASAS sensor events into one fixed-length feature vector per
resident-day. This is the "session vector" equivalent for the smart-home
domain (analogous to a keystroke-dynamics session vector), and is what the
contrastive encoder actually consumes -- never raw sensor events directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_loader import sensor_type

N_HOUR_BINS = 24


@dataclass
class FeatureSchema:
    """Fixed feature layout, learned once from a reference window so every
    day (including days with sensors not present are handled) produces a
    vector of the same length and in the same column order."""
    motion_sensors: list[str]
    door_sensors: list[str]
    temperature_sensors: list[str]
    has_presence: bool = False  # whether ENTERHOME/LEAVEHOME events exist in this home
    ref_stats: dict = field(default_factory=dict)  # per-feature (min, max) for scaling

    @property
    def dim(self) -> int:
        d = (
            len(self.motion_sensors)          # motion event counts
            + len(self.door_sensors)           # door event counts
            + N_HOUR_BINS                      # hour-of-day activity histogram
            + 2 * len(self.temperature_sensors)  # mean + std per temp sensor
        )
        if self.has_presence:
            d += 4  # leave_count, enter_count, first_leave_hour, last_enter_hour
        return d

    @property
    def feature_names(self) -> list[str]:
        names = [f"motion_{s}" for s in self.motion_sensors]
        names += [f"door_{s}" for s in self.door_sensors]
        names += [f"hour_{h:02d}" for h in range(N_HOUR_BINS)]
        for s in self.temperature_sensors:
            names += [f"temp_{s}_mean", f"temp_{s}_std"]
        if self.has_presence:
            names += ["leave_count", "enter_count", "first_leave_hour", "last_enter_hour"]
        return names


def build_schema(events: pd.DataFrame) -> FeatureSchema:
    """Determine the fixed sensor vocabulary from the full event log.
    """
    types = events["sensor"].apply(sensor_type)
    motion = sorted(events.loc[types == "motion", "sensor"].unique())
    door = sorted(events.loc[types == "door", "sensor"].unique())
    temp = sorted(events.loc[types == "temperature", "sensor"].unique())
    has_presence = bool((types == "presence").any())
    return FeatureSchema(motion_sensors=motion, door_sensors=door,
                          temperature_sensors=temp, has_presence=has_presence)


def _daily_vector(day_events: pd.DataFrame, schema: FeatureSchema) -> np.ndarray:
    types = day_events["sensor"].apply(sensor_type)

    motion_counts = np.array(
        [(day_events.loc[types == "motion", "sensor"] == s).sum() for s in schema.motion_sensors],
        dtype=np.float64,
    )
    door_counts = np.array(
        [(day_events.loc[types == "door", "sensor"] == s).sum() for s in schema.door_sensors],
        dtype=np.float64,
    )

    # Hour-of-day histogram over motion + door events only (temperature is
    # ambient/continuous and not informative of *when* the resident is active).
    activity_mask = types.isin(["motion", "door"])
    hours = day_events.loc[activity_mask, "timestamp"].dt.hour
    hour_hist = np.bincount(hours, minlength=N_HOUR_BINS)[:N_HOUR_BINS].astype(np.float64)

    temp_feats = []
    for s in schema.temperature_sensors:
        vals = pd.to_numeric(day_events.loc[day_events["sensor"] == s, "value"], errors="coerce").dropna()
        if len(vals) > 0:
            temp_feats.extend([vals.mean(), vals.std(ddof=0) if len(vals) > 1 else 0.0])
        else:
            temp_feats.extend([np.nan, np.nan])  # filled by forward-fill across days later

    parts = [motion_counts, door_counts, hour_hist, np.array(temp_feats)]

    if schema.has_presence:
        presence_events = day_events[types == "presence"]
        leave_events = presence_events[presence_events["sensor"] == "LEAVEHOME"]
        enter_events = presence_events[presence_events["sensor"] == "ENTERHOME"]
        leave_count = float(len(leave_events))
        enter_count = float(len(enter_events))
        # NaN (filled cross-day like temperature) when the resident didn't
        # leave/return that day at all, rather than a misleading 0 or 24.
        first_leave_hour = leave_events["timestamp"].dt.hour.min() if len(leave_events) else np.nan
        last_enter_hour = enter_events["timestamp"].dt.hour.max() if len(enter_events) else np.nan
        parts.append(np.array([leave_count, enter_count,
                                float(first_leave_hour) if pd.notna(first_leave_hour) else np.nan,
                                float(last_enter_hour) if pd.notna(last_enter_hour) else np.nan]))

    return np.concatenate(parts)


def build_daily_features(
    events: pd.DataFrame,
    schema: FeatureSchema,
    reference_days: int = 14,
) -> tuple[pd.DataFrame, FeatureSchema]:
    """Turn a resident's full event log into one row per calendar day.

    Returns
    daily : DataFrame indexed by date, columns = schema.feature_names, plus
        a leading 'n_events' column (raw event count, useful as a sanity
        check / optional extra feature, not scaled).
    schema : the same schema, now with ref_stats populated for scaling.
    """
    events = events.copy()
    events["date"] = events["timestamp"].dt.date
    dates = sorted(events["date"].unique())

    rows = []
    n_events_per_day = []
    for d in dates:
        day_events = events[events["date"] == d]
        rows.append(_daily_vector(day_events, schema))
        n_events_per_day.append(len(day_events))

    raw = np.vstack(rows)
    daily = pd.DataFrame(raw, columns=schema.feature_names, index=pd.to_datetime(dates))
    daily.insert(0, "n_events", n_events_per_day)

    # Forward/back-fill any missing temperature readings (rare days with sensor gaps)
    # and missing presence-hour features (days with no leave/enter event at all).
    fill_cols = [c for c in daily.columns if c.startswith("temp_")]
    fill_cols += [c for c in ("first_leave_hour", "last_enter_hour") if c in daily.columns]
    daily[fill_cols] = daily[fill_cols].ffill().bfill()

    # Fit z-score standardisation on a reference window only (first
    # `reference_days`), consistent with an online setting where you can't
    # normalise using future data.
    ref = daily.iloc[:reference_days]
    feature_cols = schema.feature_names
    ref_mean = ref[feature_cols].mean()
    ref_std = ref[feature_cols].std(ddof=0).replace(0, 1.0)  # guard constant features
    schema.ref_stats = {"mean": ref_mean.to_dict(), "std": ref_std.to_dict()}

    scaled = (daily[feature_cols] - ref_mean) / ref_std
    # Days after the reference window may have large |z| if behaviour
    # genuinely shifts
    daily[feature_cols] = scaled

    return daily, schema


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import parse_casas_file

    path = sys.argv[1] if len(sys.argv) > 1 else "../data/raw/aruba.txt"
    events, stats = parse_casas_file(path)
    schema = build_schema(events)
    daily, schema = build_daily_features(events, schema)

    print(f"Schema dim: {schema.dim} ({len(schema.motion_sensors)} motion, "
          f"{len(schema.door_sensors)} door, {N_HOUR_BINS} hour bins, "
          f"{len(schema.temperature_sensors)} temp sensors x2)")
    print(f"Daily feature matrix: {daily.shape}")
    print(daily.iloc[:3, :6])
