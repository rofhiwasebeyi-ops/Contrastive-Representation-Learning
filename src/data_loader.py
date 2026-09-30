"""
CASAS raw-text parser.

CASAS smart-home text files (aruba.txt, milan.txt) are whitespace-delimited,
one sensor event per line, in roughly this form:
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_LINE_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+"
    r"(?P<sensor>\S+)\s+"
    r"(?P<value>\S+)"
    r"(?:\s+(?P<label>.*))?$"
)


@dataclass
class ParseStats:
    total_lines: int
    parsed: int
    skipped: int


def parse_casas_file(path: str | Path) -> tuple[pd.DataFrame, ParseStats]:
    """Parse a single CASAS home file (e.g. aruba.txt) into a tidy DataFrame.

    Returns
    events : DataFrame with columns
        [timestamp, sensor, value, activity_label, activity_event]
        activity_event is one of {None, "begin", "end"} when a label is present.
    stats : ParseStats
        Counts for sanity-checking the parse (flag if skipped is high).
    """
    path = Path(path)
    rows = []
    total = 0
    skipped = 0

    with path.open("r", errors="replace") as f:
        for line in f:
            total += 1
            line = line.strip()
            if not line:
                skipped += 1
                continue
            m = _LINE_RE.match(line)
            if not m:
                skipped += 1
                continue

            ts_str = f"{m.group('date')} {m.group('time')}"
            try:
                ts = pd.Timestamp(ts_str)
            except ValueError:
                skipped += 1
                continue

            label_raw = m.group("label")
            activity_label = None
            activity_event = None
            if label_raw:
                label_raw = label_raw.strip()
                lower = label_raw.lower()
                if lower.endswith("begin"):
                    activity_event = "begin"
                    activity_label = label_raw[: -len("begin")].strip()
                elif lower.endswith("end"):
                    activity_event = "end"
                    activity_label = label_raw[: -len("end")].strip()
                else:
                    activity_label = label_raw

            rows.append(
                {
                    "timestamp": ts,
                    "sensor": m.group("sensor"),
                    "value": m.group("value"),
                    "activity_label": activity_label,
                    "activity_event": activity_event,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)

    stats = ParseStats(total_lines=total, parsed=len(df), skipped=skipped)
    return df, stats


def sensor_type(sensor_id: str) -> str:
    """Classify a CASAS sensor ID into a coarse type by its prefix.
    """
    sid = sensor_id.upper()
    if sid in ("ENTERHOME", "LEAVEHOME"):
        return "presence"
    if sid.startswith("M"):
        return "motion"
    if sid.startswith("D"):
        return "door"
    if sid.startswith("T"):
        return "temperature"
    if sid.startswith("AD") or sid.startswith("L"):
        return "ambient"
    return "other"


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python data_loader.py <path_to_home.txt>")
        sys.exit(1)

    events, stats = parse_casas_file(sys.argv[1])
    print(f"Parsed {stats.parsed}/{stats.total_lines} lines ({stats.skipped} skipped)")
    if not events.empty:
        print(events.head())
        print("\nSensor type counts:")
        print(events["sensor"].apply(sensor_type).value_counts())
        print(f"\nDate range: {events['timestamp'].min()} -> {events['timestamp'].max()}")
