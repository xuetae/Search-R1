#!/usr/bin/env python3
"""Extract numeric step metrics from a veRL/Search-R1 training log."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
METRIC_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_./-]*)\s*[:=]\s*"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
STEP_KEYS = ("training/global_step", "global_step", "trainer/global_step", "step")


def extract_metrics(log_path: Path) -> list[tuple[int, str, float]]:
    rows: list[tuple[int, str, float]] = []
    current_step: int | None = None

    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = ANSI_RE.sub("", raw_line)
            values = {match.group("key"): match.group("value") for match in METRIC_RE.finditer(line)}
            for key in STEP_KEYS:
                if key in values:
                    try:
                        current_step = int(float(values[key]))
                    except ValueError:
                        pass
                    break
            if current_step is None:
                continue
            for key, value in values.items():
                if key in STEP_KEYS:
                    continue
                try:
                    numeric_value = float(value)
                except ValueError:
                    continue
                rows.append((current_step, key, numeric_value))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = extract_metrics(args.log_path) if args.log_path.exists() else []
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("step", "metric", "value"))
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
