#!/usr/bin/env python3
"""Render a compact PNG report for a profiled 7B training run."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "search-r1-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def parse_wall_time(value: str) -> dt.datetime | None:
    if not value or value == "nvidia-smi not found":
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def read_gpu_rows(path: Path) -> dict[str, list[tuple[float, float, float]]]:
    series: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    if not path.exists():
        return series

    start_time: dt.datetime | None = None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            wall_time = parse_wall_time(row.get("wall_time", ""))
            if wall_time is None:
                continue
            if start_time is None:
                start_time = wall_time
            elapsed_min = (wall_time - start_time).total_seconds() / 60.0
            gpu_index = (row.get("gpu_index") or "unknown").strip()
            try:
                memory = float((row.get("memory_used_mb") or "").strip())
                util = float((row.get("gpu_util_percent") or "").strip())
            except ValueError:
                continue
            series[gpu_index].append((elapsed_min, memory, util))
    return series


def format_duration(seconds: str) -> str:
    try:
        total = int(float(seconds))
    except (TypeError, ValueError):
        return "n/a"
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def count_checkpoints(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def render_report(run_dir: Path, output: Path) -> None:
    summary = read_key_values(run_dir / "summary.txt")
    env = read_key_values(run_dir / "env.txt")
    gpu_series = read_gpu_rows(run_dir / "gpu_memory.csv")
    ckpt_count = count_checkpoints(run_dir / "checkpoints.txt")

    experiment = summary.get("experiment_name") or env.get("EXPERIMENT_NAME") or run_dir.name
    exit_code = summary.get("exit_code", "n/a")
    status = "SUCCESS" if exit_code == "0" else f"EXIT {exit_code}"
    duration = format_duration(summary.get("duration_seconds", ""))
    peak_mem = summary.get("peak_gpu_memory", "n/a")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": "#f7f8fa",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#d0d5dd",
            "grid.color": "#e5e7eb",
            "grid.linewidth": 0.8,
        }
    )

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[0.9, 2.2, 1.6])

    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(0.0, 0.86, "Search-R1 7B Training Report", fontsize=20, weight="bold", color="#111827")
    title_ax.text(0.0, 0.50, experiment, fontsize=12, color="#374151")

    cards = [
        ("Status", status),
        ("Duration", duration),
        ("Peak GPU Memory", peak_mem),
        ("Checkpoints", str(ckpt_count)),
    ]
    for idx, (label, value) in enumerate(cards):
        x = 0.02 + idx * 0.24
        title_ax.text(x, 0.08, label, fontsize=9, color="#667085")
        title_ax.text(x, 0.24, value, fontsize=14, weight="bold", color="#111827")

    mem_ax = fig.add_subplot(gs[1, :])
    if gpu_series:
        for gpu_index in sorted(gpu_series, key=lambda x: int(x) if x.isdigit() else x):
            points = gpu_series[gpu_index]
            mem_ax.plot(
                [item[0] for item in points],
                [item[1] for item in points],
                linewidth=1.8,
                label=f"GPU {gpu_index}",
            )
        mem_ax.set_title("GPU Memory Usage")
        mem_ax.set_xlabel("Elapsed minutes")
        mem_ax.set_ylabel("Memory used (MB)")
        mem_ax.grid(True)
        mem_ax.legend(ncol=min(4, max(1, len(gpu_series))), fontsize=8, frameon=False)
    else:
        mem_ax.text(0.5, 0.5, "No GPU samples found", ha="center", va="center", fontsize=13, color="#667085")
        mem_ax.set_axis_off()

    util_ax = fig.add_subplot(gs[2, :2])
    if gpu_series:
        for gpu_index in sorted(gpu_series, key=lambda x: int(x) if x.isdigit() else x):
            points = gpu_series[gpu_index]
            util_ax.plot(
                [item[0] for item in points],
                [item[2] for item in points],
                linewidth=1.3,
                label=f"GPU {gpu_index}",
            )
        util_ax.set_title("GPU Utilization")
        util_ax.set_xlabel("Elapsed minutes")
        util_ax.set_ylabel("Utilization (%)")
        util_ax.set_ylim(0, 100)
        util_ax.grid(True)
    else:
        util_ax.text(0.5, 0.5, "No utilization samples", ha="center", va="center", color="#667085")
        util_ax.set_axis_off()

    details_ax = fig.add_subplot(gs[2, 2:])
    details_ax.axis("off")
    details = [
        ("Run mode", summary.get("run_mode", env.get("RUN_MODE", "n/a"))),
        ("Algorithm", summary.get("algo", env.get("ALGO", "n/a"))),
        ("Train samples", env.get("TRAIN_DATA_NUM", "n/a")),
        ("Val samples", env.get("VAL_DATA_NUM", "n/a")),
        ("Total steps", env.get("TOTAL_TRAINING_STEPS", "n/a")),
        ("Save freq", env.get("SAVE_FREQ", "n/a")),
        ("Batch size", env.get("TRAIN_BATCH_SIZE", "n/a")),
        ("Model", env.get("BASE_MODEL", "n/a")),
    ]
    details_ax.text(0.0, 0.95, "Run Configuration", fontsize=12, weight="bold", color="#111827")
    for row_idx, (label, value) in enumerate(details):
        y = 0.82 - row_idx * 0.10
        details_ax.text(0.00, y, label, color="#667085")
        details_ax.text(0.34, y, str(value), color="#111827")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output = args.output.resolve() if args.output else run_dir / "report.png"
    render_report(run_dir, output)
    print(output)


if __name__ == "__main__":
    main()
