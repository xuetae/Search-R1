#!/usr/bin/env python3
"""xFusion training-task entrypoint for the 7B-base reproduction.

The platform training-task form only accepts a Python command. This wrapper
starts the local retriever, waits until it is healthy, then runs the profiled
two-GPU training script.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_PROJECT_DIRS = [
    Path("/filesdir/code/search-r1/projects/Search-R1"),
    Path("/workspace/filesdir/projects/Search-R1"),
    Path("/workspace/filesdir/search-r1/projects/Search-R1"),
    Path("/workspace/filesdir/code/search-r1/projects/Search-R1"),
]


def project_dir() -> Path:
    configured = os.environ.get("WORK_DIR") or os.environ.get("SEARCH_R1_PROJECT_DIR")
    if configured:
        return Path(configured).resolve()
    for candidate in DEFAULT_PROJECT_DIRS:
        if candidate.exists():
            return candidate
    return Path(__file__).resolve().parents[2]


def wait_for_retriever(url: str, proc: subprocess.Popen[str], timeout: int) -> None:
    docs_url = url.rsplit("/", 1)[0] + "/docs"
    deadline = time.time() + timeout
    last_error = ""

    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"retriever exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(docs_url, timeout=5) as response:
                if response.status < 500:
                    return
        except urllib.error.URLError as exc:
            last_error = str(exc)
        time.sleep(5)

    raise TimeoutError(f"retriever did not become ready in {timeout}s: {last_error}")


def stream_process(command: list[str], env: dict[str, str], cwd: Path) -> int:
    proc = subprocess.Popen(command, cwd=cwd, env=env)
    return proc.wait()


def terminate_process(proc: subprocess.Popen[str], grace_seconds: int = 20) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", default=os.environ.get("ALGO", "grpo"), choices=["grpo", "ppo"])
    parser.add_argument("--run-mode", default=os.environ.get("RUN_MODE", "two_gpu_paper"))
    parser.add_argument("--base-model-name", default=os.environ.get("BASE_MODEL_NAME", "llama-7b"))
    parser.add_argument(
        "--local-base-model",
        default=os.environ.get("LOCAL_BASE_MODEL"),
    )
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("TWO_GPU_CUDA_VISIBLE_DEVICES", "0,1"))
    parser.add_argument("--retriever-topk", default=os.environ.get("RETRIEVER_TOPK", "3"))
    parser.add_argument("--retriever-timeout", type=int, default=int(os.environ.get("RETRIEVER_TIMEOUT", "900")))
    parser.add_argument("--data_url", default=os.environ.get("DATA_URL"), help="xFusion injected dataset path.")
    parser.add_argument("--train_out", default=os.environ.get("TRAIN_OUT"), help="xFusion injected persistent output path.")
    parser.add_argument("--train_log", default=os.environ.get("TRAIN_LOG"), help="xFusion injected log output path.")
    parser.add_argument(
        "--skip-retriever",
        action="store_true",
        default=os.environ.get("SKIP_RETRIEVER", "").lower() in {"1", "true", "yes"},
        help="Use this only when another service already provides RETRIEVER_URL.",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[entry] ignoring unknown platform args: {unknown}", flush=True)
    return args


def main() -> int:
    args = parse_args()
    root = project_dir()
    script_dir = root / "reproduction" / "7b_base"
    algorithm_root = Path("/workspace/algorithm")
    if (root == algorithm_root or algorithm_root in root.parents) and Path("/workspace/filesdir").exists():
        persistent_root = Path("/workspace/filesdir")
    elif root.name == "Search-R1" and root.parent.name == "projects":
        persistent_root = root.parents[1]
    else:
        persistent_root = root
    local_base_model = args.local_base_model or str(persistent_root / "models" / "7b_base" / "llama-7b")
    platform_out_root = Path(args.train_out) if args.train_out else Path("/workspace/model_out")
    model_out_root = platform_out_root / "search-r1"

    env = os.environ.copy()
    env.setdefault("WORK_DIR", str(root))
    env.setdefault("SEARCH_R1_ROOT", str(persistent_root))
    if platform_out_root.exists() or str(platform_out_root).startswith("/workspace/model_out"):
        model_out_root.mkdir(parents=True, exist_ok=True)
        env.setdefault("OUTPUT_ROOT", str(model_out_root / "outputs"))
        env.setdefault("LOG_ROOT", str(Path(args.train_log) / "search-r1" if args.train_log else model_out_root / "logs"))
    env.setdefault("BASE_MODEL_NAME", args.base_model_name)
    env.setdefault("LOCAL_BASE_MODEL", local_base_model)
    env.setdefault("ALGO", args.algo)
    env.setdefault("RUN_MODE", args.run_mode)
    env.setdefault("TWO_GPU_CUDA_VISIBLE_DEVICES", args.cuda_visible_devices)
    env.setdefault("RETRIEVER_TOPK", args.retriever_topk)
    env.setdefault("RETRIEVER_URL", "http://127.0.0.1:8000/retrieve")
    env.setdefault("PYTHONUNBUFFERED", "1")

    print(f"[entry] project_dir={root}", flush=True)
    print(f"[entry] persistent_root={env['SEARCH_R1_ROOT']}", flush=True)
    print(f"[entry] output_root={env.get('OUTPUT_ROOT', '')}", flush=True)
    print(f"[entry] train_out={args.train_out or ''}", flush=True)
    print(f"[entry] train_log={args.train_log or ''}", flush=True)
    print(f"[entry] data_url={args.data_url or ''}", flush=True)
    print(f"[entry] algo={env['ALGO']} run_mode={env['RUN_MODE']}", flush=True)
    print(f"[entry] cuda_visible_devices={env['TWO_GPU_CUDA_VISIBLE_DEVICES']}", flush=True)
    print(f"[entry] retriever_url={env['RETRIEVER_URL']}", flush=True)

    retriever_proc: subprocess.Popen[str] | None = None
    try:
        if not args.skip_retriever:
            retriever_cmd = ["bash", str(script_dir / "launch_retriever.sh")]
            print(f"[entry] starting retriever: {' '.join(retriever_cmd)}", flush=True)
            retriever_proc = subprocess.Popen(retriever_cmd, cwd=root, env=env)
            wait_for_retriever(env["RETRIEVER_URL"], retriever_proc, args.retriever_timeout)
            print("[entry] retriever is ready", flush=True)

        train_cmd = ["bash", str(script_dir / "run_profiled_train.sh")]
        print(f"[entry] starting training: {' '.join(train_cmd)}", flush=True)
        return stream_process(train_cmd, env=env, cwd=root)
    finally:
        if retriever_proc is not None:
            print("[entry] stopping retriever", flush=True)
            terminate_process(retriever_proc)


if __name__ == "__main__":
    sys.exit(main())
