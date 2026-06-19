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
    parser.add_argument("--rollout-name", default=os.environ.get("ROLLOUT_NAME"), choices=["vllm", "hf", None])
    parser.add_argument("--tensor-model-parallel-size", default=os.environ.get("TENSOR_MODEL_PARALLEL_SIZE"))
    parser.add_argument("--rollout-do-sample", default=os.environ.get("ROLLOUT_DO_SAMPLE"))
    parser.add_argument("--rollout-dtype", default=os.environ.get("ROLLOUT_DTYPE"))
    parser.add_argument("--rollout-enforce-eager", default=os.environ.get("ROLLOUT_ENFORCE_EAGER"))
    parser.add_argument("--rollout-disable-custom-all-reduce", default=os.environ.get("ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE"))
    parser.add_argument("--actor-model-dtype", default=os.environ.get("ACTOR_MODEL_DTYPE"))
    parser.add_argument("--model-attn-implementation", default=os.environ.get("MODEL_ATTN_IMPLEMENTATION"))
    parser.add_argument("--use-remove-padding", default=os.environ.get("USE_REMOVE_PADDING"))
    parser.add_argument("--hf-summon-full-params", default=os.environ.get("HF_SUMMON_FULL_PARAMS"))
    parser.add_argument("--hf-use-cache", default=os.environ.get("HF_USE_CACHE"))
    parser.add_argument("--train-data-num", default=os.environ.get("TRAIN_DATA_NUM"))
    parser.add_argument("--val-data-num", default=os.environ.get("VAL_DATA_NUM"))
    parser.add_argument("--total-training-steps", default=os.environ.get("TOTAL_TRAINING_STEPS"))
    parser.add_argument("--train-batch-size", default=os.environ.get("TRAIN_BATCH_SIZE"))
    parser.add_argument("--val-batch-size", default=os.environ.get("VAL_BATCH_SIZE"))
    parser.add_argument("--ppo-mini-batch-size", default=os.environ.get("PPO_MINI_BATCH_SIZE"))
    parser.add_argument("--ppo-micro-batch-size", default=os.environ.get("PPO_MICRO_BATCH_SIZE"))
    parser.add_argument("--log-prob-micro-batch-size", default=os.environ.get("LOG_PROB_MICRO_BATCH_SIZE"))
    parser.add_argument("--n-agent", default=os.environ.get("N_AGENT"))
    parser.add_argument("--max-turns", default=os.environ.get("MAX_TURNS"))
    parser.add_argument("--rollout-gpu-memory-utilization", default=os.environ.get("ROLLOUT_GPU_MEMORY_UTILIZATION"))
    parser.add_argument("--max-num-batched-tokens", default=os.environ.get("MAX_NUM_BATCHED_TOKENS"))
    parser.add_argument("--max-num-seqs", default=os.environ.get("MAX_NUM_SEQS"))
    parser.add_argument("--save-freq", default=os.environ.get("SAVE_FREQ"))
    parser.add_argument("--test-freq", default=os.environ.get("TEST_FREQ"))
    parser.add_argument("--val-before-train", default=os.environ.get("VAL_BEFORE_TRAIN"))
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
    if args.train_out:
        platform_out_root = Path(args.train_out)
    elif Path("/workspace/model_out").exists() or Path("/workspace").exists():
        platform_out_root = Path("/workspace/model_out")
    else:
        platform_out_root = Path("/workspace/model-out")
    model_out_root = platform_out_root / "search-r1"

    env = os.environ.copy()
    env.setdefault("WORK_DIR", str(root))
    env.setdefault("SEARCH_R1_ROOT", str(persistent_root))
    if (
        platform_out_root.exists()
        or str(platform_out_root).startswith("/workspace/model_out")
        or str(platform_out_root).startswith("/workspace/model-out")
    ):
        model_out_root.mkdir(parents=True, exist_ok=True)
        env.setdefault("OUTPUT_ROOT", str(model_out_root / "outputs"))
        env.setdefault("LOG_ROOT", str(Path(args.train_log) / "search-r1" if args.train_log else model_out_root / "logs"))
        ray_tmp_dir = model_out_root / "ray_tmp"
        tmp_dir = model_out_root / "tmp"
        ray_tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        env.setdefault("RAY_TMPDIR", str(ray_tmp_dir))
        env.setdefault("TMPDIR", str(tmp_dir))
        env.setdefault("TEMP", str(tmp_dir))
        env.setdefault("TMP", str(tmp_dir))
    # argparse defaults already read from the environment, so assigning here
    # preserves env-based configuration while allowing explicit CLI flags to
    # override platform-injected values.
    env["BASE_MODEL_NAME"] = args.base_model_name
    env["LOCAL_BASE_MODEL"] = local_base_model
    env["ALGO"] = args.algo
    env["RUN_MODE"] = args.run_mode
    env["TWO_GPU_CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["RETRIEVER_TOPK"] = args.retriever_topk
    if args.rollout_name:
        env["ROLLOUT_NAME"] = args.rollout_name
    if args.tensor_model_parallel_size:
        env["TENSOR_MODEL_PARALLEL_SIZE"] = args.tensor_model_parallel_size
    if args.rollout_do_sample:
        env["ROLLOUT_DO_SAMPLE"] = args.rollout_do_sample
    if args.rollout_dtype:
        env["ROLLOUT_DTYPE"] = args.rollout_dtype
    if args.rollout_enforce_eager:
        env["ROLLOUT_ENFORCE_EAGER"] = args.rollout_enforce_eager
    if args.rollout_disable_custom_all_reduce:
        env["ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE"] = args.rollout_disable_custom_all_reduce
    if args.actor_model_dtype:
        env["ACTOR_MODEL_DTYPE"] = args.actor_model_dtype
    if args.model_attn_implementation:
        env["MODEL_ATTN_IMPLEMENTATION"] = args.model_attn_implementation
    if args.use_remove_padding:
        env["USE_REMOVE_PADDING"] = args.use_remove_padding
    if args.hf_summon_full_params:
        env["HF_SUMMON_FULL_PARAMS"] = args.hf_summon_full_params
    if args.hf_use_cache:
        env["HF_USE_CACHE"] = args.hf_use_cache

    if env.get("ROLLOUT_NAME") == "hf":
        env.setdefault("ROLLOUT_DTYPE", "float16")
        env.setdefault("ACTOR_MODEL_DTYPE", "float16")
        env.setdefault("MODEL_ATTN_IMPLEMENTATION", "sdpa")
        env.setdefault("USE_REMOVE_PADDING", "false")
        env.setdefault("HF_USE_CACHE", "false")
    cli_env = {
        "TRAIN_DATA_NUM": args.train_data_num,
        "VAL_DATA_NUM": args.val_data_num,
        "TOTAL_TRAINING_STEPS": args.total_training_steps,
        "TRAIN_BATCH_SIZE": args.train_batch_size,
        "VAL_BATCH_SIZE": args.val_batch_size,
        "PPO_MINI_BATCH_SIZE": args.ppo_mini_batch_size,
        "PPO_MICRO_BATCH_SIZE": args.ppo_micro_batch_size,
        "LOG_PROB_MICRO_BATCH_SIZE": args.log_prob_micro_batch_size,
        "N_AGENT": args.n_agent,
        "MAX_TURNS": args.max_turns,
        "ROLLOUT_GPU_MEMORY_UTILIZATION": args.rollout_gpu_memory_utilization,
        "MAX_NUM_BATCHED_TOKENS": args.max_num_batched_tokens,
        "MAX_NUM_SEQS": args.max_num_seqs,
        "SAVE_FREQ": args.save_freq,
        "TEST_FREQ": args.test_freq,
        "VAL_BEFORE_TRAIN": args.val_before_train,
    }
    for key, value in cli_env.items():
        if value is not None:
            env[key] = value
    env.setdefault("RETRIEVER_URL", "http://127.0.0.1:8000/retrieve")
    env.setdefault("PYTHONUNBUFFERED", "1")

    print(f"[entry] project_dir={root}", flush=True)
    print(f"[entry] persistent_root={env['SEARCH_R1_ROOT']}", flush=True)
    print(f"[entry] output_root={env.get('OUTPUT_ROOT', '')}", flush=True)
    print(f"[entry] ray_tmpdir={env.get('RAY_TMPDIR', '')}", flush=True)
    print(f"[entry] tmpdir={env.get('TMPDIR', '')}", flush=True)
    print(f"[entry] train_out={args.train_out or ''}", flush=True)
    print(f"[entry] train_log={args.train_log or ''}", flush=True)
    print(f"[entry] data_url={args.data_url or ''}", flush=True)
    print(f"[entry] algo={env['ALGO']} run_mode={env['RUN_MODE']}", flush=True)
    print(f"[entry] rollout_name={env.get('ROLLOUT_NAME', '')}", flush=True)
    print(f"[entry] tensor_model_parallel_size={env.get('TENSOR_MODEL_PARALLEL_SIZE', '')}", flush=True)
    print(f"[entry] rollout_dtype={env.get('ROLLOUT_DTYPE', '')}", flush=True)
    print(f"[entry] rollout_enforce_eager={env.get('ROLLOUT_ENFORCE_EAGER', '')}", flush=True)
    print(f"[entry] rollout_disable_custom_all_reduce={env.get('ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE', '')}", flush=True)
    print(f"[entry] actor_model_dtype={env.get('ACTOR_MODEL_DTYPE', '')}", flush=True)
    print(f"[entry] model_attn_implementation={env.get('MODEL_ATTN_IMPLEMENTATION', '')}", flush=True)
    print(f"[entry] use_remove_padding={env.get('USE_REMOVE_PADDING', '')}", flush=True)
    print(f"[entry] hf_use_cache={env.get('HF_USE_CACHE', '')}", flush=True)
    print(f"[entry] train_data_num={env.get('TRAIN_DATA_NUM', '')}", flush=True)
    print(f"[entry] val_data_num={env.get('VAL_DATA_NUM', '')}", flush=True)
    print(f"[entry] total_training_steps={env.get('TOTAL_TRAINING_STEPS', '')}", flush=True)
    print(f"[entry] train_batch_size={env.get('TRAIN_BATCH_SIZE', '')}", flush=True)
    print(f"[entry] val_batch_size={env.get('VAL_BATCH_SIZE', '')}", flush=True)
    print(f"[entry] max_num_seqs={env.get('MAX_NUM_SEQS', '')}", flush=True)
    print(f"[entry] rollout_gpu_memory_utilization={env.get('ROLLOUT_GPU_MEMORY_UTILIZATION', '')}", flush=True)
    print(f"[entry] val_before_train={env.get('VAL_BEFORE_TRAIN', '')}", flush=True)
    print(f"[entry] test_freq={env.get('TEST_FREQ', '')}", flush=True)
    print(f"[entry] n_agent={env.get('N_AGENT', '')}", flush=True)
    print(f"[entry] max_turns={env.get('MAX_TURNS', '')}", flush=True)
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
