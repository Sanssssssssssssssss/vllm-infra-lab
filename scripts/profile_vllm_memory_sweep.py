#!/usr/bin/env python3
"""Profile vLLM startup memory and KV-cache capacity across launch configs."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "run_id",
    "timestamp",
    "backend",
    "profile",
    "model",
    "quantization",
    "vllm_version",
    "gpu_name",
    "vram_total_mb",
    "route",
    "max_model_len",
    "gpu_memory_utilization",
    "max_num_seqs",
    "max_num_batched_tokens",
    "enforce_eager",
    "block_size",
    "prefix_caching_enabled",
    "chunked_prefill_enabled",
    "async_scheduling_enabled",
    "status",
    "oom",
    "model_memory_gib",
    "kv_cache_memory_gib",
    "gpu_blocks",
    "gpu_blocks_source",
    "gpu_kv_tokens",
    "max_concurrency_tokens_per_request",
    "max_concurrency",
    "cuda_graph_capture",
    "startup_seconds",
    "gpu_mem_used_mb_after_start",
    "exit_code",
    "log_path",
    "error_excerpt",
    "notes",
]


MODEL_MEMORY_RE = re.compile(r"Model loading took ([0-9.]+) GiB memory")
KV_MEMORY_RE = re.compile(r"Available KV cache memory: ([0-9.]+) GiB")
GPU_KV_TOKENS_RE = re.compile(r"GPU KV cache size: ([0-9,]+) tokens")
MAX_CONCURRENCY_RE = re.compile(
    r"Maximum concurrency for ([0-9,]+) tokens per request: ([0-9.]+)x"
)
FAILURE_PATTERNS = [
    "Engine core initialization failed",
    "RuntimeError:",
    "ValueError:",
    "Traceback",
    "out of memory",
    "OutOfMemoryError",
    "CUDA error",
]
OOM_PATTERNS = [
    "out of memory",
    "OutOfMemoryError",
    "less than desired GPU memory utilization",
    "CUDA error: out of memory",
]


@dataclass(frozen=True)
class LaunchConfig:
    max_model_len: int
    gpu_memory_utilization: float
    max_num_seqs: int
    max_num_batched_tokens: int
    enforce_eager: bool


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_int_csv(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_float_csv(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_bool_csv(raw: str) -> list[bool]:
    values = []
    for item in raw.split(","):
        text = item.strip().lower()
        if not text:
            continue
        if text in {"1", "true", "yes", "on"}:
            values.append(True)
        elif text in {"0", "false", "no", "off"}:
            values.append(False)
        else:
            raise ValueError(f"Invalid boolean value: {item}")
    return values


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def default_output_prefix() -> Path:
    date = datetime.now().strftime("%Y-%m-%d")
    return Path("reports") / "memory" / f"{date}-vllm-gguf-memory-profile"


def query_gpu() -> tuple[str, int | None, int | None]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except Exception:
        return "", None, None

    first = output.strip().splitlines()[0] if output.strip() else ""
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 3:
        return first, None, None

    try:
        total = int(parts[1])
    except ValueError:
        total = None
    try:
        used = int(parts[2])
    except ValueError:
        used = None
    return parts[0], total, used


def detect_vllm_version() -> str:
    try:
        import vllm  # type: ignore

        return getattr(vllm, "__version__", "")
    except Exception:
        return ""


def healthcheck(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def kill_existing_vllm() -> None:
    subprocess.run(
        ["pkill", "-f", "vllm serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    time.sleep(3)


def wait_for_gpu_memory_below(threshold_mb: int, timeout_seconds: int) -> tuple[bool, int | None]:
    if threshold_mb <= 0:
        _, _, used = query_gpu()
        return True, used

    deadline = time.perf_counter() + timeout_seconds
    last_used: int | None = None
    while time.perf_counter() < deadline:
        _, _, used = query_gpu()
        last_used = used
        if used is None or used <= threshold_mb:
            return True, used
        time.sleep(2)
    return False, last_used


def terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            process.kill()
        process.wait(timeout=10)


def build_command(args: argparse.Namespace, cfg: LaunchConfig) -> list[str]:
    cmd = [
        str(Path(args.venv_dir) / "bin" / "vllm"),
        "serve",
        args.model_path,
        "--tokenizer",
        args.tokenizer_path,
        "--hf-config-path",
        args.hf_config_path,
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--api-key",
        args.api_key,
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(cfg.max_model_len),
        "--gpu-memory-utilization",
        f"{cfg.gpu_memory_utilization:.2f}",
        "--max-num-seqs",
        str(cfg.max_num_seqs),
        "--max-num-batched-tokens",
        str(cfg.max_num_batched_tokens),
        "--block-size",
        str(args.block_size),
        "--generation-config",
        args.generation_config,
    ]

    if args.prefix_caching_enabled:
        cmd.append("--enable-prefix-caching")
    if args.chunked_prefill_enabled:
        cmd.append("--enable-chunked-prefill")
    if args.async_scheduling_enabled:
        cmd.append("--async-scheduling")
    if cfg.enforce_eager:
        cmd.append("--enforce-eager")
    return cmd


def read_log(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def extract_float(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def extract_int(pattern: re.Pattern[str], text: str) -> int | None:
    match = pattern.search(text)
    if not match:
        return None
    try:
        return int(match.group(1).replace(",", ""))
    except ValueError:
        return None


def parse_max_concurrency(text: str) -> tuple[int | None, float | None]:
    match = MAX_CONCURRENCY_RE.search(text)
    if not match:
        return None, None
    try:
        tokens = int(match.group(1).replace(",", ""))
    except ValueError:
        tokens = None
    try:
        concurrency = float(match.group(2))
    except ValueError:
        concurrency = None
    return tokens, concurrency


def error_excerpt(text: str) -> str:
    lines = text.splitlines()
    selected: list[str] = []
    for index, line in enumerate(lines):
        lower = line.lower()
        if any(pattern.lower() in lower for pattern in FAILURE_PATTERNS + OOM_PATTERNS):
            selected.extend(lines[max(0, index - 2) : min(len(lines), index + 5)])
            break
    if not selected and lines:
        selected = lines[-12:]
    return "\n".join(selected)[-1200:]


def status_from_log(text: str, process: subprocess.Popen[Any]) -> str:
    if "Application startup complete" in text:
        return "ready"
    if any(pattern in text for pattern in FAILURE_PATTERNS):
        return "failed"
    if process.poll() is not None:
        return "exited"
    return "timeout"


def is_oom(text: str) -> bool:
    lower = text.lower()
    return any(pattern.lower() in lower for pattern in OOM_PATTERNS)


def profile_one(
    *,
    args: argparse.Namespace,
    cfg: LaunchConfig,
    run_id: str,
    timestamp: str,
    index: int,
    gpu_name: str,
    vram_total_mb: int | None,
    vllm_version: str,
) -> dict[str, Any]:
    if args.kill_existing:
        kill_existing_vllm()
    elif healthcheck(args.health_host, args.port):
        raise RuntimeError(
            f"Port {args.port} already has a healthy server. Re-run with --kill-existing."
        )
    cooled, gpu_used_before = wait_for_gpu_memory_below(
        args.gpu_cooldown_threshold_mb,
        args.gpu_cooldown_timeout,
    )
    if not cooled:
        print(
            f"Warning: GPU memory is still {gpu_used_before} MB before launch "
            f"(threshold {args.gpu_cooldown_threshold_mb} MB).",
            file=sys.stderr,
            flush=True,
        )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (
        f"{run_id}-{index:03d}-len{cfg.max_model_len}-gpu{cfg.gpu_memory_utilization:.2f}-"
        f"seq{cfg.max_num_seqs}-batch{cfg.max_num_batched_tokens}-"
        f"eager{str(cfg.enforce_eager).lower()}.log"
    )

    env = os.environ.copy()
    env["HF_HOME"] = args.hf_home
    env["HF_HUB_DISABLE_XET"] = "1"
    cmd = build_command(args, cfg)

    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("COMMAND: " + " ".join(cmd) + "\n\n")
        log_handle.flush()
        process = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            text=True,
        )

    status = "timeout"
    while time.perf_counter() - started < args.startup_timeout:
        text = read_log(log_path)
        if "Application startup complete" in text:
            status = "ready"
            break
        if process.poll() is not None:
            status = status_from_log(text, process)
            break
        if any(pattern in text for pattern in FAILURE_PATTERNS):
            status = "failed"
            break
        time.sleep(args.poll_interval)
    else:
        status = status_from_log(read_log(log_path), process)

    startup_seconds = time.perf_counter() - started
    text = read_log(log_path)
    _, _, gpu_mem_used = query_gpu()

    model_memory_gib = extract_float(MODEL_MEMORY_RE, text)
    kv_cache_memory_gib = extract_float(KV_MEMORY_RE, text)
    gpu_kv_tokens = extract_int(GPU_KV_TOKENS_RE, text)
    max_concurrency_tokens, max_concurrency = parse_max_concurrency(text)
    cuda_graph_capture = "Graph capturing finished" in text
    gpu_blocks = ""
    gpu_blocks_source = ""
    if gpu_kv_tokens is not None and args.block_size > 0:
        gpu_blocks = math.floor(gpu_kv_tokens / args.block_size)
        gpu_blocks_source = "derived_from_gpu_kv_tokens_div_block_size"

    row = {
        "run_id": run_id,
        "timestamp": timestamp,
        "backend": "vllm",
        "profile": args.profile,
        "model": args.served_model_name,
        "quantization": args.quantization,
        "vllm_version": vllm_version,
        "gpu_name": gpu_name,
        "vram_total_mb": vram_total_mb or "",
        "route": args.route,
        "max_model_len": cfg.max_model_len,
        "gpu_memory_utilization": f"{cfg.gpu_memory_utilization:.2f}",
        "max_num_seqs": cfg.max_num_seqs,
        "max_num_batched_tokens": cfg.max_num_batched_tokens,
        "enforce_eager": bool_text(cfg.enforce_eager),
        "block_size": args.block_size,
        "prefix_caching_enabled": bool_text(args.prefix_caching_enabled),
        "chunked_prefill_enabled": bool_text(args.chunked_prefill_enabled),
        "async_scheduling_enabled": bool_text(args.async_scheduling_enabled),
        "status": status,
        "oom": bool_text(is_oom(text)),
        "model_memory_gib": model_memory_gib if model_memory_gib is not None else "",
        "kv_cache_memory_gib": kv_cache_memory_gib if kv_cache_memory_gib is not None else "",
        "gpu_blocks": gpu_blocks,
        "gpu_blocks_source": gpu_blocks_source,
        "gpu_kv_tokens": gpu_kv_tokens if gpu_kv_tokens is not None else "",
        "max_concurrency_tokens_per_request": (
            max_concurrency_tokens if max_concurrency_tokens is not None else ""
        ),
        "max_concurrency": max_concurrency if max_concurrency is not None else "",
        "cuda_graph_capture": bool_text(cuda_graph_capture),
        "startup_seconds": f"{startup_seconds:.2f}",
        "gpu_mem_used_mb_after_start": gpu_mem_used or "",
        "exit_code": process.poll() if process.poll() is not None else "",
        "log_path": str(log_path),
        "error_excerpt": "" if status == "ready" else error_excerpt(text),
        "notes": args.notes
        if cooled
        else f"{args.notes}; gpu_prelaunch_used_mb={gpu_used_before}; cooldown_not_reached",
        "command": cmd,
    }

    terminate_process(process)
    time.sleep(args.cooldown_seconds)
    return row


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def pilot_configs() -> list[LaunchConfig]:
    return [
        LaunchConfig(1024, 0.72, 1, 1024, False),
        LaunchConfig(2048, 0.80, 8, 4096, False),
        LaunchConfig(2048, 0.84, 8, 4096, False),
        LaunchConfig(2048, 0.80, 8, 4096, True),
    ]


def stage2_configs(args: argparse.Namespace) -> list[LaunchConfig]:
    return [
        LaunchConfig(max_len, gpu_util, max_seqs, batched_tokens, eager)
        for max_len, gpu_util, max_seqs, batched_tokens, eager in itertools.product(
            parse_int_csv(args.max_model_lens),
            parse_float_csv(args.gpu_memory_utilizations),
            parse_int_csv(args.max_num_seqs_values),
            parse_int_csv(args.max_num_batched_tokens_values),
            parse_bool_csv(args.enforce_eager_values),
        )
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile vLLM startup memory and KV cache capacity.")
    parser.add_argument("--preset", choices=["pilot", "stage2"], default="pilot")
    parser.add_argument("--confirm-large-sweep", action="store_true")
    parser.add_argument("--max-model-lens", default="1024,2048,4096")
    parser.add_argument("--gpu-memory-utilizations", default="0.72,0.76,0.80,0.84")
    parser.add_argument("--max-num-seqs-values", default="1,2,4,8")
    parser.add_argument("--max-num-batched-tokens-values", default="1024,2048,4096")
    parser.add_argument("--enforce-eager-values", default="true,false")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-prefix", type=Path, default=default_output_prefix())
    parser.add_argument("--log-dir", default="logs/memory_profile")
    parser.add_argument("--startup-timeout", type=int, default=420)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--cooldown-seconds", type=float, default=4.0)
    parser.add_argument("--gpu-cooldown-threshold-mb", type=int, default=1800)
    parser.add_argument("--gpu-cooldown-timeout", type=int, default=90)
    parser.add_argument("--kill-existing", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--health-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "change-this-before-lan-use"))
    parser.add_argument("--model-path", default="/mnt/e/GPTProject2/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf")
    parser.add_argument("--tokenizer-path", default="/mnt/e/GPTProject2/models/Qwen3-8B")
    parser.add_argument("--hf-config-path", default="/mnt/e/GPTProject2/models/Qwen3-8B")
    parser.add_argument("--served-model-name", default="Qwen3-8B-GGUF-vLLM-local")
    parser.add_argument("--profile", default="qwen3_8b_gguf_vllm_optimized")
    parser.add_argument("--quantization", default="gguf-q4_k_m")
    parser.add_argument("--route", default="WSL")
    parser.add_argument("--venv-dir", default=str(Path.home() / ".venvs" / "gptproject2-vllm"))
    parser.add_argument("--hf-home", default="/mnt/e/GPTProject2/hf-cache")
    parser.add_argument("--dtype", default="half")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--generation-config", default="vllm")
    parser.add_argument("--prefix-caching-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunked-prefill-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--async-scheduling-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--notes", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    configs = pilot_configs() if args.preset == "pilot" else stage2_configs(args)
    if args.limit > 0:
        configs = configs[: args.limit]
    if args.preset == "stage2" and len(configs) > 20 and not args.confirm_large_sweep:
        print(
            f"Refusing to run {len(configs)} startup configs without --confirm-large-sweep.",
            file=sys.stderr,
        )
        return 2

    run_id = str(uuid.uuid4())
    timestamp = now_iso()
    vllm_version = detect_vllm_version()
    gpu_name, vram_total_mb, _ = query_gpu()
    rows: list[dict[str, Any]] = []

    print(f"Profiling {len(configs)} vLLM startup configs; run_id={run_id}")
    for index, cfg in enumerate(configs, start=1):
        print(
            f"[{index}/{len(configs)}] len={cfg.max_model_len} "
            f"gpu={cfg.gpu_memory_utilization:.2f} seqs={cfg.max_num_seqs} "
            f"batched={cfg.max_num_batched_tokens} eager={cfg.enforce_eager}",
            flush=True,
        )
        row = profile_one(
            args=args,
            cfg=cfg,
            run_id=run_id,
            timestamp=timestamp,
            index=index,
            gpu_name=gpu_name,
            vram_total_mb=vram_total_mb,
            vllm_version=vllm_version,
        )
        rows.append(row)
        print(
            "  "
            f"status={row['status']} oom={row['oom']} "
            f"model_gib={row['model_memory_gib']} kv_gib={row['kv_cache_memory_gib']} "
            f"kv_tokens={row['gpu_kv_tokens']} graph={row['cuda_graph_capture']}",
            flush=True,
        )

    csv_path = args.output_prefix.with_suffix(".csv")
    jsonl_path = args.output_prefix.with_suffix(".jsonl")
    append_csv(csv_path, rows)
    append_jsonl(jsonl_path, rows)
    print(f"Wrote {len(rows)} memory profile rows:")
    print(f"  {csv_path}")
    print(f"  {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
