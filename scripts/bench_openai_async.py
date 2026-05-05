#!/usr/bin/env python3
"""Async benchmark runner for OpenAI-compatible chat endpoints.

The runner intentionally uses only the Python standard library so it can run
from the WSL vLLM venv, Windows Python, or a second LAN client without extra
packages. It drives streaming chat completions and measures TTFT, ITL, E2E
latency, aggregate output TPS, and basic GPU memory metadata.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
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
    "vram_total",
    "route",
    "workload",
    "target_prompt_tokens",
    "target_output_tokens",
    "prompt_tokens",
    "output_tokens",
    "total_prompt_tokens",
    "total_output_tokens",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "gpu_memory_utilization",
    "block_size",
    "prefix_caching_enabled",
    "chunked_prefill_enabled",
    "async_scheduling_enabled",
    "enforce_eager",
    "streaming",
    "concurrency",
    "request_rate",
    "request_count",
    "success_count",
    "ttft_ms_p50",
    "ttft_ms_p95",
    "itl_ms_p50",
    "itl_ms_p95",
    "e2e_latency_ms_p50",
    "e2e_latency_ms_p95",
    "output_tps",
    "total_tps",
    "gpu_mem_used_mb",
    "error_count",
    "notes",
]


@dataclass(frozen=True)
class Workload:
    name: str
    target_prompt_tokens: int
    target_output_tokens: int
    shared_prefix: bool = False
    purpose: str = ""


WORKLOADS = {
    "short_chat": Workload(
        name="short_chat",
        target_prompt_tokens=128,
        target_output_tokens=128,
        purpose="basic chat latency and throughput",
    ),
    "long_prefill": Workload(
        name="long_prefill",
        target_prompt_tokens=1024,
        target_output_tokens=32,
        purpose="prefill and TTFT sensitivity",
    ),
    "long_decode": Workload(
        name="long_decode",
        target_prompt_tokens=128,
        target_output_tokens=512,
        purpose="decode and ITL sensitivity",
    ),
    "shared_prefix": Workload(
        name="shared_prefix",
        target_prompt_tokens=1024,
        target_output_tokens=32,
        shared_prefix=True,
        purpose="automatic prefix caching behavior",
    ),
}


@dataclass
class RequestResult:
    request_id: int
    ok: bool
    ttft_ms: float | None
    itl_ms: list[float]
    e2e_latency_ms: float
    prompt_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    chunk_count: int
    error: str | None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_int_csv(raw: str) -> list[int]:
    return [int(item) for item in parse_csv_list(raw)]


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def percentile(values: list[float], pct: float) -> float | None:
    clean = sorted(v for v in values if v is not None and math.isfinite(v))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return clean[int(rank)]
    weight = rank - lower
    return clean[lower] * (1 - weight) + clean[upper] * weight


def round_or_blank(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def mean_int(values: list[int | None]) -> int | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return round(statistics.fmean(clean))


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
    if not first:
        return "", None, None

    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 3:
        return first, None, None

    name = parts[0]
    try:
        total = int(parts[1])
    except ValueError:
        total = None
    try:
        used = int(parts[2])
    except ValueError:
        used = None
    return name, total, used


def detect_vllm_version() -> str:
    try:
        import vllm  # type: ignore

        return getattr(vllm, "__version__", "")
    except Exception:
        return ""


def filler_words(word_count: int) -> str:
    # Common plain words are close to one token each on Qwen tokenizers. When a
    # local tokenizer is available, build_messages calibrates this count.
    return ("benchmark " * max(0, word_count)).strip()


def count_chat_tokens(
    tokenizer: Any | None,
    messages: list[dict[str, str]],
    disable_thinking: bool,
) -> int | None:
    if tokenizer is None:
        return None
    try:
        return len(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=not disable_thinking,
            )
        )
    except TypeError:
        try:
            return len(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
        except Exception:
            return None
    except Exception:
        return None


def calibrate_filler_count(
    *,
    tokenizer: Any | None,
    target_tokens: int,
    disable_thinking: bool,
    make_messages: Any,
) -> int:
    if tokenizer is None:
        return target_tokens

    low = 0
    high = max(8, target_tokens * 2)
    high_count = count_chat_tokens(tokenizer, make_messages(high), disable_thinking)
    while high_count is not None and high_count < target_tokens and high < target_tokens * 8:
        high *= 2
        high_count = count_chat_tokens(tokenizer, make_messages(high), disable_thinking)

    best_count = 0
    best_distance = float("inf")
    while low <= high:
        mid = (low + high) // 2
        token_count = count_chat_tokens(tokenizer, make_messages(mid), disable_thinking)
        if token_count is None:
            return target_tokens
        distance = abs(token_count - target_tokens)
        if distance < best_distance:
            best_distance = distance
            best_count = mid
        if token_count < target_tokens:
            low = mid + 1
        elif token_count > target_tokens:
            high = mid - 1
        else:
            return mid
    return best_count


def build_messages(
    workload: Workload,
    request_id: int,
    tokenizer: Any | None,
    disable_thinking: bool,
) -> list[dict[str, str]]:
    if workload.shared_prefix:
        def make_shared_messages(filler_count: int) -> list[dict[str, str]]:
            shared = filler_words(filler_count)
            return [
                {
                    "role": "system",
                    "content": (
                        "Shared benchmark prefix. Keep this text identical across "
                        f"requests for prefix-cache measurement. {shared}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Request {request_id}: repeat the word benchmark separated "
                        "by spaces until close to the response limit. Do not explain."
                    ),
                },
            ]

        filler_count = calibrate_filler_count(
            tokenizer=tokenizer,
            target_tokens=workload.target_prompt_tokens,
            disable_thinking=disable_thinking,
            make_messages=make_shared_messages,
        )
        return make_shared_messages(filler_count)

    def make_regular_messages(filler_count: int) -> list[dict[str, str]]:
        prompt_body = filler_words(filler_count)
        return [
            {
                "role": "system",
                "content": "You are a benchmark target. Follow the requested output shape.",
            },
            {
                "role": "user",
                "content": (
                    f"{prompt_body}\n\n"
                    f"Request {request_id}: produce benchmark text until close to "
                    f"{workload.target_output_tokens} output tokens. Use plain words."
                ),
            },
        ]

    filler_count = calibrate_filler_count(
        tokenizer=tokenizer,
        target_tokens=workload.target_prompt_tokens,
        disable_thinking=disable_thinking,
        make_messages=make_regular_messages,
    )
    return make_regular_messages(filler_count)


def load_tokenizer(path: str, calibrate_prompts: bool) -> Any | None:
    if not calibrate_prompts:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(path)
    except Exception as exc:
        print(f"Warning: failed to load tokenizer for prompt calibration: {exc}", file=sys.stderr)
        return None


def request_stream(
    *,
    url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: int,
    disable_thinking: bool,
    temperature: float,
    request_id: int,
) -> RequestResult:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    start = time.perf_counter()
    first_token_at: float | None = None
    last_token_at: float | None = None
    itl_ms: list[float] = []
    chunk_count = 0
    usage: dict[str, Any] | None = None

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            while True:
                raw_line = response.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if not content:
                    continue
                now = time.perf_counter()
                chunk_count += 1
                if first_token_at is None:
                    first_token_at = now
                elif last_token_at is not None:
                    itl_ms.append((now - last_token_at) * 1000)
                last_token_at = now
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return RequestResult(
            request_id=request_id,
            ok=False,
            ttft_ms=None,
            itl_ms=[],
            e2e_latency_ms=(time.perf_counter() - start) * 1000,
            prompt_tokens=None,
            output_tokens=None,
            total_tokens=None,
            chunk_count=chunk_count,
            error=f"HTTP {exc.code}: {detail[:500]}",
        )
    except Exception as exc:
        return RequestResult(
            request_id=request_id,
            ok=False,
            ttft_ms=None,
            itl_ms=[],
            e2e_latency_ms=(time.perf_counter() - start) * 1000,
            prompt_tokens=None,
            output_tokens=None,
            total_tokens=None,
            chunk_count=chunk_count,
            error=repr(exc),
        )

    end = time.perf_counter()
    completion_tokens = None
    prompt_tokens = None
    total_tokens = None
    if usage:
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")

    if completion_tokens is None:
        completion_tokens = chunk_count
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return RequestResult(
        request_id=request_id,
        ok=True,
        ttft_ms=(first_token_at - start) * 1000 if first_token_at is not None else None,
        itl_ms=itl_ms,
        e2e_latency_ms=(end - start) * 1000,
        prompt_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=total_tokens,
        chunk_count=chunk_count,
        error=None,
    )


def request_non_stream(
    *,
    url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: int,
    disable_thinking: bool,
    temperature: float,
    request_id: int,
) -> RequestResult:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return RequestResult(
            request_id=request_id,
            ok=False,
            ttft_ms=None,
            itl_ms=[],
            e2e_latency_ms=(time.perf_counter() - start) * 1000,
            prompt_tokens=None,
            output_tokens=None,
            total_tokens=None,
            chunk_count=0,
            error=f"HTTP {exc.code}: {detail[:500]}",
        )
    except Exception as exc:
        return RequestResult(
            request_id=request_id,
            ok=False,
            ttft_ms=None,
            itl_ms=[],
            e2e_latency_ms=(time.perf_counter() - start) * 1000,
            prompt_tokens=None,
            output_tokens=None,
            total_tokens=None,
            chunk_count=0,
            error=repr(exc),
        )

    usage = result.get("usage") or {}
    e2e_ms = (time.perf_counter() - start) * 1000
    return RequestResult(
        request_id=request_id,
        ok=True,
        ttft_ms=None,
        itl_ms=[],
        e2e_latency_ms=e2e_ms,
        prompt_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        chunk_count=0,
        error=None,
    )


async def run_one_request(
    *,
    args: argparse.Namespace,
    workload: Workload,
    request_id: int,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    async with semaphore:
        messages = build_messages(
            workload,
            request_id,
            args.local_tokenizer,
            args.disable_thinking,
        )
        fn = request_stream if args.streaming else request_non_stream
        return await asyncio.to_thread(
            fn,
            url=f"http://{args.host}:{args.port}/v1/chat/completions",
            api_key=args.api_key,
            model=args.model,
            messages=messages,
            max_tokens=workload.target_output_tokens,
            timeout=args.timeout,
            disable_thinking=args.disable_thinking,
            temperature=args.temperature,
            request_id=request_id,
        )


async def run_combo(
    *,
    args: argparse.Namespace,
    workload: Workload,
    concurrency: int,
) -> tuple[float, list[RequestResult]]:
    total_requests = concurrency * args.waves
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []
    start = time.perf_counter()

    for request_id in range(1, total_requests + 1):
        tasks.append(
            asyncio.create_task(
                run_one_request(
                    args=args,
                    workload=workload,
                    request_id=request_id,
                    semaphore=semaphore,
                )
            )
        )
        if args.request_rate > 0 and request_id < total_requests:
            await asyncio.sleep(1 / args.request_rate)

    results = await asyncio.gather(*tasks)
    wall_seconds = time.perf_counter() - start
    return wall_seconds, results


def summarize_combo(
    *,
    args: argparse.Namespace,
    run_id: str,
    timestamp: str,
    workload: Workload,
    concurrency: int,
    wall_seconds: float,
    results: list[RequestResult],
    gpu_name: str,
    vram_total: int | None,
    gpu_mem_used: int | None,
    vllm_version: str,
) -> dict[str, Any]:
    successes = [result for result in results if result.ok]
    errors = [result for result in results if not result.ok]
    ttfts = [result.ttft_ms for result in successes if result.ttft_ms is not None]
    e2es = [result.e2e_latency_ms for result in successes]
    itls = [value for result in successes for value in result.itl_ms]

    total_prompt_tokens = sum(result.prompt_tokens or 0 for result in successes)
    total_output_tokens = sum(result.output_tokens or 0 for result in successes)
    total_tokens = total_prompt_tokens + total_output_tokens
    output_tps = total_output_tokens / wall_seconds if wall_seconds > 0 else 0
    total_tps = total_tokens / wall_seconds if wall_seconds > 0 else 0

    notes = args.notes
    if errors:
        kinds = sorted({(error.error or "unknown").splitlines()[0][:120] for error in errors})
        notes = f"{notes}; errors={kinds}" if notes else f"errors={kinds}"

    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "backend": args.backend,
        "profile": args.profile,
        "model": args.model,
        "quantization": args.quantization,
        "vllm_version": vllm_version,
        "gpu_name": gpu_name,
        "vram_total": vram_total or "",
        "route": args.route,
        "workload": workload.name,
        "target_prompt_tokens": workload.target_prompt_tokens,
        "target_output_tokens": workload.target_output_tokens,
        "prompt_tokens": mean_int([result.prompt_tokens for result in successes]) or "",
        "output_tokens": mean_int([result.output_tokens for result in successes]) or "",
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "block_size": args.block_size,
        "prefix_caching_enabled": bool_text(args.prefix_caching_enabled),
        "chunked_prefill_enabled": bool_text(args.chunked_prefill_enabled),
        "async_scheduling_enabled": bool_text(args.async_scheduling_enabled),
        "enforce_eager": bool_text(args.enforce_eager),
        "streaming": bool_text(args.streaming),
        "concurrency": concurrency,
        "request_rate": args.request_rate,
        "request_count": len(results),
        "success_count": len(successes),
        "ttft_ms_p50": round_or_blank(percentile(ttfts, 0.50)),
        "ttft_ms_p95": round_or_blank(percentile(ttfts, 0.95)),
        "itl_ms_p50": round_or_blank(percentile(itls, 0.50)),
        "itl_ms_p95": round_or_blank(percentile(itls, 0.95)),
        "e2e_latency_ms_p50": round_or_blank(percentile(e2es, 0.50)),
        "e2e_latency_ms_p95": round_or_blank(percentile(e2es, 0.95)),
        "output_tps": round_or_blank(output_tps),
        "total_tps": round_or_blank(total_tps),
        "gpu_mem_used_mb": gpu_mem_used or "",
        "error_count": len(errors),
        "notes": notes,
        "requests": [
            {
                "request_id": result.request_id,
                "ok": result.ok,
                "ttft_ms": result.ttft_ms,
                "itl_ms": result.itl_ms,
                "e2e_latency_ms": result.e2e_latency_ms,
                "prompt_tokens": result.prompt_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.total_tokens,
                "chunk_count": result.chunk_count,
                "error": result.error,
            }
            for result in results
        ],
        "wall_seconds": wall_seconds,
        "purpose": workload.purpose,
    }


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


def default_output_prefix() -> Path:
    date = datetime.now().strftime("%Y-%m-%d")
    return Path("reports") / "benchmarks" / f"{date}-vllm-gguf-matrix"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark an OpenAI-compatible chat endpoint with streaming TTFT/ITL metrics."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "change-this-before-lan-use"))
    parser.add_argument("--model", default="Qwen3-8B-GGUF-vLLM-local")
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--profile", default="qwen3_8b_gguf_vllm_optimized")
    parser.add_argument("--quantization", default="gguf-q4_k_m")
    parser.add_argument("--route", default="WSL")
    parser.add_argument("--workloads", default="short_chat,long_prefill,long_decode,shared_prefix")
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--waves", type=int, default=1, help="Total requests per combo = concurrency * waves.")
    parser.add_argument("--request-rate", type=float, default=0.0, help="Requests per second. 0 means burst.")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-prefix", type=Path, default=default_output_prefix())
    parser.add_argument("--notes", default="")
    parser.add_argument("--tokenizer-path", default="/mnt/e/GPTProject2/models/Qwen3-8B")
    parser.add_argument("--calibrate-prompts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-model-len", default=os.environ.get("VLLM_MAX_MODEL_LEN", "2048"))
    parser.add_argument("--max-num-seqs", default=os.environ.get("VLLM_MAX_NUM_SEQS", "8"))
    parser.add_argument(
        "--max-num-batched-tokens",
        default=os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", "4096"),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        default=os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.80"),
    )
    parser.add_argument("--block-size", default=os.environ.get("VLLM_BLOCK_SIZE", "16"))
    parser.add_argument("--prefix-caching-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunked-prefill-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--async-scheduling-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    return parser


async def async_main(args: argparse.Namespace) -> int:
    selected_workloads = parse_csv_list(args.workloads)
    unknown = [name for name in selected_workloads if name not in WORKLOADS]
    if unknown:
        print(f"Unknown workloads: {', '.join(unknown)}", file=sys.stderr)
        return 2

    concurrency_values = parse_int_csv(args.concurrency)
    if not concurrency_values:
        print("At least one concurrency value is required.", file=sys.stderr)
        return 2
    if args.waves < 1:
        print("--waves must be >= 1.", file=sys.stderr)
        return 2

    run_id = str(uuid.uuid4())
    timestamp = now_iso()
    vllm_version = detect_vllm_version()
    gpu_name, vram_total, _ = query_gpu()
    args.local_tokenizer = load_tokenizer(args.tokenizer_path, args.calibrate_prompts)
    rows: list[dict[str, Any]] = []

    max_workers = max(concurrency_values) + 4
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=max_workers)
    loop.set_default_executor(executor)

    try:
        for workload_name in selected_workloads:
            workload = WORKLOADS[workload_name]
            for concurrency in concurrency_values:
                print(
                    f"Running workload={workload.name} concurrency={concurrency} "
                    f"requests={concurrency * args.waves}",
                    flush=True,
                )
                wall_seconds, results = await run_combo(
                    args=args,
                    workload=workload,
                    concurrency=concurrency,
                )
                _, _, gpu_mem_used = query_gpu()
                row = summarize_combo(
                    args=args,
                    run_id=run_id,
                    timestamp=timestamp,
                    workload=workload,
                    concurrency=concurrency,
                    wall_seconds=wall_seconds,
                    results=results,
                    gpu_name=gpu_name,
                    vram_total=vram_total,
                    gpu_mem_used=gpu_mem_used,
                    vllm_version=vllm_version,
                )
                rows.append(row)
                print(
                    "  "
                    f"ok={row['success_count']}/{row['request_count']} "
                    f"ttft_p50={row['ttft_ms_p50']}ms "
                    f"itl_p50={row['itl_ms_p50']}ms "
                    f"e2e_p50={row['e2e_latency_ms_p50']}ms "
                    f"output_tps={row['output_tps']}",
                    flush=True,
                )
    finally:
        executor.shutdown(wait=True)

    csv_path = args.output_prefix.with_suffix(".csv")
    jsonl_path = args.output_prefix.with_suffix(".jsonl")
    append_csv(csv_path, rows)
    append_jsonl(jsonl_path, rows)

    print(f"Wrote {len(rows)} benchmark rows:")
    print(f"  {csv_path}")
    print(f"  {jsonl_path}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    system = platform.system()
    if args.route == "WSL" and system == "Windows":
        print("Warning: route is WSL but this script is running under Windows.", file=sys.stderr)
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
