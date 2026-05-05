#!/usr/bin/env python3
"""Stage 3 benchmark for vLLM prefix caching and KV block reuse.

This runner is intentionally narrow: it sends three fixed PagedAttention
prefix-cache cases and records both client-side latency and vLLM /metrics
prefix-cache counter deltas.
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
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench_openai_async import (
    RequestResult,
    bool_text,
    calibrate_filler_count,
    count_chat_tokens,
    detect_vllm_version,
    load_tokenizer,
    percentile,
    query_gpu,
    request_non_stream,
    request_stream,
    round_or_blank,
)


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
    "case",
    "case_label",
    "prompt_style",
    "target_prompt_tokens",
    "shared_prefix_tokens",
    "expected_shared_blocks",
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
    "warmup_count",
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
    "prefix_cache_queries_delta",
    "prefix_cache_hits_delta",
    "prefix_cache_hit_rate",
    "kv_cache_usage_perc_before",
    "kv_cache_usage_perc_after",
    "prompt_tokens_total_delta",
    "generation_tokens_total_delta",
    "request_prefill_time_seconds_sum_delta",
    "request_decode_time_seconds_sum_delta",
    "error_count",
    "notes",
]


@dataclass(frozen=True)
class PrefixCase:
    name: str
    label: str
    prompt_style: str
    target_prompt_tokens: int
    shared_prefix_tokens: int
    target_output_tokens: int = 32
    purpose: str = ""

    @property
    def has_shared_prefix(self) -> bool:
        return self.shared_prefix_tokens > 0


CASES = {
    "case_a_distinct_1024": PrefixCase(
        name="case_a_distinct_1024",
        label="Case A: distinct prompts",
        prompt_style="distinct",
        target_prompt_tokens=1024,
        shared_prefix_tokens=0,
        purpose="control for the 1024-token shared system-prefix case",
    ),
    "case_a_distinct_2048": PrefixCase(
        name="case_a_distinct_2048",
        label="Case A control: distinct 2048-token prompts",
        prompt_style="distinct",
        target_prompt_tokens=2048,
        shared_prefix_tokens=0,
        purpose="optional control for the 2048-token shared document-prefix case",
    ),
    "case_b_shared_system_1024": PrefixCase(
        name="case_b_shared_system_1024",
        label="Case B: requests share a 1024-token system prefix",
        prompt_style="shared_system",
        target_prompt_tokens=1088,
        shared_prefix_tokens=1024,
        purpose="measure APC benefit on a long repeated system prefix",
    ),
    "case_c_shared_document_2048": PrefixCase(
        name="case_c_shared_document_2048",
        label="Case C: requests share a 2048-token document prefix",
        prompt_style="shared_document",
        target_prompt_tokens=2112,
        shared_prefix_tokens=2048,
        purpose="measure APC benefit on repeated long-document Q&A",
    ),
}

DEFAULT_CASES = "case_a_distinct_1024,case_b_shared_system_1024,case_c_shared_document_2048"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def mean_int(values: list[int | None]) -> int | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return round(statistics.fmean(clean))


def repeated_word(word: str, count: int) -> str:
    return (" ".join([word] * max(0, count))).strip()


def count_text_tokens(tokenizer: Any | None, text: str) -> int | None:
    if tokenizer is None:
        return None
    try:
        encoded = tokenizer(text, add_special_tokens=False)
        input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else encoded.input_ids
        return len(input_ids)
    except Exception:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return None


def calibrate_text_repetitions(
    *,
    tokenizer: Any | None,
    target_tokens: int,
    word: str,
    header: str,
) -> int:
    if tokenizer is None:
        return target_tokens

    low = 0
    high = max(8, target_tokens * 2)

    def make_text(count: int) -> str:
        return f"{header} {repeated_word(word, count)}".strip()

    while True:
        token_count = count_text_tokens(tokenizer, make_text(high))
        if token_count is None or token_count >= target_tokens or high >= target_tokens * 8:
            break
        high *= 2

    best_count = 0
    best_distance = float("inf")
    while low <= high:
        mid = (low + high) // 2
        token_count = count_text_tokens(tokenizer, make_text(mid))
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


def build_shared_prefix(
    *,
    case: PrefixCase,
    tokenizer: Any | None,
) -> str:
    if case.prompt_style == "shared_system":
        header = "PagedAttention shared system prefix. Keep this prefix byte-identical."
        word = "systemprefix"
    elif case.prompt_style == "shared_document":
        header = (
            "Document prefix for PagedAttention block reuse. The following synthetic "
            "document is identical across all measured requests."
        )
        word = "documentprefix"
    else:
        return ""

    repetitions = calibrate_text_repetitions(
        tokenizer=tokenizer,
        target_tokens=case.shared_prefix_tokens,
        word=word,
        header=header,
    )
    return f"{header} {repeated_word(word, repetitions)}".strip()


def build_messages(
    *,
    case: PrefixCase,
    request_id: int,
    tokenizer: Any | None,
    disable_thinking: bool,
) -> list[dict[str, str]]:
    if case.prompt_style == "distinct":
        unique_word = f"distinctrequest{request_id}"

        def make_distinct_messages(filler_count: int) -> list[dict[str, str]]:
            body = repeated_word(unique_word, filler_count)
            return [
                {
                    "role": "system",
                    "content": (
                        f"Distinct control system prompt for request {request_id}. "
                        f"{body}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{body}\n\n"
                        "Produce a short benchmark answer using plain words. "
                        f"Unique request id: {request_id}."
                    ),
                },
            ]

        filler_count = calibrate_filler_count(
            tokenizer=tokenizer,
            target_tokens=case.target_prompt_tokens,
            disable_thinking=disable_thinking,
            make_messages=make_distinct_messages,
        )
        return make_distinct_messages(filler_count)

    shared_prefix = build_shared_prefix(case=case, tokenizer=tokenizer)

    if case.prompt_style == "shared_system":
        return [
            {
                "role": "system",
                "content": shared_prefix,
            },
            {
                "role": "user",
                "content": (
                    f"Request {request_id}: answer with a concise benchmark response. "
                    "Use plain words and do not explain the benchmark."
                ),
            },
        ]

    if case.prompt_style == "shared_document":
        return [
            {
                "role": "system",
                "content": "Answer using the supplied document. Keep the answer brief.",
            },
            {
                "role": "user",
                "content": (
                    f"{shared_prefix}\n\n"
                    f"Question {request_id}: extract one concise observation from the document."
                ),
            },
        ]

    raise ValueError(f"Unknown prompt style: {case.prompt_style}")


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split("{", 1)[0]
        try:
            value = float(parts[1])
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        metrics[name] = metrics.get(name, 0.0) + value
    return metrics


def fetch_metrics(metrics_url: str, timeout: int) -> dict[str, float]:
    request = urllib.request.Request(metrics_url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return parse_prometheus_metrics(response.read().decode("utf-8", errors="replace"))


def safe_fetch_metrics(args: argparse.Namespace) -> tuple[dict[str, float], str | None]:
    if args.no_metrics:
        return {}, "metrics disabled"
    try:
        return fetch_metrics(args.metrics_url, args.metrics_timeout), None
    except Exception as exc:
        return {}, repr(exc)


def metric_value(metrics: dict[str, float], names: list[str]) -> float | None:
    values = [metrics[name] for name in names if name in metrics]
    if not values:
        return None
    return sum(values)


def metric_delta(
    before: dict[str, float],
    after: dict[str, float],
    names: list[str],
) -> float | None:
    after_value = metric_value(after, names)
    before_value = metric_value(before, names)
    if after_value is None or before_value is None:
        return None
    return after_value - before_value


PREFIX_CACHE_QUERIES = [
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_queries_total",
    "vllm_prefix_cache_queries",
    "vllm_prefix_cache_queries_total",
]
PREFIX_CACHE_HITS = [
    "vllm:prefix_cache_hits",
    "vllm:prefix_cache_hits_total",
    "vllm_prefix_cache_hits",
    "vllm_prefix_cache_hits_total",
]
KV_CACHE_USAGE = ["vllm:kv_cache_usage_perc", "vllm_kv_cache_usage_perc"]
PROMPT_TOKENS_TOTAL = ["vllm:prompt_tokens_total", "vllm_prompt_tokens_total"]
GENERATION_TOKENS_TOTAL = [
    "vllm:generation_tokens_total",
    "vllm_generation_tokens_total",
]
PREFILL_TIME_SUM = [
    "vllm:request_prefill_time_seconds_sum",
    "vllm_request_prefill_time_seconds_sum",
]
DECODE_TIME_SUM = [
    "vllm:request_decode_time_seconds_sum",
    "vllm_request_decode_time_seconds_sum",
]


def metric_text(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def default_output_prefix() -> Path:
    date = datetime.now().strftime("%Y-%m-%d")
    return Path("reports") / "benchmarks" / f"{date}-vllm-prefix-cache-blocks"


async def send_one_request(
    *,
    args: argparse.Namespace,
    case: PrefixCase,
    request_id: int,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    async with semaphore:
        messages = build_messages(
            case=case,
            request_id=request_id,
            tokenizer=args.local_tokenizer,
            disable_thinking=args.disable_thinking,
        )
        fn = request_stream if args.streaming else request_non_stream
        return await asyncio.to_thread(
            fn,
            url=f"http://{args.host}:{args.port}/v1/chat/completions",
            api_key=args.api_key,
            model=args.model,
            messages=messages,
            max_tokens=max_tokens,
            timeout=args.timeout,
            disable_thinking=args.disable_thinking,
            temperature=args.temperature,
            request_id=request_id,
        )


async def run_requests(
    *,
    args: argparse.Namespace,
    case: PrefixCase,
    request_count: int,
    concurrency: int,
    max_tokens: int,
    request_id_offset: int = 0,
) -> tuple[float, list[RequestResult]]:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []
    start = time.perf_counter()

    for idx in range(1, request_count + 1):
        request_id = request_id_offset + idx
        tasks.append(
            asyncio.create_task(
                send_one_request(
                    args=args,
                    case=case,
                    request_id=request_id,
                    max_tokens=max_tokens,
                    semaphore=semaphore,
                )
            )
        )
        if args.request_rate > 0 and idx < request_count:
            await asyncio.sleep(1 / args.request_rate)

    results = await asyncio.gather(*tasks)
    return time.perf_counter() - start, results


async def warm_prefix_cache(args: argparse.Namespace, case: PrefixCase) -> list[RequestResult]:
    if not case.has_shared_prefix or args.warmup_count <= 0:
        return []
    _, results = await run_requests(
        args=args,
        case=case,
        request_count=args.warmup_count,
        concurrency=1,
        max_tokens=args.warmup_max_tokens,
        request_id_offset=10_000,
    )
    return results


def summarize_case(
    *,
    args: argparse.Namespace,
    run_id: str,
    timestamp: str,
    case: PrefixCase,
    wall_seconds: float,
    results: list[RequestResult],
    warmup_results: list[RequestResult],
    metrics_before: dict[str, float],
    metrics_after: dict[str, float],
    metrics_error: str | None,
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

    query_delta = metric_delta(metrics_before, metrics_after, PREFIX_CACHE_QUERIES)
    hit_delta = metric_delta(metrics_before, metrics_after, PREFIX_CACHE_HITS)
    hit_rate = None
    if query_delta is not None and query_delta > 0 and hit_delta is not None:
        hit_rate = hit_delta / query_delta

    notes = args.notes
    note_parts = []
    if metrics_error:
        note_parts.append(f"metrics={metrics_error[:160]}")
    if errors:
        kinds = sorted({(error.error or "unknown").splitlines()[0][:120] for error in errors})
        note_parts.append(f"errors={kinds}")
    warmup_errors = [result for result in warmup_results if not result.ok]
    if warmup_errors:
        note_parts.append(f"warmup_errors={len(warmup_errors)}")
    if case.shared_prefix_tokens + case.target_output_tokens > int(args.max_model_len):
        note_parts.append("case may exceed configured max_model_len")
    if note_parts:
        notes = f"{notes}; {'; '.join(note_parts)}" if notes else "; ".join(note_parts)

    block_size = int(args.block_size)
    expected_shared_blocks = case.shared_prefix_tokens // block_size if block_size > 0 else 0

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
        "case": case.name,
        "case_label": case.label,
        "prompt_style": case.prompt_style,
        "target_prompt_tokens": case.target_prompt_tokens,
        "shared_prefix_tokens": case.shared_prefix_tokens,
        "expected_shared_blocks": expected_shared_blocks,
        "target_output_tokens": case.target_output_tokens,
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
        "concurrency": args.concurrency,
        "request_rate": args.request_rate,
        "request_count": len(results),
        "warmup_count": len(warmup_results),
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
        "prefix_cache_queries_delta": metric_text(query_delta, digits=0),
        "prefix_cache_hits_delta": metric_text(hit_delta, digits=0),
        "prefix_cache_hit_rate": metric_text(hit_rate),
        "kv_cache_usage_perc_before": metric_text(metric_value(metrics_before, KV_CACHE_USAGE)),
        "kv_cache_usage_perc_after": metric_text(metric_value(metrics_after, KV_CACHE_USAGE)),
        "prompt_tokens_total_delta": metric_text(
            metric_delta(metrics_before, metrics_after, PROMPT_TOKENS_TOTAL),
            digits=0,
        ),
        "generation_tokens_total_delta": metric_text(
            metric_delta(metrics_before, metrics_after, GENERATION_TOKENS_TOTAL),
            digits=0,
        ),
        "request_prefill_time_seconds_sum_delta": metric_text(
            metric_delta(metrics_before, metrics_after, PREFILL_TIME_SUM),
        ),
        "request_decode_time_seconds_sum_delta": metric_text(
            metric_delta(metrics_before, metrics_after, DECODE_TIME_SUM),
        ),
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
        "warmup_requests": [
            {
                "request_id": result.request_id,
                "ok": result.ok,
                "e2e_latency_ms": result.e2e_latency_ms,
                "prompt_tokens": result.prompt_tokens,
                "output_tokens": result.output_tokens,
                "error": result.error,
            }
            for result in warmup_results
        ],
        "wall_seconds": wall_seconds,
        "purpose": case.purpose,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM PagedAttention prefix-cache block reuse through OpenAI-compatible chat."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "change-this-before-lan-use"))
    parser.add_argument("--model", default="Qwen3-8B-GGUF-vLLM-local")
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--profile", default="qwen3_8b_gguf_vllm_optimized")
    parser.add_argument("--quantization", default="gguf-q4_k_m")
    parser.add_argument("--route", default="WSL")
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--request-count", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--override-output-tokens",
        type=int,
        default=0,
        help="When > 0, replace each case's target output length with this value.",
    )
    parser.add_argument("--request-rate", type=float, default=0.0)
    parser.add_argument("--warmup-count", type=int, default=1)
    parser.add_argument("--warmup-max-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-prefix", type=Path, default=default_output_prefix())
    parser.add_argument("--notes", default="")
    parser.add_argument("--tokenizer-path", default="/mnt/e/GPTProject2/models/Qwen3-8B")
    parser.add_argument("--calibrate-prompts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metrics-url", default="")
    parser.add_argument("--metrics-timeout", type=int, default=10)
    parser.add_argument("--no-metrics", action="store_true")
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
    selected_cases = parse_csv_list(args.cases)
    unknown = [name for name in selected_cases if name not in CASES]
    if unknown:
        print(f"Unknown cases: {', '.join(unknown)}", file=sys.stderr)
        print(f"Available cases: {', '.join(sorted(CASES))}", file=sys.stderr)
        return 2
    if args.request_count < 1:
        print("--request-count must be >= 1", file=sys.stderr)
        return 2
    if args.concurrency < 1:
        print("--concurrency must be >= 1", file=sys.stderr)
        return 2

    if not args.metrics_url:
        args.metrics_url = f"http://{args.host}:{args.port}/metrics"

    run_id = str(uuid.uuid4())
    timestamp = now_iso()
    vllm_version = detect_vllm_version()
    gpu_name, vram_total, _ = query_gpu()
    args.local_tokenizer = load_tokenizer(args.tokenizer_path, args.calibrate_prompts)

    rows: list[dict[str, Any]] = []
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=args.concurrency + 4)
    loop.set_default_executor(executor)

    try:
        for case_name in selected_cases:
            case = CASES[case_name]
            if args.override_output_tokens > 0:
                case = replace(case, target_output_tokens=args.override_output_tokens)
            print(f"Running {case.name}: {case.label}", flush=True)
            warmup_results = await warm_prefix_cache(args, case)
            if warmup_results:
                ok = sum(1 for result in warmup_results if result.ok)
                print(f"  warmup ok={ok}/{len(warmup_results)}", flush=True)

            metrics_before, metrics_error = safe_fetch_metrics(args)
            wall_seconds, results = await run_requests(
                args=args,
                case=case,
                request_count=args.request_count,
                concurrency=args.concurrency,
                max_tokens=case.target_output_tokens,
            )
            metrics_after, after_error = safe_fetch_metrics(args)
            metrics_error = metrics_error or after_error
            _, _, gpu_mem_used = query_gpu()

            row = summarize_case(
                args=args,
                run_id=run_id,
                timestamp=timestamp,
                case=case,
                wall_seconds=wall_seconds,
                results=results,
                warmup_results=warmup_results,
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                metrics_error=metrics_error,
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
                f"e2e_p50={row['e2e_latency_ms_p50']}ms "
                f"output_tps={row['output_tps']} "
                f"prefix_hit_rate={row['prefix_cache_hit_rate'] or 'n/a'}",
                flush=True,
            )
    finally:
        executor.shutdown(wait=True)

    csv_path = args.output_prefix.with_suffix(".csv")
    jsonl_path = args.output_prefix.with_suffix(".jsonl")
    append_csv(csv_path, rows)
    append_jsonl(jsonl_path, rows)
    print(f"Wrote {len(rows)} prefix-cache benchmark rows:")
    print(f"  {csv_path}")
    print(f"  {jsonl_path}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.route == "WSL" and platform.system() == "Windows":
        print("Warning: route is WSL but this script is running under Windows.", file=sys.stderr)
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
