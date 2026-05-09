#!/usr/bin/env python3
"""Small deterministic quality regression set for OpenAI-compatible chat.

This is not a full eval harness. It is a guardrail for infrastructure changes
such as FP8 KV cache where we need the answers saved beside benchmark artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import time
import urllib.error
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
    "kv_cache_dtype",
    "vllm_version",
    "gpu_name",
    "vram_total",
    "route",
    "case_id",
    "category",
    "case_name",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "gpu_memory_utilization",
    "block_size",
    "attention_backend",
    "prefix_caching_enabled",
    "chunked_prefill_enabled",
    "async_scheduling_enabled",
    "enforce_eager",
    "temperature",
    "max_tokens",
    "latency_ms",
    "prompt_tokens",
    "output_tokens",
    "total_tokens",
    "passed",
    "score",
    "required_terms",
    "missing_terms",
    "response_excerpt",
    "error",
    "notes",
]


@dataclass(frozen=True)
class QualityCase:
    case_id: str
    category: str
    name: str
    messages: list[dict[str, str]]
    max_tokens: int
    required_terms: tuple[str, ...]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def detect_vllm_version() -> str:
    try:
        import vllm  # type: ignore

        return getattr(vllm, "__version__", "")
    except Exception:
        return ""


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


def long_context_text() -> str:
    facts = (
        "Project Alpha latency p95 is 42 ms. "
        "Project Beta memory ceiling is 6.5 GB. "
        "Project Gamma rollback is disabled until the canary passes. "
    )
    filler = (
        "This synthetic operations note is repeated to create a long context. "
        "Ignore repeated filler and preserve exact numeric facts in the final summary. "
    )
    return (facts + filler) * 55


def cases() -> list[QualityCase]:
    return [
        QualityCase(
            case_id="cn_qa",
            category="中文问答",
            name="KV cache explanation in Chinese",
            messages=[
                {"role": "system", "content": "你是一个简洁、准确的 AI infra 助手。"},
                {
                    "role": "user",
                    "content": (
                        "用三句话解释 KV cache 在大语言模型推理中的作用，"
                        "并说明它主要影响 prefill 还是 decode。"
                    ),
                },
            ],
            max_tokens=180,
            required_terms=("KV", "cache|缓存", "prefill|预填充", "decode|解码"),
        ),
        QualityCase(
            case_id="en_qa",
            category="英文问答",
            name="Continuous batching explanation in English",
            messages=[
                {"role": "system", "content": "You are a concise AI infrastructure tutor."},
                {
                    "role": "user",
                    "content": (
                        "In two bullet points, explain why continuous batching improves "
                        "LLM serving throughput."
                    ),
                },
            ],
            max_tokens=128,
            required_terms=(
                "batch|batching",
                "throughput|handled per unit time|requests handled",
                "request|requests",
            ),
        ),
        QualityCase(
            case_id="math_code",
            category="数学/代码短任务",
            name="Short Python function",
            messages=[
                {"role": "system", "content": "Return concise, correct code with a one-line explanation."},
                {
                    "role": "user",
                    "content": (
                        "Write a Python function add_even_numbers(nums) that returns the "
                        "sum of the even integers in nums. Include one tiny example."
                    ),
                },
            ],
            max_tokens=220,
            required_terms=("def add_even_numbers", "sum", "% 2|mod"),
        ),
        QualityCase(
            case_id="long_context_summary",
            category="长上下文摘要",
            name="Long context factual summary",
            messages=[
                {"role": "system", "content": "Summarize long operational notes without losing exact numbers."},
                {
                    "role": "user",
                    "content": (
                        long_context_text()
                        + "\n\nSummarize the three project facts in exactly three bullets."
                    ),
                },
            ],
            max_tokens=96,
            required_terms=("42", "6.5", "rollback|disabled|禁用"),
        ),
        QualityCase(
            case_id="agent_tool_use",
            category="agent tool-use 风格 system prompt",
            name="Tool selection plan",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an agent planner. Available tools are "
                        "search_docs(query) and open_file(path). Return compact JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Plan tool calls to investigate high TTFT in a vLLM service when "
                        "many requests share the same system prefix. First search docs, "
                        "then open the local benchmark config and server log."
                    ),
                },
            ],
            max_tokens=220,
            required_terms=("search_docs", "open_file", "TTFT|ttft", "prefix"),
        ),
    ]


def request_json(url: str, payload: dict[str, Any], api_key: str, timeout: int) -> dict[str, Any]:
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
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def check_terms(text: str, terms: tuple[str, ...]) -> tuple[bool, int, list[str]]:
    lower = text.lower()
    missing: list[str] = []
    for term in terms:
        alternatives = [part.strip().lower() for part in term.split("|") if part.strip()]
        if not any(alt in lower for alt in alternatives):
            missing.append(term)
    passed = not missing
    score = len(terms) - len(missing)
    return passed, score, missing


def run_case(args: argparse.Namespace, case: QualityCase, base: dict[str, Any]) -> dict[str, Any]:
    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": case.messages,
        "temperature": args.temperature,
        "max_tokens": case.max_tokens,
    }
    if args.disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    started = time.perf_counter()
    error = ""
    response_text = ""
    prompt_tokens = ""
    output_tokens = ""
    total_tokens = ""
    try:
        response = request_json(url, payload, args.api_key, args.timeout)
        latency_ms = (time.perf_counter() - started) * 1000
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        response_text = message.get("content") or ""
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", "")
        output_tokens = usage.get("completion_tokens", "")
        total_tokens = usage.get("total_tokens", "")
    except urllib.error.HTTPError as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        error = f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}"
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        error = str(exc)

    passed, score, missing = check_terms(response_text, case.required_terms)
    if error:
        passed = False

    row = {
        **base,
        "case_id": case.case_id,
        "category": case.category,
        "case_name": case.name,
        "temperature": args.temperature,
        "max_tokens": case.max_tokens,
        "latency_ms": f"{latency_ms:.2f}",
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "passed": bool_text(passed),
        "score": score,
        "required_terms": ";".join(case.required_terms),
        "missing_terms": ";".join(missing),
        "response_excerpt": response_text.replace("\r", " ").replace("\n", " ")[:600],
        "error": error,
        "notes": args.notes,
        "messages": case.messages,
        "response": response_text,
    }
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


def default_output_prefix() -> Path:
    date = datetime.now().strftime("%Y-%m-%d")
    return Path("reports") / "quality" / f"{date}-openai-quality-regression"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small OpenAI-compatible quality regression set.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "change-this-before-lan-use"))
    parser.add_argument("--model", default="Qwen3-8B-AWQ-vLLM-local")
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--profile", default="qwen3_8b_awq_marlin_quality")
    parser.add_argument("--quantization", default="awq-marlin-int4")
    parser.add_argument("--kv-cache-dtype", default=os.environ.get("VLLM_KV_CACHE_DTYPE", "auto"))
    parser.add_argument("--route", default="WSL")
    parser.add_argument("--max-model-len", default=os.environ.get("VLLM_MAX_MODEL_LEN", "4096"))
    parser.add_argument("--max-num-seqs", default=os.environ.get("VLLM_MAX_NUM_SEQS", "1"))
    parser.add_argument(
        "--max-num-batched-tokens",
        default=os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", "2048"),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        default=os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.85"),
    )
    parser.add_argument("--block-size", default=os.environ.get("VLLM_BLOCK_SIZE", "16"))
    parser.add_argument("--attention-backend", default=os.environ.get("VLLM_ATTENTION_BACKEND", ""))
    parser.add_argument("--prefix-caching-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunked-prefill-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--async-scheduling-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-prefix", type=Path, default=default_output_prefix())
    parser.add_argument("--notes", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = str(uuid.uuid4())
    timestamp = now_iso()
    gpu_name, vram_total, _ = query_gpu()
    base = {
        "run_id": run_id,
        "timestamp": timestamp,
        "backend": args.backend,
        "profile": args.profile,
        "model": args.model,
        "quantization": args.quantization,
        "kv_cache_dtype": args.kv_cache_dtype,
        "vllm_version": detect_vllm_version(),
        "gpu_name": gpu_name,
        "vram_total": vram_total or "",
        "route": args.route,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "block_size": args.block_size,
        "attention_backend": args.attention_backend,
        "prefix_caching_enabled": bool_text(args.prefix_caching_enabled),
        "chunked_prefill_enabled": bool_text(args.chunked_prefill_enabled),
        "async_scheduling_enabled": bool_text(args.async_scheduling_enabled),
        "enforce_eager": bool_text(args.enforce_eager),
    }

    if args.route == "WSL" and platform.system() == "Windows":
        print("Warning: route is WSL but this script is running under Windows.")

    rows = []
    for case in cases():
        print(f"Running {case.case_id}: {case.name}", flush=True)
        row = run_case(args, case, base)
        rows.append(row)
        print(
            f"  passed={row['passed']} score={row['score']}/{len(case.required_terms)} "
            f"latency={row['latency_ms']}ms missing={row['missing_terms'] or '-'}",
            flush=True,
        )

    append_csv(args.output_prefix.with_suffix(".csv"), rows)
    append_jsonl(args.output_prefix.with_suffix(".jsonl"), rows)
    print(f"Wrote {len(rows)} quality rows:")
    print(f"  {args.output_prefix.with_suffix('.csv')}")
    print(f"  {args.output_prefix.with_suffix('.jsonl')}")
    return 0 if all(row["passed"] == "true" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
