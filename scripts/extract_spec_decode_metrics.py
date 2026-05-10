#!/usr/bin/env python3
"""Summarize vLLM speculative decoding counters from Prometheus snapshots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COUNTERS = {
    "vllm:spec_decode_num_drafts": "num_drafts",
    "vllm:spec_decode_num_draft_tokens": "num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens": "num_accepted_tokens",
}

FIELDNAMES = [
    "timestamp",
    "variant",
    "profile",
    "speculative_config",
    "metrics_exposed",
    "num_drafts_delta",
    "num_draft_tokens_delta",
    "num_accepted_tokens_delta",
    "acceptance_rate",
    "mean_acceptance_length",
    "per_position_acceptance_rate",
    "before_metrics",
    "after_metrics",
    "notes",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def metric_name(raw: str) -> str:
    name = raw.split("{", 1)[0]
    if name.endswith("_total"):
        name = name[: -len("_total")]
    return name


def parse_labels(raw: str) -> dict[str, str]:
    if "{" not in raw:
        return {}
    body = raw.split("{", 1)[1].rsplit("}", 1)[0]
    labels: dict[str, str] = {}
    for key, value in re.findall(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"', body):
        labels[key] = value.replace(r"\"", '"')
    return labels


def parse_prometheus(path: Path) -> tuple[dict[str, float], dict[int, float]]:
    counters: dict[str, float] = {alias: 0.0 for alias in COUNTERS.values()}
    accepted_by_position: dict[int, float] = {}

    if not path.exists():
        return counters, accepted_by_position

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = metric_name(parts[0])
        try:
            value = float(parts[1])
        except ValueError:
            continue

        alias = COUNTERS.get(name)
        if alias is not None:
            counters[alias] += value
            continue

        if name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            labels = parse_labels(parts[0])
            try:
                position = int(labels.get("position", ""))
            except ValueError:
                continue
            accepted_by_position[position] = accepted_by_position.get(position, 0.0) + value

    return counters, accepted_by_position


def clean_float(value: float | None, digits: int = 6) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute speculative decoding counter deltas from vLLM /metrics snapshots."
    )
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--speculative-config", default="")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--notes", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    before, before_pos = parse_prometheus(args.before)
    after, after_pos = parse_prometheus(args.after)

    deltas = {
        key: max(0.0, after.get(key, 0.0) - before.get(key, 0.0))
        for key in COUNTERS.values()
    }
    position_deltas = {
        position: max(0.0, after_pos.get(position, 0.0) - before_pos.get(position, 0.0))
        for position in sorted(set(before_pos) | set(after_pos))
    }

    drafts = deltas["num_drafts"]
    draft_tokens = deltas["num_draft_tokens"]
    accepted_tokens = deltas["num_accepted_tokens"]
    metrics_exposed = draft_tokens > 0 or accepted_tokens > 0 or drafts > 0
    acceptance_rate = accepted_tokens / draft_tokens if draft_tokens > 0 else None
    mean_acceptance_length = 1 + accepted_tokens / drafts if drafts > 0 else None
    per_position = {
        str(position): (value / drafts if drafts > 0 else None)
        for position, value in position_deltas.items()
    }

    row = {
        "timestamp": now_iso(),
        "variant": args.variant,
        "profile": args.profile,
        "speculative_config": args.speculative_config,
        "metrics_exposed": "true" if metrics_exposed else "false",
        "num_drafts_delta": int(drafts),
        "num_draft_tokens_delta": int(draft_tokens),
        "num_accepted_tokens_delta": int(accepted_tokens),
        "acceptance_rate": clean_float(acceptance_rate),
        "mean_acceptance_length": clean_float(mean_acceptance_length),
        "per_position_acceptance_rate": json.dumps(
            {
                position: (None if value is None else round(value, 6))
                for position, value in per_position.items()
            },
            sort_keys=True,
        ),
        "notes": args.notes,
        "before_metrics": str(args.before),
        "after_metrics": str(args.after),
    }

    append_csv(args.output_prefix.with_suffix(".csv"), row)
    append_jsonl(args.output_prefix.with_suffix(".jsonl"), row)
    print(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
