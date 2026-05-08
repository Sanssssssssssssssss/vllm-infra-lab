# 2026-05-08 AWQ-Marlin Startup Memory Sweep

## Goal

Find the real startup memory and KV-cache boundary for the current
`Qwen3-8B-AWQ + awq_marlin + enforce_eager` vLLM route. The earlier Stage 2
memory notes were based on the GGUF route, where the model loaded at about
`4.82 GiB`. The AWQ-Marlin route loads at about `5.71 GiB`, so the old KV-cache
capacity cannot be reused.

This run only measured startup. It did not run the request benchmark matrix.

## Script Change

`scripts/profile_vllm_memory_sweep.py` now supports:

```text
--vllm-quantization awq_marlin
```

This value is passed to `vllm serve --quantization`. The existing
`--quantization` field remains the reporting label, for example
`awq-marlin-int4`.

## Sweep Matrix

```text
max_model_len = 1024,2048,4096
gpu_memory_utilization = 0.82,0.84,0.85,0.86,0.88
max_num_seqs = 1,2,4
max_num_batched_tokens = 2048,4096
enforce_eager = true
```

Total: `90` server starts.

Artifacts:

```text
reports/memory/2026-05-08-vllm-awq-marlin-memory-sweep.csv
reports/memory/2026-05-08-vllm-awq-marlin-memory-sweep.jsonl
reports/memory/2026-05-08-vllm-awq-marlin-candidate-restarts.csv
reports/memory/2026-05-08-vllm-awq-marlin-candidate-restarts.jsonl
```

## Result Summary

| status | oom | count |
| --- | --- | ---: |
| ready | false | 33 |
| failed | false | 21 |
| failed | true | 36 |

All `0.86` and `0.88` configurations failed with OOM-style startup boundaries.
The highest useful `gpu_memory_utilization` in this sweep was `0.85`.

`max_num_batched_tokens=4096` consistently reduced remaining KV capacity and
caused more non-OOM startup failures. For capacity-focused startup profiles,
`2048` is the better first candidate.

## Retained Candidates

| candidate | max_model_len | gpu_util | max_num_seqs | max_num_batched_tokens | sweep KV tokens | restart KV tokens | max concurrency | GPU used after start |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| best_interactive_capacity | 2048 | 0.85 | 2 | 2048 | 4480 | 4544, 4544 | 2.22x | 7484-7498 MB |
| best_c4_capacity | 1024 | 0.85 | 4 | 2048 | 4416 | 4544, 4544 | 4.44x | 7516 MB |
| best_long_context_capacity | 4096 | 0.85 | 1 | 2048 | 4368 | 4544, 4544 | 1.11x | 7498-7499 MB |

The `best_c4_capacity` candidate is deliberately a `1024` context profile. No
AWQ-Marlin startup profile in this sweep can support four full `2048`-token
contexts. The `2048, max_num_seqs=4, batch=2048` profile starts, but its
effective max concurrency at 2048 tokens/request is about `2.0-2.2x`, so it is
not a true c4 full-context capacity candidate.

## Current Baseline Comparison

The request benchmark baseline used:

```text
max_model_len = 2048
gpu_memory_utilization = 0.85
max_num_seqs = 2
max_num_batched_tokens = 4096
```

That startup profile is ready, but the sweep measured only `3056` GPU KV tokens.
Switching `max_num_batched_tokens` from `4096` to `2048` raised the same
interactive shape to `4480` GPU KV tokens in the sweep and `4544` tokens in both
restart checks.

## Interpretation

The AWQ-Marlin route is stable at `gpu_memory_utilization=0.85`, but it is a
tight 8GB profile. Candidate GPU memory after startup is around `7.48-7.52 GB`
out of `8.19 GB`, leaving roughly `0.67-0.70 GB` of headroom. This is not fully
pegged, but it is close enough that larger batching, CUDA graph changes, or
background GPU load can move the profile across the boundary.

Use the retained candidates as startup-capacity candidates only. Before changing
the serving default, rerun the request benchmark matrix for the chosen profile.
