# 2026-05-09 AWQ-Marlin FP8 KV Cache

## Goal

Test whether FP8 KV cache can expand usable KV capacity for longer context or
more concurrency, and require a small quality regression set before treating it
as a profile candidate.

Test order:

```text
A: kv_cache_dtype = auto
B: kv_cache_dtype = fp8_e5m2
C: kv_cache_dtype = fp8_e4m3
```

## Startup Results

First, the current graph-oriented 4096-token shape was tested:

```text
max_model_len = 4096
max_num_seqs = 2
max_num_batched_tokens = 4096
gpu_memory_utilization = 0.85
enforce_eager = false
```

| kv dtype | status | KV GiB | GPU KV tokens | max concurrency | note |
| --- | --- | ---: | ---: | ---: | --- |
| auto | failed |  |  |  | available KV memory was negative |
| fp8_e5m2 | failed | 0.41 | 6032 | 1.47x | capacity improved but still below 2x4096 |
| fp8_e4m3 | failed | 0.41 | 6032 | 1.47x | capacity improved but still below 2x4096 |

Then the 4096-token long-context shape was tested:

```text
max_model_len = 4096
max_num_seqs = 2
max_num_batched_tokens = 2048
gpu_memory_utilization = 0.85
enforce_eager = true
```

| kv dtype | attention backend | status | KV GiB | GPU KV tokens | max concurrency | note |
| --- | --- | --- | ---: | ---: | ---: | --- |
| auto | default | ready | 0.58 | 4240 | 1.04x | fits one 4096-token request only |
| fp8_e5m2 | default | failed | 0.65 | 9408 | 2.30x | FlashInfer JIT required `nvcc` |
| fp8_e4m3 | default | failed | 0.65 | 9408 | 2.30x | FlashInfer JIT required `nvcc` |
| auto | TRITON_ATTN | ready | 0.65 | 4704 | 1.15x | quality baseline |
| fp8_e5m2 | TRITON_ATTN | failed |  |  |  | Triton path asserted only `fp8/fp8_e4m3` |
| fp8_e4m3 | TRITON_ATTN | ready | 0.59 | 8512 | 2.08x | long-context candidate |

The key capacity result is `fp8_e4m3 + TRITON_ATTN`: it starts at
`max_model_len=4096`, `max_num_seqs=2` and exposes `8512` GPU KV tokens, enough
for roughly two 4096-token requests. The comparable auto/TRITON row exposes
`4704` GPU KV tokens.

## Quality Regression

The quality guardrail covers:

- Chinese QA
- English QA
- math/code short task
- long-context summary
- agent tool-use style system prompt

Both rows used:

```text
max_model_len = 4096
max_num_seqs = 2
max_num_batched_tokens = 2048
attention_backend = TRITON_ATTN
enforce_eager = true
temperature = 0
```

| kv dtype | cn qa | en qa | math/code | long summary | agent tool-use | quality exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| auto | pass | pass | pass | pass | pass | 0 |
| fp8_e4m3 | pass | pass | pass | pass | pass | 0 |

Prompt/response token samples:

| kv dtype | case | prompt tokens | output tokens | latency ms |
| --- | --- | ---: | ---: | ---: |
| auto | cn_qa | 54 | 78 | 2639.38 |
| auto | en_qa | 40 | 74 | 2548.03 |
| auto | math_code | 53 | 49 | 1727.36 |
| auto | long_context_summary | 3286 | 44 | 3109.86 |
| auto | agent_tool_use | 77 | 82 | 2385.77 |
| fp8_e4m3 | cn_qa | 54 | 73 | 2977.67 |
| fp8_e4m3 | en_qa | 40 | 69 | 2486.59 |
| fp8_e4m3 | math_code | 53 | 49 | 1682.69 |
| fp8_e4m3 | long_context_summary | 3286 | 44 | 3224.01 |
| fp8_e4m3 | agent_tool_use | 77 | 81 | 2429.35 |

## Decision

Do not change the default interactive profile.

Add `fp8_e4m3 + TRITON_ATTN + enforce_eager` as an experimental long-context
profile:

```bash
bash ./scripts/start_vllm_qwen3_awq_fp8kv_wsl.sh /mnt/e/GPTProject2/vLLM
```

Reasoning:

- FP8 KV materially increases capacity: `4704` to `8512` GPU KV tokens in the
  comparable TRITON rows.
- The quality regression set passed for `fp8_e4m3`.
- The profile requires a backend change to `TRITON_ATTN` and eager mode, so it
  is not a default for short chat or the current 2048-token interactive route.
- `fp8_e5m2` is not a candidate in this environment.
- Default FlashInfer with FP8 currently needs an nvcc/CUDA toolkit path for JIT,
  which this WSL setup does not have.

## Artifacts

```text
reports/memory/2026-05-09-vllm-awq-marlin-kv-dtype-graph4096.csv
reports/memory/2026-05-09-vllm-awq-marlin-kv-dtype-graph4096.jsonl
reports/memory/2026-05-09-vllm-awq-marlin-kv-dtype-eager4096.csv
reports/memory/2026-05-09-vllm-awq-marlin-kv-dtype-eager4096.jsonl
reports/memory/2026-05-09-vllm-awq-marlin-kv-dtype-eager4096-triton.csv
reports/memory/2026-05-09-vllm-awq-marlin-kv-dtype-eager4096-triton.jsonl
reports/memory/2026-05-09-vllm-awq-marlin-kv-dtype-eager4096-triton-auto.csv
reports/memory/2026-05-09-vllm-awq-marlin-kv-dtype-eager4096-triton-auto.jsonl
reports/quality/2026-05-09-vllm-awq-marlin-kv-quality-auto-triton_attn.csv
reports/quality/2026-05-09-vllm-awq-marlin-kv-quality-auto-triton_attn.jsonl
reports/quality/2026-05-09-vllm-awq-marlin-kv-quality-auto-triton_attn.status
reports/quality/2026-05-09-vllm-awq-marlin-kv-quality-fp8_e4m3-triton_attn.csv
reports/quality/2026-05-09-vllm-awq-marlin-kv-quality-fp8_e4m3-triton_attn.jsonl
reports/quality/2026-05-09-vllm-awq-marlin-kv-quality-fp8_e4m3-triton_attn.status
```
