# Memory Model And KV Cache

Stage 2 focuses on understanding how vLLM spends GPU memory at startup and how much dynamic serving capacity remains for KV cache.

## Working Model

For this project, reason about VRAM as:

```text
total VRAM ~= model weights
           + KV cache
           + CUDA graph / activation / workspace
           + runtime overhead
           + fragmentation / allocator reserve
```

The important lesson is that fitting model weights is not enough. For a serving engine, KV cache is the dynamic capacity budget that determines how much context and concurrency the server can sustain.

## Current AWQ-Marlin Baseline Observation

The current active route is `Qwen3-8B-AWQ + awq_marlin + CUDA graph`.
It changes the memory shape materially:

- Model memory: about `5.71 GiB`
- Best retained `2048` interactive KV capacity: `4480-4544` GPU KV tokens
- Best retained `1024` c4 KV capacity: `4416-4544` GPU KV tokens
- Best retained `4096` long-context KV capacity: `4368-4544` GPU KV tokens
- `gpu_memory_utilization=0.85` is the highest useful value in the first AWQ sweep
- `gpu_memory_utilization=0.86` and `0.88` failed with OOM-style startup boundaries

The follow-up eager-vs-graph pass found graph mode with `max_num_seqs=2` is the
current runtime winner after graph cache warmup:

- Eager seq2 startup: `3328` GPU KV tokens, no graph capture
- Graph seq2 retry startup: `3600` GPU KV tokens, graph capture true
- Graph seq2 request benchmark: lower ITL p95 and higher output TPS than eager
  across concurrency `1,2,4`

The first graph seq2 attempt timed out during cold compile/cache setup, so eager
remains an explicit fallback rather than the default.

The FP8 KV cache pass adds one experimental long-context profile:

```text
kv_cache_dtype = fp8_e4m3
attention_backend = TRITON_ATTN
max_model_len = 4096
max_num_seqs = 2
max_num_batched_tokens = 2048
enforce_eager = true
```

On this profile, GPU KV tokens increased from `4704` for auto/TRITON to `8512`
for `fp8_e4m3`/TRITON. This is enough for roughly two 4096-token requests, but
it is not the default because it requires eager mode and a backend change. See
`reports/2026-05-09-vllm-awq-marlin-fp8-kv-cache.md`.

The first AWQ-Marlin startup sweep is recorded in:

- `reports/memory/2026-05-08-vllm-awq-marlin-memory-sweep.md`
- `reports/memory/2026-05-08-vllm-awq-marlin-memory-sweep.csv`
- `reports/memory/2026-05-08-vllm-awq-marlin-memory-sweep.jsonl`
- `reports/memory/2026-05-08-vllm-awq-marlin-candidate-restarts.csv`
- `reports/memory/2026-05-08-vllm-awq-marlin-candidate-restarts.jsonl`

Retained startup candidates:

| candidate | max_model_len | gpu_util | max_num_seqs | max_num_batched_tokens | GPU KV tokens | role |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `best_interactive_capacity` | 2048 | 0.85 | 2 | 2048 | 4480-4544 | Better c2 interactive KV capacity than the current `batch=4096` baseline. |
| `best_c4_capacity` | 1024 | 0.85 | 4 | 2048 | 4416-4544 | True c4 capacity at 1024-token target context. |
| `best_long_context_capacity` | 4096 | 0.85 | 1 | 2048 | 4368-4544 | Single long-context profile; not a concurrent profile. |

The previous request baseline used `max_num_batched_tokens=4096`, which started
successfully but left less KV cache. In the AWQ sweep, the comparable
`2048, gpu=0.85, seqs=2, batch=4096` row had only `3056` GPU KV tokens, while
`batch=2048` had `4480` GPU KV tokens.

## Historical GGUF Observation

The older `Qwen3-8B-GGUF Q4_K_M + vLLM + WSL` launch showed:

- Model memory: about `4.82 GiB`
- Available KV cache memory: about `1.13 GiB`
- GPU KV cache size: about `8,224 tokens`
- Maximum concurrency at `2048` tokens/request: about `4.02x`
- `gpu_memory_utilization=0.80` starts successfully
- `gpu_memory_utilization=0.86` failed because free startup VRAM was just below the requested allocation

This made `gpu_memory_utilization` a capacity knob, not a free performance
knob. Higher values can leave too little room for CUDA graphs, workspaces,
allocator reserve, and fragmentation.

## What vLLM Does At Startup

At startup, vLLM profiles the model peak memory and uses the configured GPU memory budget to decide how much space remains for KV cache. The startup logs are therefore a primary data source for capacity analysis.

The fields to capture for every launch are:

```text
model memory
KV cache memory
number of GPU blocks
GPU KV tokens
max concurrency
whether CUDA graph capture happened
whether startup OOM happened
```

In vLLM `0.17.1`, logs directly expose GPU KV tokens. The profiling script derives GPU block count as:

```text
gpu_blocks = floor(gpu_kv_tokens / block_size)
```

and records the source as `derived_from_gpu_kv_tokens_div_block_size`.

## Stage 2 Sweep Matrix

The requested full matrix is:

| variable | values |
| --- | --- |
| `max_model_len` | `1024`, `2048`, `4096` |
| `gpu_memory_utilization` | `0.72`, `0.76`, `0.80`, `0.84` |
| `max_num_seqs` | `1`, `2`, `4`, `8` |
| `max_num_batched_tokens` | `1024`, `2048`, `4096` |
| `enforce_eager` | `true`, `false` |

This is `3 * 4 * 4 * 3 * 2 = 288` server starts, so it must be automated and should be run when the machine can be left alone.

## Runner

Use:

```bash
cd /mnt/e/GPTProject2/vLLM
source ~/.venvs/gptproject2-vllm/bin/activate
python ./scripts/profile_vllm_memory_sweep.py --preset pilot --kill-existing
```

For the full Stage 2 matrix:

```bash
python ./scripts/profile_vllm_memory_sweep.py \
  --preset stage2 \
  --confirm-large-sweep \
  --kill-existing \
  --notes stage2-full-memory-sweep
```

The runner writes:

- `reports/memory/YYYY-MM-DD-vllm-gguf-memory-profile.csv`
- `reports/memory/YYYY-MM-DD-vllm-gguf-memory-profile.jsonl`

Raw launch logs go under `logs/memory_profile/` and remain ignored by git.

For the current AWQ-Marlin route, pass the vLLM quantization value explicitly:

```bash
python ./scripts/profile_vllm_memory_sweep.py \
  --preset stage2 \
  --confirm-large-sweep \
  --kill-existing \
  --max-model-lens 1024,2048,4096 \
  --gpu-memory-utilizations 0.82,0.84,0.85,0.86,0.88 \
  --max-num-seqs-values 1,2,4 \
  --max-num-batched-tokens-values 2048,4096 \
  --enforce-eager-values true \
  --model-path /mnt/e/GPTProject2/vLLM/models/Qwen3-8B-AWQ \
  --tokenizer-path /mnt/e/GPTProject2/vLLM/models/Qwen3-8B-AWQ \
  --hf-config-path /mnt/e/GPTProject2/vLLM/models/Qwen3-8B-AWQ \
  --served-model-name Qwen3-8B-AWQ-vLLM-local \
  --profile qwen3_8b_awq_marlin_eager_vllm \
  --quantization awq-marlin-int4 \
  --vllm-quantization awq_marlin \
  --dtype auto \
  --output-prefix reports/memory/2026-05-08-vllm-awq-marlin-memory-sweep \
  --log-dir logs/memory_profile_awq_sweep \
  --notes awq-marlin-startup-sweep-enforce-eager
```

## First Pilot

The first pilot run is recorded in:

- `reports/memory/2026-05-05-vllm-gguf-memory-profile.csv`
- `reports/memory/2026-05-05-vllm-gguf-memory-profile.jsonl`

Pilot rows:

| max_model_len | gpu_memory_utilization | max_num_seqs | max_num_batched_tokens | enforce_eager | status | model GiB | KV GiB | GPU KV tokens | max concurrency | CUDA graph |
| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1024 | 0.72 | 1 | 1024 | false | ready | 4.82 | 0.83 | 6064 | 5.92x | true |
| 2048 | 0.80 | 8 | 4096 | false | ready | 4.82 | 1.19 | 8656 | 4.23x | true |
| 2048 | 0.84 | 8 | 4096 | false | ready | 4.82 | 1.12 | 8112 | 3.96x | true |
| 2048 | 0.80 | 8 | 4096 | true | ready | 4.82 | 0.61 | 4400 | 2.15x | false |

Early interpretation:

- All pilot rows started successfully; no OOM was observed in this small sample.
- The `0.80` non-eager baseline gave more KV tokens than the `0.84` pilot row, which reinforces that startup memory is not a simple monotonic arithmetic exercise. Repeat runs or the full matrix are needed before treating this as a stable ranking.
- `enforce_eager=true` skipped CUDA graph capture as expected, but in this pilot it also left less available KV cache. Do not assume eager always frees capacity.
- The previous `0.86` failure remains useful as a boundary observation: higher utilization can fail because of CUDA graph/workspace/allocator/fragmentation and current free VRAM, not just model-weight size.

## Interpretation Rules

- Compare KV cache memory and GPU KV tokens before comparing request TPS.
- Treat startup OOM as a valid boundary result, not a failed experiment.
- Watch whether CUDA graph capture happened. `enforce_eager=true` should normally skip graph capture and changes the memory shape.
- Do not assume a higher `gpu_memory_utilization` is better. It can reduce safety margin and fail in real startup even when arithmetic looks possible.
- After narrowing a promising memory profile, run the request-level benchmark from `docs/BENCHMARKING.md`.
