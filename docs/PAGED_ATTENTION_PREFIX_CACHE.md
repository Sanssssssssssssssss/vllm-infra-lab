# PagedAttention Prefix Cache Blocks

Stage 3 turns PagedAttention from a slogan into a measurable block-level
experiment.

PagedAttention is a serving-system memory idea: vLLM partitions each request's
KV cache into fixed-size KV blocks. Logical request blocks can map to physical
blocks that live in non-contiguous GPU memory. Automatic Prefix Caching (APC)
adds another indirection: if two requests share the same prefix, their matching
logical prefix blocks can resolve to the same cached physical KV blocks instead
of being recomputed.

The mental model:

```text
request A: blocks [1, 2, 3]
request B: blocks [1, 2, 4]
request C: blocks [5, 6]
```

Requests A and B can reuse the physical KV blocks for `[1, 2]`. Request C
cannot. The optimization target is prefill work and KV memory reuse, not decode
throughput.

## Required Cases

Run the dedicated runner:

```bash
cd /mnt/e/GPTProject2/vLLM
source ~/.venvs/gptproject2-vllm/bin/activate
python ./scripts/bench_prefix_cache_blocks.py \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key change-this-before-lan-use \
  --model Qwen3-8B-AWQ-vLLM-local \
  --profile qwen3_8b_awq_marlin_stage3_prefix_blocks_block16_sha256 \
  --quantization awq-marlin-int4 \
  --request-count 10 \
  --concurrency 1 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.85 \
  --block-size 16 \
  --prefix-caching-hash-algo sha256 \
  --enforce-eager \
  --tokenizer-path /mnt/e/GPTProject2/vLLM/models/Qwen3-8B-AWQ \
  --notes stage3-prefix-cache-blocks
```

Default cases:

| case | requests | shared prefix | output | purpose |
| --- | ---: | ---: | ---: | --- |
| `case_a_distinct_1024` | 10 | none | 32 | Control with different prompts. |
| `case_b_shared_system_1024` | 10 | 1024-token system prefix | 32 | System-prompt APC. |
| `case_c_shared_document_2048` | 10 | 2048-token document prefix | 32 | Long-document APC. |

The runner also supports `case_a_distinct_2048` as an optional fairer control
for Case C:

```bash
python ./scripts/bench_prefix_cache_blocks.py \
  --cases case_a_distinct_2048,case_c_shared_document_2048 \
  --request-count 10 \
  --concurrency 1 \
  --max-model-len 4096 \
  --notes stage3-doc-prefix-control
```

## Server Preconditions

For the full Stage 3 run, start vLLM with prefix caching enabled and a long
enough model length:

```bash
./scripts/start_vllm_stage3_prefix_blocks_wsl.sh /mnt/e/GPTProject2/vLLM
```

This profile defaults to:

```text
VLLM_MAX_MODEL_LEN=4096
VLLM_MAX_NUM_SEQS=1
VLLM_MAX_NUM_BATCHED_TOKENS=2048
VLLM_GPU_MEMORY_UTILIZATION=0.85
VLLM_ENABLE_PREFIX_CACHING=1
VLLM_BLOCK_SIZE=16
VLLM_PREFIX_CACHING_HASH_ALGO=sha256
VLLM_KV_CACHE_METRICS=1
VLLM_ENFORCE_EAGER=1
```

The 4096-token AWQ-Marlin Stage 3 route uses eager mode because it sits on the
8GB VRAM KV boundary. The interactive 2048-token AWQ route can still use CUDA
graphs.

Case C has a 2048-token shared document prefix plus chat-template overhead,
unique question tokens, and 32 output tokens. A server launched with
`max_model_len=2048` is expected to reject or truncate this case.

On the current laptop profile, `block_size=16`, so the target shared-prefix
block counts are:

| case | shared prefix tokens | expected full shared blocks |
| --- | ---: | ---: |
| Case B | 1024 | 64 |
| Case C | 2048 | 128 |

These are expected blocks, not internal block IDs. vLLM does not expose the
exact physical block table through the OpenAI API.

## Warmup Rule

For shared-prefix cases, the runner sends one warmup request before taking the
`/metrics` baseline. That warmup seeds the KV blocks. The measured 10 requests
then test reuse of existing cached blocks.

Without warmup, a burst of identical-prefix requests can mix three effects:

- prefix cache behavior,
- first-time prefill cost,
- scheduler queueing and continuous batching behavior.

Warmup keeps the Stage 3 question sharp: "Do later requests reuse prefix KV
blocks?"

## Outputs

The runner writes:

```text
reports/benchmarks/YYYY-MM-DD-vllm-prefix-cache-blocks.csv
reports/benchmarks/YYYY-MM-DD-vllm-prefix-cache-blocks.jsonl
```

Important fields:

| field | meaning |
| --- | --- |
| `expected_shared_blocks` | `shared_prefix_tokens / block_size`, rounded down. |
| `prefix_caching_hash_algo` | Hash algorithm used by vLLM for prefix cache keys. |
| `ttft_ms_p50`, `ttft_ms_p95` | Main signal for reduced prefill work. |
| `e2e_latency_ms_p50`, `e2e_latency_ms_p95` | Should improve when output is short. |
| `output_tps` | Should stay roughly similar; decode is not the main APC target. |
| `prefix_cache_queries_delta` | vLLM prefix-cache query counter delta from `/metrics`. |
| `prefix_cache_hits_delta` | vLLM prefix-cache hit counter delta from `/metrics`. |
| `prefix_cache_hit_rate` | `hits_delta / queries_delta`. |
| `request_prefill_time_seconds_sum_delta` | Server-side prefill time delta, when exposed. |
| `request_decode_time_seconds_sum_delta` | Server-side decode time delta, when exposed. |

## Interpretation

Expected result shape:

- Case B should have lower TTFT than Case A after warmup.
- Case C should show an even stronger TTFT/E2E effect if `max_model_len=4096`
  is stable and prefix-cache hit metrics are present.
- Output TPS should not move much because APC does not shorten decode.
- If output is changed from 32 tokens to a long decode workload, APC benefit
  should fade because decode dominates total latency.
- If `prefix_cache_hits_delta` stays at zero for B/C, check that the server was
  launched with `--enable-prefix-caching`, prompts are byte-identical through the
  shared prefix, and `/metrics` is reachable.

Current AWQ-Marlin result:

- `reports/2026-05-09-vllm-awq-marlin-prefix-cache-blocks.md`
- APC was confirmed active: shared-prefix rows had hit rates near `0.96-0.99`
  and prefill sum dropped from about `5.0s` to about `0.46-0.51s`.
- Keep `block_size=16` and `sha256` as the default. Use `xxhash` only for a
  later high-QPS CPU-hash experiment. It requires the optional `xxhash` package.

Long-decode follow-up:

```bash
python ./scripts/bench_prefix_cache_blocks.py \
  --cases case_a_distinct_1024,case_b_shared_system_1024 \
  --request-count 10 \
  --concurrency 1 \
  --override-output-tokens 512 \
  --max-model-len 4096 \
  --max-num-seqs 2 \
  --notes stage3-prefix-cache-long-decode
```

## References

- vLLM APC user docs: <https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/>
- vLLM APC design docs: <https://docs.vllm.ai/en/v0.8.3/design/automatic_prefix_caching.html>
- vLLM metrics docs: <https://docs.vllm.ai/en/stable/design/metrics/>
- PagedAttention paper: <https://arxiv.org/abs/2309.06180>
