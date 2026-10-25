# Results

**Setup:** model `Qwen/Qwen2.5-0.5B-Instruct`, float16, hardware: Tesla T4, torch 2.11.0+cu130, transformers 5.20.2, run date 2026-10-25 20:43:23 UTC, commit `b441375`.

**Method:** greedy decoding (temperature 0), `ignore_eos` so every request generates its requested length, all engines serve the same model/dtype on the same GPU with `max_ctx` 4096 (prompts that do not fit are rejected, not truncated). TTFT is client-visible: time to the first non-empty content chunk. Open-loop Poisson arrivals; SLO = TTFT and TPOT thresholds shown with each table. 'Greedy agreement' in the precision table measures numerical consistency with fp32, not task quality; perplexity is on WikiText-2 only when the row says so.

### Key findings (auto-generated from the data)

- Sequential serving never met the SLO at any tested rate, while continuous-16 sustained 3 req/s, continuous-32 sustained 6 req/s, vllm sustained 12 req/s.
- At 3 req/s (the highest rate where continuous-16 meets the SLO), continuous batching has 195x lower p95 TTFT than static batching of the same size (0.12s vs 22.66s). Static batching meets the SLO at no tested rate.
- Against vLLM on the same GPU, our best engine (continuous-32, SLO capacity 6 req/s) reaches 50% of vLLM's SLO capacity (12 req/s).
- Closed-loop peak: **continuous-16** 303 tok/s = 11.7x sequential (26 tok/s).
- Closed-loop peak: **continuous-32** 469 tok/s = 18.2x sequential (26 tok/s).
- Closed-loop peak: **static-16** 161 tok/s = 6.3x sequential (26 tok/s).
- Closed-loop peak: **vllm** 1399 tok/s = 54.3x sequential (26 tok/s).
- Decode step time is nearly flat in batch size: within 1.5x of batch 1 (77.70 ms) up to batch **128**, where decode tokens/s is 89.0x batch 1. This is why batching raises throughput.
- At batch 128 a decode step takes 111.73 ms, 3.9% of the assumed 320 GB/s peak bandwidth: well below the memory roofline, so the step is not bandwidth-bound; kernel-launch / Python overhead is the likely limiter (check profile_decode.txt).
- Prefill of a 2048-token prompt takes 417.7 ms when only the last position's logits are computed vs 435.1 ms with full-vocab logits at every position (1.0x).
- Computing the lm_head only for the last prompt position raises closed-loop throughput (260 -> 268 tok/s, 1.03x; run-to-run spread is not shown, so treat differences under ~5% as noise) and moves the share of GPU time spent in prefill from 12% to 12%.
- float16/int8: 0.26x throughput vs fp32, perplexity 22.455 vs 22.426, greedy outputs match fp32 for the first 58% of tokens on average.
- float16/nf4: 0.81x throughput vs fp32, perplexity 25.025 vs 22.426, greedy outputs match fp32 for the first 17% of tokens on average.
- float16/none: 1.08x throughput vs fp32, perplexity 22.417 vs 22.426, greedy outputs match fp32 for the first 98% of tokens on average.
- Decode step time (kernel only, no scheduler): CUDA-graph slots are 3.6x-4.6x faster than the HF dynamic-cache step across batch sizes 1-32; the gain shrinks as batch grows because compute, not launch overhead, starts to dominate.
- Static slots + CUDA graphs move SLO capacity from 4 (in [4, 6)) to 10 (in [10, 12)) req/s on the same model, GPU and load (2.50x by the lower bound). Every rate was run with 2 seed(s); a rate counts as meeting the SLO only if every seed does.
- Of the end-to-end gain, CUDA graphs account for 2.02x closed-loop throughput on top of the static-slot engine (477 -> 964 tok/s); the static KV layout alone is the remainder.
- Closed-loop peak throughput: 414 -> 964 tok/s (2.33x); the difference is outside one standard deviation of the repeats.
- Against vLLM on the same GPU, slot+graphs reach 83% of its SLO capacity by lower bounds (10 vs 12 req/s) and 69% of its closed-loop throughput.
- Chunked prefill (chunk-256) on a workload with ~12% long (~1.3k-token) prompts changes p95 TPOT by 1.01x and p95 TTFT by 1.33x relative to whole-prompt prefill (mean over 4 rates; <1 is better).
- Prefix caching on a workload where every prompt shares a ~250-token preamble: mean prefix-token hit rate 92%, p95 TTFT 0.65x-0.84x of the uncached engine across rates (<1 is better; rates beyond saturation are dominated by queueing).
- On the larger second model the slot+graph engine reaches 391 tok/s vs 223 for the dynamic-cache engine at their best concurrency (1.75x), so the ranking holds across model sizes.

### Scheduler comparison (open-loop Poisson arrivals)

![scheduler](figures/scheduler_open_loop.png)

| config | SLO capacity (req/s) | peak output tok/s | TTFT p95 at capacity (s) | mean decode batch at max load |
|---|---|---|---|---|
| sequential | 0 (SLO never met) | 26.38 | n/a | 1.00 |
| static-16 | 0 (SLO never met) | 122.45 | n/a | 5.12 |
| continuous-16 | 3 (true value in [3, 4)) | 304.56 | 0.12 | 15.87 |
| continuous-32 | 6 (true value in [6, 7)) | 449.16 | 0.93 | 27.63 |
| vllm | 12 (true value in [12, 16)) | 1375.45 | 0.30 | n/a |

### Closed-loop throughput

![closed loop](figures/concurrency_closed_loop.png)

| config | at concurrency | peak tok/s (mean) | std |
|---|---|---|---|
| continuous-16 | 32 | 302.52 | 5.06 |
| continuous-32 | 64 | 468.79 | 7.09 |
| sequential | 4 | 25.77 | 0.24 |
| static-16 | 16 | 161.34 | 6.54 |
| vllm | 32 | 1398.68 | 7.09 |

### Where the time goes (microbenchmarks)

![microbench](figures/microbench.png)

| batch | step_ms | tok_per_s | achieved_gb_per_s | bw_utilisation |
|---|---|---|---|---|
| 1.000 | 77.698 | 12.870 | 12.757 | 0.040 |
| 2.000 | 87.797 | 22.780 | 11.326 | 0.035 |
| 4.000 | 97.287 | 41.116 | 10.286 | 0.032 |
| 8.000 | 70.559 | 113.380 | 14.360 | 0.045 |
| 16.000 | 99.710 | 160.466 | 10.414 | 0.033 |
| 32.000 | 69.966 | 457.366 | 15.561 | 0.049 |
| 64.000 | 99.183 | 645.269 | 11.992 | 0.037 |
| 128.000 | 111.731 | 1145.608 | 12.447 | 0.039 |

### Optimisation ablation: last-position logits in prefill

![ablation](figures/ablation_logits.png)

| variant | closed-loop tok/s | prefill time share |
|---|---|---|
| full-logits | 260.340 | 0.120 |
| last-logits | 268.476 | 0.119 |

### Precision / quantization trade-off

![precision](figures/precision.png)

| dtype | quant | tok_s | std | gpu_mem_peak_mb | perplexity | eval_tokens | greedy_token_agreement | greedy_prefix_match | greedy_exact_match | param_mb |
|---|---|---|---|---|---|---|---|---|---|---|
| float16 | int8 | 62.144 | 4.164 | 929.799 | 22.455 | 8176 | 0.592 | 0.585 | 0.417 | 601.043 |
| float16 | nf4 | 195.066 | 13.143 | 780.810 | 25.025 | 8176 | 0.233 | 0.168 | 0.000 | 430.418 |
| float16 | none | 259.520 | 16.123 | 1279.159 | 22.417 | 8176 | 0.977 | 0.977 | 0.917 | 942.293 |
| float32 | none | 239.800 | 16.513 | 2525.083 | 22.426 | 8176 | 1.000 | 1.000 | 1.000 | 1884.585 |

### Decode kernel microbenchmark

![decode graphs](figures/decode_graphs.png)

| batch | hf-dynamic-cache (eager) | slot + cuda graph | slot eager (static KV) |
|---|---|---|---|
| 1.00 | 36.37 | 9.04 | 29.14 |
| 2.00 | 35.73 | 8.29 | 28.64 |
| 4.00 | 39.00 | 8.49 | 29.41 |
| 8.00 | 36.94 | 8.79 | 29.32 |
| 16.00 | 36.95 | 9.08 | 29.00 |
| 32.00 | 37.66 | 10.56 | 30.32 |

### Engine comparison: dynamic cache vs static slots vs static slots + CUDA graphs

![engines](figures/engines.png)

| engine | SLO capacity (req/s) | seeds per rate | p95 TTFT at capacity (s) | peak open-loop tok/s | closed-loop peak tok/s |
|---|---|---|---|---|---|
| dynamic-32 | 4 (in [4, 6)) | 2 | 0.14 ± 0.01 | 448 ± 6 | 414 ± 13 (c=32, n=3) |
| slot-32 | 4 (in [4, 6)) | 2 | 0.11 ± 0.01 | 529 ± 5 | 477 ± 11 (c=32, n=3) |
| slot-graph-32 | 10 (in [10, 12)) | 2 | 0.69 ± 0.15 | 1029 ± 87 | 964 ± 14 (c=32, n=3) |
| vllm (baseline) | 12 (in [12, 16)) | 1 | n/a | n/a | 1399 |

*vLLM baseline: completed.* Same model, fp16, same load generator and TTFT definition (first non-empty content chunk).

### Chunked prefill (long-prompt workload)

![chunked](figures/chunked_prefill.png)

| config | offered_rps | ttft_p95 | tpot_p95 | e2e_p95 | good | att | n |
|---|---|---|---|---|---|---|---|
| chunk-256 | 0.500 | 0.169 | 0.013 | 1.430 | 0.552 | 1.000 | 2 |
| chunk-256 | 1.000 | 0.152 | 0.013 | 1.488 | 0.863 | 1.000 | 2 |
| chunk-256 | 2.000 | 0.252 | 0.015 | 1.850 | 1.950 | 1.000 | 2 |
| chunk-256 | 3.000 | 0.292 | 0.021 | 2.521 | 3.377 | 1.000 | 2 |
| no-chunking | 0.500 | 0.146 | 0.013 | 1.432 | 0.552 | 1.000 | 2 |
| no-chunking | 1.000 | 0.120 | 0.013 | 1.456 | 0.863 | 1.000 | 2 |
| no-chunking | 2.000 | 0.180 | 0.015 | 1.779 | 1.951 | 1.000 | 2 |
| no-chunking | 3.000 | 0.193 | 0.020 | 2.395 | 3.389 | 1.000 | 2 |

### Prefix caching (shared-preamble workload)

![prefix](figures/prefix_cache.png)

| config | offered_rps | ttft_p95 | tok | good | att | n |
|---|---|---|---|---|---|---|
| no-prefix-cache | 2 | 0.091 | 84.693 | 1.823 | 1.000 | 2 |
| no-prefix-cache | 4 | 0.129 | 178.002 | 3.810 | 1.000 | 2 |
| no-prefix-cache | 8 | 0.146 | 363.645 | 7.576 | 1.000 | 2 |
| no-prefix-cache | 12 | 0.252 | 513.363 | 10.765 | 1.000 | 2 |
| no-prefix-cache | 16 | 3.798 | 643.541 | 3.962 | 0.292 | 2 |
| prefix-cache | 2 | 0.072 | 84.767 | 1.825 | 1.000 | 2 |
| prefix-cache | 4 | 0.104 | 177.966 | 3.809 | 1.000 | 2 |
| prefix-cache | 8 | 0.123 | 363.691 | 7.577 | 1.000 | 2 |
| prefix-cache | 12 | 0.183 | 515.108 | 10.802 | 1.000 | 2 |
| prefix-cache | 16 | 2.481 | 680.176 | 6.327 | 0.440 | 2 |

### Second model size

![scale](figures/scale_model.png)

| config | concurrency | mean | std | count |
|---|---|---|---|---|
| dynamic-16 | 1 | 22.9 | 0.2 | 3 |
| dynamic-16 | 8 | 129.8 | 6.8 | 3 |
| dynamic-16 | 16 | 223.1 | 6.9 | 3 |
| slot-graph-16 | 1 | 52.1 | 0.1 | 3 |
| slot-graph-16 | 8 | 254.9 | 16.7 | 3 |
| slot-graph-16 | 16 | 391.4 | 13.5 | 3 |
