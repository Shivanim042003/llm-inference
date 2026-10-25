# Results

**Setup:** model `Qwen/Qwen2.5-0.5B-Instruct`, float16, hardware: Tesla T4, torch 2.11.0+cu130, transformers 5.20.2, run date 2026-10-25 15:40:22 UTC, commit `86d73d4`.

### Key findings (auto-generated from the data)

- Sequential serving never met the SLO at any tested rate, while continuous-16 sustained 4 req/s, continuous-32 sustained 4 req/s, vllm sustained 12 req/s.
- At 8 req/s, continuous batching cuts p95 TTFT 1.9x versus static batching of the same size (43.14s vs 22.48s).
- SLO capacity: continuous-16 4 req/s vs static-16 0 req/s.
- Against vLLM on the same GPU, our best engine (continuous-16) reaches 33% of vLLM's SLO capacity (4 vs 12 req/s).
- Closed-loop peak: **continuous-16** 322 tok/s = 10.8x sequential (30 tok/s).
- Closed-loop peak: **static-16** 170 tok/s = 5.7x sequential (30 tok/s).
- Closed-loop peak: **vllm** 1437 tok/s = 48.0x sequential (30 tok/s).
- Decode step time is nearly flat in batch size: within 1.5x of batch 1 (35.33 ms) up to batch **32**, where decode tokens/s is 32.8x batch 1. This is why batching raises throughput.
- At batch 64 a decode step takes 57.64 ms, 6.4% of the assumed 320 GB/s peak bandwidth: well below the memory roofline, so the step is not bandwidth-bound; kernel-launch / Python overhead is the likely limiter (check profile_decode.txt).
- float16/int8: 0.28x throughput vs fp32, perplexity 2.486 vs 2.489, 59% greedy-token agreement.
- float16/nf4: 0.88x throughput vs fp32, perplexity 2.616 vs 2.489, 23% greedy-token agreement.
- float16/none: 1.18x throughput vs fp32, perplexity 2.488 vs 2.489, 98% greedy-token agreement.

### Scheduler comparison (open-loop Poisson arrivals)

![scheduler](figures/scheduler_open_loop.png)

| config | SLO capacity (req/s) | peak output tok/s | TTFT p95 at capacity (s) | mean decode batch at max load |
|---|---|---|---|---|
| sequential | 0.00 | 29.60 | n/a | 1.00 |
| static-16 | 0.00 | 157.50 | n/a | 6.00 |
| continuous-16 | 4.00 | 347.13 | 0.12 | 15.88 |
| continuous-32 | 4.00 | 504.58 | 0.12 | 26.24 |
| vllm | 12.00 | 1055.95 | 0.25 | n/a |

### Closed-loop throughput

![closed loop](figures/concurrency_closed_loop.png)

| config | at concurrency | peak tok/s (mean) | std |
|---|---|---|---|
| continuous-16 | 32 | 322.30 | 7.62 |
| sequential | 4 | 29.93 | 0.04 |
| static-16 | 16 | 170.39 | 5.89 |
| vllm | 32 | 1436.60 | 17.30 |

### Where the time goes (microbenchmarks)

![microbench](figures/microbench.png)

| batch | step_ms | tok_per_s | achieved_gb_per_s | bw_utilisation |
|---|---|---|---|---|
| 1.000 | 35.328 | 28.307 | 28.058 | 0.088 |
| 2.000 | 33.126 | 60.376 | 30.018 | 0.094 |
| 4.000 | 32.908 | 121.552 | 30.408 | 0.095 |
| 8.000 | 33.651 | 237.733 | 30.110 | 0.094 |
| 16.000 | 32.843 | 487.162 | 31.617 | 0.099 |
| 32.000 | 34.451 | 928.848 | 31.602 | 0.099 |
| 64.000 | 57.644 | 1110.257 | 20.633 | 0.064 |
| 128.000 | n/a | n/a | n/a | n/a |

### Precision / quantization trade-off

![precision](figures/precision.png)

| dtype | quant | tok_s | std | gpu_mem_peak_mb | perplexity | eval_tokens | greedy_token_agreement | greedy_exact_match | param_mb |
|---|---|---|---|---|---|---|---|---|---|
| float16 | int8 | 67.417 | 4.422 | 1019.264 | 2.486 | 4991 | 0.592 | 0.417 | 601.043 |
| float16 | nf4 | 216.074 | 14.343 | 870.299 | 2.616 | 4991 | 0.233 | 0.000 | 430.418 |
| float16 | none | 289.302 | 19.462 | 1368.560 | 2.488 | 4991 | 0.977 | 0.917 | 942.293 |
| float32 | none | 244.690 | 14.937 | 2707.410 | 2.489 | 4991 | 1.000 | 1.000 | 1884.585 |
