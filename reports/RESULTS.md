# Results

**Setup:** model `Qwen/Qwen2.5-0.5B-Instruct`, float16, hardware: Tesla T4, torch 2.11.0+cu130, transformers 5.20.2, run date 2026-10-25 18:22:17 UTC, commit `f7ef35f`.

### Key findings (auto-generated from the data)

- Sequential serving never met the SLO at any tested rate, while continuous-16 sustained 4 req/s, continuous-32 sustained 6 req/s, vllm sustained 12 req/s.
- At 4 req/s (the highest rate where continuous-16 meets the SLO), continuous batching has 51x lower p95 TTFT than static batching of the same size (0.66s vs 33.56s). Static batching meets the SLO at no tested rate.
- Against vLLM on the same GPU, our best engine (continuous-32, SLO capacity 6 req/s) reaches 50% of vLLM's SLO capacity (12 req/s).
- Closed-loop peak: **continuous-16** 332 tok/s = 11.2x sequential (30 tok/s).
- Closed-loop peak: **continuous-32** 509 tok/s = 17.2x sequential (30 tok/s).
- Closed-loop peak: **static-16** 176 tok/s = 5.9x sequential (30 tok/s).
- Closed-loop peak: **vllm** 1413 tok/s = 47.6x sequential (30 tok/s).
- Decode step time is nearly flat in batch size: within 1.5x of batch 1 (34.09 ms) up to batch **32**, where decode tokens/s is 31.4x batch 1. This is why batching raises throughput.
- At batch 128 a decode step takes 104.29 ms, 4.2% of the assumed 320 GB/s peak bandwidth: well below the memory roofline, so the step is not bandwidth-bound; kernel-launch / Python overhead is the likely limiter (check profile_decode.txt).
- Prefill of a 2048-token prompt takes 419.1 ms when only the last position's logits are computed vs 434.0 ms with full-vocab logits at every position (1.0x).
- Computing the lm_head only for the last prompt position does not measurably change closed-loop throughput (291 -> 288 tok/s, 0.99x; run-to-run spread is not shown, so treat differences under ~5% as noise) and moves the share of GPU time spent in prefill from 12% to 12%.
- float16/int8: 0.27x throughput vs fp32, perplexity 22.455 vs 22.426, greedy outputs match fp32 for the first 58% of tokens on average.
- float16/nf4: 0.84x throughput vs fp32, perplexity 25.025 vs 22.426, greedy outputs match fp32 for the first 17% of tokens on average.
- float16/none: 1.15x throughput vs fp32, perplexity 22.417 vs 22.426, greedy outputs match fp32 for the first 98% of tokens on average.

### Scheduler comparison (open-loop Poisson arrivals)

![scheduler](figures/scheduler_open_loop.png)

| config | SLO capacity (req/s) | peak output tok/s | TTFT p95 at capacity (s) | mean decode batch at max load |
|---|---|---|---|---|
| sequential | 0 (SLO never met) | 29.18 | n/a | 1.00 |
| static-16 | 0 (SLO never met) | 134.71 | n/a | 5.15 |
| continuous-16 | 4 (true value in [4, 5)) | 350.33 | 0.66 | 15.92 |
| continuous-32 | 6 (true value in [6, 7)) | 531.73 | 0.21 | 27.48 |
| vllm | 12 (true value in [12, 16)) | 1419.89 | 0.33 | n/a |

### Closed-loop throughput

![closed loop](figures/concurrency_closed_loop.png)

| config | at concurrency | peak tok/s (mean) | std |
|---|---|---|---|
| continuous-16 | 32 | 332.18 | 11.50 |
| continuous-32 | 64 | 509.38 | 3.89 |
| sequential | 1 | 29.67 | 0.10 |
| static-16 | 16 | 175.69 | 7.62 |
| vllm | 32 | 1412.97 | 6.34 |

### Where the time goes (microbenchmarks)

![microbench](figures/microbench.png)

| batch | step_ms | tok_per_s | achieved_gb_per_s | bw_utilisation |
|---|---|---|---|---|
| 1.000 | 34.093 | 29.332 | 29.074 | 0.091 |
| 2.000 | 34.307 | 58.297 | 28.984 | 0.091 |
| 4.000 | 33.422 | 119.681 | 29.940 | 0.094 |
| 8.000 | 34.214 | 233.824 | 29.615 | 0.093 |
| 16.000 | 34.825 | 459.434 | 29.817 | 0.093 |
| 32.000 | 34.771 | 920.313 | 31.312 | 0.098 |
| 64.000 | 57.410 | 1114.789 | 20.718 | 0.065 |
| 128.000 | 104.291 | 1227.339 | 13.335 | 0.042 |

### Optimisation ablation: last-position logits in prefill

![ablation](figures/ablation_logits.png)

| variant | closed-loop tok/s | prefill time share |
|---|---|---|
| full-logits | 290.879 | 0.120 |
| last-logits | 288.350 | 0.119 |

### Precision / quantization trade-off

![precision](figures/precision.png)

| dtype | quant | tok_s | std | gpu_mem_peak_mb | perplexity | eval_tokens | greedy_token_agreement | greedy_prefix_match | greedy_exact_match | param_mb |
|---|---|---|---|---|---|---|---|---|---|---|
| float16 | int8 | 66.505 | 3.515 | 930.311 | 22.455 | 8176 | 0.592 | 0.585 | 0.417 | 601.043 |
| float16 | nf4 | 206.393 | 15.034 | 780.647 | 25.025 | 8176 | 0.233 | 0.168 | 0.000 | 430.418 |
| float16 | none | 284.910 | 18.466 | 1279.252 | 22.417 | 8176 | 0.977 | 0.977 | 0.917 | 942.293 |
| float32 | none | 246.922 | 13.681 | 2526.404 | 22.426 | 8176 | 1.000 | 1.000 | 1.000 | 1884.585 |
