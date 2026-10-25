# llm-inference

A small LLM serving stack **built to be measured**: a from-scratch continuous-batching engine for
Hugging Face causal LMs, an OpenAI-compatible server, an open-loop load generator, and a set of
experiments that explain *why* the numbers look the way they do (scheduling, roofline, precision).

Successor to [llm-inference-benchmark-lab](https://github.com/Shivanim042003/llm-inference-benchmark-lab),
which measured a naive CPU server. This project builds the serving system and benchmarks it
against a production engine (vLLM).

## What is in here

| Layer | What | Where |
|---|---|---|
| Engine | Iteration-level (continuous) batching on a left-padded KV cache; same code path runs sequential, static and continuous policies so comparisons isolate the scheduler | `src/llmserve/engine.py` |
| Server | OpenAI-compatible `/v1/chat/completions` with SSE streaming, `/stats`, `/engine_stats` | `src/llmserve/server.py` |
| Load generation | Open-loop Poisson arrivals (exposes queueing and tail latency) and closed-loop concurrency | `src/llmserve/loadgen.py` |
| Metrics | TTFT, TPOT, E2E percentiles, **SLO attainment and goodput**, bootstrap CIs | `src/llmserve/metrics.py` |
| Analysis | Prefill/decode microbenchmarks, achieved memory bandwidth vs roofline, `torch.profiler` tables | `src/llmserve/microbench.py` |
| Precision | fp32 / fp16 / int8 / nf4: throughput, GPU memory, perplexity, greedy-output agreement | `src/llmserve/quality.py` |
| Baseline | Same load against vLLM serving the same model | `src/llmserve/experiments.py` |
| Reports | Figures, tables and findings generated from the CSVs (no hand-written numbers) | `src/llmserve/report.py` |

## Design

```mermaid
flowchart LR
    C[load generator<br/>open / closed loop] -->|HTTP + SSE| S[FastAPI server]
    S -->|submit| Q[(pending queue)]
    Q --> E[engine thread]
    E -->|prefill 1 request| M[(HF model)]
    E -->|decode step: whole batch| M
    E -->|tokens| S
    E --> L[step log: batch size, prefill/decode time]
```

* **One batched forward pass per decode step.** Rows of different lengths share a left-padded KV
  cache `[B, H, T, D]` with an explicit attention mask and per-row position ids.
* **Admission** runs the new request's prefill separately and left-pads/concatenates its cache into
  the running batch; **finished rows** are dropped and all-padding columns trimmed.
* **Policies:** `sequential` (batch of 1), `static` (admit only when the batch has drained),
  `continuous` (admit every step, at most `max_prefill_per_step` prefills per step to bound decode stalls).
* **Correctness:** `tests/test_engine.py` checks that every policy produces exactly the tokens of an
  uncached, unbatched reference (fp64 tiny model), including EOS handling and the background thread.

## Methodology

* **Open-loop** load (Poisson arrivals) for latency/capacity: arrivals do not slow down when the
  server does, so overload shows up as queueing instead of being hidden (coordinated omission).
* **Closed-loop** load only for peak-throughput numbers, with repeats and spread.
* **Workload `mixed`:** prompts of 16-256 words and outputs of 16-256 tokens, fixed output length
  (`ignore_eos`) so every engine does identical work.
* **SLO:** TTFT <= 1 s and TPOT <= 100 ms. *SLO capacity* = highest offered rate where >= 95% of
  requests meet both and none fail. *Goodput* = rate of requests that meet the SLO.
* Requests still running 45 s after the last arrival are cancelled and counted as failed (overload).

## Results

<!-- RESULTS:START -->
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

![scheduler_open_loop](reports/figures/scheduler_open_loop.png)

![concurrency_closed_loop](reports/figures/concurrency_closed_loop.png)

![microbench](reports/figures/microbench.png)

![precision](reports/figures/precision.png)

Full tables and methodology: [reports/RESULTS.md](reports/RESULTS.md).
<!-- RESULTS:END -->

## Reproduce

```bash
pip install -e .            # needs a working torch install
make test                   # 19 CPU tests, ~20 s, no downloads
make smoke                  # full pipeline on CPU with a tiny random model
make t4                     # real experiments on a GPU (Qwen2.5-0.5B-Instruct)
```

The Colab notebook in this repo runs everything end to end and pushes results back.

## Limitations (read before quoting numbers)

* One small model (Qwen2.5-0.5B), one GPU class; results do not transfer to larger models or other GPUs.
* No paged attention and no chunked prefill: prefill is one request at a time, and left-padding wastes
  memory/compute when lengths differ. Greedy decoding only.
* The engine is Python + eager PyTorch; at this model size, per-step overhead (not memory bandwidth)
  is expected to dominate. vLLM comparisons measure that gap, they are not a claim of parity.
* The load generator runs on the same machine as the server (Colab has 2 vCPUs); at high request
  rates this can perturb measurements.
* Free-tier shared GPUs vary between sessions; compare numbers only within one run.
* The quantization experiment measures speed/memory/quality on this model only.

## License

MIT. See [LICENSE](LICENSE).
