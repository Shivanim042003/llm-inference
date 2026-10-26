# vLLM benchmark runs on the SupportDraft test split: configuration history

Two vLLM runs exist. **Run 2 is authoritative.** Run 1 is kept for transparency and is not used in the report tables.

| | Run 1: `vllm-run1-default-generation-config` | Run 2 (authoritative): `vllm-run2-vllm-generation-config` |
|---|---|---|
| vLLM | 0.32.0 | 0.32.0 |
| model / dtype | Qwen/Qwen2.5-1.5B-Instruct / half | same |
| max-model-len / max-num-seqs | 4096 / 16 | same |
| gpu-memory-utilization | 0.5 | 0.5 |
| generation config | **model default** (`--generation-config` not set, vLLM loads the model's `generation_config.json`) | **`--generation-config vllm`** (vLLM defaults; request sets temperature 0) |
| split / rates | test / [0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24] | same |
| prompt | final (e8c0b3569a) | same |
| passing-and-on-time rate | 31% to 35% | 46% to 49% |
| where | commit `57f0305`, archived in `results/supportdraft/archive/vllm_run1_default_generation_config/` | current `results/` files and the report tables |

## Why run 2 is authoritative
The other three engines decode greedily with no extra sampling settings. Run 1 let vLLM apply the model's own default generation settings, so the decoding differed from the other engines and its quality numbers were not comparable. Run 2 removes that difference, and correctness returned to the same 44% to 49% range as the other engines.

## What is established and what is inferred
Established: adding `--generation-config vllm` changed correctness from about 33% to about 35% up to 46% to 49%, with the same model, prompt and ticket set. Inferred, not directly logged: that the cause is the model's default sampling settings (for Qwen2.5 these include a repetition penalty, which works against drafts that repeat `[Pxx]` citations). The applied defaults were not read from the server logs.

Serving latency and capacity in run 1 were similar in shape but the run 1 timings are not used. Run 1 also did not isolate whether output length differed.
