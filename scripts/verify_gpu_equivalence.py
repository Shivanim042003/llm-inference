"""Real-model GPU check: do the engines produce the same greedy tokens?

Compares, per model, the reference dynamic-cache engine against the static-slot engine in three configurations
(eager; CUDA graphs; CUDA graphs + chunked prefill + prefix cache) on prompts of widely varying length, several of
which share a long prefix. This supports "equivalent on the tested configurations"; it is not a proof of numerical
equivalence in general (fp16 kernels can legitimately differ on other inputs).

    python scripts/verify_gpu_equivalence.py                        # both Qwen sizes, 36 prompts x 96 tokens (~6 min)
    python scripts/verify_gpu_equivalence.py --quick                # 0.5B only, 12 prompts x 48 tokens (~1 min)
    python scripts/verify_gpu_equivalence.py --models tiny-random   # CPU self-test of this script's logic
"""
import argparse
import gc

import torch

from llmserve.engine import Engine, GenRequest
from llmserve.models import load_model
from llmserve.slot_engine import SlotEngine

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=None)
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()
CUDA = torch.cuda.is_available()
models = a.models or (["Qwen/Qwen2.5-0.5B-Instruct"] if a.quick else ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"])
N_PROMPTS, NEW = (12, 48) if a.quick else (36, 96)
KMAX = 40 if CUDA else 6
TOPICS = ["the T4 GPU", "continuous batching", "memory bandwidth", "KV caches", "a bakery", "the French revolution"]
SHARED = "Background notes for a systems team. " + "Latency matters more than throughput for interactive use. " * (12 if CUDA else 2)


def build_prompts(tok):
    out = []
    for i in range(N_PROMPTS):
        if i % 4 == 3:        # every 4th prompt shares a long common prefix (exercises the prefix cache)
            text = SHARED + f"Question {i}: summarise {TOPICS[i % 6]} in two sentences."
        else:                 # varied lengths from ~20 to ~400 tokens
            k = 1 + (i * 7) % KMAX
            text = ("Write a short note about " + TOPICS[i % 6] + ". ") * k
        out.append(tok.encode_chat([{"role": "user", "content": text}]))
    return out


def serve(model, eos, dev, prompts, make):
    eng = make()
    reqs = [GenRequest(prompt_ids=list(p), max_new_tokens=NEW, ignore_eos=True) for p in prompts]
    for r in reqs:
        eng.submit(r)
    eng.run_until_idle()
    out, info = [r.out_ids for r in reqs], eng.stats()
    del eng
    gc.collect()
    if CUDA:
        torch.cuda.empty_cache()
    return out, info


def agree(x, y):
    exact = sum(p == q for p, q in zip(x, y)) / len(x)
    pref = [(next((i for i, (u, v) in enumerate(zip(p, q)) if u != v), len(p))) / len(p) for p, q in zip(x, y)]
    return exact, sum(pref) / len(pref), min(pref)


failures = []
for model_id in models:
    model, tok, eos, dev = load_model(model_id, dtype="float16" if CUDA else "float32")
    prompts = build_prompts(tok)
    lens = [len(p) for p in prompts]
    print(f"\n=== {model_id}: {len(prompts)} prompts, {min(lens)}-{max(lens)} prompt tokens, {NEW} new tokens each ===")
    kw = dict(eos_ids=eos, device=dev, max_batch=8, max_ctx=2048)
    dyn, _ = serve(model, eos, dev, prompts, lambda: Engine(model, **kw))
    runs = {}
    for name, extra in {"slot eager": {}, "slot + graphs": dict(cuda_graphs=True),
                        "slot + graphs + chunk64 + prefix": dict(cuda_graphs=True, chunk_size=64, prefix_cache=True)}.items():
        out, info = serve(model, eos, dev, prompts, lambda: SlotEngine(model, **kw, **extra))
        runs[name] = out
        if CUDA and extra.get("cuda_graphs") and not info["cuda_graphs"]:
            failures.append(f"{model_id} / {name}: CUDA graphs did not activate ({info['graph_error']})")
        hit = info.get("prefix_token_hit_rate")
        ex, mp, worst = agree(out, dyn)
        print(f"  {name:36s} vs dynamic cache: exact {100*ex:3.0f}%  mean matching prefix {100*mp:3.0f}%  worst prompt {100*worst:3.0f}%"
              + (f"  (prefix hit rate {100*hit:.0f}%)" if hit is not None else ""))
        if mp < 0.85:
            failures.append(f"{model_id} / {name}: diverges from the dynamic-cache engine (mean matching prefix {100*mp:.0f}%)")
    ex, mp, worst = agree(runs["slot + graphs"], runs["slot eager"])
    print(f"  {'slot + graphs vs slot eager':36s}: exact {100*ex:3.0f}%  mean matching prefix {100*mp:3.0f}%  worst prompt {100*worst:3.0f}%")
    if mp < 0.98:    # same math, same shapes: graphs must not change outputs
        failures.append(f"{model_id}: CUDA graphs change outputs vs eager slot engine (mean matching prefix {100*mp:.0f}%)")
    del model
    gc.collect()
    if CUDA:
        torch.cuda.empty_cache()

if failures:
    raise SystemExit("FAILED:\n  " + "\n  ".join(failures))
print("\nOK")
