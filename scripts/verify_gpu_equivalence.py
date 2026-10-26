"""Real-model GPU check: dynamic-cache, static-slot (eager) and static-slot + CUDA-graph engines must produce the
same greedy tokens. A spot check on a small prompt set, not a proof (fp16 kernels can differ on other inputs)."""
import gc
import sys

import torch

from llmserve.engine import Engine, GenRequest
from llmserve.models import load_model
from llmserve.slot_engine import SlotEngine

model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
assert torch.cuda.is_available(), "needs a GPU"
model, tok, eos, dev = load_model(model_id, dtype="float16")
topics = ["the T4 GPU", "continuous batching", "memory bandwidth", "KV caches", "a bakery", "the French revolution"]
prompts = [tok.encode_chat([{"role": "user", "content": ("Write a short note about " + t + ". ") * k}])
           for k, t in zip([1, 3, 8, 1, 5, 2], topics)] * 2


def serve(make):
    eng = make()
    reqs = [GenRequest(prompt_ids=list(p), max_new_tokens=48, ignore_eos=True) for p in prompts]
    for r in reqs:
        eng.submit(r)
    eng.run_until_idle()
    out, info = [r.out_ids for r in reqs], eng.stats()
    del eng
    gc.collect(); torch.cuda.empty_cache()
    return out, info


def agree(a, b):
    exact = sum(x == y for x, y in zip(a, b)) / len(a)
    pref = sum((next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), len(x))) / len(x) for x, y in zip(a, b)) / len(a)
    return exact, pref


dyn, _ = serve(lambda: Engine(model, eos_ids=eos, device=dev, max_batch=8, max_ctx=1024))
eag, _ = serve(lambda: SlotEngine(model, eos_ids=eos, device=dev, max_batch=8, max_ctx=1024))
grf, info = serve(lambda: SlotEngine(model, eos_ids=eos, device=dev, max_batch=8, max_ctx=1024, cuda_graphs=True))
print(f"model={model_id} prompts={len(prompts)} graphs_active={info['cuda_graphs']} captured={info['n_graphs']} error={info['graph_error']}")
assert info["cuda_graphs"], f"CUDA graphs did not activate: {info['graph_error']}"
for name, (a, b) in {"slot+graphs vs slot eager": (grf, eag), "slot eager vs dynamic cache": (eag, dyn),
                     "slot+graphs vs dynamic cache": (grf, dyn)}.items():
    exact, pref = agree(a, b)
    print(f"{name:30s}: exact {100*exact:.0f}%  mean matching prefix {100*pref:.0f}%")
    assert pref >= 0.9, f"{name}: outputs diverge early ({100*pref:.0f}% matching prefix)"
print("OK")
