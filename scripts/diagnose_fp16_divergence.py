"""Characterise the fp16 disagreement between the slot engine and the reference dynamic-cache engine.

Teacher-forces the shared prefix of each diverging prompt through an fp32 copy of the model and reports the fp32 logit
margin between the two chosen tokens, the typical low-precision logit error, and which engine agrees with fp32.
Evidence, not proof.

    python scripts/diagnose_fp16_divergence.py --quick
    python scripts/diagnose_fp16_divergence.py
"""
import argparse
import gc
import statistics as st

import torch

from llmserve.engine import Engine, GenRequest
from llmserve.models import load_model
from llmserve.slot_engine import SlotEngine

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=None)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--low-dtype", default=None, choices=["float16", "bfloat16"])
a = ap.parse_args()
CUDA = torch.cuda.is_available()
LOW = a.low_dtype or ("float16" if CUDA else "bfloat16")
models = a.models or (["Qwen/Qwen2.5-0.5B-Instruct"] if a.quick else ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"])
N_PROMPTS, NEW = (12, 48) if a.quick else (36, 96)
KMAX = 40 if CUDA else 6
TOPICS = ["the T4 GPU", "continuous batching", "memory bandwidth", "KV caches", "a bakery", "the French revolution"]
SHARED = "Background notes for a systems team. " + "Latency matters more than throughput for interactive use. " * (12 if CUDA else 2)


def build_prompts(tok):      # same prompt set as verify_gpu_equivalence.py
    out = []
    for i in range(N_PROMPTS):
        if i % 4 == 3:
            text = SHARED + f"Question {i}: summarise {TOPICS[i % 6]} in two sentences."
        else:
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
    out = [r.out_ids for r in reqs]
    del eng
    gc.collect()
    if CUDA:
        torch.cuda.empty_cache()
    return out


@torch.no_grad()
def last_logits(model, dev, ids):
    x = torch.tensor([ids], device=dev)
    return model(input_ids=x).logits[0, -1].float().cpu()


total_div, total_prompts = 0, 0
for model_id in models:
    low, tok, eos, dev = load_model(model_id, dtype=LOW)
    prompts = build_prompts(tok)
    kw = dict(eos_ids=eos, device=dev, max_batch=8, max_ctx=2048)
    dyn = serve(low, eos, dev, prompts, lambda: Engine(low, **kw))
    slot = serve(low, eos, dev, prompts, lambda: SlotEngine(low, **kw))
    ref, _, _, rdev = load_model(model_id, dtype="float32")
    rows = []
    for i, (p, d, s) in enumerate(zip(prompts, dyn, slot)):
        j = next((k for k, (u, v) in enumerate(zip(d, s)) if u != v), None)
        if j is None:
            continue
        ctx = list(p) + list(d[:j])
        l32 = last_logits(ref, rdev, ctx)
        llow = last_logits(low, dev, ctx)
        a_tok, b_tok = d[j], s[j]
        top32 = int(l32.argmax())
        top = l32.topk(20).indices
        err = (llow[top] - l32[top]).abs()
        rows.append(dict(i=i, plen=len(p), j=j, dyn=a_tok, slot=b_tok, margin32=float(l32[a_tok] - l32[b_tok]),
                         marginlow=float(llow[a_tok] - llow[b_tok]), err_med=float(err.median()), err_max=float(err.max()),
                         dyn_ok=a_tok == top32, slot_ok=b_tok == top32))
    total_div += len(rows)
    total_prompts += len(prompts)
    print(f"\n=== {model_id} ({LOW} engines vs fp32 reference logits): {len(rows)}/{len(prompts)} prompts diverge ===")
    if rows:
        print(f"  {'prompt':>6} {'plen':>5} {'first diff':>10} {'fp32 margin':>12} {f'{LOW} margin':>12} {'low-prec err med/max':>20}  fp32 argmax agrees with")
        for r in rows:
            who = ("both?" if r["dyn_ok"] and r["slot_ok"] else "dynamic" if r["dyn_ok"] else "slot" if r["slot_ok"] else "neither")
            print(f"  {r['i']:>6} {r['plen']:>5} {r['j']:>10} {r['margin32']:>+12.3f} {r['marginlow']:>+12.3f} "
                  f"{r['err_med']:>8.3f}/{r['err_max']:<7.3f}  {who}")
        marg = [abs(r["margin32"]) for r in rows]
        noise = [r["err_max"] for r in rows]
        inside = sum(m <= n for m, n in zip(marg, noise))
        print(f"  median |fp32 margin| {st.median(marg):.3f}  vs median max {LOW} logit error {st.median(noise):.3f}; "
              f"{inside}/{len(rows)} divergences have a margin inside the {LOW} error")
        print(f"  fp32 argmax matches: dynamic {sum(r['dyn_ok'] for r in rows)}, slot {sum(r['slot_ok'] for r in rows)}, "
              f"neither {sum((not r['dyn_ok']) and (not r['slot_ok']) for r in rows)}")
    del low, ref
    gc.collect()
    if CUDA:
        torch.cuda.empty_cache()

print(f"\nTotal: {total_div}/{total_prompts} prompts diverge between the two {LOW} engines.")
