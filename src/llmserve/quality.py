"""Quality side of the precision trade-off: perplexity and greedy-output agreement vs fp32."""
from __future__ import annotations

import math

import pandas as pd
import torch

FALLBACK_TEXT = (
    "The history of computing is a story of abstraction. Early machines were programmed by "
    "rewiring circuits, then by punching cards, then by writing assembly. Each layer hid the "
    "details of the one below it so that people could think about problems instead of "
    "plumbing. Operating systems virtualised hardware, compilers virtualised instruction sets, "
    "and cloud platforms virtualised entire data centres. Machine learning added a new layer: "
    "instead of writing rules, engineers curate data and objectives, and the system derives "
    "behaviour from examples. Serving such systems efficiently is its own discipline, because "
    "a model that is accurate but too slow or too expensive to run cannot be deployed. "
) * 40

PROMPTS = [
    "Explain what a hash table is in two sentences.",
    "List three differences between TCP and UDP.",
    "Write a haiku about debugging.",
    "What is the capital of France and why is it historically important?",
    "Describe how a binary search works.",
    "Give one reason to use a queue instead of a stack.",
    "Summarize the idea of gradient descent briefly.",
    "What does the acronym GPU stand for and what is it used for?",
    "Translate 'good morning' into Spanish and French.",
    "Explain why the sky is blue in simple terms.",
    "What is the difference between latency and throughput?",
    "Name two benefits of unit testing.",
]


def load_eval_text(max_chars=60000):
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return "\n\n".join(ds["text"])[:max_chars], "wikitext-2-raw-v1/test"
    except Exception:
        return FALLBACK_TEXT[:max_chars], "built-in fallback paragraph"


@torch.no_grad()
def perplexity(model, tok, text, device, window=512, max_windows=16):
    ids = tok.tok(text, return_tensors="pt").input_ids[0]
    nll, count = 0.0, 0
    for w in range(max_windows):
        chunk = ids[w * window:(w + 1) * window]
        if len(chunk) < 8:
            break
        x = chunk.unsqueeze(0).to(device)
        logits = model(input_ids=x).logits[0, :-1].float()
        loss = torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="sum")
        nll += float(loss)
        count += x.shape[1] - 1
    return math.exp(nll / max(count, 1)), count


@torch.no_grad()
def greedy_outputs(model, tok, device, n_new=48):
    outs = []
    for p in PROMPTS:
        ids = torch.tensor([tok.encode_chat([{"role": "user", "content": p}])], device=device)
        gen = model.generate(ids, max_new_tokens=n_new, do_sample=False,
                             pad_token_id=tok.tok.eos_token_id)
        outs.append(gen[0, ids.shape[1]:].tolist())
    return outs


def agreement(ref, other):
    """Mean fraction of positions where greedy tokens match, and fraction of fully identical outputs."""
    pos, exact = [], 0
    for a, b in zip(ref, other):
        n = min(len(a), len(b))
        pos.append(sum(x == y for x, y in zip(a[:n], b[:n])) / max(max(len(a), len(b)), 1))
        exact += a == b
    return sum(pos) / len(pos), exact / len(ref)


def run(model_id, configs, outpath):
    """configs: list of (dtype, quant). First config is the reference."""
    from .models import load_model, model_footprint

    text, source = load_eval_text()
    rows, ref_out = [], None
    for dtype, quant in configs:
        try:
            model, tok, _eos, device = load_model(model_id, dtype=dtype, quant=quant)
            ppl, ntok = perplexity(model, tok, text, device)
            outs = greedy_outputs(model, tok, device)
            if ref_out is None:
                ref_out = outs
            tok_agree, exact = agreement(ref_out, outs)
            fp = model_footprint(model)
            rows.append({"dtype": dtype, "quant": quant, "perplexity": ppl, "eval_tokens": ntok,
                         "eval_text": source, "greedy_token_agreement": tok_agree,
                         "greedy_exact_match": exact, "param_mb": fp["param_bytes"] / 2**20})
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # e.g. bitsandbytes missing
            rows.append({"dtype": dtype, "quant": quant, "error": f"{type(exc).__name__}: {exc}"[:200]})
    df = pd.DataFrame(rows)
    df.to_csv(outpath, index=False)
    return df
