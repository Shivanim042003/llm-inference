"""Engine-level microbenchmarks (no HTTP): prefill and decode cost versus shape.

Answers "where does the time go?": prefill scales with prompt length (compute-bound),
decode step time is nearly flat in batch size until the GPU is saturated (memory-bound),
which is exactly why batching raises throughput.
"""
from __future__ import annotations

import time

import pandas as pd
import torch

from .engine import cache_to_tuples, forward_last_logits
from .models import load_model, model_footprint


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def _timed(fn, device, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    _sync(device)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


@torch.no_grad()
def prefill_curve(model, device, vocab, lengths, batch=1):
    """Prefill cost with full-vocab logits at every position vs only the last position."""
    rows = []
    for L in lengths:
        ids = torch.randint(3, vocab, (batch, L), device=device)
        for variant, fn in (("full_logits", lambda: model(input_ids=ids, use_cache=True)),
                            ("last_logits", lambda: forward_last_logits(model, ids, use_cache=True))):
            try:
                sec = _timed(fn, device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            rows.append({"variant": variant, "batch": batch, "prompt_len": L, "seconds": sec,
                         "tok_per_s": batch * L / sec})
    return pd.DataFrame(rows)


@torch.no_grad()
def decode_curve(model, device, vocab, batches, ctx=256, steps=8):
    """Time one decode step (all rows attend to `ctx` cached tokens) for each batch size."""
    from .engine import tuples_to_cache

    rows = []
    for b in batches:
        try:
            ids = torch.randint(3, vocab, (b, ctx), device=device)
            out = forward_last_logits(model, ids, use_cache=True)   # avoids a [B, ctx, vocab] logits tensor
            base = [(k.clone(), v.clone()) for k, v in cache_to_tuples(out.past_key_values)]
            del out
            nxt = torch.randint(3, vocab, (b, 1), device=device)
            mask = torch.ones(b, ctx + 1, dtype=torch.long, device=device)
            pos = torch.full((b, 1), ctx, device=device)

            def step():
                cache = tuples_to_cache(base)
                model(input_ids=nxt, attention_mask=mask, position_ids=pos,
                      past_key_values=cache, use_cache=True)

            secs = [_timed(step, device, warmup=3, iters=steps) for _ in range(5)]   # best of 5 medians
            sec = min(secs)
            rows.append({"batch": b, "ctx": ctx, "step_ms": sec * 1e3, "tok_per_s": b / sec,
                         "step_ms_worst": max(secs) * 1e3})
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows.append({"batch": b, "ctx": ctx, "step_ms": float("nan"), "tok_per_s": float("nan")})
            break
    return pd.DataFrame(rows)


def profile_decode(model, device, vocab, batch, ctx=256, top=12):
    """torch.profiler table (top ops by self time) for one decode step."""
    from torch.profiler import ProfilerActivity, profile

    from .engine import tuples_to_cache

    ids = torch.randint(3, vocab, (batch, ctx), device=device)
    with torch.no_grad():
        out = forward_last_logits(model, ids, use_cache=True)
        base = [(k.clone(), v.clone()) for k, v in cache_to_tuples(out.past_key_values)]
        nxt = torch.randint(3, vocab, (batch, 1), device=device)
        mask = torch.ones(batch, ctx + 1, dtype=torch.long, device=device)
        pos = torch.full((batch, 1), ctx, device=device)
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda" else [])
        model(input_ids=nxt, attention_mask=mask, position_ids=pos,
              past_key_values=tuples_to_cache(base), use_cache=True)  # warm
        _sync(device)
        with profile(activities=acts) as prof:
            model(input_ids=nxt, attention_mask=mask, position_ids=pos,
                  past_key_values=tuples_to_cache(base), use_cache=True)
            _sync(device)
    key = "self_cuda_time_total" if device == "cuda" else "self_cpu_time_total"
    try:
        return prof.key_averages().table(sort_by=key, row_limit=top)
    except Exception:
        return prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=top)


def run(model_id, dtype, outdir, tiny_scale=False, peak_bw_gbs=320.0):
    """Run all microbenchmarks; write CSVs (+ profiler text) under outdir. T4 peak BW ~320 GB/s."""
    from pathlib import Path

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    model, _tok, _eos, device = load_model(model_id, dtype=dtype)
    vocab = model.config.vocab_size
    fp = model_footprint(model)
    lengths = [16, 64, 256, 1024] if tiny_scale else [16, 64, 128, 256, 512, 1024, 2048]
    batches = [1, 2, 4, 8] if tiny_scale else [1, 2, 4, 8, 16, 32, 64, 128]

    pre = prefill_curve(model, device, vocab, lengths)
    dec = decode_curve(model, device, vocab, batches, ctx=128 if tiny_scale else 256)
    dec["weights_gb_per_s"] = fp["param_bytes"] / (dec["step_ms"] / 1e3) / 1e9
    dec["kv_gb_per_s"] = dec["batch"] * dec["ctx"] * fp["kv_bytes_per_token"] / (dec["step_ms"] / 1e3) / 1e9
    dec["achieved_gb_per_s"] = dec["weights_gb_per_s"] + dec["kv_gb_per_s"]
    dec["peak_bw_gbs"] = peak_bw_gbs
    dec["bw_utilisation"] = dec["achieved_gb_per_s"] / peak_bw_gbs
    meta = {"model": model_id, "dtype": dtype, "device": device, **fp}
    if device == "cuda":
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["gpu_mem_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
    pre.to_csv(out / "microbench_prefill.csv", index=False)
    dec.to_csv(out / "microbench_decode.csv", index=False)
    pd.Series(meta).to_json(out / "microbench_meta.json")
    prof_txt = []
    for b in ([1, 8] if tiny_scale else [1, 32]):
        prof_txt.append(f"=== decode step, batch={b}, ctx=256 ===\n{profile_decode(model, device, vocab, b)}\n")
    (out / "profile_decode.txt").write_text("\n".join(prof_txt))
    return pre, dec, meta


@torch.no_grad()
def decode_graph_curve(model, device, batches=(1, 2, 4, 8, 16, 32), ctx=256, iters=30, max_ctx=1024):
    """Decode-step time of the SlotEngine kernel path: HF dynamic cache vs slot eager vs slot + CUDA graph.

    All three do the same math for the same shape (every row attends to `ctx` cached tokens).
    """
    from .slot_engine import SlotEngine

    rows = []
    vocab = model.config.vocab_size
    dec = decode_curve(model, device, vocab, list(batches), ctx=ctx)
    for _, r in dec.iterrows():
        rows.append({"batch": int(r["batch"]), "variant": "hf-dynamic-cache (eager)", "step_ms": r["step_ms"]})
    S = max(batches)
    for graphs in (False, True):
        if graphs and device != "cuda":
            continue
        eng = SlotEngine(model, device=device, max_batch=S, max_ctx=max_ctx, cuda_graphs=graphs,
                         ctx_buckets=[ctx + 1, max_ctx], log_steps=False)
        if graphs and not eng.cuda_graphs:
            rows.append({"batch": 0, "variant": f"slot + cuda graph (FAILED: {eng.graph_error})", "step_ms": float("nan")})
            continue
        T = eng._bucket(ctx + 1, eng.ctx_buckets)
        for b in batches:
            Sb = eng._bucket(b, eng.batch_buckets)
            eng.tok_buf[:] = torch.randint(3, vocab, (S,), device=device)
            eng.pos_buf[:] = ctx
            if graphs:
                g, _ = eng.graphs[(Sb, T)]
                fn = g.replay
            else:
                fn = lambda: eng._decode_core(Sb, T)
            sec = _timed(fn, device, warmup=5, iters=iters)
            rows.append({"batch": b, "variant": "slot + cuda graph" if graphs else "slot eager (static KV)",
                         "step_ms": sec * 1e3})
        del eng
        if device == "cuda":
            torch.cuda.empty_cache()
    out = pd.DataFrame(rows)
    out["tok_per_s"] = out["batch"] / (out["step_ms"] / 1e3)
    return out


def run_graph_microbench(model_id, dtype, outdir, tiny_scale=False):
    from pathlib import Path

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    model, _tok, _eos, device = load_model(model_id, dtype=dtype)
    batches = (1, 2, 4) if tiny_scale else (1, 2, 4, 8, 16, 32)
    df = decode_graph_curve(model, device, batches, ctx=64 if tiny_scale else 256,
                            max_ctx=256 if tiny_scale else 1024, iters=5 if tiny_scale else 30)
    df.to_csv(out / "decode_graphs.csv", index=False)
    return df
