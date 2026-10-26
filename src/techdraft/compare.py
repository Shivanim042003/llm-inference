"""Run the TechQA workload against each serving configuration and record aggregate (non-licensed) results."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from supportdraft.compare import CONFIGS

from .data import KB, load, split_of
from .run import RAW, evaluate, prepare_items, save

PROC = Path("results/processed")


def _record(name, split_name, run_label, ev_rows):
    PROC.mkdir(parents=True, exist_ok=True)
    for fname, idx in (("techdraft_serving.csv", 0), ("techdraft_quality.csv", 1)):
        new = pd.DataFrame([{"backend": name, "split": split_name, "run": run_label, "offered_rps": ev[0]["offered_rps"], **ev[idx]} for ev in ev_rows])
        path = PROC / fname
        if path.exists() and path.stat().st_size:
            old = pd.read_csv(path)
            old = old[~((old["backend"] == name) & (old["split"] == split_name) & (old["run"] == run_label))]
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(path, index=False)


def run_on_url(name, url, model, rates, items, kb, split_name, run_label, e2e_slo_s=15.0, ttft_slo_s=2.0, seed=0, min_seconds=30.0, max_items=None):
    rows = []
    for rate in rates:
        recs, serving, quality = evaluate(url, model, items, kb, rate, e2e_slo_s, ttft_slo_s, seed=seed, min_seconds=min_seconds, max_items=max_items)
        save(recs, RAW / f"{name}_{split_name}_{run_label}_rate{rate:g}.jsonl")
        rows.append((serving, quality))
        print(f"  [{name}] rate={rate:>4g}/s  n={serving['n_requests']:>4}  failed={serving['failed']}  p95 TTFT={serving['ttft_p95']:.2f}s  "
              f"p95 e2e={serving['e2e_p95']:.2f}s  on-time={100 * serving['on_time_rate']:.0f}%  pass={100 * quality.get('pass_rate', float('nan')):.0f}%  "
              f"passing&on-time={100 * serving['useful_rate']:.0f}%", flush=True)
    _record(name, split_name, run_label, rows)
    return rows


def compare_backends(prof, rates, backends=("slot-graph-32+prefix",), with_vllm=False, split_name="tune", run_label="sweep", max_items=None,
                     e2e_slo_s=15.0, ttft_slo_s=2.0, min_seconds=30.0, data_path=None, k=4):
    from llmserve.experiments import Server, _require_graphs, _slot_args, _vllm_down, _vllm_up
    model = prof["model"]
    records = load(data_path)
    kb = KB(records)
    items = prepare_items([r for r in records if split_of(r["id"]) == split_name], kb, k)
    print(f"{len(records)} records | {len(kb.docs)} documents | {len(kb.chunks)} chunks | {len(items)} items in split '{split_name}'", flush=True)
    for name in backends:
        cfg = CONFIGS[name]
        print(f"== {name} ({model})", flush=True)
        with Server(prof, f"td-{name}", "continuous", cfg["max_batch"], extra=_slot_args(cfg)) as srv:
            _require_graphs(srv, cfg)
            run_on_url(name, srv.url, model, rates, items, kb, split_name, run_label, e2e_slo_s, ttft_slo_s, min_seconds=min_seconds, max_items=max_items)
    if with_vllm:
        up = _vllm_up(prof)
        if up:
            proc, url = up
            try:
                print("== vllm", flush=True)
                run_on_url("vllm", url, model, rates, items, kb, split_name, run_label, e2e_slo_s, ttft_slo_s, min_seconds=min_seconds, max_items=max_items)
            finally:
                _vllm_down(proc)
    PROC.mkdir(parents=True, exist_ok=True)
    (PROC / f"techdraft_setup_{split_name}_{run_label}.json").write_text(json.dumps(dict(
        model=model, rates=list(rates), split=split_name, run=run_label, max_items=max_items, e2e_slo_s=e2e_slo_s, ttft_slo_s=ttft_slo_s,
        k_chunks=k, backends=list(backends) + (["vllm"] if with_vllm else [])), indent=2))
