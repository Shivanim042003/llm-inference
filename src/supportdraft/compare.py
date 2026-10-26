"""Run the same ticket spike against several serving backends and collect quality, latency and cost rows."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .runner import evaluate, save
from .tickets import split

OUT = Path("results/supportdraft")
PROC = Path("results/processed")

# serving configurations that can be started from this repo (see llmserve.experiments._slot_args)
CONFIGS = {
    "dynamic-32": dict(engine="dynamic", max_batch=32),
    "slot-graph-32": dict(engine="slot", graphs=True, max_batch=32),
    "slot-graph-32+prefix": dict(engine="slot", graphs=True, prefix=True, max_batch=32),
}


def _record(name, split_name, ev_rows, tag):
    PROC.mkdir(parents=True, exist_ok=True)
    for fname, rows in (("supportdraft_serving.csv", [r[0] for r in ev_rows]), ("supportdraft_quality.csv", [r[1] for r in ev_rows])):
        new = pd.DataFrame([{"backend": name, "split": split_name, "offered_rps": ev[0]["offered_rps"], **r}
                            for r, ev in zip(rows, ev_rows)])
        path = PROC / fname
        if path.exists() and path.stat().st_size:
            old = pd.read_csv(path)
            old = old[~((old["backend"] == name) & (old["split"] == split_name))]
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(path, index=False)


def run_on_url(name, url, model, rates, split_name="test", e2e_slo_s=15.0, ttft_slo_s=2.0, seed=0, min_seconds=30.0):
    tickets = split(split_name) if split_name != "all" else split("dev") + split("test")
    rows = []
    for rate in rates:
        recs, serving, quality = evaluate(url, model, tickets, rate, e2e_slo_s, ttft_slo_s, seed=seed, min_seconds=min_seconds)
        save(recs, OUT / f"{name}_{split_name}_rate{rate:g}.jsonl")
        rows.append((serving, quality))
        print(f"  [{name}] rate={rate:>4g}/s  n={serving['n_requests']:>4}  failed={serving['failed']}  "
              f"p95 TTFT={serving['ttft_p95']:.2f}s  p95 e2e={serving['e2e_p95']:.2f}s  on-time={100*serving['on_time_rate']:.0f}%  "
              f"correct={100*quality.get('correct_rate', float('nan')):.0f}%  useful={100*serving['useful_rate']:.0f}%", flush=True)
    _record(name, split_name, rows, name)
    return rows


def compare_backends(prof, rates, backends=("dynamic-32", "slot-graph-32", "slot-graph-32+prefix"), with_vllm=True,
                     split_name="test", e2e_slo_s=15.0, ttft_slo_s=2.0, min_seconds=30.0):
    """Start each backend in turn on prof['model'] and run the spike. vLLM is best effort (separate venv, see the install cell)."""
    from llmserve.experiments import Server, _require_graphs, _slot_args, _vllm_down, _vllm_up
    model = prof["model"]
    for name in backends:
        cfg = CONFIGS[name]
        print(f"== {name} ({model})", flush=True)
        with Server(prof, f"sd-{name}", "continuous", cfg["max_batch"], extra=_slot_args(cfg)) as srv:
            _require_graphs(srv, cfg)
            run_on_url(name, srv.url, model, rates, split_name, e2e_slo_s, ttft_slo_s, min_seconds=min_seconds)
    if with_vllm:
        up = _vllm_up(prof)
        if up:
            proc, url = up
            try:
                print("== vllm", flush=True)
                run_on_url("vllm", url, model, rates, split_name, e2e_slo_s, ttft_slo_s, min_seconds=min_seconds)
            finally:
                _vllm_down(proc)
    (PROC / f"supportdraft_setup_{split_name}.json").write_text(json.dumps(dict(model=model, rates=list(rates), split=split_name, e2e_slo_s=e2e_slo_s,
                                                                  ttft_slo_s=ttft_slo_s, backends=list(backends) + (["vllm"] if with_vllm else [])), indent=2))
