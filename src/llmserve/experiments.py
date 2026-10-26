"""Experiment driver. Every number in the reports comes from here.

    python -m llmserve.experiments --profile t4 --only env scheduler
    python -m llmserve.experiments --profile smoke --only env scheduler concurrency   # CPU, seconds
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

from .loadgen import run_closed_loop, run_open_loop, warmup
from .models import TINY_ID

ROOT = Path(__file__).resolve().parents[2]
SRC = str(ROOT / "src")
RAW = ROOT / "results" / "raw"
PROC = ROOT / "results" / "processed"
LOGS = ROOT / "results" / "logs"

PROFILES = {
    "smoke": dict(
        model=TINY_ID, dtype="float32", peak_bw_gbs=20.0, tiny=True, slo_ttft=2.0, slo_tpot=0.2,
        window_s=3, drain_s=15, min_req=6,
        sched_configs={"sequential": dict(mode="sequential", rates=[2, 8]),
                       "static-8": dict(mode="static", max_batch=8, rates=[2, 8]),
                       "continuous-8": dict(mode="continuous", max_batch=8, rates=[2, 8])},
        conc_configs={"sequential": dict(mode="sequential", levels=[1, 4]),
                      "continuous-8": dict(mode="continuous", max_batch=8, levels=[1, 4, 8])},
        conc_reps=2, workload="mixed",
        conc_extra={"continuous-16": dict(mode="continuous", max_batch=16, levels=[2])},
        refine_rates={"continuous-8": [4]}, vllm_extra_rates=[],
        ablation=dict(rates=[2, 6], conc=4, reps=2),
        precision=[("float32", "none"), ("float16", "none")], prec_conc=4, prec_requests=12,
        quality_configs=[("float32", "none")],
        vllm=None,
        engines=dict(configs={"dynamic-8": dict(engine="dynamic", max_batch=8),
                              "slot-8": dict(engine="slot", max_batch=8)},
                     rates=[2, 6], seeds=[0, 1], levels=[4], reps=2),
        decode_graphs=True,
        chunked=dict(workload="longmix", rates=[1, 2], seeds=[0], chunks=[0, 128], max_batch=8),
        prefix=dict(workload="shared", rates=[2, 6], seeds=[0], max_batch=8),
        scale=None,
    ),
    "t4": dict(
        model="Qwen/Qwen2.5-0.5B-Instruct", dtype="float16", peak_bw_gbs=320.0, tiny=False,
        slo_ttft=1.0, slo_tpot=0.1, window_s=25, drain_s=45, min_req=12,
        sched_configs={
            "sequential": dict(mode="sequential", rates=[0.1, 0.2, 0.4, 0.8]),
            "static-16": dict(mode="static", max_batch=16, rates=[0.25, 0.5, 1, 2, 3, 4, 8]),
            "continuous-16": dict(mode="continuous", max_batch=16, rates=[0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12]),
            "continuous-32": dict(mode="continuous", max_batch=32, rates=[0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12]),
        },
        conc_configs={
            "sequential": dict(mode="sequential", levels=[1, 4]),
            "static-16": dict(mode="static", max_batch=16, levels=[1, 2, 4, 8, 16]),
            "continuous-16": dict(mode="continuous", max_batch=16, levels=[1, 2, 4, 8, 16, 32]),
            "continuous-32": dict(mode="continuous", max_batch=32, levels=[8, 16, 32, 64]),
        },
        conc_reps=3, workload="mixed",
        # optional patch-up experiments (used when extending an earlier run); empty in a full run
        conc_extra={}, refine_rates={}, vllm_extra_rates=[],
        # prefill optimisation ablation: full-vocab logits at every prompt position vs last position only
        ablation=dict(rates=[2, 4, 6, 8], conc=16, reps=3),
        precision=[("float32", "none"), ("float16", "none"), ("float16", "int8"), ("float16", "nf4")],
        prec_conc=16, prec_requests=64,
        quality_configs=[("float32", "none"), ("float16", "none"), ("float16", "int8"), ("float16", "nf4")],
        vllm=dict(port=8001, max_num_seqs=16, venv="/content/vllm-env",
                  rates=[0.5, 1, 2, 4, 8, 12, 16, 24, 32], levels=[1, 2, 4, 8, 16, 32]),
        # static-slot engine: same load, same session, same model; only the engine changes
        engines=dict(configs={"dynamic-32": dict(engine="dynamic", max_batch=32),
                              "slot-32": dict(engine="slot", max_batch=32),
                              "slot-graph-32": dict(engine="slot", graphs=True, max_batch=32)},
                     rates=[2, 4, 6, 8, 10, 12, 16], seeds=[0, 1], levels=[8, 32], reps=3),
        decode_graphs=True,
        chunked=dict(workload="longmix", rates=[0.5, 1, 2, 3], seeds=[0, 1], chunks=[0, 256], max_batch=32),
        prefix=dict(workload="shared", rates=[2, 4, 8, 12, 16], seeds=[0, 1], max_batch=32),
        scale=dict(model="Qwen/Qwen2.5-1.5B-Instruct", rates=[1, 2, 4, 6], levels=[1, 8, 16], reps=3, max_batch=16),
    ),
}


# ----------------------------------------------------------------------------- helpers
def _wait_healthy(url, proc=None, log=None, timeout=420):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            tail = Path(log).read_text()[-3000:] if log and Path(log).exists() else ""
            raise RuntimeError(f"server exited early (code {proc.returncode}):\n{tail}")
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"{url} not healthy after {timeout}s")


class Server:
    """Context manager that runs `python -m llmserve.server` in a subprocess."""

    def __init__(self, prof, name, mode, max_batch=16, dtype=None, quant="none", port=8000, extra=()):
        self.prof, self.name, self.port = prof, name, port
        self.args = ["--model", prof["model"], "--mode", mode, "--max-batch", str(max_batch),
                     "--dtype", dtype or prof["dtype"], "--quant", quant, "--port", str(port),
                     "--max-pending", "4096", *extra]   # benchmark: queue effectively unbounded so overload shows as latency, not 503s
        self.base = f"http://127.0.0.1:{port}"
        self.url = f"{self.base}/v1/chat/completions"

    def __enter__(self):
        LOGS.mkdir(parents=True, exist_ok=True)
        self.log = LOGS / f"server-{self.name}.log"
        env = dict(os.environ, PYTHONPATH=SRC + os.pathsep + os.environ.get("PYTHONPATH", ""))
        self.proc = subprocess.Popen([sys.executable, "-m", "llmserve.server", *self.args],
                                     stdout=open(self.log, "w"), stderr=subprocess.STDOUT, env=env)
        _wait_healthy(f"{self.base}/health", self.proc, self.log)
        asyncio.run(warmup(self.url, self.prof["model"]))
        return self

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        time.sleep(2)  # let the GPU free its memory

    def engine_stats(self, reset=True):
        try:
            return httpx.get(f"{self.base}/engine_stats", params={"reset": int(reset)}, timeout=5).json()
        except Exception:
            return {}

    def stats(self):
        try:
            return httpx.get(f"{self.base}/stats", timeout=5).json()
        except Exception:
            return {}


def _save_records(name, records):
    RAW.mkdir(parents=True, exist_ok=True)
    name = name.replace("/", "_")
    pd.DataFrame([r.__dict__ | {"tpot": r.tpot} for r in records]).to_csv(RAW / f"{name}.csv", index=False)


def _append(path, rows):
    PROC.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def _merge_save(path, new_rows, keys):
    """Merge new rows into an existing CSV (new rows win on duplicate keys)."""
    new = pd.DataFrame(new_rows)
    if Path(path).exists() and Path(path).stat().st_size > 0:
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new = new.drop_duplicates(subset=keys, keep="last").sort_values(keys).reset_index(drop=True)
    new.to_csv(path, index=False)
    return new


def _open_loop_sweep(prof, srv, cfg_name, model_name, rates, tag, seed0=0):
    rows = []
    for i, rate in enumerate(rates):
        n = max(prof["min_req"], int(rate * prof["window_s"]))
        srv.engine_stats(reset=True)
        records, summ = asyncio.run(run_open_loop(
            srv.url, model_name, rate, n, workload=prof["workload"], seed=seed0 + i,
            max_drain_s=prof["drain_s"], slo_ttft=prof["slo_ttft"], slo_tpot=prof["slo_tpot"]))
        summ.update(srv.engine_stats(reset=True))
        summ.update(config=cfg_name, workload=prof["workload"], n_requests=n)
        rows.append(summ)
        _save_records(f"{tag}-{cfg_name}-rate{rate}", records)
        print(f"  [{cfg_name}] rate={rate:>5}: ok={summ['succeeded']}/{n} "
              f"tok/s={summ['output_tok_per_s']:.1f} ttft_p95={summ['ttft_p95']:.2f}s "
              f"goodput={summ['goodput_rps']:.2f} rps", flush=True)
    return rows


# ----------------------------------------------------------------------------- experiments
def exp_env(prof):
    info = {"date_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "python": platform.python_version(), "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "profile_model": prof["model"], "dtype": prof["dtype"]}
    try:
        import torch, transformers
        info.update(torch=torch.__version__, transformers=transformers.__version__,
                    cuda_available=torch.cuda.is_available())
        import importlib.metadata as md
        for pkg in ("accelerate", "fastapi", "uvicorn", "httpx", "pandas", "numpy", "matplotlib", "bitsandbytes", "datasets"):
            try:
                info[f"pkg_{pkg}"] = md.version(pkg)
            except Exception:
                info[f"pkg_{pkg}"] = None
        if torch.cuda.is_available():
            info.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                        gpu_mem_gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
    except Exception as exc:
        info["torch_error"] = str(exc)
    try:
        info["git_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                            capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    PROC.mkdir(parents=True, exist_ok=True)
    (PROC / "environment.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


def exp_scheduler(prof):
    """Open-loop Poisson sweep: sequential vs static vs continuous batching."""
    rows = []
    for name, cfg in prof["sched_configs"].items():
        print(f"== scheduler: {name}", flush=True)
        with Server(prof, name, cfg["mode"], cfg.get("max_batch", 16)) as srv:
            rows += _open_loop_sweep(prof, srv, name, prof["model"], cfg["rates"], "sched")
        _append(PROC / "scheduler_open_loop.csv", rows)


def exp_concurrency(prof):
    """Closed-loop concurrency sweep with repeats -> peak throughput with run-to-run spread."""
    rows = []
    for name, cfg in prof["conc_configs"].items():
        print(f"== concurrency: {name}", flush=True)
        with Server(prof, name, cfg["mode"], cfg.get("max_batch", 16)) as srv:
            for c in cfg["levels"]:
                for rep in range(prof["conc_reps"]):
                    n = max(16, 4 * c)
                    srv.engine_stats(reset=True)
                    records, summ = asyncio.run(run_closed_loop(
                        srv.url, prof["model"], c, n, workload=prof["workload"], seed=100 + rep,
                        slo_ttft=prof["slo_ttft"], slo_tpot=prof["slo_tpot"]))
                    summ.update(srv.engine_stats(reset=True))
                    summ.update(config=name, concurrency=c, rep=rep, workload=prof["workload"])
                    rows.append(summ)
                    _save_records(f"conc-{name}-c{c}-rep{rep}", records)
                last = [r for r in rows if r["config"] == name and r["concurrency"] == c]
                print(f"  [{name}] c={c:>3}: tok/s="
                      f"{sum(r['output_tok_per_s'] for r in last)/len(last):.1f} "
                      f"(n={len(last)} reps)", flush=True)
            _append(PROC / "concurrency_closed_loop.csv", rows)


def exp_microbench(prof):
    from .microbench import run

    pre, dec, meta = run(prof["model"], prof["dtype"], PROC, tiny_scale=prof["tiny"],
                         peak_bw_gbs=prof["peak_bw_gbs"])
    print(dec.round(3).to_string(index=False))


def exp_precision(prof):
    """fp32 vs fp16 vs int8 vs nf4: throughput + GPU memory (quality comes from exp_quality)."""
    rows = []
    for dtype, quant in prof["precision"]:
        name = f"{dtype}-{quant}"
        print(f"== precision: {name}", flush=True)
        try:
            with Server(prof, f"prec-{name}", "continuous", 16, dtype=dtype, quant=quant) as srv:
                for rep in range(prof["conc_reps"]):
                    records, summ = asyncio.run(run_closed_loop(
                        srv.url, prof["model"], prof["prec_conc"], prof["prec_requests"],
                        workload=prof["workload"], seed=200 + rep,
                        slo_ttft=prof["slo_ttft"], slo_tpot=prof["slo_tpot"]))
                    summ.update(srv.engine_stats(reset=True))
                    summ.update(dtype=dtype, quant=quant, rep=rep, **{
                        k: v for k, v in srv.stats().items() if k.startswith("gpu_mem") or k == "param_bytes"})
                    rows.append(summ)
                print(f"  tok/s={sum(r['output_tok_per_s'] for r in rows if r['dtype']==dtype and r['quant']==quant)/prof['conc_reps']:.1f}",
                      flush=True)
        except Exception as exc:
            print(f"  skipped {name}: {exc}", flush=True)
            rows.append(dict(dtype=dtype, quant=quant, error=str(exc)[:300]))
        _append(PROC / "precision_throughput.csv", rows)


def exp_quality(prof):
    if prof["tiny"]:
        print("quality eval needs a real tokenizer; skipped in smoke profile")
        return
    from .quality import run

    df = run(prof["model"], prof["quality_configs"], PROC / "precision_quality.csv")
    print(df.round(4).to_string(index=False))


def _vllm_up(prof):
    """Start vLLM (best effort). Returns (proc, url) or None."""
    cfg = prof.get("vllm")
    if not cfg:
        print("no vllm config for this profile; skipping")
        return None
    py = Path(cfg["venv"]) / "bin" / "python"
    if not py.exists():
        print(f"vLLM venv not found at {cfg['venv']} (run the install cell first); skipping")
        return None
    base = f"http://127.0.0.1:{cfg['port']}"
    cmd = [str(py), "-m", "vllm.entrypoints.openai.api_server", "--model", prof["model"],
           "--dtype", "half", "--max-model-len", "4096", "--max-num-seqs", str(cfg["max_num_seqs"]),
           "--gpu-memory-utilization", "0.45", "--port", str(cfg["port"])]
    LOGS.mkdir(parents=True, exist_ok=True)
    for backend in ["", "TRITON_ATTN", "XFORMERS"]:
        log = LOGS / f"vllm-{backend or 'default'}.log"
        env = dict(os.environ)
        if backend:
            env["VLLM_ATTENTION_BACKEND"] = backend
        proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
        try:
            _wait_healthy(f"{base}/health", proc, log, timeout=420)
            print(f"vLLM up (attention backend: {backend or 'default'})", flush=True)
            url = f"{base}/v1/chat/completions"
            asyncio.run(warmup(url, prof["model"]))
            return proc, url
        except Exception as exc:
            print(f"vLLM attempt with backend={backend or 'default'} failed: {str(exc)[-500:]}", flush=True)
            proc.terminate()
            time.sleep(3)
    print("vLLM could not be started on this GPU; the report will omit the baseline.")
    return None


def _vllm_down(proc):
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(3)


class _Remote:
    """Stand-in for Server when the target is an external OpenAI-compatible server."""

    def __init__(self, url):
        self.url = url

    def engine_stats(self, reset=True):
        return {}


def exp_vllm(prof):
    """Baseline: the same load against vLLM serving the same model (best effort, separate venv)."""
    up = _vllm_up(prof)
    if not up:
        return
    proc, url = up
    cfg = prof["vllm"]
    try:
        rows = _open_loop_sweep(prof, _Remote(url), "vllm", prof["model"], cfg["rates"], "sched")
        _append(PROC / "vllm_open_loop.csv", rows)
        crow = []
        for c in cfg["levels"]:
            for rep in range(prof["conc_reps"]):
                records, summ = asyncio.run(run_closed_loop(
                    url, prof["model"], c, max(16, 4 * c), workload=prof["workload"], seed=100 + rep,
                    slo_ttft=prof["slo_ttft"], slo_tpot=prof["slo_tpot"]))
                summ.update(config="vllm", concurrency=c, rep=rep, workload=prof["workload"])
                crow.append(summ)
            print(f"  [vllm] c={c}: tok/s={sum(r['output_tok_per_s'] for r in crow if r['concurrency']==c)/prof['conc_reps']:.1f}",
                  flush=True)
        _append(PROC / "vllm_closed_loop.csv", crow)
    finally:
        _vllm_down(proc)


def exp_conc_extra(prof):
    """Closed-loop sweep for configs that were not in the first run; merged into the existing CSV."""
    rows = []
    for name, cfg in prof.get("conc_extra", {}).items():
        print(f"== concurrency (extra): {name}", flush=True)
        with Server(prof, name, cfg["mode"], cfg.get("max_batch", 16)) as srv:
            for c in cfg["levels"]:
                for rep in range(prof["conc_reps"]):
                    srv.engine_stats(reset=True)
                    records, summ = asyncio.run(run_closed_loop(
                        srv.url, prof["model"], c, max(16, 4 * c), workload=prof["workload"], seed=100 + rep,
                        slo_ttft=prof["slo_ttft"], slo_tpot=prof["slo_tpot"]))
                    summ.update(srv.engine_stats(reset=True))
                    summ.update(config=name, concurrency=c, rep=rep, workload=prof["workload"])
                    rows.append(summ)
                    _save_records(f"conc-{name}-c{c}-rep{rep}", records)
                last = [r for r in rows if r["config"] == name and r["concurrency"] == c]
                print(f"  [{name}] c={c:>3}: tok/s={sum(r['output_tok_per_s'] for r in last)/len(last):.1f}", flush=True)
    if rows:
        _merge_save(PROC / "concurrency_closed_loop.csv", rows, ["config", "concurrency", "rep"])


def exp_refine(prof):
    """Add rates to the open-loop grid: finer steps near each engine's knee, and higher rates for vLLM."""
    for name, rates in prof.get("refine_rates", {}).items():
        cfg = prof["sched_configs"].get(name)
        if not cfg:
            continue
        print(f"== refine: {name} at {rates}", flush=True)
        with Server(prof, name, cfg["mode"], cfg.get("max_batch", 16)) as srv:
            rows = _open_loop_sweep(prof, srv, name, prof["model"], rates, "sched", seed0=500)
        _merge_save(PROC / "scheduler_open_loop.csv", rows, ["config", "offered_rps"])
    extra = prof.get("vllm_extra_rates")
    if extra:
        up = _vllm_up(prof)
        if up:
            proc, url = up
            try:
                rows = _open_loop_sweep(prof, _Remote(url), "vllm", prof["model"], extra, "sched", seed0=500)
                _merge_save(PROC / "vllm_open_loop.csv", rows, ["config", "offered_rps"])
            finally:
                _vllm_down(proc)


def exp_ablation(prof):
    """Prefill optimisation: compute lm_head only for the last prompt position (vs every position).

    Same server, same load, same session; the only difference is the --full-logits flag.
    """
    ab = prof.get("ablation")
    if not ab:
        return
    rows = []
    for variant, extra in (("full-logits", ("--full-logits",)), ("last-logits", ())):
        print(f"== ablation: {variant}", flush=True)
        with Server(prof, f"abl-{variant}", "continuous", 16, extra=extra) as srv:
            for r in _open_loop_sweep(prof, srv, f"continuous-16/{variant}", prof["model"], ab["rates"], "abl", seed0=700):
                r.update(variant=variant, kind="open_loop")
                rows.append(r)
            for rep in range(ab["reps"]):
                srv.engine_stats(reset=True)
                records, summ = asyncio.run(run_closed_loop(
                    srv.url, prof["model"], ab["conc"], max(32, 4 * ab["conc"]), workload=prof["workload"],
                    seed=300 + rep, slo_ttft=prof["slo_ttft"], slo_tpot=prof["slo_tpot"]))
                summ.update(srv.engine_stats(reset=True))
                summ.update(config=f"continuous-16/{variant}", variant=variant, kind="closed_loop",
                            concurrency=ab["conc"], rep=rep)
                rows.append(summ)
            cl = [r for r in rows if r["variant"] == variant and r["kind"] == "closed_loop"]
            print(f"  closed-loop c={ab['conc']}: tok/s={sum(r['output_tok_per_s'] for r in cl)/len(cl):.1f}", flush=True)
    _append(PROC / "ablation_logits.csv", rows)


# ----------------------------------------------------------------------------- slot-engine experiments
def _slot_args(cfg):
    if cfg.get("engine", "dynamic") != "slot":
        return []
    a = ["--engine", "slot"]
    if cfg.get("graphs"):
        a.append("--cuda-graphs")
    if cfg.get("chunk"):
        a += ["--chunk-size", str(cfg["chunk"])]
    if cfg.get("prefix"):
        a.append("--prefix-cache")
    return a


def _require_graphs(srv, cfg):
    """A 'graphs' configuration must really run with CUDA graphs; never silently measure the eager fallback."""
    if not cfg.get("graphs"):
        return
    try:
        import torch
        if not torch.cuda.is_available():
            return                      # CPU smoke run: graphs are a no-op by design
    except Exception:
        return
    st = srv.stats()
    if not st.get("cuda_graphs"):
        raise RuntimeError(f"CUDA graph capture did not take effect ({st.get('graph_error')}); see {srv.log}")


def _sweep_seeds(prof, srv, name, rates, seeds, tag):
    rows = []
    for seed in seeds:
        for r in _open_loop_sweep(prof, srv, name, prof["model"], rates, f"{tag}-s{seed}", seed0=1000 + 100 * seed):
            r["seed"] = seed
            rows.append(r)
    return rows


def _closed_sweep(prof, srv, name, levels, reps, tag):
    rows = []
    for c in levels:
        for rep in range(reps):
            srv.engine_stats(reset=True)
            records, summ = asyncio.run(run_closed_loop(
                srv.url, prof["model"], c, max(16, 4 * c), workload=prof["workload"], seed=100 + rep,
                slo_ttft=prof["slo_ttft"], slo_tpot=prof["slo_tpot"]))
            summ.update(srv.engine_stats(reset=True))
            summ.update(config=name, concurrency=c, rep=rep, workload=prof["workload"])
            rows.append(summ)
            _save_records(f"{tag}-{name}-c{c}-rep{rep}", records)
        sel = [r["output_tok_per_s"] for r in rows if r["concurrency"] == c]
        print(f"  [{name}] c={c:>3}: tok/s={sum(sel)/len(sel):.1f} (n={len(sel)})", flush=True)
    return rows


def exp_decode_graphs(prof):
    """Decode-step time: HF dynamic cache vs static slots (eager) vs static slots + CUDA graph."""
    if not prof.get("decode_graphs"):
        return
    from .microbench import run_graph_microbench

    df = run_graph_microbench(prof["model"], prof["dtype"], PROC, tiny_scale=prof["tiny"])
    print(df.round(2).to_string(index=False))


def exp_engines(prof):
    """dynamic-cache engine vs static-slot engine vs static-slot + CUDA graphs: open-loop (seeds) and closed-loop (reps)."""
    ex = prof.get("engines")
    if not ex:
        return
    orows, crows = [], []
    for name, cfg in ex["configs"].items():
        print(f"== engines: {name}", flush=True)
        with Server(prof, name, "continuous", cfg["max_batch"], extra=_slot_args(cfg)) as srv:
            _require_graphs(srv, cfg)
            print("  server stats:", {k: v for k, v in srv.stats().items() if k in ("engine", "cuda_graphs", "n_graphs", "graph_error")}, flush=True)
            orows += _sweep_seeds(prof, srv, name, ex["rates"], ex["seeds"], "eng")
            crows += _closed_sweep(prof, srv, name, ex["levels"], ex["reps"], "engc")
        _append(PROC / "engines_open_loop.csv", orows)
        _append(PROC / "engines_closed_loop.csv", crows)


def exp_chunked(prof):
    """Chunked prefill on a workload where ~12% of prompts are ~1.3k tokens (slot engine, same graphs)."""
    ex = prof.get("chunked")
    if not ex:
        return
    p = {**prof, "workload": ex["workload"]}
    rows = []
    for chunk in ex["chunks"]:
        name = f"chunk-{chunk}" if chunk else "no-chunking"
        print(f"== chunked prefill: {name}", flush=True)
        cfg = dict(engine="slot", graphs=True, chunk=chunk)
        with Server(p, f"chk-{name}", "continuous", ex["max_batch"], extra=_slot_args(cfg)) as srv:
            _require_graphs(srv, cfg)
            for r in _sweep_seeds(p, srv, name, ex["rates"], ex["seeds"], "chk"):
                r["chunk_size"] = chunk
                rows.append(r)
        _append(PROC / "chunked_open_loop.csv", rows)


def exp_prefix(prof):
    """Prefix caching on a shared-preamble workload."""
    ex = prof.get("prefix")
    if not ex:
        return
    p = {**prof, "workload": ex["workload"]}
    rows = []
    for on in (False, True):
        name = "prefix-cache" if on else "no-prefix-cache"
        print(f"== prefix caching: {name}", flush=True)
        cfg = dict(engine="slot", graphs=True, prefix=on)
        with Server(p, f"pfx-{name}", "continuous", ex["max_batch"], extra=_slot_args(cfg)) as srv:
            _require_graphs(srv, cfg)
            for r in _sweep_seeds(p, srv, name, ex["rates"], ex["seeds"], "pfx"):
                r["prefix_cache"] = on
                rows.append(r)
        _append(PROC / "prefix_open_loop.csv", rows)


def exp_scale(prof):
    """Does the engine ranking hold on a bigger model? Dynamic vs slot+graphs on Qwen2.5-1.5B."""
    ex = prof.get("scale")
    if not ex:
        return
    p = {**prof, "model": ex["model"]}
    orows, crows = [], []
    for name, cfg in {"dynamic-16": dict(engine="dynamic"), "slot-graph-16": dict(engine="slot", graphs=True)}.items():
        print(f"== scale ({ex['model']}): {name}", flush=True)
        with Server(p, f"scale-{name}", "continuous", ex["max_batch"], extra=_slot_args(cfg)) as srv:
            _require_graphs(srv, cfg)
            orows += _sweep_seeds(p, srv, name, ex["rates"], [0], "scl")
            crows += _closed_sweep(p, srv, name, ex["levels"], ex["reps"], "sclc")
        _append(PROC / "scale_open_loop.csv", orows)
        _append(PROC / "scale_closed_loop.csv", crows)


EXPERIMENTS = {"env": exp_env, "scheduler": exp_scheduler, "concurrency": exp_concurrency,
               "microbench": exp_microbench, "precision": exp_precision, "quality": exp_quality,
               "vllm": exp_vllm, "conc_extra": exp_conc_extra, "refine": exp_refine,
               "ablation": exp_ablation, "decode_graphs": exp_decode_graphs, "engines": exp_engines,
               "chunked": exp_chunked, "prefix": exp_prefix, "scale": exp_scale}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="t4", choices=list(PROFILES))
    ap.add_argument("--only", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    a = ap.parse_args()
    prof = PROFILES[a.profile]
    for name in a.only:
        t0 = time.time()
        print(f"\n######## {name} ({a.profile}) ########", flush=True)
        EXPERIMENTS[name](prof)
        print(f"-- {name} finished in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
