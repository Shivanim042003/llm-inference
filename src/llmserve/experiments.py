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
        precision=[("float32", "none"), ("float16", "none")], prec_conc=4, prec_requests=12,
        quality_configs=[("float32", "none")],
        vllm=None,
    ),
    "t4": dict(
        model="Qwen/Qwen2.5-0.5B-Instruct", dtype="float16", peak_bw_gbs=320.0, tiny=False,
        slo_ttft=1.0, slo_tpot=0.1, window_s=25, drain_s=45, min_req=12,
        sched_configs={
            "sequential": dict(mode="sequential", rates=[0.1, 0.2, 0.4, 0.8]),
            "static-16": dict(mode="static", max_batch=16, rates=[0.25, 0.5, 1, 2, 4, 8]),
            "continuous-16": dict(mode="continuous", max_batch=16, rates=[0.5, 1, 2, 4, 8, 12]),
            "continuous-32": dict(mode="continuous", max_batch=32, rates=[0.5, 1, 2, 4, 8, 12]),
        },
        conc_configs={
            "sequential": dict(mode="sequential", levels=[1, 2, 4, 8]),
            "static-16": dict(mode="static", max_batch=16, levels=[1, 2, 4, 8, 16]),
            "continuous-16": dict(mode="continuous", max_batch=16, levels=[1, 2, 4, 8, 16, 32]),
        },
        conc_reps=3, workload="mixed",
        precision=[("float32", "none"), ("float16", "none"), ("float16", "int8"), ("float16", "nf4")],
        prec_conc=16, prec_requests=64,
        quality_configs=[("float32", "none"), ("float16", "none"), ("float16", "int8"), ("float16", "nf4")],
        vllm=dict(port=8001, max_num_seqs=16, venv="/content/vllm-env", rates=[0.5, 1, 2, 4, 8, 12],
                  levels=[1, 2, 4, 8, 16, 32]),
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
                     "--dtype", dtype or prof["dtype"], "--quant", quant, "--port", str(port), *extra]
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
    pd.DataFrame([r.__dict__ | {"tpot": r.tpot} for r in records]).to_csv(RAW / f"{name}.csv", index=False)


def _append(path, rows):
    PROC.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


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


def exp_vllm(prof):
    """Baseline: the same load against vLLM serving the same model (best effort, separate venv)."""
    cfg = prof.get("vllm")
    if not cfg:
        print("no vllm config for this profile; skipping")
        return
    py = Path(cfg["venv"]) / "bin" / "python"
    if not py.exists():
        print(f"vLLM venv not found at {cfg['venv']} (run the install cell first); skipping")
        return
    base = f"http://127.0.0.1:{cfg['port']}"
    url = f"{base}/v1/chat/completions"
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
            break
        except Exception as exc:
            print(f"vLLM attempt with backend={backend or 'default'} failed: {str(exc)[-500:]}", flush=True)
            proc.terminate()
            time.sleep(3)
    else:
        print("vLLM could not be started on this GPU; the report will omit the baseline.")
        return
    try:
        asyncio.run(warmup(url, prof["model"]))
        rows = _open_loop_sweep(prof, type("S", (), {"url": url, "engine_stats": lambda self, reset=True: {}})(),
                                "vllm", prof["model"], cfg["rates"], "sched")
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
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


EXPERIMENTS = {"env": exp_env, "scheduler": exp_scheduler, "concurrency": exp_concurrency,
               "microbench": exp_microbench, "precision": exp_precision, "quality": exp_quality,
               "vllm": exp_vllm}


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
