"""One clean path from download to report:  python -m techdraft all [--vllm]   (needs a GPU; see README steps)

Steps: download | quality | sweep | report | all.  Results go to results/processed, raw model replies to data/techdraft/raw (git-ignored).
"""
from __future__ import annotations

import argparse
import sys

QUALITY_RUNS = (("quality", "Qwen/Qwen2.5-1.5B-Instruct", 0.5), ("quality3b", "Qwen/Qwen2.5-3B-Instruct", 0.3))
SWEEP_MODEL, SWEEP_RATES, SWEEP_ITEMS = "Qwen/Qwen2.5-1.5B-Instruct", (0.25, 0.5, 0.75, 1.0, 1.5), 100


def _prof(model, vllm_gpu_util=0.5):
    from llmserve.experiments import PROFILES
    prof = dict(PROFILES["t4"])
    prof["model"] = model
    if prof.get("vllm"):
        prof["vllm"] = dict(prof["vllm"], gpu_util=vllm_gpu_util)
    return prof


def step_download():
    from .data import download
    print("saved", download())


def step_quality():
    """Every question of each split once, one model at a time (tune is reported as the split used for choices)."""
    from .compare import compare_backends
    for split in ("tune", "final"):
        for label, model, rate in QUALITY_RUNS:
            compare_backends(_prof(model), [rate], backends=("slot-graph-32",), split_name=split, run_label=label, min_seconds=0)


def step_sweep(with_vllm):
    from .compare import compare_backends
    compare_backends(_prof(SWEEP_MODEL), list(SWEEP_RATES), backends=("dynamic-32", "slot-graph-32", "slot-graph-32+prefix"), with_vllm=with_vllm,
                     split_name="final", run_label="sweep", max_items=SWEEP_ITEMS)


def step_report():
    from .report import build
    build()
    print("wrote reports/TECHDRAFT.md and results/processed/techdraft_manifest.json")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m techdraft", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["download", "quality", "sweep", "report", "all"])
    ap.add_argument("--vllm", action="store_true", help="include vLLM in the sweep (its environment must already exist)")
    a = ap.parse_args(argv)
    if a.step in ("download", "all"):
        step_download()
    if a.step in ("quality", "all"):
        step_quality()
    if a.step in ("sweep", "all"):
        step_sweep(a.vllm)
    if a.step in ("report", "all"):
        step_report()


if __name__ == "__main__":
    sys.exit(main())
