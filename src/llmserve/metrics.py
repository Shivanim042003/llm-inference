"""Latency / throughput / goodput metrics for serving benchmarks."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional


def percentile(values: Iterable[float], p: float) -> float:
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return float("nan")
    if not 0 <= p <= 100:
        raise ValueError("p must be in [0, 100]")
    pos = (len(vals) - 1) * p / 100
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def bootstrap_ci(values: List[float], stat=lambda v: sum(v) / len(v), n_boot=1000, alpha=0.05, seed=0):
    """Percentile bootstrap confidence interval for any statistic."""
    import random

    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rnd = random.Random(seed)
    stats = sorted(stat([rnd.choice(vals) for _ in vals]) for _ in range(n_boot))
    return (stats[int(alpha / 2 * n_boot)], stats[int((1 - alpha / 2) * n_boot) - 1])


@dataclass
class RequestRecord:
    """One benchmark request. Times are seconds relative to the start of the run."""
    t_arrival: float
    ok: bool
    ttft: Optional[float] = None          # time to first token
    e2e: Optional[float] = None           # end-to-end latency
    prompt_tokens: int = 0
    output_tokens: int = 0
    error: str = ""
    engine_ttft: Optional[float] = None   # server-side time from submit to first generated token (if reported)

    @property
    def tpot(self) -> Optional[float]:
        """Mean time per output token after the first (decode speed seen by the user)."""
        if not self.ok or self.ttft is None or self.e2e is None or self.output_tokens < 2:
            return None
        return (self.e2e - self.ttft) / (self.output_tokens - 1)


def summarize(records: List[RequestRecord], wall_s: float, offered_rps: Optional[float] = None,
              slo_ttft: float = 1.0, slo_tpot: float = 0.1) -> dict:
    ok = [r for r in records if r.ok]
    n = len(records)
    out_tokens = sum(r.output_tokens for r in ok)
    meets = [r for r in ok if r.ttft is not None and r.ttft <= slo_ttft
             and (r.tpot is None or r.tpot <= slo_tpot)]
    tpots = [r.tpot for r in ok if r.tpot is not None]
    return {
        "offered_rps": offered_rps,
        "requests": n,
        "succeeded": len(ok),
        "failed": n - len(ok),
        "wall_s": wall_s,
        "throughput_rps": len(ok) / wall_s if wall_s > 0 else 0.0,
        "output_tok_per_s": out_tokens / wall_s if wall_s > 0 else 0.0,
        "ttft_p50": percentile([r.ttft for r in ok], 50),
        "ttft_p95": percentile([r.ttft for r in ok], 95),
        "ttft_p99": percentile([r.ttft for r in ok], 99),
        "engine_ttft_p50": percentile([r.engine_ttft for r in ok], 50),
        "engine_ttft_p95": percentile([r.engine_ttft for r in ok], 95),
        "tpot_p50": percentile(tpots, 50),
        "tpot_p95": percentile(tpots, 95),
        "e2e_p50": percentile([r.e2e for r in ok], 50),
        "e2e_p95": percentile([r.e2e for r in ok], 95),
        "e2e_p99": percentile([r.e2e for r in ok], 99),
        "ttft_definition": "first_non_empty_content_chunk",
        "slo_ttft_s": slo_ttft,
        "slo_tpot_s": slo_tpot,
        "slo_attainment": len(meets) / n if n else 0.0,
        "goodput_rps": len(meets) / wall_s if wall_s > 0 else 0.0,
    }
