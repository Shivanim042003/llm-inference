"""Latency objectives, 'useful draft' accounting, and cost / capacity arithmetic.

A draft is USEFUL when the request succeeded, it passed the automated quality checks, and the whole draft arrived within the
end-to-end latency objective. Time to first token is reported separately and does not make a draft on time by itself.
All dollar figures depend on an ASSUMED GPU price per hour; nothing here is a measured price.
"""
from __future__ import annotations


def is_useful(rec, e2e_slo_s):
    return bool(rec["ok"] and rec["correct"] and rec["e2e_s"] is not None and rec["e2e_s"] <= e2e_slo_s)


def percentile(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def serving_summary(recs, wall_s, e2e_slo_s, ttft_slo_s):
    n = len(recs)
    ok = [r for r in recs if r["ok"]]
    useful = sum(is_useful(r, e2e_slo_s) for r in recs)
    return dict(n_requests=n, failed=n - len(ok), wall_s=wall_s,
                ttft_p50=percentile([r["ttft_s"] for r in ok], .5), ttft_p95=percentile([r["ttft_s"] for r in ok], .95),
                e2e_p50=percentile([r["e2e_s"] for r in ok], .5), e2e_p95=percentile([r["e2e_s"] for r in ok], .95),
                on_time_rate=(sum(r["e2e_s"] <= e2e_slo_s for r in ok) / n) if n else float("nan"),
                ttft_ok_rate=(sum(r["ttft_s"] is not None and r["ttft_s"] <= ttft_slo_s for r in ok) / n) if n else float("nan"),
                useful=useful, useful_rate=useful / n if n else float("nan"),
                useful_per_s=useful / wall_s if wall_s > 0 else float("nan"),
                output_tok_per_s=sum(r["output_tokens"] or 0 for r in ok) / wall_s if wall_s > 0 else float("nan"))


def sustainable_rate(rows, min_ok=0.95):
    """Highest tested offered rate at which no request failed, >= min_ok of requests were on time, and the draft rate kept up.
    rows: dicts with offered_rps, failed, on_time_rate. Returns (rate, next_tested_rate_or_None). 0.0 if none qualified."""
    rows = sorted(rows, key=lambda r: r["offered_rps"])
    good = [r["offered_rps"] for r in rows if r["failed"] == 0 and r["on_time_rate"] >= min_ok]
    cap = max(good) if good else 0.0
    higher = [r["offered_rps"] for r in rows if r["offered_rps"] > cap]
    return cap, (min(higher) if higher else None)


def saturated_cost(useful, wall_s, gpu_usd_per_hour):
    """GPU cost while the benchmark ran, per 1,000 useful drafts (GPU assumed fully used for the run)."""
    usd = wall_s / 3600 * gpu_usd_per_hour
    return dict(gpu_usd=usd, usd_per_1k_useful=(1000 * usd / useful) if useful else float("inf"))


def scenario(sustainable_rps, useful_rate, gpu_usd_per_hour, agents, drafts_per_agent_per_hour, service_hours_per_day):
    """Scenario ESTIMATE from measured numbers plus stated assumptions (all assumption arguments are inputs, not findings).

    agents_supported: how many agents the measured sustainable draft rate could serve at the assumed request frequency.
    provisioned cost: the GPU is paid for the whole service window, including idle time.
    """
    per_agent_rps = drafts_per_agent_per_hour / 3600
    supported = sustainable_rps / per_agent_rps if per_agent_rps > 0 else float("inf")
    daily_requests = agents * drafts_per_agent_per_hour * service_hours_per_day
    daily_useful = daily_requests * useful_rate
    daily_gpu = gpu_usd_per_hour * service_hours_per_day
    return dict(agents_supported=supported, daily_requests=daily_requests, daily_useful=daily_useful, daily_gpu_usd=daily_gpu,
                usd_per_1k_useful_provisioned=(1000 * daily_gpu / daily_useful) if daily_useful else float("inf"),
                utilization_of_sustainable=(agents * per_agent_rps / sustainable_rps) if sustainable_rps else float("inf"))
