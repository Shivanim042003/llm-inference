"""Send a spike of support tickets to any OpenAI-compatible backend, then score the drafts and summarise latency."""
from __future__ import annotations

import asyncio
import json
import math
import random
import time
from pathlib import Path

import httpx

from .cost import serving_summary
from .retrieval import build_messages, retrieve
from .score import score_draft, summarize_quality
from .tickets import TICKETS

TICKETS_BY_ID = {t["ticket_id"]: t for t in TICKETS}


async def _one(client, url, model, ticket, k, max_tokens, t_run0, timeout):
    ids = retrieve(ticket["question"], k)
    t0 = time.perf_counter()
    rec = dict(ticket_id=ticket["ticket_id"], kind=ticket["kind"], split=ticket["split"], retrieved_ids=ids, draft="", ok=False,
               error=None, ttft_s=None, e2e_s=None, prompt_tokens=None, output_tokens=None, t_arrival=t0 - t_run0)
    body = {"model": model, "messages": build_messages(ticket["question"], ids), "max_tokens": max_tokens, "stream": True,
            "temperature": 0, "stream_options": {"include_usage": True}}      # ignore_eos left off: drafts must stop at EOS
    parts = []
    try:
        async with client.stream("POST", url, json=body, timeout=timeout) as resp:
            if resp.status_code != 200:
                rec["error"] = f"http {resp.status_code}"
                return rec
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                if data.get("error"):
                    rec["error"] = str(data["error"].get("type", "stream_error"))
                    return rec
                if data.get("usage"):
                    rec["prompt_tokens"] = data["usage"].get("prompt_tokens")
                    rec["output_tokens"] = data["usage"].get("completion_tokens")
                for ch in data.get("choices") or []:
                    text = (ch.get("delta") or {}).get("content")
                    if text:
                        if rec["ttft_s"] is None:
                            rec["ttft_s"] = time.perf_counter() - t0       # first non-empty content chunk
                        parts.append(text)
        rec["e2e_s"] = time.perf_counter() - t0
        rec["draft"] = "".join(parts)
        rec["ok"] = rec["ttft_s"] is not None
        if not rec["ok"]:
            rec["error"] = "empty_response"
    except Exception as exc:
        rec["error"] = type(exc).__name__
    return rec


async def _spike(url, model, tickets, rate, seed, repeats, k, max_tokens, timeout):
    rnd = random.Random(seed)
    work = [t for _ in range(repeats) for t in rnd.sample(tickets, len(tickets))]
    t_run0 = time.perf_counter()
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=1000, max_keepalive_connections=1000)) as client:
        tasks, at = [], 0.0
        for t in work:
            at += rnd.expovariate(rate)
            delay = max(0.0, t_run0 + at - time.perf_counter())
            await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(_one(client, url, model, t, k, max_tokens, t_run0, timeout)))
        recs = await asyncio.gather(*tasks)
    return list(recs), time.perf_counter() - t_run0


def run_spike(url, model, tickets, rate, seed=0, repeats=1, k=3, max_tokens=200, timeout=90.0):
    return asyncio.run(_spike(url, model, tickets, rate, seed, repeats, k, max_tokens, timeout))


def score_records(recs):
    for r in recs:
        s = score_draft(TICKETS_BY_ID[r["ticket_id"]], r["draft"], r["retrieved_ids"]) if r["ok"] else dict(correct=False, failure="request_failed")
        r.update(score=s, correct=bool(s["correct"]))
    return recs


def evaluate(url, model, tickets, rate, e2e_slo_s, ttft_slo_s, seed=0, min_seconds=30.0, **kw):
    """One offered rate: run, score, summarise. repeats is chosen so that the run lasts at least ~min_seconds."""
    repeats = max(1, math.ceil(min_seconds * rate / len(tickets)))
    recs, wall = run_spike(url, model, tickets, rate, seed=seed, repeats=repeats, **kw)
    score_records(recs)
    serving = serving_summary(recs, wall, e2e_slo_s, ttft_slo_s)
    served = [(r["ticket_id"], r["score"]) for r in recs if r["ok"]]
    quality = summarize_quality(served, TICKETS_BY_ID) if served else {}
    quality["requests_failed"] = len(recs) - len(served)
    return recs, dict(offered_rps=rate, repeats=repeats, **serving), quality


def save(recs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
