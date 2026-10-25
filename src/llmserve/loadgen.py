"""Load generators: open-loop (Poisson arrivals) and closed-loop (fixed concurrency).

Open-loop is the honest way to measure a server: arrivals do not slow down when the server
does, so queueing delay and tail latency show up. Closed-loop hides queueing (coordinated
omission) but is useful for peak-throughput numbers.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from typing import List, Optional

import httpx

from .metrics import RequestRecord, summarize

WORDS = ("system latency throughput memory kernel batch token cache model request queue "
         "scheduler device tensor bandwidth compute attention decode prefill network").split()

WORKLOADS = {
    # (prompt_words choices, output_token choices)
    "uniform": ([48], [64]),
    "mixed": ([16, 48, 128, 256], [16, 32, 64, 128, 256]),
    # ~12% of prompts are ~1.3k tokens: long prefills that stall decoding unless prefill is chunked
    "longmix": ([16, 16, 16, 32, 32, 48, 48, 1000], [64, 128]),
}

# "shared": every prompt starts with the same ~250-token system-style preamble (RAG / agent / few-shot pattern),
# followed by a short unique question. This is the workload prefix caching targets.
_prefix_rng = random.Random(2024)          # ONE generator, so the deterministic preamble is varied text
SHARED_PREFIX = "You are a careful assistant for a systems team. Background notes: " + " ".join(
    _prefix_rng.choice(WORDS) for _ in range(200)) + "."


def make_prompt(rnd: random.Random, n_words: int) -> str:
    body = " ".join(rnd.choice(WORDS) for _ in range(n_words))
    return f"Continue this text in a natural way: {body}"


def sample_request(rnd: random.Random, workload: str):
    if workload == "shared":
        question = " ".join(rnd.choice(WORDS) for _ in range(rnd.choice([8, 16, 24])))
        return f"{SHARED_PREFIX}\n\nQuestion: {question}", rnd.choice([32, 64])
    pw, ot = WORKLOADS[workload]
    return make_prompt(rnd, rnd.choice(pw)), rnd.choice(ot)


async def _one_request(client: httpx.AsyncClient, url: str, model: str, prompt: str, max_tokens: int,
                       t_run0: float, timeout: float, ignore_eos: bool = True) -> RequestRecord:
    t0 = time.perf_counter()
    rec = RequestRecord(t_arrival=t0 - t_run0, ok=False)
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "stream": True, "ignore_eos": ignore_eos, "temperature": 0,
            "stream_options": {"include_usage": True}}
    chunks = 0
    try:
        async with client.stream("POST", url, json=body, timeout=timeout) as resp:
            if resp.status_code != 200:
                rec.error = f"http {resp.status_code}"
                return rec
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                if data.get("usage"):
                    rec.prompt_tokens = data["usage"].get("prompt_tokens", 0)
                    rec.output_tokens = data["usage"].get("completion_tokens", 0)
                if data.get("error"):
                    rec.error = str(data["error"].get("type", "stream_error"))
                    return rec
                if data.get("usage") and data["usage"].get("engine_ttft_s") is not None:
                    rec.engine_ttft = data["usage"]["engine_ttft_s"]
                choices = data.get("choices") or []
                if choices:
                    # Client-visible TTFT = first chunk that carries non-empty text. Role-only / empty-content
                    # chunks (vLLM sends one first) and finish-only chunks are not generated text.
                    text = choices[0].get("delta", {}).get("content")
                    if text:
                        chunks += 1
                        if rec.ttft is None:
                            rec.ttft = time.perf_counter() - t0
        rec.e2e = time.perf_counter() - t0
        if rec.output_tokens == 0:
            rec.output_tokens = chunks
        rec.ok = rec.ttft is not None
    except Exception as exc:  # network error, timeout, ...
        rec.error = type(exc).__name__
    return rec


async def run_open_loop(url: str, model: str, rate: float, n_requests: int, workload: str = "mixed",
                        seed: int = 0, timeout: float = 120.0, max_drain_s: float = 120.0,
                        slo_ttft: float = 1.0, slo_tpot: float = 0.1):
    """Poisson arrivals at `rate` req/s. Requests still running `max_drain_s` after the last
    arrival are cancelled and counted as failed (the server is overloaded)."""
    rnd = random.Random(seed)
    limits = httpx.Limits(max_connections=4096, max_keepalive_connections=4096)
    async with httpx.AsyncClient(limits=limits) as client:
        t_run0 = time.perf_counter()
        tasks: List[asyncio.Task] = []
        t_next = 0.0
        for _ in range(n_requests):
            t_next += rnd.expovariate(rate)
            prompt, max_tokens = sample_request(rnd, workload)
            delay = t_next - (time.perf_counter() - t_run0)
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(
                _one_request(client, url, model, prompt, max_tokens, t_run0, timeout)))
        done, pending = await asyncio.wait(tasks, timeout=max_drain_s)
        for t in pending:
            t.cancel()
        records = []
        for t in tasks:
            if t in done and not t.cancelled() and t.exception() is None:
                records.append(t.result())
            else:
                records.append(RequestRecord(t_arrival=0.0, ok=False, error="cancelled_overload"))
        wall = time.perf_counter() - t_run0
    return records, summarize(records, wall, offered_rps=rate, slo_ttft=slo_ttft, slo_tpot=slo_tpot)


async def run_closed_loop(url: str, model: str, concurrency: int, n_requests: int,
                          workload: str = "mixed", seed: int = 0, timeout: float = 120.0,
                          slo_ttft: float = 1.0, slo_tpot: float = 0.1):
    rnd = random.Random(seed)
    work = [sample_request(rnd, workload) for _ in range(n_requests)]
    it = iter(work)
    records: List[RequestRecord] = []
    limits = httpx.Limits(max_connections=4096, max_keepalive_connections=4096)
    async with httpx.AsyncClient(limits=limits) as client:
        t_run0 = time.perf_counter()

        async def worker():
            for prompt, max_tokens in it:  # shared iterator -> work stealing
                records.append(await _one_request(client, url, model, prompt, max_tokens, t_run0, timeout))

        await asyncio.gather(*[worker() for _ in range(concurrency)])
        wall = time.perf_counter() - t_run0
    return records, summarize(records, wall, offered_rps=None, slo_ttft=slo_ttft, slo_tpot=slo_tpot)


async def warmup(url: str, model: str, n: int = 4, max_tokens: int = 8):
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        rnd = random.Random(123)
        await asyncio.gather(*[
            _one_request(client, url, model, make_prompt(rnd, 16), max_tokens, t0, 120.0)
            for _ in range(n)])
