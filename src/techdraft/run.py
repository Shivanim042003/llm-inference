"""Send prepared TechQA questions to any OpenAI-compatible backend (open-loop Poisson arrivals), then score and summarise."""
from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path

import httpx

from supportdraft.cost import serving_summary

from .data import KB, gold_docs, split_of
from .pipeline import build_messages, parse
from .score import score_item, summarize

RAW = Path("data/techdraft/raw")          # local only: contains model drafts derived from licensed text


MAX_PROMPT_TOKENS = 3796          # server limit 4096 minus the 300 generated tokens
TOKENIZER_ID = "Qwen/Qwen2.5-1.5B-Instruct"
EXCLUDED = []                      # ids of questions dropped because the prompt cannot fit the server limit (filled by prepare_items)
_TOK = {}


def _prompt_tokens(messages):
    if "tok" not in _TOK:
        try:
            from transformers import AutoTokenizer
            _TOK["tok"] = AutoTokenizer.from_pretrained(TOKENIZER_ID)
        except Exception:
            _TOK["tok"] = None
    t = _TOK["tok"]
    if t is None:
        return None
    return len(t(t.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)).input_ids)


def prepare_items(records, kb, k=4, max_prompt_tokens=MAX_PROMPT_TOKENS):
    """Retrieval and prompt construction happen once, before timing, so client CPU does not distort latency.
    Questions whose prompt would exceed the server context limit are excluded (recorded in EXCLUDED), not truncated."""
    items = []
    EXCLUDED.clear()
    for r in records:
        ret = kb.retrieve(r["question"], k)
        shown = [kb.by_id[c] for c, _ in ret]
        msgs = build_messages(r["question"], shown)
        n = _prompt_tokens(msgs) if max_prompt_tokens else None
        if n is not None and n > max_prompt_tokens:
            EXCLUDED.append(r["id"])
            continue
        items.append(dict(id=r["id"], split=split_of(r["id"]), impossible=r["is_impossible"], gold_answer=r["answer"], gold_docs=gold_docs(r),
                          retrieved=ret, messages=msgs))
    return items


async def _one(client, url, model, item, max_tokens, t_run0, timeout):
    t0 = time.perf_counter()
    rec = dict(item_id=item["id"], draft="", ok=False, error=None, ttft_s=None, e2e_s=None, prompt_tokens=None, output_tokens=None,
               t_arrival=t0 - t_run0)
    body = {"model": model, "messages": item["messages"], "max_tokens": max_tokens, "stream": True, "temperature": 0,
            "stream_options": {"include_usage": True}}
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
                            rec["ttft_s"] = time.perf_counter() - t0
                        parts.append(text)
        rec["e2e_s"] = time.perf_counter() - t0
        rec["draft"] = "".join(parts)
        rec["ok"] = rec["ttft_s"] is not None
        if not rec["ok"]:
            rec["error"] = "empty_response"
    except Exception as exc:
        rec["error"] = type(exc).__name__
    return rec


async def _spike(url, model, items, rate, seed, repeats, max_tokens, timeout):
    rnd = random.Random(seed)
    work = [it for _ in range(repeats) for it in rnd.sample(items, len(items))]
    t_run0 = time.perf_counter()
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=1000, max_keepalive_connections=1000)) as client:
        tasks, at = [], 0.0
        for it in work:
            at += rnd.expovariate(rate)
            await asyncio.sleep(max(0.0, t_run0 + at - time.perf_counter()))
            tasks.append(asyncio.create_task(_one(client, url, model, it, max_tokens, t_run0, timeout)))
        recs = await asyncio.gather(*tasks)
    return list(recs), time.perf_counter() - t_run0


def evaluate(url, model, items, kb, rate, e2e_slo_s, ttft_slo_s, seed=0, min_seconds=30.0, max_items=None, max_tokens=300, timeout=120.0, k=4):
    if max_items and len(items) > max_items:
        items = random.Random(seed).sample(items, max_items)
    by = {it["id"]: it for it in items}
    repeats = max(1, -(-int(min_seconds * rate) // len(items))) if min_seconds else 1
    recs, wall = asyncio.run(_spike(url, model, items, rate, seed, repeats, max_tokens, timeout))
    for r in recs:
        it = by[r["item_id"]]
        shown = [kb.by_id[c] for c, _ in it["retrieved"][:k]]
        parsed = parse(r["draft"]) if r["ok"] else dict(decision="UNPARSED", quotes=[], answer="")
        r["score"] = score_item(it, parsed, shown)
        r["correct"] = bool(r["ok"] and r["score"]["passed"])
    serving = serving_summary(recs, wall, e2e_slo_s, ttft_slo_s)
    served = [r["score"] for r in recs if r["ok"]]
    quality = summarize(served) if served else {}
    quality["requests_failed"] = len(recs) - len(served)
    return recs, dict(offered_rps=rate, repeats=repeats, n_unique=len(items), **serving), quality


def save(recs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
