"""OpenAI-compatible chat-completions server on top of the batching Engine.

    python -m llmserve.server --model Qwen/Qwen2.5-0.5B-Instruct --mode continuous --max-batch 16
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .engine import Engine, EngineClosed, GenRequest, Overloaded
from .slot_engine import SlotEngine
from .models import load_model, model_footprint

STATE = {}


def build_app(model_id, mode="continuous", max_batch=16, dtype="float16", quant="none",
              max_prefill_per_step=None, max_ctx=4096, last_logits_only=True,
              engine_kind="dynamic", cuda_graphs=False, chunk_size=0, prefix_cache=False,
              max_pending=256, max_queued_tokens=None, request_timeout=120.0, max_tokens_limit=1024,
              shutdown_drain_s=10.0):
    model, tok, eos_ids, device = load_model(model_id, dtype=dtype, quant=quant)
    if mode == "sequential":
        max_batch, policy = 1, "continuous"
    elif mode == "static":
        policy = "static"
    elif mode == "continuous":
        policy = "continuous"
    else:
        raise ValueError(mode)
    if engine_kind == "slot":
        if quant != "none":
            raise ValueError("the slot engine drives the weights directly; use --quant none")
        engine = SlotEngine(model, eos_ids=eos_ids, device=device, max_batch=max_batch, policy=policy,
                            max_prefill_per_step=max_prefill_per_step, max_ctx=max_ctx,
                            last_logits_only=last_logits_only, cuda_graphs=cuda_graphs,
                            chunk_size=chunk_size, prefix_cache=prefix_cache,
                            max_pending=max_pending, max_queued_tokens=max_queued_tokens)
    else:
        engine = Engine(model, eos_ids=eos_ids, device=device, max_batch=max_batch, policy=policy,
                        max_prefill_per_step=2 if max_prefill_per_step is None else max_prefill_per_step,
                        max_ctx=max_ctx, last_logits_only=last_logits_only,
                        max_pending=max_pending, max_queued_tokens=max_queued_tokens)
    footprint = model_footprint(model)
    STATE.update(engine=engine, tok=tok, model_id=model_id, mode=mode, dtype=dtype, quant=quant,
                 device=device, footprint=footprint, draining=False)
    @asynccontextmanager
    async def lifespan(_app):
        engine.start()
        yield
        STATE["draining"] = True
        await asyncio.get_running_loop().run_in_executor(None, engine.stop, shutdown_drain_s)

    app = FastAPI(title="llmserve", lifespan=lifespan)

    def err(status, message, etype="invalid_request_error", code=None, headers=None):
        return JSONResponse({"error": {"message": message, "type": etype, "code": code or etype}},
                            status_code=status, headers=headers)

    def health_body():
        ok = engine.alive() and not STATE.get("draining")
        body = {"status": "healthy" if ok else "unhealthy", "model": model_id, "mode": mode, "device": device,
                "dtype": dtype, "quant": quant, "max_batch": engine.max_batch, "max_ctx": max_ctx,
                "last_logits_only": last_logits_only, "engine": engine_kind,
                "worker_alive": engine.alive()}
        if engine.failed:
            body["reason"] = engine.failed
        elif STATE.get("draining"):
            body["reason"] = "shutting down"
        return ok, body

    @app.get("/health")
    def health():
        ok, body = health_body()
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/ready")
    def ready():
        ok, body = health_body()
        ok = ok and engine.accepting
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/stats")
    def stats():
        out = dict(engine.stats(), **footprint)
        try:
            import torch
            if device == "cuda":
                out["gpu_mem_allocated_mb"] = torch.cuda.memory_allocated() / 2**20
                out["gpu_mem_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
        except Exception:
            pass
        return out

    @app.get("/engine_stats")
    def engine_stats(reset: int = 0):
        """Aggregate scheduler statistics from the engine step log (occupancy, prefill/decode split)."""
        log = list(engine.step_log)
        dec = [s for s in log if s[1] == "decode"]
        pre = [s for s in log if s[1] == "prefill"]
        out = {
            "decode_steps": len(dec),
            "mean_decode_batch": (sum(s[2] for s in dec) / len(dec)) if dec else 0.0,
            "mean_decode_ms": (1e3 * sum(s[3] for s in dec) / len(dec)) if dec else 0.0,
            "prefills": len(pre),
            "prefill_s": sum(s[3] for s in pre),
            "decode_s": sum(s[3] for s in dec),
        }
        tot = out["prefill_s"] + out["decode_s"]
        out["prefill_time_share"] = out["prefill_s"] / tot if tot else 0.0
        out.update({k: v for k, v in engine.stats().items() if k.startswith("prefix_")})
        if hasattr(engine, "pcache") and engine.pcache is not None and reset:
            engine.pcache.lookups = engine.pcache.hit_requests = engine.pcache.hit_tokens = engine.pcache.prompt_tokens = 0
        if reset:
            engine.step_log.clear()
        return out

    @app.get("/v1/models")
    def models():
        return {"data": [{"id": model_id, "object": "model"}]}

    ALLOWED = {"model", "messages", "max_tokens", "max_completion_tokens", "stream", "stream_options",
               "temperature", "top_p", "n", "ignore_eos", "timeout", "user"}

    def validate(payload):
        """Return (error_response | None, parsed dict). Unsupported options are rejected, never silently ignored."""
        if not isinstance(payload, dict):
            return err(400, "request body must be a JSON object"), None
        unknown = sorted(set(payload) - ALLOWED)
        if unknown:
            return err(400, f"unsupported parameter(s): {', '.join(unknown)}", code="unsupported_parameter"), None
        if payload.get("n", 1) != 1:
            return err(400, "only n=1 is supported", code="unsupported_parameter"), None
        if payload.get("temperature", 0) not in (0, 0.0, None):
            return err(400, "only greedy decoding (temperature=0) is supported", code="unsupported_parameter"), None
        if payload.get("top_p", 1) not in (1, 1.0, None):
            return err(400, "top_p is not supported (greedy decoding only)", code="unsupported_parameter"), None
        msgs = payload.get("messages")
        if not isinstance(msgs, list) or not msgs:
            return err(400, "messages must be a non-empty list"), None
        for m in msgs:
            if (not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant")
                    or not isinstance(m.get("content"), str)):
                return err(400, "each message needs role in {system,user,assistant} and a string content"), None
        mt = payload.get("max_tokens", payload.get("max_completion_tokens", 64))
        if isinstance(mt, bool) or not isinstance(mt, int) or not 1 <= mt <= max_tokens_limit:
            return err(400, f"max_tokens must be an integer in [1, {max_tokens_limit}]"), None
        to = payload.get("timeout", request_timeout)
        if isinstance(to, bool) or not isinstance(to, (int, float)) or not 0 < to <= request_timeout:
            return err(400, f"timeout must be a number in (0, {request_timeout}] seconds"), None
        for key in ("stream", "ignore_eos"):
            if key in payload and not isinstance(payload[key], bool):
                return err(400, f"{key} must be a boolean"), None
        return None, {"messages": msgs, "max_tokens": mt, "timeout": float(to),
                      "stream": bool(payload.get("stream", False)), "ignore_eos": bool(payload.get("ignore_eos", False))}

    FAILURE_HTTP = {"timeout": (504, "request exceeded its deadline"), "error": (500, "engine failure"),
                    "shutdown": (503, "server is shutting down"), "cancelled": (499, "request cancelled")}

    @app.post("/v1/chat/completions")
    async def chat(payload: dict, request: Request):
        bad, p = validate(payload)
        if bad:
            return bad
        prompt_ids = tok.encode_chat(p["messages"])
        if len(prompt_ids) + p["max_tokens"] > max_ctx:
            return err(400, f"prompt ({len(prompt_ids)} tokens) + max_tokens ({p['max_tokens']}) exceeds the context "
                            f"limit of {max_ctx}; shorten the input (the server never truncates silently)",
                       code="context_length_exceeded")
        loop = asyncio.get_running_loop()
        # Per-request streaming buffer. A request emits at most max_tokens token events plus one terminal event, so
        # this bound can never be hit by a healthy request; if it is, the client is cancelled instead of buffering.
        queue: asyncio.Queue = asyncio.Queue(maxsize=p["max_tokens"] + 4)

        def on_token(token_id, finished):
            # Immutable event: (token id or None, finished, finish reason). The consumer never reads req.out_ids,
            # so a slow client cannot see duplicated, merged or empty content.
            ev = (token_id, finished, req.finish_reason if finished else None)

            def put():
                try:
                    queue.put_nowait(ev)
                except asyncio.QueueFull:      # slow-client policy: stop generating for it, never buffer without bound
                    engine.cancel(req)
            loop.call_soon_threadsafe(put)

        req = GenRequest(prompt_ids=prompt_ids, max_new_tokens=p["max_tokens"], ignore_eos=p["ignore_eos"],
                         on_token=on_token, deadline_s=p["timeout"])
        try:
            engine.submit(req)
        except Overloaded as exc:
            return err(503, f"server overloaded: {exc}", etype="overloaded", headers={"Retry-After": "1"})
        except EngineClosed as exc:
            return err(503, f"server unavailable: {exc}", etype="unavailable")
        cid = "chatcmpl-" + uuid.uuid4().hex[:12]
        created = int(time.time())
        ids = []          # consumer-owned copy of the generated tokens

        async def next_event():
            """Wait for the next engine event; if the client has gone away, cancel the request."""
            while True:
                try:
                    return await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        engine.cancel(req)
                        raise asyncio.CancelledError()

        def usage():
            out = {"prompt_tokens": len(prompt_ids), "completion_tokens": len(ids),
                   "total_tokens": len(prompt_ids) + len(ids)}
            if req.t_first is not None:
                out["engine_ttft_s"] = req.t_first - req.t_submit
            return out

        if not p["stream"]:
            try:
                while True:
                    tok_id, finished, reason = await next_event()
                    if tok_id is not None:
                        ids.append(tok_id)
                    if finished:
                        break
            except asyncio.CancelledError:
                engine.cancel(req)
                raise
            if reason in FAILURE_HTTP:
                status, msg = FAILURE_HTTP[reason]
                return err(status, msg, etype=reason)
            return JSONResponse({
                "id": cid, "object": "chat.completion", "created": created, "model": model_id,
                "choices": [{"index": 0, "finish_reason": reason,
                             "message": {"role": "assistant", "content": tok.decode(ids)}}],
                "usage": usage()})

        async def events():
            sent, finished = "", False
            try:
                while True:
                    tok_id, finished, reason = await next_event()
                    if tok_id is not None:
                        ids.append(tok_id)
                    text = tok.decode(ids)
                    if text.endswith("\ufffd") and not finished:
                        delta = ""   # hold back an incomplete multibyte sequence
                    else:
                        delta, sent = text[len(sent):], text
                    if delta:
                        yield "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created,
                                                     "model": model_id, "choices": [{"index": 0, "delta": {"content": delta},
                                                                                    "finish_reason": None}]}) + "\n\n"
                    if finished:
                        break
                if reason in FAILURE_HTTP:
                    yield "data: " + json.dumps({"error": {"message": FAILURE_HTTP[reason][1], "type": reason}}) + "\n\n"
                else:
                    yield "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created,
                                                 "model": model_id, "choices": [{"index": 0, "delta": {},
                                                                                "finish_reason": reason}]}) + "\n\n"
                    yield "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created,
                                                 "model": model_id, "choices": [], "usage": usage()}) + "\n\n"
                yield "data: [DONE]\n\n"
            finally:
                if not finished:          # client went away mid-stream: stop generating, free the KV slot
                    engine.cancel(req)

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def main():
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--mode", default="continuous", choices=["sequential", "static", "continuous"])
    p.add_argument("--max-batch", type=int, default=16)
    p.add_argument("--max-prefill-per-step", type=int, default=None)
    p.add_argument("--engine", default="dynamic", choices=["dynamic", "slot"],
                   help="dynamic: left-padded growing KV cache (reference); slot: static KV slots, fixed-shape decode")
    p.add_argument("--cuda-graphs", action="store_true", help="slot engine: capture decode steps as CUDA graphs")
    p.add_argument("--chunk-size", type=int, default=0, help="slot engine: prefill chunk size in tokens (0 = whole prompt)")
    p.add_argument("--prefix-cache", action="store_true", help="slot engine: reuse KV of shared prompt prefixes")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--quant", default="none", choices=["none", "int8", "nf4"])
    p.add_argument("--max-ctx", type=int, default=4096)
    p.add_argument("--full-logits", action="store_true",
                   help="ablation: compute full-vocab logits for every prompt position during prefill")
    p.add_argument("--max-pending", type=int, default=256, help="queue bound; further requests get HTTP 503")
    p.add_argument("--max-queued-tokens", type=int, default=None,
                   help="admission budget over (prompt + max_tokens) of all unfinished requests; beyond it -> 503")
    p.add_argument("--request-timeout", type=float, default=120.0, help="per-request deadline cap in seconds")
    p.add_argument("--shutdown-drain-s", type=float, default=10.0, help="finish in-flight work this long on shutdown")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    app = build_app(a.model, a.mode, a.max_batch, a.dtype, a.quant, a.max_prefill_per_step, a.max_ctx,
                     last_logits_only=not a.full_logits, engine_kind=a.engine, cuda_graphs=a.cuda_graphs,
                     chunk_size=a.chunk_size, prefix_cache=a.prefix_cache, max_pending=a.max_pending,
                     max_queued_tokens=a.max_queued_tokens, request_timeout=a.request_timeout,
                     shutdown_drain_s=a.shutdown_drain_s)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning",
                timeout_graceful_shutdown=int(a.shutdown_drain_s) + 5)


if __name__ == "__main__":
    main()
