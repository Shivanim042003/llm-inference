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

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from .engine import Engine, GenRequest
from .models import load_model, model_footprint

STATE = {}


def build_app(model_id, mode="continuous", max_batch=16, dtype="float16", quant="none",
              max_prefill_per_step=2, max_ctx=4096):
    model, tok, eos_ids, device = load_model(model_id, dtype=dtype, quant=quant)
    if mode == "sequential":
        max_batch, policy = 1, "continuous"
    elif mode == "static":
        policy = "static"
    elif mode == "continuous":
        policy = "continuous"
    else:
        raise ValueError(mode)
    engine = Engine(model, eos_ids=eos_ids, device=device, max_batch=max_batch, policy=policy,
                    max_prefill_per_step=max_prefill_per_step, max_ctx=max_ctx)
    footprint = model_footprint(model)
    STATE.update(engine=engine, tok=tok, model_id=model_id, mode=mode, dtype=dtype, quant=quant,
                 device=device, footprint=footprint)
    @asynccontextmanager
    async def lifespan(_app):
        engine.start()
        yield
        engine.stop()

    app = FastAPI(title="llmserve", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "healthy", "model": model_id, "mode": mode, "device": device,
                "dtype": dtype, "quant": quant, "max_batch": engine.max_batch}

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
        if reset:
            engine.step_log.clear()
        return out

    @app.get("/v1/models")
    def models():
        return {"data": [{"id": model_id, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(payload: dict):
        messages = payload.get("messages") or []
        if not messages:
            raise HTTPException(400, "messages must not be empty")
        max_tokens = max(1, min(int(payload.get("max_tokens", 64)), 1024))
        stream = bool(payload.get("stream", False))
        ignore_eos = bool(payload.get("ignore_eos", False))
        prompt_ids = tok.encode_chat(messages)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def on_token(token_id, finished):
            loop.call_soon_threadsafe(queue.put_nowait, (token_id, finished))

        req = GenRequest(prompt_ids=prompt_ids, max_new_tokens=max_tokens,
                         ignore_eos=ignore_eos, on_token=on_token)
        engine.submit(req)
        cid = "chatcmpl-" + uuid.uuid4().hex[:12]
        created = int(time.time())

        def usage():
            return {"prompt_tokens": len(req.prompt_ids), "completion_tokens": len(req.out_ids),
                    "total_tokens": len(req.prompt_ids) + len(req.out_ids)}

        if not stream:
            while True:
                _, finished = await queue.get()
                if finished:
                    break
            return JSONResponse({
                "id": cid, "object": "chat.completion", "created": created, "model": model_id,
                "choices": [{"index": 0, "finish_reason": req.finish_reason,
                             "message": {"role": "assistant", "content": tok.decode(req.out_ids)}}],
                "usage": usage()})

        async def events():
            sent = ""
            while True:
                _, finished = await queue.get()
                text = tok.decode(req.out_ids)
                if text.endswith("�") and not finished:
                    delta = ""  # hold back incomplete multibyte sequence
                else:
                    delta, sent = text[len(sent):], text
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                         "model": model_id,
                         "choices": [{"index": 0, "delta": {"content": delta},
                                      "finish_reason": req.finish_reason if finished else None}]}
                yield f"data: {json.dumps(chunk)}\n\n"
                if finished:
                    break
            final = {"id": cid, "object": "chat.completion.chunk", "created": created,
                     "model": model_id, "choices": [], "usage": usage()}
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def main():
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--mode", default="continuous", choices=["sequential", "static", "continuous"])
    p.add_argument("--max-batch", type=int, default=16)
    p.add_argument("--max-prefill-per-step", type=int, default=2)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--quant", default="none", choices=["none", "int8", "nf4"])
    p.add_argument("--max-ctx", type=int, default=4096)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    app = build_app(a.model, a.mode, a.max_batch, a.dtype, a.quant, a.max_prefill_per_step, a.max_ctx)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
