"""End-to-end: real server process path (in-process via TestClient-like uvicorn thread) + load generator."""
import asyncio
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from llmserve.loadgen import run_closed_loop, run_open_loop, warmup
from llmserve.models import TINY_ID
from llmserve.server import build_app


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module", params=["sequential", "static", "continuous", "slot", "slot-chunked-prefix"])
def server(request):
    kw = {}
    mode = request.param
    if mode.startswith("slot"):
        kw = dict(engine_kind="slot", max_ctx=1024, chunk_size=16 if "chunked" in mode else 0,
                  prefix_cache="prefix" in mode)
        mode = "continuous"
    app = build_app(TINY_ID, mode=mode, max_batch=8, dtype="float32", **kw)
    port = free_port()
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    srv = uvicorn.Server(cfg)
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    yield f"http://127.0.0.1:{port}", request.param
    srv.should_exit = True
    th.join(timeout=5)


def test_non_streaming_completion(server):
    base, _ = server
    r = httpx.post(f"{base}/v1/chat/completions", timeout=30, json={
        "messages": [{"role": "user", "content": "hello"}], "max_tokens": 7, "ignore_eos": True})
    body = r.json()
    assert r.status_code == 200
    assert body["usage"]["completion_tokens"] == 7
    assert body["choices"][0]["finish_reason"] == "length"


def test_open_loop_loadgen(server):
    base, _ = server
    url = f"{base}/v1/chat/completions"
    asyncio.run(warmup(url, TINY_ID, n=2, max_tokens=4))
    recs, summ = asyncio.run(run_open_loop(url, TINY_ID, rate=20, n_requests=15, workload="uniform",
                                           max_drain_s=60))
    assert summ["succeeded"] == 15 and summ["failed"] == 0
    assert all(r.output_tokens == 64 for r in recs)
    assert summ["ttft_p50"] > 0 and summ["output_tok_per_s"] > 0


def test_closed_loop_loadgen(server):
    base, _ = server
    url = f"{base}/v1/chat/completions"
    recs, summ = asyncio.run(run_closed_loop(url, TINY_ID, concurrency=4, n_requests=8, workload="uniform"))
    assert summ["succeeded"] == 8
