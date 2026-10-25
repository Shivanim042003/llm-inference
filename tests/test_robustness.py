"""Streaming correctness, cancellation, deadlines, overload, failure propagation, validation, shutdown."""
import concurrent.futures as cf
import json
import socket
import threading
import time

import httpx
import pytest
import torch
import uvicorn

from llmserve.engine import Engine, EngineClosed, GenRequest, Overloaded
from llmserve.models import TINY_ID, build_tiny_model
from llmserve.server import build_app
from llmserve.slot_engine import SlotEngine

KINDS = ["dynamic", "slot"]


def make_engine(kind, model=None, **kw):
    model = model or build_tiny_model().double()
    if kind == "slot":
        return SlotEngine(model, device="cpu", max_ctx=128, ctx_buckets=[32, 128], **kw)
    return Engine(model, device="cpu", max_ctx=128, **kw)


def req(n_prompt=6, new=40, **kw):
    return GenRequest(prompt_ids=list(range(3, 3 + n_prompt)), max_new_tokens=new, ignore_eos=True, **kw)


@pytest.mark.parametrize("kind", KINDS)
def test_cancel_active_and_pending_frees_resources(kind):
    eng = make_engine(kind, max_batch=2)
    a, b, c = req(), req(), req()
    for r in (a, b, c):
        eng.submit(r)
    eng.tick(); eng.tick()
    assert len(eng.active) == 2 and len(eng.pending) + (1 if getattr(eng, "_pf", None) else 0) >= 1
    eng.cancel(a)            # active
    eng.cancel(c)            # pending
    eng.run_until_idle()
    assert a.finish_reason == "cancelled" and c.finish_reason == "cancelled"
    assert b.finish_reason == "length" and len(b.out_ids) == 40
    assert not eng.active and not eng.pending
    if kind == "slot":
        assert len(eng.free) == eng.S and not eng.slots
    assert eng.cancelled_count == 2


@pytest.mark.parametrize("kind", KINDS)
def test_deadline_removes_expired_requests(kind):
    eng = make_engine(kind, max_batch=2)
    slow = req(new=100, deadline_s=0.05)
    ok = req(new=5)
    eng.submit(slow); eng.submit(ok)
    eng.tick()
    time.sleep(0.08)
    eng.run_until_idle()
    assert slow.finish_reason == "timeout" and len(slow.out_ids) < 100
    assert ok.finish_reason == "length"
    assert eng.expired_count == 1 and not eng.active


@pytest.mark.parametrize("kind", KINDS)
def test_bounded_queue_and_token_budget(kind):
    eng = make_engine(kind, max_batch=1, max_pending=2)
    eng.submit(req()); eng.submit(req())
    with pytest.raises(Overloaded):
        eng.submit(req())
    assert eng.rejected == 1
    eng2 = make_engine(kind, max_batch=1, max_queued_tokens=100)
    eng2.submit(req(n_prompt=10, new=40))     # 50 tokens
    eng2.submit(req(n_prompt=10, new=30))     # 90 tokens
    with pytest.raises(Overloaded):
        eng2.submit(req(n_prompt=10, new=30))  # would be 130 > 100


@pytest.mark.parametrize("kind", KINDS)
def test_engine_failure_notifies_every_client_and_blocks_new_work(kind):
    eng = make_engine(kind, max_batch=2)
    events = []
    reqs = [req(new=20, on_token=lambda t, f, i=i: events.append((i, t, f))) for i in range(4)]
    for r in reqs:
        eng.submit(r)
    orig = eng._decode_step
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("synthetic CUDA error")
        return orig()
    eng._decode_step = boom
    eng.start()
    deadline = time.time() + 10
    while any(r.t_done is None for r in reqs) and time.time() < deadline:
        time.sleep(0.02)
    assert all(r.finish_reason in ("error", "length") for r in reqs)
    assert sum(r.finish_reason == "error" for r in reqs) >= 1
    assert eng.failed and "synthetic" in eng.failed and not eng.alive()
    with pytest.raises(EngineClosed):
        eng.submit(req())
    finals = {i for i, t, f in events if f}
    assert finals == {0, 1, 2, 3}          # every client got a terminal event


@pytest.mark.parametrize("kind", KINDS)
def test_shutdown_drains_then_fails_remaining(kind):
    eng = make_engine(kind, max_batch=1)
    quick, slow = req(new=3), req(new=100)
    eng.start()
    eng.submit(quick); eng.submit(slow)
    eng.stop(drain_s=0.0)
    assert quick.finish_reason in ("length", "shutdown") and slow.finish_reason == "shutdown"
    with pytest.raises(EngineClosed):
        eng.submit(req())


# ----------------------------------------------------------------------------- HTTP level
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def srv():
    app = build_app(TINY_ID, mode="continuous", max_batch=1, dtype="float32", max_ctx=256, max_pending=2,
                    request_timeout=30.0)
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(base + "/health", timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    yield base
    server.should_exit = True
    th.join(timeout=10)


def post(base, **body):
    body.setdefault("messages", [{"role": "user", "content": "hello there"}])
    return httpx.post(base + "/v1/chat/completions", json=body, timeout=30)


def test_health_and_ready(srv):
    assert httpx.get(srv + "/health").json()["worker_alive"] is True
    assert httpx.get(srv + "/ready").status_code == 200


@pytest.mark.parametrize("body,code", [
    ({"messages": []}, "messages must be"),
    ({"messages": [{"role": "robot", "content": "x"}]}, "role"),
    ({"messages": [{"role": "user", "content": 5}]}, "string content"),
    ({"max_tokens": 0}, "max_tokens"),
    ({"max_tokens": 5000}, "max_tokens"),
    ({"temperature": 0.8}, "greedy"),
    ({"n": 2}, "n=1"),
    ({"stop": ["x"]}, "unsupported parameter"),
    ({"logprobs": True}, "unsupported parameter"),
    ({"timeout": 9999}, "timeout"),
    ({"max_tokens": 250}, "context limit"),     # prompt + max_tokens > max_ctx=256 -> rejected, not truncated
])
def test_validation_rejects_instead_of_ignoring(srv, body, code):
    body = dict(body)
    if body.get("max_tokens") == 250:
        body["messages"] = [{"role": "user", "content": "y" * 100}]
    r = post(srv, **body)
    assert r.status_code == 400, r.text
    assert code in r.json()["error"]["message"]


def sse(base, **body):
    body.setdefault("messages", [{"role": "user", "content": "stream me"}])
    body["stream"] = True
    return httpx.stream("POST", base + "/v1/chat/completions", json=body, timeout=30)


def test_slow_streaming_consumer_gets_exact_content_without_duplication(srv):
    kw = dict(max_tokens=40, ignore_eos=True, messages=[{"role": "user", "content": "slow consumer test"}])
    ref = post(srv, **kw).json()["choices"][0]["message"]["content"]
    text, chunks, usage, finish = "", 0, None, None
    with sse(srv, **kw) as r:
        for line in r.iter_lines():
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            time.sleep(0.03)                     # consumer far slower than the engine
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                piece = c["delta"].get("content")
                if piece is not None:
                    assert piece != "", "empty content chunks must not be sent"
                    text += piece; chunks += 1
                finish = c.get("finish_reason") or finish
    assert text == ref and finish == "length"
    assert usage["completion_tokens"] == 40 and "engine_ttft_s" in usage


def test_client_disconnect_cancels_generation(srv):
    with sse(srv, max_tokens=200, ignore_eos=True) as r:
        it = r.iter_lines()
        for _ in range(3):
            next(it)
    # leaving the context closes the connection; the engine must stop and free its slot
    deadline = time.time() + 10
    while time.time() < deadline:
        st = httpx.get(srv + "/stats").json()
        if st["active"] == 0 and st["pending"] == 0:
            break
        time.sleep(0.1)
    assert st["active"] == 0 and st["cancelled"] >= 1


def test_overload_returns_503_with_retry_after(srv):
    def one(_):
        return post(srv, max_tokens=150, ignore_eos=True)
    with cf.ThreadPoolExecutor(10) as ex:
        res = list(ex.map(one, range(10)))
    codes = [r.status_code for r in res]
    assert 503 in codes and 200 in codes
    r503 = next(r for r in res if r.status_code == 503)
    assert r503.headers.get("retry-after") == "1" and r503.json()["error"]["type"] == "overloaded"
    assert httpx.get(srv + "/stats").json()["rejected"] >= 1


def test_per_request_deadline_returns_504(srv):
    time.sleep(1.0)
    r = post(srv, max_tokens=250 - 40, ignore_eos=True, timeout=0.05,
             messages=[{"role": "user", "content": "deadline"}])
    assert r.status_code == 504 and r.json()["error"]["type"] == "timeout"


def test_failed_engine_makes_health_unhealthy_and_clients_get_errors():
    app = build_app(TINY_ID, mode="continuous", max_batch=2, dtype="float32", max_ctx=256)
    from llmserve.server import STATE
    eng = STATE["engine"]
    orig = eng._decode_step
    eng._decode_step = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True); th.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(base + "/health", timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    try:
        r = post(base, max_tokens=5)
        assert r.status_code == 500 and r.json()["error"]["type"] == "error"
        h = httpx.get(base + "/health")
        assert h.status_code == 503 and "boom" in h.json()["reason"]
        assert post(base, max_tokens=5).status_code == 503
    finally:
        server.should_exit = True
        th.join(timeout=10)
