"""Extra slot-engine checks: context-bucket crossings, long generation, slot reuse, eviction exactness, cancel mid-prefill."""
import random

import torch

from llmserve.engine import GenRequest
from llmserve.models import build_tiny_model
from llmserve.slot_engine import SlotEngine
from test_engine import reference_greedy


def reqs(seed, n, plen=(3, 20), new=(1, 12), vocab=500):
    rnd = random.Random(seed)
    return [GenRequest(prompt_ids=[rnd.randrange(3, vocab) for _ in range(rnd.randint(*plen))],
                       max_new_tokens=rnd.randint(*new), ignore_eos=True) for _ in range(n)]


def serve(model, rs, **kw):
    eng = SlotEngine(model, device="cpu", **kw)
    for r in rs:
        eng.submit(r)
    eng.run_until_idle()
    return eng


def test_decode_crosses_every_context_bucket_boundary():
    model = build_tiny_model().double()
    buckets = [8, 16, 32, 64, 128, 160]
    rs = [GenRequest(prompt_ids=[5 + i, 9, 11], max_new_tokens=150, ignore_eos=True) for i in range(3)]
    eng = SlotEngine(model, device="cpu", max_batch=4, max_ctx=160, ctx_buckets=buckets)
    used, orig = set(), eng._decode_core
    eng._decode_core = lambda Sb, T: (used.add(T), orig(Sb, T))[1]      # record the context bucket of every step
    for r in rs:
        eng.submit(r)
    eng.run_until_idle()
    assert used == set(buckets), f"expected every bucket to be exercised, got {sorted(used)}"   # 3 + 150 tokens -> 160
    for r in rs:
        assert r.out_ids == reference_greedy(model, r.prompt_ids, r.max_new_tokens)


def test_long_generation_with_many_slot_reuses():
    model = build_tiny_model().double()
    rs = reqs(21, 20, plen=(3, 12), new=(30, 60))
    eng = serve(model, rs, max_batch=3, max_ctx=128, ctx_buckets=[32, 64, 128])
    for r in rs:
        assert r.out_ids == reference_greedy(model, r.prompt_ids, r.max_new_tokens)
    assert len(eng.free) == 3


def test_outputs_stay_exact_across_prefix_hits_and_real_evictions():
    model = build_tiny_model().double()
    rnd = random.Random(5)
    prefixes = [[rnd.randrange(3, 500) for _ in range(24)] for _ in range(6)]
    rs = [GenRequest(prompt_ids=prefixes[(i // 3) % 6] + [rnd.randrange(3, 500) for _ in range(rnd.randint(2, 6))],
                     max_new_tokens=5, ignore_eos=True) for i in range(36)]
    eng = SlotEngine(model, device="cpu", max_batch=2, max_ctx=128, ctx_buckets=[64, 128],
                     prefix_cache=True, prefix_block=8)
    eng.pcache.max_bytes = 70_000        # one node is ~30 KB here: room for ~2 of the 6 prefixes; groups of 3 requests share one -> hits and evictions
    for r in rs:
        eng.submit(r)
    eng.run_until_idle()
    st = eng.pcache.stats()
    assert st["prefix_evictions"] > 0, "the byte budget never forced an eviction"
    assert st["prefix_hit_requests"] > 0, "no request ever hit the cache"
    assert eng.pcache.bytes <= eng.pcache.max_bytes
    for r in rs:                          # every request, whether it hit, missed or followed an eviction
        assert r.out_ids == reference_greedy(model, r.prompt_ids, r.max_new_tokens)


def test_cancel_and_expire_during_chunked_prefill_free_everything():
    model = build_tiny_model().double()
    long_ids = [random.Random(3).randrange(3, 500) for _ in range(96)]
    eng = SlotEngine(model, device="cpu", max_batch=2, max_ctx=160, ctx_buckets=[160], chunk_size=8)
    a = GenRequest(prompt_ids=long_ids, max_new_tokens=5, ignore_eos=True)
    b = GenRequest(prompt_ids=[4, 5, 6], max_new_tokens=6, ignore_eos=True)
    eng.submit(a); eng.submit(b)
    eng.tick(); eng.tick()
    assert eng._pf is not None and eng._pf["req"] is a        # a is mid-prefill
    eng.cancel(a)
    eng.run_until_idle()
    assert a.finish_reason == "cancelled" and a.out_ids == []
    assert b.out_ids == reference_greedy(model, b.prompt_ids, 6)
    assert eng._pf is None and len(eng.free) == 2 and not eng.slots

    c = GenRequest(prompt_ids=long_ids, max_new_tokens=5, ignore_eos=True, deadline_s=0.0)
    eng.submit(c)
    eng.tick()                                                  # starts prefill
    eng.run_until_idle()
    assert c.finish_reason == "timeout" and len(eng.free) == 2 and eng._pf is None


def test_step_log_is_bounded():
    model = build_tiny_model().double()
    eng = SlotEngine(model, device="cpu", max_batch=2, max_ctx=64, ctx_buckets=[64])
    assert eng.step_log.maxlen is not None
