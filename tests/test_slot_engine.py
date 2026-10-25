import random

import pytest
import torch

from llmserve.engine import Engine, GenRequest
from llmserve.models import build_tiny_model
from llmserve.slot_engine import PrefixCache, SlotEngine
from test_engine import make_requests, reference_greedy

SMALL = dict(max_ctx=128, ctx_buckets=[16, 32, 64, 128])


def serve(model, reqs, **kw):
    eng = SlotEngine(model, device="cpu", **{**SMALL, **kw})
    for r in reqs:
        eng.submit(r)
    eng.run_until_idle()
    return eng


@pytest.mark.parametrize("kw", [
    dict(max_batch=1), dict(max_batch=4), dict(max_batch=16),
    dict(max_batch=4, policy="static"),
    dict(max_batch=4, chunk_size=5), dict(max_batch=8, chunk_size=1),
    dict(max_batch=4, prefix_cache=True, prefix_block=4),
    dict(max_batch=4, chunk_size=7, prefix_cache=True, prefix_block=4),
])
def test_slot_engine_matches_uncached_reference(kw):
    model = build_tiny_model().double()
    reqs = make_requests(seed=1, n=12)
    eng = serve(model, reqs, **kw)
    for r in reqs:
        assert r.out_ids == reference_greedy(model, r.prompt_ids, r.max_new_tokens), kw
        assert r.finish_reason == "length"
    assert eng.completed == len(reqs) and not eng.slots and len(eng.free) == eng.S


def test_slot_engine_matches_reference_engine_with_gqa_variants():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    for kvh in (1, 2, 4):  # MQA, GQA, MHA
        torch.manual_seed(3)
        cfg = Qwen2Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=kvh, tie_word_embeddings=False)
        model = Qwen2ForCausalLM(cfg).eval().double()
        reqs = make_requests(seed=5, n=6)
        serve(model, reqs, max_batch=4)
        for r in reqs:
            assert r.out_ids == reference_greedy(model, r.prompt_ids, r.max_new_tokens), kvh


def test_slots_are_reused_and_batching_happens():
    model = build_tiny_model().double()
    reqs = make_requests(seed=2, n=14)
    eng = serve(model, reqs, max_batch=3)
    sizes = [s[2] for s in eng.step_log if s[1] == "decode"]
    assert max(sizes) == 3          # more requests than slots -> slots were recycled
    for r in reqs:
        assert r.out_ids == reference_greedy(model, r.prompt_ids, r.max_new_tokens)


def test_eos_in_slot_engine():
    model = build_tiny_model().double()
    prompt = [5, 6, 7, 8]
    ref = reference_greedy(model, prompt, 6)
    eng = SlotEngine(model, eos_ids={ref[0]}, device="cpu", max_batch=2, **SMALL)
    r = GenRequest(prompt_ids=prompt, max_new_tokens=20)
    eng.submit(r); eng.run_until_idle()
    assert r.finish_reason == "stop" and r.out_ids == []
    ks = [i for i in range(1, len(ref)) if ref[i] not in ref[:i]]
    if ks:
        eng = SlotEngine(model, eos_ids={ref[ks[0]]}, device="cpu", max_batch=2, **SMALL)
        r = GenRequest(prompt_ids=prompt, max_new_tokens=20)
        eng.submit(r); eng.run_until_idle()
        assert r.out_ids == ref[:ks[0]] and r.finish_reason == "stop"


def test_chunked_prefill_interleaves_with_decode():
    model = build_tiny_model().double()
    long = GenRequest(prompt_ids=[random.Random(1).randrange(3, 500) for _ in range(60)], max_new_tokens=4, ignore_eos=True)
    short = GenRequest(prompt_ids=[7, 8, 9], max_new_tokens=30, ignore_eos=True)
    eng = SlotEngine(model, device="cpu", max_batch=4, chunk_size=8, max_ctx=128, ctx_buckets=[128])
    eng.submit(short)
    eng.tick()                       # short admitted and decoding
    eng.submit(long)
    seen_decode_during_prefill = False
    while eng.tick():
        if eng._pf is not None and eng.slots:
            seen_decode_during_prefill = True
    assert seen_decode_during_prefill
    assert long.out_ids == reference_greedy(model, long.prompt_ids, 4)
    assert short.out_ids == reference_greedy(model, short.prompt_ids, 30)
    chunks = [s for s in eng.step_log if s[1] == "prefill" and s[4] <= 8]
    assert len(chunks) >= 8


def test_prefix_cache_hits_and_is_exact():
    model = build_tiny_model().double()
    rnd = random.Random(11)
    shared = [rnd.randrange(3, 500) for _ in range(32)]
    reqs = [GenRequest(prompt_ids=shared + [rnd.randrange(3, 500) for _ in range(rnd.randint(2, 10))],
                       max_new_tokens=6, ignore_eos=True) for _ in range(8)]
    eng = serve(model, reqs, max_batch=2, prefix_cache=True, prefix_block=8)
    st = eng.pcache.stats()
    assert st["prefix_hit_requests"] >= 6 and st["prefix_token_hit_rate"] > 0.5
    for r in reqs:
        assert r.out_ids == reference_greedy(model, r.prompt_ids, r.max_new_tokens)
    # prefix hit skips compute: later prefills process fewer tokens than the prompt length
    pf_tokens = [s[4] for s in eng.step_log if s[1] == "prefill"]
    assert min(pf_tokens) < 32


def test_prefix_cache_never_covers_whole_prompt_and_evicts():
    pc = PrefixCache(block=4, max_bytes=10_000)
    kv = [(torch.zeros(1, 1, 16, 4), torch.zeros(1, 1, 16, 4))]
    ids = list(range(16))
    pc.insert(ids, kv)
    t, got = pc.lookup(ids)              # identical prompt: must leave >= 1 token to compute
    assert t == 12 and got[0][0].shape[2] == 12
    for i in range(50):                  # force eviction
        pc.insert([100 + i] * 16, kv)
    assert pc.bytes <= 10_000 + 1024 and len(pc.nodes) >= 1
    assert all(nid in pc.nodes for nid in pc.index.values())


def test_bucket_selection_covers_long_contexts():
    model = build_tiny_model().double()
    eng = SlotEngine(model, device="cpu", max_batch=5, max_ctx=100)
    assert eng.batch_buckets == [1, 2, 4, 5] and eng.ctx_buckets[-1] == 100
    assert eng._bucket(3, eng.batch_buckets) == 4 and eng._bucket(101, eng.ctx_buckets) == 100


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_cuda_graphs_match_eager_slot_engine_and_reference():
    model = build_tiny_model().cuda()
    def go(**kw):
        reqs = make_requests(seed=7, n=10)
        eng = SlotEngine(model, device="cuda", max_ctx=128, ctx_buckets=[64, 128], **kw)
        for r in reqs:
            eng.submit(r)
        eng.run_until_idle()
        return eng, [r.out_ids for r in reqs], reqs
    e0, eager, reqs = go(max_batch=4, cuda_graphs=False)
    e1, graph, _ = go(max_batch=4, cuda_graphs=True)
    assert e1.cuda_graphs and e1.graph_error is None and len(e1.graphs) > 0, e1.graph_error
    agree = sum(a == b for a, b in zip(eager, graph)) / len(eager)
    assert agree >= 0.9
    cpu_model = build_tiny_model()   # same seed -> same weights; the uncached reference runs on CPU
    ref_ok = sum(o == reference_greedy(cpu_model, r.prompt_ids, r.max_new_tokens) for o, r in zip(graph, reqs)) / len(reqs)
    assert ref_ok >= 0.8
