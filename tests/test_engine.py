import random

import torch

from llmserve.engine import Engine, GenRequest
from llmserve.models import build_tiny_model


def reference_greedy(model, prompt, n):
    """Ground truth: recompute the full sequence every step, no KV cache, no batching."""
    ids = list(prompt)
    out = []
    with torch.no_grad():
        for _ in range(n):
            logits = model(input_ids=torch.tensor([ids])).logits[0, -1]
            tok = int(logits.argmax())
            out.append(tok)
            ids.append(tok)
    return out


def make_requests(seed, n, vocab=500):
    rnd = random.Random(seed)
    reqs = []
    for _ in range(n):
        prompt = [rnd.randrange(3, vocab) for _ in range(rnd.randint(3, 40))]
        reqs.append(GenRequest(prompt_ids=prompt, max_new_tokens=rnd.randint(1, 24), ignore_eos=True))
    return reqs


def run(model, reqs, **kw):
    eng = Engine(model, device="cpu", **kw)
    for r in reqs:
        eng.submit(r)
    eng.run_until_idle()
    return eng


def test_all_policies_match_uncached_reference():
    model = build_tiny_model().double()
    for policy, max_batch in [("continuous", 1), ("continuous", 4), ("continuous", 16), ("static", 4)]:
        reqs = make_requests(seed=1, n=12)
        run(model, reqs, max_batch=max_batch, policy=policy)
        for r in reqs:
            assert r.out_ids == reference_greedy(model, r.prompt_ids, r.max_new_tokens), (policy, max_batch)
            assert r.finish_reason == "length"


def test_continuous_batching_actually_batches():
    model = build_tiny_model().double()
    reqs = make_requests(seed=2, n=10)
    eng = run(model, reqs, max_batch=8, policy="continuous")
    sizes = [s[2] for s in eng.step_log if s[1] == "decode"]
    assert max(sizes) > 1
    assert eng.completed == len(reqs)


def test_static_waits_for_batch_to_drain():
    model = build_tiny_model().double()
    reqs = make_requests(seed=3, n=6)
    eng = Engine(model, device="cpu", max_batch=2, policy="static")
    for r in reqs:
        eng.submit(r)
    max_seen = 0
    while eng.tick():
        max_seen = max(max_seen, len(eng.active))
    assert max_seen <= 2


def test_eos_stops_generation_and_is_not_emitted():
    # EOS on the very first generated token -> nothing emitted
    model = build_tiny_model().double()
    prompt = [5, 6, 7, 8]
    ref = reference_greedy(model, prompt, 4)
    eng = Engine(model, eos_ids={ref[0]}, device="cpu", max_batch=2)
    req = GenRequest(prompt_ids=prompt, max_new_tokens=20)
    eng.submit(req)
    eng.run_until_idle()
    assert req.finish_reason == "stop" and req.out_ids == []

    # EOS in the middle of a generation: find a seed/prompt whose greedy path has a novel token
    for seed in range(20):
        model = build_tiny_model(seed=seed).double()
        rnd = random.Random(seed)
        prompt = [rnd.randrange(3, 500) for _ in range(8)]
        ref = reference_greedy(model, prompt, 16)
        ks = [i for i in range(1, len(ref)) if ref[i] not in ref[:i]]
        if ks:
            k = ks[0]
            eng = Engine(model, eos_ids={ref[k]}, device="cpu", max_batch=2)
            req = GenRequest(prompt_ids=prompt, max_new_tokens=20)
            eng.submit(req)
            eng.run_until_idle()
            assert req.finish_reason == "stop"
            assert req.out_ids == ref[:k]
            return
    raise AssertionError("no seed produced a novel token")


def test_background_thread_serves_requests():
    model = build_tiny_model().double()
    reqs = make_requests(seed=4, n=5)
    eng = Engine(model, device="cpu", max_batch=4)
    eng.start()
    for r in reqs:
        eng.submit(r)
    import time
    deadline = time.time() + 30
    while eng.completed < len(reqs) and time.time() < deadline:
        time.sleep(0.05)
    eng.stop()
    assert eng.completed == len(reqs)
    for r in reqs:
        assert r.out_ids == reference_greedy(model, r.prompt_ids, r.max_new_tokens)


def test_last_logits_prefill_matches_full_logits():
    from llmserve.engine import forward_last_logits

    model = build_tiny_model().double()
    ids = torch.tensor([[5, 9, 11, 40, 7, 3]])
    with torch.no_grad():
        full = model(input_ids=ids, use_cache=True).logits[0, -1]
        last = forward_last_logits(model, ids, use_cache=True).logits[0, -1]
    assert torch.allclose(full, last)
    for flag in (True, False):
        reqs = make_requests(seed=9, n=6)
        run(model, reqs, max_batch=4, policy="continuous", last_logits_only=flag)
        for r in reqs:
            assert r.out_ids == reference_greedy(model, r.prompt_ids, r.max_new_tokens)
